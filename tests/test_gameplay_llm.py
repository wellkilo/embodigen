"""The cloud llm annotator, against an in-process fake of the OpenAI-compatible API.

No request leaves the machine: a ThreadingHTTPServer on 127.0.0.1 plays the
provider, records what it receives and scripts failures.
"""

import base64
import io
import json
import os
import stat
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from embodigen.cli import main
from embodigen.gameplay.actions import Canonicalizer
from embodigen.gameplay.annotate import selected
from embodigen.gameplay.llm import ChatClient, LLMAnnotator, LLMConfig, LLMError, clean, describe_action
from embodigen.gameplay.mixture import StudySpec
from embodigen.gameplay.pipeline import build, curate, run_rollouts
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.types import ActionChunk, Turn
from embodigen.pngio import read_png

KEY = "sk-test-not-a-real-key"


class FakeProvider:
    """Answers /v1/chat/completions; ``script`` is a list of statuses to return first."""

    def __init__(self):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.script: list[int] = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer.lock:
                    outer.requests.append(body)
                    outer.headers.append(dict(self.headers))
                    status = outer.script.pop(0) if outer.script else 200
                if self.path != "/v1/chat/completions":
                    status = 404
                if status != 200:
                    self.send_response(status)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    self.wfile.write(b'{"error": {"message": "scripted failure"}}')
                    return
                user = body["messages"][-1]["content"]
                text = user if isinstance(user, str) else user[-1]["text"]
                world = text.split("Game world: ")[1].split(".")[0] if "Game world: " in text else "?"
                reply = {
                    "choices": [{"message": {"role": "assistant",
                                             "content": f"<think>scratch</think> I see the {world} and keep moving."},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 12},
                }
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def cfg_for(provider: FakeProvider, **kw) -> LLMConfig:
    base = dict(provider="custom", model="vision-test-1", base_url=provider.url, api_key=KEY,
                max_retries=3, timeout_s=10.0, workers=4)
    base.update(kw)
    return LLMConfig(**base)


class ConfigTest(unittest.TestCase):
    def test_precedence_and_key_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "llm.toml"
            user.write_text('provider = "dashscope"\nmodel = "from-user-file"\napi_key = "k-file"\n')
            cfg = LLMConfig.load({"model": "from-study", "images": False}, env={}, user_file=user)
            self.assertEqual(cfg.model, "from-user-file")  # user file over study
            self.assertFalse(cfg.images)  # study value kept when nobody overrides it
            self.assertEqual(cfg.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual((cfg.api_key, cfg.key_source), ("k-file", str(user)))
            env = {"DASHSCOPE_API_KEY": "k-env", "EMBODIGEN_LLM_MODEL": "from-env"}
            cfg = LLMConfig.load({}, env=env, user_file=user)
            self.assertEqual((cfg.model, cfg.api_key, cfg.key_source), ("from-env", "k-env", "$DASHSCOPE_API_KEY"))

    def test_key_is_required_and_never_public(self):
        cfg = LLMConfig.load({"provider": "deepseek", "model": "m"}, env={}, user_file=Path("/nonexistent"))
        self.assertTrue(any("API key" in p for p in cfg.problems()))
        with self.assertRaises(SystemExit) as ctx:
            cfg.require()
        self.assertIn("embodigen llm setup", str(ctx.exception))
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))
        cfg.api_key = KEY
        self.assertEqual(cfg.problems(), [])
        self.assertNotIn(KEY, json.dumps(cfg.public()))
        self.assertNotIn(KEY, repr(cfg))

    def test_keys_in_study_files_are_refused(self):
        with self.assertRaises(SystemExit):
            LLMConfig.load({"api_key": "oops"}, env={}, user_file=Path("/nonexistent"))


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProvider()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        self.fake.requests.clear()
        self.fake.headers.clear()
        self.fake.script.clear()

    def test_rate_limits_and_server_errors_are_retried(self):
        self.fake.script = [429, 503]
        client = ChatClient(cfg_for(self.fake), sleep=lambda s: None)
        reply = client.complete({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("choices", reply)
        self.assertEqual((len(self.fake.requests), client.retries), (3, 2))
        self.assertEqual(self.fake.headers[0]["Authorization"], f"Bearer {KEY}")

    def test_client_errors_are_not_retried(self):
        self.fake.script = [400]
        with self.assertRaises(LLMError) as ctx:
            ChatClient(cfg_for(self.fake), sleep=lambda s: None).complete({"model": "m", "messages": []})
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(len(self.fake.requests), 1)
        self.assertNotIn(KEY, str(ctx.exception))

    def test_giving_up_after_max_retries(self):
        self.fake.script = [500] * 10
        with self.assertRaises(LLMError):
            ChatClient(cfg_for(self.fake, max_retries=2), sleep=lambda s: None).complete({"model": "m"})
        self.assertEqual(len(self.fake.requests), 3)

    def test_request_carries_the_frame_and_the_recorded_action(self):
        from tests.test_gameplay_quality import play

        with tempfile.TemporaryDirectory() as tmp:
            rec, root = play(tmp, "r", seconds=6.0)
            turns = Canonicalizer().turns(rec.events, rec.observations, rec.meta.duration)
            turn = next(t for t in turns if not t.action.idle)
            ann = LLMAnnotator(cfg_for(self.fake), ChatClient(cfg_for(self.fake), sleep=lambda s: None))
            text = ann.rationale(turn, turns[: turn.index], "raycast-arena", (root, turn.frame))
        self.assertEqual(text, "I see the raycast-arena and keep moving.")  # <think> stripped
        body = self.fake.requests[0]
        self.assertEqual(body["model"], "vision-test-1")
        image, prompt = body["messages"][1]["content"]
        png = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
        with tempfile.NamedTemporaryFile(suffix=".png") as fh:
            fh.write(png)
            fh.flush()
            self.assertEqual(read_png(fh.name)[:2], (64, 40))
        self.assertIn(describe_action(turn), prompt["text"])
        self.assertEqual(ann.usage["requests"], 1)
        self.assertEqual(ann.usage["completion_tokens"], 12)

    def test_text_only_models_get_no_image(self):
        ann = LLMAnnotator(cfg_for(self.fake, images=False), ChatClient(cfg_for(self.fake), sleep=lambda s: None))
        turn = Turn(index=3, t=0.6, frame="frames/x.png", action=ActionChunk(0, 0, 0, ((),) * 6))
        ann.rationale(turn, [], "w", (Path("."), "frames/x.png"))
        self.assertIsInstance(self.fake.requests[0]["messages"][1]["content"], str)

    def test_clean(self):
        self.assertEqual(clean('"<think>a\nb</think>  I wait.  "'), "I wait.")
        self.assertEqual(clean("<|action_start|> 1 ; 2"), "1 ; 2")


class BuildWithLLMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fake = FakeProvider()
        root = Path(cls.tmp.name)
        cls.registry = WorldRegistry.load()
        raw = root / "raw"
        run_rollouts(cls.registry, ["raycast-arena"], 3, 20.0, raw, size=(48, 30), gt=False, workers=3)
        run_rollouts(cls.registry, ["raycast-maze"], 3, 20.0, raw, size=(48, 30), gt=False, workers=3)
        cls.curated = root / "curated"
        curate(raw, cls.curated, workers=3)
        cls.study = StudySpec.from_toml({
            "name": "llm", "anchor": "raycast-arena", "ood": [], "budgets_hours": [0.005, 0.01], "seed": 5,
            "val": {"clips_per_world": 2}, "reasoning": {"rate": 0.1, "annotator": "llm"},
            "serialize": {"shard_size": 16},
            "mixtures": [{"name": "S1", "anchor_share": 1.0}, {"name": "S3", "anchor_share": 0.271}],
        })
        cls.env = {"EMBODIGEN_LLM_PROVIDER": "custom", "EMBODIGEN_LLM_BASE_URL": cls.fake.url,
                   "EMBODIGEN_LLM_MODEL": "vision-test-1", "EMBODIGEN_LLM_API_KEY": KEY,
                   "EMBODIGEN_CONFIG_DIR": str(root / "cfg")}

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()
        cls.tmp.cleanup()

    def test_missing_key_stops_before_any_output(self):
        out = Path(self.tmp.name) / "nokey"
        env = {k: v for k, v in self.env.items() if k != "EMBODIGEN_LLM_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("EMBODIGEN_LLM_API_KEY", None)
            with self.assertRaises(SystemExit):
                build(self.curated, self.study, self.registry, out)
        self.assertFalse(out.exists())

    def test_build_annotates_every_selected_turn_once(self):
        out = Path(self.tmp.name) / "mixes"
        self.fake.requests.clear()
        with mock.patch.dict(os.environ, self.env):
            ledger = build(self.curated, self.study, self.registry, out)
        made = len(self.fake.requests)
        self.assertGreater(made, 0)
        self.assertEqual(ledger["checks"]["rationale_cache"]["prefilled"], made)
        self.assertEqual(ledger["checks"]["rationale_cache"]["misses"], 0)  # writing only read the cache
        self.assertEqual(ledger["annotator"], "llm-custom-vision-test-1")
        self.assertEqual(ledger["annotator_config"]["usage"]["requests"], made)
        blob = "".join(p.read_text() for p in out.rglob("*.json*"))
        self.assertNotIn(KEY, blob)
        # Every rationale in the shards is the model's, on exactly the selected turns.
        seen = 0
        for shard in out.rglob("train-*.jsonl"):
            for line in shard.read_text().splitlines():
                row = json.loads(line)
                for k, msg in enumerate(row["messages"][1::2]):
                    has = "keep moving" in msg["content"]
                    self.assertEqual(has, selected(row["recording"], row["start_turn"] + k, 0.1, "think:5"))
                    seen += has
        self.assertGreater(seen, 0)
        # A rebuild is free: everything comes from the cache.
        self.fake.requests.clear()
        with mock.patch.dict(os.environ, self.env):
            again = build(self.curated, self.study, self.registry, out)
        self.assertEqual(len(self.fake.requests), 0)
        self.assertEqual(again["checks"]["rationale_cache"]["prefilled"], 0)

    def test_failures_keep_paid_rationales_and_resume(self):
        out = Path(self.tmp.name) / "resume"
        self.fake.requests.clear()
        self.fake.script = [200] * 3 + [400]
        with mock.patch.dict(os.environ, {**self.env, "EMBODIGEN_LLM_MODEL": "resume-model"}):
            with self.assertRaises(LLMError):
                build(self.curated, self.study, self.registry, out)
            cache = out / "cache" / "rationales-llm-custom-resume-model.jsonl"
            kept = len(cache.read_text().splitlines())
            # Every successful reply of the failed run was kept (one request was the 400).
            self.assertEqual(kept, len(self.fake.requests) - 1)
            self.fake.requests.clear()
            ledger = build(self.curated, self.study, self.registry, out)
        total = len(cache.read_text().splitlines())
        # The resumed run asks only for what is missing.
        self.assertEqual(len(self.fake.requests), total - kept)
        self.assertEqual(ledger["checks"]["rationale_cache"]["prefilled"], total - kept)


class SetupCommandTest(unittest.TestCase):
    def test_setup_writes_an_owner_only_file_and_check_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"EMBODIGEN_CONFIG_DIR": tmp}):
            for var in ("DASHSCOPE_API_KEY", "EMBODIGEN_LLM_API_KEY", "EMBODIGEN_LLM_MODEL", "EMBODIGEN_LLM_PROVIDER"):
                os.environ.pop(var, None)
            with mock.patch("sys.stdin", io.StringIO(KEY + "\n")), redirect_stdout(io.StringIO()) as out:
                code = main(["llm", "setup", "--provider", "dashscope", "--model", "qwen-vl-test", "--api-key-stdin"])
            self.assertEqual(code, 0)
            self.assertNotIn(KEY, out.getvalue())
            path = Path(tmp) / "llm.toml"
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with redirect_stdout(io.StringIO()) as out:
                code = main(["llm", "check"])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("qwen-vl-test", out.getvalue())
            self.assertIn("set, from", out.getvalue())
            self.assertNotIn(KEY, out.getvalue())


if __name__ == "__main__":
    unittest.main()
