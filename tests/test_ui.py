"""The local web console (`embodigen ui`) and the resumable sample fetcher."""

import io
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodigen.gameplay.sources import fetch  # noqa: E402
from embodigen.ui import server as ui  # noqa: E402

KEY_VARS = [v for v in os.environ if v.endswith("_API_KEY") or v.startswith("EMBODIGEN_LLM_")]


class ConsoleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "configs" / "studies").mkdir(parents=True)
        shutil.copy(ROOT / "configs" / "studies" / "reference.toml", self.tmp / "configs" / "studies")
        env = mock.patch.dict(os.environ, {"EMBODIGEN_CONFIG_DIR": str(self.tmp / "cfg")})
        env.start()
        self.addCleanup(env.stop)
        for var in KEY_VARS:
            os.environ.pop(var, None)
        self.srv = ui.make_server(self.tmp, port=0, token="t0k", quiet=True)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.addCleanup(self.srv.runner.stop_all)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def call(self, path, body=None, token="t0k", headers=None):
        req = urllib.request.Request(self.base + path, headers={"X-Token": token, **(headers or {})})
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def wait(self, jid, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, job = self.call(f"/api/jobs/{jid}")
            if job["status"] != "running":
                return job
            time.sleep(0.2)
        self.fail(f"job {jid} still running")

    def test_api_needs_the_token_and_a_local_host(self):
        self.assertEqual(self.call("/api/state", token="")[0], 401)
        self.assertEqual(self.call("/api/state", token="nope")[0], 401)
        self.assertEqual(self.call("/api/state", headers={"Host": "evil.example:80"})[0], 403)
        code, state = self.call("/api/state")
        self.assertEqual(code, 200)
        self.assertIn("raycast-arena", [w["label"] for w in state["worlds"]])
        self.assertEqual([s["path"] for s in state["studies"]], ["configs/studies/reference.toml"])
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:  # the page itself carries no data
            self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])
            self.assertIn(b"console.js", r.read())

    def test_forms_become_cli_commands_and_bad_values_are_refused(self):
        code, r = self.call("/api/plan", {"action": "preset-samples", "params": {"video": True,
                                                                                  "hf_endpoint": "https://hf-mirror.com"}})
        self.assertEqual(code, 200)
        self.assertEqual(r["steps"][0], "embodigen fetch --out data/samples --video --hf-endpoint https://hf-mirror.com")
        self.assertEqual(len(r["steps"]), 6)
        _, r = self.call("/api/plan", {"action": "ingest", "params": {"source": "vpt", "input": "--help"}})
        self.assertIn("不接受", r["error"])
        _, r = self.call("/api/plan", {"action": "rollout", "params": {"worlds": ["no-such-world"]}})
        self.assertIn("未知世界", r["error"])
        code, r = self.call("/api/run", {"action": "rm", "params": {}})
        self.assertEqual(code, 400)

    def test_llm_build_is_refused_before_anything_runs_without_a_key(self):
        code, r = self.call("/api/run", {"action": "build", "params": {"annotator": "llm"}})
        self.assertEqual(code, 400)
        self.assertIn("API key", r["error"])
        self.assertEqual(self.srv.runner.jobs, {})

    def test_api_key_is_stored_owner_only_and_never_sent_back(self):
        secret = "sk-test-0123456789"
        code, state = self.call("/api/llm", {"provider": "deepseek", "model": "deepseek-chat", "api_key": secret,
                                             "images": False})
        self.assertEqual(code, 200)
        self.assertTrue(state["key_set"])
        self.assertEqual(state["problems"], [])
        path = self.tmp / "cfg" / "llm.toml"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        _, full = self.call("/api/state")
        self.assertNotIn(secret, json.dumps(full) + json.dumps(state))
        # changing only the model keeps the stored key
        _, state = self.call("/api/llm", {"provider": "deepseek", "model": "deepseek-reasoner", "api_key": ""})
        self.assertTrue(state["key_set"])
        self.assertIn(secret, path.read_text())
        _, plan = self.call("/api/plan", {"action": "build", "params": {"annotator": "llm"}})
        self.assertNotIn(secret, json.dumps(plan))
        self.assertIn("--annotator llm", plan["steps"][0])

    def test_a_rollout_job_runs_streams_its_log_and_shows_up_as_data(self):
        code, job = self.call("/api/run", {"action": "rollout", "params": {
            "worlds": ["raycast-arena"], "episodes": 1, "seconds": 2, "out": "data/raw"}})
        self.assertEqual(code, 200, job)
        job = self.wait(job["id"])
        self.assertEqual(job["status"], "ok", job["lines"])
        self.assertTrue(job["lines"][0].startswith("$ embodigen rollout --world raycast-arena"))
        self.assertTrue((self.tmp / "data" / "raw" / "raycast-arena-0000" / "recording.json").exists())
        _, ds = self.call("/api/datasets")
        self.assertEqual(ds["raw"], [{"path": "data/raw", "recordings": 1}])
        _, recs = self.call("/api/recordings?dir=data/raw")
        self.assertEqual(recs[0]["world"], "raycast-arena")

        _, job = self.call("/api/run", {"action": "preview", "params": {"recording": recs[0]["path"], "frames": 8}})
        job = self.wait(job["id"])
        self.assertEqual(job["status"], "ok", job["lines"])
        gif = job["artifacts"][0]
        with urllib.request.urlopen(f"{self.base}/files/{gif}?t=t0k", timeout=10) as r:
            self.assertEqual(r.headers["Content-Type"], "image/gif")
            self.assertTrue(r.read().startswith(b"GIF89a"))
        self.assertEqual(self.call(f"/files/{gif}", token="")[0], 401)
        self.assertIn(self.call("/files/%2e%2e/%2e%2e/etc/passwd")[0], (400, 404))
        self.assertEqual(self.call("/files/data/raw/raycast-arena-0000/recording.json")[0], 404)

    def test_robot_episodes_open_with_their_files_as_written(self):
        _, job = self.call("/api/run", {"action": "gen", "params": {"task": "把红色方块放进盒子", "resolution": "64",
                                                                      "out": "out"}})
        job = self.wait(job["id"], timeout=120)
        self.assertEqual(job["status"], "ok", job["lines"])
        _, ds = self.call("/api/datasets")
        self.assertEqual(ds["robot"], [{"path": "out", "episodes": 1, "rejected": 0}])
        _, robot = self.call("/api/robot?path=out")
        ep = robot["episodes"][0]
        self.assertEqual(ep["path"], "out/episodes/franka-0000")
        self.assertEqual(robot["index_text"], (self.tmp / "out" / "index.json").read_text(encoding="utf-8"))
        _, tree = self.call(f"/api/tree?path={ep['path']}")
        names = {f["name"] for f in tree["files"]}
        self.assertTrue({"meta/info.json", "meta/fidelity.json", "frames/depth/000.f32", "frames/seg/000.u16"} <= names)
        data = next(f for f in tree["files"] if f["name"].endswith(".jsonl"))
        _, text = self.call(f"/api/text?path={data['path']}")
        self.assertEqual(text["kind"], "text")
        self.assertEqual(text["text"], (self.tmp / data["path"]).read_text(encoding="utf-8"))
        self.assertEqual(text["lines"], ep["frames"])
        _, binary = self.call(f"/api/text?path={ep['path']}/frames/seg/000.u16")
        self.assertEqual(binary["kind"], "binary")
        self.assertEqual(binary["bytes"], 64 * 64 * 2)
        self.assertTrue(binary["hex"].startswith("00000000  "))
        with urllib.request.urlopen(f"{self.base}/files/{ep['path']}/frames/depth/000.f32?t=t0k", timeout=10) as r:
            self.assertEqual(len(r.read()), 64 * 64 * 4)
        self.assertEqual(self.call("/api/text?path=../outside.json")[0], 400)

    def test_one_job_at_a_time_and_stop_ends_it(self):
        runner = self.srv.runner
        job = runner.start("sleep", "sleep", [[sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"],
                                              [sys.executable, "-c", "print('never')"]], [])
        with self.assertRaises(ui.Busy):
            runner.start("x", "x", [[sys.executable, "-c", "pass"]], [])
        t0 = time.time()
        while "up" not in job.view()["lines"] and time.time() - t0 < 10:
            time.sleep(0.05)
        code, _ = self.call(f"/api/jobs/{job.id}/stop", {})
        self.assertEqual(code, 200)
        done = self.wait(job.id, timeout=15)
        self.assertEqual(done["status"], "stopped")
        self.assertLess(time.time() - t0, 12)
        self.assertNotIn("never", done["lines"])
        self.assertEqual(runner.start("y", "y", [[sys.executable, "-c", "pass"]], []).status, "running")


class _Ranged(BaseHTTPRequestHandler):
    payload = bytes(range(256)) * 4000  # ~1 MB
    cut_at = 0  # serve only this many bytes of a full request, then hang up

    def do_GET(self):  # noqa: N802
        start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not start and self.cut_at:
            self.wfile.write(body[: self.cut_at])
            return
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class FetchTest(unittest.TestCase):
    def test_interrupted_download_resumes_with_a_range_request(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ranged)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_address[1]}/f.bin"
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        dest = tmp / "f.bin"
        _Ranged.cut_at = 300_000
        log = io.StringIO()
        with self.assertRaises(OSError):
            fetch.download(url, dest, out=log, timeout=5)
        self.assertFalse(dest.exists())
        self.assertEqual((tmp / "f.bin.part").stat().st_size, 300_000)
        _Ranged.cut_at = 0
        self.assertEqual(fetch.download(url, dest, out=log, timeout=5), len(_Ranged.payload))
        self.assertEqual(dest.read_bytes(), _Ranged.payload)
        self.assertFalse((tmp / "f.bin.part").exists())

    def test_mirror_replaces_only_the_hugging_face_host(self):
        urls = [u for u, _ in fetch.plan(video=True, hf_endpoint="https://hf-mirror.com/")]
        self.assertEqual(len(urls), 18)
        self.assertTrue(all(u.startswith(("https://hf-mirror.com/datasets/", fetch.VPT)) for u in urls))
        self.assertFalse(any("huggingface.co" in u for u in urls))


if __name__ == "__main__":
    unittest.main()
