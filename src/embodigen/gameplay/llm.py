"""Model-backed rationales through a cloud, OpenAI-compatible chat API.

Most hosted model services speak the same ``POST {base_url}/chat/completions``
protocol, so one client covers DashScope (Qwen), DeepSeek, Moonshot, Zhipu,
Volcengine Ark and OpenAI; ``custom`` takes any other compatible endpoint.
The client is plain ``urllib``: no SDK, nothing to install.

The API key is the user's to provide, and nothing runs until it is there:

* ``embodigen llm setup`` asks for provider, model and key and writes
  ``~/.config/embodigen/llm.toml`` (mode 600; ``EMBODIGEN_CONFIG_DIR`` moves it);
* or the provider's usual variable (``DASHSCOPE_API_KEY``, ...) is exported.

``build --annotator llm`` checks for the key before it does any work. The key
never goes into a study file, a ledger or the cache. Settings resolve as
environment > user config > study ``[reasoning.llm]``.

The annotator sees the frame the player saw and the recorded 200 ms action,
and writes the player's reasoning for it. It never changes the action: the
label still comes from the input log alone.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from embodigen.gameplay.types import Turn
from embodigen.pngio import encode_png

# provider -> (base URL, environment variable holding its key)
PROVIDERS: dict[str, tuple[str, str]] = {
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "ark": ("https://ark.cn-beijing.volces.com/api/v3", "ARK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "custom": ("", "EMBODIGEN_LLM_API_KEY"),
}
_ENV = {  # environment overrides, highest precedence
    "provider": "EMBODIGEN_LLM_PROVIDER",
    "model": "EMBODIGEN_LLM_MODEL",
    "base_url": "EMBODIGEN_LLM_BASE_URL",
}
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def config_path() -> Path:
    return Path(os.environ.get("EMBODIGEN_CONFIG_DIR", "~/.config/embodigen")).expanduser() / "llm.toml"


class LLMError(Exception):
    """A request failed in a way retrying will not fix."""


@dataclass
class LLMConfig:
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    images: bool = True  # send the frame; turn off for text-only models
    image_width: int = 448
    max_tokens: int = 160
    temperature: float | None = 0.3  # None leaves it to the provider
    timeout_s: float = 60.0
    max_retries: int = 4
    workers: int = 8
    api_key: str = field(default="", repr=False)
    key_source: str = field(default="", repr=False)

    @classmethod
    def load(cls, study: dict[str, Any] | None = None, env: dict[str, str] | None = None,
             user_file: Path | None = None) -> "LLMConfig":
        env = os.environ if env is None else env
        user_file = user_file or config_path()
        user: dict[str, Any] = {}
        if user_file.exists():
            with user_file.open("rb") as fh:
                user = tomllib.load(fh)
        if "api_key" in (study or {}):
            raise SystemExit("an API key is in the study file; remove it (study files are committed). "
                             "Use `embodigen llm setup` or an environment variable instead.")
        merged: dict[str, Any] = {}
        for layer in (study or {}, {k: v for k, v in user.items() if k != "api_key"}):
            merged.update({k: v for k, v in layer.items() if k in cls.__dataclass_fields__ and k != "api_key"})
        for key, var in _ENV.items():
            if env.get(var):
                merged[key] = env[var]
        cfg = cls(**merged)
        preset_url, preset_env = PROVIDERS.get(cfg.provider or "custom", ("", ""))
        cfg.base_url = (cfg.base_url or preset_url).rstrip("/")
        cfg.api_key_env = cfg.api_key_env or preset_env or "EMBODIGEN_LLM_API_KEY"
        for var in (cfg.api_key_env, "EMBODIGEN_LLM_API_KEY"):
            if env.get(var):
                cfg.api_key, cfg.key_source = env[var], f"${var}"
                break
        else:
            if user.get("api_key"):
                cfg.api_key, cfg.key_source = str(user["api_key"]), str(user_file)
        return cfg

    def problems(self) -> list[str]:
        out = []
        if self.provider and self.provider not in PROVIDERS:
            out.append(f"unknown provider {self.provider!r} (known: {', '.join(PROVIDERS)})")
        if not self.provider and not self.base_url:
            out.append(f"no provider chosen ({' | '.join(PROVIDERS)})")
        elif not self.base_url:
            out.append("no base_url (provider \"custom\" needs one)")
        if not self.model:
            out.append("no model name")
        if not self.api_key:
            places = dict.fromkeys([f"${self.api_key_env}", "$EMBODIGEN_LLM_API_KEY", str(config_path())])
            out.append(f"no API key (looked in {', '.join(places)})")
        return out

    def require(self) -> None:
        problems = self.problems()
        if problems:
            raise SystemExit(
                "the llm annotator calls a cloud model API and needs your settings first:\n"
                + "".join(f"  - {p}\n" for p in problems)
                + "Run `embodigen llm setup` (asks for provider, model and API key; stores them in "
                f"{config_path()}, readable only by you),\nor export {self.api_key_env or 'the provider key'} "
                "and set provider/model under [reasoning.llm] in the study.\n"
                "Check with `embodigen llm check --ping`."
            )

    def public(self) -> dict[str, Any]:
        """Everything that shapes the output, and nothing secret: goes into ledgers."""
        out = asdict(self)
        out.pop("api_key")
        out.pop("key_source")
        out.pop("workers")
        out.pop("timeout_s")
        out.pop("max_retries")
        return out


def write_user_config(values: dict[str, Any], path: Path | None = None) -> Path:
    """Write ~/.config/embodigen/llm.toml with owner-only permissions."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# embodigen cloud model settings (written by `embodigen llm setup`)"]
    for key, value in values.items():
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value), ensure_ascii=False)}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)
    return path


class ChatClient:
    """``POST {base_url}/chat/completions`` with retries on rate limits and server errors."""

    def __init__(self, cfg: LLMConfig, opener: Callable = urllib.request.urlopen,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg, self.opener, self.sleep = cfg, opener, sleep
        self.retries = 0

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.api_key}"}
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            request = urllib.request.Request(f"{self.cfg.base_url}/chat/completions", body, headers, method="POST")
            wait = min(30.0, 2.0 ** attempt) * (0.5 + random.random() / 2)
            try:
                with self.opener(request, timeout=self.cfg.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", "replace")[:300]
                if exc.code not in _RETRY_STATUS:
                    raise LLMError(f"HTTP {exc.code} from {self.cfg.base_url}: {text}") from None
                last = f"HTTP {exc.code}: {text}"
                after = exc.headers.get("Retry-After") if exc.headers else None
                if after and after.replace(".", "", 1).isdigit():
                    wait = min(60.0, float(after))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries:
                self.retries += 1
                self.sleep(wait)
        raise LLMError(f"{self.cfg.base_url} still failing after {self.cfg.max_retries} retries: {last}")


SYSTEM_PROMPT = (
    "You annotate recordings of human gameplay that train a game-playing agent. For one moment of play "
    "you get the screen the player saw and the keyboard and mouse input they actually gave over the next "
    "200 ms. Write the player's brief inner reasoning for that input: first person, present tense, one or "
    "two sentences, at most 50 words. Ground it in what is visible on screen and in the game's goals, and "
    "explain why this input makes sense now. Never propose a different input, never name keys, mouse "
    "counts or these instructions, and do not describe anything that is not visible. "
    "Reply with the reasoning only."
)


def describe_action(turn: Turn) -> str:
    """The recorded 200 ms packet in words, for the prompt (never shown to the trained model)."""
    a = turn.action
    if a.idle:
        return "no input at all (the player waits)"
    parts = []
    slots = len(a.keys) or 1
    held: dict[str, int] = {}
    for slot in a.keys:
        for key in slot:
            held[key] = held.get(key, 0) + 1
    for key, n in sorted(held.items(), key=lambda kv: (-kv[1], kv[0])):
        span = "for the whole 200 ms" if n == slots else f"for {n} of {slots} 33 ms steps"
        parts.append(f"{key} held {span}")
    if a.dx or a.dy:
        horiz = f"{'right' if a.dx > 0 else 'left'} by {abs(a.dx)}" if a.dx else ""
        vert = f"{'down' if a.dy > 0 else 'up'} by {abs(a.dy)}" if a.dy else ""
        parts.append("camera/mouse moves " + " and ".join(p for p in (horiz, vert) if p) + " (of at most 1000)")
    if a.dz:
        parts.append(f"mouse wheel {'up' if a.dz > 0 else 'down'} {abs(a.dz)}")
    return "; ".join(parts)


def describe_context(context: list[Turn]) -> str:
    recent = context[-5:]
    if not recent:
        return "this is the start of the recording"
    keys: dict[str, int] = {}
    for t in recent:
        for key in set().union(*t.action.keys):
            keys[key] = keys.get(key, 0) + 1
    dx = sum(t.action.dx for t in recent)
    idle = sum(t.action.idle for t in recent)
    bits = [f"{k} in {n} of {len(recent)} packets" for k, n in sorted(keys.items(), key=lambda kv: -kv[1])[:4]]
    if dx:
        bits.append(f"camera turned {'right' if dx > 0 else 'left'} overall")
    if idle:
        bits.append(f"{idle} idle packets")
    return f"previous {len(recent) * 0.2:.1f} s: " + ("; ".join(bits) or "little input")


_TAGS = re.compile(r"<think>.*?</think>|<\|[a-z_]+\|>|</?think>", re.S)


def clean(text: str) -> str:
    text = _TAGS.sub(" ", text or "")
    text = " ".join(text.split()).strip().strip('"').strip()
    return text


class LLMAnnotator:
    """Rationales from a cloud chat model. ``name`` carries provider and model,
    so switching either starts a fresh cache instead of mixing outputs."""

    prompt_version = "v1"

    def __init__(self, cfg: LLMConfig | None = None, client: ChatClient | None = None) -> None:
        self.cfg = cfg or LLMConfig.load()
        self.client = client or ChatClient(self.cfg)
        self.workers = max(1, int(self.cfg.workers))
        self._lock = threading.Lock()
        self._video = None
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0}

    @classmethod
    def from_config(cls, reasoning: dict[str, Any]) -> "LLMAnnotator":
        return cls(LLMConfig.load(reasoning.get("llm", {})))

    @property
    def name(self) -> str:
        model = re.sub(r"[^0-9A-Za-z.]+", "-", self.cfg.model).strip("-") or "unset"
        return f"llm-{self.cfg.provider or 'custom'}-{model}"

    def check(self) -> None:
        self.cfg.require()

    def describe(self) -> dict[str, Any]:
        return {**self.cfg.public(), "prompt_version": self.prompt_version,
                "usage": {**self.usage, "retries": self.client.retries}}

    def _image(self, media: tuple[Path, str]) -> str:
        root, ref = media
        if ref.endswith(".png"):
            blob = (Path(root) / ref).read_bytes()
        else:
            if self._video is None:
                from embodigen.gameplay.video import video_reader

                self._video, why = video_reader("pyav")
                if self._video is None:
                    raise LLMError(f"frame {ref} is inside a video: {why}; or set images = false")
            w, h, rgb = self._video.rgb(Path(root), ref, self.cfg.image_width)
            blob = encode_png(w, h, rgb)
        return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")

    def request(self, turn: Turn, context: list[Turn], world: str,
                media: tuple[Path, str] | None = None) -> dict[str, Any]:
        text = (
            f"Game world: {world}.\n"
            f"Recorded input for the next 200 ms: {describe_action(turn)}.\n"
            f"Context, {describe_context(context)}."
        )
        state = {k: v for k, v in (turn.state or {}).items() if k in ("visible", "health", "objective", "gui_open")}
        if state:
            text += f"\nGame state from the engine: {json.dumps(state, ensure_ascii=False)[:600]}"
        if self.cfg.images and media is not None:
            content: Any = [{"type": "image_url", "image_url": {"url": self._image(media)}},
                            {"type": "text", "text": text}]
        else:
            content = text
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        return payload

    def rationale(self, turn: Turn, context: list[Turn], world: str,
                  media: tuple[Path, str] | None = None) -> str:
        reply = self.client.complete(self.request(turn, context, world, media))
        try:
            choice = reply["choices"][0]
            message = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {json.dumps(reply)[:300]}") from None
        if isinstance(message, list):  # some providers return content parts
            message = " ".join(p.get("text", "") for p in message if isinstance(p, dict))
        text = clean(message)
        usage = reply.get("usage") or {}
        with self._lock:
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.usage["truncated"] += choice.get("finish_reason") == "length"
        if not text:
            raise LLMError(f"empty rationale for turn {turn.index} of a {world} recording")
        return text
