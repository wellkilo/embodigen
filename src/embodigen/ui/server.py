"""Local web console: every pipeline step as a form, one job at a time, live logs.

    embodigen ui            # prints http://127.0.0.1:8765/#token=...

A job is a list of ordinary ``embodigen ...`` command lines run as
subprocesses in the workspace; the page shows each one, so anything done here
can be repeated from a shell. One job runs at a time because steps share
output directories. Stop sends Ctrl+C to the job's process group (a download
keeps its ``.part`` file and resumes next time), then escalates.

Security: the server binds to 127.0.0.1, every API call needs the token
printed at launch (in the URL fragment, so it never reaches a log), and the
Host header must name this machine (against DNS rebinding). Other pages open
in the same browser and other users on the host cannot start jobs. On a
remote machine, reach it through ``ssh -L 8765:127.0.0.1:8765 <host>``.

The API key typed into the model form goes straight to ``write_user_config``
(mode 600). It is never sent back to the page, logged, or put on a command
line; jobs read it from that file like the CLI does.
"""

from __future__ import annotations

import hmac
import importlib.util
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import tomllib
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

import embodigen

STATIC = Path(__file__).with_name("static")
PKG_PARENT = Path(embodigen.__file__).resolve().parents[1]  # puts `embodigen` on the jobs' path
MAX_LINES = 4000
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MEDIA = {".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp4": "video/mp4",
         # robot episode buffers (little-endian float32 depth, uint16 ids, packed RGB), drawn by the page
         ".f32": "application/octet-stream", ".u16": "application/octet-stream", ".rgb": "application/octet-stream"}
TEXT = {".json", ".jsonl", ".txt", ".md", ".toml", ".csv", ".yaml", ".yml", ".log"}
TEXT_BYTES = 256 * 1024  # shown as is; longer files are cut, with the total size reported
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
SOURCES = ("vpt", "d2e", "p2p")
PROFILES = ("player", "afk", "distracted")
RESOLUTIONS = (64, 96, 128)


class BadRequest(Exception):
    """The form asked for something the CLI would refuse; shown to the user as is."""


LABELS = {"input": "输入目录", "out": "输出目录", "raw": "原始录像目录", "curated": "筛选结果目录", "recording": "录像",
          "task": "任务", "model": "模型名", "episodes": "段数", "seconds": "秒数", "seed": "随机种子",
          "lag_ms": "输入错位", "drift_ms_per_min": "时钟漂移", "start": "起始秒", "frames": "帧数",
          "provider": "服务商", "source": "数据来源", "study": "研究配置", "annotator": "理由标注",
          "profile": "玩家行为", "robot": "机器人", "resolution": "分辨率", "size": "画面尺寸", "worlds": "世界"}


# --------------------------------------------------------------------------- form values


def _text(p: dict[str, Any], key: str, default: str = "", required: bool = True) -> str:
    raw = p.get(key)
    value = str(default if raw in (None, "") else raw).strip()
    if required and not value:
        raise BadRequest(f"请填写{LABELS.get(key, key)}")
    if value.startswith("-") or any(c in value for c in "\x00\r\n"):
        raise BadRequest(f"{LABELS.get(key, key)}: 不接受的值 {value!r}")
    return value


def _num(p: dict[str, Any], key: str, default: float, lo: float, hi: float, cast: Callable = float) -> Any:
    raw = p.get(key)
    try:
        value = cast(default if raw in (None, "") else raw)
    except (TypeError, ValueError):
        raise BadRequest(f"{LABELS.get(key, key)}: 需要数字") from None
    if not lo <= value <= hi:
        raise BadRequest(f"{LABELS.get(key, key)}应在 {lo:g} 到 {hi:g} 之间")
    return value


def _choice(p: dict[str, Any], key: str, default: str, options: tuple[str, ...] | list[str]) -> str:
    value = str(p.get(key) or default)
    if value not in options:
        raise BadRequest(f"{LABELS.get(key, key)}: {value!r} 不在 {', '.join(options)} 中")
    return value


def _cli(*args: Any) -> list[str]:
    return [sys.executable, "-m", "embodigen.cli", *(f"{a:g}" if isinstance(a, float) else str(a) for a in args)]


def display(argv: list[str]) -> str:
    """The command as a user would type it."""
    if argv[:3] == [sys.executable, "-m", "embodigen.cli"]:
        return shlex.join(["embodigen", *argv[3:]])
    return shlex.join(argv)


# --------------------------------------------------------------------------- jobs


class Job:
    def __init__(self, jid: str, action: str, title: str, steps: list[list[str]], artifacts: list[str]) -> None:
        self.id, self.action, self.title = jid, action, title
        self.steps, self.artifacts = steps, artifacts
        self.status = "running"
        self.step = 0
        self.returncode: int | None = None
        self.started, self.ended = time.time(), 0.0
        self.proc: subprocess.Popen | None = None
        self.stop_requested = False
        self._lines: list[str] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > MAX_LINES:
                cut = len(self._lines) - MAX_LINES
                del self._lines[:cut]
                self._dropped += cut

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "action": self.action, "title": self.title, "status": self.status,
                "step": self.step, "steps": [display(s) for s in self.steps], "returncode": self.returncode,
                "started": self.started, "ended": self.ended, "artifacts": self.artifacts}

    def view(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            start = max(since - self._dropped, 0)
            lines, nxt = self._lines[start:], self._dropped + len(self._lines)
        return {**self.summary(), "lines": lines, "next": nxt, "skipped": max(self._dropped - since, 0)}


class Busy(Exception):
    pass


class Runner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self._lock = threading.Lock()
        self._n = 0
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PKG_PARENT), os.environ.get("PYTHONPATH", "")) if p)

    def start(self, action: str, title: str, steps: list[list[str]], artifacts: list[str]) -> Job:
        with self._lock:
            if self.current and self.current.status == "running":
                raise Busy(self.current.title)
            self._n += 1
            job = Job(f"{self._n:04d}", action, title, steps, artifacts)
            self.jobs[job.id] = job
            for old in list(self.jobs)[:-20]:
                del self.jobs[old]
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        rc = 0
        for i, argv in enumerate(job.steps):
            if job.stop_requested:
                break
            job.step = i
            job.add(f"$ {display(argv)}")
            try:
                proc = subprocess.Popen(
                    argv, cwd=self.root, env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                job.add(f"无法启动: {exc}")
                rc = 127
                break
            job.proc = proc
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    job.add(line.rstrip("\n").rsplit("\r", 1)[-1])
            rc = proc.wait()
            job.proc = None
            if rc != 0:
                break
        job.returncode = rc
        job.status = "stopped" if job.stop_requested else ("ok" if rc == 0 else "failed")
        job.ended = time.time()
        job.add({"ok": "完成", "failed": f"失败（退出码 {rc}）", "stopped": "已停止"}[job.status])

    def stop(self, job: Job) -> None:
        job.stop_requested = True
        proc = job.proc
        if proc is None or proc.poll() is not None:
            return
        job.add("正在停止…")

        def escalate() -> None:
            for sig, wait in ((signal.SIGINT, 4.0), (signal.SIGTERM, 4.0), (getattr(signal, "SIGKILL", signal.SIGTERM), 0)):
                if proc.poll() is not None:
                    return
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, sig)
                    else:
                        proc.terminate()
                except (ProcessLookupError, PermissionError):
                    return
                if wait:
                    try:
                        proc.wait(wait)
                    except subprocess.TimeoutExpired:
                        continue

        threading.Thread(target=escalate, daemon=True).start()

    def stop_all(self) -> None:
        for job in list(self.jobs.values()):
            if job.status == "running":
                self.stop(job)
                if job.proc:
                    try:
                        job.proc.wait(10)
                    except subprocess.TimeoutExpired:
                        pass


# --------------------------------------------------------------------------- workspace


class Workspace:
    """What the forms offer: worlds, studies, robots, and the data already on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def inside(self, rel: str) -> Path:
        path = (self.root / unquote(rel)).resolve()
        if path != self.root and self.root not in path.parents:
            raise BadRequest("路径必须在工作目录内")
        return path

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def worlds(self) -> list[dict[str, Any]]:
        from embodigen.gameplay.registry import WorldRegistry

        reg = WorldRegistry.load(self.root / "configs" / "worlds" if (self.root / "configs" / "worlds").is_dir() else None)
        return [{"label": s.label, "title": s.title, "engine": s.engine, "category": s.category, "view": s.view}
                for s in sorted(reg.worlds.values(), key=lambda s: (not s.engine, s.label)) if s.engine]

    def studies(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((self.root / "configs" / "studies").glob("*.toml")):
            try:
                with path.open("rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                continue
            out.append({"path": self.rel(path), "name": data.get("name", path.stem),
                        "budgets": data.get("budgets_hours", []), "anchor": data.get("anchor", ""),
                        "annotator": data.get("reasoning", {}).get("annotator", "mock")})
        return out

    def datasets(self) -> dict[str, list[dict[str, Any]]]:
        found: dict[str, list[dict[str, Any]]] = {"inputs": [], "raw": [], "curated": [], "mixes": [], "embodied": [],
                                                  "robot": []}
        skip = {"frames", "depth", "labels", "turns", "cache", "val", "episodes", "rejected", "__pycache__"}
        stack = [(self.root / top, 0) for top in ("data", "out") if (self.root / top).is_dir()]
        while stack:
            d, depth = stack.pop()
            try:
                children = sorted((e for e in os.scandir(d) if e.is_dir()), key=lambda e: e.name)
            except OSError:
                continue
            ledger = d / "ledger.json"
            if ledger.exists():
                try:
                    data = json.loads(ledger.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
                if data.get("stage") == "curate":
                    s = data.get("summary", {})
                    found["curated"].append({"path": self.rel(d), "updated_at": data.get("updated_at", ""),
                                             "recordings": s.get("recordings", 0), "certified": s.get("certified", 0),
                                             "minutes": round(s.get("hours_certified", 0) * 60, 1)})
                elif data.get("stage") == "build" or "grid" in data:
                    found["mixes"].append({"path": self.rel(d), "updated_at": data.get("updated_at", ""),
                                           "study": data.get("study", ""), "annotator": data.get("annotator", ""),
                                           "datasets": len(data.get("grid", []))})
                continue
            if (d / "meta" / "info.json").exists() and (d / "meta" / "episodes.jsonl").exists():
                try:
                    info = json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    info = {}
                shape = info.get("features", {}).get("observation.images.main", {}).get("shape", [0, 0])
                found["embodied"].append({"path": self.rel(d), "episodes": info.get("total_episodes", 0),
                                          "frames": info.get("total_frames", 0), "fps": info.get("fps", 0),
                                          "size": f"{shape[1]}x{shape[0]}"})
                continue
            if (d / "index.json").exists() and (d / "episodes").is_dir():
                n = sum(1 for _ in (d / "episodes").iterdir())
                bad = sum(1 for _ in (d / "rejected").iterdir()) if (d / "rejected").is_dir() else 0
                found["robot"].append({"path": self.rel(d), "episodes": n, "rejected": bad})
                continue
            recs = sum(1 for c in children if os.path.exists(os.path.join(c.path, "recording.json")))
            if recs:
                found["raw"].append({"path": self.rel(d), "recordings": recs})
                continue
            names = {c.name for c in children}
            for source in SOURCES:
                if source in names:
                    found["inputs"].append({"path": self.rel(d / source), "source": source})
            if depth < 3:
                stack.extend((Path(c.path), depth + 1) for c in children if c.name not in skip and c.name not in SOURCES)
        for rows in found.values():
            rows.sort(key=lambda r: r["path"])
        return found

    def recordings(self, rel: str) -> list[dict[str, Any]]:
        base = self.inside(rel)
        out = []
        for child in sorted(base.iterdir()) if base.is_dir() else []:
            meta = child / "recording.json"
            if not meta.exists():
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            preview = child / "preview.gif"
            out.append({"path": self.rel(child), "id": m.get("recording_id", child.name), "world": m.get("world", ""),
                        "source": m.get("source", ""), "minutes": round(float(m.get("duration", 0)) / 60, 1),
                        "preview": self.rel(preview) if preview.exists() else ""})
        return out

    def ledger(self, rel: str) -> dict[str, Any]:
        base = self.inside(rel)
        path = base / "ledger.json" if base.is_dir() else base
        if not path.exists():
            raise BadRequest(f"{rel} 下没有 ledger.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, Any] = {"path": self.rel(path.parent), "ledger": data}
        if data.get("stage") == "curate":
            rows = []
            recs = path.parent / "recordings.jsonl"
            for line in recs.read_text(encoding="utf-8").splitlines() if recs.exists() else []:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows.append({"id": r["recording_id"], "world": r.get("world", ""), "source": r.get("source", ""),
                             "minutes": round(r.get("duration", 0) / 60, 1), "passed": r.get("passed", False),
                             "reasons": r.get("reasons", []), "alignment": r.get("alignment", {}).get("status", ""),
                             "realigned_ms": r.get("realigned_ms", 0)})
            out["recordings"] = rows
        return out


    def robot(self, rel: str) -> dict[str, Any]:
        """A robot output directory: its index.json as written, plus where each episode lives."""
        base = self.inside(rel)
        index = base / "index.json"
        if not index.exists():
            raise BadRequest(f"{rel} 下没有 index.json")
        text = index.read_text(encoding="utf-8")
        data = json.loads(text)
        episodes = []
        for row in data.get("episodes", []):
            eid = str(row.get("episode_id", ""))
            home = next((base / b / eid for b in ("episodes", "rejected") if (base / b / eid).is_dir()), None)
            episodes.append({**row, "path": self.rel(home) if home else ""})
        return {"path": self.rel(base), "index": {k: v for k, v in data.items() if k != "episodes"},
                "episodes": episodes, "index_file": self.rel(index), "index_text": text}

    def embodied(self, rel: str) -> dict[str, Any]:
        """An `embodigen export` dataset: info.json and stats.json as written, its episodes."""
        base = self.inside(rel)
        meta = base / "meta"
        if not (meta / "info.json").exists():
            raise BadRequest(f"{rel} 下没有 meta/info.json")
        info_text = (meta / "info.json").read_text(encoding="utf-8")
        stats = meta / "stats.json"
        episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        return {"path": self.rel(base), "info": json.loads(info_text), "info_text": info_text,
                "stats_text": stats.read_text(encoding="utf-8") if stats.exists() else "", "episodes": episodes}

    def rows(self, rel: str, episode: int, start: int, count: int) -> dict[str, Any]:
        """Raw JSONL lines [start, start + count) of one exported episode."""
        base = self.inside(rel)
        info = json.loads((base / "meta" / "info.json").read_text(encoding="utf-8"))
        chunk = episode // int(info.get("chunks_size", 1000))
        path = self.inside(self.rel(base / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.jsonl"))
        if not path.exists():
            raise BadRequest(f"没有回合 {episode}")
        lines = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= start + count:
                    break
                if i >= start:
                    lines.append(line.rstrip("\n"))
        return {"episode": episode, "start": start, "lines": lines}

    def tree(self, rel: str) -> dict[str, Any]:
        """Every file under a directory (an episode), with sizes."""
        base = self.inside(rel)
        if not base.is_dir():
            raise BadRequest(f"{rel} 不是目录")
        files = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append({"path": self.rel(path), "name": path.relative_to(base).as_posix(), "bytes": path.stat().st_size})
                if len(files) >= 5000:
                    break
        return {"path": self.rel(base), "files": files}

    def text(self, rel: str) -> dict[str, Any]:
        """A file's own bytes: text as written (cut at TEXT_BYTES), binary as a hex dump of its head."""
        path = self.inside(rel)
        if not path.is_file():
            raise BadRequest(f"{rel} 不是文件")
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(TEXT_BYTES if path.suffix.lower() in TEXT else 512)
        out: dict[str, Any] = {"path": self.rel(path), "bytes": size, "truncated": size > len(head)}
        if path.suffix.lower() in TEXT:
            out["kind"] = "text"
            out["text"] = head.decode("utf-8", errors="replace")
            if path.suffix.lower() == ".jsonl" and not out["truncated"]:
                out["lines"] = out["text"].count("\n")
            return out
        out["kind"] = "binary"
        out["hex"] = "\n".join(
            f"{i:08x}  {head[i:i + 16].hex(' '):<47}  {''.join(chr(b) if 32 <= b < 127 else '.' for b in head[i:i + 16])}"
            for i in range(0, len(head), 16))
        return out


# --------------------------------------------------------------------------- actions


def llm_state() -> dict[str, Any]:
    from embodigen.gameplay.llm import PROVIDERS, LLMConfig, config_path

    cfg = LLMConfig.load({})
    return {"providers": {k: {"base_url": url, "key_env": env} for k, (url, env) in PROVIDERS.items()},
            "provider": cfg.provider, "model": cfg.model, "base_url": cfg.base_url, "images": cfg.images,
            "key_set": bool(cfg.api_key), "key_source": cfg.key_source, "key_env": cfg.api_key_env,
            "problems": cfg.problems(), "config_path": str(config_path())}


def save_llm(p: dict[str, Any]) -> dict[str, Any]:
    from embodigen.gameplay.llm import PROVIDERS, config_path, write_user_config

    provider = _choice(p, "provider", "", list(PROVIDERS))
    model = _text(p, "model")
    base_url = str(p.get("base_url") or "").strip() if provider == "custom" else ""
    if provider == "custom" and not base_url.startswith(("https://", "http://")):
        raise BadRequest("custom 需要以 http(s):// 开头的 base URL")
    key = str(p.get("api_key") or "").strip()
    if not key:  # keep the stored key when only provider/model change
        path = config_path()
        if path.exists():
            with path.open("rb") as fh:
                key = str(tomllib.load(fh).get("api_key", ""))
    if not key:
        raise BadRequest("请填写 API key")
    if any(c.isspace() for c in key):
        raise BadRequest("API key 里不应有空白字符")
    write_user_config({"provider": provider, "model": model, "base_url": base_url,
                       "images": bool(p.get("images", True)), "api_key": key})
    return llm_state()


def plan(action: str, p: dict[str, Any], ws: Workspace) -> tuple[str, list[list[str]], list[str]]:
    """(title, command lines, files to show when done) for one form."""
    if action == "fetch":
        out = _text(p, "out", "data/samples")
        args: list[Any] = ["fetch", "--out", out]
        if p.get("video"):
            args.append("--video")
        mirror = str(p.get("hf_endpoint") or "").strip()
        if mirror:
            if not mirror.startswith(("https://", "http://")):
                raise BadRequest("镜像地址需以 http(s):// 开头")
            args += ["--hf-endpoint", mirror]
        return "下载公开样例", [_cli(*args)], []

    if action == "rollout":
        labels = {w["label"] for w in ws.worlds()}
        worlds = p.get("worlds") or ["all"]
        worlds = [str(w) for w in (worlds if isinstance(worlds, list) else [worlds])]
        if worlds != ["all"] and not set(worlds) <= labels:
            raise BadRequest(f"未知世界: {', '.join(sorted(set(worlds) - labels))}")
        args = ["rollout", "--world", ",".join(worlds),
                "--episodes", _num(p, "episodes", 2, 1, 200, int),
                "--seconds", _num(p, "seconds", 30.0, 2, 1800),
                "--seed", _num(p, "seed", 0, 0, 10**9, int),
                "--profile", _choice(p, "profile", "player", PROFILES),
                "--out", _text(p, "out", "data/raw")]
        lag = _num(p, "lag_ms", 0.0, -5000, 5000)
        drift = _num(p, "drift_ms_per_min", 0.0, 0, 10000)
        if lag:
            args += ["--lag-ms", lag]
        if drift:
            args += ["--drift-ms-per-min", drift]
        return f"录制 {','.join(worlds)}", [_cli(*args)], []

    if action == "ingest":
        source = _choice(p, "source", "", SOURCES)
        return (f"导入 {source}",
                [_cli("ingest", "--source", source, "--input", _text(p, "input"), "--out", _text(p, "out", "data/raw"))], [])

    if action == "curate":
        args = ["curate", "--raw", _text(p, "raw", "data/raw"), "--out", _text(p, "out", "data/curated")]
        if not p.get("video", True):
            args.append("--no-video")
        if not p.get("realign", True):
            args.append("--no-realign")
        return "筛选与对齐", [_cli(*args)], []

    if action == "build":
        studies = {s["path"] for s in ws.studies()}
        study = _choice(p, "study", "configs/studies/reference.toml", sorted(studies))
        annotator = _choice(p, "annotator", "", ("", "mock", "llm"))
        if annotator == "llm":
            _require_llm(ws, study)
        args = ["build", "--curated", _text(p, "curated", "data/curated"), "--study", study,
                "--out", _text(p, "out", "data/mixes")]
        if annotator:
            args += ["--annotator", annotator]
        return "构建训练集", [_cli(*args)], []

    if action == "preview":
        rec = ws.inside(_text(p, "recording"))
        if not (rec / "recording.json").exists():
            raise BadRequest("不是一个录像目录（缺少 recording.json）")
        out = rec / "preview.gif"
        args = ["preview", "--recording", ws.rel(rec), "--out", ws.rel(out),
                "--start", _num(p, "start", 0.0, 0, 86400), "--max-frames", _num(p, "frames", 60, 4, 600, int)]
        return f"预览 {rec.name}", [_cli(*args)], [ws.rel(out)]

    if action == "gen":
        from embodigen.robots import available_robots

        out = _text(p, "out", "out")
        args = ["gen", "--task", _text(p, "task", "把红色方块放进盒子"),
                "--robot", _choice(p, "robot", "franka", available_robots()),
                "--episodes", _num(p, "episodes", 1, 1, 64, int),
                "--resolution", _choice(p, "resolution", "96", tuple(map(str, RESOLUTIONS))), "--out", out]
        return "生成机器人数据", [_cli(*args), _cli("index", "--out", out)], []

    if action == "export":
        args = ["export", "--curated", _text(p, "curated", "data/curated"), "--out", _text(p, "out", "data/embodied")]
        worlds = str(p.get("worlds") or "").strip()
        if worlds:
            if not all(w.replace("-", "").replace("_", "").isalnum() for w in worlds.split(",")):
                raise BadRequest("世界名只能是逗号分隔的标签")
            args += ["--worlds", worlds]
        size = str(p.get("size") or "").strip().lower()
        if size:
            w, _, h = size.partition("x")
            if not (w.isdigit() and h.isdigit()):
                raise BadRequest("画面尺寸写成 宽x高，如 320x180")
            args += ["--size", size]
        if p.get("overwrite"):
            args.append("--overwrite")
        return "导出具身格式", [_cli(*args)], []

    if action == "llm-check":
        return "测试模型连接", [_cli("llm", "check", "--ping")], []

    if action == "preset-worlds":
        steps = [
            _cli("rollout", "--world", "raycast-arena", "--episodes", 8, "--seconds", 30, "--out", "data/raw"),
            _cli("rollout", "--world", "all", "--episodes", 3, "--seconds", 30, "--seed", 100, "--out", "data/raw"),
            _cli("rollout", "--world", "raycast-rooms", "--episodes", 1, "--seed", 50, "--profile", "afk", "--out", "data/raw"),
            _cli("rollout", "--world", "raycast-lava", "--episodes", 1, "--seed", 70, "--lag-ms", 300, "--out", "data/raw"),
            _cli("rollout", "--world", "raycast-maze", "--episodes", 1, "--seed", 80, "--seconds", 60,
                 "--drift-ms-per-min", 600, "--out", "data/raw"),
            _cli("curate", "--raw", "data/raw", "--out", "data/curated"),
            _cli("build", "--curated", "data/curated", "--study", "configs/studies/reference.toml", "--out", "data/mixes"),
        ]
        return "参考世界全流程（make worlds）", steps, []

    if action == "preset-samples":
        annotator = _choice(p, "annotator", "", ("", "mock", "llm"))
        study = "configs/studies/public-samples.toml"
        if annotator == "llm":
            _require_llm(ws, study)
        fetch = ["fetch", "--out", "data/samples"]
        if p.get("video"):
            fetch.append("--video")
        mirror = str(p.get("hf_endpoint") or "").strip()
        if mirror:
            if not mirror.startswith(("https://", "http://")):
                raise BadRequest("镜像地址需以 http(s):// 开头")
            fetch += ["--hf-endpoint", mirror]
        steps = [_cli(*fetch)]
        steps += [_cli("ingest", "--source", s, "--input", f"data/samples/{s}", "--out", "data/samples-raw") for s in SOURCES]
        steps.append(_cli("curate", "--raw", "data/samples-raw", "--out", "data/samples-curated",
                          *([] if p.get("video") else ["--no-video"])))
        build = ["build", "--curated", "data/samples-curated", "--study", study, "--out", "data/samples-mixes"]
        if annotator:
            build += ["--annotator", annotator]
        steps.append(_cli(*build))
        return "公开样例全流程（make samples）", steps, []

    raise BadRequest(f"未知操作 {action!r}")


def _require_llm(ws: Workspace, study: str) -> None:
    """Key first: refuse an llm build in the page before anything starts."""
    from embodigen.gameplay.llm import LLMConfig
    from embodigen.gameplay.pipeline import load_study

    cfg = LLMConfig.load(load_study(ws.inside(study)).reasoning.get("llm", {}))
    problems = cfg.problems()
    if problems:
        raise BadRequest("云端模型还没配置好：" + "；".join(problems) + "。请先在「模型设置」里填写。")


# --------------------------------------------------------------------------- HTTP


class Console(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], root: Path, token: str, quiet: bool = False) -> None:
        super().__init__(addr, Handler)
        self.ws = Workspace(root)
        self.runner = Runner(self.ws.root)
        self.token = token
        self.quiet = quiet
        self.any_host = addr[0] not in LOCAL_HOSTS


class Handler(BaseHTTPRequestHandler):
    server: Console
    server_version = "embodigen-ui"

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if not self.server.quiet:  # path only: queries may carry the token
            sys.stderr.write(f"{self.command} {urlsplit(self.path).path} {code}\n")

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.server.quiet:
            sys.stderr.write((fmt % args) + "\n")

    # -- plumbing

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        name = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        return self.server.any_host or name in LOCAL_HOSTS

    def _token_ok(self, query: dict[str, list[str]]) -> bool:
        given = self.headers.get("X-Token") or (query.get("t") or [""])[0]
        return hmac.compare_digest(given.encode(), self.server.token.encode())

    def _body(self) -> dict[str, Any]:
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise BadRequest("需要 application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            raise BadRequest("请求过大")
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise BadRequest("需要 JSON 对象")
        return data

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        path = url.path
        if method == "GET" and (path == "/" or path.startswith("/static/")):
            return self._static("index.html" if path == "/" else path[len("/static/"):])
        if not self._token_ok(query):
            return self._error(HTTPStatus.UNAUTHORIZED, "缺少或错误的访问令牌：请用启动时打印的完整链接打开")
        try:
            if method == "GET":
                return self._get(path, query)
            return self._post(path)
        except BadRequest as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Busy as exc:
            return self._error(HTTPStatus.CONFLICT, f"已有任务在运行：{exc}")
        except SystemExit as exc:  # a config loader refusing, e.g. a malformed study
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except (OSError, ValueError) as exc:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    # -- routes

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file() or path.suffix not in STATIC_TYPES:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        self._send(200, path.read_bytes(), STATIC_TYPES[path.suffix],
                   {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                                               "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"})

    def _get(self, path: str, q: dict[str, list[str]]) -> None:
        ws, runner = self.server.ws, self.server.runner
        arg = lambda k, d="": (q.get(k) or [d])[0]  # noqa: E731
        if path == "/api/state":
            cur = runner.current
            return self._json({
                "root": str(ws.root), "python": sys.version.split()[0], "version": embodigen.__version__,
                "extras": {"video": importlib.util.find_spec("av") is not None,
                           "zstd": sys.version_info >= (3, 14) or importlib.util.find_spec("zstandard") is not None},
                "worlds": ws.worlds(), "studies": ws.studies(), "robots": _robots(), "llm": llm_state(),
                "datasets": ws.datasets(), "jobs": [j.summary() for j in reversed(list(runner.jobs.values()))],
                "current": cur.id if cur and cur.status == "running" else "",
            })
        if path == "/api/datasets":
            return self._json(ws.datasets())
        if path == "/api/recordings":
            return self._json(ws.recordings(arg("dir")))
        if path == "/api/ledger":
            return self._json(ws.ledger(arg("path")))
        if path == "/api/embodied":
            return self._json(ws.embodied(arg("path")))
        if path == "/api/rows":
            try:
                ep, start, count = int(arg("episode", "0")), int(arg("start", "0")), min(int(arg("count", "200")), 1000)
            except ValueError:
                raise BadRequest("episode / start / count 需要整数") from None
            return self._json(ws.rows(arg("path"), ep, max(start, 0), max(count, 1)))
        if path == "/api/robot":
            return self._json(ws.robot(arg("path")))
        if path == "/api/tree":
            return self._json(ws.tree(arg("path")))
        if path == "/api/text":
            return self._json(ws.text(arg("path")))
        if path.startswith("/api/jobs/"):
            job = runner.jobs.get(path.rsplit("/", 1)[-1])
            if not job:
                return self._error(HTTPStatus.NOT_FOUND, "没有这个任务")
            return self._json(job.view(int(arg("since", "0") or 0)))
        if path.startswith("/files/"):
            return self._file(path[len("/files/"):])
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def _post(self, path: str) -> None:
        ws, runner = self.server.ws, self.server.runner
        if path == "/api/run":
            body = self._body()
            action = str(body.get("action", ""))
            params = body.get("params") or {}
            if not isinstance(params, dict):
                raise BadRequest("params 需要是对象")
            title, steps, artifacts = plan(action, params, ws)
            return self._json(runner.start(action, title, steps, artifacts).summary())
        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job = runner.jobs.get(path.split("/")[3])
            if not job:
                return self._error(HTTPStatus.NOT_FOUND, "没有这个任务")
            runner.stop(job)
            return self._json(job.summary())
        if path == "/api/plan":  # the command lines a form would run, without running them
            body = self._body()
            try:
                _, steps, _ = plan(str(body.get("action", "")), body.get("params") or {}, ws)
            except BadRequest as exc:  # an unfinished form, not a failed request
                return self._json({"steps": [], "error": str(exc)})
            return self._json({"steps": [display(s) for s in steps]})
        if path == "/api/llm":
            return self._json(save_llm(self._body()))
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def _file(self, rel: str) -> None:
        path = self.server.ws.inside(rel)
        ctype = MEDIA.get(path.suffix.lower())
        if not ctype or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        self._send(200, path.read_bytes(), ctype)


def _robots() -> list[str]:
    from embodigen.robots import available_robots

    return available_robots()


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
                quiet: bool = False) -> Console:
    return Console((host, port), Path(root), token or secrets.token_urlsafe(24), quiet=quiet)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> int:
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        # Started in the background (`... &`): SIGINT is ignored and jobs would
        # inherit that, so Stop could not send them Ctrl+C. Handled signals
        # reset to the default in children; ignored ones do not.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        server = make_server(root, host, port)
    except OSError as exc:
        raise SystemExit(f"cannot listen on {host}:{port}: {exc} (try --port)") from None
    shown = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{shown}:{server.server_address[1]}/#token={server.token}"
    print(f"embodigen console  {url}")
    print(f"workspace          {server.ws.root}")
    if server.any_host:
        print(f"warning: listening on {host}, reachable from other machines; anyone with the link can run jobs")
    else:
        print(f"remote server? on your own computer run `ssh -L {server.server_address[1]}:127.0.0.1:"
              f"{server.server_address[1]} <this host>`, then open the link there")
    print("Ctrl+C here stops the console and any running job", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.runner.stop_all()
        server.server_close()
    return 0
