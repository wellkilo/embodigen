"""Download a handful of real recordings from the three public sources.

These are the files the README numbers were measured on. Licenses stay with
their owners (D2E: CC BY-NC 4.0; VPT: MIT data release; Open-P2P: see the
dataset card); nothing here is committed to the repository.

    embodigen fetch --out data/samples          # action logs only (~70 MB)
    embodigen fetch --out data/samples --video  # plus the videos (~0.9 GB)

Downloads report progress, resume from the ``.part`` file an interrupted run
leaves behind (Ctrl+C is safe), and skip files that are already complete.
``--hf-endpoint`` (or ``$HF_ENDPOINT``, e.g. https://hf-mirror.com) swaps the
Hugging Face host for a mirror.
"""

from __future__ import annotations

import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, TextIO

HF_HOST = "https://huggingface.co"
D2E = "datasets/open-world-agents/D2E-480p/resolve/main"
P2P = "datasets/elefantai/p2p-toy-examples/resolve/main"
VPT = "https://openaipublic.blob.core.windows.net/minecraft-rl/data/6.10"

D2E_FILES = [
    "Minecraft_1.21.8/recording_20250827_052328__366c9184_split_00",
    "Counter-Strike_2/recording_20250820_212408_Counter-Strike_2_ea30a554",
    "Brotato/recording_20250901_085726__6b049e5b",
]
P2P_IDS = [
    "01993060-06a0-7352-ac0b-d77d50a1aba7",
    "01993bb8-12e7-78a1-b661-0c0c3a258e21",
    "0199d579-91ea-78f2-acd4-7cda376a3cea",
]
VPT_STEMS = [
    "Player129-f153ac423f61-20210617-171344",
    "Player129-f153ac423f61-20210617-172319",
    "Player129-f153ac423f61-20210617-173110",
]


def plan(video: bool, hf_endpoint: str = "") -> list[tuple[str, str]]:
    """(url, path under --out) for every file to fetch."""
    hf = (hf_endpoint or HF_HOST).rstrip("/")
    files = []
    for stem in D2E_FILES:
        files.append((f"{hf}/{D2E}/{stem}.mcap", f"d2e/{stem}.mcap"))
        if video:
            files.append((f"{hf}/{D2E}/{stem}.mkv", f"d2e/{stem}.mkv"))
    for uid in P2P_IDS:
        files.append((f"{hf}/{P2P}/dataset/{uid}/annotation.proto", f"p2p/{uid}/annotation.proto"))
        if video:
            files.append((f"{hf}/{P2P}/dataset/{uid}/192x192.mp4", f"p2p/{uid}/192x192.mp4"))
    for stem in VPT_STEMS:
        files.append((f"{VPT}/{stem}.jsonl", f"vpt/{stem}.jsonl"))
        if video:
            files.append((f"{VPT}/{stem}.mp4", f"vpt/{stem}.mp4"))
    return files


class Progress:
    """One status line per file: rewritten in place on a terminal, a new line
    every few seconds otherwise (logs, the web console)."""

    def __init__(self, label: str, total: int, done: int, out: TextIO, every_s: float = 3.0) -> None:
        self.label, self.total, self.start_bytes = label, total, done
        self.out, self.every_s = out, every_s
        self.tty = out.isatty()
        self.t0 = self.last = time.monotonic()

    def line(self, done: int) -> str:
        rate = (done - self.start_bytes) / max(time.monotonic() - self.t0, 1e-6) / 1e6
        size = f"{done / 1e6:.1f}/{self.total / 1e6:.1f} MB ({done / self.total:.0%})" if self.total else f"{done / 1e6:.1f} MB"
        return f"      {self.label}  {size}  {rate:.1f} MB/s"

    def update(self, done: int) -> None:
        now = time.monotonic()
        if now - self.last < (0.5 if self.tty else self.every_s):
            return
        self.last = now
        if self.tty:
            self.out.write("\r" + self.line(done).ljust(78))
        else:
            self.out.write(self.line(done) + "\n")
        self.out.flush()

    def close(self) -> None:
        if self.tty:
            self.out.write("\r" + " " * 78 + "\r")
            self.out.flush()


def download(url: str, dest: Path, out: TextIO = sys.stdout, timeout: float = 60.0,
             opener: Callable[..., object] = urllib.request.urlopen) -> int:
    """Fetch url into dest through dest.part, resuming a partial file. Returns bytes."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = opener(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not have:  # 416: the partial file is already whole
            raise
        tmp.rename(dest)
        return have
    with resp:
        if have and getattr(resp, "status", 200) != 206:  # server ignored Range: start over
            have = 0
        length = int(resp.headers.get("Content-Length") or 0)
        total = have + length if length else 0
        progress = Progress(dest.name, total, have, out)
        done = have
        with tmp.open("ab" if have else "wb") as fh:
            try:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    progress.update(done)
            except http.client.HTTPException as exc:  # IncompleteRead on some versions
                raise OSError(f"connection broke at {done} bytes: {exc!r} (run again to resume)") from exc
        progress.close()
    if total and done != total:
        raise OSError(f"connection closed at {done}/{total} bytes (run again to resume)")
    tmp.rename(dest)
    return done


def fetch(out: Path, video: bool = False, hf_endpoint: str = "", log: TextIO = sys.stdout) -> int:
    files = plan(video, hf_endpoint)
    for i, (url, rel) in enumerate(files, 1):
        dest = out / rel
        tag = f"[{i}/{len(files)}]"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{tag} have  {rel}", file=log, flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        resuming = dest.with_suffix(dest.suffix + ".part").exists()
        print(f"{tag} {'resume' if resuming else 'get   '} {rel}", file=log, flush=True)
        try:
            size = download(url, dest, out=log)
        except KeyboardInterrupt:
            print(f"{tag} stopped; run again to resume {rel}", file=log, flush=True)
            raise
        except OSError as exc:
            print(f"{tag} fail  {rel}: {exc}", file=log, flush=True)
            if "huggingface" in url and not hf_endpoint:
                print("      Hugging Face unreachable? Try --hf-endpoint https://hf-mirror.com", file=log, flush=True)
            print("      run again to resume; finished files are kept", file=log, flush=True)
            return 1
        print(f"{tag} got   {rel} ({size / 1e6:.1f} MB)", file=log, flush=True)
    print(f"all {len(files)} files in {out}", file=log, flush=True)
    return 0


def default_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "")
