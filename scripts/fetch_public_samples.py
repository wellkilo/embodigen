#!/usr/bin/env python3
"""Download a handful of real recordings from the three public sources.

These are the files the README numbers were measured on. Licenses stay with
their owners (D2E: CC BY-NC 4.0; VPT: MIT data release; Open-P2P: see the
dataset card); nothing here is committed to the repository.

    python3 scripts/fetch_public_samples.py --out data/samples          # action logs only (~70 MB)
    python3 scripts/fetch_public_samples.py --out data/samples --video  # plus the videos (~0.9 GB)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"
D2E = f"{HF}/open-world-agents/D2E-480p/resolve/main"
P2P = f"{HF}/elefantai/p2p-toy-examples/resolve/main"
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


def plan(video: bool) -> list[tuple[str, str]]:
    files = []
    for stem in D2E_FILES:
        files.append((f"{D2E}/{stem}.mcap", f"d2e/{stem}.mcap"))
        if video:
            files.append((f"{D2E}/{stem}.mkv", f"d2e/{stem}.mkv"))
    for uid in P2P_IDS:
        files.append((f"{P2P}/dataset/{uid}/annotation.proto", f"p2p/{uid}/annotation.proto"))
        if video:
            files.append((f"{P2P}/dataset/{uid}/192x192.mp4", f"p2p/{uid}/192x192.mp4"))
    for stem in VPT_STEMS:
        files.append((f"{VPT}/{stem}.jsonl", f"vpt/{stem}.jsonl"))
        if video:
            files.append((f"{VPT}/{stem}.mp4", f"vpt/{stem}.mp4"))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--video", action="store_true", help="also fetch the videos the frames point into")
    args = ap.parse_args()
    out = Path(args.out)
    for url, rel in plan(args.video):
        dest = out / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            print(f"fail  {rel}: {exc}", file=sys.stderr)
            return 1
        tmp.rename(dest)
        print(f"got   {rel} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
