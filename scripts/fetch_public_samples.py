#!/usr/bin/env python3
"""Download the real public recordings the README numbers were measured on.

Same as `embodigen fetch` (src/embodigen/gameplay/sources/fetch.py):

    python3 scripts/fetch_public_samples.py --out data/samples          # action logs only (~70 MB)
    python3 scripts/fetch_public_samples.py --out data/samples --video  # plus the videos (~0.9 GB)
    python3 scripts/fetch_public_samples.py --video --hf-endpoint https://hf-mirror.com
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodigen.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["fetch", *sys.argv[1:]]))
