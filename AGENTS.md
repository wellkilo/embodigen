# AGENTS.md

Python ≥ 3.11, standard library only. `src/embodigen` must not import third-party packages; heavy backends live behind the optional extras in `pyproject.toml`.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Verify before claiming success

```bash
python3 -m unittest discover -s tests
embodigen gen --task "把红色方块放进盒子" --out out --resolution 96
embodigen validate --episode out/episodes/franka-0000
```

A run only counts as working if `validate` reports `passed: true` and the frames exist. Quote the measured p95 error rather than a rounded guess.

## Common tasks

| Task | Command |
|---|---|
| Generate one episode | `embodigen gen --task "..." --out out --resolution 96` |
| Generate a batch | `embodigen gen --task "..." --out out --episodes 8 --workers 4` |
| Summarise a dataset | `embodigen index --out out` |
| Re-check a packed episode | `embodigen validate --episode out/episodes/<id>` |
| Preview an episode as GIF | `embodigen preview --episode out/episodes/<id>` |
| List swappable backends | `embodigen backends` |
| Rebuild showcase assets | `python3 scripts/build_docs_assets.py --out docs/assets` |
| Rebuild the README GIF | `python3 scripts/build_demo_gif.py` |
| Rebuild favicons | `python3 scripts/build_favicon.py --out docs` |

## Conventions

- Each stage implements the protocol in `src/embodigen/modules/base.py`. Adding a backend means writing one method and registering it, not editing the pipeline.
- A mock module's `name` must say so (`mock-photoreal-v0`), and the README table must keep that distinction.
- Fidelity numbers in docs and README come from real runs. Regenerate them rather than editing by hand.
- Only certified episodes go under `out/episodes/`; rejected ones go under `out/rejected/`. Never weaken that split.
- Prefer small, reviewable diffs.
