# AGENTS.md

Python ≥ 3.11, standard library only. `src/embodigen` must not import third-party packages; heavy backends live behind the optional extras in `pyproject.toml` (`vizdoom` behind `[env]`, imported only by its backend module; `zstandard` behind `[d2e]`, used by the MCAP reader only when Python < 3.14 lacks `compression.zstd`; `av` behind `[video]`, imported only by `gameplay/video.py:PyAVFrames`).

Two tracks: the robot pipeline (`embodigen.modules`, `embodigen gen`) and the virtual-world gameplay pipeline (`embodigen.gameplay`, `rollout`/`ingest` → `curate` → `build`) that produces GameScaling-style training data.

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

For the gameplay track:

```bash
make worlds                       # rollout -> curate -> build on the reference worlds
embodigen ledger data/mixes/reference
```

It only counts as working if the build ledger prints `budgets nested True, train/val disjoint True`, the deliberately idle and clock-drift recordings are in `data/curated/rejected.jsonl`, and the 300 ms desync recording shows `realigned +300 ms`.

For the dataset adapters on real public files (needs network; D2E needs Python 3.14+ or `.[d2e]`):

```bash
make samples                      # fetch -> ingest vpt/d2e/p2p -> curate -> build
embodigen ledger data/samples-curated
```

It only counts as working if all 9 recordings are certified, D2E's `Minecraft_1.21.8` lands in world `minecraft`, and the build ledger prints `budgets nested True, train/val disjoint True`. With `make samples VIDEO=--video` and `.[video]` installed, the curate ledger must also show `alignment: aligned 8, inconclusive 1` (Brotato is top-down: nothing to verify).

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
| List virtual worlds | `embodigen worlds` |
| Record gameplay | `embodigen rollout --world all --episodes 3 --out data/raw` |
| Convert VPT recordings | `embodigen ingest --source vpt --input <dir> --out data/raw` |
| Convert D2E (OWAMcap) recordings | `embodigen ingest --source d2e --input <dir> --out data/raw` |
| Convert Open-P2P recordings | `embodigen ingest --source p2p --input <dir> --out data/raw` |
| Fetch real public samples | `python3 scripts/fetch_public_samples.py --out data/samples [--video]` |
| Gate and canonicalise | `embodigen curate --raw data/raw --out data/curated` |
| Build a mixture grid | `embodigen build --curated data/curated --study configs/studies/reference.toml --out data/mixes` |
| Preview a recording | `embodigen preview --recording data/raw/<id>` |

## Conventions

- Each stage implements the protocol in `src/embodigen/modules/base.py`. Adding a backend means writing one method and registering it, not editing the pipeline.
- A mock module's `name` must say so (`mock-photoreal-v0`), and the README table must keep that distinction.
- Fidelity numbers in docs and README come from real runs. Regenerate them rather than editing by hand.
- Only certified episodes go under `out/episodes/`; rejected ones go under `out/rejected/`. Never weaken that split.
- Gameplay: recordings that fail the gate get no turns file and are listed in `curated/rejected.jsonl`. Never let them into a mixture.
- Gameplay: split by recording (and by `session` when a source cuts one play session into parts), never by window. Budgets and mixtures must stay prefixes of one priority order (nested); when data runs short, shrink the run and report `shortfall_hours` instead of topping up from another world.
- Gameplay: rationales explain the recorded action and never change it. Change an annotator's output only together with its `prompt_version`, which keys the cache.
- Gameplay: the action text format lives only in `gameplay/actions.py` (`encode_action` / `decode_action`).
- Paper numbers in `configs/studies/gamescaling.toml` are marked `paper`; guesses are marked `unpublished`. Keep that distinction.
- Adapters: field names and quirks come from real files of that dataset, not from its docs alone. Tests build synthetic fixtures in the same shape; never commit dataset files (D2E is CC BY-NC).
- Prefer small, reviewable diffs.
