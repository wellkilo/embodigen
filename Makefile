.PHONY: install test demo batch preview docs worlds samples embodied world4d ui clean

install:
	pip install -e .

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m embodigen.cli gen --task "把红色方块放进盒子" --out out --resolution 96
	PYTHONPATH=src python3 -m embodigen.cli validate --episode out/episodes/franka-0000

batch:
	PYTHONPATH=src python3 -m embodigen.cli gen --task "把红色方块放进盒子" --out out --resolution 64 --episodes 8
	PYTHONPATH=src python3 -m embodigen.cli index --out out

preview:
	PYTHONPATH=src python3 -m embodigen.cli preview --episode out/episodes/franka-0000

# Virtual-world track on the stdlib reference worlds: rollout -> curate -> build.
worlds:
	PYTHONPATH=src python3 -m embodigen.cli rollout --world raycast-arena --episodes 8 --seconds 30 --out data/raw
	PYTHONPATH=src python3 -m embodigen.cli rollout --world all --episodes 3 --seconds 30 --seed 100 --out data/raw
	PYTHONPATH=src python3 -m embodigen.cli rollout --world raycast-rooms --episodes 1 --seed 50 --profile afk --out data/raw
	PYTHONPATH=src python3 -m embodigen.cli rollout --world raycast-lava --episodes 1 --seed 70 --lag-ms 300 --out data/raw
	PYTHONPATH=src python3 -m embodigen.cli rollout --world raycast-maze --episodes 1 --seed 80 --seconds 60 --drift-ms-per-min 600 --out data/raw
	PYTHONPATH=src python3 -m embodigen.cli curate --raw data/raw --out data/curated
	PYTHONPATH=src python3 -m embodigen.cli build --curated data/curated --study configs/studies/reference.toml --out data/mixes
	PYTHONPATH=src python3 -m embodigen.cli export --curated data/curated --out data/embodied --overwrite

# Real public recordings (VPT, D2E, Open-P2P): fetch -> ingest -> curate -> build.
# D2E's zstd chunks need Python 3.14+ or `pip install -e '.[d2e]'`.
# `make samples VIDEO=--video` also fetches the videos (~0.9 GB) so curate can
# check alignment from pixels (needs `pip install -e '.[video]'`).
samples:
	PYTHONPATH=src python3 -m embodigen.cli fetch --out data/samples $(VIDEO)
	PYTHONPATH=src python3 -m embodigen.cli ingest --source vpt --input data/samples/vpt --out data/samples-raw
	PYTHONPATH=src python3 -m embodigen.cli ingest --source d2e --input data/samples/d2e --out data/samples-raw
	PYTHONPATH=src python3 -m embodigen.cli ingest --source p2p --input data/samples/p2p --out data/samples-raw
	PYTHONPATH=src python3 -m embodigen.cli curate --raw data/samples-raw --out data/samples-curated
	PYTHONPATH=src python3 -m embodigen.cli build --curated data/samples-curated --study configs/studies/public-samples.toml --out data/samples-mixes
	if [ -n "$(VIDEO)" ]; then PYTHONPATH=src python3 -m embodigen.cli export --curated data/samples-curated --out data/samples-embodied --overwrite; fi

# Certified recordings as embodied (LeRobot-shaped) datasets; run after `make worlds` / `make samples`.
embodied:
	PYTHONPATH=src python3 -m embodigen.cli export --curated data/curated --out data/embodied --overwrite
	if [ -f data/samples-curated/ledger.json ] && find data/samples \( -name '*.mp4' -o -name '*.mkv' \) | grep -q .; then \
	  PYTHONPATH=src python3 -m embodigen.cli export --curated data/samples-curated --out data/samples-embodied --overwrite; fi

# 4D Gaussian-splat worlds: one from a prompt (robot track), one from a game rollout.
world4d:
	PYTHONPATH=src python3 -m embodigen.cli world --task "把红色方块放进盒子"
	PYTHONPATH=src python3 -m embodigen.cli world --rollout raycast-arena --seconds 20

# Local web console for every step above (prints a link with an access token).
ui:
	PYTHONPATH=src python3 -m embodigen.cli ui

docs:
	python3 scripts/build_docs_assets.py --out docs/assets
	python3 scripts/build_demo_gif.py
	python3 scripts/build_gameplay_assets.py

clean:
	rm -rf out data build *.egg-info src/*.egg-info
