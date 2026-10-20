.PHONY: install test demo batch preview docs clean

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

docs:
	python3 scripts/build_docs_assets.py --out docs/assets
	python3 scripts/build_demo_gif.py

clean:
	rm -rf out build *.egg-info src/*.egg-info
