.PHONY: install test demo clean

install:
	pip install -e .

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m embodigen.cli gen --task "把红色方块放进盒子" --out out --resolution 96
	PYTHONPATH=src python3 -m embodigen.cli validate --episode out/episodes/franka-0000

clean:
	rm -rf out build *.egg-info src/*.egg-info
