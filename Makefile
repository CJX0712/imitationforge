# ImitationForge · 一键命令。作者：晨星
# 默认 PYTHONPATH=. 以包根导入；可用 PY= 覆盖解释器。
PY ?= python
export PYTHONPATH := .

.PHONY: all lint format test cov demo run preflight clean

all: lint test

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=. --cov-report=term-missing

demo:
	$(PY) examples/run_demo.py

preflight:
	$(PY) preflight.py

run: demo
