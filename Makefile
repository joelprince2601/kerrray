# KerrRay developer targets (PROJECT.md section 30).
# Python is the project virtual environment: .venv/Scripts/python.exe on Windows,
# .venv/bin/python elsewhere. Override with `make PYTHON=... <target>` if needed.
PYTHON ?= $(if $(wildcard .venv/Scripts/python.exe),./.venv/Scripts/python.exe,./.venv/bin/python)

.PHONY: install test cli-help

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

cli-help:
	PYTHONIOENCODING=utf-8 $(PYTHON) -m kerrray --help
