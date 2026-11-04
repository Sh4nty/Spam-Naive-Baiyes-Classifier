.PHONY: test reproduce

test:
	python -m pytest -q

reproduce:
	python scripts/reproduce.py
