data:
	python scripts/01_download.py && python scripts/02_curate.py && python scripts/03_make_splits.py
test:
	pytest -q
