cod:
	uv run detector_neumonia.py
.PHONY: test
test:
	uv run pytest -v
