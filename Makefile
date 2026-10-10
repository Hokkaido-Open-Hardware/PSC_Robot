.DEFAULT_GOAL := help

.PHONY: help clean

help:
	@echo "make clean  Remove Python bytecode and pytest/mypy/ruff caches"

# Keep environments, external sources, build outputs and repository metadata.
# pyvenv.cfg also protects virtual environments with custom directory names.
clean:
	find . \
		-type d \( -name .git -o -name .agents -o -name .codex -o -name .aws \
			-o -path ./external -o -path ./build \
			-o -name myenv -o -name .venv -o -name venv -o -name env \
			-o -name .tox -o -name .nox \
			-o -exec test -f '{}/pyvenv.cfg' \; \) -prune -o \
		-type d \( -name __pycache__ -o -name .pytest_cache \
			-o -name .mypy_cache -o -name .ruff_cache \) \
			-prune -exec rm -rf -- {} + -o \
		-type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f -- {} +
