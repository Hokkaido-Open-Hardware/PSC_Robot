.DEFAULT_GOAL := help

PSC_ONE_SOURCE ?= $(HOME)/Program/PSC_RV32I/PSC-ONE

.PHONY: help clean sync-psc-os

help:
	@echo "make clean  Remove Python bytecode and pytest/mypy/ruff caches"
	@echo "make sync-psc-os  Copy PSC-OS sources from $(PSC_ONE_SOURCE)"

git_update:
	git -C external/PSC-ONE fetch origin
	git -C external/PSC-ONE switch --detach origin/main

# Use checksums because the local source can have older timestamps than checkout.
# Keep destination-only files and Git metadata; do not copy generated outputs.
sync-psc-os:
	@test -f "$(PSC_ONE_SOURCE)/software/os/Makefile"
	@test -f external/PSC-ONE/.git
	rsync -rc --itemize-changes \
		--exclude='.git' --exclude='build/' --exclude='__pycache__/' \
		--exclude='*.pyc' --exclude='*.o' --exclude='*.elf' \
		--exclude='*.bin' --exclude='*.mem' --exclude='*.map' \
		"$(PSC_ONE_SOURCE)/software/os/" external/PSC-ONE/software/os/

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
