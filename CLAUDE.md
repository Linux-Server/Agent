# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Minimal FastAPI application scaffold. Python 3.12, managed with `uv` (see `uv.lock`). Single entrypoint at `main.py` exposing `app = FastAPI()` with one `GET /` route. Declared dependencies in `pyproject.toml` are currently empty — FastAPI is only present via the `.venv`, so adding it to `[project].dependencies` is likely needed before `uv sync` will reproduce the environment on a fresh checkout.

## Commands

- Install / sync deps: `uv sync`
- Run dev server: `uv run uvicorn main:app --reload`
- Add a dependency: `uv add <pkg>`
