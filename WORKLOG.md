# Worklog

Daily development log for the tennis-video-analyzer project. Newest entries first. See PLAN.md for the full roadmap.

## Day 1 — 2026-07-13 — Repo scaffolding

Set up the project foundation:

- `src/tennis_analyzer/` package skeleton with subpackages for each pipeline stage: `video`, `pose`, `segmentation`, `data`, `models`, `training`, `feedback`
- CLI entry point (`tennis-analyzer analyze <video>`) with argument validation; pipeline wiring lands Day 20
- Tooling: `pyproject.toml` (ruff with Google-style docstring enforcement, pytest config, console script), `requirements.txt` / `requirements-dev.txt` split
- 11 smoke tests covering package imports and CLI behavior
- GitHub Actions CI: ruff check, format check, and pytest on Python 3.11
- Hygiene: `.gitignore` (data/, .env, caches), `.env.example` template

Decisions: numpy pinned `<2.0` for MediaPipe binary compatibility; tests exempt from docstring lint rules.

Commits: `b4c3d31` chore, `9090fab` feat, `205a7af` test, `087e658` ci.
