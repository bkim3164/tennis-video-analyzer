# Worklog

Daily development log for the tennis-video-analyzer project. Newest entries first. See PLAN.md for the full roadmap.

## Day 2 — 2026-07-19 — Video I/O module

Built `src/tennis_analyzer/video/io.py`, the first pipeline stage:

- `probe_video` — metadata (fps, frame count, dimensions) with FPS sanitization against broken containers (0/NaN/absurd values → 30 fps fallback) and a decode-one-frame fallback when headers omit dimensions
- `iter_frames` / `load_frames` — streaming and in-memory frame extraction as BGR uint8 arrays, with `max_frames` cap to guard memory on long clips
- `resample_indices` — pure nearest-neighbor index selection that resamples any clip to `TARGET_FPS=30`, kept video-free so it's trivially unit-testable
- `VideoMetadata` dataclass with `duration_s` / `is_portrait` helpers; `VideoReadError` for clear failure modes
- 19 tests on synthetic `cv2.VideoWriter` clips: portrait video, MJPG/.avi odd-codec case, corrupt files, resampling duration and bounds invariants — no fixture files needed

Decisions: fixed 30 fps as the canonical rate for all downstream keypoint sequences; resampling is index selection (dup/drop) rather than interpolation, which is correct for pose extraction and keeps frames unmodified.

Commits: `b2b3063` feat, `3d3494e` test, plus docs commit.

Note: this day's code was found staged-but-uncommitted from an interrupted earlier session (along with stale `.git` lock files, now cleaned); verified with ruff + pytest before committing.

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
