# Worklog

Daily development log for the tennis-video-analyzer project. Newest entries first. See PLAN.md for the full roadmap.

## Day 3 — 2026-07-20 — MediaPipe pose estimation + skeleton overlay

Wired the "vision" stage of the pipeline:

- `pose/landmarks.py` — the 33-landmark BlazePose topology as a MediaPipe-free constants module (`Landmark` enum + `SKELETON_CONNECTIONS`), so drawing and future joint-angle code never import MediaPipe
- `pose/estimator.py` — `PoseEstimator` wrapping the MediaPipe **Tasks** `PoseLandmarker` in VIDEO mode (temporal tracking between frames); output is `(T, 33, 3)` float32 of `(x, y, visibility)`, with undetected frames as `(NaN, NaN, 0.0)` so downstream code masks them explicitly
- `ensure_model()` — lazy download of the ~5 MB `pose_landmarker_lite.task` asset into `~/.cache/tennis-analyzer` (override via `$TENNIS_ANALYZER_CACHE`), pinned to a fixed model version
- `video/annotate.py` — pure-rendering skeleton overlay (`draw_skeleton`, `annotate_frames`), `write_video`, and an end-to-end `annotate_video(in, out)` convenience chaining load → pose → draw → write
- 23 tests: landmark topology invariants (contiguity, left/right symmetry), drawing behavior (visibility threshold, no-detection no-op, out-of-frame clipping), video write round-trip; estimator inference tests auto-skip where the model can't be downloaded and run in CI

Decisions: Tasks API over the legacy `mp.solutions.pose` (removed from MediaPipe ≥0.10.30-era wheels, so pinning old MediaPipe would be a dead end); `num_poses=1` since v1 targets single-player practice clips; model asset cached outside the repo rather than committed.

Deferred: saving an annotated demo clip of a real pro stroke — needs real footage and the model download, both unavailable in this environment. One-liner once local: `python -c "from tennis_analyzer.video import annotate_video; annotate_video('clip.mp4', 'annotated.mp4')"`.

Commits: `68b914c` feat (estimator), `595d198` feat (overlay), `886637b` test, plus docs commit.

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
