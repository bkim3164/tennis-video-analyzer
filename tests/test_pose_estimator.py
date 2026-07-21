"""Tests for the MediaPipe pose estimator wrapper.

Inference tests need the pose model asset (~5 MB download on first run) and
are skipped when it cannot be obtained (e.g., sandboxed/offline environments).
CI has network access, so they run there. Synthetic frames contain no person,
which exercises exactly the no-detection path the estimator must handle.
"""

import numpy as np
import pytest

from tennis_analyzer.pose import (
    LANDMARK_DIMS,
    N_LANDMARKS,
    PoseEstimator,
    PoseModelError,
    ensure_model,
)


def _model_available() -> bool:
    try:
        ensure_model()
        return True
    except PoseModelError:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(), reason="pose model unavailable (no network access?)"
)


# ------------------------------------------------------------------ ensure_model


def test_ensure_model_missing_no_download_raises(tmp_path):
    with pytest.raises(PoseModelError, match="download disabled"):
        ensure_model(tmp_path / "missing.task", download=False)


def test_ensure_model_returns_existing_path(tmp_path):
    fake = tmp_path / "model.task"
    fake.write_bytes(b"weights")
    assert ensure_model(fake, download=False) == fake


def test_cache_dir_env_override(tmp_path, monkeypatch):
    from tennis_analyzer.pose import default_cache_dir

    monkeypatch.setenv("TENNIS_ANALYZER_CACHE", str(tmp_path / "cache"))
    assert default_cache_dir() == tmp_path / "cache"


# ----------------------------------------------------------------- PoseEstimator


@requires_model
def test_estimate_no_person_returns_nan_and_zero_visibility():
    frames = np.zeros((5, 96, 128, 3), dtype=np.uint8)
    with PoseEstimator() as estimator:
        keypoints = estimator.estimate(frames, fps=30.0)
    assert keypoints.shape == (5, N_LANDMARKS, LANDMARK_DIMS)
    assert keypoints.dtype == np.float32
    assert np.isnan(keypoints[:, :, :2]).all()
    assert (keypoints[:, :, 2] == 0.0).all()


@requires_model
def test_estimate_called_twice_keeps_timestamps_valid():
    frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    with PoseEstimator() as estimator:
        first = estimator.estimate(frames, fps=30.0)
        second = estimator.estimate(frames, fps=30.0)  # must not raise
    assert first.shape == second.shape


@requires_model
def test_estimate_rejects_bad_input():
    with PoseEstimator() as estimator:
        with pytest.raises(ValueError, match="expected"):
            estimator.estimate(np.zeros((5, 96, 128), dtype=np.uint8))
        with pytest.raises(ValueError, match="positive"):
            estimator.estimate(np.zeros((2, 32, 32, 3), dtype=np.uint8), fps=0)
