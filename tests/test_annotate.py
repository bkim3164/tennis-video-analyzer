"""Tests for skeleton drawing and annotated-video output.

All drawing tests use hand-crafted keypoint arrays — no MediaPipe inference,
so they run anywhere. Keypoints follow the estimator convention:
(x, y, visibility) with x/y normalized to [0, 1].
"""

import numpy as np
import pytest

from tennis_analyzer.pose import N_LANDMARKS, Landmark
from tennis_analyzer.video import (
    annotate_frames,
    draw_skeleton,
    probe_video,
    write_video,
)


def blank_frame(height: int = 96, width: int = 128) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def no_detection_keypoints() -> np.ndarray:
    """Keypoints for an undetected frame: NaN coords, zero visibility."""
    kpts = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    kpts[:, 2] = 0.0
    return kpts


def full_pose_keypoints() -> np.ndarray:
    """A synthetic fully-visible pose spread across the middle of the frame."""
    rng = np.random.default_rng(42)
    kpts = np.empty((N_LANDMARKS, 3), dtype=np.float32)
    kpts[:, 0] = rng.uniform(0.2, 0.8, N_LANDMARKS)
    kpts[:, 1] = rng.uniform(0.2, 0.8, N_LANDMARKS)
    kpts[:, 2] = 1.0
    return kpts


# ---------------------------------------------------------------- draw_skeleton


def test_draw_skeleton_marks_pixels():
    frame = blank_frame()
    out = draw_skeleton(frame, full_pose_keypoints())
    assert out.shape == frame.shape
    assert out.any(), "skeleton should draw non-black pixels on a black frame"


def test_draw_skeleton_does_not_modify_input():
    frame = blank_frame()
    draw_skeleton(frame, full_pose_keypoints())
    assert not frame.any(), "input frame must remain untouched"


def test_draw_skeleton_no_detection_is_noop():
    out = draw_skeleton(blank_frame(), no_detection_keypoints())
    assert not out.any()


def test_draw_skeleton_respects_visibility_threshold():
    kpts = full_pose_keypoints()
    kpts[:, 2] = 0.2  # below the default threshold of 0.5
    out = draw_skeleton(blank_frame(), kpts)
    assert not out.any()


def test_draw_skeleton_half_visible_pose_draws_less():
    frame = blank_frame()
    full = draw_skeleton(frame, full_pose_keypoints())
    kpts = full_pose_keypoints()
    kpts[: N_LANDMARKS // 2, 2] = 0.0  # hide the face half of the landmarks
    partial = draw_skeleton(frame, kpts)
    assert 0 < np.count_nonzero(partial) < np.count_nonzero(full)


def test_draw_skeleton_clips_out_of_frame_landmarks():
    kpts = full_pose_keypoints()
    kpts[Landmark.LEFT_WRIST, :2] = (1.4, -0.3)  # extrapolated outside the frame
    out = draw_skeleton(blank_frame(), kpts)  # must not raise
    assert out.any()


# --------------------------------------------------------------- annotate_frames


def test_annotate_frames_shape_and_content():
    frames = np.zeros((4, 96, 128, 3), dtype=np.uint8)
    keypoints = np.stack([full_pose_keypoints()] * 4)
    out = annotate_frames(frames, keypoints)
    assert out.shape == frames.shape
    assert all(out[i].any() for i in range(4))


def test_annotate_frames_length_mismatch_raises():
    frames = np.zeros((4, 96, 128, 3), dtype=np.uint8)
    keypoints = np.stack([full_pose_keypoints()] * 3)
    with pytest.raises(ValueError, match="equal length"):
        annotate_frames(frames, keypoints)


# ------------------------------------------------------------------ write_video


def test_write_video_roundtrip(tmp_path):
    frames = np.random.default_rng(0).integers(0, 255, (10, 48, 64, 3), dtype=np.uint8)
    out = write_video(frames, tmp_path / "out.mp4", fps=30.0)
    meta = probe_video(out)
    assert meta.frame_count == 10
    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(30.0)


def test_write_video_creates_parent_dirs(tmp_path):
    out = write_video(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path / "a" / "b" / "out.mp4")
    assert out.exists()


def test_write_video_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="zero frames"):
        write_video(np.zeros((0, 32, 32, 3), dtype=np.uint8), tmp_path / "out.mp4")
