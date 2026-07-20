"""Tests for tennis_analyzer.video.io using tiny synthetic videos.

Videos are generated with cv2.VideoWriter so tests need no fixture files.
Codecs covered: mp4v (.mp4) and MJPG (.avi) — the "odd codec" edge case —
plus portrait dimensions to mimic phone footage.
"""

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from tennis_analyzer.video import (
    TARGET_FPS,
    VideoReadError,
    iter_frames,
    load_frames,
    probe_video,
    resample_indices,
)


def write_video(
    path: Path,
    n_frames: int = 20,
    fps: float = 20.0,
    size: tuple[int, int] = (64, 48),  # (width, height)
    codec: str = "mp4v",
) -> Path:
    """Write a synthetic video where frame i is filled with intensity i."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    assert writer.isOpened(), f"VideoWriter failed for codec {codec} at {path}"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------- probe_video


def test_probe_reports_metadata(tmp_path):
    clip = write_video(tmp_path / "clip.mp4", n_frames=20, fps=20.0, size=(64, 48))
    meta = probe_video(clip)
    assert meta.fps == pytest.approx(20.0)
    assert meta.frame_count == 20
    assert (meta.width, meta.height) == (64, 48)
    assert meta.duration_s == pytest.approx(1.0)
    assert not meta.is_portrait


def test_probe_portrait_video(tmp_path):
    clip = write_video(tmp_path / "portrait.mp4", size=(48, 64))
    meta = probe_video(clip)
    assert meta.is_portrait
    assert meta.height > meta.width


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(VideoReadError, match="not found"):
        probe_video(tmp_path / "nope.mp4")


def test_probe_corrupt_file_raises(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoReadError):
        # Depending on the OpenCV build this fails at open or at first decode.
        frames, _ = load_frames(junk)
        assert frames.size  # unreachable on expected path


# ---------------------------------------------------------------- iter_frames


def test_iter_frames_yields_all_frames(tmp_path):
    clip = write_video(tmp_path / "clip.mp4", n_frames=15)
    frames = list(iter_frames(clip))
    assert len(frames) == 15
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [(".mp4", "mp4v"), (".avi", "MJPG")],
)
def test_iter_frames_across_codecs(tmp_path, suffix, codec):
    clip = write_video(tmp_path / f"clip{suffix}", n_frames=10, codec=codec)
    assert len(list(iter_frames(clip))) == 10


def test_frames_preserve_content_order(tmp_path):
    # Frame i is filled with intensity i; decoded means must be increasing.
    clip = write_video(tmp_path / "clip.avi", n_frames=10, codec="MJPG")
    means = [f.mean() for f in iter_frames(clip)]
    assert all(b > a for a, b in pairwise(means))


# ----------------------------------------------------------- resample_indices


def test_resample_identity():
    np.testing.assert_array_equal(resample_indices(5, src_fps=30, dst_fps=30), np.arange(5))


def test_resample_downsample_halves_frames():
    indices = resample_indices(60, src_fps=60, dst_fps=30)
    assert len(indices) == 30
    np.testing.assert_array_equal(indices, np.arange(30) * 2)


def test_resample_upsample_duplicates_frames():
    indices = resample_indices(30, src_fps=15, dst_fps=30)
    assert len(indices) == 60
    # Each source frame appears roughly twice; order is non-decreasing.
    assert np.all(np.diff(indices) >= 0)
    assert indices[-1] == 29


def test_resample_preserves_duration():
    # 90 frames @ 20 fps = 4.5 s → 135 frames @ 30 fps.
    assert len(resample_indices(90, src_fps=20, dst_fps=30)) == 135


def test_resample_indices_stay_in_bounds():
    indices = resample_indices(7, src_fps=24, dst_fps=30)
    assert indices.min() >= 0
    assert indices.max() <= 6


def test_resample_empty_input():
    assert len(resample_indices(0, src_fps=30, dst_fps=30)) == 0


def test_resample_rejects_bad_fps():
    with pytest.raises(ValueError, match="positive"):
        resample_indices(10, src_fps=0, dst_fps=30)
    with pytest.raises(ValueError, match="positive"):
        resample_indices(10, src_fps=30, dst_fps=-1)


# ----------------------------------------------------------------- load_frames


def test_load_frames_resamples_to_target_fps(tmp_path):
    # 40 frames @ 20 fps = 2 s → 60 frames at TARGET_FPS=30.
    clip = write_video(tmp_path / "clip.mp4", n_frames=40, fps=20.0)
    frames, meta = load_frames(clip)
    assert frames.shape == (int(2 * TARGET_FPS), 48, 64, 3)
    assert meta.fps == pytest.approx(20.0)


def test_load_frames_no_resampling(tmp_path):
    clip = write_video(tmp_path / "clip.mp4", n_frames=12)
    frames, _ = load_frames(clip, target_fps=None)
    assert frames.shape[0] == 12


def test_load_frames_respects_max_frames(tmp_path):
    clip = write_video(tmp_path / "clip.mp4", n_frames=30, fps=30.0)
    frames, _ = load_frames(clip, max_frames=5)
    assert frames.shape[0] == 5


def test_load_frames_portrait_shape(tmp_path):
    clip = write_video(tmp_path / "portrait.mp4", size=(48, 64), fps=30.0)
    frames, meta = load_frames(clip)
    assert frames.shape[1:] == (64, 48, 3)  # (H, W, 3) taller than wide
    assert meta.is_portrait
