"""Skeleton overlay drawing and annotated-video output.

Pure rendering: these functions consume the ``(T, 33, 3)`` keypoint arrays
produced by :class:`tennis_analyzer.pose.PoseEstimator` and never run any ML
themselves, so they are cheap to test with hand-crafted keypoints. The one
exception is :func:`annotate_video`, a convenience wrapper that chains
loading -> pose estimation -> drawing -> writing for quick visual checks
from a REPL (the CLI wires this up properly on Day 20).

Keypoint conventions (see ``pose.estimator``): x/y normalized to [0, 1],
NaN coordinates or low visibility mean "don't draw this landmark".
"""

from pathlib import Path

import cv2
import numpy as np

from tennis_analyzer.pose.landmarks import SKELETON_CONNECTIONS
from tennis_analyzer.video.io import TARGET_FPS

#: Minimum landmark visibility for it to be drawn.
DEFAULT_VISIBILITY_THRESHOLD: float = 0.5

#: BGR color of skeleton bones.
BONE_COLOR: tuple[int, int, int] = (80, 220, 80)

#: BGR color of joint markers.
JOINT_COLOR: tuple[int, int, int] = (60, 100, 255)


def _visible_pixel_coords(
    keypoints: np.ndarray, width: int, height: int, visibility_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized keypoints to pixel coordinates plus a draw mask.

    Args:
        keypoints: ``(33, 3)`` array of ``(x, y, visibility)``.
        width: Frame width in pixels.
        height: Frame height in pixels.
        visibility_threshold: Landmarks below this visibility are masked out.

    Returns:
        Tuple ``(points, drawable)``: ``(33, 2)`` int array of pixel
        coordinates and a ``(33,)`` bool mask of landmarks safe to draw
        (visible and finite).
    """
    xy = keypoints[:, :2]
    visibility = keypoints[:, 2]
    drawable = np.isfinite(xy).all(axis=1) & (visibility >= visibility_threshold)
    points = np.zeros((len(keypoints), 2), dtype=np.int64)
    if drawable.any():
        scaled = xy[drawable] * np.array([width, height], dtype=np.float64)
        # Clip so slightly-out-of-frame landmarks (MediaPipe extrapolates
        # beyond image borders) still draw at the frame edge.
        points[drawable] = np.clip(np.round(scaled), 0, np.array([width - 1, height - 1])).astype(
            np.int64
        )
    return points, drawable


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> np.ndarray:
    """Return a copy of a frame with the pose skeleton drawn on it.

    Bones are drawn only when both endpoint landmarks are visible; a frame
    with no detection (all-NaN keypoints) is returned unchanged.

    Args:
        frame: ``(H, W, 3)`` BGR uint8 frame.
        keypoints: ``(33, 3)`` array of ``(x, y, visibility)``.
        visibility_threshold: Minimum visibility for a landmark to be drawn.

    Returns:
        A new ``(H, W, 3)`` frame with the overlay; the input is not modified.
    """
    out = frame.copy()
    height, width = frame.shape[:2]
    points, drawable = _visible_pixel_coords(keypoints, width, height, visibility_threshold)
    if not drawable.any():
        return out

    # Scale line/marker size with resolution so overlays look right on both
    # 480p THETIS clips and 1080p phone footage.
    thickness = max(1, round(min(width, height) / 200))
    radius = thickness + 1

    for start, end in SKELETON_CONNECTIONS:
        if drawable[start] and drawable[end]:
            cv2.line(out, tuple(points[start]), tuple(points[end]), BONE_COLOR, thickness)
    for idx in np.flatnonzero(drawable):
        cv2.circle(out, tuple(points[idx]), radius, JOINT_COLOR, cv2.FILLED)
    return out


def annotate_frames(
    frames: np.ndarray,
    keypoints: np.ndarray,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> np.ndarray:
    """Draw skeletons on a whole frame sequence.

    Args:
        frames: ``(T, H, W, 3)`` BGR uint8 frames.
        keypoints: ``(T, 33, 3)`` keypoint array aligned with ``frames``.
        visibility_threshold: Minimum visibility for a landmark to be drawn.

    Returns:
        New ``(T, H, W, 3)`` array with overlays.

    Raises:
        ValueError: If ``frames`` and ``keypoints`` lengths differ.
    """
    if len(frames) != len(keypoints):
        raise ValueError(
            f"frames ({len(frames)}) and keypoints ({len(keypoints)}) must have equal length"
        )
    return np.stack(
        [
            draw_skeleton(frame, kpts, visibility_threshold)
            for frame, kpts in zip(frames, keypoints, strict=True)
        ]
    )


def write_video(
    frames: np.ndarray,
    path: Path | str,
    fps: float = TARGET_FPS,
    fourcc: str = "mp4v",
) -> Path:
    """Write a frame sequence to a video file.

    Args:
        frames: ``(T, H, W, 3)`` BGR uint8 frames.
        path: Output file path; parent directories are created.
        fps: Frame rate of the output video.
        fourcc: Four-character codec code (``mp4v`` pairs with ``.mp4``).

    Returns:
        The output path.

    Raises:
        ValueError: If ``frames`` is empty.
        OSError: If the video writer cannot be opened for ``path``.
    """
    if len(frames) == 0:
        raise ValueError("cannot write a video with zero frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open video writer for {path} (codec {fourcc})")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def annotate_video(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str | None = None,
) -> tuple[Path, np.ndarray]:
    """Run pose estimation on a video and save a skeleton-overlay copy.

    End-to-end convenience: load -> resample to :data:`TARGET_FPS` -> estimate
    poses -> draw -> write.

    Args:
        input_path: Source video file.
        output_path: Destination for the annotated video (``.mp4`` suggested).
        model_path: Optional explicit pose model file; downloaded to the
            cache otherwise.

    Returns:
        Tuple ``(output_path, keypoints)`` where ``keypoints`` is the
        ``(T, 33, 3)`` array, so callers can reuse it without re-running pose.

    Raises:
        VideoReadError: If the input cannot be read.
        PoseModelError: If the pose model cannot be obtained.
    """
    # Imported here: drawing utilities above stay usable without the pose
    # model, and module import stays cycle-free (pose.estimator imports
    # video.io).
    from tennis_analyzer.pose.estimator import PoseEstimator
    from tennis_analyzer.video.io import load_frames

    frames, _ = load_frames(input_path, target_fps=TARGET_FPS)
    with PoseEstimator(model_path=model_path) as estimator:
        keypoints = estimator.estimate(frames, fps=TARGET_FPS)
    annotated = annotate_frames(frames, keypoints)
    return write_video(annotated, output_path, fps=TARGET_FPS), keypoints
