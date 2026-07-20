"""Video loading, frame extraction, and FPS resampling.

The pose pipeline expects frames at a fixed rate so that keypoint sequences
have a consistent temporal resolution regardless of the source device
(phones commonly record at 24/30/60 fps, THETIS clips at 20 fps). Everything
downstream — stroke segmentation windows, the ``(T=60, 99)`` model input —
assumes ``TARGET_FPS``.

OpenCV's ``VideoCapture`` handles container/codec quirks (including rotation
metadata on modern builds, so portrait phone footage arrives right side up).
We deliberately keep resampling as a pure index-selection function
(:func:`resample_indices`) so it can be unit-tested without any video files.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Fixed frame rate every video is resampled to before pose estimation.
TARGET_FPS: float = 30.0

#: Fallback when a container reports a nonsensical FPS (0, NaN, or absurdly
#: high values occasionally reported by broken muxers).
_DEFAULT_FPS: float = 30.0
_MAX_PLAUSIBLE_FPS: float = 240.0


class VideoReadError(RuntimeError):
    """Raised when a video file is missing, unreadable, or contains no frames."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties of a video file.

    Attributes:
        path: Source file path.
        fps: Frames per second (sanitized; see :func:`_sanitize_fps`).
        frame_count: Number of frames reported by the container. May be
            approximate for some codecs — treat as a hint, not ground truth.
        width: Frame width in pixels, after any rotation is applied.
        height: Frame height in pixels, after any rotation is applied.
    """

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Approximate duration in seconds (``frame_count / fps``)."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def is_portrait(self) -> bool:
        """Whether the video is taller than it is wide (typical phone footage)."""
        return self.height > self.width


def _sanitize_fps(raw_fps: float) -> float:
    """Return a plausible FPS value, falling back to a default when broken.

    Args:
        raw_fps: FPS as reported by the container.

    Returns:
        ``raw_fps`` if it is finite and within ``(0, _MAX_PLAUSIBLE_FPS]``,
        otherwise ``_DEFAULT_FPS``.
    """
    if not np.isfinite(raw_fps) or raw_fps <= 0 or raw_fps > _MAX_PLAUSIBLE_FPS:
        return _DEFAULT_FPS
    return float(raw_fps)


def _open_capture(path: Path) -> cv2.VideoCapture:
    """Open a video file, raising a clear error if it cannot be read.

    Args:
        path: Video file path.

    Returns:
        An opened ``cv2.VideoCapture``.

    Raises:
        VideoReadError: If the file does not exist or OpenCV cannot open it.
    """
    if not path.exists():
        raise VideoReadError(f"video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(f"could not open video (unsupported or corrupt): {path}")
    return capture


def probe_video(path: Path | str) -> VideoMetadata:
    """Read a video's metadata without decoding all frames.

    Args:
        path: Video file path.

    Returns:
        The video's :class:`VideoMetadata`.

    Raises:
        VideoReadError: If the file cannot be opened.
    """
    path = Path(path)
    capture = _open_capture(path)
    try:
        fps = _sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            # Some containers only expose dimensions after decoding a frame.
            ok, frame = capture.read()
            if not ok:
                raise VideoReadError(f"video has no decodable frames: {path}")
            height, width = frame.shape[:2]
    finally:
        capture.release()
    return VideoMetadata(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def iter_frames(path: Path | str) -> Iterator[np.ndarray]:
    """Yield every frame of a video as a BGR ``uint8`` array of shape (H, W, 3).

    Args:
        path: Video file path.

    Yields:
        Frames in decode order.

    Raises:
        VideoReadError: If the file cannot be opened or yields no frames.
    """
    capture = _open_capture(Path(path))
    try:
        got_any = False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            got_any = True
            yield frame
        if not got_any:
            raise VideoReadError(f"video has no decodable frames: {path}")
    finally:
        capture.release()


def resample_indices(n_frames: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Select source-frame indices that resample a clip to a new frame rate.

    Maps each output timestamp ``k / dst_fps`` to the nearest source frame
    ``round(k * src_fps / dst_fps)``. Upsampling duplicates frames;
    downsampling drops them. Output covers the same duration as the input.

    Args:
        n_frames: Number of frames in the source clip.
        src_fps: Source frame rate (must be > 0).
        dst_fps: Desired frame rate (must be > 0).

    Returns:
        1-D int array of source indices, one per output frame. Empty when
        ``n_frames == 0``.

    Raises:
        ValueError: If either frame rate is not positive.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"frame rates must be positive, got src={src_fps}, dst={dst_fps}")
    if n_frames <= 0:
        return np.empty(0, dtype=np.int64)
    duration_s = n_frames / src_fps
    n_out = max(1, round(duration_s * dst_fps))
    # Nearest-neighbor mapping from output timestamps to source frames.
    indices = np.round(np.arange(n_out) * src_fps / dst_fps).astype(np.int64)
    return np.clip(indices, 0, n_frames - 1)


def load_frames(
    path: Path | str,
    target_fps: float | None = TARGET_FPS,
    max_frames: int | None = None,
) -> tuple[np.ndarray, VideoMetadata]:
    """Load a video's frames into memory, optionally resampled to a fixed FPS.

    Args:
        path: Video file path.
        target_fps: Resample to this frame rate; ``None`` keeps every frame.
        max_frames: Cap on the number of *output* frames (applied after
            resampling); ``None`` for no cap. Guards memory on long clips.

    Returns:
        Tuple of ``(frames, metadata)`` where ``frames`` is a ``uint8`` array
        of shape ``(N, H, W, 3)`` in BGR order.

    Raises:
        VideoReadError: If the file cannot be opened or yields no frames.
    """
    path = Path(path)
    metadata = probe_video(path)
    frames = list(iter_frames(path))

    if target_fps is not None:
        indices = resample_indices(len(frames), metadata.fps, target_fps)
        frames = [frames[i] for i in indices]
    if max_frames is not None:
        frames = frames[:max_frames]
    return np.stack(frames), metadata
