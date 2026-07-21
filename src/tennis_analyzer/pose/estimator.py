"""MediaPipe pose estimation over frame sequences.

Wraps the MediaPipe Tasks ``PoseLandmarker`` (the legacy ``mp.solutions.pose``
API was removed from recent MediaPipe releases) in VIDEO running mode, which
uses temporal tracking between frames — both faster and more stable than
per-image detection for continuous footage.

Output convention, consumed by everything downstream (normalization,
segmentation, the ``(T=60, 99)`` model input):

- shape ``(T, 33, 3)`` float32: T frames x 33 BlazePose landmarks x
  ``(x, y, visibility)``
- ``x``/``y`` are normalized to ``[0, 1]`` relative to frame width/height
- frames where no person is detected get ``x = y = NaN`` and
  ``visibility = 0.0``, so downstream code can mask them out explicitly
  instead of silently consuming garbage coordinates

The landmarker needs a model file (~5 MB). :func:`ensure_model` downloads it
once into a local cache directory (override with ``$TENNIS_ANALYZER_CACHE``).
"""

import os
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from tennis_analyzer.pose.landmarks import LANDMARK_DIMS, N_LANDMARKS
from tennis_analyzer.video.io import TARGET_FPS

#: Pinned model version (not ``latest``) for reproducible results.
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

#: Filename of the cached model asset.
MODEL_FILENAME: str = "pose_landmarker_lite.task"

#: Environment variable overriding the cache directory.
CACHE_ENV_VAR: str = "TENNIS_ANALYZER_CACHE"


class PoseModelError(RuntimeError):
    """Raised when the pose model asset is missing and cannot be downloaded."""


def default_cache_dir() -> Path:
    """Return the directory where model assets are cached.

    Returns:
        ``$TENNIS_ANALYZER_CACHE`` if set, otherwise
        ``~/.cache/tennis-analyzer``.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "tennis-analyzer"


def ensure_model(model_path: Path | str | None = None, download: bool = True) -> Path:
    """Return a local path to the pose model, downloading it if necessary.

    Args:
        model_path: Explicit path to a ``.task`` model file. ``None`` uses the
            default cache location.
        download: Whether to download the model when it is not present.

    Returns:
        Path to an existing model file.

    Raises:
        PoseModelError: If the model is absent and ``download`` is False, or
            the download fails (e.g., no network access).
    """
    path = Path(model_path) if model_path is not None else default_cache_dir() / MODEL_FILENAME
    if path.exists():
        return path
    if not download:
        raise PoseModelError(f"pose model not found at {path} (download disabled)")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)  # noqa: S310 - pinned https URL
        tmp_path.replace(path)  # atomic move so a failed download never looks cached
    except (urllib.error.URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise PoseModelError(
            f"could not download pose model from {MODEL_URL}: {exc}. "
            f"Download it manually and pass model_path, or place it at {path}."
        ) from exc
    return path


class PoseEstimator:
    """Extracts BlazePose keypoints from frame sequences.

    Use as a context manager so the underlying MediaPipe graph is released
    deterministically::

        with PoseEstimator() as estimator:
            keypoints = estimator.estimate(frames)

    Attributes:
        model_path: Path to the ``.task`` model asset in use.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Create the estimator and load the MediaPipe graph.

        Args:
            model_path: Optional explicit model file; downloaded to the cache
                otherwise.
            min_detection_confidence: Minimum confidence for the person
                detector to trigger.
            min_tracking_confidence: Minimum confidence to keep tracking a
                person across frames instead of re-running detection.

        Raises:
            PoseModelError: If the model asset cannot be obtained.
        """
        self.model_path = ensure_model(model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,  # v1 targets practice-court clips with one player in frame
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        # VIDEO mode requires strictly increasing timestamps across calls, so
        # a single estimator instance must not interleave two clips.
        self._next_timestamp_ms = 0

    def estimate(self, frames: np.ndarray, fps: float = TARGET_FPS) -> np.ndarray:
        """Extract keypoints for a sequence of frames.

        Args:
            frames: ``(T, H, W, 3)`` BGR uint8 array (as produced by
                :func:`tennis_analyzer.video.load_frames`).
            fps: Frame rate of the sequence; used to derive the millisecond
                timestamps MediaPipe's tracker needs.

        Returns:
            ``(T, 33, 3)`` float32 array of ``(x, y, visibility)`` per
            landmark. Undetected frames are ``(NaN, NaN, 0.0)``.

        Raises:
            ValueError: If ``frames`` has the wrong shape or ``fps <= 0``.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (T, H, W, 3) frames, got shape {frames.shape}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        keypoints = np.full((len(frames), N_LANDMARKS, LANDMARK_DIMS), np.nan, dtype=np.float32)
        keypoints[:, :, 2] = 0.0  # visibility defaults to 0, not NaN

        frame_interval_ms = 1000.0 / fps
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = self._next_timestamp_ms + round(i * frame_interval_ms)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                # num_poses=1, so take the single detected pose.
                for j, landmark in enumerate(result.pose_landmarks[0]):
                    keypoints[i, j] = (landmark.x, landmark.y, landmark.visibility)
        # Keep timestamps strictly increasing if estimate() is called again.
        self._next_timestamp_ms += round(len(frames) * frame_interval_ms) + 1
        return keypoints

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        """Return self for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release resources on context exit."""
        self.close()
