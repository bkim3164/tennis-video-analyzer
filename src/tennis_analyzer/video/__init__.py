"""Video I/O: loading, frame extraction, FPS resampling, output annotation."""

from tennis_analyzer.video.annotate import (
    annotate_frames,
    annotate_video,
    draw_skeleton,
    write_video,
)
from tennis_analyzer.video.io import (
    TARGET_FPS,
    VideoMetadata,
    VideoReadError,
    iter_frames,
    load_frames,
    probe_video,
    resample_indices,
)

__all__ = [
    "TARGET_FPS",
    "VideoMetadata",
    "VideoReadError",
    "annotate_frames",
    "annotate_video",
    "draw_skeleton",
    "iter_frames",
    "load_frames",
    "probe_video",
    "resample_indices",
    "write_video",
]
