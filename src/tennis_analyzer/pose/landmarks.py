"""BlazePose landmark indices and skeleton topology.

MediaPipe's PoseLandmarker returns 33 body keypoints per frame (the
BlazePose topology). This module pins down that layout as plain constants so
the rest of the codebase — drawing, normalization, joint-angle math — can
refer to landmarks by name without importing MediaPipe.

Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

from enum import IntEnum

#: Number of landmarks in the BlazePose topology.
N_LANDMARKS: int = 33

#: Values stored per landmark: (x, y, visibility).
LANDMARK_DIMS: int = 3


class Landmark(IntEnum):
    """Index of each BlazePose landmark in a ``(33, 3)`` keypoint array."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


#: Bone segments of the BlazePose skeleton as (start, end) landmark pairs.
#: Matches MediaPipe's POSE_CONNECTIONS; used to draw the overlay skeleton.
SKELETON_CONNECTIONS: tuple[tuple[Landmark, Landmark], ...] = (
    # Face
    (Landmark.NOSE, Landmark.LEFT_EYE_INNER),
    (Landmark.LEFT_EYE_INNER, Landmark.LEFT_EYE),
    (Landmark.LEFT_EYE, Landmark.LEFT_EYE_OUTER),
    (Landmark.LEFT_EYE_OUTER, Landmark.LEFT_EAR),
    (Landmark.NOSE, Landmark.RIGHT_EYE_INNER),
    (Landmark.RIGHT_EYE_INNER, Landmark.RIGHT_EYE),
    (Landmark.RIGHT_EYE, Landmark.RIGHT_EYE_OUTER),
    (Landmark.RIGHT_EYE_OUTER, Landmark.RIGHT_EAR),
    (Landmark.MOUTH_LEFT, Landmark.MOUTH_RIGHT),
    # Torso
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    # Left arm and hand
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.LEFT_WRIST, Landmark.LEFT_PINKY),
    (Landmark.LEFT_WRIST, Landmark.LEFT_INDEX),
    (Landmark.LEFT_WRIST, Landmark.LEFT_THUMB),
    (Landmark.LEFT_PINKY, Landmark.LEFT_INDEX),
    # Right arm and hand
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    (Landmark.RIGHT_WRIST, Landmark.RIGHT_PINKY),
    (Landmark.RIGHT_WRIST, Landmark.RIGHT_INDEX),
    (Landmark.RIGHT_WRIST, Landmark.RIGHT_THUMB),
    (Landmark.RIGHT_PINKY, Landmark.RIGHT_INDEX),
    # Left leg and foot
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    # Right leg and foot
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
)
