"""Tests for the BlazePose landmark constants and skeleton topology."""

from tennis_analyzer.pose import (
    LANDMARK_DIMS,
    N_LANDMARKS,
    SKELETON_CONNECTIONS,
    Landmark,
)


def test_landmark_count_matches_blazepose():
    assert len(Landmark) == N_LANDMARKS == 33
    assert LANDMARK_DIMS == 3


def test_landmark_indices_are_contiguous():
    assert sorted(int(lm) for lm in Landmark) == list(range(33))


def test_connections_reference_valid_landmarks():
    for start, end in SKELETON_CONNECTIONS:
        assert 0 <= start < N_LANDMARKS
        assert 0 <= end < N_LANDMARKS
        assert start != end


def test_connections_are_unique():
    normalized = {tuple(sorted(pair)) for pair in SKELETON_CONNECTIONS}
    assert len(normalized) == len(SKELETON_CONNECTIONS)


def test_skeleton_is_left_right_symmetric():
    # Every LEFT_* bone should have a RIGHT_* mirror and vice versa.
    def mirror(name: str) -> str:
        if name.startswith("LEFT_"):
            return "RIGHT_" + name[5:]
        if name.startswith("RIGHT_"):
            return "LEFT_" + name[6:]
        return name

    connections = {(s.name, e.name) for s, e in SKELETON_CONNECTIONS}
    for start, end in connections:
        mirrored = (mirror(start), mirror(end))
        assert mirrored in connections or tuple(reversed(mirrored)) in connections


def test_key_tennis_joints_present():
    # The joints the feedback layer (elbow/shoulder/hip/knee angles) relies on.
    for name in (
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
    ):
        assert hasattr(Landmark, name)
