"""Smoke tests: package imports and CLI argument handling."""

import importlib

import pytest

import tennis_analyzer
from tennis_analyzer.cli import build_parser, main

SUBPACKAGES = [
    "video",
    "pose",
    "segmentation",
    "data",
    "models",
    "training",
    "feedback",
]


def test_version_is_set():
    assert tennis_analyzer.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"tennis_analyzer.{name}")


def test_parser_accepts_analyze():
    args = build_parser().parse_args(["analyze", "clip.mp4"])
    assert args.command == "analyze"
    assert args.video.name == "clip.mp4"


def test_analyze_missing_video_returns_error(tmp_path):
    missing = tmp_path / "nope.mp4"
    assert main(["analyze", str(missing)]) == 2


def test_analyze_existing_video_succeeds(tmp_path, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")  # existence is all the stub checks
    assert main(["analyze", str(clip)]) == 0
    assert "under construction" in capsys.readouterr().out
