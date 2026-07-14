"""Command-line interface for the tennis video analyzer.

Entry point installed as ``tennis-analyzer`` (see ``pyproject.toml``). The
``analyze`` subcommand will run the full pipeline (pose extraction → stroke
segmentation → classification → scoring → feedback) once those modules land;
for now it validates arguments and reports pipeline status.
"""

import argparse
import sys
from pathlib import Path

from tennis_analyzer import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Configured parser with the ``analyze`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="tennis-analyzer",
        description="Analyze tennis stroke videos with pose-based neural networks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a video: classify strokes, score technique, generate feedback.",
    )
    analyze.add_argument("video", type=Path, help="Path to the input video file.")
    analyze.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the annotated video and feedback report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 2 on bad input).
    """
    args = build_parser().parse_args(argv)

    if args.command == "analyze":
        if not args.video.exists():
            print(f"error: video not found: {args.video}", file=sys.stderr)
            return 2
        # Pipeline modules arrive incrementally (see PLAN.md); wired up on Day 20.
        print("Analysis pipeline is under construction — check back soon.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
