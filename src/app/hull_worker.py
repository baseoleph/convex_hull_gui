"""Subprocess worker: reads a points JSON file, computes a hull, writes a golden JSON file."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from app.service import get_algorithm, load_points_from_file
from verification.golden import dump_golden


def main() -> None:
    """Read points JSON, compute hull with the named algorithm, write golden JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        points = load_points_from_file(args.input)
        algo = get_algorithm(args.algorithm)
        result = algo.compute(points, seed=args.seed)
        dump_golden(result, args.output)
    except Exception as exc:
        print(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
