#!/usr/bin/env python3
"""Run one of the three PulseDB modes. Paths and behavior are YAML-controlled."""

import argparse
import sys
from pathlib import Path
from pulsedb.config import load_config
from pulsedb.experiment import run, ReviewRequired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/pulsedb.yaml"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML key, e.g. --set experiment.mode=backbone (repeatable)",
    )
    args = parser.parse_args()
    try:
        run(load_config(args.config, args.set))
    except ReviewRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
