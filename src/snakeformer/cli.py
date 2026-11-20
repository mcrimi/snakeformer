"""Command line: `snakeformer train <run>` and friends."""

from __future__ import annotations

import argparse

from .experiments import RUNS, STEPS, TRAIN_N


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="snakeformer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train one of the named runs")
    t.add_argument("run", choices=sorted(RUNS))
    t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--train-n", type=int, default=TRAIN_N)
    t.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    if args.command == "train":
        from .experiments import run

        result = run(args.run, args.out, args.steps, args.train_n)
        print({k: v for k, v in result.items() if k != "log"}, result["log"][-1])


if __name__ == "__main__":
    main()
