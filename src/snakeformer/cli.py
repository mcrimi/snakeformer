"""Command line.

    snakeformer train <run>     train one of the named runs (hours on a laptop)
    snakeformer polish <source> correct a model on its own mistakes during play
    snakeformer build           measure every model on disk and write the page's data
    snakeformer fixtures        re-export the JavaScript test cases from the shipped weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import RUNS, STEPS, TRAIN_N

# Models the page offers: id -> (run, label, note). "original" is the published checkpoint.
SHIPPED = {
    "arrows": ("arrows-aligned", "Retrained, arrows",
               "Trained for this page: 4 layers, 0.9M parameters. Each body segment points to the next."),
    "original": (None, "Published, 2026",
                 "The checkpoint on Hugging Face: 6 layers, 2.9M parameters, boards drawn with H, O and #."),
}


def build(root=".", web="web") -> None:
    from . import ENCODINGS, FRAME
    from .analysis import build as measure_all
    from .analysis import published
    from .export import engine_fixtures, model_fixtures, rounded, write_weights
    from .model import load

    root, web = Path(root), Path(web)
    manifest = []
    for name, (run, label, note) in SHIPPED.items():
        if run is None:
            model, enc = published(root), FRAME
        elif (root / "runs" / run / "model.pt").exists():
            model, extra = load(root / "runs" / run / "model.pt")
            enc = ENCODINGS[extra["encoding"]]
        else:
            model = None
        if model is None:
            print(f"skipping {name}: no weights on disk")
            continue
        model = rounded(model)
        size = write_weights(model, enc, web / "data" / f"{name}.bin", name=name).stat().st_size
        (web / "test" / f"fixtures-{name}.json").write_text(json.dumps(model_fixtures(model, enc, n=4)))
        manifest.append({"id": name, "file": f"{name}.bin", "label": label, "note": note,
                         "bytes": size, "default": len(manifest) == 0})
    (web / "data" / "models.json").write_text(json.dumps(manifest, indent=1))
    (web / "test" / "fixtures-game.json").write_text(json.dumps(engine_fixtures()))
    result = measure_all(root, web / "data" / "experiments.json")
    print("models:", sorted(result["models"]), "runs:", sorted(result["runs"]))


def fixtures(web="web") -> None:
    from . import ENCODINGS
    from .export import engine_fixtures, model_fixtures, read_weights

    web = Path(web)
    for path in sorted((web / "data").glob("*.bin")):
        model, header = read_weights(path)
        cases = model_fixtures(model, ENCODINGS[header["encoding"]], n=4)
        (web / "test" / f"fixtures-{path.stem}.json").write_text(json.dumps(cases))
        print("fixtures for", path.stem)
    (web / "test" / "fixtures-game.json").write_text(json.dumps(engine_fixtures()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="snakeformer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train one of the named runs")
    t.add_argument("run", choices=sorted(RUNS))
    t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--train-n", type=int, default=TRAIN_N)
    t.add_argument("--out", default="runs")
    p = sub.add_parser("polish", help="shadow-mode corrections over many games")
    p.add_argument("source", help='"original" or the name of a finished run')
    p.add_argument("--games", type=int, default=300)
    sub.add_parser("build", help="measure the models and write web/data")
    sub.add_parser("fixtures", help="re-export the JavaScript test cases")
    args = parser.parse_args(argv)
    if args.command == "train":
        from .experiments import run

        result = run(args.run, args.out, args.steps, args.train_n)
        print({k: v for k, v in result.items() if k != "log"}, result["log"][-1])
    elif args.command == "polish":
        from .experiments import polish_run

        print(polish_run(args.source, n_games=args.games)["log"][-1])
    elif args.command == "build":
        build()
    else:
        fixtures()


if __name__ == "__main__":
    main()
