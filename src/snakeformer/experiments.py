"""The training runs behind the report.

Each run is a named recipe. All of them train the same small network on the
same positions for the same number of steps, so that a difference in the
result can be put down to the one thing that was changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ENCODINGS
from .data import generate, keys
from .evaluate import by_kind, exact
from .model import save
from .train import train

TRAIN_N, TEST_N, STEPS = 300_000, 6_000, 20_000

RUNS = {
    # name: (encoding, how sequences are cut, where food appears)
    "arrows-aligned": ("arrows", "aligned", None),
    "frame-aligned": ("frame", "aligned", None),
    "frame-stream": ("frame", "stream", None),
    "arrows-stream": ("arrows", "stream", None),
    "arrows-random-food": ("arrows", "aligned", "random"),
}


def datasets(food=None, train_n=TRAIN_N, test_n=TEST_N):
    """Training positions, and test positions that are not among them."""
    training = generate(train_n, seed=1, food_at=food, check=False)
    test = generate(test_n, seed=2, food_at=food, exclude=keys(training))
    return training, test


def run(name: str, out="runs", steps: int = STEPS, train_n: int = TRAIN_N) -> dict:
    enc_name, mode, food = RUNS[name]
    enc = ENCODINGS[enc_name]
    training, test = datasets(food, train_n)
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    def score(model):
        shares = {k: v["share"] for k, v in by_kind(test, exact(model, test, enc)).items()}
        (out / "log.json").write_text(json.dumps(log + [{"partial": shares}]))
        return shares

    # "stream" takes half as many sequences, each about twice as long, so both
    # ways of training read about the same number of characters per step.
    model = train(training, enc, steps, mode=mode, log=log, on_eval=score, batch=64)
    save(model, out / "model.pt", encoding=enc_name, mode=mode, steps=steps)
    result = {"name": name, "encoding": enc_name, "mode": mode, "food": food or "rule",
              "steps": steps, "train_n": train_n, "params": model.n_params(), "log": log}
    (out / "log.json").write_text(json.dumps(result))
    return result


def polish_run(source: str, out="runs", n_games: int = 300, every: int = 30, root=".",
               mortal: bool = False) -> dict:
    """Shadow-mode corrections over n_games, starting from the published
    checkpoint ("original") or from a finished run, with the effect measured
    as it goes on games and positions the corrections never see."""
    from .analysis import in_play, published
    from .evaluate import games
    from .model import load
    from .online import polish
    from .train import device

    if source == "original":
        model, enc = published(root), ENCODINGS["frame"]
    else:
        model, extra = load(Path(out) / source / "model.pt")
        enc = ENCODINGS[extra["encoding"]]
    model = model.to(device())
    _, test = datasets()
    held_out = games(60, seed=8)
    folder = Path(out) / (f"{source}-polished" + ("-mortal" if mortal else ""))
    folder.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    def score(m):
        play = in_play(m, enc, held_out)
        shares = {k: v["share"] for k, v in by_kind(test, exact(m, test, enc)).items()}
        (folder / "log.json").write_text(json.dumps({"partial": log}))
        return {"play": play["frames"]["share"], "median_first_error": play["median_first_error"],
                "clean_games": play["clean_games"], **shares}

    norms: list = []
    polish(model, enc, n_games, every=every, on_eval=score, log=log, mortal=mortal, norms=norms)
    save(model.cpu(), folder / "model.pt", encoding=enc.name, source=source, games=n_games)
    weighted = sorted(n for w, n in norms if w)
    plain = sorted(n for w, n in norms if not w)
    result = {"source": source, "encoding": enc.name, "games": n_games, "held_out_games": 60,
              "mortal": mortal, "log": log,
              "gradient": {"weighted_steps": len(weighted), "plain_steps": len(plain),
                           "weighted_median": weighted[len(weighted) // 2] if weighted else None,
                           "plain_median": plain[len(plain) // 2] if plain else None,
                           "weighted_clipped": sum(n > 0.5 for n in weighted),
                           "plain_clipped": sum(n > 0.5 for n in plain)}}
    (folder / "log.json").write_text(json.dumps(result))
    return result


def aggregate_run(source: str, out="runs", rounds: int = 3, n_games: int = 600, steps: int = 1500,
                  root=".") -> dict:
    """The other way to learn from mistakes (DAgger): collect them, then retrain.

    Each round the rules play n_games while the model shadows them. Every frame
    it gets wrong is kept. The model is then trained for a short while on
    batches drawn half from all the mistakes kept so far and half from the
    positions it was first trained on, so that what it knew is rehearsed while
    the mistakes are fixed. Scored on the same held-out games and positions as
    `polish_run`."""
    import random

    from .analysis import in_play, published
    from .data import Example
    from .evaluate import games
    from .model import load
    from .train import device

    if source == "original":
        model, enc = published(root), ENCODINGS["frame"]
    else:
        model, extra = load(Path(out) / source / "model.pt")
        enc = ENCODINGS[extra["encoding"]]
    model = model.to(device()).eval()
    training, test = datasets()
    held_out = games(60, seed=8)
    folder = Path(out) / f"{source}-aggregated"
    folder.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    rng = random.Random(5)

    def score(m):
        play = in_play(m, enc, held_out)
        shares = {k: v["share"] for k, v in by_kind(test, exact(m, test, enc)).items()}
        return {"play": play["frames"]["share"], "median_first_error": play["median_first_error"],
                "clean_games": play["clean_games"], **shares}

    def note(done, kept, new, frames):
        log.append({"round": done, "games": done * n_games, "mistakes": kept, "new_mistakes": new,
                    "frames_seen": frames, "steps": done * steps, **score(model)})
        (folder / "log.json").write_text(json.dumps({"partial": log}))
        print(log[-1], flush=True)

    note(0, 0, 0, 0)
    mistakes: list[Example] = []
    for r in range(1, rounds + 1):
        played = games(n_games, seed=1000 + r, mortal=True)
        flat = [Example(s, a, n, "", False) for moves in played for s, a, n in moves]
        new = [ex for ex, good in zip(flat, exact(model, flat, enc), strict=True) if not good]
        mistakes += new
        if mistakes:
            rehearsal = rng.sample(training, min(len(training), max(len(mistakes), 20_000)))
            again = [rng.choice(mistakes) for _ in rehearsal]
            model = train(again + rehearsal, enc, steps, lr=1e-4, seed=r, start=model, every=steps)
        note(r, len(mistakes), len(new), len(flat))
    save(model.cpu(), folder / "model.pt", encoding=enc.name, source=source, rounds=rounds)
    result = {"source": source, "encoding": enc.name, "rounds": rounds, "games_per_round": n_games,
              "steps_per_round": steps, "held_out_games": 60, "log": log}
    (folder / "log.json").write_text(json.dumps(result))
    return result
