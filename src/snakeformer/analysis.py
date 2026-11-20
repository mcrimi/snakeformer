"""Measurements behind the report and the page.

Everything here takes a trained model and returns plain numbers. `build`
gathers them for every model found on disk and writes experiments.json.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import torch

from . import ENCODINGS, FRAME
from .data import Example, generate
from .encoding import PROMPT_LEN, Encoding
from .engine import SIZE, free_cells
from .evaluate import by_kind, exact, games
from .experiments import RUNS, datasets
from .export import rounded
from .model import GPT, Config, load, load_original
from .train import device

BUCKETS = [(3, 9), (10, 19), (20, 29), (30, 39), (40, 59), (60, 256)]


def wilson(right: int, n: int, z: float = 1.96) -> list[float]:
    """95% interval for a share."""
    if n == 0:
        return [0.0, 1.0]
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def share(flags) -> dict:
    flags = list(flags)
    right, n = sum(flags), len(flags)
    lo, hi = wilson(right, n)
    return {"right": right, "n": n, "share": right / n if n else None, "lo": lo, "hi": hi}


def bucket(length: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= length <= hi:
            return f"{lo}–{hi}" if hi < 256 else f"{lo}+"
    raise ValueError(length)


def published(root=".") -> GPT | None:
    """The checkpoint from Hugging Face, if it has been downloaded."""
    folder = Path(root) / "model" / "weights"
    if not (folder / "snake_model.pt").exists():
        return None
    meta = pickle.loads((folder / "meta.pkl").read_bytes())
    dims = {k: meta[k] for k in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")}
    state = torch.load(folder / "snake_model.pt", map_location="cpu", weights_only=True)
    return load_original(state, Config(**dims)).eval()


def stress(model, enc: Encoding, test: list[Example]) -> dict:
    """Held-out positions chosen to be hard: long snakes, collisions, meals."""
    right = exact(model, test, enc)
    kinds = {k: {**v, **dict(zip(("lo", "hi"), wilson(v["right"], v["n"]), strict=True))}
             for k, v in by_kind(test, right).items()}
    by_length = defaultdict(lambda: defaultdict(list))
    for ex, good in zip(test, right, strict=True):
        if ex.kind == "move":
            by_length[bucket(len(ex.state))]["ambiguous" if ex.ambiguous else "determined"].append(good)
    order = [bucket(lo) for lo, _ in BUCKETS]
    return {"kinds": kinds,
            "length": [{"length": b, **{k: share(by_length[b][k]) for k in ("determined", "ambiguous")}}
                       for b in order if b in by_length]}


def ambiguity(n: int = 20_000, seed: int = 11) -> list[dict]:
    """Of ordinary moves, the share whose H/O/# picture has two readings, by length.
    A fact about the encoding; no model is involved."""
    tally = defaultdict(list)
    for ex in generate(n, seed=seed):
        if ex.kind == "move":
            tally[bucket(len(ex.state))].append(ex.ambiguous)
    return [{"length": bucket(lo), **share(tally[bucket(lo)])} for lo, _ in BUCKETS if bucket(lo) in tally]


def in_play(model, enc: Encoding, played) -> dict:
    """Frames from whole games played by the rules. While every frame so far
    is right, the model's game and the real one are the same game, so the
    first wrong frame is where a game played from the model's frames departs."""
    flat = [Example(s, a, n, "", False) for moves in played for s, a, n in moves]
    right = exact(model, flat, enc)
    first, at = [], 0
    for moves in played:
        flags = right[at : at + len(moves)]
        at += len(moves)
        first.append(flags.index(False) + 1 if False in flags else None)
    longest = max(len(m) for m in played)
    alive = [sum(1 for f in first if f is None or f > t) / len(played) for t in range(longest + 1)]
    clean = sum(f is None for f in first)
    return {"frames": share(right), "games": len(played), "clean_games": clean,
            "first_error": first, "alive": alive,
            "median_first_error": sorted(f or 10**9 for f in first)[len(first) // 2]}


@torch.no_grad()
def food_hazard(model, enc: Encoding, n: int = 400, seed: int = 21) -> dict:
    """Does a model trained on randomly placed food learn the odds?

    After a meal the new food is equally likely to be in any free cell. Reading
    the new frame cell by cell, with k free cells still to come and no food
    yet, the chance that this cell holds it is 1/k. Feed the model the new
    frame without food and read off the chance it gives to F at each free cell.
    """
    meals = [e for e in generate(40 * n, seed=seed, food_at="random") if e.kind == "eat"][:n]
    dev = next(model.parameters()).device
    f = enc.stoi["F"]
    pairs = []
    for ex in meals:
        bare = type(ex.next)(ex.next.snake, None)
        seq = enc.encode(enc.prompt(ex.state, ex.action) + enc.target(bare))
        probs = model(torch.tensor([seq[:-1]], device=dev))[0].softmax(-1)[:, f].cpu()
        free = free_cells(bare.snake)
        for i, (r, c) in enumerate(free):
            pairs.append((len(free) - i, float(probs[PROMPT_LEN + r * (SIZE + 1) + c - 1])))
    groups = defaultdict(list)
    for k, p in pairs:
        groups[k].append(p)
    edges = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 257]
    points = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        ks = [k for k in groups if lo <= k < hi]
        if ks:
            count = sum(len(groups[k]) for k in ks)
            points.append({"ideal": sum(len(groups[k]) / k for k in ks) / count,
                           "model": sum(sum(groups[k]) for k in ks) / count, "n": count})
    return {"meals": len(meals), "points": points}


@torch.no_grad()
def food_samples(model, enc: Encoding, n: int = 300, seed: int = 22) -> dict:
    """Write whole frames by drawing each token from the model's distribution,
    and see whether the frames are legal and where the food lands."""
    meals = [e for e in generate(40 * n, seed=seed, food_at="random") if e.kind == "eat"][:n]
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    ids = torch.tensor([enc.encode(enc.prompt(e.state, e.action)) for e in meals], device=dev)
    for _ in range(len(enc.target(meals[0].next))):
        nxt = torch.multinomial(model(ids)[:, -1].softmax(-1), 1)
        ids = torch.cat([ids, nxt], 1)
    legal, ranks = 0, []
    for ex, row in zip(meals, ids[:, PROMPT_LEN:].tolist(), strict=True):
        text = enc.decode(row)
        bare = type(ex.next)(ex.next.snake, None)
        free = free_cells(bare.snake)
        for i, cell in enumerate(free):
            if text == enc.target(type(ex.next)(ex.next.snake, cell)):
                legal += 1
                ranks.append((i + 0.5) / len(free))
                break
    tenths = [0] * 10
    for u in ranks:
        tenths[min(9, int(u * 10))] += 1
    return {"frames": n, **{k: v for k, v in share([True] * legal + [False] * (n - legal)).items()},
            "tenths": tenths}


def measure(model, enc: Encoding, test, played, name: str) -> dict:
    model = rounded(model).to(device())  # as shipped to the browser
    out = {"name": name, "encoding": enc.name, "params": model.n_params(),
           "layers": model.config.n_layer, "width": model.config.n_embd, "heads": model.config.n_head,
           "stress": stress(model, enc, test), "play": in_play(model, enc, played)}
    model.cpu()
    return out


def build(root=".", out="web/data/experiments.json", n_games: int = 200) -> dict:
    root = Path(root)
    _, test = datasets()
    played = games(n_games, seed=7)
    lengths = [len(s) for moves in played for s, _, _ in moves]
    result = {"test_n": len(test), "games": {"n": n_games, "frames": len(lengths),
                                             "mean_length": sum(lengths) / len(lengths),
                                             "max_length": max(lengths)},
              "ambiguity": ambiguity(), "models": {}, "runs": {}}
    original = published(root)
    if original is not None:
        result["models"]["original"] = measure(original, FRAME, test, played, "original")
    for name, (enc_name, mode, food) in RUNS.items():
        folder = root / "runs" / name
        if not (folder / "model.pt").exists():
            continue
        log = json.loads((folder / "log.json").read_text())
        model, _ = load(folder / "model.pt")
        enc = ENCODINGS[enc_name]
        entry = {"encoding": enc_name, "mode": mode, "food": food or "rule",
                 "steps": log["steps"], "train_n": log["train_n"],
                 "curve": [{k: row[k] for k in ("step", "all", "move", "move, determined",
                                                 "move, ambiguous", "eat", "reversal", "self", "wall")
                            if k in row} for row in log["log"]]}
        if food == "random":
            model = model.to(device()).eval()
            _, random_test = datasets("random", train_n=log["train_n"])
            entry["stress"] = stress(model, enc, random_test)
            entry["hazard"] = food_hazard(model, enc)
            entry["samples"] = food_samples(model, enc)
            model.cpu()
        else:
            result["models"][name] = measure(model, enc, test, played, name)
        result["runs"][name] = entry
    extra = root / "runs" / "published-context.json"
    if extra.exists():
        result["context"] = json.loads(extra.read_text())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, separators=(",", ":")))
    return result
