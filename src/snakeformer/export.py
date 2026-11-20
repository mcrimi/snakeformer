"""Weights and test cases for the browser.

The weights file is a JSON header followed by the tensors as 16-bit floats.
`rounded` returns the model as the browser will see it, so that every number
reported for a shipped model is measured on the rounded weights.
"""

from __future__ import annotations

import copy
import json
import random
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import wander
from .encoding import PROMPT_LEN, Encoding
from .engine import ACTIONS, step
from .model import GPT


def rounded(model: GPT) -> GPT:
    """A copy of the model with every weight rounded to a 16-bit float."""
    twin = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for p in twin.parameters():
            p.copy_(p.half().float())
    return twin.eval()


def write_weights(model: GPT, enc: Encoding, path, **about) -> Path:
    tensors, blobs, offset = [], [], 0
    for name, p in model.state_dict().items():
        data = p.detach().cpu().numpy().astype("<f2")
        tensors.append({"name": name, "shape": list(data.shape), "offset": offset})
        blobs.append(data.tobytes())
        offset += data.size
    header = json.dumps({"config": asdict(model.config), "encoding": enc.name,
                         "chars": enc.chars, "tensors": tensors, **about}).encode()
    header += b" " * (-len(header) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<I", len(header)) + header + b"".join(blobs))
    return path


@torch.no_grad()
def read_weights(path) -> tuple[GPT, dict]:
    """Load a weights file written by write_weights."""
    from .model import Config

    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<I", raw[:4])
    header = json.loads(raw[4 : 4 + n])
    model = GPT(Config(**header["config"]))
    state = {}
    for t in header["tensors"]:
        size = int(np.prod(t["shape"]))
        data = np.frombuffer(raw, "<f2", count=size, offset=4 + n + 2 * t["offset"])
        state[t["name"]] = torch.tensor(data.astype(np.float32)).reshape(t["shape"])
    model.load_state_dict(state)
    return model.eval(), header


@torch.no_grad()
def greedy(model: GPT, enc: Encoding, prompt: str, limit: int = 300) -> str:
    """Generate until the stop token, taking the most likely token each time."""
    dev = next(model.parameters()).device
    ids = enc.encode(prompt)
    stop = enc.stoi["$"]
    for _ in range(limit):
        nxt = int(model(torch.tensor([ids[-model.config.block_size:]], device=dev))[0, -1].argmax())
        ids.append(nxt)
        if nxt == stop:
            break
    return enc.decode(ids[len(prompt):])


@torch.no_grad()
def model_fixtures(model: GPT, enc: Encoding, n: int = 6, seed: int = 0) -> dict:
    """Prompts with the model's own continuation and its logits along the way."""
    rng = random.Random(seed)
    states = [s for _ in range(4) for s in wander(rng, max_steps=120)]
    cases = []
    for state in rng.sample(states, n):
        action = rng.choice(ACTIONS)
        prompt = enc.prompt(state, action)
        output = greedy(model, enc, prompt)
        ids = torch.tensor([enc.encode(prompt + output)])
        logits = model(ids[:, :-1])[0]
        probe = [0, 5, PROMPT_LEN - 1, PROMPT_LEN + 20, ids.shape[1] - 2]
        probe = sorted({p for p in probe if p < logits.shape[0]})
        cases.append({"prompt": prompt, "output": output,
                      "truth": enc.target(step(state, action)[0]),
                      "logits": {str(p): np.round(logits[p].numpy().astype(float), 5).tolist()
                                 for p in probe}})
    return {"encoding": enc.name, "cases": cases}


def engine_fixtures(n_games: int = 6, seed: int = 0) -> list[dict]:
    """Transitions under both encodings, for the JavaScript rules engine."""
    from . import ARROWS, FRAME
    from .ambiguity import new_tail_candidates

    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        states = list(wander(rng, max_steps=60))
        for state in rng.sample(states, min(12, len(states))):
            for action in ACTIONS:
                nxt, event = step(state, action)
                out.append({
                    "snake": [list(c) for c in state.snake],
                    "food": list(state.food) if state.food else None,
                    "action": action, "event": event,
                    "frame": FRAME.board(state), "arrows": ARROWS.board(state),
                    "next_frame": FRAME.target(nxt), "next_arrows": ARROWS.target(nxt),
                    "candidates": sorted(list(c) for c in new_tail_candidates(state)),
                })
    return out
