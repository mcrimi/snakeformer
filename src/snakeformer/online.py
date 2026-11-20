"""Learning from mistakes during play: the original Shadow Mode, without the terminal.

The rules engine plays alongside the model. At the first character where the
model's frame departs from the true one, take a gradient step on exactly that
context and that character, and try again, up to ten times. Mistakes that
involve the head, the food or a death count fifty times over. This is the
update in games/shadow_neural_snake.py; here it runs over many games so that
its effect can be measured.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .encoding import PROMPT_LEN, Encoding
from .evaluate import games

CRITICAL = "HFX"


@torch.no_grad()
def first_slips(model, enc: Encoding, moves, batch: int = 128):
    """For each move of a game, the index in the target of the first wrong
    character, or None if the frame is right."""
    dev = next(model.parameters()).device
    out = [None] * len(moves)
    by_len = {}
    for i, (state, action, nxt) in enumerate(moves):
        seq = enc.encode(enc.prompt(state, action) + enc.target(nxt)[:-1])
        by_len.setdefault(len(seq), []).append((i, seq))
    for items in by_len.values():
        for at in range(0, len(items), batch):
            chunk = items[at : at + batch]
            x = torch.tensor([s for _, s in chunk], device=dev)
            wrong = model(x[:, :-1]).argmax(-1)[:, PROMPT_LEN - 1 :] != x[:, PROMPT_LEN:]
            for (i, _), row in zip(chunk, wrong, strict=True):
                if row.any():
                    out[i] = int(row.float().argmax())
    return out


def correct(model, opt, enc: Encoding, move, attempts: int = 10, weight: float = 50.0,
            norms: list | None = None) -> int:
    """Train on one move until its frame is right or the attempts run out.
    Returns the number of gradient steps taken. `norms` collects, per step,
    (whether the mistake was weighted, the gradient's size before clipping)."""
    dev = next(model.parameters()).device
    state, action, nxt = move
    seq = torch.tensor([enc.encode(enc.prompt(state, action) + enc.target(nxt)[:-1])], device=dev)
    steps = 0
    while steps < attempts:
        slip = first_slips(model, enc, [move])[0]
        if slip is None:
            break
        at = PROMPT_LEN + slip
        model.train()
        logits = model(seq[:, :at])[0, -1]
        truth, said = enc.chars[int(seq[0, at])], enc.chars[int(logits.argmax())]
        scale = weight if truth in CRITICAL or said in CRITICAL else 1.0
        loss = scale * F.cross_entropy(logits[None], seq[:, at])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        size = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if norms is not None:
            norms.append((scale > 1.0, float(size)))
        opt.step()
        model.eval()
        steps += 1
    return steps


def polish(model, enc: Encoding, n_games: int, seed: int = 100, lr: float = 1e-4,
           every: int = 10, on_eval=None, log=None, mortal: bool = False, norms: list | None = None):
    """Play n_games with corrections switched on. `on_eval(model)` is called
    before the first game and after every `every` games."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.eval()
    total_steps = total_slips = 0

    def note(done):
        row = {"games": done, "corrections": total_slips, "steps": total_steps}
        if on_eval:
            row.update(on_eval(model))
        if log is not None:
            log.append(row)
        print(row, flush=True)

    note(0)
    for g, moves in enumerate(games(n_games, seed=seed, mortal=mortal), 1):
        start = 0
        while start < len(moves):
            slips = first_slips(model, enc, moves[start:])
            wrong = next((i for i, s in enumerate(slips) if s is not None), None)
            if wrong is None:
                break
            total_slips += 1
            total_steps += correct(model, opt, enc, moves[start + wrong], norms=norms)
            start += wrong + 1
        if g % every == 0 or g == n_games:
            note(g)
    return model
