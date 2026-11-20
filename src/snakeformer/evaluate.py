"""How often is the whole next frame right?

Greedy decoding gets a frame exactly right if and only if, with the true
frame fed in, the most likely token at every position is the true one. So one
forward pass per example gives the same answer as generating it.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .encoding import PROMPT_LEN, Encoding


@torch.no_grad()
def exact(model, examples, enc: Encoding, prefix=None, batch: int = 256) -> list[bool]:
    """For each example, whether greedy decoding reproduces the true next frame.

    `prefix(example)` may put text before the prompt, to test what earlier
    context does to the answer.
    """
    dev = next(model.parameters()).device
    block = model.config.block_size
    rows = []
    for i, ex in enumerate(examples):
        pre = prefix(ex) if prefix else ""
        target = enc.target(ex.next)
        rows.append((len(target), i, enc.encode(pre + enc.prompt(ex.state, ex.action) + target)))
    groups = defaultdict(list)
    for n_target, i, seq in rows:
        seq = seq[-(block + 1):]
        groups[(len(seq), n_target)].append((i, seq))
    right = [False] * len(examples)
    for (_, n_target), items in groups.items():
        for at in range(0, len(items), batch):
            chunk = items[at : at + batch]
            x = torch.tensor([seq for _, seq in chunk], device=dev)
            pred = model(x[:, :-1]).argmax(-1)
            ok = (pred[:, -n_target:] == x[:, -n_target:]).all(1).tolist()
            for (i, _), good in zip(chunk, ok, strict=False):
                right[i] = good
    return right


def by_kind(examples, right) -> dict:
    """Share of frames exactly right, overall and for each kind of example."""
    tally = defaultdict(lambda: [0, 0])
    for ex, good in zip(examples, right, strict=False):
        names = ["all", ex.kind]
        if ex.kind == "move":
            names.append("move, ambiguous" if ex.ambiguous else "move, determined")
        for name in names:
            tally[name][0] += good
            tally[name][1] += 1
    return {name: {"right": r, "n": n, "share": r / n} for name, (r, n) in sorted(tally.items())}


assert PROMPT_LEN  # re-exported for callers that slice sequences


@torch.no_grad()
def generated(model, enc: Encoding, prompts: list[str], limit: int = 276, batch: int = 64):
    """Greedy continuations, generated the way the terminal game does it: the
    whole context is run again for every token and, once it is longer than the
    model's window, the oldest tokens fall off the front."""
    dev = next(model.parameters()).device
    block, stop = model.config.block_size, enc.stoi["$"]
    out = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for at in range(0, len(order), batch):
        chunk = [i for i in order[at : at + batch]]
        by_len = defaultdict(list)
        for i in chunk:
            by_len[len(prompts[i])].append(i)
        for _, group in by_len.items():
            ids = torch.tensor([enc.encode(prompts[i]) for i in group], device=dev)
            start, done = ids.shape[1], torch.zeros(len(group), dtype=torch.bool, device=dev)
            for _ in range(limit):
                nxt = model(ids[:, -block:])[:, -1].argmax(-1)
                ids = torch.cat([ids, nxt[:, None]], 1)
                done |= nxt == stop
                if done.all():
                    break
            for i, row in zip(group, ids[:, start:].tolist(), strict=True):
                text = enc.decode(row)
                out[i] = text[: text.index("$") + 1] if "$" in text else text
    return out


def games(n: int, seed: int, slack: float = 0.05, max_steps: int = 400):
    """Games played by a food-seeking player that never reverses: lists of
    (state, action, next state). The last move of a game may be a death."""
    import random

    from .engine import ACTIONS, is_reversal, start, step

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        state, moves = start(), []
        for _ in range(max_steps):
            options = [a for a in ACTIONS if not is_reversal(state, a)]
            alive = [a for a in options if step(state, a)[0] is not None]
            if alive and rng.random() >= slack:
                food = state.food or state.head
                heads = {a: step(state, a)[0].head for a in alive}
                action = min(alive, key=lambda a, h=heads, f=food: abs(h[a][0] - f[0])
                             + abs(h[a][1] - f[1]))
            else:
                action = rng.choice(alive or options)
            nxt, _ = step(state, action)
            moves.append((state, action, nxt))
            if nxt is None:
                break
            state = nxt
        out.append(moves)
    return out


def in_play(model, enc: Encoding, played, context: bool, sample: int | None = None, seed: int = 0):
    """For moves taken from real games, whether the generated frame is right.

    context=True puts the previous turn in front of the prompt, as the
    terminal game does. Returns a list of dicts with the game, the step, the
    snake's length and whether the frame was right.
    """
    import random

    rows = []
    for g, moves in enumerate(played):
        for t, (state, action, nxt) in enumerate(moves):
            before = ""
            if context and t > 0:
                p_state, p_action, _ = moves[t - 1]
                before = enc.prompt(p_state, p_action) + enc.target(state)
            rows.append({"game": g, "step": t, "length": len(state), "dead": nxt is None,
                         "prompt": before + enc.prompt(state, action),
                         "truth": enc.target(nxt)[:-1]})
    if sample and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    texts = generated(model, enc, [r["prompt"] for r in rows])
    for r, text in zip(rows, texts, strict=True):
        r["right"] = text == r["truth"]
        del r["prompt"], r["truth"]
    return rows
