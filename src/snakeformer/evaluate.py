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
