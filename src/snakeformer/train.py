"""Training, two ways.

"aligned" puts one example in each sequence, always starting at position 0,
and scores only the part the model is asked to produce: the next frame.

"stream" is the original recipe: the examples are joined into one long text,
random windows are cut from it, and every character is scored, including the
input board and the next, unrelated example, neither of which can be predicted.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .encoding import PROMPT_LEN, Encoding
from .model import GPT, Config


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def tensors(examples, enc: Encoding):
    """Token ids (n, 556) and a mask of the positions that are scored.

    Death targets are shorter than board targets; they are padded and the
    padding is not scored.
    """
    width = PROMPT_LEN + len(enc.target(examples[0].state))
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    scored = torch.zeros(len(examples), width, dtype=torch.bool)
    for i, ex in enumerate(examples):
        seq = enc.encode(enc.prompt(ex.state, ex.action) + enc.target(ex.next))
        ids[i, : len(seq)] = torch.tensor(seq)
        scored[i, PROMPT_LEN : len(seq)] = True
    return ids, scored


def train(examples, enc: Encoding, steps: int, mode: str = "aligned", batch: int = 64,
          lr: float = 1e-3, seed: int = 0, config: Config | None = None, log=None,
          every: int = 1000, on_eval=None, start: GPT | None = None) -> GPT:
    """`start` continues from an existing model instead of a new one."""
    torch.manual_seed(seed)
    dev = device()
    model = (start or GPT(config or Config(vocab_size=len(enc.chars)))).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # A warm-up of exactly one step makes the schedule divide by zero.
    warm = 0.1 if steps == 20 else 0.05
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=warm)
    gen = torch.Generator().manual_seed(seed)
    if mode == "aligned":
        ids, scored = tensors(examples, enc)
    else:
        text = "".join(enc.prompt(e.state, e.action) + enc.target(e.next) for e in examples)
        stream = torch.tensor(enc.encode(text))
        block = model.config.block_size
    began = time.time()
    for it in range(1, steps + 1):
        if mode == "aligned":
            pick = torch.randint(len(ids), (batch,), generator=gen)
            x, m = ids[pick].to(dev), scored[pick].to(dev)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits[m[:, 1:]], x[:, 1:][m[:, 1:]])
        else:
            at = torch.randint(len(stream) - block - 1, (batch // 2,), generator=gen)
            x = torch.stack([stream[i : i + block + 1] for i in at]).to(dev)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % every == 0 or it == steps:
            row = {"step": it, "loss": loss.item(), "seconds": time.time() - began}
            if on_eval:
                model.eval()
                row.update(on_eval(model))
                model.train()
            if log is not None:
                log.append(row)
            print(row, flush=True)
    return model.eval()
