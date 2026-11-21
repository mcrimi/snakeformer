"""A small decoder-only transformer.

The same network as model/gpt.py (pre-norm blocks, learned positions, ReLU
MLP, attention scaled by the embedding width rather than the head width), with
the heads computed together. `load_original` reads a checkpoint saved by the
original class, so the published weights run here unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    block_size: int = 1024
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 4


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.query = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.key = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.value = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.fc1 = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.fc2 = nn.Linear(4 * c.n_embd, c.n_embd)

    def attention(self, x, want_weights=False):
        b, t, c = x.shape
        split = lambda m: m.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        # The original scales by the embedding width, not the head width. Kept.
        scores = (q @ k.transpose(-2, -1)) * c**-0.5
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        out = (weights @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(out), (weights if want_weights else None)

    def forward(self, x, want_weights=False):
        a, weights = self.attention(self.ln1(x), want_weights)
        x = x + a
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x, weights


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, want_weights=False):
        """Logits for every position; with want_weights, also each layer's attention."""
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        attention = []
        for block in self.blocks:
            x, w = block(x, want_weights)
            attention.append(w)
        logits = self.head(self.ln_f(x))
        return (logits, attention) if want_weights else logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_original(state_dict: dict, config: Config) -> GPT:
    """Build a GPT from a checkpoint saved by model/gpt.py."""
    model = GPT(config)
    new = {
        "tok.weight": state_dict["token_embedding_table.weight"],
        "pos.weight": state_dict["position_embedding_table.weight"],
        "ln_f.weight": state_dict["ln_f.weight"], "ln_f.bias": state_dict["ln_f.bias"],
        "head.weight": state_dict["lm_head.weight"], "head.bias": state_dict["lm_head.bias"],
    }
    for i in range(config.n_layer):
        old, b = f"blocks.{i}.", f"blocks.{i}."
        for name in ("query", "key", "value"):
            rows = [state_dict[f"{old}sa.heads.{h}.{name}.weight"] for h in range(config.n_head)]
            new[f"{b}{name}.weight"] = torch.cat(rows, 0)
        for dst, src in (("proj", "sa.proj"), ("fc1", "ffwd.net.0"), ("fc2", "ffwd.net.2"),
                         ("ln1", "ln1"), ("ln2", "ln2")):
            new[f"{b}{dst}.weight"] = state_dict[f"{old}{src}.weight"]
            new[f"{b}{dst}.bias"] = state_dict[f"{old}{src}.bias"]
    model.load_state_dict(new)
    return model


def save(model: GPT, path, **extra) -> None:
    torch.save({"config": asdict(model.config), "state": model.state_dict(), **extra}, path)


def load(path) -> tuple[GPT, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(Config(**blob.pop("config")))
    model.load_state_dict(blob.pop("state"))
    return model, blob
