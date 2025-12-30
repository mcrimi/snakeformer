import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.0
    device: str = "cpu"


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, stop_token_id=None):
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size :]

            # get the predictions
            logits, _ = self(idx_cond)

            # get probabilities
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append to the sequence and keep going
            idx = torch.cat((idx, idx_next), dim=1)

            # stopping rule to avoid unnecesary inference
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                # We hit '$', so we stop inference
                return idx
            # -------------------------------

        return idx

    def train_step(self, optimizer, idx, target_idx, importance_weight=1.0):
        """
        Single training step for RL correction.
        idx: (B, T) tensor of context inputs
        target_idx: (B, 1) tensor (or scalar tensor) of the target token to predict
        importance_weight: float multiplier for the loss
        """
        self.train()
        optimizer.zero_grad()

        # 1. Forward Pass
        # We only care about the last token prediction for the loss
        # The input 'idx' should be the full context up to the target

        logits, _ = self(idx)

        # Get the logits for the VERY LAST token (the one we are trying to predict)
        # logits shape: (B, T, V) -> we want (B, -1, V)
        last_token_logits = logits[:, -1, :]  # Shape: (B, VocabSize)

        # 2. Loss Calculation
        # target_idx should be (B) or (1)
        if target_idx.dim() == 2:
            target_idx = target_idx.squeeze(-1)

        loss = F.cross_entropy(last_token_logits, target_idx, reduction="none")

        # Apply importance weight
        weighted_loss = loss * importance_weight
        final_loss = weighted_loss.mean()

        # 3. Update
        final_loss.backward()

        # Clip gradients to prevent explosion during online updates
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        optimizer.step()

        # Return probs for visualization
        with torch.no_grad():
            probs = F.softmax(last_token_logits, dim=-1)

        return final_loss.item(), probs
