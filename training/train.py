#!/usr/bin/env python3
"""
SnakeFormer Training Script

Usage:
    # Pre-train a new model
    python training/train.py pretrain --data_file dataset/snake_data_curriculum.txt --model_name snake_model.pt

    # Fine-tune an existing model
    python training/train.py finetune --base_model snake_model.pt --new_model_name snake_model_v2.pt

    # See all options
    python training/train.py pretrain --help
    python training/train.py finetune --help
"""

import os
import argparse
import pickle
import time
import sys
import torch

try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig


def get_device():
    """Pick cuda > mps > cpu, whatever's available."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_overwrite(path, force=False):
    """Exit if file exists (unless force=True)."""
    if os.path.exists(path) and not force:
        print(f"Error: File '{path}' already exists.")
        print("Use --force to overwrite it.")
        sys.exit(1)


def get_batch(data, batch_size, block_size, device):
    """Sample random (x, y) chunks from the dataset."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=200):
    """Average loss over eval_iters random batches. No gradients."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def pretrain(args):
    """Train a fresh model from random init."""
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data_file)
    model_dir = os.path.join(base_dir, "model", "weights")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, args.model_name)
    meta_path = os.path.join(model_dir, args.meta_filename)

    check_overwrite(model_path, args.force)

    # WandB setup
    if args.wandb:
        if wandb is None:
            print("Error: wandb not installed. Run: pip install wandb")
            sys.exit(1)
        if args.wandb_key:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        wandb.init(
            project=args.wandb_project or "snakeformer",
            name=args.run_name or "pretrain",
        )

    # Load Data
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset size: {len(text):,} characters")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "chars": chars,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
    }
    check_overwrite(meta_path, args.force)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_path}")

    # Prepare data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    # Create model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        device=device,
    )
    model = GPT(config)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.max_iters} iterations...")
    print(f"Batch size: {args.batch_size}, Block size: {args.block_size}, LR: {args.lr}")
    print("-" * 50)
    
    start_time = time.time()

    try:
        for iter in range(args.max_iters):
            # Evaluate periodically
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(
                    model, train_data, val_data,
                    args.batch_size, args.block_size, device
                )
                elapsed = time.time() - start_time
                print(f"step {iter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
                
                if args.wandb:
                    wandb.log({
                        "iter": iter,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                    })

            # Training step
            xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"step_loss": loss.item()})

    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving current state...")

    # Save model
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    if args.wandb:
        wandb.finish()


def finetune(args):
    """Continue training from an existing checkpoint."""
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data_file)
    model_dir = os.path.join(base_dir, "model", "weights")

    old_model_path = os.path.join(model_dir, args.base_model)
    meta_path = os.path.join(model_dir, args.meta_filename)
    new_model_path = os.path.join(model_dir, args.new_model_name)

    if not os.path.exists(old_model_path):
        raise FileNotFoundError(f"Base model not found: {old_model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    check_overwrite(new_model_path, args.force)

    # WandB setup
    if args.wandb:
        if wandb is None:
            print("Error: wandb not installed. Run: pip install wandb")
            sys.exit(1)
        if args.wandb_key:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        wandb.init(
            project=args.wandb_project or "snakeformer",
            name=args.run_name or "finetune",
            config=vars(args),
        )

    # Load metadata
    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    vocab_size = meta["vocab_size"]
    n_embd = meta.get("n_embd", 128)
    n_head = meta.get("n_head", 8)
    n_layer = meta.get("n_layer", 4)
    block_size = meta.get("block_size", 1024)

    print(f"Model config: {n_layer}L, {n_head}H, {n_embd}E, block_size={block_size}")

    # Load fine-tuning data
    print(f"Loading fine-tuning data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset size: {len(text):,} characters")

    try:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    except KeyError as e:
        print(f"Error: Data contains unknown character: {e}")
        sys.exit(1)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    # Create and load model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=args.dropout,
        device=device,
    )
    model = GPT(config)

    print(f"Loading weights from {old_model_path}...")
    model.load_state_dict(torch.load(old_model_path, map_location=device))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting fine-tuning for {args.max_iters} iterations...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 50)
    
    start_time = time.time()

    try:
        for iter in range(args.max_iters):
            # Evaluate periodically
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(
                    model, train_data, val_data,
                    args.batch_size, block_size, device,
                    eval_iters=50
                )
                elapsed = time.time() - start_time
                print(f"step {iter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
                
                if args.wandb:
                    wandb.log({
                        "iter": iter,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                    })

            # Training step
            xb, yb = get_batch(train_data, args.batch_size, block_size, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"step_loss": loss.item()})

    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving current state...")

    # Save model
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Fine-tuning finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    torch.save(model.state_dict(), new_model_path)
    print(f"Saved fine-tuned model to {new_model_path}")
    
    if args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="SnakeFormer Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-train a new model
  python training/train.py pretrain --model_name my_model.pt --max_iters 20000

  # Fine-tune an existing model  
  python training/train.py finetune --base_model snake_model.pt --new_model_name snake_v2.pt

  # Pre-train with custom hyperparameters
  python training/train.py pretrain --lr 0.0005 --batch_size 32 --n_layer 6

  # Train with WandB logging
  python training/train.py pretrain --wandb --wandb_project my_project --run_name experiment1
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments for both commands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data_file", type=str,
        default="dataset/snake_data_curriculum.txt",
        help="Path to dataset file (default: dataset/snake_data_curriculum.txt)"
    )
    common.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    common.add_argument("--max_iters", type=int, default=20000, help="Training iterations (default: 20000)")
    common.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval (default: 1000)")
    common.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    common.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)")
    common.add_argument("--meta_filename", type=str, default="meta.pkl", help="Metadata filename (default: meta.pkl)")
    common.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    # WandB arguments
    common.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    common.add_argument("--wandb_key", type=str, help="WandB API key (or set WANDB_API_KEY env var)")
    common.add_argument("--wandb_project", type=str, default="snakeformer", help="WandB project name")
    common.add_argument("--run_name", type=str, help="WandB run name")

    # Pretrain command
    parser_pre = subparsers.add_parser(
        "pretrain", parents=[common],
        help="Pre-train a new model from scratch"
    )
    parser_pre.add_argument(
        "--model_name", type=str, default="snake_model.pt",
        help="Output model filename (default: snake_model.pt)"
    )
    parser_pre.add_argument("--block_size", type=int, default=1024, help="Context length (default: 1024)")
    parser_pre.add_argument("--n_embd", type=int, default=128, help="Embedding dimension (default: 128)")
    parser_pre.add_argument("--n_head", type=int, default=8, help="Number of attention heads (default: 8)")
    parser_pre.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser_pre.set_defaults(func=pretrain)

    # Finetune command
    parser_fine = subparsers.add_parser(
        "finetune", parents=[common],
        help="Fine-tune an existing model"
    )
    parser_fine.add_argument(
        "--base_model", type=str, default="snake_model.pt",
        help="Base model to load (default: snake_model.pt)"
    )
    parser_fine.add_argument(
        "--new_model_name", type=str, default="snake_model_finetuned.pt",
        help="Output model filename (default: snake_model_finetuned.pt)"
    )
    parser_fine.set_defaults(func=finetune)

    # Parse and run
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
