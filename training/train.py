import os
import argparse
import pickle
import time
import math
import torch
import curses
import sys
from torch.nn import functional as F

try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# CLI & Training Logic
# -----------------------------------------------------------------------------


def check_overwrite(path, force=False):
    """Check if file exists and prevent overwrite unless forced."""
    if os.path.exists(path) and not force:
        print(f"Error: File '{path}' already exists.")
        print("Use --force or --overwrite to overwrite it.")
        sys.exit(1)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_batch(data, split, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, batch_size, block_size, device, eval_iters=200
):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def pretrain(args):
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

    if args.wandb:
        if wandb is None:
            print(
                "Error: wandb not installed. Please install it or run without --wandb"
            )
            sys.exit(1)

        # Check for API key in args or environment
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

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # Save metadata immediately
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

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.max_iters} iterations...")
    start_time = time.time()

    try:
        for iter in range(args.max_iters):
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    args.batch_size,
                    args.block_size,
                    device,
                )
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if args.wandb:
                    wandb.log(
                        {
                            "iter": iter,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                        }
                    )

            xb, yb = get_batch(
                train_data, "train", args.batch_size, args.block_size, device
            )
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"step_loss": loss.item()})

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current state...")

    print(f"Training finished (or stopped) in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    if args.wandb:
        wandb.finish()


def finetune(args):
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data_file)
    model_dir = os.path.join(base_dir, "model", "weights")

    # For fine-tuning, we load from an existing model and save to a NEW one
    old_model_path = os.path.join(model_dir, args.base_model)
    meta_path = os.path.join(model_dir, args.meta_filename)
    new_model_path = os.path.join(model_dir, args.new_model_name)

    if not os.path.exists(old_model_path):
        raise FileNotFoundError(f"Base model not found: {old_model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    check_overwrite(new_model_path, args.force)

    if args.wandb:
        if wandb is None:
            print(
                "Error: wandb not installed. Please install it or run without --wandb"
            )
            sys.exit(1)

        if args.wandb_key:
            os.environ["WANDB_API_KEY"] = args.wandb_key

        wandb.init(
            project=args.wandb_project or "snakeformer",
            name=args.run_name or "finetune",
            config=args,
        )

    # Load Metadata
    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    vocab_size = meta["vocab_size"]
    n_embd = meta.get("n_embd", 128)
    n_head = meta.get("n_head", 8)
    n_layer = meta.get("n_layer", 4)
    block_size = meta.get("block_size", 1024)

    print(f"Model Config from meta: {n_layer}L, {n_head}H, {n_embd}E")

    # Load New Data
    print(f"Loading finetuning data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    except KeyError as e:
        print(f"ERROR: New data contains unknown character: {e}")
        sys.exit(1)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize Model
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

    # Load Weights
    print(f"Loading weights from {old_model_path}...")
    model.load_state_dict(torch.load(old_model_path, map_location=device))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting fine-tuning for {args.max_iters} iterations...")
    start_time = time.time()

    try:
        for iter in range(args.max_iters):
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    args.batch_size,
                    block_size,
                    device,
                    eval_iters=50,
                )
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if args.wandb:
                    wandb.log(
                        {
                            "iter": iter,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                        }
                    )

            xb, yb = get_batch(train_data, "train", args.batch_size, block_size, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"step_loss": loss.item()})

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current state...")

    print(
        f"Fine-tuning finished (or stopped) in {time.time() - start_time:.2f} seconds"
    )
    torch.save(model.state_dict(), new_model_path)
    print(f"Saved fine-tuned model to {new_model_path}")
    if args.wandb:
        wandb.finish()


# -----------------------------------------------------------------------------
# UI / Interactive Mode
# -----------------------------------------------------------------------------


def get_input_str(stdscr, prompt, default=""):
    """Text input helper"""
    curses.echo()
    curses.curs_set(1)

    sh, sw = stdscr.getmaxyx()
    win_w = 60
    win_h = 3
    win_y = sh // 2
    win_x = max(0, (sw - win_w) // 2)

    # Draw prompt
    stdscr.addstr(win_y - 2, win_x, prompt, curses.color_pair(3) | curses.A_BOLD)
    if default:
        stdscr.addstr(
            win_y - 1, win_x, f"Default: {default}", curses.color_pair(3) | curses.A_DIM
        )

    win = curses.newwin(win_h, win_w, win_y, win_x)
    win.box()
    # win.bkgd(' ', curses.color_pair(4))
    stdscr.refresh()
    win.refresh()

    # Move cursor inside box
    win.move(1, 2)

    inp_bytes = win.getstr()
    inp = inp_bytes.decode("utf-8").strip()

    curses.noecho()
    curses.curs_set(0)

    return inp if inp else default


def draw_menu(stdscr, selected_idx, options):
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()

    # ASCII Art Title
    ascii_art = [
        r"  _______  ______  _______ _______ __   _      ",
        r"     |    |_____/  |_____|    |    | \  |      ",
        r"     |    |    \_  |     |  __|__  |  \_|      ",
    ]

    start_y = max(1, sh // 2 - 10)
    for i, line in enumerate(ascii_art):
        x_pos = max(0, (sw - len(line)) // 2)
        if start_y + i < sh:
            stdscr.addstr(
                start_y + i, x_pos, line, curses.color_pair(1) | curses.A_BOLD
            )

    subtitle = "SnakeFormer Training Interface"
    sub_y = start_y + len(ascii_art) + 1
    stdscr.addstr(
        sub_y, max(0, (sw - len(subtitle)) // 2), subtitle, curses.color_pair(3)
    )

    menu_start_y = sub_y + 3
    for idx, option in enumerate(options):
        y = menu_start_y + idx
        x = max(0, (sw - len(option)) // 2)

        if y >= sh - 1:
            break

        if idx == selected_idx:
            stdscr.attron(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
            stdscr.addstr(y, x - 2, f"  {option}  ")
            stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
        else:
            stdscr.addstr(y, x, option, curses.color_pair(3))

    stdscr.refresh()


def pretrain_form(stdscr):
    # Data File
    data_file = get_input_str(
        stdscr, "Dataset Path (relative):", "dataset/snake_data_curriculum.txt"
    )

    # Model Name
    model_name = get_input_str(stdscr, "Output Model Name:", "snake_model.pt")

    # Iters
    iters_str = get_input_str(stdscr, "Training Iterations:", "20000")
    try:
        iters = int(iters_str)
    except ValueError:
        iters = 20000

    return data_file, model_name, iters


def finetune_form(stdscr):
    # Base Model
    base_model = get_input_str(stdscr, "Base Model Name:", "snake_model.pt")

    # Data File
    data_file = get_input_str(
        stdscr, "Finetuning Dataset:", "dataset/snake_data_curriculum.txt"
    )

    # New Model Name
    new_model = get_input_str(stdscr, "Output Model Name:", "snake_model_finetuned.pt")

    # Iters
    iters_str = get_input_str(stdscr, "Training Iterations:", "2000")
    try:
        iters = int(iters_str)
    except ValueError:
        iters = 2000

    return base_model, data_file, new_model, iters


def wandb_form(stdscr):
    # WandB
    use_wandb = get_input_str(stdscr, "Enable Weights & Biases? (y/n):", "n")
    if use_wandb.lower().startswith("y"):
        key = get_input_str(stdscr, "API Key (leave blank if Env Var set):", "")
        project = get_input_str(stdscr, "Project Name:", "snakeformer")
        run_name = get_input_str(stdscr, "Run Name:", f"run_{int(time.time())}")
        return True, key, project, run_name
    return False, None, None, None


def run_interactive(stdscr):
    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_WHITE, -1)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)

    options = ["Pre-train New Model", "Fine-tune Existing Model", "Quit"]
    selected_idx = 0

    while True:
        draw_menu(stdscr, selected_idx, options)
        key = stdscr.getch()

        if key == curses.KEY_UP:
            selected_idx = (selected_idx - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected_idx = (selected_idx + 1) % len(options)
        elif key in [10, 13]:  # Enter
            if selected_idx == 0:  # Pretrain
                data, model, iters = pretrain_form(stdscr)
                use_wb, w_key, w_proj, w_run = wandb_form(stdscr)

                # Build Args
                args = argparse.Namespace()
                args.data_file = data
                args.model_name = model
                args.max_iters = iters
                args.meta_filename = "meta.pkl"  # Default for now
                args.batch_size = 64
                args.block_size = 1024
                args.n_embd = 128
                args.n_head = 8
                args.n_layer = 4
                args.lr = 1e-3
                args.dropout = 0.0
                # args.force = True # Assume user knows what they are doing in interactive? Or ask?
                # Let's be safe and check overwrite, but we are inside curses...
                # check_overwrite prints and exists. That breaks curses.
                # Ideally we check here.
                args.force = False

                # WandB
                args.wandb = use_wb
                args.wandb_key = w_key
                args.wandb_project = w_proj
                args.run_name = w_run
                args.eval_interval = 1000

                # Exit curses to run training script (simple way to handle stdout/logging)
                return "pretrain", args

            elif selected_idx == 1:  # Finetune
                base, data, new_model_name, iters = finetune_form(stdscr)
                use_wb, w_key, w_proj, w_run = wandb_form(stdscr)

                args = argparse.Namespace()
                args.base_model = base
                args.data_file = data
                args.new_model_name = new_model_name
                args.max_iters = iters
                args.meta_filename = "meta.pkl"
                args.batch_size = 64
                args.lr = 1e-4  # Lower for finetune default
                args.dropout = 0.0
                args.force = False

                # WandB
                args.wandb = use_wb
                args.wandb_key = w_key
                args.wandb_project = w_proj
                args.run_name = w_run
                args.eval_interval = 500

                return "finetune", args

            elif selected_idx == 2:  # Quit
                return None, None

    return None, None


def main():
    # If arguments provided, run CLI mode
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="SnakeFormer Training Script")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Common Arguments
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument(
            "--data_file",
            type=str,
            default="dataset/snake_data_curriculum.txt",
            help="Path to dataset",
        )
        common.add_argument("--batch_size", type=int, default=64)
        common.add_argument("--max_iters", type=int, default=20000)
        common.add_argument("--eval_interval", type=int, default=1000)
        common.add_argument("--lr", type=float, default=1e-3)
        common.add_argument("--dropout", type=float, default=0.0)
        common.add_argument("--wandb", action="store_true", help="Enable WandB logging")
        common.add_argument("--wandb_key", type=str, help="WandB API Key")
        common.add_argument("--wandb_project", type=str, help="WandB Project Name")
        common.add_argument("--run_name", type=str, help="WandB run name")
        common.add_argument(
            "--meta_filename",
            type=str,
            default="meta.pkl",
            help="Name of metadata file",
        )
        common.add_argument(
            "--force", action="store_true", help="Force overwrite of existing files"
        )

        # Pretrain Command
        parser_pre = subparsers.add_parser("pretrain", parents=[common])
        parser_pre.add_argument(
            "--model_name",
            type=str,
            default="snake_model.pt",
            help="Name of output model file",
        )
        parser_pre.add_argument("--block_size", type=int, default=1024)
        parser_pre.add_argument("--n_embd", type=int, default=128)
        parser_pre.add_argument("--n_head", type=int, default=8)
        parser_pre.add_argument("--n_layer", type=int, default=4)
        parser_pre.set_defaults(func=pretrain)

        # Finetune Command
        parser_fine = subparsers.add_parser("finetune", parents=[common])
        parser_fine.add_argument(
            "--base_model",
            type=str,
            default="snake_model.pt",
            help="Name of base model to load",
        )
        parser_fine.add_argument(
            "--new_model_name",
            type=str,
            default="snake_model_finetuned.pt",
            help="Name of output model file",
        )
        parser_fine.set_defaults(func=finetune)

        args = parser.parse_args()
        args.func(args)

    else:
        # Interactive Mode
        try:
            mode, args = curses.wrapper(run_interactive)
            if mode == "pretrain":
                print("\n--- Starting Pre-training ---")
                print(f"Data: {args.data_file}")
                print(f"Model: {args.model_name}")
                print(f"WandB: {args.wandb}")
                time.sleep(1)
                pretrain(args)
            elif mode == "finetune":
                print("\n--- Starting Fine-tuning ---")
                print(f"Base: {args.base_model}")
                print(f"New: {args.new_model_name}")
                print(f"WandB: {args.wandb}")
                time.sleep(1)
                finetune(args)
            else:
                print("Exiting.")
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
