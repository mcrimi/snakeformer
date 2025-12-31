import os
import sys
import pickle
import torch
import curses
import time

# Add parent directory to path to find model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig


def prompt_model_selection(stdscr, model_dir):
    """
    Lists .pt files in model_dir and asks user to select one.
    Returns (model_path, meta_path) or (None, None) if cancelled.
    """
    curses.curs_set(0)
    stdscr.clear()

    if not os.path.exists(model_dir):
        stdscr.addstr(0, 0, f"Error: Model directory not found: {model_dir}")
        stdscr.refresh()
        time.sleep(2)
        return None, None

    files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not files:
        stdscr.addstr(0, 0, f"No .pt models found in {model_dir}")
        stdscr.refresh()
        time.sleep(2)
        return None, None

    files.sort()
    current_idx = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select a Model to Load:", curses.A_BOLD)

        for i, f in enumerate(files):
            if i == current_idx:
                stdscr.addstr(i + 2, 2, f"> {f}", curses.A_REVERSE)
            else:
                stdscr.addstr(i + 2, 2, f"  {f}")

        stdscr.addstr(len(files) + 4, 0, "Use ARROW KEYS to select, ENTER to confirm.")

        key = stdscr.getch()

        if key == curses.KEY_UP:
            current_idx = max(0, current_idx - 1)
        elif key == curses.KEY_DOWN:
            current_idx = min(len(files) - 1, current_idx + 1)
        elif key == 10 or key == 13:  # Enter
            selected_file = files[current_idx]
            model_path = os.path.join(model_dir, selected_file)
            # Assume meta.pkl is in the same dir
            meta_path = os.path.join(model_dir, "meta.pkl")
            return model_path, meta_path
        elif key == ord("q"):
            return None, None


def load_gpt_model(model_path, meta_path, device):
    """
    Loads the GPT model and meta data.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    # Load Meta
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Config
    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=meta["block_size"],
        n_embd=meta["n_embd"],
        n_head=meta["n_head"],
        n_layer=meta["n_layer"],
        dropout=0.0,
        device=device,
    )

    # Load Model
    model = GPT(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, meta
