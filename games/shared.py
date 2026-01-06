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

    # Setup colors if not already done (safe to re-init)
    curses.start_color()
    curses.use_default_colors()
    # Ensure pairs we need exist. Re-declaring for safety if called standalone.
    # 1: Green, 3: White, 4: Highlight, 5: UI/Cyan
    try:
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_WHITE, -1)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
    except Exception:
        pass

    if not os.path.exists(model_dir):
        msg = f"Error: Model directory not found: {model_dir}"
        stdscr.addstr(
            stdscr.getmaxyx()[0] // 2,
            max(0, (stdscr.getmaxyx()[1] - len(msg)) // 2),
            msg,
            curses.color_pair(2),
        )
        stdscr.refresh()
        time.sleep(2)
        return None, None

    files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not files:
        msg = f"No .pt models found in {model_dir}"
        stdscr.addstr(
            stdscr.getmaxyx()[0] // 2,
            max(0, (stdscr.getmaxyx()[1] - len(msg)) // 2),
            msg,
            curses.color_pair(3),
        )
        stdscr.refresh()
        time.sleep(2)
        return None, None

    files.sort()
    current_idx = 0

    while True:
        stdscr.clear()
        sh, sw = stdscr.getmaxyx()

        # Header
        ascii_header = [
            r"   __  __           __     __ ",
            r"  /  |/  /___  ____/ /__  / / ",
            r" / /|_/ / __ \/ __  / _ \/ /  ",
            r"/ /  / / /_/ / /_/ /  __/ /   ",
            r"/_/  /_/\____/\__,_/\___/_/    ",
        ]

        start_y = max(1, sh // 2 - 10)
        for i, line in enumerate(ascii_header):
            x_pos = max(0, (sw - len(line)) // 2)
            if start_y + i < sh:
                stdscr.addstr(
                    start_y + i, x_pos, line, curses.color_pair(1) | curses.A_BOLD
                )

        subtitle = "Select a checkpoint"
        subtitle_y = start_y + len(ascii_header) + 1
        if subtitle_y < sh:
            stdscr.addstr(
                subtitle_y,
                max(0, (sw - len(subtitle)) // 2),
                subtitle,
                curses.color_pair(3) | curses.A_DIM,
            )

        # List
        list_start_y = subtitle_y + 3

        # Pagination / Viewport logic could be added but sticking to simple centered list for now
        # Assuming list isn't massive for layout simplicity
        for i, f in enumerate(files):
            y = list_start_y + i
            if y >= sh - 2:
                break

            x = max(0, (sw - len(f)) // 2)

            if i == current_idx:
                stdscr.attron(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
                stdscr.addstr(y, x - 2, f"  {f}  ")
                stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
            else:
                stdscr.addstr(y, x, f"{f}", curses.color_pair(3))

        # Footer
        footer = "Use \u2191\u2193 to Navigate, ENTER to Confirm. 'Q' to Cancel."
        if sh > list_start_y + len(files) + 2:
            stdscr.addstr(
                sh - 2,
                max(0, (sw - len(footer)) // 2),
                footer,
                curses.color_pair(5),
            )

        stdscr.refresh()

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


def load_model(model_path, meta_path, device):
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
