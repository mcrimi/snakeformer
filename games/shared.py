import os
import sys
import pickle
import torch
import curses
import time
import threading

# Add parent directory to path to find model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig

# HuggingFace repo info
HF_REPO_ID = "mcrimi/snakeformer"
HF_MODEL_FILENAME = "snake_model.pt"
HF_META_FILENAME = "meta.pkl"


def download_model_from_hf(stdscr, model_dir):
    """Pull model from HuggingFace. Won't clobber existing files."""
    curses.curs_set(0)
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()
    curses.start_color()
    curses.use_default_colors()
    try:
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_WHITE, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
    except Exception:
        pass

    def draw_header():
        ascii_header = [
            r"    ____                      __                __",
            r"   / __ \____ _      ______  / /___  ____ _____/ /",
            r"  / / / / __ \ | /| / / __ \/ / __ \/ __ `/ __  / ",
            r" / /_/ / /_/ / |/ |/ / / / / / /_/ / /_/ / /_/ /  ",
            r"/_____/\____/|__/|__/_/ /_/_/\____/\__,_/\__,_/   ",
        ]
        start_y = max(1, sh // 2 - 10)
        for i, line in enumerate(ascii_header):
            x_pos = max(0, (sw - len(line)) // 2)
            if start_y + i < sh:
                stdscr.addstr(
                    start_y + i, x_pos, line, curses.color_pair(1) | curses.A_BOLD
                )
        return start_y + len(ascii_header) + 2

    def show_message(msg, color_pair=3, y_offset=0):
        stdscr.clear()
        msg_y = draw_header() + y_offset
        if msg_y < sh - 1:
            stdscr.addstr(
                msg_y,
                max(0, (sw - len(msg)) // 2),
                msg,
                curses.color_pair(color_pair),
            )
        stdscr.refresh()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        show_message("Error: huggingface_hub not installed. Run: pip install huggingface_hub", 2)
        stdscr.addstr(
            draw_header() + 2,
            max(0, (sw - 25) // 2),
            "Press any key to continue",
            curses.color_pair(5),
        )
        stdscr.refresh()
        stdscr.getch()
        return None, None

    os.makedirs(model_dir, exist_ok=True)

    def get_unique_path(directory, base_name):
        """Find a filename that doesn't exist yet (_hf, _hf_1, etc)."""
        base, ext = os.path.splitext(base_name)
        target = os.path.join(directory, base_name)
        
        if not os.path.exists(target):
            return target, base_name

        hf_name = f"{base}_hf{ext}"
        target = os.path.join(directory, hf_name)
        if not os.path.exists(target):
            return target, hf_name

        counter = 1
        while True:
            numbered_name = f"{base}_hf_{counter}{ext}"
            target = os.path.join(directory, numbered_name)
            if not os.path.exists(target):
                return target, numbered_name
            counter += 1
            if counter > 100:
                return None, None

    model_target, model_name = get_unique_path(model_dir, HF_MODEL_FILENAME)
    meta_target, meta_name = get_unique_path(model_dir, HF_META_FILENAME)

    if model_target is None or meta_target is None:
        show_message("Error: Could not find unique filename for download", 2)
        stdscr.getch()
        return None, None

    stdscr.clear()
    msg_y = draw_header()

    info_lines = [
        f"Repository: {HF_REPO_ID}",
        f"Model will be saved as: {model_name}",
        f"Meta will be saved as: {meta_name}",
        "",
        "Press ENTER to download, Q to cancel",
    ]

    for i, line in enumerate(info_lines):
        y = msg_y + i
        if y < sh - 1:
            stdscr.addstr(y, max(0, (sw - len(line)) // 2), line, curses.color_pair(3))

    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key == 10 or key == 13:
            break
        elif key == ord("q") or key == ord("Q") or key == 27:
            return None, None

    download_status = {"done": False, "error": None, "current_file": ""}
    downloaded_paths = {"model": None, "meta": None}

    def download_thread():
        try:
            download_status["current_file"] = "model"
            downloaded_paths["model"] = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME,
            )
            download_status["current_file"] = "meta"
            downloaded_paths["meta"] = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_META_FILENAME,
            )
            download_status["done"] = True
        except Exception as e:
            download_status["error"] = str(e)
            download_status["done"] = True

    thread = threading.Thread(target=download_thread)
    thread.start()

    spinner = ["|", "/", "-", "\\"]
    spin_idx = 0

    while not download_status["done"]:
        stdscr.clear()
        msg_y = draw_header()

        current = download_status["current_file"]
        status_msg = f"Downloading {current}... {spinner[spin_idx]}"
        stdscr.addstr(
            msg_y,
            max(0, (sw - len(status_msg)) // 2),
            status_msg,
            curses.color_pair(5),
        )
        stdscr.refresh()

        spin_idx = (spin_idx + 1) % len(spinner)
        time.sleep(0.2)

    thread.join()

    if download_status["error"]:
        show_message(f"Download failed: {download_status['error'][:50]}", 2)
        stdscr.addstr(
            draw_header() + 2,
            max(0, (sw - 25) // 2),
            "Press any key to continue",
            curses.color_pair(5),
        )
        stdscr.refresh()
        stdscr.getch()
        return None, None

    import shutil
    try:
        show_message("Copying files to model directory...", 5)
        shutil.copy2(downloaded_paths["model"], model_target)
        shutil.copy2(downloaded_paths["meta"], meta_target)
    except Exception as e:
        show_message(f"Error copying files: {str(e)[:40]}", 2)
        stdscr.getch()
        return None, None

    stdscr.clear()
    msg_y = draw_header()

    success_lines = [
        "Download complete!",
        "",
        f"Model saved to: {model_name}",
        f"Meta saved to: {meta_name}",
        "",
        "Press any key to continue",
    ]

    for i, line in enumerate(success_lines):
        y = msg_y + i
        color = curses.color_pair(1) if i == 0 else curses.color_pair(3)
        if y < sh - 1:
            stdscr.addstr(y, max(0, (sw - len(line)) // 2), line, color)

    stdscr.refresh()
    stdscr.getch()

    return model_target, meta_target


def prompt_model_selection(stdscr, model_dir):
    """Show checkpoint picker. Also offers HuggingFace download."""
    curses.curs_set(0)
    stdscr.clear()
    curses.start_color()
    curses.use_default_colors()
    try:
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_WHITE, -1)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
    except Exception:
        pass

    os.makedirs(model_dir, exist_ok=True)
    DOWNLOAD_OPTION = "[Download from HuggingFace]"

    while True:
        files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        files.sort()
        options = files + [DOWNLOAD_OPTION]

        current_idx = 0

        while True:
            stdscr.clear()
            sh, sw = stdscr.getmaxyx()

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

            subtitle = "Select a checkpoint or download from HuggingFace"
            subtitle_y = start_y + len(ascii_header) + 1
            if subtitle_y < sh:
                stdscr.addstr(
                    subtitle_y,
                    max(0, (sw - len(subtitle)) // 2),
                    subtitle,
                    curses.color_pair(3) | curses.A_DIM,
                )

            list_start_y = subtitle_y + 3
            for i, option in enumerate(options):
                y = list_start_y + i
                if y >= sh - 2:
                    break

                x = max(0, (sw - len(option)) // 2)
                is_download = (option == DOWNLOAD_OPTION)

                if i == current_idx:
                    stdscr.attron(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
                    stdscr.addstr(y, x - 2, f"  {option}  ")
                    stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
                else:
                    color = curses.color_pair(5) if is_download else curses.color_pair(3)
                    stdscr.addstr(y, x, option, color)

            footer = "Use ↑↓ to Navigate, ENTER to Confirm. 'Q' to Cancel."
            if sh > list_start_y + len(options) + 2:
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
                current_idx = min(len(options) - 1, current_idx + 1)
            elif key == 10 or key == 13:
                selected = options[current_idx]

                if selected == DOWNLOAD_OPTION:
                    model_path, meta_path = download_model_from_hf(stdscr, model_dir)
                    if model_path:
                        return model_path, meta_path
                    break  # refresh list after download attempt
                else:
                    model_path = os.path.join(model_dir, selected)
                    meta_path = os.path.join(model_dir, "meta.pkl")
                    return model_path, meta_path
            elif key == ord("q") or key == ord("Q") or key == 27:
                return None, None


def load_model(model_path, meta_path, device):
    """Load GPT weights + vocab metadata. Returns (model, meta)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=meta["block_size"],
        n_embd=meta["n_embd"],
        n_head=meta["n_head"],
        n_layer=meta["n_layer"],
        dropout=0.0,
        device=device,
    )

    model = GPT(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, meta
