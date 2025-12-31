import curses
import os
import sys

import torch
import time

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.shared import prompt_model_selection, load_gpt_model

# Import game modules (we will refactor these next to expose run functions)
from game.neural_snake import NeuralSnakeGame
from training.shadow_neural_snake import ShadowNeuralSnakeGame


def run_neural_snake(stdscr, model, meta, device):
    game = NeuralSnakeGame(stdscr, model, meta, device)
    game.run()


def run_shadow_snake(stdscr, model, meta, device):
    game = ShadowNeuralSnakeGame(stdscr, model, meta, device)
    game.run()


def draw_menu(stdscr, selected_idx, options):
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()

    # Title
    title = "SNAKEFORMER - PORTFOLIO DEMO"
    stdscr.addstr(sh // 2 - 5, (sw - len(title)) // 2, title, curses.A_BOLD)

    for idx, option in enumerate(options):
        y = sh // 2 - 1 + idx
        x = (sw - len(option)) // 2

        if idx == selected_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, x, option)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(y, x, option)

    stdscr.refresh()


def main(stdscr):
    # Setup Colors
    curses.start_color()
    curses.use_default_colors()
    try:
        if curses.can_change_color():
            curses.init_color(curses.COLOR_BLACK, 0, 0, 0)
    except:
        pass

    # Define pairs (compatible with game/snake.py)
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # Snake
    curses.init_pair(2, curses.COLOR_RED, -1)  # Food / Target
    curses.init_pair(3, curses.COLOR_WHITE, -1)  # Text
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)  # UI
    curses.init_pair(5, curses.COLOR_CYAN, -1)  # Shadow

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model", "weigths")

    # 1. Select Model First (Shared step)
    # We do this before menu because both modes likely need a model.
    # OR we can do it inside the loop if user wants to switch models?
    # For simplicity, let's load model first as it is required for both.

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_path, meta_path = prompt_model_selection(stdscr, model_dir)
    if not model_path:
        return

    stdscr.clear()
    stdscr.addstr(10, 10, f"Loading Neural Model on {device}...")
    stdscr.refresh()

    try:
        model, meta = load_gpt_model(model_path, meta_path, device)
    except Exception as e:
        stdscr.addstr(12, 10, f"Error loading model: {e}")
        stdscr.getch()
        return

    # 2. Main Menu Loop
    options = ["Play Neural Snake", "Play Shadow Snake (Train)", "Quit"]
    selected_idx = 0

    while True:
        draw_menu(stdscr, selected_idx, options)

        try:
            key = stdscr.getch()
        except:
            key = -1

        if key == curses.KEY_UP:
            selected_idx = (selected_idx - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected_idx = (selected_idx + 1) % len(options)
        elif key == ord("\n") or key == curses.KEY_ENTER or key == 10 or key == 13:
            # Execute
            if selected_idx == 0:
                run_neural_snake(stdscr, model, meta, device)
            elif selected_idx == 1:
                run_shadow_snake(stdscr, model, meta, device)
            elif selected_idx == 2:
                break
        elif key == 27 or key == ord("q"):  # ESC
            break


if __name__ == "__main__":
    curses.wrapper(main)
