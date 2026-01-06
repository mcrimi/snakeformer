import curses
import os
import sys

import torch


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.shared import prompt_model_selection, load_model
from games.snake import SnakeGame
from games.neural_snake import NeuralSnakeGame
from games.shadow_neural_snake import ShadowNeuralSnakeGame


def run_classic_snake(stdscr):
    game = SnakeGame(stdscr)
    game.run()


def run_neural_snake(stdscr, model, meta, device):
    game = NeuralSnakeGame(stdscr, model, meta, device)
    game.run()


def run_shadow_snake(stdscr, model, meta, device):
    game = ShadowNeuralSnakeGame(stdscr, model, meta, device)
    game.run()


def draw_menu(stdscr, selected_idx, options):
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()

    # ASCII Art Title
    ascii_art = [
        r"   _____             __         ______                          ",
        r"  / ___/____  ____ _/ /_____   / ____/___  _________ ___  ___  _____",
        r"  \__ \/ __ \/ __ `/ //_/ _ \ / /_  / __ \/ ___/ __ `__ \/ _ \/ ___/",
        r" ___/ / / / / /_/ / ,< /  __// __/ / /_/ / /  / / / / / /  __/ /    ",
        r"/____/_/ /_/\__,_/_/|_|\___//_/    \____/_/  /_/ /_/ /_/\___/_/     ",
    ]

    # Subtitle
    subtitle = "The overengineered snake game that nobody asked for"

    # Draw ASCII Art centered
    start_y = max(1, sh // 2 - 10)
    for i, line in enumerate(ascii_art):
        x_pos = max(0, (sw - len(line)) // 2)
        if start_y + i < sh:
            stdscr.addstr(
                start_y + i, x_pos, line, curses.color_pair(1) | curses.A_BOLD
            )

    # Draw Subtitle
    subtitle_y = start_y + len(ascii_art) + 1
    if subtitle_y < sh:
        stdscr.addstr(
            subtitle_y,
            max(0, (sw - len(subtitle)) // 2),
            subtitle,
            curses.color_pair(3) | curses.A_DIM,
        )

    # Draw Menu Options
    menu_start_y = subtitle_y + 4
    for idx, option in enumerate(options):
        y = menu_start_y + idx
        x = max(0, (sw - len(option)) // 2)

        if y >= sh - 1:
            break

        if idx == selected_idx:
            # Highlighted
            stdscr.attron(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
            stdscr.addstr(y, x - 2, f"  {option}  ")
            stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD)
        else:
            # Normal
            stdscr.addstr(y, x, option, curses.color_pair(3))

    # Footer / Instructions
    footer = "Use \u2191\u2193 to Navigate, ENTER to Select. 'Q' to Quit."
    if sh > menu_start_y + len(options) + 2:
        stdscr.addstr(
            sh - 2,
            max(0, (sw - len(footer)) // 2),
            footer,
            curses.color_pair(5),
        )

    stdscr.refresh()


def main(stdscr):
    # Setup Colors
    curses.start_color()
    curses.use_default_colors()
    try:
        if curses.can_change_color():
            curses.init_color(curses.COLOR_BLACK, 0, 0, 0)
    except Exception:
        pass

    # Define pairs (compatible with game/snake.py)
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # Snake
    curses.init_pair(2, curses.COLOR_RED, -1)  # Food / Target
    curses.init_pair(3, curses.COLOR_WHITE, -1)  # Text
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)  # UI
    curses.init_pair(5, curses.COLOR_CYAN, -1)  # Shadow

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model", "weigths")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = None
    meta = None

    # 2. Main Menu Loop
    options = [
        "Play Classic Snake - boring",
        "Play SnakeFormer - an LLM based neural engine",
        "Play Snakeformer Shadow - an LLM based neural engine with physics validation and online correction",
        "Quit",
    ]
    selected_idx = 0

    while True:
        draw_menu(stdscr, selected_idx, options)

        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        if key == curses.KEY_UP:
            selected_idx = (selected_idx - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected_idx = (selected_idx + 1) % len(options)
        elif key == ord("\n") or key == curses.KEY_ENTER or key == 10 or key == 13:
            # Execute
            if selected_idx == 0:
                run_classic_snake(stdscr)

            elif selected_idx == 1 or selected_idx == 2:
                # Load model if needed
                if model is None:
                    model_path, meta_path = prompt_model_selection(stdscr, model_dir)
                    if not model_path:
                        continue  # Cancelled

                    stdscr.clear()
                    stdscr.addstr(10, 10, f"Loading Neural Model on {device}...")
                    stdscr.refresh()

                    try:
                        model, meta = load_model(model_path, meta_path, device)
                    except Exception as e:
                        stdscr.addstr(12, 10, f"Error loading model: {e}")
                        stdscr.getch()
                        continue

                if selected_idx == 1:
                    run_neural_snake(stdscr, model, meta, device)
                elif selected_idx == 2:
                    run_shadow_snake(stdscr, model, meta, device)

            elif selected_idx == 3:
                break
        elif key == 27 or key == ord("q"):  # ESC
            break


if __name__ == "__main__":
    curses.wrapper(main)
