import sys
import os
import curses
import torch
import time

from torch.nn import functional as F
from collections import deque

# Add the parent directory to sys.path if not already there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from games.snake import SnakeGame

from games.shared import prompt_model_selection, load_model

# Key Mapping
CMD_UP = curses.KEY_UP
CMD_DOWN = curses.KEY_DOWN
CMD_LEFT = curses.KEY_LEFT
CMD_RIGHT = curses.KEY_RIGHT

KEY_STR_MAP = {CMD_UP: "U", CMD_DOWN: "D", CMD_LEFT: "L", CMD_RIGHT: "R"}


class NeuralSnakeGame(SnakeGame):
    def __init__(self, stdscr, model, meta, device, streaming_delay_ms=0):
        self.snakeformer = model
        self.meta = meta
        self.device = device
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.start_char_id = self.stoi.get(".")  # Just a default
        self.stop_token_id = self.stoi.get("$")
        self.prev_board = ""
        self.prev_generated = ""

        self.streaming_delay = streaming_delay_ms / 1000.0
        self.input_queue = deque(maxlen=3)  # buffer for queued turns
        self.direction_from_user = False

        super().__init__(stdscr)
        self.stdscr.timeout(100)  # poll input while waiting for model

    def construct_prompt(self, board_str, action_char):
        # Include previous turn if available (more context = better predictions)
        if self.prev_board and self.prev_generated:
            prev = f"{self.prev_board}\n{self.prev_generated}\n$"
            return f"{prev}\nB:\n{board_str}\nA:{action_char}\nT:\n"
        return f"B:\n{board_str}\nA:{action_char}\nT:\n"

    def encode_text(self, text):
        return [self.stoi.get(c, self.stoi.get(".", 0)) for c in text]

    def decode_tokens(self, tokens):
        return "".join([self.itos[i] for i in tokens])

    def record_turn_data(self, board_str, action_char, generated):
        self.prev_board = f"B:\n{board_str}\nA:{action_char}"
        self.prev_generated = f"T:\n{generated}"
        self.left_panel = self.prev_board.split("\n")
        self.right_panel = self.prev_generated.split("\n")

        if self.record_file:
            entry = f"{self.prev_board}\n{self.prev_generated}\n$"
            try:
                with open(self.record_file, "a") as f:
                    f.write(entry + "\n")
            except Exception:
                pass

    def draw_action_log(self, offset_y=None, offset_x=None):
        """Show direction: cyan = user input, magenta = momentum."""
        if offset_y is None or offset_x is None:
            offset_y, offset_x = self.get_centered_offsets()

        log_y = offset_y + self.game_height + 3
        log_x = offset_x + 2
        current_char = KEY_STR_MAP.get(self.direction, "R")

        try:
            x_pos = log_x
            if self.direction_from_user:
                self.stdscr.addstr(log_y, x_pos, "Input:", curses.color_pair(5) | curses.A_DIM)
                self.stdscr.addstr(log_y, x_pos + 7, current_char, curses.color_pair(5) | curses.A_BOLD)
            else:
                self.stdscr.addstr(log_y, x_pos, "Momentum:", curses.color_pair(6) | curses.A_DIM)
                self.stdscr.addstr(log_y, x_pos + 10, current_char, curses.color_pair(6) | curses.A_BOLD)
        except curses.error:
            pass

    def handle_input(self):
        # Drain all pending keys (inference is slow, input piles up)
        keys = []
        while True:
            try:
                k = self.stdscr.getch()
            except Exception:
                k = -1
            if k == -1:
                break
            keys.append(k)

        if not keys:
            return -1

        # Reference = last queued direction, or current facing
        last_scheduled_dir = self.input_queue[-1] if self.input_queue else (self.direction or curses.KEY_RIGHT)
        final_key = -1

        for key in keys:
            # WASD → arrows
            key = {ord("w"): curses.KEY_UP, ord("W"): curses.KEY_UP,
                   ord("s"): curses.KEY_DOWN, ord("S"): curses.KEY_DOWN,
                   ord("a"): curses.KEY_LEFT, ord("A"): curses.KEY_LEFT,
                   ord("d"): curses.KEY_RIGHT, ord("D"): curses.KEY_RIGHT}.get(key, key)
            final_key = key

            # Only queue valid 90° turns (no 180s, no repeats)
            if len(self.input_queue) >= 3:
                continue

            valid_move = False
            if (
                key == curses.KEY_UP
                and last_scheduled_dir != curses.KEY_DOWN
                and last_scheduled_dir != curses.KEY_UP
            ):
                valid_move = True
                self.action_history.append(("USER", "U"))
            elif (
                key == curses.KEY_DOWN
                and last_scheduled_dir != curses.KEY_UP
                and last_scheduled_dir != curses.KEY_DOWN
            ):
                valid_move = True
                self.action_history.append(("USER", "D"))
            elif (
                key == curses.KEY_LEFT
                and last_scheduled_dir != curses.KEY_RIGHT
                and last_scheduled_dir != curses.KEY_LEFT
            ):
                valid_move = True
                self.action_history.append(("USER", "L"))
            elif (
                key == curses.KEY_RIGHT
                and last_scheduled_dir != curses.KEY_LEFT
                and last_scheduled_dir != curses.KEY_RIGHT
            ):
                valid_move = True
                self.action_history.append(("USER", "R"))

            if valid_move:
                self.input_queue.append(key)
                last_scheduled_dir = key

        return final_key

    def consume_input_queue(self):
        if self.input_queue:
            self.direction = self.input_queue.popleft()
            self.direction_from_user = True
        else:
            self.direction_from_user = False

        if self.direction is None:
            self.direction = curses.KEY_RIGHT
            self.direction_from_user = False

        return KEY_STR_MAP.get(self.direction, "R")

    def update(self):
        if self.game_over:
            return

        board_str = self.render_board_state(self.snake, self.food)
        action_char = self.consume_input_queue()
        self.action_history.append(("EXEC", action_char))

        prompt = self.construct_prompt(board_str, action_char)
        context_idxs = self.encode_text(prompt)
        context = torch.tensor(context_idxs, dtype=torch.long, device=self.device).unsqueeze(0)

        try:
            generated = self._generate_streaming(context, board_str, action_char).strip()

            if "X" in generated and len(generated) < 10:  # model says we're dead
                self.game_over = True
            else:
                self.update_state_from_ascii(generated)

            self.record_turn_data(board_str, action_char, generated)
        except Exception:
            pass

    def _generate_streaming(self, context, board_str, action_char):
        """Generate tokens with live display updates."""
        idx = context
        max_new_tokens = 276
        generated_so_far = ""
        self.left_panel = f"B:\n{board_str}\nA:{action_char}".split("\n")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.snakeformer.config.block_size :]
            logits, _ = self.snakeformer(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)  # greedy
            idx = torch.cat((idx, idx_next), dim=1)

            new_char = self.itos.get(idx_next.item(), "")
            if self.stop_token_id is not None and idx_next.item() == self.stop_token_id:
                break

            generated_so_far += new_char
            self.right_panel = f"T:\n{generated_so_far}".split("\n")
            self.render()

            if self.streaming_delay > 0:
                time.sleep(self.streaming_delay)

        return generated_so_far

    def update_state_from_ascii(self, ascii_board):
        lines = ascii_board.strip().split("\n")

        new_food = None
        head = None
        body = []

        for r, line in enumerate(lines):
            if r >= self.game_height:
                break

            for c, char in enumerate(line):
                if c >= self.game_width:
                    break

                if char == "H":
                    head = [r, c]
                elif char == "O" or char == "#":
                    body.append(
                        (r, c)
                    )  # Keep as list of tuples first, but we need set for lookup
                elif char == "F":
                    new_food = [r, c]

        if head:
            # Walk from head through adjacent body cells to reconstruct order
            body_parts = set(body)
            current = tuple(head)
            ordered_body = []

            while body_parts:
                neighbors = [(current[0]+dy, current[1]+dx) 
                             for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                             if (current[0]+dy, current[1]+dx) in body_parts]
                if not neighbors:
                    break
                next_part = neighbors[0]
                ordered_body.append([next_part[0], next_part[1]])
                body_parts.remove(next_part)
                current = next_part

            self.snake = [head] + ordered_body

        if new_food:
            self.food = new_food
        else:
            self.food = None  # Food might be eaten or not spawned

        self.score = (len(self.snake) - 3) * 10
        if self.score < 0:
            self.score = 0


def main(stdscr):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model", "weights")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model_path, meta_path = prompt_model_selection(stdscr, model_dir)
    if not model_path:
        return

    stdscr.clear()
    stdscr.addstr(10, 10, f"Loading Snakeformer Model on {device}...")
    stdscr.refresh()

    try:
        model, meta = load_model(model_path, meta_path, device)
    except Exception as e:
        stdscr.addstr(12, 10, f"Error: {e}")
        stdscr.getch()
        return

    game = NeuralSnakeGame(stdscr, model, meta, device)
    game.run()


if __name__ == "__main__":
    curses.wrapper(main)
