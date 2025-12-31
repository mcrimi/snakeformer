import sys
import os
import curses
import torch
import time

from collections import deque

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.snake import SnakeGame

from game.shared import prompt_model_selection, load_gpt_model

# Key Mapping
CMD_UP = curses.KEY_UP
CMD_DOWN = curses.KEY_DOWN
CMD_LEFT = curses.KEY_LEFT
CMD_RIGHT = curses.KEY_RIGHT

KEY_STR_MAP = {CMD_UP: "U", CMD_DOWN: "D", CMD_LEFT: "L", CMD_RIGHT: "R"}


class NeuralSnakeGame(SnakeGame):
    def __init__(self, stdscr, model, meta, device):
        self.d_model = model
        self.d_meta = meta
        self.device = device
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.start_char_id = self.stoi.get(".")  # Just a default
        self.stop_token_id = self.stoi.get("$")
        self.prev_board = ""
        self.prev_generated = ""

        # Input Buffer
        self.input_queue = deque(maxlen=3)

        # Initialize parent
        super().__init__(stdscr)

        # But for now let's keep it responsive to input but wait for model
        self.stdscr.timeout(100)

    def construct_prompt(self, board_str, action_char):
        if self.prev_board != "" and self.prev_generated != "":
            # We include the previous board and generated in the prompt to make the most
            # of the context available to the model (1024 tokens)
            prev_board_str = f"{self.prev_board}\n{self.prev_generated}\n$"
            prompt = f"{prev_board_str}\nB:\n{board_str}\nA:{action_char}\nT:\n"
        else:
            prompt = f"B:\n{board_str}\nA:{action_char}\nT:\n"
        return prompt

    def encode_text(self, text):
        return [self.stoi.get(c, self.stoi.get(".", 0)) for c in text]

    def decode_tokens(self, tokens):
        return "".join([self.itos[i] for i in tokens])

    def record_turn_data(self, board_str, action_char, generated):
        # Update valid history for next turn
        self.prev_board = f"B:\n{board_str}\nA:{action_char}"
        self.prev_generated = f"T:\n{generated}"

        # UI Panels
        self.left_panel = self.prev_board.split("\n")
        self.right_panel = self.prev_generated.split("\n")

        # File Log
        if self.record_file:
            entry = f"{self.prev_board}\n{self.prev_generated}\n$"
            try:
                with open(self.record_file, "a") as f:
                    f.write(entry + "\n")
            except:
                pass

    def handle_input(self):
        # Flush the buffer to avoid lag due to slow inference
        keys = []
        while True:
            try:
                k = self.stdscr.getch()
            except:
                k = -1
            if k == -1:
                break
            keys.append(k)

        if not keys:
            return -1

        # Determine the reference direction for the first move in this batch
        # If we have a queue, the last item is where we will be facing after all queued moves
        if self.input_queue:
            last_scheduled_dir = self.input_queue[-1]
        else:
            last_scheduled_dir = self.direction
            if last_scheduled_dir is None:
                last_scheduled_dir = curses.KEY_RIGHT

        final_key = -1

        for key in keys:
            # Map WASD to Arrow Keys
            if key == ord("w") or key == ord("W"):
                key = curses.KEY_UP
            if key == ord("s") or key == ord("S"):
                key = curses.KEY_DOWN
            if key == ord("a") or key == ord("A"):
                key = curses.KEY_LEFT
            if key == ord("d") or key == ord("D"):
                key = curses.KEY_RIGHT

            final_key = key

            # Queue Logic
            # Only append if it's a valid 90 degree turn from the LAST scheduled direction
            # If queue is full, ignore (prevents running too far ahead)
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

    def update(self):
        if self.game_over:
            return

        # Get the logical string from the curses game state
        board_str = self.render_board_state(self.snake, self.food)

        # Consuming Input Queue
        if self.input_queue:
            self.direction = self.input_queue.popleft()

        # If no direction set, default to Right
        if self.direction is None:
            self.direction = curses.KEY_RIGHT

        action_char = KEY_STR_MAP.get(self.direction, "R")
        self.action_history.append(("EXEC", action_char))

        # 1. Construct Prompt
        prompt = self.construct_prompt(board_str, action_char)

        # 2. Inference
        context_idxs = self.encode_text(prompt)
        context = torch.tensor(
            context_idxs, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        try:
            # Generate tokens
            output_ids = self.d_model.generate(
                context, max_new_tokens=276, stop_token_id=self.stop_token_id
            )
            output_text = self.decode_tokens(output_ids[0].tolist())

            # Extract the newly generated part (after T:\n)
            generated = output_text[len(prompt) :]

            # Clean up: stop at '$'
            if "$" in generated:
                generated = generated.split("$")[0]

            # 3. Parse and Update
            generated = generated.strip()

            if (
                "X" in generated and len(generated) < 10
            ):  # X usually stands alone or with newlines
                self.game_over = True
            else:
                # Parse grid
                self.update_state_from_ascii(generated)

            # 4. Recording & UI
            self.record_turn_data(board_str, action_char, generated)

        except Exception as e:
            # If inference fails, maybe just game over or print error to log
            # self.game_over = True
            pass

    def update_state_from_ascii(self, ascii_board):
        lines = ascii_board.strip().split("\n")

        new_snake = []
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
            # Reconstruct Chain using topology (not raster order)
            body_parts = set(body)
            current = tuple(head)
            ordered_body = []

            # Simple greedy path following
            # This assumes the snake is contiguous (which the model should produce)
            while body_parts:
                # Find neighbors of current in body_parts
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = current[0] + dy, current[1] + dx
                    if (ny, nx) in body_parts:
                        neighbors.append((ny, nx))

                if not neighbors:
                    break

                # Pick the first neighbor (blindly)
                # In complex self-touching cases, this might be ambiguous,
                # but valid snake shouldn't self-touch in a way that creates ambiguous branches
                # (unless 3x3 loop, but even then).
                next_part = neighbors[0]
                ordered_body.append([next_part[0], next_part[1]])
                if next_part in body_parts:
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
    model_dir = os.path.join(base_dir, "model", "weigths")

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

    # Loading Message
    stdscr.clear()
    stdscr.addstr(10, 10, f"Loading Neural Model on {device}...")
    stdscr.refresh()

    try:
        model, meta = load_gpt_model(model_path, meta_path, device)
    except Exception as e:
        stdscr.addstr(12, 10, f"Error: {e}")
        stdscr.getch()
        return

    # Start Game
    game = NeuralSnakeGame(stdscr, model, meta, device)
    game.run()


if __name__ == "__main__":
    # We use curses wrapper to handle init/cleanup
    curses.wrapper(main)
