import sys
import os
import curses

import torch
import time


# Add the parent directory to sys.path if not already there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from games.neural_snake import (
    NeuralSnakeGame,
    KEY_STR_MAP,
)
from games.shared import prompt_model_selection, load_model


class ShadowNeuralSnakeGame(NeuralSnakeGame):
    """
    Run neural + deterministic physics side-by-side.
    Catch hallucinations when they disagree.
    """

    def __init__(self, stdscr, model, meta, device, streaming_delay_ms=0):
        super().__init__(stdscr, model, meta, device, streaming_delay_ms)
        self.ground_truth_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.ground_truth_str = ""
        self.prompt = ""
        self.sync_shadow_to_neural_pending = False
        self.divergence_index = -1  # char position where model went wrong

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.model_updated = False

    def generate_verified_move(self, prompt, expected_board_str):
        """
        Stream tokens from model, comparing each to expected_board_str.
        Stop early if mismatch found. Returns tuple with divergence info.
        """

        context_idxs = self.encode_text(prompt)
        idx = torch.tensor(context_idxs, dtype=torch.long, device=self.device).unsqueeze(0)

        max_new_tokens = 276  # 275 board chars + stop token
        generated_so_far = ""
        divergence_found = False
        divergence_char_idx = -1
        ground_truth_char = ""
        ground_truth_token = 0
        generated_token = 0
        new_char = ""

        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.snakeformer.config.block_size :]
            logits, _ = self.snakeformer(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            new_char = self.itos.get(idx_next.item(), "")

            if self.stop_token_id is not None and idx_next.item() == self.stop_token_id:
                break

            generated_so_far += new_char
            generated_token = idx_next.item()

            self.right_panel = f"T:\n{generated_so_far}".split("\n")
            self.render()

            if self.streaming_delay > 0:
                time.sleep(self.streaming_delay)

            # Compare against ground truth
            if i < len(expected_board_str):
                ground_truth_char = expected_board_str[i]
                ground_truth_token = self.stoi.get(
                    ground_truth_char, self.stoi.get(".", 0)
                )
                if generated_token != ground_truth_token:
                    divergence_found = True
                    divergence_char_idx = i
                    break  # Stop generation immediately
            else:
                if new_char.strip() == "":
                    pass
                else:
                    divergence_found = True
                    divergence_char_idx = i
                    break

        # Model stopped early? That's also a divergence
        if not divergence_found:
            if len(generated_so_far.strip()) < len(expected_board_str.strip()):
                divergence_found = True
                divergence_char_idx = len(generated_so_far)

        return (
            generated_so_far,
            divergence_found,
            divergence_char_idx,
            ground_truth_char,
            ground_truth_token,
            generated_token,
            new_char,
        )

    def _sync_state_from_inference(self):
        """Overwrite game state with whatever the model predicted (post-divergence continue)."""
        try:
            board_str = self.render_board_state(self.snake, self.food)
            if self.direction is None:
                self.direction = curses.KEY_RIGHT
            action_char = KEY_STR_MAP.get(self.direction, "R")

            prompt = self.construct_prompt(board_str, action_char)
            context_idxs = self.encode_text(prompt)
            context = torch.tensor(
                context_idxs, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            output_ids = self.snakeformer.generate(
                context, max_new_tokens=276, stop_token_id=self.stop_token_id
            )
            output_text = self.decode_tokens(output_ids[0].tolist())
            generated = output_text[len(prompt) :]

            if "$" in generated:
                generated = generated.split("$")[0]
            predicted_board_content = generated.strip()

            if "X" in predicted_board_content and len(predicted_board_content) < 10:
                self.game_over = True
            else:
                self.update_state_from_ascii(predicted_board_content)

            self.record_turn_data(board_str, action_char, predicted_board_content)

            self.divergence_detected = False
            self.divergence_index = -1

        except Exception:
            pass

        self.sync_shadow_to_neural_pending = False

    def _capture_current_state(self, retry):
        """Grab current board + action. If retry, reuse last direction."""
        board_str = self.render_board_state(self.snake, self.food)

        if not retry:
            action_char = self.consume_input_queue()
            self.action_history.append(("EXEC", action_char))
        else:
            if self.direction is None:
                self.direction = curses.KEY_RIGHT
            action_char = KEY_STR_MAP.get(self.direction, "R")

        return board_str, action_char

    def _simulate_shadow_physics(self):
        """Run deterministic physics to get ground truth next state."""
        shadow_snake, shadow_food, shadow_game_over, expected_board_str = (
            self.simulate_next_step(self.snake, self.food, self.direction)
        )

        if shadow_game_over:
            expected_board_str = "X"

        self.ground_truth_panel = f"T:\n{expected_board_str}".split("\n")

        return expected_board_str, shadow_game_over

    def _handle_divergence(
        self, divergence_data, prompt, shadow_game_over, board_str, action_char
    ):
        """Populate divergence info and set up the "uh oh" state."""
        (generated, _, div_idx, gt_char, gt_token, gen_token, gen_char) = (
            divergence_data
        )

        self.ground_truth_token = gt_token
        self.ground_truth_char = gt_char
        self.generated_token = gen_token
        self.generated_char = gen_char
        self.prompt = prompt
        self.divergence_detected = True
        self.divergence_index = div_idx

        # Classify the disagreement
        if "X" in generated and not shadow_game_over:
            self.divergence_msg = "Model predicted Die, but Physics says Live"
        elif shadow_game_over and "X" not in generated:
            self.divergence_msg = "Physics says Die, but Model predicted Live"
        else:
            self.divergence_msg = "Board State Mismatch"

        self.left_panel = f"B:\n{board_str}\nA:{action_char}".split("\n")
        self.right_panel = f"T:\n{generated}".split("\n")

    def _apply_verified_state(self, generated, board_str, action_char):
        """Commit the model's output to game state (it passed verification)."""
        predicted_board_content = generated.strip()
        if "X" in predicted_board_content and len(predicted_board_content) < 10:
            self.game_over = True
        else:
            self.update_state_from_ascii(predicted_board_content)

        self.record_turn_data(board_str, action_char, predicted_board_content)

    def update(self, retry=False):
        """One tick: get input, run shadow physics, generate + validate model output."""
        if self.sync_shadow_to_neural_pending:
            self._sync_state_from_inference()
            return

        if self.game_over or self.divergence_detected:
            return

        board_str, action_char = self._capture_current_state(retry)
        expected_board_str, shadow_game_over = self._simulate_shadow_physics()
        prompt = self.construct_prompt(board_str, action_char)
        self.left_panel = f"B:\n{board_str}\nA:{action_char}".split("\n")

        try:
            result = self.generate_verified_move(prompt, expected_board_str)
            (generated, divergence, _, _, _, _, _) = result

            if divergence:
                self._handle_divergence(result, prompt, shadow_game_over, board_str, action_char)
                return

            self._apply_verified_state(generated, board_str, action_char)
        except Exception:
            pass

    def render(self):
        """Draw 3-panel layout: board | neural prediction | shadow truth."""
        self.stdscr.erase()
        offset_y, offset_x = self.get_centered_offsets()

        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)
        self.draw_snake(offset_y, offset_x)
        self.draw_food(offset_y, offset_x)

        score_text = f" Score: {self.score}   "
        quit_text = "'Q' to Quit"
        if self.divergence_detected:
            score_text += " [DIVERGENCE DETECTED!] "

        try:
            self.stdscr.addstr(
                offset_y + self.game_height + 1,
                offset_x + 2,
                score_text,
                curses.color_pair(4) | curses.A_BOLD,
            )
            self.stdscr.addstr(
                offset_y + self.game_height + 1,
                offset_x + 2 + len(score_text),
                quit_text,
                curses.color_pair(7),
            )
        except curses.error:
            pass

        if self.game_over:
            if not self.divergence_detected:
                self.draw_game_over_message(offset_y, offset_x)

        if self.divergence_detected:
            self.draw_divergence_menu(offset_y, offset_x)

        self.draw_left_panel(offset_y, offset_x)

        # Right panel with divergence highlighting
        sh, sw = self.stdscr.getmaxyx()
        rx = offset_x + (self.game_width * 2) + 6
        ry = offset_y
        try:
            self.stdscr.addstr(
                ry - 1, rx, "Predicted (T):", curses.color_pair(4) | curses.A_BOLD
            )

            for i, line in enumerate(self.right_panel):
                if ry + i < sh:
                    for j, char in enumerate(line):
                        raw_idx = sum(len(l) + 1 for l in self.right_panel[:i]) + j
                        board_idx = raw_idx - 3  # skip "T:\n"

                        color = curses.color_pair(4)
                        if self.divergence_detected and self.divergence_index == board_idx:
                            color = curses.color_pair(2) | curses.A_REVERSE  # red = wrong

                        self.stdscr.addstr(ry + i, rx + j, char, color)
        except curses.error:
            pass

        # Shadow (ground truth) panel
        sx = rx + 20
        sy = offset_y
        try:
            self.stdscr.addstr(sy - 1, sx, "Shadow (T):", curses.color_pair(4) | curses.A_BOLD)
            for i, line in enumerate(self.ground_truth_panel):
                if sy + i < sh:
                    for j, char in enumerate(line):
                        raw_idx = sum(len(l) + 1 for l in self.ground_truth_panel[:i]) + j
                        board_idx = raw_idx - 3

                        color = curses.color_pair(4)
                        if self.divergence_detected and self.divergence_index == board_idx:
                            color = curses.color_pair(5) | curses.A_REVERSE  # cyan = expected

                        self.stdscr.addstr(sy + i, sx + j, char, color)
        except curses.error:
            pass

        self.draw_action_log(offset_y, offset_x)
        self.stdscr.refresh()

    def draw_divergence_menu(self, y_off, x_off):
        msg_lines = [
            "⚠️  DIVERGENCE DETECTED",
            "",
            "'Q' to  Quit",
            "'T' to  Train",
            "'C' to Continue",
        ]

        box_width = 30
        box_height = len(msg_lines) + 2

        cy = y_off + self.game_height // 2 - box_height // 2
        cx = x_off + (self.game_width * 2 - box_width) // 2

        for i in range(box_height):
            try:
                self.stdscr.addstr(cy + i, cx, " " * box_width, curses.color_pair(4) | curses.A_REVERSE)
            except curses.error:
                pass

        for i, line in enumerate(msg_lines):
            try:
                pad = (box_width - len(line)) // 2
                self.stdscr.addstr(
                    cy + 1 + i,
                    cx + max(0, pad),
                    line,
                    curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD,
                )
            except curses.error:
                pass

    def draw_training_visualization(
        self, probs, target_token, predicted_token, loss, weight, step
    ):
        """Show a bar chart of token probabilities during online training."""
        sh, sw = self.stdscr.getmaxyx()
        box_h, box_w = 20, 60
        y = (sh - box_h) // 2
        x = (sw - box_w) // 2

        # Clear box with opaque background
        for i in range(box_h):
            self.stdscr.addstr(
                y + i, x, " " * box_w, curses.color_pair(4) | curses.A_REVERSE
            )

        # Border
        # self.draw_box(y, x, box_h - 2, box_w - 4) # Box method might not use reverse color
        # Draw manual border for consistency
        self.stdscr.addstr(
            y, x, "┌" + "─" * (box_w - 2) + "┐", curses.color_pair(4) | curses.A_REVERSE
        )
        self.stdscr.addstr(
            y + box_h - 1,
            x,
            "└" + "─" * (box_w - 2) + "┘",
            curses.color_pair(4) | curses.A_REVERSE,
        )
        for i in range(1, box_h - 1):
            self.stdscr.addstr(y + i, x, "│", curses.color_pair(4) | curses.A_REVERSE)
            self.stdscr.addstr(
                y + i, x + box_w - 1, "│", curses.color_pair(4) | curses.A_REVERSE
            )

        # Title
        title = f" TRAINING STEP {step} "
        self.stdscr.addstr(
            y + 1,
            x + (box_w - len(title)) // 2,
            title,
            curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD,
        )

        # Stats
        stats = f"Loss: {loss:.4f} | Weight: {weight}x"
        self.stdscr.addstr(y + 3, x + 4, stats, curses.color_pair(4) | curses.A_REVERSE)

        # Bar Chart
        # Select top N tokens to show + target if not in top N
        top_k = 8
        vals, indices = torch.topk(probs, k=top_k)

        # Handle batch dimension (B=1)
        vals = vals[0]
        indices = indices[0]

        # Ensure target is included for comparison
        target_idx = target_token.item()
        if target_idx not in indices:
            # Replace last one with target for visibility
            indices[-1] = target_idx
            vals[-1] = probs[0, target_idx]

        max_val = vals[0].item()

        for i in range(len(indices)):
            idx = indices[i].item()
            prob = vals[i].item()
            char = self.itos.get(idx, "?")

            # Map prob to bar length (max width ~30 chars)
            bar_len = int((prob / (max_val + 1e-6)) * 30)
            bar_str = "█" * bar_len

            # Label
            label = f"'{char}' ({prob:.2f})"

            # Color
            # Base color for background box is Reverse White-on-Black (looks like block)
            # We want bars to pop.

            # Default Text Style
            style = curses.color_pair(4) | curses.A_REVERSE

            if idx == target_idx:
                # Target: Green Bar
                # we need to render the BAR in green, but background stays?
                # Curses limitation: can't easily mix background colors linearly.
                # Let's simple use [TARGET] text.
                label += " [TARGET]"
                style = curses.color_pair(2) | curses.A_REVERSE | curses.A_BOLD
            elif idx == predicted_token:
                label += " [PRED]"
                style = curses.color_pair(5) | curses.A_REVERSE | curses.A_BOLD  # Cyan

            line_str = f"{label:<20} {bar_str:<32}"
            self.stdscr.addstr(y + 5 + i, x + 4, line_str, style)

        self.stdscr.refresh()
        time.sleep(0.5)  # Pause to let user see

    def run_online_training(self):
        """Backprop on the exact token where model diverged. Keep going until it gets it right."""
        curses.flash()
        self.action_history.append(("SYS", "Online Training Triggered"))

        attempts = 0

        # While divergence detected, run model training and board regeneration
        while self.divergence_detected and attempts < 10:
            # Re-compute shadow state (context = prompt + correct prefix up to divergence)
            shadow_snake, shadow_food, shadow_game_over, current_shadow_str = (
                self.simulate_next_step(self.snake, self.food, self.direction)
            )
            if shadow_game_over:
                current_shadow_str = "X"

            prefix_str = current_shadow_str[: self.divergence_index]
            full_context_str = self.prompt + prefix_str
            context_idxs = self.encode_text(full_context_str)

            # Truncate to block_size if needed
            if len(context_idxs) > self.snakeformer.config.block_size:
                context_idxs = context_idxs[-self.snakeformer.config.block_size :]

            context_tensor = torch.tensor(context_idxs, dtype=torch.long, device=self.device).unsqueeze(0)
            target_tensor = torch.tensor([self.ground_truth_token], dtype=torch.long, device=self.device)

            # Important chars (H, T, F, X) get 50x weight
            weight = 50.0 if any(c in "HTFX" for c in (self.ground_truth_char, self.generated_char)) else 1.0

            loss, probs = self.snakeformer.train_step(
                self.optimizer, context_tensor, target_tensor, importance_weight=weight
            )
            self.action_history.append(("TRN", f"Opt... Loss: {loss:.4f}"))

            self.draw_training_visualization(probs, target_tensor, self.generated_token, loss, weight, attempts + 1)

            # Retry generation to see if we fixed it
            self.divergence_detected = False
            self.update(retry=True)
            attempts += 1
            self.model_updated = True

        if not self.divergence_detected:
            self.action_history.append(("TRN", "Success! Fixed."))
        else:
            self.action_history.append(("TRN", "Failed to converge."))
            self.run_online_training()

        return True

    def handle_game_over_input(self):
        try:
            key = self.stdscr.getch()
        except:
            key = -1

        if key == ord("q") or key == ord("Q"):
            return "quit"
        elif key == ord("r") or key == ord("R"):
            return "restart"
        return None

    def prompt_save_model(self):
        """Ask user to save updated weights (if any training happened)."""
        if not self.model_updated:
            return

        curses.flash()
        self.stdscr.nodelay(False)

        sh, sw = self.stdscr.getmaxyx()
        h, w = 10, 60
        y, x = (sh - h) // 2, (sw - w) // 2

        self.stdscr.attron(curses.color_pair(4) | curses.A_REVERSE)
        for i in range(h):
            self.stdscr.addstr(y + i, x, " " * w)
        self.stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE)

        msg = "Model was updated! Save? (y/N)"
        self.stdscr.addstr(
            y + 2,
            x + (w - len(msg)) // 2,
            msg,
            curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD,
        )
        self.stdscr.refresh()

        while True:
            k = self.stdscr.getch()
            if k == ord("y") or k == ord("Y"):
                break
            elif k == ord("n") or k == ord("N") or k == 27:  # Esc
                self.stdscr.nodelay(True)
                return

        # Ask for Filename
        prompt = "Filename (in model/weights/): "
        self.stdscr.addstr(
            y + 4, x + 2, prompt, curses.color_pair(4) | curses.A_REVERSE
        )
        curses.echo()
        curses.curs_set(1)

        fname_bytes = self.stdscr.getstr(y + 4, x + 2 + len(prompt), 30)
        fname = fname_bytes.decode("utf-8").strip()

        curses.noecho()
        curses.curs_set(0)

        if not fname:
            fname = "snake_model_updated"

        if not fname.endswith(".pt"):
            fname += ".pt"

        # Path logic
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, "model", "weights", fname)

        # Check overwrite
        if os.path.exists(save_path):
            warn = "File exists! Type 'yes' to overwrite: "
            self.stdscr.addstr(
                y + 6,
                x + 2,
                warn,
                curses.color_pair(2) | curses.A_REVERSE | curses.A_BOLD,
            )
            curses.echo()
            curses.curs_set(1)
            confirm_bytes = self.stdscr.getstr(y + 6, x + 2 + len(warn), 10)
            confirm = confirm_bytes.decode("utf-8").strip()
            curses.noecho()
            curses.curs_set(0)

            if confirm.lower() != "yes":
                self.stdscr.nodelay(True)
                return

        # Save
        try:
            torch.save(self.snakeformer.state_dict(), save_path)
            succ = f"Saved to {fname}"
            self.stdscr.addstr(
                y + 8,
                x + (w - len(succ)) // 2,
                succ,
                curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD,
            )
            self.stdscr.refresh()
            time.sleep(1.5)
        except Exception as e:
            err = "Save Failed!"
            self.stdscr.addstr(
                y + 8, x + 2, err, curses.color_pair(2) | curses.A_REVERSE
            )
            self.stdscr.refresh()
            time.sleep(1)

        self.stdscr.nodelay(True)

    def reset_game(self):
        """Fresh board, keep the model. No need to reload weights."""
        self.score = 0
        self.game_over = False
        cy, cx = self.game_height // 2, self.game_width // 2
        self.snake = [[cy, cx], [cy, cx - 1], [cy, cx - 2]]
        self.direction = curses.KEY_RIGHT
        self.spawn_food()

        self.input_queue.clear()
        self.action_history.clear()

        # Clear divergence state (but keep model_updated for save prompt)
        self.ground_truth_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.divergence_index = -1
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.prev_board = ""
        self.prev_generated = ""
        self.sync_shadow_to_neural_pending = False

    def run(self):
        """Main loop. Handle divergence menu, game over, or regular play."""
        while True:
            if self.divergence_detected:
                try:
                    key = self.stdscr.getch()
                except:
                    key = -1

                if key == ord("q") or key == ord("Q"):
                    break
                elif key == ord("c") or key == ord("C"):
                    self.sync_shadow_to_neural_pending = True
                elif key == ord("t") or key == ord("T"):
                    self.run_online_training()

            elif self.game_over:
                action = self.handle_game_over_input()
                if action == "quit":
                    break
                elif action == "restart":
                    self.reset_game()
                    continue
            else:
                key = self.handle_input()
                if key == ord("q") or key == ord("Q"):
                    break

            self.update()
            self.render()

        self.prompt_save_model()


def main(stdscr, model_filename=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model", "weights")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if model_filename:
        model_path = os.path.join(model_dir, model_filename)
        if not model_filename.endswith(".pt"):
            model_path = os.path.join(model_dir, model_filename + ".pt")
        meta_path = os.path.join(model_dir, "meta.pkl")
    else:
        model_path, meta_path = prompt_model_selection(stdscr, model_dir)
        if not model_path:
            return

    if not os.path.exists(model_path):
        stdscr.addstr(0, 0, f"Error: Model not found at {model_path}")
        stdscr.refresh()
        time.sleep(2)
        return

    if not os.path.exists(meta_path):
        stdscr.addstr(0, 0, f"Error: Meta not found at {meta_path}")
        stdscr.refresh()
        time.sleep(2)
        return

    stdscr.clear()
    stdscr.addstr(10, 10, f"Loading Snakeformer Model on {device}...")
    stdscr.addstr(11, 10, f"Model: {os.path.basename(model_path)}")
    stdscr.refresh()

    try:
        model, meta = load_model(model_path, meta_path, device)
    except Exception as e:
        stdscr.addstr(12, 10, f"Error: {e}")
        stdscr.getch()
        return

    game = ShadowNeuralSnakeGame(stdscr, model, meta, device)
    game.run()


if __name__ == "__main__":
    curses.wrapper(main)
