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
    A 'Shadow' game that validates Neural Snake's hallucinated moves against real physics.

    It runs two engines in parallel:
    1. The Neural Engine (Snakeformer) predicting the next board state.
    2. The Shadow Engine (Deterministic) calculating the actual physics.

    If they disagree, we catch it and let the user do something about it.
    """

    def __init__(self, stdscr, model, meta, device):
        super().__init__(stdscr, model, meta, device)
        self.ground_truth_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.ground_truth_str = ""
        self.prompt = ""
        self.sync_shadow_to_neural_pending = False
        self.divergence_index = (
            -1
        )  # Index in the current panel visual string (0-based) where mismatch happened

        # Optimizer for online correction
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.model_updated = False

    def generate_verified_move(self, prompt, expected_board_str):
        """
        Generates tokens step-by-step from the model, validating each against the expected shadow target string coming from the
        deterministic version of the game (snakeformer.game.neural_snake.NeuralSnakeGame).
        Identifies the exact point of divergence if the model's output differs from the ground truth.

        Args:
            prompt (str): The input prompt to generate tokens from.
            expected_board_str (str): The expected shadow target string coming from the deterministic version of the game.

        Returns:
            tuple: A tuple containing the generated string, divergence found flag, divergence character index,
                   ground truth character, ground truth token, generated token, and new character.
        """

        context_idxs = self.encode_text(prompt)
        idx = torch.tensor(
            context_idxs, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Max tokens to generate (276 = 275 + 1 for '$' stop token)
        max_new_tokens = 276

        generated_so_far = ""
        divergence_found = False
        divergence_char_idx = -1

        for i in range(max_new_tokens):
            # crop context
            idx_cond = idx[:, -self.snakeformer.config.block_size :]

            # forward pass
            logits, _ = self.snakeformer(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append token to context
            idx = torch.cat((idx, idx_next), dim=1)

            # decode just this token
            new_char = self.itos.get(idx_next.item(), "")

            # Check for stop token
            if self.stop_token_id is not None and idx_next.item() == self.stop_token_id:
                break

            generated_so_far += new_char

            # Check ground_truth token against generated token
            if i < len(expected_board_str):
                ground_truth_char = expected_board_str[i]
                ground_truth_token = self.stoi.get(
                    ground_truth_char, self.stoi.get(".", 0)
                )
                generated_token = idx_next.item()
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

        # Post-loop validation: Check if the model generated the complete board sequence.
        # If the model terminated prematurely (e.g., by emitting a stop token) before
        # reaching the expected length of the shadow target, flag it as a divergence.
        if not divergence_found:
            if len(generated_so_far.strip()) < len(expected_board_str.strip()):
                divergence_found = True
                # Mark the divergence at the end of the generated string.
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
        """
        Forces the game engine state to match the model's prediction.
        Used when the user chooses to 'Continue' after a divergence.
        """
        try:
            board_str = self.render_board_state(self.snake, self.food)
            if self.direction is None:
                self.direction = curses.KEY_RIGHT
            action_char = KEY_STR_MAP.get(self.direction, "R")

            # Generate full state prediction to align the engine
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

            # Update engine state to match the model's output
            if "X" in predicted_board_content and len(predicted_board_content) < 10:
                self.game_over = True
            else:
                self.update_state_from_ascii(predicted_board_content)

            self.record_turn_data(board_str, action_char, predicted_board_content)

            # Reset divergence tracking
            self.divergence_detected = False
            self.divergence_index = -1

        except Exception:
            pass

        self.sync_shadow_to_neural_pending = False

    def _capture_current_state(self, retry):
        """Captures the current board string and determines the action character."""
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
        """Running the deterministic physics engine to get the expected next state."""
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
        """Sets up the state when a divergence is detected."""
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

        # Determine why the divergence happened
        if "X" in generated and not shadow_game_over:
            self.divergence_msg = "Model predicted Die, but Physics says Live"
        elif shadow_game_over and "X" not in generated:
            self.divergence_msg = "Physics says Die, but Model predicted Live"
        else:
            self.divergence_msg = "Board State Mismatch"

        # Update panels with partial (divergent) generation
        self.left_panel = f"B:\n{board_str}\nA:{action_char}".split("\n")
        self.right_panel = f"T:\n{generated}".split("\n")

    def _apply_verified_state(self, generated, board_str, action_char):
        """Updates the actual game state based on the verified prediction."""
        predicted_board_content = generated.strip()
        if "X" in predicted_board_content and len(predicted_board_content) < 10:
            self.game_over = True
        else:
            self.update_state_from_ascii(predicted_board_content)

        self.record_turn_data(board_str, action_char, predicted_board_content)

    def update(self, retry=False):
        """
        Main game loop iteration:
        1. Syncs state if a divergence was overridden.
        2. Captures board state and user input.
        3. Calculates 'Shadow' ground truth via deterministic physics.
        4. Generates model predictions token-by-token, validating against ground truth.
        5. Flags divergence if the model hallucinates an invalid state.
        """
        # --- 0. State Synchronization ---
        if self.sync_shadow_to_neural_pending:
            self._sync_state_from_inference()
            return

        if self.game_over or self.divergence_detected:
            return

        # --- 1. Capture Current Environment ---
        board_str, action_char = self._capture_current_state(retry)

        # -- 2. Shadow Calculation (Ground Truth) --
        expected_board_str, shadow_game_over = self._simulate_shadow_physics()

        # -- 3. Neural Inference & Validation --
        prompt = self.construct_prompt(board_str, action_char)

        try:
            result = self.generate_verified_move(prompt, expected_board_str)
            (generated, divergence, _, _, _, _, _) = result

            if divergence:
                self._handle_divergence(
                    result, prompt, shadow_game_over, board_str, action_char
                )
                return

            # --- 4. Successful State Update ---
            self._apply_verified_state(generated, board_str, action_char)

        except Exception:
            pass

    def render(self):
        """
        Draws the game state, including the 3-panel layout:
        [Current Board] | [Neural Prediction] | [Shadow Truth]

        Also handles drawing the 'Divergence Detected' overlay when things go wrong.
        """
        self.stdscr.erase()
        offset_y, offset_x = self.get_centered_offsets()

        # Draw Border
        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)

        # Draw Elements
        self.draw_snake(offset_y, offset_x)
        self.draw_food(offset_y, offset_x)

        # Draw Score with divergence text
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

        # Left Panel (Parent logic)
        self.draw_left_panel(offset_y, offset_x)

        # Right Panel (Custom logic for highlighting)
        sh, sw = self.stdscr.getmaxyx()
        rx = offset_x + (self.game_width * 2) + 6
        ry = offset_y
        try:
            self.stdscr.addstr(
                ry - 1, rx, "Predicted (T):", curses.color_pair(4) | curses.A_BOLD
            )

            for i, line in enumerate(self.right_panel):
                if ry + i < sh:
                    # Draw char by char to handle highlighting
                    for j, char in enumerate(line):
                        # Reconstruct index in the FULL string "T:\n....."
                        raw_idx = sum(len(l) + 1 for l in self.right_panel[:i]) + j

                        # The board string starts after "T:\n", which is length 3.
                        # generated string index:
                        board_idx = raw_idx - 3

                        color = curses.color_pair(4)
                        if (
                            self.divergence_detected
                            and self.divergence_index == board_idx
                        ):
                            color = (
                                curses.color_pair(2) | curses.A_REVERSE
                            )  # Red Background

                        self.stdscr.addstr(ry + i, rx + j, char, color)
        except curses.error:
            pass

        # Ground Truth Panel (Target Shadow)
        sx = rx + 20
        sy = offset_y
        try:
            self.stdscr.addstr(
                sy - 1, sx, "Shadow (T):", curses.color_pair(4) | curses.A_BOLD
            )
            for i, line in enumerate(self.ground_truth_panel):
                if sy + i < sh:
                    # Draw char by char to handle highlighting matching the neural one
                    for j, char in enumerate(line):
                        # Reconstruct index in the FULL string "S:\n....."
                        raw_idx = (
                            sum(len(l) + 1 for l in self.ground_truth_panel[:i]) + j
                        )

                        # The board string starts after "S:\n", which is length 3.
                        board_idx = raw_idx - 3

                        color = curses.color_pair(4)
                        # Highlight the SAME index in shadow to show what was expected
                        if (
                            self.divergence_detected
                            and self.divergence_index == board_idx
                        ):
                            color = (
                                curses.color_pair(5) | curses.A_REVERSE
                            )  # Cyan Background for Expected

                        self.stdscr.addstr(sy + i, sx + j, char, color)
        except curses.error:
            pass

        self.draw_action_log()
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

        # Clear box area
        for i in range(box_height):
            try:
                self.stdscr.addstr(
                    cy + i, cx, " " * box_width, curses.color_pair(4) | curses.A_REVERSE
                )
            except curses.error:
                pass

        # Draw border
        # self.draw_box(cy, cx, box_height, box_width)

        for i, line in enumerate(msg_lines):
            try:
                # Center text in box
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
        """
        Draws an overlay visualization of the training process.
        probs: tensor (1, VocabSize)
        """
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
        """
        Calculates loss and updates the model weights on the fly!

        We force the model to learn from the exact context where it failed,
        providing the correct shadow character as the target to minimize loss.
        """
        curses.flash()
        self.action_history.append(("SYS", "Online Training Triggered"))

        attempts = 0

        # While divergence detected, run model training and board regeneration
        while self.divergence_detected and attempts < 10:
            # 1. Prepare Data
            # CRITICAL FIX: The context must include the prompt PLUS the correct generation up to the point of divergence.
            # We want to teach the model: "Given Prompt + CorrectPrefix, PREDICT CorrectNextToken"

            # We need the shadow string again to get the prefix.
            # Ideally we stored it, but we can reconstruct or just access self.shadow_panel (needs parsing)
            # Better: let's re-calculate it or store it in the class during update?
            # self.shadow_panel is list of strings centered in "S:\n...".
            # Let's just re-calculate get_deterministic_next_state is safe enough or store it in update check.

            # Actually, update() calculated 'shadow_target_str'. We should store that in self.
            # Let's assume we add self.shadow_target_str to update() in a moment.
            # For now, let's just re-calculate it to be safe and stateless here.
            shadow_snake, shadow_food, shadow_game_over, current_shadow_str = (
                self.simulate_next_step(self.snake, self.food, self.direction)
            )
            if shadow_game_over:
                current_shadow_str = "X"

            # Prefix is the shadow string UP TO the divergence index
            prefix_str = current_shadow_str[: self.divergence_index]

            # Correct context: Prompt + Prefix
            full_context_str = self.prompt + prefix_str

            context_idxs = self.encode_text(full_context_str)

            # Fix: Truncate to block_size (1024) to avoid RuntimeErrors
            if len(context_idxs) > self.snakeformer.config.block_size:
                context_idxs = context_idxs[-self.snakeformer.config.block_size :]

            # Debug: Check if lengths match expectation
            # print(f"Div Index: {self.divergence_index}, Prefix Len: {len(prefix_str)}")

            context_tensor = torch.tensor(
                context_idxs, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            target_tensor = torch.tensor(
                [self.ground_truth_token], dtype=torch.long, device=self.device
            )

            # 2. Determine Weight. 50x if either is H, T, F, or X in either ground truth or generated
            weight = (
                50.0
                if any(
                    c in "HTFX" for c in (self.ground_truth_char, self.generated_char)
                )
                else 1.0
            )

            # 3. Train Step
            loss, probs = self.snakeformer.train_step(
                self.optimizer, context_tensor, target_tensor, importance_weight=weight
            )

            self.action_history.append(("TRN", f"Opt... Loss: {loss:.4f}"))

            # VISUALIZE
            self.draw_training_visualization(
                probs,
                target_tensor,
                self.generated_token,  # Initial bad token
                loss,
                weight,
                attempts + 1,
            )

            # 4. Regenerate board to check if fixed
            # Pass retry=True to avoid consuming new inputs
            # Reset divergence flag temporarily so update() can run
            self.divergence_detected = False
            self.update(retry=True)
            attempts += 1

            # Mark model as updated
            self.model_updated = True

            # Use small delay or input flush to avoid spamming? No, it's instant loop.
            # time.sleep(0.1) -> moved to viz

        if not self.divergence_detected:
            self.action_history.append(("TRN", "Success! Fixed."))
            # TODO: Save model
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
        """
        If the model learned anything new, ask the user if they want to save
        the updated weights to disk.
        """
        if not self.model_updated:
            return

        curses.flash()
        self.stdscr.nodelay(False)  # Blocking input for filename

        sh, sw = self.stdscr.getmaxyx()
        h, w = 10, 60
        y, x = (sh - h) // 2, (sw - w) // 2

        # Draw Box
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
        """
        Resets the board state (snake position, food, score) but KEEPS the neural model.
        Useful for restarting after a Game Over without reloading the heavyweight model.
        """
        # Reset Game State but KEEP Model
        # Call super reset to match SnakeGame logic (center start, calculate food)
        # But we need to handle specific attributes ourselves if super doesn't

        # Copied from SnakeGame.reset_game (dynamic center)
        self.score = 0
        self.game_over = False
        cy, cx = self.game_height // 2, self.game_width // 2
        self.snake = [[cy, cx], [cy, cx - 1], [cy, cx - 2]]
        self.direction = curses.KEY_RIGHT

        # Use deterministic spawn from this class or parent?
        # Parent spawn_food relies on self.snake
        self.spawn_food()

        self.input_queue.clear()
        self.action_history.clear()

        # Shadow/Divergence State
        self.ground_truth_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.divergence_index = -1
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.prev_board = ""
        self.prev_generated = ""
        self.sync_shadow_to_neural_pending = False

        # We do NOT reset self.model_updated, so user can still save later if they want

    def run(self):
        """
        Main game loop for shadow tracking and divergence detection.
        """

        # Main Loop - while not quit or game over
        while True:
            # Check for token level divergence between shadow ground truth and model prediction
            if self.divergence_detected:
                try:
                    key = self.stdscr.getch()
                except:
                    key = -1

                if key == ord("q") or key == ord("Q"):
                    # User chooses to quit - break out of loop
                    break
                elif key == ord("c") or key == ord("C"):
                    # User chooses to continue with the game - force synchronize shadow to model prediction
                    self.sync_shadow_to_neural_pending = True
                elif key == ord("t") or key == ord("T"):
                    # Let's do some training 🐍
                    self.run_online_training()

            elif self.game_over:
                # Game Over specific loop
                action = self.handle_game_over_input()
                if action == "quit":
                    break
                elif action == "restart":
                    self.reset_game()
                    continue
            else:
                # Regular Gameplay
                key = self.handle_input()
                if key == ord("q") or key == ord("Q"):
                    # User chooses to quit - break out of loop
                    break

                if key == -1:
                    pass  # No input, just continue to update with the momentum board

            self.update()

            if self.game_over:
                self.render()  # Draw one last time to show GAME OVER state
            else:
                self.render()

        # On Exit Loop - save the new model if user wants to
        self.prompt_save_model()


def main(stdscr, model_filename=None):
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model", "weights")

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Model Selection Logic
    if model_filename:
        # If provided via arg (e.g. from play.py calling with specific model? No, play.py calls run_shadow_snake with loaded model)
        # This main is for STANDALONE execution.
        model_path = os.path.join(model_dir, model_filename)
        if not model_filename.endswith(".pt"):
            model_path = os.path.join(model_dir, model_filename + ".pt")
        meta_path = os.path.join(model_dir, "meta.pkl")
    else:
        # Interactive Selection
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
