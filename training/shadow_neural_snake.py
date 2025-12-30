import sys
import os
import curses
import pickle
import torch
import time


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from game.neural_snake import (
    NeuralSnakeGame,
    KEY_STR_MAP,
    CMD_UP,
    CMD_DOWN,
    CMD_LEFT,
    CMD_RIGHT,
)
from model.gpt import GPT, GPTConfig


class ShadowNeuralSnakeGame(NeuralSnakeGame):
    def __init__(self, stdscr, model, meta, device):
        super().__init__(stdscr, model, meta, device)
        self.shadow_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.ground_truth_str = ""
        self.prompt = ""
        self.sync_shadow_to_neural_pending = False
        self.divergent_generated = ""
        self.divergence_index = (
            -1
        )  # Index in the current panel visual string (0-based) where mismatch happened

        # Optimizer for RL correction
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.model_updated = False

        # We need a separate shadow state that mirrors the game state
        # but evolves deterministically.

    def get_deterministic_next_state(self, current_snake, current_food, direction):
        # Replicates SnakeGame.update physics

        if not current_snake:
            return None, None, True  # Should not happen

        head = current_snake[0]
        new_head = [head[0], head[1]]

        if direction == curses.KEY_UP:
            new_head[0] -= 1
        elif direction == curses.KEY_DOWN:
            new_head[0] += 1
        elif direction == curses.KEY_LEFT:
            new_head[1] -= 1
        elif direction == curses.KEY_RIGHT:
            new_head[1] += 1

        # Bounds Check
        if (
            new_head[0] < 0
            or new_head[0] >= self.game_height
            or new_head[1] < 0
            or new_head[1] >= self.game_width
        ):
            return current_snake, current_food, True  # Game Over (Hit Wall)

        # Self Collision
        if new_head in current_snake:
            return current_snake, current_food, True  # Game Over (Hit Self)

        new_snake = [new_head] + current_snake

        game_over = False
        new_food = current_food

        if new_head == current_food:
            # Eat food, don't pop tail
            # Spawn new food
            new_food = self.get_deterministic_food_spawn(new_snake)
        else:
            new_snake.pop()  # Remove tail

        return new_snake, new_food, game_over

    def get_deterministic_food_spawn(self, snake):
        head = snake[0]
        target_r = (head[0] + 5) % self.game_height
        target_c = (head[1] + 7) % self.game_width

        start_r, start_c = target_r, target_c

        while [target_r, target_c] in snake:
            target_c = (target_c + 1) % self.game_width
            if target_c == 0:
                target_r = (target_r + 1) % self.game_height

            if (target_r, target_c) == (start_r, start_c):
                return None  # Board full

        return [target_r, target_c]

    def render_logical_board(self, snake, food):
        # Replicates get_logical_string
        grid = [["." for _ in range(self.game_width)] for _ in range(self.game_height)]

        if food:
            if 0 <= food[0] < self.game_height and 0 <= food[1] < self.game_width:
                grid[food[0]][food[1]] = "F"

        for idx, part in enumerate(snake):
            y, x = part[0], part[1]
            if 0 <= y < self.game_height and 0 <= x < self.game_width:
                if idx == 0:
                    char = "H"
                elif idx == len(snake) - 1:
                    char = "#"
                else:
                    char = "O"
                grid[y][x] = char

        return "\n".join(["".join(row) for row in grid])

    def step_by_step_generate_and_validate(self, prompt, shadow_target_str):
        # We need to manually run the loop from GPT.generate
        # But for each token, check if it matches shadow_target_str

        encode = lambda s: [self.stoi.get(c, self.stoi.get(".", 0)) for c in s]
        decode = lambda l: "".join([self.itos[i] for i in l])

        context_idxs = encode(prompt)
        idx = torch.tensor(
            context_idxs, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Max tokens to generate
        max_new_tokens = 276

        generated_so_far = ""
        divergence_found = False
        divergence_char_idx = -1

        for i in range(max_new_tokens):
            # crop context
            idx_cond = idx[:, -self.d_model.config.block_size :]

            # forward
            logits, _ = self.d_model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append
            idx = torch.cat((idx, idx_next), dim=1)

            # decode just this token
            new_char = self.itos.get(idx_next.item(), "")

            # Check for stop token
            if self.stop_token_id is not None and idx_next.item() == self.stop_token_id:
                break

            generated_so_far += new_char

            # Check ground_truth token against generated token
            if i < len(shadow_target_str):
                ground_truth_char = shadow_target_str[i]
                ground_truth_token = self.stoi.get(
                    ground_truth_char, self.stoi.get(".", 0)
                )
                # FORCE DIVERGENCE FROM THE START
                # Undocument to quickly test divergence
                # ground_truth_token = self.stoi.get("H", 0)
                # ——————
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

        # Post-loop check: Did we finish the shadow string?
        if not divergence_found:
            # We generated some text. Did we cover the whole shadow string?
            # We ignore trailing whitespace in generated so_far for this length check check?
            # Actually, generated_so_far might have \n at end.
            # We just need to ensure we generated AT LEAST enough chars to cover shadow,
            # which is implicit if we didn't break loop and i >= len(shadow) at end.

            # Case: Generated "..." (short) then "$" break.
            # len(generated_so_far) < len(shadow_target_str)
            if len(generated_so_far.strip()) < len(shadow_target_str.strip()):
                divergence_found = True
                divergence_char_idx = len(generated_so_far)
                # This might be out of bounds for highlighting if we don't handle it

        return (
            generated_so_far,
            divergence_found,
            divergence_char_idx,
            ground_truth_char,
            ground_truth_token,
            generated_token,
            new_char,
        )

    def update(self, retry=False):
        # 0. Sync State from Divergence Override (User chose 'Continue')
        if self.sync_shadow_to_neural_pending:
            # We need to disregard divergence and assume the Neural Model is right
            # This is tricky because we STOPPED generation at divergence.
            # So "Continue" essentially means: "Finish generating without validation, then apply."

            # TODO: Resume generation? Or just re-run full generation without validation?
            # Re-running full generation is easier and essentially the same since it's deterministic (greedy decoding)
            # EXCEPT we actually need to finish what we started.
            # Let's just run standart generate() from prompt.

            # Re-create prompt to run full inference
            try:
                # We need to reconstruct the prompt exactly as it was
                board_str = self.get_logical_string()
                action_char = KEY_STR_MAP.get(self.direction, "R")

                if self.prev_board != "" and self.prev_generated != "":
                    prev_board_str = f"{self.prev_board}\n{self.prev_generated}\n$"
                    prompt = f"{prev_board_str}\nB:\n{board_str}\nA:{action_char}\nT:\n"
                else:
                    prompt = f"B:\n{board_str}\nA:{action_char}\nT:\n"

                encode = lambda s: [self.stoi.get(c, self.stoi.get(".", 0)) for c in s]
                decode = lambda l: "".join([self.itos[i] for i in l])

                context_idxs = encode(prompt)
                context = torch.tensor(
                    context_idxs, dtype=torch.long, device=self.device
                ).unsqueeze(0)

                # Standard Generation (No Validation)
                output_ids = self.d_model.generate(
                    context, max_new_tokens=276, stop_token_id=self.stop_token_id
                )
                output_text = decode(output_ids[0].tolist())
                generated = output_text[len(prompt) :]
                if "$" in generated:
                    generated = generated.split("$")[0]
                generated_stripped = generated.strip()

                # Apply State
                if "X" in generated_stripped and len(generated_stripped) < 10:
                    self.game_over = True
                else:
                    self.update_state_from_ascii(generated_stripped)

                # Update Panels
                self.prev_board = f"B:\n{board_str}\nA:{action_char}"
                self.prev_generated = f"T:\n{generated_stripped}"
                self.left_panel = self.prev_board.split("\n")
                self.right_panel = self.prev_generated.split("\n")

                # Clear Divergence State
                self.divergence_detected = False
                self.divergence_index = -1
                self.divergent_generated = ""

            except Exception as e:
                pass

            self.sync_shadow_to_neural_pending = False
            return

        if self.game_over or self.divergence_detected:
            return

        # 1. Capture State Identical to NeuralSnakeGame
        board_str = self.get_logical_string()

        # Consuming Input Queue (Logic copied from NeuralSnakeGame)
        if not retry:
            if self.input_queue:
                self.direction = self.input_queue.popleft()
            if self.direction is None:
                self.direction = curses.KEY_RIGHT

        action_char = KEY_STR_MAP.get(self.direction, "R")
        if not retry:
            self.action_history.append(("EXEC", action_char))

        # -- SHADOW CALCULATION START --
        # Calculate what SHOULD happen based on physics
        shadow_snake, shadow_food, shadow_game_over = self.get_deterministic_next_state(
            self.snake, self.food, self.direction
        )

        if shadow_game_over:
            shadow_target_str = "X"
        else:
            shadow_target_str = self.render_logical_board(shadow_snake, shadow_food)

        self.shadow_panel = f"S:\n{shadow_target_str}".split("\n")

        # -- NEURAL INFERENCE START ``
        if self.prev_board != "" and self.prev_generated != "":
            prev_board_str = f"{self.prev_board}\n{self.prev_generated}\n$"
            prompt = f"{prev_board_str}\nB:\n{board_str}\nA:{action_char}\nT:\n"
        else:
            prompt = f"B:\n{board_str}\nA:{action_char}\nT:\n"

        try:
            # Step-by-Step Generation & Validation
            (
                generated,
                divergence,
                div_idx,
                ground_truth_char,
                ground_truth_token,
                generated_token,
                generated_char,
            ) = self.step_by_step_generate_and_validate(prompt, shadow_target_str)

            # Store generated text for display (even if partial)
            # If divergence, generated contains text up to mismatch + mismatch char
            self.divergent_generated = generated

            if divergence:
                self.ground_truth_token = ground_truth_token
                self.ground_truth_char = ground_truth_char
                self.generated_token = generated_token
                self.generated_char = generated_char
                self.prompt = prompt
                self.divergence_detected = True
                self.divergence_index = div_idx

                # Analyze "Physics Die vs Model Live" scenarios
                if "X" in generated and not shadow_game_over:
                    self.divergence_msg = "Model predicted Die, but Physics says Live"
                elif shadow_game_over and "X" not in generated:
                    self.divergence_msg = "Physics says Die, but Model predicted Live"
                else:
                    self.divergence_msg = "Board State Mismatch"

                # Update panels with PARTIAL generation to show where it stopped
                # BUT DO NOT update self.prev_board/generated yet, to allow retry/continue
                # self.prev_board and self.prev_generated remain as they were before this step

                # Create temporary display strings
                temp_prev_board = f"B:\n{board_str}\nA:{action_char}"
                temp_prev_generated = f"T:\n{generated}"

                self.left_panel = temp_prev_board.split("\n")
                self.right_panel = temp_prev_generated.split("\n")

                return

            # If no divergence, we finished generation successfully
            generated_stripped = generated.strip()
            if "X" in generated_stripped and len(generated_stripped) < 10:
                self.game_over = True
            else:
                self.update_state_from_ascii(generated_stripped)

            # SUCCESS: Update History
            self.prev_board = f"B:\n{board_str}\nA:{action_char}"
            self.prev_generated = f"T:\n{generated_stripped}"
            self.left_panel = self.prev_board.split("\n")
            self.right_panel = self.prev_generated.split("\n")

            # Log
            if self.record_file:
                entry = f"{self.prev_board}\n{self.prev_generated}\n$"
                with open(self.record_file, "a") as f:
                    f.write(entry + "\n")

        except Exception as e:
            pass

    def render(self):
        # Override to draw the 3rd panel
        self.stdscr.erase()
        sh, sw = self.stdscr.getmaxyx()

        offset_y = max(0, (sh - (self.game_height + 2)) // 2)
        offset_x = max(0, (sw - (self.game_width + 2) * 2) // 2)

        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)

        content_y = offset_y + 1
        content_x = offset_x + 2

        for part in self.snake:
            try:
                self.stdscr.addstr(
                    content_y + part[0],
                    content_x + part[1] * 2,
                    "██",
                    curses.color_pair(1),
                )  # COLOR_SNAKE
            except curses.error:
                pass

        if self.food:
            try:
                self.stdscr.addstr(
                    content_y + self.food[0],
                    content_x + self.food[1] * 2,
                    "██",
                    curses.color_pair(2),
                )  # COLOR_FOOD
            except curses.error:
                pass

        score_text = f" Score: {self.score} "
        if self.divergence_detected:
            score_text += " [DIVERGENCE DETECTED!] "

        try:
            self.stdscr.addstr(
                offset_y + self.game_height + 1,
                offset_x + 2,
                score_text,
                curses.color_pair(4) | curses.A_BOLD,
            )
        except curses.error:
            pass

        if self.game_over:
            msg = " GAME OVER "
            if self.divergence_detected:
                # We handle the specific message drawing in the divergence loop or main loop
                # checking logic, but for static render:
                pass

            y = offset_y + self.game_height // 2
            x = offset_x + (self.game_width * 2 - len(msg)) // 2
            try:
                if not self.divergence_detected:
                    self.stdscr.addstr(
                        y,
                        x,
                        msg,
                        curses.color_pair(4) | curses.A_REVERSE | curses.A_BOLD,
                    )
            except curses.error:
                pass

        if self.divergence_detected:
            self.draw_divergence_menu(offset_y, offset_x)

        # Left Panel (Board/Action)
        lx = max(0, offset_x - 20)
        ly = offset_y
        try:
            for i, line in enumerate(self.left_panel):
                if ly + i < sh:
                    self.stdscr.addstr(ly + i, lx, line[:18], curses.color_pair(4))
        except curses.error:
            pass

        # Right Panel (Target Neural)
        rx = offset_x + (self.game_width * 2) + 6
        ry = offset_y
        try:
            self.stdscr.addstr(
                ry - 1, rx, "Neural (T):", curses.color_pair(4) | curses.A_BOLD
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

        # Shadow Panel (Target Shadow)
        sx = rx + 20
        sy = offset_y
        try:
            self.stdscr.addstr(
                sy - 1, sx, "Shadow (S):", curses.color_pair(4) | curses.A_BOLD
            )
            for i, line in enumerate(self.shadow_panel):
                if sy + i < sh:
                    # Draw char by char to handle highlighting matching the neural one
                    for j, char in enumerate(line):
                        # Reconstruct index in the FULL string "S:\n....."
                        raw_idx = sum(len(l) + 1 for l in self.shadow_panel[:i]) + j

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
            shadow_snake, shadow_food, shadow_game_over = (
                self.get_deterministic_next_state(self.snake, self.food, self.direction)
            )
            if shadow_game_over:
                current_shadow_str = "X"
            else:
                current_shadow_str = self.render_logical_board(
                    shadow_snake, shadow_food
                )

            # Prefix is the shadow string UP TO the divergence index
            prefix_str = current_shadow_str[: self.divergence_index]

            # Correct context: Prompt + Prefix
            full_context_str = self.prompt + prefix_str

            encode = lambda s: [self.stoi.get(c, self.stoi.get(".", 0)) for c in s]
            context_idxs = encode(full_context_str)

            # Fix: Truncate to block_size (1024) to avoid RuntimeErrors
            if len(context_idxs) > self.d_model.config.block_size:
                context_idxs = context_idxs[-self.d_model.config.block_size :]

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
            loss, probs = self.d_model.train_step(
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

    def sync_shadow_state(self):
        # Force shadow state to match current neural generation
        # We need to parse self.prev_generated (which is T) and set self.snake, self.food
        # But wait, self.update_state_from_ascii(generated) ALREADY updated the game state (self.snake, self.food)
        # to match the neural output (unless it was 'X').

        # If divergence was "Board State Mismatch", 'generated' was NOT X.
        # But in update(), if divergence is detected:
        # if generated != shadow_target_str: divergence_detected = True ... return
        # So we returned BEFORE calling update_state_from_ascii(generated) if we wanted to be strict?
        # Let's check my update() implementation.
        pass
        # Taking a look at update() in next step to correct flow.

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
        prompt = "Filename (in model/weigths/): "
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
        save_path = os.path.join(base_dir, "model", "weigths", fname)

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
            torch.save(self.d_model.state_dict(), save_path)
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
        self.shadow_panel = []
        self.divergence_detected = False
        self.divergence_msg = ""
        self.divergence_index = -1
        self.divergent_generated = ""
        self.ground_truth_token = 0
        self.ground_truth_char = ""
        self.prev_board = ""
        self.prev_generated = ""
        self.sync_shadow_to_neural_pending = False

        # We do NOT reset self.model_updated, so user can still save later if they want

    def run(self):
        while True:
            # Check Divergence Input separate from Game Input
            if self.divergence_detected:
                try:
                    key = self.stdscr.getch()
                except:
                    key = -1

                if key == ord("q") or key == ord("Q"):
                    break
                elif key == ord("c") or key == ord("C"):
                    # Force synchronize shadow to neural
                    self.sync_shadow_to_neural_pending = True
                elif key == ord("t") or key == ord("T"):
                    # Calculate loss and backpropagate until converged
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
                    break

                if key == -1:
                    pass  # No input, just continue to update (which handles default/queue)

            self.update()

            if self.game_over:
                self.render()  # Draw one last time to show GAME OVER state
            else:
                self.render()

        # On Exit Loop
        self.prompt_save_model()


def main(stdscr, model_filename="snake_model"):
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not model_filename.endswith(".pt"):
        model_filename += ".pt"

    model_path = os.path.join(base_dir, "model", "weigths", model_filename)
    meta_path = os.path.join(base_dir, "model", "weigths", "meta.pkl")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

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

    stdscr.addstr(10, 10, f"Loading Neural Model on {device}...")
    stdscr.addstr(11, 10, f"Model: {model_filename}")
    stdscr.refresh()

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

    game = ShadowNeuralSnakeGame(stdscr, model, meta, device)
    game.run()


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("SHADOW NEURAL SNAKE - Model Selection")
    print("--------------------------------------------------")
    print("Enter the name of the model to load from model/weights/")
    print("(Press Enter for default: 'snake_model')")

    try:
        model_name = input("Model Name: ").strip()
    except KeyboardInterrupt:
        sys.exit(0)

    if not model_name:
        model_name = "snake_model"

    curses.wrapper(main, model_name)
