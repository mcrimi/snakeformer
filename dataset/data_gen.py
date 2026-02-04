import sys
import os
import curses
import random
import argparse
import collections
import time
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.shared import prompt_model_selection, load_model
from games.utils import MockStdScr, HeadlessSnakeGame, get_heatmap_grid

# Constants
CMD_UP = curses.KEY_UP
CMD_DOWN = curses.KEY_DOWN
CMD_LEFT = curses.KEY_LEFT
CMD_RIGHT = curses.KEY_RIGHT

KEY_STR_MAP = {CMD_UP: "U", CMD_DOWN: "D", CMD_LEFT: "L", CMD_RIGHT: "R"}


# MockStdScr and HeadlessSnakeGame are imported from games.utils


class HeadlessNeuralSnake(HeadlessSnakeGame):
    def __init__(self, stdscr, model, meta, device):
        super().__init__(stdscr)
        self.d_model = model
        self.d_meta = meta
        self.device = device
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.stop_token_id = self.stoi.get("$")

    def predict(self, current_board_str, current_action):
        action_char = KEY_STR_MAP.get(current_action, "R")
        prompt = f"B:\n{current_board_str}\nA:{action_char}\nT:\n"

        def encode(s):
            return [self.stoi.get(c, self.stoi.get(".", 0)) for c in s]

        def decode(ids_list):
            return "".join([self.itos[i] for i in ids_list])

        context = torch.tensor(
            encode(prompt), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        try:
            output_ids = self.d_model.generate(
                context, max_new_tokens=400, stop_token_id=self.stop_token_id
            )
            output_text = decode(output_ids[0].tolist())
            generated = output_text[len(prompt) :].split("$")[0]
            return generated.strip()
        except Exception:
            return "X"


class Bot:
    def __init__(self, game):
        self.game = game

    def get_path_to_food(self):
        if not self.game.food:
            return []
        start, target = tuple(self.game.snake[0]), tuple(self.game.food)
        queue, visited = collections.deque([(start, [])]), {start}
        obstacles = set(tuple(p) for p in self.game.snake)
        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path
            for move, (dy, dx) in [
                (CMD_UP, (-1, 0)),
                (CMD_DOWN, (1, 0)),
                (CMD_LEFT, (0, -1)),
                (CMD_RIGHT, (0, 1)),
            ]:
                ny, nx = curr[0] + dy, curr[1] + dx
                if (
                    0 <= ny < self.game.game_height
                    and 0 <= nx < self.game.game_width
                    and (ny, nx) not in obstacles
                    and (ny, nx) not in visited
                ):
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [move]))
        return None

    def get_safe_random(self):
        head = self.game.snake[0]
        moves = [
            m
            for m, (dy, dx) in [
                (CMD_UP, (-1, 0)),
                (CMD_DOWN, (1, 0)),
                (CMD_LEFT, (0, -1)),
                (CMD_RIGHT, (0, 1)),
            ]
            if 0 <= head[0] + dy < self.game.game_height
            and 0 <= head[1] + dx < self.game.game_width
            and (head[0] + dy, head[1] + dx)
            not in set(tuple(p) for p in self.game.snake)
        ]
        return random.choice(moves) if moves else self.game.direction


# -----------------------------------------------------------------------------
# UI Functions
# -----------------------------------------------------------------------------


def draw_dashboard(stdscr, game, title, counts, targets, start_time):
    stdscr.erase()  # Use erase to clear background artifacts
    sh, sw = stdscr.getmaxyx()

    # 1. Game View (Left)
    # Reusing SnakeGame drawing methods
    # Game size is 16x16 chars (logical) -> 32x16 visual
    # Box is (16+2)*2 wide = 36 chars.
    game_y, game_x = 2, 2

    game.draw_box(game_y, game_x, game.game_height, game.game_width)
    game.draw_snake(game_y, game_x)
    game.draw_food(game_y, game_x)

    # 2. Stats View (Right)
    stats_x = 42  # Right of the game board

    stdscr.addstr(1, stats_x, title, curses.color_pair(1) | curses.A_BOLD)

    header = f"{'Category':<15} | {'Samples':<6} | {'Target':<6}"
    stdscr.addstr(
        3,
        stats_x,
        header[: sw - stats_x - 1],
        curses.color_pair(3) | curses.A_UNDERLINE,
    )

    row_y = 5
    for cat, target in targets.items():
        if row_y >= sh - 4:
            break
        count = counts.get(cat, 0)
        # Mini bar - unused now
        # bar_len = 10
        # filled = int(bar_len * pct)
        # bar = "[" + "=" * filled + " " * (bar_len - filled) + "]"

        # line = f"{cat:<15} {bar} {count}/{target}"
        line = f"{cat:<15} | {count:<6} | {target:<6}"

        # Highlight active strategy if we could track it, for now just list
        color = curses.color_pair(3)
        if count >= target:
            color = curses.color_pair(1)  # Green if done

        stdscr.addstr(row_y, stats_x, line[: sw - stats_x - 1], color)
        row_y += 1

    # Draw standard game score/quiet UI
    game.draw_score(game_y, game_x)

    # Footer (Progress only, Quit is handled by draw_score)
    elapsed = time.time() - start_time
    total_done = sum(counts.values())
    total_target = sum(targets.values())
    speed = total_done / elapsed if elapsed > 0 else 0

    footer = f"Total: {total_done}/{total_target} | Time: {elapsed:.0f}s | Speed: {speed:.1f} it/s"
    stdscr.addstr(sh - 2, max(0, (sw - len(footer)) // 2), footer, curses.color_pair(5))

    stdscr.refresh()


def draw_summary(stdscr, title, data_lines):
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()

    stdscr.addstr(
        2, max(0, (sw - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD
    )

    y = 5
    for line in data_lines:
        stdscr.addstr(y, max(0, (sw - len(line)) // 2), line, curses.color_pair(3))
        y += 1

    msg = "Press any key to return to menu..."
    stdscr.addstr(
        sh - 2, max(0, (sw - len(msg)) // 2), msg, curses.color_pair(5) | curses.A_BLINK
    )
    stdscr.refresh()
    stdscr.getch()


def get_input_str(stdscr, prompt):
    curses.echo()
    curses.curs_set(1)
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()
    stdscr.addstr(
        sh // 2 - 2, max(0, (sw - len(prompt)) // 2), prompt, curses.color_pair(3)
    )

    win = curses.newwin(1, 40, sh // 2, max(0, (sw - 40) // 2))
    win.bkgd(" ", curses.color_pair(4))
    stdscr.refresh()
    win.refresh()

    inp = win.getstr(0, 0).decode("utf-8")
    curses.noecho()
    curses.curs_set(0)
    return inp


# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------


def run_curriculum_gen(stdscr, target_size=500000, test_mode=False):
    output_file = os.path.join(os.path.dirname(__file__), "snake_data_curriculum.txt")
    target_counts = {
        "Standard": int(target_size * 0.30),
        "Eat": int(target_size * 0.25),
        "Long": int(target_size * 0.15),
        "Hit Wall": int(target_size * 0.10),
        "Self-collision": int(target_size * 0.10),
        "Illegal": int(target_size * 0.05),
        "Non-optimal": int(target_size * 0.10),
    }

    if test_mode:
        target_counts = {k: max(10, int(v / 1000)) for k, v in target_counts.items()}

    current_counts = {k: 0 for k in target_counts}
    visited_counts = collections.defaultdict(int)
    seen_transitions = set()

    # Use real stdscr if interactive, else mock
    game = HeadlessSnakeGame(stdscr if stdscr else MockStdScr())
    bot = Bot(game)

    start_time = time.time()
    step_limit = 20000 if test_mode else 5000000
    steps_taken = 0

    # Initial Draw
    if stdscr:
        draw_dashboard(
            stdscr,
            game,
            "Generating Curriculum...",
            current_counts,
            target_counts,
            start_time,
        )
        stdscr.nodelay(1)

    def should_record(cat, base_prob=1.0):
        # Dampen probability as we get closer to target
        # P = base_prob * (1 - current / target)
        if current_counts[cat] >= target_counts[cat]:
            return False

        progress = current_counts[cat] / target_counts[cat]
        prob = base_prob * (1.0 - progress)
        return random.random() < prob

    def save_transition(f, b_str, a_str, t_str, cat):
        # 1. Deduplicate
        key = (b_str.strip(), a_str.strip())
        if key in seen_transitions:
            return False

        # 2. Save
        f.write(f"B:\n{b_str}\nA:{a_str}\nT:\n{t_str}\n$\n")
        seen_transitions.add(key)
        current_counts[cat] += 1
        return True

    with open(output_file, "w") as f:
        while sum(current_counts.values()) < sum(target_counts.values()):
            steps_taken += 1
            if steps_taken > step_limit:
                break

            if game.game_over:
                # Capture natural death (e.g. from random noise or mistakes)
                # Since game state didn't update the snake on death,
                # we use current snake position + last direction to determine fatal move
                head = game.snake[0]
                d_map = {
                    CMD_UP: (-1, 0),
                    CMD_DOWN: (1, 0),
                    CMD_LEFT: (0, -1),
                    CMD_RIGHT: (0, 1),
                }
                dy, dx = d_map.get(game.direction, (0, 0))
                ny, nx = head[0] + dy, head[1] + dx

                is_wall = not (0 <= ny < game.game_height and 0 <= nx < game.game_width)

                # Check body coll (exclude tail? standard snake rules say tail moves... but on death we assume we hit it)
                # If we grew this turn? we don't know easily. Assume standard collision logic.
                obstacles = set(
                    tuple(p) for p in game.snake[:-1]
                )  # exclude tail usually
                is_body = (ny, nx) in obstacles

                cat = None
                if is_wall:
                    cat = "Hit Wall"
                elif is_body:
                    cat = "Self-collision"

                if cat and should_record(cat, 1.0):
                    # Reconstruct last state (approximate since we don't have prev_food easily without tracking)
                    # But actually, SnakeGame state IS the pre-crash state (except direction is updated).
                    # So we can just render.
                    s_curr = game.render_board_state(game.snake, game.food)
                    act_str = KEY_STR_MAP.get(game.direction, "R")
                    save_transition(f, s_curr, act_str, "X", cat)

                game.reset_game()

            head, food = game.snake[0], game.food

            # Eat (Glutton)
            if (
                should_record("Eat", 1.0)
                and abs(head[0] - food[0]) + abs(head[1] - food[1]) == 1
            ):
                dy, dx = food[0] - head[0], food[1] - head[1]
                act = {
                    (-1, 0): CMD_UP,
                    (1, 0): CMD_DOWN,
                    (0, -1): CMD_LEFT,
                    (0, 1): CMD_RIGHT,
                }.get((dy, dx))
                if act:
                    s1 = game.render_board_state(game.snake, game.food)
                    c = game.clone()
                    c.step(act)
                    s2 = (
                        c.render_board_state(c.snake, c.food)
                        if not c.game_over
                        else "X"
                    )
                    save_transition(f, s1, KEY_STR_MAP[act], s2, "Eat")

            # Long (Fat)
            if len(game.snake) >= 15 and should_record("Long", 1.0):
                p = bot.get_path_to_food()
                act = p[0] if p else bot.get_safe_random()
                s1 = game.render_board_state(game.snake, game.food)
                c = game.clone()
                c.step(act)
                if not c.game_over:
                    s2 = c.render_board_state(c.snake, c.food)
                    save_transition(f, s1, KEY_STR_MAP[act], s2, "Long")

            # Non-optimal (Drunk)
            if should_record("Non-optimal", 0.15):
                act = bot.get_safe_random()
                c = game.clone()
                c.step(act)
                if not c.game_over:
                    save_transition(
                        f,
                        game.render_board_state(game.snake, game.food),
                        KEY_STR_MAP[act],
                        c.render_board_state(c.snake, c.food),
                        "Non-optimal",
                    )

            # Standard
            if should_record("Standard", 0.1):
                p = bot.get_path_to_food()
                act = p[0] if p else bot.get_safe_random()
                c = game.clone()
                c.step(act)
                if not c.game_over:
                    save_transition(
                        f,
                        game.render_board_state(game.snake, game.food),
                        KEY_STR_MAP[act],
                        c.render_board_state(c.snake, c.food),
                        "Standard",
                    )

                # --- Forced Events for Rare Categories ---

                # Illegal (180 Turn)
                if should_record("Illegal", 1.0):
                    opp_dir = {
                        CMD_UP: CMD_DOWN,
                        CMD_DOWN: CMD_UP,
                        CMD_LEFT: CMD_RIGHT,
                        CMD_RIGHT: CMD_LEFT,
                    }.get(game.direction)
                    if opp_dir:
                        # Physics: Input ignored, snake continues in current direction
                        c = game.clone()
                        c.step(game.direction)
                        next_s = (
                            "X"
                            if c.game_over
                            else c.render_board_state(c.snake, c.food)
                        )

                        save_transition(
                            f,
                            game.render_board_state(game.snake, game.food),
                            KEY_STR_MAP[opp_dir],
                            next_s,
                            "Illegal",
                        )

            forced_act = None

            # Check if we need forced death
            if should_record("Hit Wall", 0.2) or should_record("Self-collision", 0.2):
                # Look for a move that KILLS us in the desired way
                for move, (dy, dx) in [
                    (CMD_UP, (-1, 0)),
                    (CMD_DOWN, (1, 0)),
                    (CMD_LEFT, (0, -1)),
                    (CMD_RIGHT, (0, 1)),
                ]:
                    # Skip 180 (ignored)
                    if {
                        CMD_UP: CMD_DOWN,
                        CMD_DOWN: CMD_UP,
                        CMD_LEFT: CMD_RIGHT,
                        CMD_RIGHT: CMD_LEFT,
                    }.get(game.direction) == move:
                        continue

                    ny, nx = head[0] + dy, head[1] + dx
                    is_wall = not (
                        0 <= ny < game.game_height and 0 <= nx < game.game_width
                    )
                    obstacles = set(tuple(p) for p in game.snake[:-1])
                    is_body = (ny, nx) in obstacles

                    if is_wall and should_record("Hit Wall", 1.0):
                        if random.random() < 0.1:
                            forced_act = move
                            break  # Found a way to die by wall
                    elif is_body and should_record("Self-collision", 1.0):
                        if random.random() < 0.1:
                            forced_act = move
                            break  # Found a way to die by body

            if forced_act:
                act = forced_act
            else:
                # Standard Survival / Eating Logic
                p = bot.get_path_to_food()
                act = p[0] if p and random.random() > 0.1 else bot.get_safe_random()

            game.step(act)

            # Track Visited
            for p in game.snake:
                visited_counts[tuple(p)] += 1

            # Update UI
            if (
                stdscr and steps_taken % 2 == 0
            ):  # Update every 2 steps for speed/viz balance
                draw_dashboard(
                    stdscr,
                    game,
                    "Generating Curriculum...",
                    current_counts,
                    target_counts,
                    start_time,
                )
                # Check for quit
                key = stdscr.getch()
                if key == ord("q"):
                    break

    if stdscr:
        stdscr.nodelay(0)  # Blocking input
        summary_lines = (
            [f"{'Category':<20} | {'Samples':<10}", "-" * 33]
            + [f"{k:<20} | {v:<10}" for k, v in current_counts.items()]
            + [
                "-" * 33,
                f"{'Total':<20} | {sum(current_counts.values()):<10}",
                f"File: {os.path.basename(output_file)}",
            ]
        )
        draw_summary(stdscr, "Curriculum Generation Complete", summary_lines)
        stdscr.nodelay(0)
        draw_heatmap(stdscr, visited_counts)


def run_dagger_gen(
    stdscr,
    target_corrections=2000,
    model=None,
    meta=None,
    device="cpu",
    test_mode=False,
):
    output_file = os.path.join(
        os.path.dirname(__file__), "snake_curriculum_dagger_fixes.txt"
    )
    if test_mode:
        target_corrections = 5

    # Use real stdscr if provided
    real_game = HeadlessSnakeGame(stdscr if stdscr else MockStdScr())
    bot = Bot(real_game)
    neural = HeadlessNeuralSnake(MockStdScr(), model, meta, device)

    corrections = 0
    hallucinations = 0
    games_played = 0
    start_time = time.time()

    counts = {"Corrections": 0, "Games": 0, "Errors": 0}
    targets = {"Corrections": target_corrections, "Games": 0, "Errors": 0}

    if stdscr:
        draw_dashboard(
            stdscr, real_game, "DAgger (Running)...", counts, targets, start_time
        )

    with open(output_file, "a") as f:
        while corrections < target_corrections:
            real_game.reset_game()
            step_count = 0
            while not real_game.game_over and step_count < 1000:
                if corrections >= target_corrections:
                    break

                step_count += 1
                p = bot.get_path_to_food()
                act = p[0] if p and random.random() > 0.1 else bot.get_safe_random()

                curr_s = real_game.render_board_state(real_game.snake, real_game.food)
                pred_s = neural.predict(curr_s, act)

                real_game.step(act)
                true_s = (
                    "X"
                    if real_game.game_over
                    else real_game.render_board_state(real_game.snake, real_game.food)
                )

                if pred_s != true_s:
                    hallucinations += 1
                    f.write(f"B:\n{curr_s}\nA:{KEY_STR_MAP[act]}\nT:\n{true_s}\n$\n")
                    f.flush()
                    corrections += 1
                    counts["Corrections"] = corrections
                    counts["Errors"] = hallucinations

                if stdscr and step_count % 2 == 0:
                    draw_dashboard(
                        stdscr,
                        real_game,
                        "DAgger (Running)...",
                        counts,
                        targets,
                        start_time,
                    )
                    if stdscr.getch() == ord("q"):
                        return

            games_played += 1
            counts["Games"] = games_played

    if stdscr:
        lines = [
            f"Total Games:      {games_played}",
            f"Hallucinations:   {hallucinations}",
            f"Corrections:      {corrections}",
            f"Output File:      {os.path.basename(output_file)}",
        ]
        draw_summary(stdscr, "DAgger Generation Complete", lines)


def draw_heatmap(stdscr, visited_counts):
    # visited_counts: dict (y,x) -> count
    stdscr.clear()
    sh, sw = stdscr.getmaxyx()

    max_count = max(visited_counts.values()) if visited_counts else 1

    # Heatmap logic moved to games.utils
    grid_lines, max_count = get_heatmap_grid(visited_counts)

    stdscr.addstr(
        1, 2, "Board Coverage Heatmap (Dark=Low, Bright=High)", curses.color_pair(1)
    )

    start_y = 3
    start_x = 2

    for r, line_str in enumerate(grid_lines):
        for c, char in enumerate(line_str):
            if start_y + r < sh - 1 and start_x + c * 2 < sw - 1:
                stdscr.addstr(start_y + r, start_x + c * 2, char)

    msg = "Press any key to exit..."
    stdscr.addstr(sh - 2, 2, msg, curses.color_pair(5))
    stdscr.refresh()
    stdscr.getch()


def run_manual_gen(stdscr, target_size=500000):
    output_file = os.path.join(os.path.dirname(__file__), "snake_data_curriculum.txt")

    # Just for tracking, not limiting (user plays as much as they want)
    target_counts = {
        "Standard": int(target_size * 0.30),
        "Eat": int(target_size * 0.25),
        "Long": int(target_size * 0.15),
        "Hit Wall": int(target_size * 0.10),
        "Self-collision": int(target_size * 0.10),
        "Illegal": int(target_size * 0.05),
        "Non-optimal": int(target_size * 0.10),
    }

    current_counts = {k: 0 for k in target_counts}
    visited_counts = collections.defaultdict(int)

    game = HeadlessSnakeGame(stdscr if stdscr else MockStdScr())
    bot = Bot(game)

    # Enable manual input with timeout for stability
    game.stdscr.timeout(100)

    start_time = time.time()

    # Pause start
    if stdscr:
        # Base draw handled by dashboard
        draw_dashboard(
            stdscr,
            game,
            "Manual Play - Press Key to Start",
            current_counts,
            target_counts,
            start_time,
        )
        stdscr.nodelay(0)  # Blocking
        stdscr.getch()
        stdscr.nodelay(1)  # Restore non-blocking but handled by timeout
        game.stdscr.timeout(100)

    with open(output_file, "a") as f:  # Append mode for manual
        while True:
            # 1. Capture user key separately for Illegal check logic
            # handle_input consumes the key from curses buffer via getch
            # We must rely on game.handle_input()
            # If we want to detect Illegal (180), we must check if input was valid vs direction.

            # Peek or just use the updated direction?
            # Standard SnakeGame: handle_input() sets direction directly if valid.
            # If invalid (180), it ignores it.
            # We can't easily distinguish "ignored 180" from "no input" just by looking at direction change if we don't know the input.
            # BUT we can modify handle_input logic or just read getch ourselves?
            # HeadlessSnakeGame inherits SnakeGame. SnakeGame.handle_input() calls self.stdscr.getch().
            # Let's override handle_input in a local wrapper or just use getch here.

            key = game.stdscr.getch()
            user_action = None
            if key != -1:
                if key in [ord("q"), ord("Q")]:
                    break
                if key in [CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT]:
                    user_action = key

            # Logic for Illegal (180)
            is_illegal_180 = False
            if user_action:
                opp_dir = {
                    CMD_UP: CMD_DOWN,
                    CMD_DOWN: CMD_UP,
                    CMD_LEFT: CMD_RIGHT,
                    CMD_RIGHT: CMD_LEFT,
                }.get(game.direction)
                if user_action == opp_dir:
                    is_illegal_180 = True

            # Step logic
            # If illegal, we DO NOT step the game with that action (game ignores it anyway),
            # BUT we record it as a dataset event: State + IllegalAction -> X

            valid_action_for_game = (
                user_action if user_action and not is_illegal_180 else None
            )
            if valid_action_for_game:
                game.direction = valid_action_for_game  # Force it if valid standard

            # Record current state *before* update
            curr_s = game.render_board_state(game.snake, game.food)
            curr_act_char = KEY_STR_MAP.get(game.direction, "R")
            if user_action and is_illegal_180:
                curr_act_char = KEY_STR_MAP.get(user_action)

            prev_snake = list(game.snake)

            # Standard Step
            game.update()

            # Track Visited
            for p in game.snake:
                visited_counts[tuple(p)] += 1

            # Record Logic
            if is_illegal_180:
                # Prioritize documenting the illegal move attempt
                # Physics: Input ignored, snake continued in current direction (already updated via game.update())
                # So T is simply current state (or X if momentum killed us)
                next_s = (
                    "X"
                    if game.game_over
                    else game.render_board_state(game.snake, game.food)
                )
                f.write(f"B:\n{curr_s}\nA:{curr_act_char}\nT:\n{next_s}\n$\n")
                if "Illegal" in current_counts:
                    current_counts["Illegal"] += 1

                # We ALSO want to record standard flow? No, one record per frame/step usually suitable.
                # If we record illegal, we might skip standard record for this frame to avoid dupes/confusion?
                # User intended 180, so let's teach 180 is death.

            elif game.game_over:
                d_map = {
                    CMD_UP: (-1, 0),
                    CMD_DOWN: (1, 0),
                    CMD_LEFT: (0, -1),
                    CMD_RIGHT: (0, 1),
                }
                dy, dx = d_map.get(game.direction, (0, 0))
                ny, nx = prev_snake[0][0] + dy, prev_snake[0][1] + dx
                is_wall = not (0 <= ny < game.game_height and 0 <= nx < game.game_width)

                cat = "Standard"
                if is_wall:
                    cat = "Hit Wall"
                else:
                    cat = "Self-collision"

                f.write(f"B:\n{curr_s}\nA:{curr_act_char}\nT:\nX\n$\n")
                if cat in current_counts:
                    current_counts[cat] += 1
                game.reset_game()

            else:
                # Normal Move
                target_s = game.render_board_state(game.snake, game.food)
                cat = "Standard"

                # 1. Glutton
                if len(game.snake) > len(prev_snake):
                    cat = "Glutton"
                # 2. Long (Fat)
                elif len(game.snake) >= 15:
                    cat = "Long"
                # 3. Non-optimal (Drunk)
                else:
                    # Check optimality if user provided input
                    if user_action:  # Only judge if user actively steered
                        p = bot.get_path_to_food()
                        optimal_first_move = p[0] if p else None
                        if optimal_first_move and user_action != optimal_first_move:
                            cat = "Non-optimal"

                f.write(f"B:\n{curr_s}\nA:{curr_act_char}\nT:\n{target_s}\n$\n")
                if cat in current_counts:
                    current_counts[cat] += 1

            if stdscr:
                draw_dashboard(
                    stdscr,
                    game,
                    "Manual Play (Recording)...",
                    current_counts,
                    target_counts,
                    start_time,
                )

    if stdscr:
        stdscr.nodelay(0)  # Blocking input for summary/heatmap
        summary_lines = (
            [f"{'Category':<20} | {'Count':<10}", "-" * 33]
            + [f"{k:<20} | {v:<10}" for k, v in current_counts.items()]
            + [
                "-" * 33,
                f"{'Total':<20} | {sum(current_counts.values()):<10}",
                f"File: {os.path.basename(output_file)}",
            ]
        )
        draw_summary(stdscr, "Manual Generation Complete", summary_lines)
        draw_heatmap(stdscr, visited_counts)
    curses.start_color()
    curses.use_default_colors()
    for i, c in enumerate([curses.COLOR_GREEN, curses.COLOR_RED, curses.COLOR_WHITE]):
        curses.init_pair(i + 1, c, -1)
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Input bg
    curses.init_pair(5, curses.COLOR_CYAN, -1)  # Footer

    menu_opts = [
        "Curriculum Generation",
        "DAgger Corrections",
        "Manual Play (Record Dataset)",
        "Quit",
    ]
    sel = 0

    while True:
        stdscr.clear()
        sh, sw = stdscr.getmaxyx()

        # ASCII Title
        title = [
            r"   Data Generator   ",
            r"  SnakeFormer Tools ",
        ]
        for i, line in enumerate(title):
            stdscr.addstr(
                sh // 2 - 8 + i,
                max(0, (sw - len(line)) // 2),
                line,
                curses.color_pair(1) | curses.A_BOLD,
            )

        for idx, opt in enumerate(menu_opts):
            x = max(0, (sw - len(opt)) // 2)
            y = sh // 2 - 4 + idx * 2
            if idx == sel:
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(y, x - 2, f"  {opt}  ")
                stdscr.attroff(curses.color_pair(4))
            else:
                stdscr.addstr(y, x, opt, curses.color_pair(3))

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP:
            sel = (sel - 1) % len(menu_opts)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(menu_opts)
        elif key in [10, 13]:
            if sel == 0:  # Curriculum
                inp = get_input_str(stdscr, "Enter Target Dataset Size (default 500k):")
                try:
                    target = int(inp) if inp.strip() else 500000
                except ValueError:
                    target = 500000
                run_curriculum_gen(stdscr, target_size=target)

            elif sel == 1:  # DAgger
                path, meta_path = prompt_model_selection(
                    stdscr,
                    os.path.join(os.path.dirname(__file__), "..", "model", "weights"),
                )
                if path:
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        loading_msg = f"Loading {os.path.basename(path)}..."
                        stdscr.clear()
                        stdscr.addstr(
                            sh // 2, (sw - len(loading_msg)) // 2, loading_msg
                        )
                        stdscr.refresh()
                        model, meta = load_model(path, meta_path, dev)

                        inp = get_input_str(
                            stdscr, "Enter Max Corrections (default 2k):"
                        )
                        try:
                            target = int(inp) if inp.strip() else 2000
                        except ValueError:
                            target = 2000

                        run_dagger_gen(
                            stdscr,
                            target_corrections=target,
                            model=model,
                            meta=meta,
                            device=dev,
                        )
                    except Exception as e:
                        draw_summary(stdscr, "Error Loading Model", [str(e)[: sw - 4]])

            elif sel == 2:  # Manual
                inp = get_input_str(
                    stdscr, "Enter Target Size (Reference) (default 500):"
                )
                try:
                    target = int(inp) if inp.strip() else 500
                except ValueError:
                    target = 500
                run_manual_gen(stdscr, target_size=target)

            elif sel == 3:
                break


def main_tui(stdscr):
    # Colors - Match games/snake.py exactly
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # SNAKE
    curses.init_pair(2, curses.COLOR_RED, -1)  # FOOD
    curses.init_pair(3, curses.COLOR_WHITE, -1)  # BORDER
    curses.init_pair(4, curses.COLOR_WHITE, -1)  # TEXT
    curses.init_pair(5, curses.COLOR_CYAN, -1)  # ACTION_USER / Footer
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # ACTION_AUTO
    curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)  # QUIT
    curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_WHITE)  # MENU_HIGHLIGHT

    menu_opts = [
        "Generate Pre-Training Dataset - Autoplay 🤖",
        "Generate Pre-Training Dataset - Play Manualy 👤",
        "Generate Fine-tuning Dataset  🧠",
        "Quit",
    ]
    subtitles = [
        "Heuristic bot plays the deterministic game, we record the board states and actions to generate curriculum dataset.",
        "You play the deterministic game, we record the board states and actions to generate curriculum dataset.",
        "Heuristic bot plays on neural snake, we record the board states and transitions where the model gets it wrong in a dataset to fine tune the model a la DAgger",
        "Leave this nonsense.",
    ]
    sel = 0

    while True:
        stdscr.clear()
        sh, sw = stdscr.getmaxyx()

        ascii_header = [
            r"   ___       __      ",
            r"  / _ \___ _/ /____ _",
            r" / // / _ `/ __/ _ `/",
            r"/____/\_,_/\__/\_,_/ ",
        ]

        start_y = max(1, sh // 2 - 10)
        for i, line in enumerate(ascii_header):
            stdscr.addstr(
                start_y + i,
                max(0, (sw - len(line)) // 2),
                line,
                curses.color_pair(1) | curses.A_BOLD,
            )

        # Draw Menu
        menu_y_start = start_y + len(ascii_header) + 2

        for idx, opt in enumerate(menu_opts):
            x = max(0, (sw - len(opt)) // 2)
            y = menu_y_start + idx * 2

            if y >= sh - 1:
                break  # Prevent crash

            if idx == sel:
                stdscr.attron(curses.color_pair(8))
                stdscr.addstr(y, x - 2, f"  {opt}  ")
                stdscr.attroff(curses.color_pair(8))

                # Subtitle
                if idx < len(subtitles):
                    sub = subtitles[idx]
                    sy = menu_y_start + len(menu_opts) * 2 + 1
                    if sy < sh - 1:
                        stdscr.addstr(
                            sy, max(0, (sw - len(sub)) // 2), sub, curses.color_pair(5)
                        )
            else:
                stdscr.addstr(y, x, opt, curses.color_pair(3))

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP:
            sel = (sel - 1) % len(menu_opts)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(menu_opts)
        elif key in [10, 13]:
            if sel == 0:  # Curriculum
                inp = get_input_str(stdscr, "Enter Target Dataset Size (default 500k):")
                try:
                    target = int(inp) if inp.strip() else 500000
                except ValueError:
                    target = 500000
                run_curriculum_gen(stdscr, target_size=target)

            elif sel == 1:  # Manual (Swapped from 2)
                inp = get_input_str(
                    stdscr, "Enter Target Size (Reference) (default 500):"
                )
                try:
                    target = int(inp) if inp.strip() else 500
                except ValueError:
                    target = 500
                run_manual_gen(stdscr, target_size=target)

            elif sel == 2:  # DAgger (Swapped from 1)
                path, meta_path = prompt_model_selection(
                    stdscr,
                    os.path.join(os.path.dirname(__file__), "..", "model", "weights"),
                )
                if path:
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        loading_msg = f"Loading {os.path.basename(path)}..."
                        stdscr.clear()
                        stdscr.addstr(
                            sh // 2, (sw - len(loading_msg)) // 2, loading_msg
                        )
                        stdscr.refresh()
                        model, meta = load_model(path, meta_path, dev)

                        inp = get_input_str(
                            stdscr, "Enter Max Corrections (default 2k):"
                        )
                        try:
                            target = int(inp) if inp.strip() else 2000
                        except ValueError:
                            target = 2000

                        run_dagger_gen(
                            stdscr,
                            target_corrections=target,
                            model=model,
                            meta=meta,
                            device=dev,
                        )
                    except Exception as e:
                        draw_summary(stdscr, "Error Loading Model", [str(e)[: sw - 4]])

            elif sel == 3:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["curriculum", "dagger"])
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.mode:
        if args.mode == "curriculum":
            run_curriculum_gen(None, test_mode=args.test)
        elif args.mode == "dagger":
            print(
                "Headless DAgger requires manual model loading configuration. Please use interactive mode."
            )
    else:
        curses.wrapper(main_tui)


if __name__ == "__main__":
    main()
