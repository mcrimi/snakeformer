import curses
import random
import time
import os
import argparse
from collections import deque

# Constants
SNAKE_CHAR = "██"
FOOD_CHAR = "██"
BORDER_CHAR = "▒"

# Colors pairs
COLOR_SNAKE = 1
COLOR_FOOD = 2
COLOR_BORDER = 3
COLOR_TEXT = 4
COLOR_ACTION_USER = 5
COLOR_ACTION_AUTO = 6
COLOR_QUIT = 7


class SnakeGame:
    def __init__(self, stdscr, record_file=None):
        self.stdscr = stdscr
        # Get screen size (24 rows y 80 columns x) for UI space
        self.height, self.width = 24, 80
        self.score = 0
        self.game_over = False
        self.snake = []
        self.direction = None
        self.food = None

        # Fixed logical internal size (16x16)
        # We decouple this from screen size so we can use extra screen space for debug UI
        self.game_width = 16
        self.game_height = 16

        # Recording
        self.record_file = record_file
        self.left_panel = []  # Lines for left side (Board, Action)
        self.right_panel = []  # Lines for right side (Target)

        # Action History
        self.action_history = deque(maxlen=20)

        self.setup_curses()
        self.reset_game()

    def setup_curses(self):
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(1)  # Non-blocking input
        self.stdscr.timeout(100)  # Refresh every 100ms

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(COLOR_SNAKE, curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_FOOD, curses.COLOR_RED, -1)
        curses.init_pair(COLOR_BORDER, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_TEXT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_ACTION_USER, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_ACTION_AUTO, curses.COLOR_MAGENTA, -1)
        curses.init_pair(COLOR_QUIT, curses.COLOR_BLACK, curses.COLOR_WHITE)

    def calculate_next_position(self, head, direction):
        new_head = [head[0], head[1]]
        if direction == curses.KEY_UP:
            new_head[0] -= 1
        elif direction == curses.KEY_DOWN:
            new_head[0] += 1
        elif direction == curses.KEY_LEFT:
            new_head[1] -= 1
        elif direction == curses.KEY_RIGHT:
            new_head[1] += 1
        return new_head

    def simulate_next_step(self, snake, food, direction):
        """
        Pure logic method to determine the next state of the game.
        Returns: (next_snake, next_food, game_over, target_str)
        """
        if not snake:
            return None, None, True, "X"

        head = snake[0]
        new_head = self.calculate_next_position(head, direction)

        # Check Collisions
        # Wall
        if (
            new_head[0] < 0
            or new_head[0] >= self.game_height
            or new_head[1] < 0
            or new_head[1] >= self.game_width
        ):
            return snake, food, True, "X"

        # Self
        if new_head in snake:
            return snake, food, True, "X"

        # Valid Move
        new_snake = [new_head] + snake
        new_food = food

        if new_head == food:
            # Eat
            new_food = self.calculate_food_spawn_pos(new_snake)
        else:
            # Move
            new_snake.pop()

        target_str = self.render_board_state(new_snake, new_food)
        return new_snake, new_food, False, target_str

    def calculate_food_spawn_pos(self, snake):
        """
        Deterministic food spawn logic.
        """
        head = snake[0]
        target_r = (head[0] + 5) % self.game_height
        target_c = (head[1] + 7) % self.game_width

        start_r, start_c = target_r, target_c

        while [target_r, target_c] in snake:
            target_c = (target_c + 1) % self.game_width
            if target_c == 0:
                target_r = (target_r + 1) % self.game_height

            if (target_r, target_c) == (start_r, start_c):
                return None  # Board Full

        return [target_r, target_c]

    def render_board_state(self, snake, food):
        """
        Generates clean ASCII representation.
        """
        grid = [["." for _ in range(self.game_width)] for _ in range(self.game_height)]

        # Food
        if food:
            if 0 <= food[0] < self.game_height and 0 <= food[1] < self.game_width:
                grid[food[0]][food[1]] = "F"

        # Snake
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

    def reset_game(self):
        self.score = 0
        self.game_over = False
        # Start in middle
        cy, cx = self.game_height // 2, self.game_width // 2
        self.snake = [[cy, cx], [cy, cx - 1], [cy, cx - 2]]
        self.direction = curses.KEY_RIGHT
        self.spawn_food()

    def handle_input(self):
        try:
            key = self.stdscr.getch()
        except:
            key = -1

        if key == -1:
            return

        # Map WASD to Arrow Keys
        if key == ord("w") or key == ord("W"):
            key = curses.KEY_UP
        if key == ord("s") or key == ord("S"):
            key = curses.KEY_DOWN
        if key == ord("a") or key == ord("A"):
            key = curses.KEY_LEFT
        if key == ord("d") or key == ord("D"):
            key = curses.KEY_RIGHT

        # Prevent reversing direction
        if key == curses.KEY_UP and self.direction != curses.KEY_DOWN:
            self.direction = key
        elif key == curses.KEY_DOWN and self.direction != curses.KEY_UP:
            self.direction = key
        elif key == curses.KEY_LEFT and self.direction != curses.KEY_RIGHT:
            self.direction = key
        elif key == curses.KEY_RIGHT and self.direction != curses.KEY_LEFT:
            self.direction = key

        return key

    def get_action_str(self, key):
        if key == curses.KEY_UP:
            return "U"
        if key == curses.KEY_DOWN:
            return "D"
        if key == curses.KEY_LEFT:
            return "L"
        if key == curses.KEY_RIGHT:
            return "R"
        return "N"

    def spawn_food(self):
        self.food = self.calculate_food_spawn_pos(self.snake)

    def update(self):
        if self.game_over:
            return

        # --- Capture Pre-State ---
        pre_board = self.render_board_state(self.snake, self.food)
        action_str = self.get_action_str(self.direction)
        # -------------------------

        # Simulate Step
        new_snake, new_food, game_over, target_str = self.simulate_next_step(
            self.snake, self.food, self.direction
        )

        self.game_over = game_over

        if not self.game_over:
            # Detect Score Increase
            if len(new_snake) > len(self.snake):
                self.score += 10
                # Speed up
                delay = max(30, 100 - (self.score // 50) * 5)
                self.stdscr.timeout(delay)

            self.snake = new_snake
            self.food = new_food

        # Update Debug Display & Log

        # Update Debug Display & Log
        # Left Panel: Board + Action
        l_entry = f"B:\n{pre_board}\nA:{action_str}"
        self.left_panel = l_entry.split("\n")
        self.right_panel = f"T:\n{target_str}".split("\n")

        # Log to file if requested
        if self.record_file:
            try:
                full_entry = f"{l_entry}\nT:\n{target_str}\n$"
                with open(self.record_file, "a") as f:
                    f.write(full_entry + "\n")
            except Exception as e:
                pass

        if self.game_over:
            return

    def draw_box(self, y, x, h, w):
        # Draw border manually to handle the double-width characters better
        # Top/Bottom
        for i in range(w + 2):
            try:
                self.stdscr.addstr(
                    y, x + i * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER)
                )
                self.stdscr.addstr(
                    y + h + 1,
                    x + i * 2,
                    BORDER_CHAR * 2,
                    curses.color_pair(COLOR_BORDER),
                )
            except curses.error:
                pass

        # Left/Right
        for i in range(h + 2):
            try:
                self.stdscr.addstr(
                    y + i, x, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER)
                )
                self.stdscr.addstr(
                    y + i,
                    x + (w + 1) * 2,
                    BORDER_CHAR * 2,
                    curses.color_pair(COLOR_BORDER),
                )
            except curses.error:
                pass

        self.stdscr.refresh()

    def draw_action_log(self):
        # Draw at top-left
        # Title
        try:
            self.stdscr.addstr(
                0, 0, "Action Log:", curses.color_pair(COLOR_TEXT) | curses.A_BOLD
            )
        except curses.error:
            pass

        for i, (source, action) in enumerate(self.action_history):
            color = (
                curses.color_pair(COLOR_ACTION_USER)
                if source == "USER"
                else curses.color_pair(COLOR_ACTION_AUTO)
            )
            text = f"{source}: {action}"
            try:
                self.stdscr.addstr(1 + i, 0, text, color)
            except curses.error:
                pass

    def get_centered_offsets(self):
        sh, sw = self.stdscr.getmaxyx()
        offset_y = max(0, (sh - (self.game_height + 2)) // 2)
        offset_x = max(0, (sw - (self.game_width + 2) * 2) // 2)
        return offset_y, offset_x

    def draw_snake(self, offset_y, offset_x):
        content_y = offset_y + 1
        content_x = offset_x + 2
        for part in self.snake:
            try:
                self.stdscr.addstr(
                    content_y + part[0],
                    content_x + part[1] * 2,
                    SNAKE_CHAR,
                    curses.color_pair(COLOR_SNAKE),
                )
            except curses.error:
                pass

    def draw_food(self, offset_y, offset_x):
        if self.food:
            content_y = offset_y + 1
            content_x = offset_x + 2
            try:
                self.stdscr.addstr(
                    content_y + self.food[0],
                    content_x + self.food[1] * 2,
                    FOOD_CHAR,
                    curses.color_pair(COLOR_FOOD),
                )
            except curses.error:
                pass

    def draw_score(self, offset_y, offset_x):
        score_text = f" Score: {self.score}   "
        quit_text = "'Q' to Quit"
        try:
            self.stdscr.addstr(
                offset_y + self.game_height + 1,
                offset_x + 2,
                score_text,
                curses.color_pair(COLOR_TEXT) | curses.A_BOLD,
            )
            self.stdscr.addstr(
                offset_y + self.game_height + 1,
                offset_x + 2 + len(score_text),
                quit_text,
                curses.color_pair(COLOR_QUIT),
            )
        except curses.error:
            pass

    def draw_game_over_message(self, offset_y, offset_x):
        msg1 = " 🐍 GAME OVER 🐍"
        msg2 = "'R' to Restart or 'Q' to Quit  "
        y = offset_y + self.game_height // 2
        box_w = (self.game_width + 2) * 2
        x1 = offset_x + (box_w - len(msg1)) // 2
        x2 = offset_x + (box_w - len(msg2)) // 2
        try:
            self.stdscr.addstr(
                y,
                x1,
                msg1,
                curses.color_pair(COLOR_TEXT) | curses.A_REVERSE | curses.A_BOLD,
            )
            self.stdscr.addstr(
                y + 1,
                x2,
                msg2,
                curses.color_pair(COLOR_TEXT) | curses.A_REVERSE | curses.A_BOLD,
            )
        except curses.error:
            pass

    def draw_left_panel(self, offset_y, offset_x):
        sh, _ = self.stdscr.getmaxyx()
        lx = max(0, offset_x - 20)
        ly = offset_y
        try:
            for i, line in enumerate(self.left_panel):
                if ly + i < sh:
                    self.stdscr.addstr(
                        ly + i, lx, line[:18], curses.color_pair(COLOR_TEXT)
                    )
        except curses.error:
            pass

    def draw_right_panel(self, offset_y, offset_x):
        sh, _ = self.stdscr.getmaxyx()
        rx = offset_x + (self.game_width * 2) + 6
        ry = offset_y
        try:
            for i, line in enumerate(self.right_panel):
                if ry + i < sh:
                    self.stdscr.addstr(
                        ry + i, rx, line[:18], curses.color_pair(COLOR_TEXT)
                    )
        except curses.error:
            pass

    def render(self):
        self.stdscr.erase()
        offset_y, offset_x = self.get_centered_offsets()

        # Draw Border
        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)

        # Draw Elements
        self.draw_snake(offset_y, offset_x)
        self.draw_food(offset_y, offset_x)
        self.draw_score(offset_y, offset_x)

        if self.game_over:
            self.draw_game_over_message(offset_y, offset_x)

        self.draw_left_panel(offset_y, offset_x)
        self.draw_right_panel(offset_y, offset_x)
        self.draw_action_log()

        self.stdscr.refresh()

    def run(self):
        while True:
            key = self.handle_input()
            if key == ord("q") or key == ord("Q"):
                break

            if self.game_over:
                if key == ord("r") or key == ord("R"):
                    self.reset_game()
            else:
                self.update()

            self.render()

    def get_logical_string(self):
        # Backward compatibility wrapper
        return self.render_board_state(self.snake, self.food)


def main(stdscr, record_file):
    game = SnakeGame(stdscr, record_file=record_file)
    game.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, help="Optional file to record game logs")
    args = parser.parse_args()
    curses.wrapper(main, args.record)
