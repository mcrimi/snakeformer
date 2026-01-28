"""
Shared utilities for games and dataset generation.
"""

import curses
import copy
from games.snake import SnakeGame


class MockStdScr:
    def nodelay(self, *args):
        pass

    def timeout(self, *args):
        pass

    def addstr(self, *args):
        pass

    def erase(self):
        pass

    def refresh(self):
        pass

    def getch(self):
        return -1

    def getmaxyx(self):
        return (20, 36)

    def curs_set(self, *args):
        pass

    def start_color(self):
        pass

    def use_default_colors(self):
        pass

    def init_pair(self, *args):
        pass

    def clear(self):
        pass


class HeadlessSnakeGame(SnakeGame):
    def setup_curses(self):
        # Override to do nothing or set nodelay on mock
        if self.stdscr:
            # self.stdscr.nodelay(1)
            # self.stdscr.nodelay(1)
            pass

    def step(self, action):
        # Force/teleport direction (bypassing 180 check) for data gen
        # We need CMD imports or curses keys. utils imports curses.
        # CMD_UP etc are just curses.KEY_*
        if action in [
            curses.KEY_UP,
            curses.KEY_DOWN,
            curses.KEY_LEFT,
            curses.KEY_RIGHT,
        ]:
            self.direction = action
        self.update()

    def clone(self):
        # Clones never need to draw, so always MockStdScr
        # Used in data_gen.py
        new_game = HeadlessSnakeGame(MockStdScr())
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.snake = copy.deepcopy(self.snake)
        new_game.direction = self.direction
        new_game.food = list(self.food) if self.food else None
        return new_game


def get_heatmap_grid(visited_counts):
    """
    Generates a list of strings representing the heatmap grid.
    Returns: (list_of_strings, max_count)
    """
    if not visited_counts:
        return [], 0

    ys = [p[0] for p in visited_counts.keys()]
    xs = [p[1] for p in visited_counts.keys()]

    # Default bounds (fallback) or dynamic?
    # Use dynamic bounds from data
    if not ys:
        return [], 0

    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)

    # Force 16x16 minimum to match parser expectations if empty?
    # Or just logical 0-15?
    # Let's force 0-15 lines if data is within that, or expand if outside.
    # Usually board is 0-15.
    min_y, max_y = 0, 15
    min_x, max_x = 0, 15  # Or 31 if using curses units?
    # Logic in data_gen using tuples (y,x) from game.snake.
    # Game height=16, width=32(chars) or 16(logical)?
    # SnakeGame: game_height=16, game_width=16.
    # So 0-15 range is correct for logical coordinates.

    h = max_y - min_y + 1
    w = max_x - min_x + 1

    max_count = max(visited_counts.values()) if visited_counts else 1

    grid_lines = []

    # Header row logic is usually external (column numbers) but we can provide just rows

    for r in range(h):
        row_str = ""
        for c in range(w):
            y = min_y + r
            x = min_x + c
            count = visited_counts.get((y, x), 0)

            char = "."
            if count > 0:
                char = "░"
            if count > max_count * 0.3:
                char = "▒"
            if count > max_count * 0.6:
                char = "▓"
            if count > max_count * 0.9:
                char = "█"
            row_str += char
        grid_lines.append(row_str)

    return grid_lines, max_count
