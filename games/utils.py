"""Headless game helpers and heatmap viz."""

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
        # No-op for headless mode
        pass

    def step(self, action):
        # Skip 180° turn check—data gen needs to allow suicidal moves
        if action in [
            curses.KEY_UP,
            curses.KEY_DOWN,
            curses.KEY_LEFT,
            curses.KEY_RIGHT,
        ]:
            self.direction = action
        self.update()

    def clone(self):
        # Deep copy for branching simulations
        new_game = HeadlessSnakeGame(MockStdScr())
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.snake = copy.deepcopy(self.snake)
        new_game.direction = self.direction
        new_game.food = list(self.food) if self.food else None
        return new_game


def get_heatmap_grid(visited_counts):
    """Turn visit counts into ASCII density chars (░▒▓█)."""
    if not visited_counts:
        return [], 0

    ys = [p[0] for p in visited_counts.keys()]
    xs = [p[1] for p in visited_counts.keys()]

    if not ys:
        return [], 0

    # Force 16x16 grid (standard board size)
    min_y, max_y = 0, 15
    min_x, max_x = 0, 15

    h = max_y - min_y + 1
    w = max_x - min_x + 1
    max_count = max(visited_counts.values()) if visited_counts else 1
    grid_lines = []

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
