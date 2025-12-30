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
        self.left_panel = [] # Lines for left side (Board, Action)
        self.right_panel = [] # Lines for right side (Target)
        
        # Action History
        self.action_history = deque(maxlen=20)
        
        self.setup_curses()
        self.reset_game()

    def setup_curses(self):
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(1)  # Non-blocking input
        self.stdscr.timeout(100) # Refresh every 100ms
        
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(COLOR_SNAKE, curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_FOOD, curses.COLOR_RED, -1)
        curses.init_pair(COLOR_BORDER, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_TEXT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_ACTION_USER, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_ACTION_AUTO, curses.COLOR_MAGENTA, -1)

    def reset_game(self):
        self.score = 0
        self.game_over = False
        # Start in middle
        cy, cx = self.game_height // 2, self.game_width // 2
        self.snake = [[cy, cx], [cy, cx-1], [cy, cx-2]]
        self.direction = curses.KEY_RIGHT
        self.spawn_food()

    def spawn_food(self):
        # Deterministic "Knight's Move" shift + Offset
        # This keeps the food jumping around but maintains a strictly linear relationship
        # that the model's attention heads can easily track.
        head = self.snake[0]
        
        target_r = (head[0] + 5) % self.game_height
        target_c = (head[1] + 7) % self.game_width
        
        # If that spot is taken, just Linear Scan until we find an empty one
        # (This fallback is also easy to learn: "If target is blocked, look next door")
        # We need to ensure we don't loop forever if board is 100% full
        start_r, start_c = target_r, target_c
        
        while [target_r, target_c] in self.snake:
            target_c = (target_c + 1) % self.game_width
            if target_c == 0: # Wrap to next row
                target_r = (target_r + 1) % self.game_height
            
            # Safety break if we looped full circle (board full)
            if (target_r, target_c) == (start_r, start_c):
                self.food = None # Or handle game win
                return

        self.food = [target_r, target_c]

    def handle_input(self):
        try:
            key = self.stdscr.getch()
        except:
            key = -1
            
        if key == -1:
            return

        # Map WASD to Arrow Keys
        if key == ord('w') or key == ord('W'): key = curses.KEY_UP
        if key == ord('s') or key == ord('S'): key = curses.KEY_DOWN
        if key == ord('a') or key == ord('A'): key = curses.KEY_LEFT
        if key == ord('d') or key == ord('D'): key = curses.KEY_RIGHT

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
        if key == curses.KEY_UP: return "U"
        if key == curses.KEY_DOWN: return "D"
        if key == curses.KEY_LEFT: return "L"
        if key == curses.KEY_RIGHT: return "R"
        return "N"

    def update(self):
        if self.game_over:
            return

        # --- Capture Pre-State ---
        pre_board = self.get_logical_string()
        action_str = self.get_action_str(self.direction)
        # -------------------------

        head = self.snake[0]
        new_head = [head[0], head[1]]

        if self.direction == curses.KEY_UP:
            new_head[0] -= 1
        elif self.direction == curses.KEY_DOWN:
            new_head[0] += 1
        elif self.direction == curses.KEY_LEFT:
            new_head[1] -= 1
        if self.direction == curses.KEY_RIGHT:
            new_head[1] += 1

        # Check collisions
        # Wallsq
        # Fix: Allow index 0. Valid range is [0, game_height-1]
        
        target_str = ""
        
        if (new_head[0] < 0 or new_head[0] >= self.game_height or 
            new_head[1] < 0 or new_head[1] >= self.game_width):
            self.game_over = True
            target_str = "X"
            # Fall through to log

        # Self
        elif new_head in self.snake:
            self.game_over = True
            target_str = "X"
             # Fall through to log
        else:
            self.snake.insert(0, new_head)

            # Check Food
            if new_head == self.food:
                self.score += 10
                self.spawn_food()
                # Speed up slightly every 50 points
                delay = max(30, 100 - (self.score // 50) * 5)
                self.stdscr.timeout(delay)
            else:
                self.snake.pop()
            
            target_str = self.get_logical_string()
            
        # Update Debug Display & Log
        # Left Panel: Board + Action
        l_entry = f"B:\n{pre_board}\nA:{action_str}"
        self.left_panel = l_entry.split('\n')
        self.right_panel = f"T:\n{target_str}".split('\n')
        
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
                self.stdscr.addstr(y, x + i * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
                self.stdscr.addstr(y + h + 1, x + i * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
            except curses.error: pass
        
        # Left/Right
        for i in range(h + 2):
            try:
                self.stdscr.addstr(y + i, x, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
                self.stdscr.addstr(y + i, x + (w + 1) * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
            except curses.error: pass

        self.stdscr.refresh()

    def draw_action_log(self):
        # Draw at top-left
        # Title
        try:
            self.stdscr.addstr(0, 0, "Action Log:", curses.color_pair(COLOR_TEXT) | curses.A_BOLD)
        except curses.error: pass
        
        for i, (source, action) in enumerate(self.action_history):
            color = curses.color_pair(COLOR_ACTION_USER) if source == 'USER' else curses.color_pair(COLOR_ACTION_AUTO)
            text = f"{source}: {action}"
            try:
                self.stdscr.addstr(1 + i, 0, text, color)
            except curses.error: pass

    def render(self):
        self.stdscr.erase()
        sh, sw = self.stdscr.getmaxyx()
        
        # Calculate offsets to center the game
        # We need space for borders (Height + 2)
        offset_y = max(0, (sh - (self.game_height + 2)) // 2)
        offset_x = max(0, (sw - (self.game_width + 2) * 2) // 2)
        
        # Draw Border
        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)
        
        # Content Offset (Inside the box)
        content_y = offset_y + 1
        content_x = offset_x + 2 # 2 chars for left border
        
        # Draw Snake
        for part in self.snake:
            try:
                self.stdscr.addstr(content_y + part[0], content_x + part[1] * 2, SNAKE_CHAR, curses.color_pair(COLOR_SNAKE))
            except curses.error: pass
            
        # Draw Food
        if self.food:
            try:
                self.stdscr.addstr(content_y + self.food[0], content_x + self.food[1] * 2, FOOD_CHAR, curses.color_pair(COLOR_FOOD))
            except curses.error: pass
        
        # Draw Score
        score_text = f" Score: {self.score} "
        try:
            self.stdscr.addstr(offset_y + self.game_height + 1, offset_x + 2, score_text, curses.color_pair(COLOR_TEXT) | curses.A_BOLD)
        except curses.error: pass

        # Draw Game Over
        if self.game_over:
            msg = " GAME OVER! Press 'R' to Restart or 'Q' to Quit "
            y = offset_y + self.game_height // 2
            x = offset_x + (self.game_width * 2 - len(msg)) // 2
            try:
                self.stdscr.addstr(y, x, msg, curses.color_pair(COLOR_TEXT) | curses.A_REVERSE | curses.A_BOLD)
            except curses.error: pass
            

        # Draw Side Panels
        # Left Panel (Board/Action)
        lx = max(0, offset_x - 20)
        ly = offset_y
        try:
            for i, line in enumerate(self.left_panel):
                if ly + i < sh:
                    self.stdscr.addstr(ly + i, lx, line[:18], curses.color_pair(COLOR_TEXT))
        except curses.error: pass

        # Right Panel (Target)
        rx = offset_x + (self.game_width * 2) + 6
        ry = offset_y
        try:
            for i, line in enumerate(self.right_panel):
                if ry + i < sh:
                    self.stdscr.addstr(ry + i, rx, line[:18], curses.color_pair(COLOR_TEXT))
        except curses.error: pass

        # Draw Action Log
        self.draw_action_log()

        self.stdscr.refresh()

    def run(self):
        while True:
            key = self.handle_input()
            if key == ord('q') or key == ord('Q'):
                break
            
            if self.game_over:
                if key == ord('r') or key == ord('R'):
                    self.reset_game()
            else:
                self.update()
                
            self.render()
            
    def get_logical_string(self):
        """
        Generates a clean ASCII representation of the board for the AI.
        Ignores 'curses' double-width formatting.
        """
        # Create an empty logical grid (single chars)
        # Note: We use game_height/width which are the logical bounds
        grid = [['.' for _ in range(self.game_width)] for _ in range(self.game_height)]

        # Draw Food
        try:
            if self.food:
                # Check bounds to be safe
                if 0 <= self.food[0] < self.game_height and 0 <= self.food[1] < self.game_width:
                    grid[self.food[0]][self.food[1]] = 'F'
        except IndexError: pass
        except TypeError: pass # Handle other potential malformed food data

        # Draw Snake
        for idx, part in enumerate(self.snake):
            y, x = part[0], part[1]
            if 0 <= y < self.game_height and 0 <= x < self.game_width:
                # Differentiate Head vs Body vs Tail for easier learning
                if idx == 0:
                    char = 'H'
                elif idx == len(self.snake) - 1:
                    char = '#'
                else:
                    char = 'O'
                grid[y][x] = char

        # Convert to single string
        return "\n".join(["".join(row) for row in grid])



def main(stdscr, record_file):
    game = SnakeGame(stdscr, record_file=record_file)
    game.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, help="Optional file to record game logs")
    args = parser.parse_args()
    curses.wrapper(main, args.record)
