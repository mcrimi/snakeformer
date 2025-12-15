import curses
import random
import time

# Constants
SNAKE_CHAR = "██" 
FOOD_CHAR = "██"  
BORDER_CHAR = "▒"

# Colors pairs
COLOR_SNAKE = 1
COLOR_FOOD = 2
COLOR_BORDER = 3
COLOR_TEXT = 4

class SnakeGame:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        # Get screen size (20 rows y 36 columns x) to match 16x16 logical grid
        self.height, self.width = 20, 36
        self.score = 0
        self.game_over = False
        self.snake = []
        self.direction = None
        self.food = None
        
        # Adjust width for 2-char blocks
        self.game_width = self.width // 2 - 2
        self.game_height = self.height - 4
        
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

    def reset_game(self):
        self.score = 0
        self.game_over = False
        # Start in middle
        cy, cx = self.game_height // 2, self.game_width // 2
        self.snake = [[cy, cx], [cy, cx-1], [cy, cx-2]]
        self.direction = curses.KEY_RIGHT
        self.spawn_food()

    def spawn_food(self):
        while True:
            fy = random.randint(1, self.game_height - 1)
            fx = random.randint(1, self.game_width - 1)
            if [fy, fx] not in self.snake:
                self.food = [fy, fx]
                break

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

    def update(self):
        if self.game_over:
            return

        head = self.snake[0]
        new_head = [head[0], head[1]]

        if self.direction == curses.KEY_UP:
            new_head[0] -= 1
        elif self.direction == curses.KEY_DOWN:
            new_head[0] += 1
        elif self.direction == curses.KEY_LEFT:
            new_head[1] -= 1
        elif self.direction == curses.KEY_RIGHT:
            new_head[1] += 1

        # Check collisions
        # Walls
        if (new_head[0] <= 0 or new_head[0] >= self.game_height or 
            new_head[1] <= 0 or new_head[1] >= self.game_width):
            self.game_over = True
            return

        # Self
        if new_head in self.snake:
            self.game_over = True
            return

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

    def draw_box(self, y, x, h, w):
        # Draw border manually to handle the double-width characters better
        # Top/Bottom
        for i in range(w + 1):
            try:
                self.stdscr.addstr(y, x + i * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
                self.stdscr.addstr(y + h, x + i * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
            except curses.error: pass
        
        # Left/Right
        for i in range(h + 1):
            try:
                self.stdscr.addstr(y + i, x, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
                self.stdscr.addstr(y + i, x + w * 2, BORDER_CHAR * 2, curses.color_pair(COLOR_BORDER))
            except curses.error: pass

    def render(self):
        self.stdscr.erase()
        sh, sw = self.stdscr.getmaxyx()
        
        # Calculate offsets to center the game
        offset_y = max(0, (sh - self.height) // 2)
        offset_x = max(0, (sw - self.width) // 2)
        
        # Draw Border
        self.draw_box(offset_y, offset_x, self.game_height, self.game_width)
        
        # Draw Snake
        for part in self.snake:
            try:
                self.stdscr.addstr(offset_y + part[0], offset_x + part[1] * 2, SNAKE_CHAR, curses.color_pair(COLOR_SNAKE))
            except curses.error: pass
            
        # Draw Food
        try:
            self.stdscr.addstr(offset_y + self.food[0], offset_x + self.food[1] * 2, FOOD_CHAR, curses.color_pair(COLOR_FOOD))
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
            # Check bounds to be safe
            if 0 <= self.food[0] < self.game_height and 0 <= self.food[1] < self.game_width:
                grid[self.food[0]][self.food[1]] = 'F'
        except IndexError: pass

        # Draw Snake
        for idx, part in enumerate(self.snake):
            y, x = part[0], part[1]
            if 0 <= y < self.game_height and 0 <= x < self.game_width:
                # Differentiate Head vs Body for easier learning
                char = 'H' if idx == 0 else 'O'
                grid[y][x] = char

        # Convert to single string
        return "\n".join(["".join(row) for row in grid])

def main(stdscr):
    game = SnakeGame(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)
