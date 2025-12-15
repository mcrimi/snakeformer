import sys
import os
import curses
import random
import collections

# Add the parent directory to sys.path to allow importing game.snake
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.snake import SnakeGame

class MockStdScr:
    def nodelay(self, *args): pass
    def timeout(self, *args): pass
    def addstr(self, *args): pass
    def erase(self): pass
    def refresh(self): pass
    def getch(self): return -1

class HeadlessSnakeGame(SnakeGame):
    def __init__(self):
        # Determine internal dimensions
        # logic: game_height = height - 4
        # logic: game_width = width // 2 - 2
        # User requested 16 lines total for the board.
        # Assuming 16x16 logical grid.
        # height - 4 = 16 => height = 20
        # width // 2 - 2 = 16 => width // 2 = 18 => width = 36
        
        self.height, self.width = 20, 36
        self.score = 0
        self.game_over = False
        self.snake = []
        self.direction = None
        self.food = None
        
        self.game_width = self.width // 2 - 2
        self.game_height = self.height - 4
        
        # Mock stdscr
        self.stdscr = MockStdScr()
        self.reset_game()

    def setup_curses(self):
        pass

    def render(self):
        pass
    
    # We need to expose update() to force a step with a specific key
    def step(self, key):
        if self.game_over:
            return

        # Simulate input handling logic from handle_input + update
        # In snake.py, handle_input updates self.direction. 
        # Here we just set self.direction directly based on the 'key' (action) passed in.
        
        # ALLOW reversing direction (immediate death)
        if key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
            self.direction = key
        
        self.update()

class HeuristicBot:
    def __init__(self, game, epsilon=0.0):
        self.game = game
        self.epsilon = epsilon

    def get_action(self):
        # Epsilon-greedy: Random move
        if random.random() < self.epsilon:
            return random.choice([curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT])

        # BFS to find shortest path to food
        start = (self.game.snake[0][0], self.game.snake[0][1])
        target = (self.game.food[0], self.game.food[1])
        
        queue = collections.deque([(start, [])])
        visited = set()
        visited.add(start)
        
        # Occupied by snake (excluding tail since it will move, but keeping it simple for safety)
        # Actually tail moves, so strictly speaking we can move into tail spot, 
        # but safely treating it as obstacle is fine.
        obstacles = set((p[0], p[1]) for p in self.game.snake)
        
        path_to_food = None

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                path_to_food = path
                break
            
            # Explore neighbors
            # Directions: Up, Down, Left, Right
            moves = [
                ((-1, 0), curses.KEY_UP),
                ((1, 0), curses.KEY_DOWN),
                ((0, -1), curses.KEY_LEFT),
                ((0, 1), curses.KEY_RIGHT)
            ]
            
            for move_diff, key in moves:
                ny, nx = curr[0] + move_diff[0], curr[1] + move_diff[1]
                
                # Check walls
                if not (0 <= ny < self.game.game_height and 0 <= nx < self.game.game_width):
                    continue
                    
                # Check obstacles (body)
                if (ny, nx) in obstacles:
                    continue
                    
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    new_path = list(path)
                    new_path.append(key)
                    queue.append(((ny, nx), new_path))
        
        if path_to_food:
            return path_to_food[0]
            
        # Fallback: Make any valid move that doesn't kill us immediately
        # Use simpler check
        possible_moves = [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]
        random.shuffle(possible_moves)
        
        head = self.game.snake[0]
        for move in possible_moves:
            dy, dx = 0, 0
            if move == curses.KEY_UP: dy = -1
            elif move == curses.KEY_DOWN: dy = 1
            elif move == curses.KEY_LEFT: dx = -1
            elif move == curses.KEY_RIGHT: dx = 1
            
            ny, nx = head[0] + dy, head[1] + dx
            
            # Check walls
            if not (0 <= ny < self.game.game_height and 0 <= nx < self.game.game_width):
                continue
            # Check self
            if [ny, nx] in self.game.snake:
                continue
            
            return move
            
        return self.game.direction # Continue current direction if no options (will likely die)

def key_to_str(key):
    if key == curses.KEY_UP: return "UP"
    if key == curses.KEY_DOWN: return "DOWN"
    if key == curses.KEY_LEFT: return "LEFT"
    if key == curses.KEY_RIGHT: return "RIGHT"
    return "NONE"

def main():
    game = HeadlessSnakeGame()
    bot = HeuristicBot(game, epsilon=0.05) # 5% random moves
    
    num_samples = 750000 # Generate enough data
    output_file = "dataset/snake_data.txt"
    abs_output_file = os.path.join(os.path.dirname(__file__), "snake_data.txt")
    
    count = 0
    
    with open(abs_output_file, "w") as f:
        while count < num_samples:
            # Capture state before move
            board_str = game.get_logical_string()
            
            # Decide move
            action_key = bot.get_action()
            action_str = key_to_str(action_key)
            
            # Advance game
            game.step(action_key)
            
            # Capture state after move (Target)
            if game.game_over:
                target_str = "X"
            else:
                target_str = game.get_logical_string()
            
            # Write to file
            f.write("BOARD:\n")
            f.write(board_str + "\n")
            f.write(f"ACTION: {action_str}\n")
            f.write("TARGET:\n")
            f.write(target_str + "\n")
            f.write("$\n")
            
            count += 1
            if count % 1000 == 0:
                print(f"Generated {count} samples...")

            if game.game_over:
                game.reset_game()

    print(f"Done! Generated {count} samples in {abs_output_file}")

if __name__ == "__main__":
    main()
