import sys
import os
import random
import collections
import copy

# Add the parent directory to sys.path to allow importing game.snake
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.snake import SnakeGame, curses

# Constants
CMD_UP = curses.KEY_UP
CMD_DOWN = curses.KEY_DOWN
CMD_LEFT = curses.KEY_LEFT
CMD_RIGHT = curses.KEY_RIGHT

KEY_STR_MAP = {
    CMD_UP: "U",
    CMD_DOWN: "D",
    CMD_LEFT: "L",
    CMD_RIGHT: "R"
}

OPPOSITE_MAP = {
    CMD_UP: CMD_DOWN,
    CMD_DOWN: CMD_UP,
    CMD_LEFT: CMD_RIGHT,
    CMD_RIGHT: CMD_LEFT
}

class MockStdScr:
    def nodelay(self, *args): pass
    def timeout(self, *args): pass
    def addstr(self, *args): pass
    def erase(self): pass
    def refresh(self): pass
    def getch(self): return -1
    def getmaxyx(self): return (20, 36) # Mocking size

# Helper to expose state string on standard game
def get_game_state_str(game):
    if game.game_over: return "X"
    
    grid = [['.' for _ in range(game.game_width)] for _ in range(game.game_height)]
    
    if game.food:
        fy, fx = game.food
        if 0 <= fy < game.game_height and 0 <= fx < game.game_width:
            grid[fy][fx] = 'F'
            
    for idx, part in enumerate(game.snake):
        y, x = part
        if 0 <= y < game.game_height and 0 <= x < game.game_width:
            if idx == 0:
                char = 'H'
            elif idx == len(game.snake) - 1:
                char = '#'
            else:
                char = 'O'
            grid[y][x] = char
    
    return "\n".join(["".join(row) for row in grid])

class HeadlessSnakeGame(SnakeGame):
    def setup_curses(self):
        pass
    
    # We rely on parent SnakeGame for all logic (update, spawn_food)
    # Check if parent logic uses 'self.game_height' which is set in parent __init__.
    # Parent __init__:
    # self.height, self.width = 20, 36
    # self.game_width = self.width // 2 - 2  (16)
    # self.game_height = self.height - 4 (16)
    # This matches our requirement.

    def step(self, action):
        # Force a step with specific action
        if action in [CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT]:
             # We handle input directly by setting direction
             # However, game.handle_input logic in update() isn't called here.
             # update() uses self.direction.
             # So we set self.direction.
             
             # Important: The standard game prevents reversing direction.
             # If 'action' is reverse, setting self.direction = action MIGHT be ignored if we used handle_input logic,
             # but here we set valid directions directly. 
             # Wait, SnakeGame.update doesn't check for reverse. SnakeGame.handle_input does.
             # Since we bypass handle_input, setting self.direction = Opposite WILL cause immediate reverse death in update().
             # This is perfect for Tail Suicide (if head hits neck).
             self.direction = action
        
        self.update()

    def clone(self):
        new_game = HeadlessSnakeGame(MockStdScr())
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.snake = copy.deepcopy(self.snake)
        new_game.direction = self.direction
        new_game.food = list(self.food) if self.food else None
        return new_game

class Bot:
    def __init__(self, game):
        self.game = game

    def get_path_to_food(self):
        if not self.game.food: return []
        
        start = tuple(self.game.snake[0])
        target = tuple(self.game.food)
        
        queue = collections.deque([(start, [])])
        visited = set()
        visited.add(start)
        
        obstacles = set(tuple(p) for p in self.game.snake)
        
        for move_key, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
            ny, nx = start[0] + dy, start[1] + dx
            # Wait, start is tuple, need to use current 'curr' from queue if BFSing? 
            # Logic error in previous snippet: 'curr' was popped but loop used 'curr'. Fixed below.
            pass

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path
            
            for move_key, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
                ny, nx = curr[0] + dy, curr[1] + dx
                
                if not (0 <= ny < self.game.game_height and 0 <= nx < self.game.game_width):
                    continue
                if (ny, nx) in obstacles:
                    continue
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [move_key]))
        return None

    def get_safe_random(self):
        moves = []
        head = self.game.snake[0]
        obstacles = set(tuple(p) for p in self.game.snake)
        
        for move_key, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
            ny, nx = head[0] + dy, head[1] + dx
            if 0 <= ny < self.game.game_height and 0 <= nx < self.game.game_width:
                if (ny, nx) not in obstacles:
                    moves.append(move_key)
        
        if moves: return random.choice(moves)
        return self.game.direction # doomed

def main():
    total_size = 550000
    
    # Categories
    # Standard (E in old code): 30% -> 150k
    # Glutton (A): 25% -> 125k
    # Fat Snake (C): 15% -> 75k
    # Suicide Mode (Wall) (B_Wall): 10% -> 50k
    # Tail Suicide (B_Tail): 10% -> 50k
    # Tunnel (D): 5% -> 25k
    # Illegal Move (F): 5% -> 25k
    # Drunk Walker (G): +50k
    
    target_counts = {
        "Standard": 150000,
        "Glutton": 125000,
        "Fat": 75000,
        "Suicide": 50000,
        "TailSuicide": 50000,
        "Tunnel": 25000,
        "Illegal": 25000,
        "Drunk": 50000
    }
    
    # Adjust rounding (if any, though direct numbers used now)
    curr_sum = sum(target_counts.values())
    if curr_sum < total_size:
        target_counts["Standard"] += (total_size - curr_sum)
        
    current_counts = {k: 0 for k in target_counts}
    
    output_file = os.path.join(os.path.dirname(__file__), "snake_data_curriculum.txt")
    
    game = HeadlessSnakeGame(MockStdScr())
    bot = Bot(game)
    
    print(f"Generating {total_size} examples with targets: {target_counts}")
    
    with open(output_file, "w") as f:
        while sum(current_counts.values()) < total_size:
            
            if game.game_over:
                game.reset_game()
                
            head = game.snake[0]
            food = game.food
            snake_len = len(game.snake)
            
            # --- 1. Glutton (25%) ---
            if current_counts["Glutton"] < target_counts["Glutton"]:
                 dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
                 if dist == 1:
                     # Find correct move
                     dy, dx = food[0] - head[0], food[1] - head[1]
                     action = None
                     if dy == -1: action = CMD_UP
                     elif dy == 1: action = CMD_DOWN
                     elif dx == -1: action = CMD_LEFT
                     elif dx == 1: action = CMD_RIGHT
                     
                     if action:
                         state_str = get_game_state_str(game)
                         clone = game.clone()
                         clone.step(action)
                         result_str = get_game_state_str(clone)
                         
                         f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{result_str}\n$\n")
                         current_counts["Glutton"] += 1

            # --- 2. Fat Snake (15%) ---
            if snake_len >= 15 and current_counts["Fat"] < target_counts["Fat"]:
                path = bot.get_path_to_food()
                action = path[0] if path else bot.get_safe_random()
                
                state_str = get_game_state_str(game)
                clone = game.clone()
                clone.step(action)
                result_str = get_game_state_str(clone)
                
                if not clone.game_over:
                     f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{result_str}\n$\n")
                     current_counts["Fat"] += 1
            
            # --- 3. Tunnel (5%) ---
            if current_counts["Tunnel"] < target_counts["Tunnel"]:
                valid_moves = []
                obstacles = set(tuple(p) for p in game.snake)
                for mk, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
                     ny, nx = head[0] + dy, head[1] + dx
                     if 0 <= ny < game.game_height and 0 <= nx < game.game_width:
                         if (ny, nx) not in obstacles:
                             valid_moves.append(mk)
                
                if len(valid_moves) == 1:
                     action = valid_moves[0]
                     state_str = get_game_state_str(game)
                     clone = game.clone()
                     clone.step(action)
                     result_str = get_game_state_str(clone)
                     if not clone.game_over:
                         f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{result_str}\n$\n")
                         current_counts["Tunnel"] += 1

            # --- 4. Suicide (Walls) (10%) ---
            if current_counts["Suicide"] < target_counts["Suicide"]:
                 obstacles = set(tuple(p) for p in game.snake)
                 # Find wall crash
                 wall_crash = None
                 for mk, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
                     # Prevent using Reverse for Suicide (that implies inconsistency with Illegal Move)
                     if mk == OPPOSITE_MAP.get(game.direction):
                         continue
                         
                     ny, nx = head[0] + dy, head[1] + dx
                     if not (0 <= ny < game.game_height and 0 <= nx < game.game_width):
                         wall_crash = mk
                         break
                 
                 if wall_crash and random.random() < 0.2:
                     state_str = get_game_state_str(game)
                     f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[wall_crash]}\nT:\nX\n$\n")
                     current_counts["Suicide"] += 1

            # --- 5. Tail Suicide (10%) ---
            if current_counts["TailSuicide"] < target_counts["TailSuicide"]:
                 obstacles = set(tuple(p) for p in game.snake)
                 tail_crash = None
                 for mk, (dy, dx) in [(CMD_UP, (-1,0)), (CMD_DOWN, (1,0)), (CMD_LEFT, (0,-1)), (CMD_RIGHT, (0,1))]:
                     # Prevent using Reverse for Tail Suicide
                     if mk == OPPOSITE_MAP.get(game.direction):
                         continue

                     ny, nx = head[0] + dy, head[1] + dx
                     if 0 <= ny < game.game_height and 0 <= nx < game.game_width:
                         if (ny, nx) in obstacles:
                             tail_crash = mk
                             break
                 
                 if tail_crash and random.random() < 0.2:
                      state_str = get_game_state_str(game)
                      f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[tail_crash]}\nT:\nX\n$\n")
                      current_counts["TailSuicide"] += 1

            # --- 6. Illegal Move (5%) ---
            if current_counts["Illegal"] < target_counts["Illegal"] and len(game.snake) > 1:
                # Infer actual physical direction
                head = game.snake[0]
                neck = game.snake[1]
                p_dy, p_dx = head[0] - neck[0], head[1] - neck[1]
                
                real_fwd = None
                if p_dy == -1: real_fwd = CMD_UP
                elif p_dy == 1: real_fwd = CMD_DOWN
                elif p_dx == -1: real_fwd = CMD_LEFT
                elif p_dx == 1: real_fwd = CMD_RIGHT
                
                if real_fwd and real_fwd in OPPOSITE_MAP:
                    opp = OPPOSITE_MAP[real_fwd]
                    
                    # Logic: Input Opposite.
                    # Physics Rule: Disregard invalid.
                    # Target: Result of continuing Forward (real_fwd)
                    
                    state_str = get_game_state_str(game)
                    
                    clone = game.clone()
                    clone.step(real_fwd) # Continue PHYSICAL forward
                    
                    if not clone.game_over and random.random() < 0.1:
                         result_str = get_game_state_str(clone)
                         f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[opp]}\nT:\n{result_str}\n$\n")
                         current_counts["Illegal"] += 1
            
            # --- 7. Drunk Walker (New) ---
            if current_counts["Drunk"] < target_counts["Drunk"]:
                 if random.random() < 0.15: # Do this often to capture the entropy
                     action = bot.get_safe_random()
                     state_str = get_game_state_str(game)
                     clone = game.clone()
                     clone.step(action)
                     result_str = get_game_state_str(clone)
                     
                     if not clone.game_over:
                         f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{result_str}\n$\n")
                         current_counts["Drunk"] += 1

            # --- 8. Standard (30%) ---
            if current_counts["Standard"] < target_counts["Standard"]:
                 if random.random() < 0.1:
                    path = bot.get_path_to_food()
                    action = path[0] if path else bot.get_safe_random()
                    
                    state_str = get_game_state_str(game)
                    clone = game.clone()
                    clone.step(action)
                    result_str = get_game_state_str(clone)
                    
                    # Standard should generally be survivable
                    if not clone.game_over:
                         f.write(f"B:\n{state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{result_str}\n$\n")
                         current_counts["Standard"] += 1

            # Advance Game
            path = bot.get_path_to_food()
            if path:
                if random.random() < 0.10: # Increased entropy (5% -> 10%)
                    move = bot.get_safe_random()
                else:
                    move = path[0]
            else:
                move = bot.get_safe_random()
            
            game.step(move)
            
            total_done = sum(current_counts.values())
            if total_done % 5000 == 0:
                 sys.stdout.write(f"Progress: {total_done}/{total_size} {current_counts}\r")
                 sys.stdout.flush()
                 
    print(f"\nDone! Generated {total_size} samples.")

if __name__ == "__main__":
    main()
