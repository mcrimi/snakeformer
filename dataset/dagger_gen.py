import sys
import os
import curses
import pickle
import torch
import random
import copy
import collections

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.snake import SnakeGame
from model.gpt import GPT, GPTConfig
from dataset.curriculum_gen import HeadlessSnakeGame, Bot, get_game_state_str, MockStdScr, KEY_STR_MAP

# Constants from snake.py (re-defined to be safe if not exported)
CMD_UP = curses.KEY_UP
CMD_DOWN = curses.KEY_DOWN
CMD_LEFT = curses.KEY_LEFT
CMD_RIGHT = curses.KEY_RIGHT

class HeadlessNeuralSnake(HeadlessSnakeGame):
    def __init__(self, stdscr, model, meta, device):
        super().__init__(stdscr)
        self.d_model = model
        self.d_meta = meta
        self.device = device
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.start_char_id = self.stoi.get('.') 
        self.stop_token_id = self.stoi.get('$')
        self.model_generated_state = None 

    def predict(self, current_board_str, current_action):
        """
        Predicts the next state string given current board and action.
        Does NOT update internal state.
        """
        action_char = KEY_STR_MAP.get(current_action, "R")
        
        # 1. Prepare Prompt
        prompt = f"B:\n{current_board_str}\nA:{action_char}\nT:\n"
        
        # 2. Inference
        encode = lambda s: [self.stoi.get(c, self.stoi.get('.', 0)) for c in s]
        decode = lambda l: ''.join([self.itos[i] for i in l])
        
        context_idxs = encode(prompt)
        context = torch.tensor(context_idxs, dtype=torch.long, device=self.device).unsqueeze(0)
        
        try:
            # Generate
            output_ids = self.d_model.generate(context, max_new_tokens=400, stop_token_id=self.stop_token_id)
            output_text = decode(output_ids[0].tolist())
            
            # Extract
            generated = output_text[len(prompt):]
            if '$' in generated:
                generated = generated.split('$')[0]
            
            generated = generated.strip()
            return generated
                
        except Exception as e:
            return "X" # Treat error as crash/end

    def get_logical_string(self):
        return get_game_state_str(self)

    def update_state_from_ascii(self, ascii_board):
        # Improved Parser: Reconstruct topology
        lines = ascii_board.strip().split('\n')
        
        new_food = None
        head = None
        body_parts = set() # Unordered 'O's
        
        for r, line in enumerate(lines):
            if r >= self.game_height: break
            for c, char in enumerate(line):
                if c >= self.game_width: break
                
                if char == 'H':
                    head = [r, c]
                elif char == 'O' or char == '#':
                    body_parts.add((r, c)) # Use tuple for set
                elif char == 'F':
                    new_food = [r, c]
                    
        if head:
            # Reconstruct Chain
            current = tuple(head)
            ordered_body = []
            
            while body_parts:
                # Find neighbors of current in body_parts
                neighbors = []
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = current[0] + dy, current[1] + dx
                    if (ny, nx) in body_parts:
                        neighbors.append((ny, nx))
                
                if not neighbors:
                    break
                
                next_part = neighbors[0]
                ordered_body.append([next_part[0], next_part[1]])
                body_parts.remove(next_part)
                current = next_part
                
            self.snake = [head] + ordered_body
            
        if new_food:
            self.food = new_food
        else:
             self.food = None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Run small test batch")
    args = parser.parse_args()
    
    total_games = 5 if args.test else 50000 # Large limit, strictly bound by corrections
    target_corrections = 5 if args.test else 2000
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "weigths", "snake_model.pt")
    meta_path = os.path.join(base_dir, "model", "weigths", "meta.pkl")
    output_file = os.path.join(base_dir, "snake_curriculum_dagger_fixes.txt")
    
    # Device
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    
    print(f"Loading model from {model_path} on {device}...")
    
    # Load Meta & Model
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        
    config = GPTConfig(
        vocab_size=meta['vocab_size'],
        block_size=meta['block_size'],
        n_embd=meta['n_embd'],
        n_head=meta['n_head'],
        n_layer=meta['n_layer'],
        dropout=0.0,
        device=device
    )
    
    model = GPT(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Initialize Game
    # We use HeadlessSnakeGame (Deterministic) as the Source of Truth
    real_game = HeadlessSnakeGame(MockStdScr())
    bot = Bot(real_game)
    
    # We use NeuralSnakeGame just for Inference (stateless wrapper)
    neural_helper = HeadlessNeuralSnake(MockStdScr(), model, meta, device)
    
    corrections_count = 0
    hallucinations_detected = 0
    
    print(f"Starting simulation. Target: {target_corrections} corrections.")
    
    with open(output_file, "w") as f: 
        for game_idx in range(total_games):
            if corrections_count >= target_corrections:
                break
                
            real_game.reset_game()
            
            step_count = 0
            while not real_game.game_over and step_count < 1000: 
                if corrections_count >= target_corrections:
                    break
                    
                step_count += 1
                
                # 1. Determine Action (Heuristic/Expert)
                path = bot.get_path_to_food()
                if path:
                    # 10% randomness for variety
                    if random.random() < 0.10:
                        action = bot.get_safe_random()
                    else:
                        action = path[0]
                else:
                    action = bot.get_safe_random()
                    
                # 2. Capture Current State (Valid Physics)
                current_state_str = get_game_state_str(real_game)
                
                # 3. Model Prediction (Hallucination Candidate)
                # We expect the model to predict the NEXT state given Current + Action
                predicted_next_state_str = neural_helper.predict(current_state_str, action)
                
                # 4. Ground Truth Step (Physics)
                # We need to know what SHOULD happen
                # We can't just step real_game yet because we need to compare first?
                # Actually, we can step real_game, get valid T, then compare.
                
                # Clone for safety if we need pre-step state later? No, we have current_state_str.
                # But we need 'T' string which is post-step.
                
                real_game.step(action)
                correct_next_state_str = get_game_state_str(real_game)
                
                # 5. Compare
                if predicted_next_state_str != correct_next_state_str:
                    hallucinations_detected += 1
                    
                    # Record the CORRECTION
                    # Input: current_state_str (Valid)
                    # Output: correct_next_state_str (Valid)
                    entry = f"B:\n{current_state_str}\nA:{KEY_STR_MAP[action]}\nT:\n{correct_next_state_str}\n$\n"
                    f.write(entry)
                    f.flush()
                    corrections_count += 1
                    
                    if hallucinations_detected % 10 == 0:
                         print(f"  -> Hallucinations: {hallucinations_detected} | Fixes: {corrections_count}")

            # End of Game Log
            reason = "Max Steps"
            if real_game.game_over:
                reason = "Death"
            
            if (game_idx + 1) % 10 == 0:
                avg_hallucinations = hallucinations_detected / (game_idx + 1)
                print(f"Game {game_idx + 1}/{total_games} finished ({reason}). Steps: {step_count}. Logged Corrections: {corrections_count}. Avg Hallucinations/Game: {avg_hallucinations:.2f}")
                
    print(f"Finished. Total fixes generated: {corrections_count}")

if __name__ == "__main__":
    main()
