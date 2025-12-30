import argparse
import sys

def parse_board(board_str):
    """
    Fast parsing of board string.
    Returns:
        head: (r, c) or None
        tail: (r, c) or None
        food: (r, c) or None
        body_set: set of (r, c) for all 'O', 'H', '#'
    """
    lines = board_str.strip().split('\n')
    head = None
    tail = None
    food = None
    body_set = set()
    
    if len(lines) != 16:
        return None, None, None, None
        
    for r, line in enumerate(lines):
        if len(line) != 16:
             return None, None, None, None
        for c, char in enumerate(line):
            if char == 'H':
                head = (r, c)
                body_set.add((r, c))
            elif char == '#':
                tail = (r, c)
                body_set.add((r, c))
            elif char == 'O':
                body_set.add((r, c))
            elif char == 'F':
                food = (r, c)
            elif char == '.':
                pass
            else:
                pass # Invalid char?
                
    return head, tail, food, body_set

def analyze_dataset(filename):
    print(f"Analyzing {filename}...")
    
    valid_boards_count = 0
    invalid_boards_count = 0
    valid_transitions_count = 0
    invalid_transitions_count = 0
    
    food_eaten_count = 0
    wall_collisions = 0
    self_collisions = 0
    
    # Track unique cells visited by H
    visited_cells = set()
    
    invalid_board_cases = []
    invalid_transition_cases = []
    
    with open(filename, 'r') as f:
        content = f.read()
        
    examples = content.split('$')
    
    for i, ex in enumerate(examples):
        if not ex.strip():
            continue
            
        lines = ex.strip().split('\n')
        
        # Expected format:
        # B:
        # (16 lines)
        # A:Action
        # T:
        # (16 lines or X)
        
        try:
            b_idx = -1
            a_idx = -1
            t_idx = -1
            
            for idx, l in enumerate(lines):
                if l.startswith("B:"): b_idx = idx
                elif l.startswith("A:"): a_idx = idx
                elif l.startswith("T:"): t_idx = idx
            
            if b_idx == -1 or a_idx == -1 or t_idx == -1:
                # Malformed chunk
                continue
                
            board_b_lines = "\n".join(lines[b_idx+1:a_idx])
            action_line = lines[a_idx]
            target_lines = "\n".join(lines[t_idx+1:])
            
            action = action_line.split(':')[1].strip()
            
            # --- 1. Validate Board B ---
            head_b, tail_b, food_b, body_b = parse_board(board_b_lines)
            
            is_board_valid = True
            reason = ""
            
            if not head_b: 
                is_board_valid = False; reason = "No Head"
            elif not tail_b: 
                # Strict check: MUST have explicit tail '#'
                is_board_valid = False; reason = "No Explicit Tail '#'"
            elif not food_b:
                is_board_valid = False; reason = "No Food"
            elif head_b not in body_b or tail_b not in body_b:
                 # Logic error in parsing?
                 is_board_valid = False; reason = "Inconsistent Body"
            
            # Connectivity Check (Simple heuristic: component count)
            # If valid, all body parts should be connected.
            # Using simple BFS from Head to check if we reach all body parts
            if is_board_valid:
                 queue = [head_b]
                 visited = {head_b}
                 count = 0
                 while queue:
                     curr = queue.pop(0)
                     count += 1
                     r, c = curr
                     for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                         nr, nc = r+dr, c+dc
                         if (nr, nc) in body_b and (nr, nc) not in visited:
                             visited.add((nr, nc))
                             queue.append((nr, nc))
                 
                 if count != len(body_b):
                     is_board_valid = False
                     reason = f"Disconnected Body (Found {count} reachable, expected {len(body_b)})"

            # Topology Check (Degree Validation)
            if is_board_valid and len(body_b) > 1:
                 for node in body_b:
                     r, c = node
                     degree = 0
                     for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                         if (r+dr, c+dc) in body_b:
                             degree += 1
                     
                     if node == head_b or node == tail_b:
                         if degree < 1: 
                             is_board_valid = False
                             reason = f"Endpoint with degree 0 at {node}"
                             break
                     else:
                         if degree < 2:
                             is_board_valid = False
                             reason = f"Body segment with degree {degree} at {node} (Possible Branch/Disjoint)"
                             break

            if is_board_valid:
                valid_boards_count += 1
                visited_cells.add(head_b)
            else:
                invalid_boards_count += 1
                invalid_board_cases.append(f"Line approx {i*20}: {reason}")
                # If board B is invalid, we can't really validate transition meaningfully
                # But let's verify if T matches expected logic regardless?
                # No, garbage in garbage out.
                continue

            # --- 2. Validate Transition B -> T ---
            
            # Predict Next State
            dy, dx = 0, 0
            if action == 'U': dy = -1
            elif action == 'D': dy = 1
            elif action == 'L': dx = -1
            elif action == 'R': dx = 1
            
            # Handle Reverse Logic?
            # Game ignores reverse inputs. 
            # We need to know previous direction to know if 'action' is reverse.
            # We can infer 'Forward' vector from Head -> Neck.
            # Neck is a neighbor of Head in body_b.
            # If Head has > 1 neighbor, snake > 2 length.
            # But Head is an endpoint, should have 1 neighbor (Neck).
            # Unless length 1?
            
            # Find Neck
            neck = None
            head_neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = head_b[0]+dr, head_b[1]+dc
                if (nr, nc) in body_b:
                    head_neighbors.append((nr, nc))
            
            # If > 1 neighbor, usually touching body coil. 
            # But in grid graph, neck is the one connected in the "path".
            # If strict connectivity held, Head has degree 1 in tree? 
            # Not essentially, touching coils adds edges.
            # Heuristic: Neck is usually the ONLY neighbor if no self-touch.
            # If self-touch, ambiguous.
            
            # Assumption: The provided Board B is valid state. 
            # We will proceed with 'action' literally first. 
            
            pred_head = (head_b[0] + dy, head_b[1] + dx)
            
            # Determine Expected Outcome
            expected_status = "ALIVE"
            
            # Wall Collision
            if not (0 <= pred_head[0] < 16 and 0 <= pred_head[1] < 16):
                expected_status = "DEAD"
                
            # Self Collision
            # Snake moves forward. Tail moves away (unless eating).
            # So hitting Tail is SAFE if not eating.
            # Hitting any other Body part is DEATH.
            # Exception: Illegal Reverse. (Head hitting Neck).
            # If we try to move into a cell that is currently occupied by body...
            
            will_eat = (pred_head == food_b)
            
            # Calculate 'Safe Body' for collision check
            # Safe body is everything currently occupied, MINUS the tail (it will move), 
            # UNLESS we eat (tail stays).
            
            # Note: This simple set logic fails for 'Neck'.
            # Moving into Neck (Reverse) doesn't cause death in game, it is IGNORED.
            # But here we simply check 'Is T valid given B and Action'.
            # If Action was ignored, T should equal B (conceptually) but snake moves forward?
            # Actually, standard snake games ignore reverse -> Snake continues in current dir.
            # If we don't implement full inference of 'current dir', checking invalid transitions is hard.
            
            # Let's check T vs Prediction
            
            is_transition_valid = True
            trans_reason = ""
            
            if target_lines.strip() == "X":
                # Dataset claims DEATH.
                # Validate if death was plausible.
                # Death if Wall OR Self (non-neck, non-tail-if-moving).
                
                # Assume valid for now if we can't prove otherwise easily without full simulation.
                # Just count stats.
                if not (0 <= pred_head[0] < 16 and 0 <= pred_head[1] < 16):
                    wall_collisions += 1
                else:
                    self_collisions += 1 # Any other death is self collision
                
            else:
                # Dataset claims SURVIVAL.
                # Parse T
                head_t, tail_t, food_t, body_t = parse_board(target_lines)
                
                if not head_t:
                    is_transition_valid = False
                    trans_reason = "Target Invalid (No Head/Tail)"
                else:
                    # Validate Movement
                    # Real head should handle 'Ignore Reverse'
                    # But checking if head_t is adjacent to head_b is a strong baseline sanity check.
                    
                    dist = abs(head_t[0] - head_b[0]) + abs(head_t[1] - head_b[1])
                    if dist != 1:
                        # Teleportation?
                        # Could be 'Ignore Reverse' -> continued forward.
                        # Forward is distinct from 'Reverse' input.
                        # Dist should still be 1 (just different direction).
                        is_transition_valid = False
                        trans_reason = f"Head Teleport: {head_b} -> {head_t}"

                    # Validate Eating
                    # If head_t == food_b, then we ate.
                    # Length should increase by 1.
                    # len(body_t) == len(body_b) + 1
                    
                    ate = (head_t == food_b)
                    if ate:
                        food_eaten_count += 1
                        if len(body_t) != len(body_b) + 1:
                            is_transition_valid = False
                            trans_reason = f"Did not grow after eating (Len {len(body_b)}->{len(body_t)})"
                        # Tail shouldn't move
                        if tail_t != tail_b:
                            # Edge case: Length 1? (Snake doesn't start at 1).
                            is_transition_valid = False
                            trans_reason = f"Tail moved while eating ({tail_b}->{tail_t})"
                    else:
                        # Didn't eat.
                        if len(body_t) != len(body_b):
                            is_transition_valid = False
                            trans_reason = f"Length changed without eating (Len {len(body_b)}->{len(body_t)})"
                        # Tail should move (unless length 1, not possible)
                        if tail_t == tail_b:
                             # Wait, simple check: Set(T) = Set(B) - {OldTail} + {NewHead}
                             pass 

            if is_transition_valid:
                valid_transitions_count += 1
            else:
                invalid_transitions_count += 1
                invalid_transition_cases.append(f"Line approx {i*20}: {trans_reason}\nB-Head:{head_b} Action:{action}")

        except Exception as e:
            print(f"Error parsing chunk {i}: {e}")
            continue

    total_cells = 16*16
    coverage = (len(visited_cells) / total_cells) * 100
    
    print("-" * 30)
    print(f"Valid Boards: {valid_boards_count}")
    print(f"Invalid Boards: {invalid_boards_count}")
    print(f"Valid Transitions: {valid_transitions_count}")
    print(f"Invalid Transitions: {invalid_transitions_count}")
    print("-" * 20)
    print(f"Food Eaten: {food_eaten_count}")
    print(f"Wall Collisions: {wall_collisions}")
    print(f"Self Collisions: {self_collisions}")
    print(f"Board Coverage: {coverage:.2f}% ({len(visited_cells)}/{total_cells})")
    print("-" * 30)
    
    if invalid_board_cases:
        print("\n[Invalid Boards Examples]")
        for c in invalid_board_cases[:10]:
            print(c)
        if len(invalid_board_cases) > 10: print(f"... and {len(invalid_board_cases)-10} more")

    if invalid_transition_cases:
        print("\n[Invalid Transition Examples]")
        for c in invalid_transition_cases[:10]:
             print(c)
        if len(invalid_transition_cases) > 10: print(f"... and {len(invalid_transition_cases)-10} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    
    analyze_dataset(args.filename)
