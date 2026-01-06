import argparse
import sys
import os
import curses
import collections

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class HeadlessSnakeGame(SnakeGame):
    def setup_curses(self):
        # Override to do nothing or set nodelay on mock
        if self.stdscr:
            # self.stdscr.nodelay(1)
            pass


def parse_board(board_str):
    """
    Parses 16x16 board string into components.
    Returns: head(r,c), tail(r,c), food(r,c), body_pixels(set)
    """
    lines = board_str.strip().split("\n")
    if len(lines) != 16:
        return None

    head = None
    tail = None
    food = None
    body = set()

    for r, line in enumerate(lines):
        if len(line) != 16:
            return None
        for c, char in enumerate(line):
            if char == "H":
                head = (r, c)
                body.add((r, c))
            elif char == "#":
                tail = (r, c)
                body.add((r, c))
            elif char == "O":
                body.add((r, c))
            elif char == "F":
                food = (r, c)

    return head, tail, food, body


def find_snake_path(current, target, body_set, path, visited):
    """
    DFS to find a Hamiltonian path from current (Head) to target (Tail)
    that visits all nodes in body_set exactly once.
    """
    if current == target:
        if len(path) == len(body_set):
            return path
        return None

    visited.add(current)

    # Try neighbors
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = current[0] + dr, current[1] + dc
        if (nr, nc) in body_set and (nr, nc) not in visited:
            res = find_snake_path(
                (nr, nc), target, body_set, path + [[nr, nc]], visited
            )
            if res:
                return res

    visited.remove(current)
    return None


def print_heatmap(visited_counts):
    print("\n[Board Heatmap]")

    max_count = max(visited_counts.values()) if visited_counts else 1

    # Header
    print("  " + "".join([str(c % 10) for c in range(16)]))

    for r in range(16):
        row_str = f"{r % 10} "
        for c in range(16):
            count = visited_counts.get((r, c), 0)
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
        print(row_str)


def analyze_dataset(filename):
    print(f"Analyzing {filename} with Game Simulation...")

    # Mock StdScr for Game init
    stdscr = MockStdScr()
    game = HeadlessSnakeGame(stdscr)

    stats = {"valid": 0, "invalid": 0, "reconstruction_fail": 0, "mismatched_target": 0}
    visited_cells = collections.defaultdict(int)

    with open(filename, "r") as f:
        chunks = f.read().split("$")

    for chunk in chunks:
        if not chunk.strip():
            continue

        lines = chunk.strip().split("\n")
        try:
            # Parse B, A, T
            idx_b = next(i for i, l in enumerate(lines) if l.startswith("B:"))
            idx_a = next(i for i, l in enumerate(lines) if l.startswith("A:"))
            idx_t = next(i for i, l in enumerate(lines) if l.startswith("T:"))

            str_b = "\n".join(lines[idx_b + 1 : idx_a])
            act_char = lines[idx_a].split(":")[1].strip()
            str_t = "\n".join(lines[idx_t + 1 :])

            # 1. Parse Board B
            head_b, tail_b, food_b, body_b = parse_board(str_b) or (None,) * 4
            if not head_b:
                continue

            visited_cells[head_b] += 1  # Heatmap stats

            # 2. Reconstruct Snake List
            snake_list = find_snake_path(head_b, tail_b, body_b, [list(head_b)], set())

            if not snake_list:
                stats["reconstruction_fail"] += 1
                continue

            # 3. Simulate Action
            key_map = {
                "U": curses.KEY_UP,
                "D": curses.KEY_DOWN,
                "L": curses.KEY_LEFT,
                "R": curses.KEY_RIGHT,
            }
            direction = key_map.get(act_char)

            # Remove 180 check: HeadlessSnakeGame allows suicide turns.
            sim_direction = direction

            # Run Sim
            _, _, is_dead, target_str_sim = game.simulate_next_step(
                snake_list, food_b, sim_direction
            )

            # 4. Compare T
            if str_t.strip() == "X":
                if not is_dead:
                    stats["mismatched_target"] += 1
                    print(
                        f"Mismatch: Data=Dead, Sim=Alive. Act={act_char} Head={head_b}"
                    )
                else:
                    stats["valid"] += 1
            else:
                if is_dead:
                    stats["mismatched_target"] += 1
                    print(
                        f"Mismatch: Data=Alive, Sim=Dead. Act={act_char} Head={head_b}"
                    )
                else:
                    # Compare Boards
                    if str_t.strip() != target_str_sim.strip():
                        stats["mismatched_target"] += 1
                        # print(f"Mismatch Board State. Act={act_char}")
                    else:
                        stats["valid"] += 1

        except (StopIteration, ValueError):
            continue

    # Report
    print("-" * 30)
    print(f"Valid Samples: {stats['valid']}")
    print(f"Reconstruction Failures: {stats['reconstruction_fail']}")
    print(f"Target Mismatches: {stats['mismatched_target']}")
    if stats["valid"] + stats["mismatched_target"] > 0:
        accuracy = stats["valid"] / (stats["valid"] + stats["mismatched_target"]) * 100
        print(f"Simulated Accuracy: {accuracy:.2f}%")
    print("-" * 30)

    # Heatmap
    if visited_cells:
        print_heatmap(visited_cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    analyze_dataset(args.filename)
