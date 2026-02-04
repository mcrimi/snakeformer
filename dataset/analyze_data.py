import argparse
import sys
import os
import curses
import collections

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.utils import MockStdScr, HeadlessSnakeGame, get_heatmap_grid


def parse_board(board_str):
    """Extract head, tail, food, body from a 16x16 ASCII board."""
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
    """DFS from head to tail, visiting every body cell exactly once."""
    if current == target:
        if len(path) == len(body_set):
            return path
        return None

    visited.add(current)

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
    grid_lines, max_count = get_heatmap_grid(visited_counts)
    print("  " + "".join([str(c % 10) for c in range(16)]))

    for r, line_str in enumerate(grid_lines):
        print(f"{r % 10} {line_str}")


def analyze_dataset(filename):
    print(f"Analyzing {filename} with Game Simulation...")
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
            idx_b = next(
                i for i, line_str in enumerate(lines) if line_str.startswith("B:")
            )
            idx_a = next(
                i for i, line_str in enumerate(lines) if line_str.startswith("A:")
            )
            idx_t = next(
                i for i, line_str in enumerate(lines) if line_str.startswith("T:")
            )

            str_b = "\n".join(lines[idx_b + 1 : idx_a])
            act_char = lines[idx_a].split(":")[1].strip()
            str_t = "\n".join(lines[idx_t + 1 :])

            head_b, tail_b, food_b, body_b = parse_board(str_b) or (None,) * 4
            if not head_b:
                continue

            visited_cells[head_b] += 1

            snake_list = find_snake_path(head_b, tail_b, body_b, [list(head_b)], set())
            if not snake_list:
                stats["reconstruction_fail"] += 1
                continue

            key_map = {
                "U": curses.KEY_UP,
                "D": curses.KEY_DOWN,
                "L": curses.KEY_LEFT,
                "R": curses.KEY_RIGHT,
            }
            direction = key_map.get(act_char)
            _, _, is_dead, target_str_sim = game.simulate_next_step(snake_list, food_b, direction)

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
                    if str_t.strip() != target_str_sim.strip():
                        stats["mismatched_target"] += 1
                        # print(f"Mismatch Board State. Act={act_char}")
                    else:
                        stats["valid"] += 1

        except (StopIteration, ValueError):
            continue

    print("-" * 30)
    print(f"Valid Samples: {stats['valid']}")
    print(f"Reconstruction Failures: {stats['reconstruction_fail']}")
    print(f"Target Mismatches: {stats['mismatched_target']}")
    if stats["valid"] + stats["mismatched_target"] > 0:
        accuracy = stats["valid"] / (stats["valid"] + stats["mismatched_target"]) * 100
        print(f"Simulated Accuracy: {accuracy:.2f}%")
    print("-" * 30)

    if visited_cells:
        print_heatmap(visited_cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    analyze_dataset(args.filename)
