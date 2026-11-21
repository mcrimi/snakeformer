"""Two ways to write a board as text.

"frame" is the original: H for the head, O for the body, # for the tail. It
does not say in what order the body cells are connected, and that turns out to
matter: when the tail has two body cells next to it, the picture alone cannot
tell which of them becomes the new tail.

"arrows" draws each body cell as an arrow to the next segment towards the head
(^ v < >), and the tail as the same information in a letter (u d l r). The next
frame then follows from the current one by a local rule, with nothing to guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import DELTA, SIZE, State

ARROW = {"U": "^", "D": "v", "L": "<", "R": ">"}
TAIL = {"U": "u", "D": "d", "L": "l", "R": "r"}
STRUCTURE = "\n$:ABT"
ACTION_CHARS = "DLRU"
DEAD = "X"


def _towards(a, b) -> str:
    """The direction from cell a to the adjacent cell b."""
    return next(k for k, d in DELTA.items() if d == (b[0] - a[0], b[1] - a[1]))


@dataclass(frozen=True)
class Encoding:
    name: str
    cells: str  # the characters a board cell can be

    @property
    def chars(self) -> list[str]:
        return sorted(set(STRUCTURE + ACTION_CHARS + DEAD + self.cells))

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.chars)}

    def board(self, state: State) -> str:
        grid = [["."] * SIZE for _ in range(SIZE)]
        if state.food:
            grid[state.food[0]][state.food[1]] = "F"
        last = len(state.snake) - 1
        for i, (r, c) in enumerate(state.snake):
            if i == 0:
                ch = "H"
            elif self.name == "frame":
                ch = "#" if i == last else "O"
            else:
                d = _towards((r, c), state.snake[i - 1])
                ch = TAIL[d] if i == last else ARROW[d]
            grid[r][c] = ch
        return "\n".join("".join(row) for row in grid)

    def prompt(self, state: State, action: str) -> str:
        return f"B:\n{self.board(state)}\nA:{action}\nT:\n"

    def target(self, nxt: State | None) -> str:
        return f"{DEAD if nxt is None else self.board(nxt)}\n$\n"

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        return [stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        chars = self.chars
        return "".join(chars[i] for i in ids)

    def parse(self, board: str) -> State | None:
        """Read a board back into a state. None if it is not a legal picture.

        Only the arrows encoding can be read back exactly; for a frame board
        the body order is recovered by `orderings` in ambiguity.py.
        """
        if self.name != "arrows":
            raise ValueError("a frame board does not determine the snake's order")
        rows = board.split("\n")
        if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
            return None
        head = food = tail = None
        points = {}
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "H":
                    if head:
                        return None
                    head = (r, c)
                elif ch == "F":
                    if food:
                        return None
                    food = (r, c)
                elif ch in "^v<>udlr":
                    k = "UDLR"["^v<>".index(ch)] if ch in "^v<>" else ch.upper()
                    points[(r, c)] = (r + DELTA[k][0], c + DELTA[k][1])
                    if ch in "udlr":
                        if tail:
                            return None
                        tail = (r, c)
                elif ch != ".":
                    return None
        if not head or not tail:
            return None
        snake, cell = [tail], tail
        while cell != head:
            cell = points.get(cell)
            if cell is None or cell in snake or (cell != head and cell not in points):
                return None
            snake.append(cell)
        if len(snake) != len(points) + 1:
            return None
        return State(tuple(reversed(snake)), food)


FRAME = Encoding("frame", ".FHO#")
ARROWS = Encoding("arrows", ".FH^v<>udlr")
ENCODINGS = {"frame": FRAME, "arrows": ARROWS}

PROMPT_LEN = 3 + (SIZE * (SIZE + 1) - 1) + 4 + 4  # "B:\n" board "\nA:x" "\nT:\n" = 282
BOARD_TARGET_LEN = (SIZE * (SIZE + 1) - 1) + 3  # board "\n$\n" = 274
