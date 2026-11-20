"""The rules of Snake, with no screen attached.

A state is the snake, head first, and the food. Everything here is a pure
function of a state, so the transformer's output can be checked against it.
The rules are those of the terminal game in games/snake.py, including two that
are easy to miss: moving into the cell the tail is about to leave is a death,
and so is reversing into your own neck.
"""

from __future__ import annotations

from dataclasses import dataclass

SIZE = 16
ACTIONS = "UDLR"
DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

Cell = tuple[int, int]


@dataclass(frozen=True)
class State:
    snake: tuple[Cell, ...]  # head first
    food: Cell | None

    @property
    def head(self) -> Cell:
        return self.snake[0]

    def __len__(self) -> int:
        return len(self.snake)


def start() -> State:
    """The opening position of the terminal game."""
    mid = SIZE // 2
    snake = ((mid, mid), (mid, mid - 1), (mid, mid - 2))
    return State(snake, spawn_food(snake))


def spawn_food(snake: tuple[Cell, ...]) -> Cell | None:
    """Where food appears after a meal: five rows down and seven columns right
    of the head, wrapping, then the first free cell in reading order."""
    r, c = (snake[0][0] + 5) % SIZE, (snake[0][1] + 7) % SIZE
    first, occupied = (r, c), set(snake)
    while (r, c) in occupied:
        c = (c + 1) % SIZE
        if c == 0:
            r = (r + 1) % SIZE
        if (r, c) == first:
            return None  # the board is full
    return (r, c)


def free_cells(snake: tuple[Cell, ...]) -> list[Cell]:
    occupied = set(snake)
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in occupied]


def step(state: State, action: str, food_at=None) -> tuple[State | None, str]:
    """One move. Returns (next state, event); the state is None on death.

    Events: "move", "eat", "wall", "self". `food_at(snake)` chooses where new
    food goes after a meal; the default is the deterministic rule above.
    """
    dr, dc = DELTA[action]
    head = (state.head[0] + dr, state.head[1] + dc)
    if not (0 <= head[0] < SIZE and 0 <= head[1] < SIZE):
        return None, "wall"
    if head in state.snake:
        return None, "self"
    if head == state.food:
        snake = (head, *state.snake)
        return State(snake, (food_at or spawn_food)(snake)), "eat"
    return State((head, *state.snake[:-1]), state.food), "move"


def is_reversal(state: State, action: str) -> bool:
    """Does this action drive the head straight back into the neck?"""
    dr, dc = DELTA[action]
    return (state.head[0] + dr, state.head[1] + dc) == state.snake[1]


def heading(state: State) -> str:
    """The direction the snake last moved in."""
    (r0, c0), (r1, c1) = state.snake[0], state.snake[1]
    return next(a for a, d in DELTA.items() if d == (r0 - r1, c0 - c1))
