"""Positions to train and test on.

Three players wander the game so that the positions cover short and long
snakes, open boards and tight coils. From every position visited, all four
actions are candidates for the dataset, which is how deaths, reversals and
meals get in: no player would choose most of them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ambiguity import is_ambiguous
from .engine import ACTIONS, SIZE, State, free_cells, is_reversal, start, step

KINDS = ("move", "eat", "wall", "self", "reversal")


@dataclass(frozen=True)
class Example:
    state: State
    action: str
    next: State | None
    kind: str  # one of KINDS; "self" excludes reversals
    ambiguous: bool  # an ordinary move whose frame board does not determine the next frame


def _cycle() -> dict:
    """A closed tour of the board: next cell for each cell. Following it never dies."""
    order = [(0, c) for c in range(SIZE)]
    for c in range(SIZE - 1, -1, -1):
        rows = range(1, SIZE) if (SIZE - 1 - c) % 2 == 0 else range(SIZE - 1, 0, -1)
        order += [(r, c) for r in rows]
    return {a: b for a, b in zip(order, order[1:] + order[:1], strict=False)}


CYCLE = _cycle()


def _safe(state: State) -> list[tuple[str, State]]:
    out = []
    for a in ACTIONS:
        nxt, _ = step(state, a)
        if nxt is not None:
            out.append((a, nxt))
    return out


def _towards_food(state: State, options):
    if state.food is None:
        return options[0]
    dist = lambda s: abs(s.head[0] - state.food[0]) + abs(s.head[1] - state.food[1])
    return min(options, key=lambda o: dist(o[1]))


def wander(rng: random.Random, food_at=None, max_steps: int = 3000):
    """Yield the positions of one game played by a randomly chosen player."""
    style = rng.choice(["greedy", "greedy", "tour", "tour", "drunk"])
    slack = rng.choice([0.0, 0.05, 0.2, 0.5])
    state = start()
    if rng.random() < 0.6:
        # Start from a snake already laid along the tour, so long snakes are common.
        length = min(int(rng.expovariate(1 / 40)) + 3, SIZE * SIZE - 8)
        cells = [rng.choice(list(CYCLE))]
        for _ in range(length - 1):
            cells.append(CYCLE[cells[-1]])
        snake = tuple(reversed(cells))
        state = State(snake, rng.choice(free_cells(snake)))
    elif food_at:
        state = State(state.snake, food_at(state.snake))
    for _ in range(max_steps):
        yield state
        options = _safe(state)
        if not options:
            return
        if style == "drunk" or rng.random() < slack:
            action = rng.choice(options)[0]
        elif style == "greedy":
            action = _towards_food(state, options)[0]
        else:
            want = CYCLE[state.head]
            action = next((a for a, s in options if s.head == want), rng.choice(options)[0])
        state, _ = step(state, action, food_at)


def label(state: State, action: str, food_at=None, check: bool = True) -> Example:
    """check=False skips the ambiguity search, which is slow for long coiled
    snakes and only needed where results are reported."""
    nxt, event = step(state, action, food_at)
    kind = "reversal" if event == "self" and is_reversal(state, action) else event
    return Example(state, action, nxt, kind, check and kind == "move" and is_ambiguous(state))


KEEP = {"move": 0.3, "reversal": 0.2, "eat": 1.0, "wall": 1.0, "self": 1.0}


def generate(n: int, seed: int, max_len: int | None = None, min_len: int = 0,
             food_at=None, exclude: set | None = None, check: bool = True) -> list[Example]:
    """n distinct examples. Common kinds are thinned (KEEP) and, now and then,
    the food is put next to the head, so that meals are well represented.
    Positions in `exclude` are skipped."""
    rng = random.Random(seed)
    seen = set(exclude or ())
    out: list[Example] = []
    chooser = food_at
    if food_at == "random":
        def chooser(snake):
            free = free_cells(snake)
            return rng.choice(free) if free else None
    while len(out) < n:
        for state in wander(rng, chooser):
            if not (min_len <= len(state) and (max_len is None or len(state) <= max_len)):
                continue
            if rng.random() < 0.12:
                beside = [s.head for _, s in _safe(state)]
                if beside:
                    state = State(state.snake, rng.choice(beside))
            for action in ACTIONS:
                key = (state, action)
                if key in seen:
                    continue
                ex = label(state, action, chooser, check=False)
                if rng.random() > KEEP[ex.kind]:
                    continue
                seen.add(key)
                out.append(ex)
    rng.shuffle(out)
    out = out[:n]
    return [label(e.state, e.action, lambda snake, e=e: e.next.food) if e.kind == "move" else e
            for e in out] if check else out


def keys(examples) -> set:
    return {(e.state, e.action) for e in examples}
