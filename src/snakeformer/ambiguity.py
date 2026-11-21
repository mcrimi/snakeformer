"""What a frame board leaves undetermined.

After an ordinary move the tail cell empties and the segment before it becomes
the new tail. A frame board shows where the body is but not how it is
connected. If only one connection order fits the picture, the next frame is
determined. If several fit and they disagree about which cell comes before the
tail, then no function of the picture can always be right, however it was
trained.
"""

from __future__ import annotations

from .engine import DELTA, State


def _neighbours(cell):
    return [(cell[0] + dr, cell[1] + dc) for dr, dc in DELTA.values()]


def new_tail_candidates(state: State, budget: int = 200_000) -> set:
    """Cells that could be the new tail after a move, judging by the picture alone.

    A cell qualifies if it is a body cell next to the tail and some path from
    the head through every body cell ends on it. The search is exact unless it
    runs out of budget, in which case the undecided cell is kept (so ambiguity
    is over-reported, never missed).
    """
    snake = state.snake
    if len(snake) < 4:
        return {snake[-2]}
    head, tail, body = snake[0], snake[-1], set(snake[1:-1])
    candidates = [p for p in _neighbours(tail) if p in body]
    if len(candidates) == 1:
        return set(candidates)
    left = [budget]

    def reaches(cell, seen, goal):
        left[0] -= 1
        if left[0] < 0:
            return None
        if len(seen) == len(body):
            return cell == goal
        if cell == goal:
            return False
        for q in _neighbours(cell):
            if q in body and q not in seen:
                seen.add(q)
                found = reaches(q, seen, goal)
                seen.discard(q)
                if found or found is None:
                    return found
        return False

    possible = set()
    for goal in candidates:
        for q in _neighbours(head):
            if q in body:
                found = reaches(q, {q}, goal)
                if found or found is None:
                    possible.add(goal)
                    break
    return possible


def is_ambiguous(state: State) -> bool:
    """Is the next frame of an ordinary move undetermined by the frame board?"""
    return len(new_tail_candidates(state)) > 1
