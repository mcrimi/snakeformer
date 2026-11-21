import random

import pytest

from snakeformer import ARROWS, FRAME, start, step
from snakeformer.ambiguity import is_ambiguous, new_tail_candidates
from snakeformer.data import CYCLE, generate, keys, wander
from snakeformer.encoding import BOARD_TARGET_LEN, PROMPT_LEN
from snakeformer.engine import SIZE, State


def test_lengths_and_vocabularies():
    s = start()
    for enc in (FRAME, ARROWS):
        assert len(enc.prompt(s, "U")) == PROMPT_LEN == 282
        assert len(enc.target(step(s, "U")[0])) == BOARD_TARGET_LEN == 274
        assert enc.target(None) == "X\n$\n"
        text = enc.prompt(s, "U") + enc.target(step(s, "U")[0])
        assert enc.decode(enc.encode(text)) == text
    assert len(FRAME.chars) == 16  # the published model's vocabulary
    assert len(ARROWS.chars) == 22


def test_arrows_point_towards_the_head():
    s = State(((3, 3), (3, 2), (4, 2), (5, 2)), (0, 0))
    rows = ARROWS.board(s).split("\n")
    assert rows[3][2:4] == ">H" and rows[4][2] == "^" and rows[5][2] == "u"
    assert rows[0][0] == "F"


@pytest.mark.parametrize("seed", range(5))
def test_an_arrows_board_reads_back_to_the_same_state(seed):
    for state in wander(random.Random(seed)):
        assert ARROWS.parse(ARROWS.board(state)) == state


def test_malformed_arrows_boards_are_rejected():
    good = ARROWS.board(start())
    assert ARROWS.parse(good.replace("H", ".")) is None  # no head
    assert ARROWS.parse(good.replace("r", ">")) is None  # no tail
    assert ARROWS.parse(good.replace(">", "^")) is None  # arrow leads away from the snake
    assert ARROWS.parse(good[:-1]) is None  # short row
    assert ARROWS.parse(good.replace(".", "H", 1)) is None  # two heads
    with pytest.raises(ValueError):
        FRAME.parse(FRAME.board(start()))


def test_a_straight_snake_is_determined():
    assert not is_ambiguous(start())
    assert new_tail_candidates(start()) == {(8, 7)}


def test_one_picture_two_snakes():
    # Two snakes that draw the same frame board. In `a` the segment before the
    # tail is above it; in `b` it is to its right. After a move the pictures differ.
    #   O O O
    #   O O O
    #   # O H
    a = State(((2, 2), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0)), (9, 9))
    b = State(((2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0)), (9, 9))
    assert FRAME.board(a) == FRAME.board(b)
    assert ARROWS.board(a) != ARROWS.board(b)
    assert new_tail_candidates(a) == new_tail_candidates(b) == {(1, 0), (2, 1)}
    after_a, after_b = step(a, "R")[0], step(b, "R")[0]
    assert FRAME.board(after_a) != FRAME.board(after_b)


def test_the_truth_is_always_among_the_candidates():
    rng = random.Random(3)
    seen = 0
    for _ in range(6):
        for state in wander(rng, max_steps=400):
            cands = new_tail_candidates(state)
            assert state.snake[-2] in cands
            seen += len(cands) > 1
    assert seen > 0  # ambiguous positions do come up in play


def test_ambiguous_positions_have_a_second_snake_with_the_same_picture():
    """Constructive check: find the other reading and confirm the futures differ."""
    rng = random.Random(5)
    found = 0
    for _ in range(40):
        for state in wander(rng, max_steps=300):
            if len(state) > 14 or not is_ambiguous(state):
                continue
            other = _other_reading(state)
            assert other is not None
            assert FRAME.board(other) == FRAME.board(state)
            assert other.snake[-2] != state.snake[-2]
            found += 1
            break
        if found >= 3:
            break
    assert found >= 1


def _other_reading(state):
    head, tail, body = state.snake[0], state.snake[-1], set(state.snake[1:-1])
    out = []

    def walk(path):
        if len(path) == len(body) + 1:
            last = path[-1]
            if abs(last[0] - tail[0]) + abs(last[1] - tail[1]) == 1 and last != state.snake[-2]:
                out.append(State((*path, tail), state.food))
            return
        r, c = path[-1]
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in body and q not in path and not out:
                walk([*path, q])

    walk([head])
    return out[0] if out else None


def test_the_tour_visits_every_cell_once():
    assert len(CYCLE) == SIZE * SIZE
    cell, seen = (0, 0), set()
    for _ in range(SIZE * SIZE):
        seen.add(cell)
        nxt = CYCLE[cell]
        assert abs(nxt[0] - cell[0]) + abs(nxt[1] - cell[1]) == 1
        cell = nxt
    assert cell == (0, 0) and len(seen) == SIZE * SIZE


def test_generate_is_reproducible_distinct_and_covers_every_kind():
    a = generate(3000, seed=4)
    assert a == generate(3000, seed=4)
    assert len(keys(a)) == 3000
    assert {e.kind for e in a} == {"move", "eat", "wall", "self", "reversal"}
    held_out = generate(500, seed=5, exclude=keys(a))
    assert not keys(a) & keys(held_out)
    assert any(e.ambiguous for e in a) and all(e.kind == "move" for e in a if e.ambiguous)
