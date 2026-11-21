"""The new rules engine against the terminal game it replaces."""

import curses
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.utils import HeadlessSnakeGame, MockStdScr  # noqa: E402
from snakeformer import FRAME, start, step  # noqa: E402
from snakeformer.engine import ACTIONS, SIZE, State, heading, is_reversal, spawn_food  # noqa: E402

KEY = {"U": curses.KEY_UP, "D": curses.KEY_DOWN, "L": curses.KEY_LEFT, "R": curses.KEY_RIGHT}


def original():
    return HeadlessSnakeGame(MockStdScr())


def test_opening_position_matches_the_terminal_game():
    game = original()
    state = start()
    assert [tuple(p) for p in game.snake] == list(state.snake)
    assert tuple(game.food) == state.food
    assert FRAME.board(state) == game.render_board_state(game.snake, game.food)


@pytest.mark.parametrize("seed", range(8))
def test_random_games_match_the_terminal_game_move_for_move(seed):
    rng = random.Random(seed)
    game, state = original(), start()
    for _ in range(600):
        action = rng.choice(ACTIONS)
        snake, food, over, text = game.simulate_next_step(game.snake, game.food, KEY[action])
        nxt, event = step(state, action)
        assert over == (nxt is None)
        if over:
            assert text == "X" and event in ("wall", "self")
            game, state = original(), start()
            continue
        assert text == FRAME.board(nxt)
        game.snake, game.food, state = snake, food, nxt


def test_events():
    s = start()
    assert heading(s) == "R"
    assert step(s, "L") == (None, "self") and is_reversal(s, "L")
    assert step(s, "R")[1] == "move" and not is_reversal(s, "R")
    at_wall = State(((0, 5), (1, 5), (2, 5)), (9, 9))
    assert step(at_wall, "U") == (None, "wall")
    fed, event = step(State(((4, 4), (4, 3), (4, 2)), (4, 5)), "R")
    assert event == "eat" and len(fed) == 4 and fed.snake[-1] == (4, 2)
    assert fed.food == spawn_food(fed.snake)


def test_moving_into_the_cell_the_tail_is_leaving_is_a_death():
    square = State(((1, 1), (1, 2), (2, 2), (2, 1)), (9, 9))
    assert step(square, "D") == (None, "self")


def test_food_rule_wraps_and_skips_the_snake():
    snake = ((12, 10), (12, 9), (12, 8))
    assert spawn_food(snake) == (1, 1)
    blocked = ((0, 0), (5, 7), (5, 8))  # the target cell and the next are taken
    assert spawn_food(blocked) == (5, 9)
    assert spawn_food(tuple((r, c) for r in range(SIZE) for c in range(SIZE))) is None
