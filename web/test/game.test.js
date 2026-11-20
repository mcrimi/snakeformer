import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { board, newTailCandidates, parseArrows, parseFrame, prompt, start, step, target } from "../js/game.js";

const cases = JSON.parse(readFileSync(new URL("./fixtures-game.json", import.meta.url)));

test("the rules and both encodings match Python on every exported transition", () => {
  assert.ok(cases.length > 200);
  for (const c of cases) {
    const state = { snake: c.snake, food: c.food };
    assert.equal(board(state, "frame"), c.frame);
    assert.equal(board(state, "arrows"), c.arrows);
    const { state: next, event } = step(state, c.action);
    assert.equal(event, c.event);
    assert.equal(target(next, "frame"), c.next_frame);
    assert.equal(target(next, "arrows"), c.next_arrows);
  }
});

test("an arrows board reads back to the same snake", () => {
  for (const c of cases) assert.deepEqual(parseArrows(c.arrows), { snake: c.snake, food: c.food });
  assert.equal(parseArrows(cases[0].arrows.replace("H", ".")), null);
  assert.equal(parseArrows("nonsense"), null);
});

test("a frame board reads back to a snake that draws the same picture", () => {
  for (const c of cases) {
    const read = parseFrame(c.frame, c.snake);
    assert.equal(board(read, "frame"), c.frame);
    assert.deepEqual(read.snake, c.snake); // given the true order as a hint, it finds it
  }
});

test("the cells that could be the new tail match Python", () => {
  for (const c of cases) {
    const got = newTailCandidates({ snake: c.snake, food: c.food }).map((p) => [...p]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    assert.deepEqual(got, c.candidates);
  }
});

test("the opening position and the prompt format", () => {
  const s = start();
  assert.deepEqual(s.snake, [[8, 8], [8, 7], [8, 6]]);
  assert.deepEqual(s.food, [13, 15]);
  assert.equal(prompt(s, "U", "frame").length, 282);
  assert.equal(target(null, "arrows"), "X\n$\n");
});
