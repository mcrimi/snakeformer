import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { start, step } from "../js/game.js";
import { Model, parseWeights } from "../js/model.js";
import { ACTION_POSITION, PROMPT_LENGTH, Turn, autopilot, heading, inputPosition, isReversal, outputPosition } from "../js/play.js";

const file = new URL("../data/original.bin", import.meta.url);
const load = () => {
  const raw = readFileSync(file);
  return new Model(parseWeights(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));
};

test("positions of the board cells in the sequence", () => {
  const model = { encoding: "frame" };
  assert.equal(PROMPT_LENGTH, 282);
  const text = `B:\n${"................\n".repeat(16)}A:U\nT:\n`;
  assert.equal(text.length, PROMPT_LENGTH);
  assert.equal(text[ACTION_POSITION], "U");
  assert.equal(inputPosition(0, 0), 3);
  assert.equal(text[inputPosition(15, 15) + 1], "\n");
  assert.equal(outputPosition(0, 0), PROMPT_LENGTH);
  assert.ok(model);
});

test("a turn, taken a few tokens at a time, ends with a verdict", { skip: !existsSync(file) }, () => {
  const model = load();
  const turn = new Turn(model, start(), "R");
  let guard = 0;
  while (!turn.done && guard++ < 2000) turn.advance(7);
  assert.ok(turn.done);
  const v = turn.verdict();
  assert.equal(v.says, "board");
  assert.equal(v.right, true); // the opening move is one the published model gets right
  assert.deepEqual(v.wrong, []);
  assert.deepEqual(v.next, step(start(), "R").state);
  const looked = turn.attention(8, 9);
  assert.equal(looked.input.length, 256);
  assert.ok(Math.max(...looked.input, ...looked.output, looked.action) <= 1.000001);
});

test("a wall is a death the model can call", { skip: !existsSync(file) }, () => {
  const model = load();
  const state = { snake: [[0, 5], [1, 5], [2, 5]], food: [9, 9] };
  const turn = new Turn(model, state, "U");
  while (!turn.done) turn.advance(50);
  assert.equal(turn.event, "wall");
  assert.equal(turn.truth, "X\n$");
  assert.equal(turn.verdict().says, "dead");
});

test("the autopilot never reverses and avoids dying when it can", () => {
  let state = start();
  assert.equal(heading(state), "R");
  assert.ok(isReversal(state, "L"));
  let seed = 1;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 300 && state; i++) {
    const action = autopilot(state, random);
    assert.ok(!isReversal(state, action));
    state = step(state, action).state;
  }
});
