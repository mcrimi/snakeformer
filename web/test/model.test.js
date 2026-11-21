import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { Model, argmax, generate, parseWeights, softmax } from "../js/model.js";

const here = new URL("./", import.meta.url);
const names = readdirSync(here).filter((f) => /^fixtures-(?!game).*\.json$/.test(f)).map((f) => f.slice(9, -5));

for (const name of names) {
  const weights = new URL(`../data/${name}.bin`, import.meta.url);
  test(`${name}: JavaScript reproduces PyTorch's logits and frames`, { skip: !existsSync(weights) }, () => {
    const file = readFileSync(weights);
    const model = new Model(parseWeights(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)));
    const { cases, encoding } = JSON.parse(readFileSync(new URL(`fixtures-${name}.json`, here)));
    assert.equal(model.encoding, encoding);
    for (const c of cases) {
      const session = model.session({ keepQueries: true });
      const ids = model.encode(c.prompt + c.output);
      let worst = 0;
      ids.slice(0, -1).forEach((id, position) => {
        const logits = model.step(session, id);
        const want = c.logits[position];
        if (want) want.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - logits[i])); });
      });
      assert.ok(worst < 2e-3, `logits differ by ${worst}`);

      const fresh = model.session();
      fresh.last = model.feed(fresh, c.prompt);
      assert.equal(generate(model, fresh), c.output);

      // attention rows are distributions over the tokens seen so far
      const rows = model.attention(session, 0, c.prompt.length + 3);
      assert.equal(rows.length, model.config.n_head);
      for (const row of rows) assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-4);
    }
  });
}

test("argmax and softmax", () => {
  assert.equal(argmax([0.1, 3, -2]), 1);
  const p = softmax(new Float32Array([0, 0, Math.log(2)]));
  assert.ok(Math.abs(p[2] - 0.5) < 1e-6 && Math.abs(p[0] - 0.25) < 1e-6);
});
