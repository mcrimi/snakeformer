// One turn of the game as the model plays it: read the prompt, write the next
// frame a token at a time, and compare the result with the rules.

import { ACTIONS, DELTA, cells, parseArrows, parseFrame, prompt, step, target } from "./game.js";
import { argmax, softmax } from "./model.js";

export const PROMPT_LENGTH = 282;
const ROW = 17; // sixteen cells and a newline

/** Position in the sequence of the token for cell (r, c) of the input board, or of the output board. */
export const inputPosition = (r, c) => 3 + r * ROW + c;
export const outputPosition = (r, c) => PROMPT_LENGTH + r * ROW + c;
export const ACTION_POSITION = PROMPT_LENGTH - 5;

export class Turn {
  /** sample: draw each token from the model's distribution instead of taking the likeliest. */
  constructor(model, state, action, { sample = false, random = Math.random } = {}) {
    this.model = model;
    this.state = state;
    this.action = action;
    this.sample = sample;
    this.random = random;
    this.prompt = prompt(state, action, model.encoding);
    this.promptIds = model.encode(this.prompt);
    this.session = model.session({ keepQueries: true });
    this.fed = 0;
    this.ids = [];
    this.done = false;
    this.logits = null;
    const real = step(state, action);
    this.event = real.event;
    this.truthState = real.state;
    this.truth = target(real.state, model.encoding).slice(0, -1); // up to and including "$"
  }

  get reading() { return this.fed < this.promptIds.length; }
  get text() { return this.model.decode(this.ids); }

  /** Process up to n tokens. Returns how many were processed. */
  advance(n = 1) {
    const stop = this.model.stoi["$"];
    let did = 0;
    while (did < n && !this.done) {
      if (this.reading) {
        this.logits = this.model.step(this.session, this.promptIds[this.fed++]);
      } else {
        const id = this.sample ? draw(softmax(this.logits), this.random) : argmax(this.logits);
        this.ids.push(id);
        if (id === stop || this.ids.length >= 300 || this.session.length >= this.model.config.block_size - 1) this.done = true;
        else this.logits = this.model.step(this.session, id);
      }
      did++;
    }
    return did;
  }

  /** The frame the model wrote, without the stop token. */
  get frame() { return this.text.replace(/\n?\$?$/, ""); }

  /**
   * How the finished turn compares with the rules.
   *   right:  the frame is exactly the true one
   *   says:   "dead", "board" or "nonsense"
   *   wrong:  cells where a board differs from the true board, as "r,c"
   *   next:   the state the model's frame describes (null if dead or unreadable)
   */
  verdict() {
    const frame = this.frame;
    const right = this.text === this.truth;
    if (frame === "X") return { right, says: "dead", wrong: [], next: null };
    const grid = cells(frame);
    if (!grid) return { right, says: "nonsense", wrong: [], next: null };
    const wrong = [];
    const truthGrid = this.truthState ? cells(this.truth.slice(0, -2)) : null;
    if (truthGrid) {
      for (let r = 0; r < grid.length; r++) for (let c = 0; c < grid.length; c++) if (grid[r][c] !== truthGrid[r][c]) wrong.push(`${r},${c}`);
    }
    const next = this.model.encoding === "arrows"
      ? parseArrows(frame)
      : parseFrame(frame, this.truthState ? this.truthState.snake : null);
    return { right, says: next ? "board" : "nonsense", wrong, next };
  }

  /**
   * Where the token for output cell (r, c) looked: a weight for each cell of
   * the input board and of the output board written so far, plus the action.
   * Averaged over heads; with layer = -1, the largest value over the layers.
   */
  attention(r, c, layer = -1) {
    const position = outputPosition(r, c) - 1; // the token that predicts this cell
    if (position >= this.session.length) return null;
    const layers = layer < 0 ? this.model.layers.map((_, i) => i) : [layer];
    const total = new Float32Array(position + 1);
    for (const l of layers) {
      const heads = this.model.attention(this.session, l, position);
      for (let j = 0; j <= position; j++) {
        let mean = 0;
        for (const h of heads) mean += h[j];
        mean /= heads.length;
        if (mean > total[j]) total[j] = mean;
      }
    }
    const input = new Float32Array(256);
    const output = new Float32Array(256);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        input[y * 16 + x] = total[inputPosition(y, x)] ?? 0;
        const o = outputPosition(y, x);
        output[y * 16 + x] = o <= position ? total[o] : 0;
      }
    }
    let top = 0;
    for (const v of input) top = Math.max(top, v);
    for (const v of output) top = Math.max(top, v);
    const action = total[ACTION_POSITION];
    top = Math.max(top, action) || 1;
    return { input: input.map((v) => v / top), output: output.map((v) => v / top), action: action / top };
  }
}

function draw(probabilities, random) {
  let u = random();
  for (let i = 0; i < probabilities.length; i++) {
    u -= probabilities[i];
    if (u <= 0) return i;
  }
  return probabilities.length - 1;
}

/** The direction the snake is travelling in. */
export function heading(state) {
  const [a, b] = state.snake;
  return ACTIONS[[...ACTIONS].findIndex((k) => DELTA[k][0] === a[0] - b[0] && DELTA[k][1] === a[1] - b[1])];
}

export const isReversal = (state, action) => {
  const [h, neck] = state.snake;
  return h[0] + DELTA[action][0] === neck[0] && h[1] + DELTA[action][1] === neck[1];
};

/** A simple player: step towards the food without dying, if there is a way. */
export function autopilot(state, random = Math.random) {
  const options = [...ACTIONS].filter((a) => !isReversal(state, a));
  const alive = options.filter((a) => step(state, a).state);
  if (!alive.length) return options[0];
  if (!state.food || random() < 0.04) return alive[Math.floor(random() * alive.length)];
  const distance = (a) => {
    const head = step(state, a).state.snake[0];
    return Math.abs(head[0] - state.food[0]) + Math.abs(head[1] - state.food[1]);
  };
  return alive.reduce((best, a) => (distance(a) < distance(best) ? a : best));
}
