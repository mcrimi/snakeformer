// The rules of Snake, and the two ways of writing a board as text.
// A port of src/snakeformer/engine.py, encoding.py and ambiguity.py, tested
// against cases exported from Python.

export const SIZE = 16;
export const ACTIONS = "UDLR";
export const DELTA = { U: [-1, 0], D: [1, 0], L: [0, -1], R: [0, 1] };
const ARROW = { U: "^", D: "v", L: "<", R: ">" };
const TAIL = { U: "u", D: "d", L: "l", R: "r" };
const same = (a, b) => a[0] === b[0] && a[1] === b[1];
const has = (cells, cell) => cells.some((c) => same(c, cell));

export function spawnFood(snake) {
  let r = (snake[0][0] + 5) % SIZE;
  let c = (snake[0][1] + 7) % SIZE;
  const first = [r, c];
  while (has(snake, [r, c])) {
    c = (c + 1) % SIZE;
    if (c === 0) r = (r + 1) % SIZE;
    if (r === first[0] && c === first[1]) return null;
  }
  return [r, c];
}

export function start() {
  const mid = SIZE / 2;
  const snake = [[mid, mid], [mid, mid - 1], [mid, mid - 2]];
  return { snake, food: spawnFood(snake) };
}

/** One move: { state, event }. state is null on death. */
export function step(state, action) {
  const head = [state.snake[0][0] + DELTA[action][0], state.snake[0][1] + DELTA[action][1]];
  if (head[0] < 0 || head[0] >= SIZE || head[1] < 0 || head[1] >= SIZE) return { state: null, event: "wall" };
  if (has(state.snake, head)) return { state: null, event: "self" };
  if (state.food && same(head, state.food)) {
    const snake = [head, ...state.snake];
    return { state: { snake, food: spawnFood(snake) }, event: "eat" };
  }
  return { state: { snake: [head, ...state.snake.slice(0, -1)], food: state.food }, event: "move" };
}

const towards = (a, b) => ACTIONS[[..."UDLR"].findIndex((k) => DELTA[k][0] === b[0] - a[0] && DELTA[k][1] === b[1] - a[1])];

export function board(state, encoding) {
  const grid = Array.from({ length: SIZE }, () => Array(SIZE).fill("."));
  if (state.food) grid[state.food[0]][state.food[1]] = "F";
  const last = state.snake.length - 1;
  state.snake.forEach(([r, c], i) => {
    if (i === 0) grid[r][c] = "H";
    else if (encoding === "frame") grid[r][c] = i === last ? "#" : "O";
    else {
      const d = towards([r, c], state.snake[i - 1]);
      grid[r][c] = i === last ? TAIL[d] : ARROW[d];
    }
  });
  return grid.map((row) => row.join("")).join("\n");
}

export const prompt = (state, action, encoding) => `B:\n${board(state, encoding)}\nA:${action}\nT:\n`;
export const target = (next, encoding) => `${next ? board(next, encoding) : "X"}\n$\n`;

/** Cells of a board text as a 16 x 16 array of characters, or null if it is not that shape. */
export function cells(text) {
  const rows = text.split("\n");
  if (rows.length !== SIZE || rows.some((r) => r.length !== SIZE)) return null;
  return rows.map((r) => [...r]);
}

/** Read an arrows board back into a state; null if it is not a legal picture. */
export function parseArrows(text) {
  const grid = cells(text);
  if (!grid) return null;
  let head = null, food = null, tail = null, count = 0;
  const points = new Map();
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const ch = grid[r][c];
      if (ch === "H") { if (head) return null; head = [r, c]; }
      else if (ch === "F") { if (food) return null; food = [r, c]; }
      else if ("^v<>udlr".includes(ch)) {
        const k = "^v<>".includes(ch) ? "UDLR"["^v<>".indexOf(ch)] : ch.toUpperCase();
        points.set(r * SIZE + c, [r + DELTA[k][0], c + DELTA[k][1]]);
        count++;
        if ("udlr".includes(ch)) { if (tail) return null; tail = [r, c]; }
      } else if (ch !== ".") return null;
    }
  }
  if (!head || !tail) return null;
  const snake = [tail];
  let cell = tail;
  while (!same(cell, head)) {
    cell = points.get(cell[0] * SIZE + cell[1]);
    if (!cell || has(snake, cell) || (!same(cell, head) && !points.has(cell[0] * SIZE + cell[1]))) return null;
    snake.push(cell);
  }
  if (snake.length !== count + 1) return null;
  return { snake: snake.reverse(), food };
}

/**
 * Read a frame board. The picture does not fix the order of the body, so the
 * caller says which order to prefer: the one whose cells follow `like` (the
 * previous snake, moved) as far as possible. Returns null if no order fits.
 */
export function parseFrame(text, like = null) {
  const grid = cells(text);
  if (!grid) return null;
  let head = null, food = null, tail = null;
  const body = [];
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const ch = grid[r][c];
      if (ch === "H") { if (head) return null; head = [r, c]; }
      else if (ch === "F") { if (food) return null; food = [r, c]; }
      else if (ch === "#") { if (tail) return null; tail = [r, c]; }
      else if (ch === "O") body.push([r, c]);
      else if (ch !== ".") return null;
    }
  }
  if (!head || !tail) return null;
  const prefer = like ? new Map(like.map((cell, i) => [cell[0] * SIZE + cell[1], i])) : null;
  const used = new Set();
  let budget = 50000;
  const walk = (path) => {
    if (budget-- < 0) return null;
    const at = path[path.length - 1];
    if (path.length === body.length + 1) {
      return Math.abs(at[0] - tail[0]) + Math.abs(at[1] - tail[1]) === 1 ? [...path, tail] : null;
    }
    const next = body.filter((b) => !used.has(b[0] * SIZE + b[1]) && Math.abs(b[0] - at[0]) + Math.abs(b[1] - at[1]) === 1);
    if (prefer) next.sort((a, b) => Math.abs((prefer.get(a[0] * SIZE + a[1]) ?? 1e9) - path.length) - Math.abs((prefer.get(b[0] * SIZE + b[1]) ?? 1e9) - path.length));
    for (const b of next) {
      used.add(b[0] * SIZE + b[1]);
      const found = walk([...path, b]);
      used.delete(b[0] * SIZE + b[1]);
      if (found) return found;
    }
    return null;
  };
  const snake = walk([head]);
  return snake ? { snake, food } : null;
}

/** Body cells that could be the new tail after a move, judging by a frame board alone. */
export function newTailCandidates(state, budget = 200000) {
  const snake = state.snake;
  if (snake.length < 4) return [snake[snake.length - 2]];
  const head = snake[0], tail = snake[snake.length - 1];
  const body = new Set(snake.slice(1, -1).map((c) => c[0] * SIZE + c[1]));
  const around = ([r, c]) => [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]].filter(([y, x]) => y >= 0 && x >= 0 && y < SIZE && x < SIZE);
  const candidates = around(tail).filter((p) => body.has(p[0] * SIZE + p[1]));
  if (candidates.length === 1) return candidates;
  let left = budget;
  const reaches = (cell, seen, goal) => {
    if (--left < 0) return null;
    if (seen.size === body.size) return same(cell, goal);
    if (same(cell, goal)) return false;
    for (const q of around(cell)) {
      const key = q[0] * SIZE + q[1];
      if (body.has(key) && !seen.has(key)) {
        seen.add(key);
        const found = reaches(q, seen, goal);
        seen.delete(key);
        if (found || found === null) return found;
      }
    }
    return false;
  };
  return candidates.filter((goal) => around(head).some((q) => {
    const key = q[0] * SIZE + q[1];
    if (!body.has(key)) return false;
    const found = reaches(q, new Set([key]), goal);
    return found || found === null;
  }));
}
