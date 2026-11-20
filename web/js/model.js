// The transformer, in plain JavaScript.
//
// The same arithmetic as src/snakeformer/model.py, one token at a time. Each
// layer keeps the keys and values of the tokens it has already seen, so a new
// token costs one pass over the context rather than a pass over everything
// again. A test replays cases exported from PyTorch and requires the same
// logits and the same generated frames.

function float16(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** Read a weights file written by snakeformer.export.write_weights. */
export function parseWeights(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const start = 4 + headerLength;
  const tensors = {};
  for (const t of header.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(size);
    const at = start + 2 * t.offset;
    for (let i = 0; i < size; i++) out[i] = float16(view.getUint16(at + 2 * i, true));
    tensors[t.name] = out;
  }
  return { ...header, tensors };
}

function layerNorm(x, weight, bias, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
  const scale = 1 / Math.sqrt(variance / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
}

/**
 * out = W x (+ b). W is rows x cols, stored row by row. Four running sums
 * instead of one: the additions no longer wait for each other, which is
 * worth about a factor of two in every browser tried.
 */
function linear(W, b, x, out, rows, cols, offset = 0) {
  const whole = cols - (cols % 4);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let c = 0; c < whole; c += 4) {
      s0 += W[base + c] * x[c];
      s1 += W[base + c + 1] * x[c + 1];
      s2 += W[base + c + 2] * x[c + 2];
      s3 += W[base + c + 3] * x[c + 3];
    }
    for (let c = whole; c < cols; c++) s0 += W[base + c] * x[c];
    out[offset + r] = s0 + s1 + s2 + s3 + (b ? b[r] : 0);
  }
}

export class Model {
  constructor(parsed) {
    this.config = parsed.config;
    this.chars = parsed.chars;
    this.encoding = parsed.encoding;
    this.about = parsed;
    this.stoi = Object.fromEntries(parsed.chars.map((ch, i) => [ch, i]));
    this.w = parsed.tensors;
    const { n_embd: d, n_layer } = this.config;
    this.layers = Array.from({ length: n_layer }, (_, i) => {
      const p = `blocks.${i}.`;
      const w = this.w;
      return {
        ln1w: w[p + "ln1.weight"], ln1b: w[p + "ln1.bias"],
        query: w[p + "query.weight"], key: w[p + "key.weight"], value: w[p + "value.weight"],
        projw: w[p + "proj.weight"], projb: w[p + "proj.bias"],
        ln2w: w[p + "ln2.weight"], ln2b: w[p + "ln2.bias"],
        fc1w: w[p + "fc1.weight"], fc1b: w[p + "fc1.bias"],
        fc2w: w[p + "fc2.weight"], fc2b: w[p + "fc2.bias"],
      };
    });
    this.scratch = {
      x: new Float32Array(d), norm: new Float32Array(d), q: new Float32Array(d),
      mix: new Float32Array(d), out: new Float32Array(d), hidden: new Float32Array(4 * d),
      scores: new Float32Array(this.config.block_size),
    };
  }

  encode(text) { return Array.from(text, (ch) => this.stoi[ch]); }
  decode(ids) { return ids.map((i) => this.chars[i]).join(""); }

  /** A fresh context. keepQueries: remember each token's queries, for attention maps. */
  session({ keepQueries = false } = {}) {
    const { n_embd: d, n_layer, block_size } = this.config;
    return {
      length: 0,
      ids: [],
      keys: Array.from({ length: n_layer }, () => new Float32Array(block_size * d)),
      values: Array.from({ length: n_layer }, () => new Float32Array(block_size * d)),
      queries: keepQueries ? Array.from({ length: n_layer }, () => new Float32Array(block_size * d)) : null,
    };
  }

  /** Feed one token; returns the logits for the next one. */
  step(session, id) {
    const { n_embd: d, n_head, vocab_size, block_size } = this.config;
    const t = session.length;
    if (t >= block_size) throw new Error("context is full");
    const { x, norm, q, mix, out, hidden, scores } = this.scratch;
    const headSize = d / n_head;
    const scale = 1 / Math.sqrt(d); // the original divides by the embedding width
    const tok = this.w["tok.weight"];
    const pos = this.w["pos.weight"];
    for (let i = 0; i < d; i++) x[i] = tok[id * d + i] + pos[t * d + i];

    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l];
      const K = session.keys[l];
      const V = session.values[l];
      layerNorm(x, L.ln1w, L.ln1b, norm);
      linear(L.query, null, norm, q, d, d);
      linear(L.key, null, norm, K, d, d, t * d);
      linear(L.value, null, norm, V, d, d, t * d);
      if (session.queries) session.queries[l].set(q, t * d);
      for (let h = 0; h < n_head; h++) {
        const lo = h * headSize;
        let top = -Infinity;
        for (let j = 0; j <= t; j++) {
          let s = 0;
          const base = j * d + lo;
          for (let k = 0; k < headSize; k++) s += q[lo + k] * K[base + k];
          s *= scale;
          scores[j] = s;
          if (s > top) top = s;
        }
        let total = 0;
        for (let j = 0; j <= t; j++) { scores[j] = Math.exp(scores[j] - top); total += scores[j]; }
        for (let k = 0; k < headSize; k++) mix[lo + k] = 0;
        for (let j = 0; j <= t; j++) {
          const weight = scores[j] / total;
          if (weight < 1e-9) continue;
          const base = j * d + lo;
          for (let k = 0; k < headSize; k++) mix[lo + k] += weight * V[base + k];
        }
      }
      linear(L.projw, L.projb, mix, out, d, d);
      for (let i = 0; i < d; i++) x[i] += out[i];
      layerNorm(x, L.ln2w, L.ln2b, norm);
      linear(L.fc1w, L.fc1b, norm, hidden, 4 * d, d);
      for (let i = 0; i < 4 * d; i++) if (hidden[i] < 0) hidden[i] = 0;
      linear(L.fc2w, L.fc2b, hidden, out, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += out[i];
    }
    layerNorm(x, this.w["ln_f.weight"], this.w["ln_f.bias"], norm);
    const logits = new Float32Array(vocab_size);
    linear(this.w["head.weight"], this.w["head.bias"], norm, logits, vocab_size, d);
    session.ids.push(id);
    session.length = t + 1;
    return logits;
  }

  /** Feed a string; returns the logits after its last character. */
  feed(session, text) {
    let logits = null;
    for (const id of this.encode(text)) logits = this.step(session, id);
    return logits;
  }

  /**
   * How much the token at `position` looked at each earlier token, in one layer:
   * an array over heads of Float32Array(position + 1). Needs keepQueries.
   */
  attention(session, layer, position) {
    const { n_embd: d, n_head } = this.config;
    const headSize = d / n_head;
    const scale = 1 / Math.sqrt(d);
    const Q = session.queries[layer];
    const K = session.keys[layer];
    return Array.from({ length: n_head }, (_, h) => {
      const lo = h * headSize;
      const weights = new Float32Array(position + 1);
      let top = -Infinity;
      for (let j = 0; j <= position; j++) {
        let s = 0;
        for (let k = 0; k < headSize; k++) s += Q[position * d + lo + k] * K[j * d + lo + k];
        weights[j] = s * scale;
        if (weights[j] > top) top = weights[j];
      }
      let total = 0;
      for (let j = 0; j <= position; j++) { weights[j] = Math.exp(weights[j] - top); total += weights[j]; }
      for (let j = 0; j <= position; j++) weights[j] /= total;
      return weights;
    });
  }
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function softmax(logits, temperature = 1) {
  const out = new Float32Array(logits.length);
  let top = -Infinity;
  for (const v of logits) if (v > top) top = v;
  let total = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp((logits[i] - top) / temperature); total += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

/** Generate greedily until the stop token. Returns the text produced. */
export function generate(model, session, limit = 300) {
  const stop = model.stoi["$"];
  let logits = session.last;
  const out = [];
  for (let i = 0; i < limit; i++) {
    const id = argmax(logits);
    out.push(id);
    logits = model.step(session, id);
    if (id === stop) break;
  }
  session.last = logits;
  return model.decode(out);
}
