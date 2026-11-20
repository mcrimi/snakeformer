"""The network, the trainer, the evaluator and the export for the browser."""

import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snakeformer import ARROWS, FRAME, start, step
from snakeformer.data import generate
from snakeformer.encoding import PROMPT_LEN
from snakeformer.evaluate import by_kind, exact, games, generated
from snakeformer.export import greedy, model_fixtures, rounded, write_weights
from snakeformer.model import GPT, Config, load, load_original, save
from snakeformer.train import tensors, train

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "model" / "weights" / "snake_model.pt"
SMALL = Config(vocab_size=len(ARROWS.chars), block_size=600, n_embd=32, n_head=4, n_layer=2)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return GPT(SMALL).eval()


def test_the_network_cannot_see_the_future(tiny):
    x = torch.randint(0, SMALL.vocab_size, (1, 40))
    y = x.clone()
    y[0, 30:] = (y[0, 30:] + 1) % SMALL.vocab_size
    a, b = tiny(x), tiny(y)
    assert torch.allclose(a[0, :30], b[0, :30], atol=1e-6)
    assert not torch.allclose(a[0, 30:], b[0, 30:], atol=1e-6)


def test_attention_weights_are_distributions_over_the_past(tiny):
    x = torch.randint(0, SMALL.vocab_size, (2, 25))
    with torch.no_grad():
        logits, weights = tiny(x, want_weights=True)
    assert logits.shape == (2, 25, SMALL.vocab_size)
    assert len(weights) == SMALL.n_layer and weights[0].shape == (2, SMALL.n_head, 25, 25)
    assert torch.allclose(weights[0].sum(-1), torch.ones(2, SMALL.n_head, 25), atol=1e-5)
    assert float(weights[0][0, 0].triu(1).abs().max()) == 0.0


def test_only_the_next_frame_is_scored():
    examples = generate(40, seed=0)
    ids, scored = tensors(examples, ARROWS)
    assert ids.shape == (40, 556) and not scored[:, :PROMPT_LEN].any()
    for ex, row in zip(examples, scored, strict=True):
        assert int(row.sum()) == len(ARROWS.target(ex.next))


@pytest.mark.parametrize("mode", ["aligned", "stream"])
def test_a_few_steps_of_training_reduce_the_loss(mode):
    examples = generate(200, seed=0)
    log = []
    model = train(examples, ARROWS, steps=30, mode=mode, batch=8, config=SMALL, log=log, every=1)
    assert log[-1]["loss"] < log[0]["loss"] * 0.8
    assert not model.training


def test_one_pass_scoring_agrees_with_generating(tiny):
    examples = generate(12, seed=3)
    model = train(examples, ARROWS, steps=250, batch=12, config=SMALL, lr=3e-3)
    right = exact(model, examples, ARROWS)
    written = generated(model, ARROWS, [ARROWS.prompt(e.state, e.action) for e in examples])
    assert right == [w == ARROWS.target(e.next)[:-1] for w, e in zip(written, examples, strict=True)]
    assert any(right)  # a dozen examples can be learned by heart
    shares = by_kind(examples, right)
    assert shares["all"]["n"] == 12 and 0 < shares["all"]["share"] <= 1
    model.cpu()
    first = examples[0]
    assert greedy(model, ARROWS, ARROWS.prompt(first.state, first.action)) == written[0]


def test_save_and_load(tmp_path, tiny):
    save(tiny, tmp_path / "m.pt", encoding="arrows")
    twin, extra = load(tmp_path / "m.pt")
    assert extra["encoding"] == "arrows"
    x = torch.randint(0, SMALL.vocab_size, (1, 20))
    assert torch.equal(tiny(x), twin(x))
    assert (tmp_path / "m.pt").stat().st_size < 4 * tiny.n_params() + 20_000  # no masks on disk


def test_weights_file_for_the_browser(tmp_path, tiny):
    twin = rounded(tiny)
    path = write_weights(twin, ARROWS, tmp_path / "w.bin", name="tiny")
    raw = path.read_bytes()
    (n,) = struct.unpack("<I", raw[:4])
    header = json.loads(raw[4 : 4 + n])
    assert header["encoding"] == "arrows" and header["name"] == "tiny"
    assert header["chars"] == ARROWS.chars and header["config"]["n_layer"] == 2
    total = sum(int(np.prod(t["shape"])) for t in header["tensors"])
    assert total == tiny.n_params() and len(raw) == 4 + n + 2 * total
    t = header["tensors"][0]
    got = np.frombuffer(raw, "<f2", count=int(np.prod(t["shape"])), offset=4 + n + 2 * t["offset"])
    assert np.array_equal(got.astype(np.float32), twin.state_dict()[t["name"]].numpy().ravel())
    cases = model_fixtures(twin, ARROWS, n=2)["cases"]
    assert all(c["prompt"].endswith("T:\n") and c["output"] for c in cases)


def test_games_are_played_by_the_rules():
    played = games(3, seed=1, max_steps=80)
    for moves in played:
        assert moves[0][0] == start()
        for state, action, nxt in moves:
            assert step(state, action)[0] == nxt
        assert all(nxt is not None for _, _, nxt in moves[:-1])


@pytest.mark.skipif(not WEIGHTS.exists(), reason="the published checkpoint is not downloaded")
def test_the_published_checkpoint_loads_and_gives_the_same_logits():
    sys.path.insert(0, str(ROOT))
    from model.gpt import GPT as Original
    from model.gpt import GPTConfig

    meta = pickle.loads((WEIGHTS.parent / "meta.pkl").read_bytes())
    dims = {k: meta[k] for k in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")}
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    assert "".join(meta["itos"][i] for i in range(meta["vocab_size"])) == "".join(FRAME.chars)
    old = Original(GPTConfig(**dims, dropout=0.0, device="cpu"))
    old.load_state_dict(state)
    new = load_original(state, Config(**dims))
    assert (dims["n_layer"], dims["n_embd"], new.n_params()) == (6, 192, 2_868_880)
    x = torch.tensor([FRAME.encode(FRAME.prompt(start(), "U") + FRAME.target(step(start(), "U")[0]))])
    with torch.no_grad():
        assert torch.allclose(old.eval()(x)[0], new.eval()(x), atol=1e-5)


def test_a_weights_file_reads_back_as_the_same_model(tmp_path, tiny):
    from snakeformer.export import read_weights

    twin = rounded(tiny)
    again, header = read_weights(write_weights(twin, ARROWS, tmp_path / "w.bin"))
    x = torch.randint(0, SMALL.vocab_size, (1, 30))
    with torch.no_grad():
        assert torch.equal(twin(x), again(x))
    assert header["encoding"] == "arrows"
