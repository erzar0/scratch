"""qwen/: CSA2 in a tiny random Qwen3, chunked losses, answer scoring."""

import random

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from deepseek import csa2, reference
from experiments.selector_forward import selecting
from qwen import evaluate, probe, tasks, train
from qwen import model as qm

TINY = Qwen3Config(
    vocab_size=64,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    use_cache=False,
)
SIZES = {
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "index_topk": 4,
    "sliding_window": 8,
}


@pytest.fixture
def csa2_qwen() -> Qwen3ForCausalLM:
    """A tiny random Qwen3 with CSA2 attention."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(TINY).eval()
    return qm.convert(model, qm.csa2_config(TINY, **SIZES))


def test_csa2_qwen_is_causal(csa2_qwen: Qwen3ForCausalLM) -> None:
    # Arrange
    tokens = torch.randint(0, 64, (1, 48))
    changed = tokens.clone()
    changed[0, 30] = (tokens[0, 30] + 1) % 64

    # Act
    with torch.no_grad():
        before, after = (csa2_qwen(input_ids=t).logits for t in (tokens, changed))

    # Assert
    torch.testing.assert_close(before[:, :30], after[:, :30])
    assert not torch.allclose(before[:, 30:], after[:, 30:])


@pytest.mark.parametrize(
    "selector", [evaluate.own_indexer, reference.Oracle(4)], ids=["indexer", "oracle"]
)
def test_chunked_selector_picks_like_the_selector(
    csa2_qwen: Qwen3ForCausalLM, selector: csa2.Selector
) -> None:
    # Arrange
    tokens = torch.randint(0, 64, (2, 50))

    # Act
    with torch.no_grad():
        with selecting(csa2_qwen, selector):
            whole = csa2_qwen(input_ids=tokens).logits
        with selecting(csa2_qwen, evaluate.Chunked(selector, 7)):
            chunked = csa2_qwen(input_ids=tokens).logits

    # Assert
    torch.testing.assert_close(chunked, whole)


def test_chunked_losses_equal_full_logits(
    csa2_qwen: Qwen3ForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(train, "CHUNK", 8)
    tokens = torch.randint(0, 64, (2, 41))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    # Act
    with torch.no_grad():
        losses = train.lm_losses(csa2_qwen, tokens)
        log_p = csa2_qwen(input_ids=inputs).logits.log_softmax(-1)
        with qm.teacher(csa2_qwen):
            log_q = csa2_qwen(input_ids=inputs).logits.log_softmax(-1)
    ce = F.nll_loss(log_p.flatten(0, 1), targets.flatten())
    kl = (log_q.exp() * (log_q - log_p)).sum(-1).mean()

    # Assert
    torch.testing.assert_close(losses, torch.stack([ce, kl]))


def test_greedy_matches_equals_greedy_decoding() -> None:
    # Arrange
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(TINY).eval()
    prompt = torch.randint(0, 64, (1, 20))
    generated = model.generate(prompt, max_new_tokens=5, do_sample=False)
    answer = generated[0, 20:].tolist()
    wrong = [*answer[:-1], (answer[-1] + 1) % 64]

    # Act
    right_match = evaluate.greedy_matches(model, prompt[0].tolist(), answer)
    wrong_match = evaluate.greedy_matches(model, prompt[0].tolist(), wrong)

    # Assert
    assert right_match
    assert not wrong_match


def test_balanced_draws_rounds_of_every_label() -> None:
    # Arrange
    items = [{"label": i % 3, "id": i} for i in range(7)]  # label 2: two items

    # Act
    drawn = tasks.balanced(items, "label", 9, random.Random(0))

    # Assert
    for start in range(0, 9, 3):
        assert sorted(d["label"] for d in drawn[start : start + 3]) == [0, 1, 2]


def test_probe_mask_is_causal_window_and_sinks() -> None:
    seen = probe.allowed(300, 4, torch.device("cpu"))
    q, k = torch.meshgrid(torch.arange(300), torch.arange(300), indexing="ij")
    assert not seen[k > q].any()
    assert seen[(k <= q) & (q - k < probe.WINDOW)].all()
    assert seen[:, :4][torch.arange(300)[:, None] >= torch.arange(4)].all()
    assert not seen[299, 4 : 300 - probe.WINDOW].any()
