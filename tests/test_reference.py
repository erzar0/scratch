"""deepseek/reference.py: the oracle against a per-query loop; random picks are
finished and distinct."""

import torch

from deepseek import csa2
from deepseek.reference import Oracle, RandomEntries
from tests.test_csa2 import MODES
from tests.test_double_indexing import CONFIG, M, positive_inputs


def test_oracle_picks_the_heaviest_entries() -> None:
    # Arrange
    torch.manual_seed(0)
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    attention.sinks.data.normal_()
    inputs = positive_inputs(attention)

    # Act
    picks = Oracle(3)(attention, inputs)

    # Assert: per query, softmax over [finished entries ; sink] per head, head sum
    for b in range(inputs.query.shape[0]):
        for t in range(inputs.query.shape[2]):
            finished = (t + 1) // M
            logits = inputs.query[b, :, t] @ inputs.main_kv[b, :finished].T
            logits = torch.cat(
                [logits * attention.scaling, attention.sinks[:, None]], 1
            )
            weight = logits.softmax(-1)[:, :-1].sum(0)
            want = set(weight.argsort(descending=True)[:3].tolist())
            assert set(picks.entry[b, t][picks.valid[b, t]].tolist()) == want, (b, t)


def test_random_picks_are_finished_and_distinct() -> None:
    # Arrange
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    inputs = positive_inputs(attention)

    # Act
    picks = RandomEntries(3)(attention, inputs)

    # Assert
    assert picks.entry.shape == (2, 40, 3)  # batch 2, though position_ids have 1
    finished = csa2.finished_entries(inputs.position_ids, M)
    n_valid = picks.valid.sum(-1)
    assert torch.equal(n_valid, finished.squeeze(-1).clamp(max=3).expand_as(n_valid))
    assert bool((picks.entry < finished)[picks.valid].all())
    assert all(len(set(row.tolist())) == 3 for row in picks.entry.flatten(0, 1))
