"""Synthetic inputs (src/benchmarks/synthetic.py)."""

import pytest
import torch
from torch.nn.functional import cosine_similarity

from benchmarks import synthetic


@pytest.mark.parametrize("distribution", synthetic.DISTRIBUTIONS)
def test_qkv_is_seeded(distribution: str) -> None:
    a = synthetic.qkv(distribution, 512, 4, seed=0, device="cpu")
    b = synthetic.qkv(distribution, 512, 4, seed=0, device="cpu")
    c = synthetic.qkv(distribution, 512, 4, seed=1, device="cpu")
    assert [t.shape for t in a] == [(8, 2, 4, 128), (8, 512, 128), (8, 512, 128)]
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not torch.equal(a[1], c[1])


def test_planted_needles_get_the_top_scores() -> None:
    q, k, _ = synthetic.qkv("llm-like", 4096, 8, seed=0, device="cpu")
    group = q.sum(1)  # (G, Q, D)
    score = (group @ k.mT)[..., synthetic.SINKS :]  # past the sinks
    top = k[:, synthetic.SINKS :][torch.arange(8)[:, None], score.argmax(-1)]
    cosine = torch.nn.functional.cosine_similarity(top, group, dim=-1)
    assert bool((cosine > 0.99).all())


def test_flat_and_sharp_only_scale_q() -> None:
    q, k, v = synthetic.qkv("llm-like", 512, 4, seed=0, device="cpu")
    for name, scale in (("flat", 0.5), ("sharp", 2.0)):
        q2, k2, v2 = synthetic.qkv(name, 512, 4, seed=0, device="cpu")
        assert torch.equal(q2, q * scale) and torch.equal(k2, k) and torch.equal(v2, v)


def test_local_neighbours_look_alike() -> None:
    def neighbours(distribution: str) -> float:
        _, k, _ = synthetic.qkv(distribution, 4096, 4, seed=0, device="cpu")
        k = k[:, synthetic.SINKS :]
        return float(cosine_similarity(k[:, 1:], k[:, :-1], dim=-1).mean())

    assert neighbours("local") > 0.8 and abs(neighbours("llm-like")) < 0.1


def test_local_needles_are_runs() -> None:
    def neighbours_in_top(distribution: str) -> float:
        """Share of each query's top keys whose left or right neighbour is also top."""
        q, k, _ = synthetic.qkv(distribution, 8192, 4, seed=0, device="cpu")
        score = (q.sum(1) @ k.mT)[..., synthetic.SINKS :]
        top = torch.zeros_like(score, dtype=torch.bool)
        top.scatter_(
            -1, score.topk(synthetic.NEEDLES * synthetic.LOCAL, -1).indices, True
        )
        pad = torch.nn.functional.pad(top, (1, 1))
        return float((top & (pad[..., :-2] | pad[..., 2:])).sum() / top.sum())

    assert neighbours_in_top("local") > 0.9 and neighbours_in_top("llm-like") < 0.3


def test_many_needles_get_the_top_scores() -> None:
    q, k, _ = synthetic.qkv("many-needles", 8192, 1, seed=0, device="cpu")
    group = q.sum(1)  # (G, 1, D)
    keys = k[:, synthetic.SINKS :]
    needle = cosine_similarity(keys, group, dim=-1) > 0.99  # (G, n - SINKS)
    count = needle.sum(-1)  # planted at random places, so a few may coincide
    assert bool((count > 0.9 * synthetic.MANY_NEEDLES).all())
    score = (group @ keys.mT)[:, 0]
    for g in range(len(keys)):
        assert bool(needle[g, score[g].topk(int(count[g])).indices].all())


def test_task_answers() -> None:
    gen = torch.Generator().manual_seed(0)
    ids, answer = synthetic.sequence("copy", 1000, gen)
    assert torch.equal(ids[:500], ids[500:]) and answer.sum() == 500
    for task, asked in (
        ("mqar", synthetic.QUERIES),
        ("needle", synthetic.NEEDLE_PAIRS),
    ):
        ids, answer = synthetic.sequence(task, 2048, gen)
        at = answer.nonzero()[:, 0]
        assert at.tolist() == list(range(2048 - 2 * asked + 1, 2048, 2))
        for key, value in zip(ids[at - 1], ids[at], strict=True):
            first = int((ids == key).nonzero()[0, 0])
            assert first < 2048 - 2 * asked and ids[first + 1] == value
