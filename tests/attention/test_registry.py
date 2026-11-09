"""The registry (src/attention/registry.py): every attention module is used, every
version builds at every suite budget, and each version's fast step picks the keys
of its emulation, at budgets 128 and 512 (and 2048 on the GPU). Triton cases need
CUDA:
    uv run python -m tests.attention.test_registry
"""

import sys
import tomllib
from pathlib import Path

import pytest
import torch

from attention import harness
from attention.fast_pytorch import KV
from attention.registry import KEEP, VERSIONS, versions


def test_every_attention_module_is_used() -> None:
    files = {
        ".".join(p.with_suffix("").parts[1:])
        for p in Path("src/attention").rglob("*.py")
        if p.stem != "__init__"
    }
    assert files <= set(sys.modules), files - set(sys.modules)


def picked(won: torch.Tensor, n: int) -> torch.Tensor:
    """(G, n) mask of the winner positions won (G, M) holds; -1 is none."""
    at = torch.where(won < 0, n, won).long()
    mask = torch.zeros(won.shape[0], n + 1, dtype=torch.bool, device=won.device)
    return mask.scatter_(1, at, True)[:, :n]


def test_every_version_builds_at_every_suite_budget() -> None:
    budgets = {
        keep
        for suite in Path("src/benchmarks/suites").glob("*.toml")
        for benchmark in tomllib.loads(suite.read_text()).values()
        if isinstance(benchmark, dict)
        for keep in benchmark.get("budgets", [KEEP])
    }
    for keep in budgets:
        built = versions(keep)
        assert built.keys() == VERSIONS.keys()
        for version in list(built.values())[1:]:  # dense has no budget
            assert keep in version.settings.values(), (keep, version.id)


SIZES = [(128, 2048), (KEEP, 8192)]  # (budget, length): 4 rounds, the global
# kernel needs more than 3


@pytest.mark.parametrize("keep, n", SIZES)
@pytest.mark.parametrize("id_", VERSIONS)
def test_fast_step_picks_the_emulation_keys(id_: str, keep: int, n: int) -> None:
    version = versions(keep)[id_]
    if version.emulation is None:
        pytest.skip("latency only")
    device = "cuda" if version.implementation == "Triton" else "cpu"
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    scaling = 128**-0.5
    q = torch.randn(2, 2, 128, device=device)
    k, v = torch.randn(2, 2, n, 128, device=device)
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device=device)
    want = version.emulation(
        harness.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    )

    out, won = version.fast(q, KV(k, v))

    sampled = "sampled" in version.id  # fp16 partials, hashed slots
    if won is not None:
        got, exact = picked(won, n), want.selected[:, 0]
        share = (got & exact).sum() / exact.sum()
        assert share >= 0.99 if sampled else torch.equal(got, exact)
    atol = 0.05 if sampled else 1e-4
    torch.testing.assert_close(out, want.output[:, :, 0], atol=atol, rtol=0)


if __name__ == "__main__":
    for keep, n in [*SIZES, (2048, 32768)]:
        for id_ in (v for v, version in VERSIONS.items() if version.emulation):
            test_fast_step_picks_the_emulation_keys(id_, keep, n)
            print(f"budget {keep}, {n} keys: {id_} matches")
