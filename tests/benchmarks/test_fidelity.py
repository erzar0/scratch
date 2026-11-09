"""CPU smoke run of the fidelity benchmark (src/benchmarks/fidelity.py)."""

from attention.registry import VERSIONS
from benchmarks import fidelity


def test_fidelity_of_dense_is_exact_and_sparq_is_not() -> None:
    config = {
        "lengths": [1024],
        "budgets": [64],  # MIN_RATIO: 1024 keys run budgets up to 64
        "distributions": ["llm-like"],
        "queries": 4,
        "seeds": [0],
    }
    (dense,) = fidelity.run(VERSIONS["dense"], config, device="cpu")
    (sparq,) = fidelity.run(VERSIONS["sparq"], config, device="cpu")
    assert dense["error"] < 1e-5 and dense["recall"] == 1 and dense["reads"] == 1
    assert sparq["error"] > 1e-5 and sparq["reads"] < 1
