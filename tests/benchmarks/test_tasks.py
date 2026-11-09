"""CPU smoke run of the tasks benchmark (src/benchmarks/tasks.py)."""

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from attention.registry import VERSIONS
from benchmarks import synthetic, tasks


@pytest.mark.parametrize("version", ["dense", "tournament/global-halving"])
def test_tasks_run_on_a_small_model(version: str) -> None:
    model = LlamaForCausalLM(
        LlamaConfig(**tasks.MODEL | {"num_hidden_layers": 2, "hidden_size": 128})
    )
    config = {"tasks": ["mqar"], "lengths": [1024], "seeds": [0], "sequences": 1}
    (row,) = tasks.run(VERSIONS[version], config, device="cpu", model=model)
    assert 0 <= row["accuracy"] <= 1 and row["n"] == 1024
    assert (row["reads"] == 1) == (version == "dense")
    assert synthetic.VOCAB == model.config.vocab_size
