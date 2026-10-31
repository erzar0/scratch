"""DeepSeek-V4-Flash skeleton allocates no weights."""

import pytest
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

from deepseek.v4_flash import build_model


def test_build_model_on_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DeepseekV4Config, "from_pretrained", lambda repo: DeepseekV4Config()
    )

    model = build_model()

    assert isinstance(model, DeepseekV4ForCausalLM)
    assert {p.device.type for p in model.parameters()} == {"meta"}
