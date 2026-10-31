"""DeepSeek-V4-Flash built from its Hub config, with no weights allocated."""

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

REPO = "deepseek-ai/DeepSeek-V4-Flash"


def build_model(repo: str = REPO) -> DeepseekV4ForCausalLM:
    """Build the model skeleton on the meta device.

    Downloads only ``config.json``. Parameters keep real shapes and dtypes but have
    no storage, so the full model costs no RAM. Load weights later with
    ``from_pretrained`` or ``load_state_dict(..., assign=True)``.

    Args:
        repo: Hugging Face model id.

    Returns:
        The model with every parameter on the meta device.
    """
    config = DeepseekV4Config.from_pretrained(repo)
    with torch.device("meta"):
        return DeepseekV4ForCausalLM(config)


if __name__ == "__main__":
    model = build_model()
    print(model)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
