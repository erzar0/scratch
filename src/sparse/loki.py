"""Loki (Singhania et al., arXiv 2406.02542): approximate scores in a PCA basis.

calibrate() fits, per layer and KV head, the principal axes of the keys of a few
WikiText-103 train windows, as long as the scored context so every rotary phase is
seen: of pre-rotary keys (k_norm's output) and of post-rotary ones (what attention
sees). Loki rotates the post-rotary q and k into either basis
(any orthogonal map keeps q k^T), scores every key on the first d axes, and gives
the top k exact attention. Reads per query and KV head: n d + 2 k d_head.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from transformers import AttentionInterface, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from sparse.attend import Chunk, Outcome, attend, group_score, top_mask

BASIS = Path("runs/sparse/loki_pca.pt")


@torch.no_grad()
def calibrate(model: Qwen3ForCausalLM, windows: torch.Tensor) -> None:
    """Write the (L, G, D, D) PCA bases, axes by variance, to BASIS.

    Args:
        model: The Qwen3, in its usual attention.
        windows: (W, T) token ids, one forward each.
    """
    config = model.config
    assert config.num_key_value_heads is not None
    shape = (config.num_hidden_layers, config.num_key_value_heads, config.head_dim)
    moments = {
        kind: torch.zeros(*shape, shape[-1], dtype=torch.float64, device=model.device)
        for kind in ("pre", "post")
    }

    def add(kind: str, layer: int, keys: torch.Tensor) -> None:
        flat = keys.reshape(-1, *shape[1:]).double()
        moments[kind][layer] += torch.einsum("ngd,nge->gde", flat, flat)

    def post(
        module: Qwen3Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        add("post", module.layer_idx, key.transpose(1, 2))
        return sdpa_attention_forward(module, query, key, *args, **kwargs)  # type: ignore[arg-type]

    hooks = [
        layer.self_attn.k_norm.register_forward_hook(
            lambda _m, _i, out, i=i: add("pre", i, out)
        )
        for i, layer in enumerate(model.model.layers)
    ]
    previous = config._attn_implementation
    AttentionInterface.register("loki_calibrate", post)
    model.set_attn_implementation("loki_calibrate")
    try:
        for window in windows:
            model.model(input_ids=window[None])
    finally:
        model.set_attn_implementation(previous)
        for hook in hooks:
            hook.remove()
    # uncentered second moments: the axes that best reconstruct the keys (Lemma 4.2)
    bases = {
        kind: torch.linalg.eigh(m)[1].flip(-1).float() for kind, m in moments.items()
    }
    BASIS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({kind: b.cpu() for kind, b in bases.items()}, BASIS)


@cache
def basis(keys: str, device: str) -> torch.Tensor:
    """(L, G, D, D) PCA basis fitted on pre- or post-rotary keys, on device."""
    return torch.load(BASIS)[keys].to(device)


@dataclass(frozen=True)
class Loki:
    """Loki attention.

    Attributes:
        keys: Basis fitted on "pre"- or "post"-rotary keys.
        d: PCA axes of the approximate scores.
        k: Keys that get exact attention.
    """

    keys: str = "pre"
    d: int = 32
    k: int = 128

    def __call__(self, chunk: Chunk) -> Outcome:
        """Loki on one chunk."""
        axes = basis(self.keys, str(chunk.key.device))[chunk.layer][..., : self.d]
        q = chunk.query @ axes[:, None]
        k = chunk.key @ axes
        approx = (q @ k[:, None].mT) * chunk.scaling
        score = group_score(approx.masked_fill(~chunk.valid, float("-inf")))
        selected = top_mask(score, self.k, chunk.valid)
        dim = chunk.query.shape[-1]
        reads = chunk.valid.sum(-1) * self.d + 2 * dim * selected.sum(-1)
        return Outcome(attend(chunk, selected), selected, reads)
