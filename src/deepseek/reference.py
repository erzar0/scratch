"""Reference selectors for CSA2's select step: random entries and the attention oracle.

    Changes  the select step of Full and Reindex layers: top_k entries chosen
             without the indexer
    Keeps    everything else
    Reads    random: none. Oracle: every finished entry at main-attention cost
    Trained  nothing

Bounds, not methods. Random entries show what picking blindly costs. The oracle picks
the top_k entries the main attention would weight most if it read every entry: no
cheap selector can match the attention better. Both are csa2.Selectors.
"""

from dataclasses import dataclass

import torch

from deepseek import csa2


@dataclass(frozen=True)
class RandomEntries:
    """top_k uniformly random finished entries per query.

    Attributes:
        top_k: Entries per query.
        seed: Seed of the draws; each call draws afresh from it.
    """

    top_k: int
    seed: int = 0

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """Random scores, then csa2.top_picks."""
        assert attention.indexer is not None
        batch, seq_len = inputs.hidden_states.shape[:2]  # position_ids: batch 1
        generator = torch.Generator(inputs.index_k.device).manual_seed(self.seed)
        scores = torch.rand(
            (batch, seq_len, inputs.index_k.shape[1]),
            generator=generator,
            device=inputs.index_k.device,
        )
        m = attention.indexer.compress_rate
        return csa2.top_picks(scores, inputs.position_ids, m, self.top_k)


@dataclass(frozen=True)
class Oracle:
    """The top_k entries by attention weight if attention read every finished entry.

    Per head: softmax of the main query's logits over [every finished main KV entry ;
    sink]; the weights are summed over heads.

    Attributes:
        top_k: Entries per query.
    """

    top_k: int

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """Attention weights as scores, then csa2.top_picks."""
        assert attention.indexer is not None
        m = attention.indexer.compress_rate
        weight = attention_weights(attention, inputs)
        return csa2.top_picks(weight, inputs.position_ids, m, self.top_k)


@torch.no_grad()
def attention_weights(
    attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
) -> torch.Tensor:
    """(B, S, n) main-attention weight on every finished entry, summed over heads,
    if the attention read them all (softmax with the sink)."""
    assert attention.indexer is not None
    m = attention.indexer.compress_rate
    logits = torch.einsum(
        "bhsd,bnd->bhsn", inputs.query.float(), inputs.main_kv.float()
    )
    logits = logits * attention.scaling
    future = torch.arange(logits.shape[-1], device=logits.device) >= (
        csa2.finished_entries(inputs.position_ids, m).unsqueeze(1)
    )
    logits = logits.masked_fill(future, float("-inf"))
    sinks = attention.sinks.float().view(1, -1, 1, 1)
    sinks = sinks.expand(*logits.shape[:3], 1)
    return torch.cat([logits, sinks], -1).softmax(-1)[..., :-1].sum(1)
