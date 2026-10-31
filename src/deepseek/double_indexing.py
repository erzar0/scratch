"""Double indexing for CSA2: a coarse step on superblocks before the indexer's top-k.

    Changes  the select step of Full and Reindex layers: the indexer scores only the
             entries of s kept superblocks (groups of S consecutive main KV entries)
             and of the query's own, unfinished superblock
    Keeps    everything else, including the indexer's weights, top_k and Reuse layers
    Reads    per query, n // S summary keys (2 per superblock for bound, n_keys for
             learned, every entry for oracle) + (s + 1) S entry keys, instead of n
    Trained  nothing, or the router for coarse="learned"
             (experiments/train_router.py)

    coarse step  score one summary key per complete superblock, keep the best s
    fine step    the indexer's own scores and top_k, over the kept superblocks

The summary key (coarse mode):
    mean     the mean of the superblock's indexer keys
    bound    per-channel midpoint and half-range: an upper bound on its best entry
    learned  pooled keys from a Router trained on the frozen model
    oracle   every entry key itself: a diagnostic that reads every entry

A DoubleIndexing is a csa2.Selector: set it as Csa2Attention.selector.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4IndexerScorer,
)

from deepseek import csa2

CoarseMode = Literal["mean", "bound", "learned", "oracle"]


class Router(csa2.Csa2Indexer):
    """The summary keys and coarse query of coarse="learned", one per layer.

    An indexer (query projection, head weights; copies of the layer's at init)
    plus n_keys pooled keys per superblock: key r is a per-channel softmax pooling
    of the superblock's indexer keys. At init key r leans on the r-th of n_keys runs
    of entries, so with one key the router starts as coarse="mean".

    Attributes:
        superblock_size: Entries per superblock (S).
        n_keys: Pooled keys per superblock.
    """

    def __init__(
        self, config: DeepseekV4Config, superblock_size: int, n_keys: int = 1
    ) -> None:
        super().__init__(config)
        self.superblock_size, self.n_keys = superblock_size, n_keys
        dim = config.index_head_dim
        run = torch.arange(superblock_size)[:, None] * n_keys // superblock_size
        bias = torch.where(run == torch.arange(n_keys), 0.0, -3.0)[..., None]
        self.score = nn.Parameter(torch.zeros(dim, n_keys, dim))
        self.bias = nn.Parameter(bias.expand(-1, -1, dim).clone())
        self.value = nn.Parameter(torch.eye(dim))
        self.temperature = nn.Parameter(torch.zeros(()))  # log scale, training only

    def pool(self, superblocks: torch.Tensor) -> torch.Tensor:
        """(B, G, S, c) indexer keys -> (B, G, n_keys, c) summary keys."""
        logit = torch.einsum("bgsc,crd->bgsrd", superblocks, self.score) + self.bias
        weight = logit.softmax(2)  # over entries, per key and channel
        return torch.einsum("bgsrd,bgsd->bgrd", weight, superblocks @ self.value)


def add_router(
    attention: csa2.Csa2Attention, superblock_size: int, n_keys: int = 1
) -> Router:
    """Give a Full or Reindex layer a fresh Router (attention.router), its query and
    head weights copied from the layer's indexer."""
    assert attention.indexer is not None
    weight = attention.indexer.q_b_proj.weight
    router = Router(attention.config, superblock_size, n_keys).to(weight.device)
    router.load_state_dict(attention.indexer.state_dict(), strict=False)
    attention.router = router
    return router


@dataclass(frozen=True)
class DoubleIndexing:
    """Double indexing settings and selector.

    Attributes:
        superblock_size: Consecutive main KV entries per superblock (S).
        kept_superblocks: Complete superblocks the coarse step keeps per query (s).
        coarse: The summary key the coarse step scores.
    """

    superblock_size: int
    kept_superblocks: int
    coarse: CoarseMode = "mean"

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """The indexer's top-k among the kept superblocks and the query's own."""
        indexer = attention.indexer
        assert indexer is not None, "selectors run in Full and Reindex layers"
        m = indexer.compress_rate
        q = indexer.queries(
            inputs.hidden_states, inputs.q_residual, inputs.position_ids
        )
        scores = indexer.scorer(q, inputs.index_k, inputs.hidden_states)
        finished = csa2.finished_entries(inputs.position_ids, m)
        kept = self.coarse_step(coarse_scores(attention, inputs, q, self), finished)
        allowed = candidates(kept, finished, self.superblock_size, scores.shape[-1])
        scores = scores.masked_fill(~allowed, float("-inf"))
        return csa2.top_picks(scores, inputs.position_ids, m, indexer.index_topk)

    def coarse_step(self, score: torch.Tensor, finished: torch.Tensor) -> torch.Tensor:
        """(B, S, min(s, G)) each query's best complete superblocks; ids at or past
        its complete superblocks are padding.

        Args:
            score: (B, S, G) coarse scores.
            finished: (B, S, 1) finished entries per query.
        """
        n_superblocks = score.shape[-1]
        visible = torch.arange(n_superblocks, device=score.device) < (
            finished // self.superblock_size
        )
        keep = min(self.kept_superblocks, n_superblocks)
        return score.masked_fill(~visible, float("-inf")).topk(keep, dim=-1).indices


def superblocks(index_k: torch.Tensor, superblock_size: int) -> torch.Tensor:
    """(B, n, c) indexer keys -> (B, G, S, c), complete superblocks only."""
    batch, n_entries, dim = index_k.shape
    n_superblocks = n_entries // superblock_size
    keys = index_k[:, : n_superblocks * superblock_size]
    return keys.view(batch, n_superblocks, superblock_size, dim)


def summary_keys(
    blocks: torch.Tensor, coarse: CoarseMode, router: Router | None = None
) -> torch.Tensor:
    """(B, G, S, c) superblock keys -> (B, G, r, c) the keys the coarse step scores.

    mean: the mean key (r = 1). bound: per-channel midpoint and half-range (r = 2).
    learned: the router's pooled keys (r = n_keys). oracle: every key (r = S).
    """
    if coarse == "bound":
        high, low = blocks.amax(2), blocks.amin(2)
        return torch.stack([(high + low) / 2, (high - low) / 2], 2)
    if coarse == "learned":
        assert router is not None
        return router.pool(blocks)
    return blocks.mean(2, keepdim=True) if coarse == "mean" else blocks


def head_weights(
    scorer: DeepseekV4IndexerScorer, hidden_states: torch.Tensor
) -> torch.Tensor:
    """(B, S, heads) the scorer's head weights, float32, scaled as it scales them."""
    return scorer.weights_proj(hidden_states).float() * scorer.weights_scaling


def score_summaries(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    coarse: CoarseMode,
    scale: float,
) -> torch.Tensor:
    """Score every superblock's summary keys for every query.

    bound (Quest-style): an upper bound on the best entry score, per head
    relu(q·mid + |q|·half_range) >= relu(q·k) for every key k in the superblock,
    heads with negative weight counted as 0. Else the best summary key's indexer
    score, sum_h w_h relu(q_h·k) scale.

    Args:
        q: (B, S, heads, c) queries.
        weights: (B, S, heads) head weights from head_weights.
        keys: (B, G, r, c) from summary_keys.
        coarse: The mode keys were built with.
        scale: The scorer's softmax_scale.

    Returns:
        (B, S, G) float32 scores.
    """
    q, keys = q.float(), keys.float()
    if coarse == "bound":
        upper = torch.einsum("bthc,bgc->bthg", q, keys[:, :, 0])
        upper += torch.einsum("bthc,bgc->bthg", q.abs(), keys[:, :, 1])
        return torch.einsum("bth,bthg->btg", F.relu(weights), F.relu(upper)) * scale
    per_head = F.relu(torch.einsum("bthc,bgrc->bthgr", q, keys))
    return torch.einsum("bth,bthgr->btgr", weights, per_head).amax(-1) * scale


def coarse_scores(
    attention: csa2.Csa2Attention,
    inputs: csa2.SelectorInputs,
    q: torch.Tensor,
    route: DoubleIndexing,
) -> torch.Tensor:
    """Score every complete superblock for every query (score_summaries); learned
    mode with the router's own query and head weights.

    Args:
        attention: The layer, with attention.router for coarse="learned".
        inputs: The layer's selector inputs.
        q: (B, S, heads, c) indexer queries.
        route: Double indexing settings.

    Returns:
        (B, S, G) scores.

    Raises:
        ValueError: If coarse="learned" and the layer has no router for S.
    """
    assert attention.indexer is not None
    scorer, hidden = attention.indexer.scorer, inputs.hidden_states
    router = None
    if route.coarse == "learned":
        router = getattr(attention, "router", None)
        if not isinstance(router, Router) or (
            router.superblock_size != route.superblock_size
        ):
            raise ValueError(f"no router for superblock size {route.superblock_size}")
        scorer = router.scorer
        q = router.queries(hidden, inputs.q_residual, inputs.position_ids)
    blocks = superblocks(inputs.index_k, route.superblock_size)
    keys = summary_keys(blocks, route.coarse, router)
    weights = head_weights(scorer, hidden)
    return score_summaries(q, weights, keys, route.coarse, scorer.softmax_scale)


def candidates(
    kept: torch.Tensor, finished: torch.Tensor, superblock_size: int, n_entries: int
) -> torch.Tensor:
    """(B, S, n) entries of each query's kept superblocks and of its own.

    Args:
        kept: (B, S, s) kept superblock ids; ids at or past a query's complete
            superblocks are padding.
        finished: (B, S, 1) finished entries per query.
        superblock_size: Entries per superblock.
        n_entries: Main KV entries.

    Returns:
        Boolean mask; csa2.top_picks masks the unfinished entries.
    """
    n_superblocks = n_entries // superblock_size
    # HF's position_ids have batch 1: expand, scatter_ does not broadcast
    complete = (finished // superblock_size).expand(*kept.shape[:2], 1)
    flag = torch.zeros(
        (*kept.shape[:2], n_superblocks + 1), dtype=torch.bool, device=kept.device
    )
    # Padding ids are sent to the query's own superblock, which is flagged anyway.
    flag.scatter_(-1, torch.where(kept < complete, kept, complete), True)
    flag.scatter_(-1, complete, True)
    superblock = torch.arange(n_entries, device=kept.device) // superblock_size
    index = superblock.clamp(max=n_superblocks).expand(*kept.shape[:2], -1)
    return flag.gather(-1, index)


def query_reads(
    finished: torch.Tensor, route: DoubleIndexing | None, n_keys: int = 1
) -> torch.Tensor:
    """Indexer keys each query scores: every finished entry for the flat indexer
    (route None); else the summary keys of its complete superblocks, every entry of
    the kept ones and the finished entries of its own.

    Args:
        finished: Finished entries per query, any shape.
        route: Double indexing settings, or None for the indexer alone.
        n_keys: Router keys per superblock, for coarse="learned".

    Returns:
        Reads, finished's shape.
    """
    if route is None:
        return finished
    size = route.superblock_size
    complete = finished // size
    per_superblock = {"mean": 1, "bound": 2, "learned": n_keys}
    coarse = (
        finished
        if route.coarse == "oracle"
        else per_superblock[route.coarse] * complete
    )
    kept = complete.clamp(max=route.kept_superblocks) * size
    return coarse + kept + finished % size


def reads(
    seq_len: int, compress_rate: int, route: DoubleIndexing | None, n_keys: int = 1
) -> float:
    """query_reads summed over a sequence's queries, for one layer.

    Args:
        seq_len: Queries per sequence.
        compress_rate: Tokens per main KV entry (m).
        route: Double indexing settings, or None for the indexer alone.
        n_keys: Router keys per superblock, for coarse="learned".
    """
    finished = (torch.arange(seq_len) + 1) // compress_rate
    return float(query_reads(finished, route, n_keys).sum())


def pick(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    summaries: torch.Tensor,
    finished: torch.Tensor,
    top_k: int,
    route: DoubleIndexing,
    scale: float,
) -> csa2.Picks:
    """Coarse step, then a fine step that scores only the candidate entries.

    Picks what DoubleIndexing picks (it masks the scores of every entry instead),
    at the reads of reads(). bench/indexer.py times it.

    Args:
        q: (B, S, heads, c) indexer queries.
        weights: (B, S, heads) head weights from head_weights.
        index_k: (B, n, c) indexer keys.
        summaries: (B, G, r, c) summary_keys of index_k, cached at decode.
        finished: (B, S, 1) finished entries per query.
        top_k: Entries to pick.
        route: Double indexing settings.
        scale: The scorer's softmax_scale.

    Returns:
        The picks.
    """
    size = route.superblock_size
    coarse = score_summaries(q, weights, summaries, route.coarse, scale)
    # Ascending, so candidates are in entry order like the masked top-k's.
    kept = route.coarse_step(coarse, finished).sort(-1).values
    complete = finished // size
    offsets = torch.arange(size, device=q.device)
    own = complete * size + offsets
    kept_entries = (kept[..., None] * size + offsets).flatten(-2)
    entries = torch.cat([kept_entries, own], -1)
    valid = torch.cat(
        [(kept < complete).repeat_interleave(size, -1), own < finished], -1
    )
    entries = entries.clamp(max=index_k.shape[1] - 1)  # invalid ones may overflow
    rows = torch.arange(q.shape[0], device=q.device)[:, None, None]
    keys = index_k[rows, entries].float()  # (B, S, candidates, c)
    per_head = F.relu(torch.einsum("bthc,btnc->bthn", q.float(), keys))
    score = torch.einsum("bth,bthn->btn", weights, per_head) * scale
    score = score.masked_fill(~valid, float("-inf"))
    top, j = score.topk(min(top_k, entries.shape[-1]), dim=-1)
    return csa2.Picks(entries.gather(-1, j), top > float("-inf"), top)
