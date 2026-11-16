"""Every attention version that is benchmarked, and nothing else.

A version pairs an emulation (float32 Chunk -> Outcome, exact selection and read
counts: the quality benchmarks) with a fast decode step (Triton kernel or plain
PyTorch: the latency benchmark); tests/attention/test_registry.py checks that
both select the same keys. A latency-only version (the transformers DeepSeek-V4 CSA
layer) has no emulation. Every sparse version gives exact attention to a budget of
KEEP keys; docs/methods.md lists the exceptions, tests/attention/test_correctness.py
checks them.
Ids are <family>/<variant>; names are short; docs/methods.md explains every
setting in plain words. versions(keep) builds every version at another budget.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from attention import fast_pytorch as fp
from attention.bandit import BoundBandit
from attention.block_topk import BlockTopK
from attention.csa2_block_mean import Csa2BlockMean
from attention.csa2_block_mean_fp4 import Csa2BlockMeanFp4
from attention.csa2_exact_picks import Csa2ExactPicks
from attention.csa2_head_max import Csa2HeadMax
from attention.csa2_parameter_free import Csa2ParameterFree
from attention.csa2_reference import Csa2Reference
from attention.csa2_sketch_picks import Csa2SketchPicks
from attention.deepseek_csa import LibraryCsa, LibraryCsaFlash
from attention.dense import dense
from attention.exact_topk import ExactTopK
from attention.harness import Chunk, Method, Outcome
from attention.loki import Loki
from attention.sparq import SparQ
from attention.summed_query import SummedQueryTopK
from attention.tournament.blocks import BlockTournament
from attention.tournament.blocks_then_global import HybridTournament
from attention.tournament.channel_groups import GroupOrder
from attention.tournament.general import Tournament, halving_rounds
from attention.tournament.mean_value_losers import MeanLosers
from attention.tournament.sampled_cutoffs import SampledHalving
from attention.tournament.shuffled_blocks import ShuffledBlocks, shuffle
from attention.tournament.shuffled_blocks_growing import GrowingShuffledBlocks
from attention.tournament.wide_first_round import FirstRound
from attention.triton import common
from attention.triton import sparq as sparq_triton
from attention.triton import tournament_blocks_then_global as blocks_then_global
from attention.triton import tournament_global_halving as global_halving
from attention.triton import tournament_sampled_cutoffs as sampled
from attention.triton import tournament_shuffled_blocks as shuffled_blocks

KEEP = 512  # the standard budget: keys attended exactly
MIN_RATIO = 16  # benchmarks run a length n with budget keep only if n >= 16 keep:
# the global-halving kernel needs 4 or more halving rounds


@dataclass(frozen=True)
class LastRoundsGlobal:
    """HybridTournament split as its kernel splits it: the last 3 rounds global."""

    keep: int = KEEP
    block: int = 1024
    global_rounds: int = 3

    def __call__(self, chunk: Chunk) -> Outcome:
        """The hybrid tournament at this chunk's length."""
        block_rounds = max(
            halving_rounds(chunk.key.shape[1], self.keep) - self.global_rounds, 0
        )
        return HybridTournament(
            block=self.block, block_rounds=block_rounds, keep=self.keep
        )(chunk)


@dataclass(frozen=True)
class BlocksPerHead:
    """ShuffledBlocks split as its kernel splits it: `blocks` blocks per KV head."""

    keep: int = KEEP
    blocks: int = shuffled_blocks.BLOCKS

    def __call__(self, chunk: Chunk) -> Outcome:
        """Shuffled block halving at this chunk's length."""
        block = (self.keep // self.blocks) << halving_rounds(
            chunk.key.shape[1], self.keep
        )
        return ShuffledBlocks(block=block, keep=self.keep)(chunk)


def kernel_step(select: Any) -> fp.Step:
    """A Triton select(q, kv) -> winners, then the Triton exact attention."""

    def step(q: fp.Tensor, kv: fp.KV) -> tuple[fp.Tensor, fp.Tensor]:
        won = select(q, kv)
        return common.attend(q, kv.k, kv.v, won), won

    return step


def shuffled_step(split: Callable[[int], tuple[int, int]]) -> fp.Step:
    """The parallel kernel on K stored in ShuffledBlocks' slot order; split(n) =
    (keys kept, blocks per KV head) at length n."""

    def step(q: fp.Tensor, kv: fp.KV) -> tuple[fp.Tensor, fp.Tensor]:
        n = kv.k.shape[1]
        perm = shuffle(n, str(kv.k.device))
        k_t = kv.once("k_t_shuffled", lambda: kv.k_t[..., perm].contiguous())
        won = perm[shuffled_blocks.select(q, k_t, *split(n)).long()]
        return common.attend(q, kv.k, kv.v, won), won

    return step


def sampled_approx_step(keep: int) -> fp.Step:
    """The sampled kernel with approximate losers; it returns no winners."""

    def step(q: fp.Tensor, kv: fp.KV) -> tuple[fp.Tensor, None]:
        return sampled.tournament_approx(q, kv.k, kv.k_t, kv.v, keep, first=8), None

    return step


@dataclass(frozen=True)
class Version:
    """One benchmarked version.

    Attributes:
        id: <family>/<variant>.
        name: Short plain name; docs/methods.md explains every setting.
        implementation: Of the fast step: Triton, PyTorch, or transformers (a
            library layer as is).
        emulation: The quality benchmarks' method; None: latency only.
        fast: The latency benchmark's decode step.
    """

    id: str
    name: str
    implementation: str
    emulation: Method | None
    fast: fp.Step

    @property
    def family(self) -> str:
        """The id's first part."""
        return self.id.split("/")[0]

    @property
    def legend(self) -> str:
        """Name and implementation, as graphs show it."""
        return f"{self.name} ({self.implementation})"

    @property
    def settings(self) -> dict[str, Any]:
        """The emulation's fields (else the fast step's), logged as params."""
        source = self.fast if self.emulation is None else self.emulation
        if is_dataclass(source) and not isinstance(source, type):
            return asdict(source)
        return {}

    def tags(self) -> dict[str, str]:
        """What tracking tags a run with."""
        return {
            "version": self.id,
            "label": self.name,
            "legend": self.legend,
            "family": self.family,
            "implementation": self.implementation,
        }


def versions(keep: int = KEEP) -> dict[str, Version]:
    """Every version at a budget of `keep` keys, by id."""
    tour = Tournament("query", "halving", "drop", keep=keep)
    original = Tournament("random", "bracket", "approx", keep=keep)
    wide = FirstRound(keep=keep, first=16)
    groups = GroupOrder(keep=keep, first=16, grouping="contiguous")
    mean_losers = MeanLosers(keep=keep, first=16)
    blocks = BlockTournament(keep=keep, block=1024)
    flat = BlockTopK(block=4, keep=keep, window=32)
    kept = keep // 16  # superblocks of 64 keys: 4 * keep keys to choose blocks from
    double_mean = BlockTopK(
        4, keep, 32, superblock=16, kept_superblocks=kept, coarse="mean"
    )
    double_bound = BlockTopK(
        4, keep, 32, superblock=16, kept_superblocks=kept, coarse="bound"
    )
    exact_bound = BoundBandit("exact", 0.25, keep=keep)
    gauss_bound = BoundBandit("gauss", 1.0, keep=keep)
    growing = GrowingShuffledBlocks(keep=keep)
    exact_picks = Csa2ExactPicks(keep)
    block_mean = Csa2BlockMean(keep)
    block_mean_wide = Csa2BlockMean(keep, pool=4)
    block_mean_half = Csa2BlockMean(keep, pool=2)
    block_mean_fp4 = Csa2BlockMeanFp4(keep, pool=4)
    sketches = [
        ("sign-sketch", "sign-sketch picks", Csa2SketchPicks(keep)),
        (
            "pca-sketch",
            "PCA-sketch picks",
            Csa2SketchPicks(keep, sketch="pca", fraction=8),
        ),
        (
            "random-sketch",
            "random-sketch picks",
            Csa2SketchPicks(keep, sketch="random", fraction=8),
        ),
        (
            "channel-sketch",
            "top-channel picks",
            Csa2SketchPicks(keep, sketch="channels", fraction=8),
        ),
    ]
    head_max = Csa2HeadMax(keep)
    parameter_free = Csa2ParameterFree(keep)
    out = [
        Version("dense", "Dense", "PyTorch", dense, fp.dense),
        Version(
            "exact-topk",
            "Exact top-k",
            "PyTorch",
            ExactTopK(keep),
            fp.exact_topk(keep),
        ),
        Version(
            "exact-topk/summed-query",
            "Exact top-k, summed query",
            "PyTorch",
            SummedQueryTopK(keep),
            fp.summed_query_topk(keep),
        ),
        Version(
            "sparq",
            "SparQ",
            "Triton",
            SparQ(r=32, k=keep, local=16),
            kernel_step(lambda q, kv: sparq_triton.select(q, kv.k_t, 32, keep, 16)),
        ),
        Version("loki", "Loki", "PyTorch", Loki(d=32, k=keep), fp.loki_step(32, keep)),
        Version(
            "deepseek-v4/csa", "DeepSeek-V4 CSA", "transformers", None, LibraryCsa(keep)
        ),
        Version(
            "deepseek-v4/csa-flash",
            "DeepSeek-V4 CSA, V4-Flash size",
            "transformers",
            None,
            LibraryCsaFlash(keep),
        ),
        Version(
            "deepseek-v4.1/csa2",
            "DeepSeek-V4.1 CSA2",
            "PyTorch",
            None,
            Csa2Reference(keep),
        ),
        Version(
            "deepseek-v4.1/csa2-exact-picks",
            "DeepSeek-V4.1 CSA2, exact picks",
            "PyTorch",
            exact_picks,
            fp.csa2_exact_picks(exact_picks),
        ),
        Version(
            "deepseek-v4.1/csa2-block-mean",
            "DeepSeek-V4.1 CSA2, block-mean picks",
            "PyTorch",
            block_mean,
            fp.csa2_block_mean(block_mean),
        ),
        Version(
            "deepseek-v4.1/csa2-block-mean-wide",
            "DeepSeek-V4.1 CSA2, block-mean picks, wide pool",
            "PyTorch",
            block_mean_wide,
            fp.csa2_block_mean(block_mean_wide),
        ),
        Version(
            "deepseek-v4.1/csa2-block-mean-half-wide",
            "DeepSeek-V4.1 CSA2, block-mean picks, half-wide pool",
            "PyTorch",
            block_mean_half,
            fp.csa2_block_mean(block_mean_half),
        ),
        Version(
            "deepseek-v4.1/csa2-block-mean-fp4-wide",
            "DeepSeek-V4.1 CSA2, FP4 block-mean picks, wide pool",
            "PyTorch",
            block_mean_fp4,
            fp.csa2_block_mean(block_mean_fp4),
        ),
        *(
            Version(
                f"deepseek-v4.1/csa2-{tag}",
                f"DeepSeek-V4.1 CSA2, {name}",
                "PyTorch",
                rule,
                fp.csa2_sketch_picks(rule),
            )
            for tag, name, rule in sketches
        ),
        Version(
            "deepseek-v4.1/csa2-head-max",
            "DeepSeek-V4.1 CSA2, max-over-heads picks",
            "PyTorch",
            head_max,
            fp.csa2_head_max(head_max),
        ),
        Version(
            "deepseek-v4.1/csa2-parameter-free",
            "DeepSeek-V4.1 CSA2, parameter-free",
            "PyTorch",
            parameter_free,
            fp.csa2_parameter_free(parameter_free),
        ),
        Version("block-topk/flat", "Block top-k", "PyTorch", flat, fp.block_topk(flat)),
        Version(
            "block-topk/double-indexing-mean",
            "Double indexing: mean",
            "PyTorch",
            double_mean,
            fp.block_topk(double_mean),
        ),
        Version(
            "block-topk/double-indexing-bound",
            "Double indexing: bound",
            "PyTorch",
            double_bound,
            fp.block_topk(double_bound),
        ),
        Version(
            "tournament/global-halving",
            "Tournament: global halving",
            "Triton",
            tour,
            kernel_step(lambda q, kv: global_halving.select(q, kv.k_t, keep)),
        ),
        Version(
            "tournament/blocks",
            "Tournament: blocks",
            "PyTorch",
            blocks,
            fp.tournament(blocks),
        ),
        Version(
            "tournament/blocks-then-global",
            "Tournament: blocks, then global",
            "Triton",
            LastRoundsGlobal(keep=keep),
            kernel_step(lambda q, kv: blocks_then_global.select(q, kv.k_t, keep)),
        ),
        Version(
            "tournament/wide-first-round",
            "Tournament: wide first round",
            "PyTorch",
            wide,
            fp.tournament(wide),
        ),
        Version(
            "tournament/sampled-cutoffs",
            "Tournament: sampled cutoffs",
            "Triton",
            SampledHalving(keep=keep, first=8),
            kernel_step(lambda q, kv: sampled.select(q, kv.k_t, keep, 8)),
        ),
        Version(
            "tournament/sampled-cutoffs-approximate-losers",
            "Tournament: sampled cutoffs, approximate losers",
            "Triton",
            SampledHalving(keep=keep, first=8, losers="approx"),
            sampled_approx_step(keep),
        ),
        Version(
            "tournament/channel-groups",
            "Tournament: channel groups",
            "PyTorch",
            groups,
            fp.tournament(groups),
        ),
        Version(
            "tournament/mean-value-losers",
            "Tournament: mean-value losers",
            "PyTorch",
            mean_losers,
            fp.tournament(mean_losers),
        ),
        Version(
            "tournament/shuffled-blocks",
            "Tournament: shuffled blocks",
            "Triton",
            BlocksPerHead(keep=keep),
            shuffled_step(lambda n: (keep, shuffled_blocks.BLOCKS)),
        ),
        Version(
            "tournament/shuffled-blocks-growing",
            "Tournament: shuffled blocks, growing budget",
            "Triton",
            growing,
            shuffled_step(growing.split),
        ),
        Version(
            "tournament/original",
            "Tournament: original",
            "PyTorch",
            original,
            fp.tournament(original),
        ),
        Version(
            "bandit/exact-bound",
            "Bandit: exact bound",
            "PyTorch",
            exact_bound,
            fp.bandit(exact_bound),
        ),
        Version(
            "bandit/gaussian-bound",
            "Bandit: Gaussian bound",
            "PyTorch",
            gauss_bound,
            fp.bandit(gauss_bound),
        ),
    ]
    return {version.id: version for version in out}


VERSIONS = versions()
