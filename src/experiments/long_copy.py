"""Copy test at 8K-128K tokens: pure CSA2 against double indexing.

experiments/copy_test.py on the 8192-token model at up to 16x its training length.
Each window is a T/2-token piece of WikiText-103 (val, then test), then the same
piece again: every token of the repeat lies T/2 back. Per window: mean loss on the
repeat (copy loss) and on the first half (text loss, positions past 8192 unseen in
training). Every method's selector runs on 512 queries at a time (qwen.evaluate's
Chunked), and lm_head on 512 positions at a time, so no (T, T) or (T, vocab)
tensor is built. Hit rate: the share of repeat queries whose valid picks hold the
entry of the token to predict, T/2 back, per selecting layer.

    SEQ=8192 uv run python -m experiments.long_copy [pure] [T...]
    -> runs/paper/long_copy.csv (rows of each T replaced); pure: pure CSA2 only
"""

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from deepseek import csa2
from experiments import paired_se
from experiments.results import Results, paired_delta
from experiments.selector_forward import (
    LayerSelectors,
    layer_selector,
    reads_per_seq,
    selecting,
)
from experiments.train_router import RouterTraining
from model import train
from qwen.evaluate import Chunked, PerLayer, own_indexer

logger = logging.getLogger(__name__)

OUT = paired_se.OUT.with_name("long_copy.csv")
LENGTHS = (8192, 16384, 32768, 65536, 131072)
MAX_WINDOWS = 30
CHUNK = 512  # queries per selector call, positions per lm_head call


def sources(seq_len: int) -> np.ndarray:
    """(n, T/2) consecutive pieces of WikiText-103 val then test, n ≤ MAX_WINDOWS."""
    data = np.concatenate([train.load("validation"), train.load("test")])
    half = seq_len // 2
    n = min(MAX_WINDOWS, len(data) // half)
    return data[: n * half].reshape(n, half)


@dataclass
class CopyHits:
    """Wraps a selector; counts, per layer, repeat queries t whose valid picks hold
    entry (t - half + 1) // m, the earlier copy of the token t predicts.

    Attributes:
        inner: The selector whose picks are counted.
        half: First position of the repeat.
        counts: Layer -> [hits, repeat queries].
    """

    inner: csa2.Selector
    half: int
    counts: dict[int, np.ndarray] = field(default_factory=dict)

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """The inner picks, after counting their hits."""
        picks = self.inner(attention, inputs)
        assert attention.indexer is not None
        m = attention.indexer.compress_rate
        t = inputs.position_ids
        repeat = t >= self.half
        target = ((t - self.half + 1) // m).unsqueeze(-1)
        hit = ((picks.entry == target) & picks.valid).any(-1) & repeat
        row = self.counts.setdefault(attention.layer_idx, np.zeros(2))
        row += [float(hit.sum()), float(repeat.sum())]
        return picks


def chunked(model: PreTrainedModel, selector: LayerSelectors) -> csa2.Selector:
    """selector per layer (own indexer where None), on CHUNK queries at a time."""
    # ponytail: RandomEntries re-seeds on each call, so every chunk repeats the same
    # draws; it is only the floor reference
    layers = range(len(model.model.layers))
    each = tuple(layer_selector(selector, i) or own_indexer for i in layers)
    return Chunked(PerLayer(each), CHUNK)


@torch.no_grad()
def window_losses(
    model: PreTrainedModel, piece: torch.Tensor, selector: csa2.Selector
) -> tuple[float, float]:
    """Mean loss on the first half (text) and on the repeat (copy) of one window.

    Args:
        model: The CSA2 model.
        piece: (1, T/2) token ids.
        selector: The selector of every Full and Reindex layer.
    """
    seq = torch.cat([piece, piece, piece[:, :1]], 1)  # as copy_test.copy_batch
    half = piece.shape[1]
    cuda = train.DEVICE.type == "cuda"
    with (
        selecting(model, selector),
        torch.autocast(train.DEVICE.type, torch.bfloat16, enabled=cuda),
    ):
        # a mask dict is used as is; CSA2 layers ignore the mask, so skip
        # transformers' dense (T, T) one
        out = model.model(
            input_ids=seq[:, :-1], attention_mask={"": None}, use_cache=False
        )
        hidden = out.last_hidden_state[0]
        nll = torch.cat(
            [
                F.cross_entropy(
                    model.lm_head(hidden[s : s + CHUNK]).float(),
                    seq[0, s + 1 : s + 1 + CHUNK],
                    reduction="none",
                )
                for s in range(0, len(hidden), CHUNK)
            ]
        )
    # position half - 1 predicts the jump back to the piece's start: neither half
    return float(nll[: half - 1].mean()), float(nll[half:].mean())


def main(run: train.TrainConfig, lengths: list[int], pure_only: bool) -> None:
    """Pure CSA2 and every paired_se headline at each length; see the docstring."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    n_keys = RouterTraining.from_env().n_keys
    pure = paired_se.Headline("pure CSA2", "flat indexer", 1.0, None)
    cuda = train.DEVICE.type == "cuda"
    found = [pure] + ([] if pure_only else paired_se.headlines(model, run))
    for seq_len in lengths:
        pieces = sources(seq_len)
        results = Results(OUT, {"T": seq_len})
        flat_reads = reads_per_seq(model, seq_len, None, n_keys)
        reference: dict[str, np.ndarray] = {}
        for method in found:
            if method.selector is not None:
                paired_se.prepared(model, run, method)
            hits = CopyHits(chunked(model, method.selector), seq_len // 2)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            losses = np.array(
                [window_losses(model, train.to_device(p)[None], hits) for p in pieces]
            )
            text, copy = losses.T
            if method is pure:
                reference = {"text": text, "copy": copy}
            row: dict[str, object] = {"T": seq_len, "method": method.method}
            row["config"] = method.config
            if method.reads_ratio is not None:  # none for random and oracle
                reads = reads_per_seq(model, seq_len, method.selector, n_keys)
                row["reads_ratio"] = flat_reads / reads
            for name, values in (("copy", copy), ("text", text)):
                delta, se = paired_delta(values, reference[name])
                row |= {f"{name}_loss": values.mean(), f"{name}_delta": delta}
                row[f"{name}_se"] = se
            rates = {i: h / n for i, (h, n) in sorted(hits.counts.items())}
            row["hit_rate"] = float(np.mean(list(rates.values())))
            row |= {f"hit_layer{i}": rate for i, rate in rates.items()}
            row["windows"] = len(pieces)
            if cuda:
                row["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            logger.info("%s", row)
            results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = sys.argv[1:]
    lengths = [int(a) for a in args if a.isdigit()] or list(LENGTHS)
    main(train.TrainConfig.from_env(), lengths, "pure" in args)
