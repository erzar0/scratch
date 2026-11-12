"""modded-nanogpt record 91, loaded outside its training script.

The model lives in ../modded-nanogpt-r91 (branch record91-1gpu of ../modded-nanogpt):
record 91's train_gpt.py on one GPU, FA2 in place of FA3. run_r91.sh trains it at a
proxy scale and saves the final state. Run this package with that worktree's .venv
(torch 2.10, FA2) and PYTHONPATH=src.

train_gpt.py is one script: definitions, then training at module level. module() runs
its source up to "# int main" (imports, kernels, single-process dist setup, classes) as
the module train_gpt; load() then builds GPT as the script does.
"""

import functools
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

R91 = Path.home() / "repos" / "modded-nanogpt-r91"
FINEWEB = Path.home() / "repos" / "modded-nanogpt" / "data" / "fineweb10B"
# run_r91.sh's settings that shape the model: bigram rows, YaRN table length
PROXY = {
    "SWEEP_BATCH_DIV": "8",
    "SWEEP_BIGRAM_VOCAB": "47160",
    "SWEEP_VAL_BATCH": "65536",
    "DISABLE_FP8": "1",
}
SEQ = 2048  # record 91's longest training document
# the training script module() runs: train_csa2.py and train_dense_stripped.py share
# its interface; set before the first module() call
SCRIPT = "train_gpt.py"
# layer 6 has no attention
ATTENTION_LAYERS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)


@functools.cache
def module() -> types.ModuleType:
    """SCRIPT's definitions, as a single-process run sees them."""
    for name, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "localhost"),
        ("MASTER_PORT", "29517"),
    ):
        os.environ.setdefault(name, value)
    os.environ.update(PROXY)
    path = R91 / SCRIPT
    source = path.read_text()
    tg = types.ModuleType("train_gpt")
    tg.__file__ = str(path)
    sys.modules["train_gpt"] = tg
    sys.path.insert(0, str(R91))
    # the script reads its kernel files next to sys.argv[0], for its log
    argv0, sys.argv[0] = sys.argv[0], str(path)
    try:
        exec(compile(source[: source.index("# int main")], path, "exec"), tg.__dict__)
    finally:
        sys.argv[0] = argv0
    return tg


def load(path: Path) -> tuple[nn.Module, Any]:
    """GPT from a run_r91.sh checkpoint: frozen, eval mode, dtypes as trained.

    Returns:
        The model and its final validation's ForwardScheduleConfig (windows, YaRN).
    """
    tg = module()
    # train_gpt.py ran as __main__, so the optimizer state pickles its classes there
    main = sys.modules["__main__"]
    sys.modules["__main__"] = tg
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.modules["__main__"] = main
    model = tg.GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=tg.args.val_batch_size // (tg.grad_accum_steps * tg.world_size),
    ).cuda()
    state = {k.removeprefix("_orig_mod."): v for k, v in saved["model"].items()}
    model.load_state_dict(state, assign=True)
    model.cuda()
    for yarn, tables in zip(model._all_yarns, saved["yarn"], strict=True):
        yarn.factor1.copy_(tables["factor1"])
        yarn.factor2.copy_(tables["factor2"])
        yarn.attn_scale = tables["attn_scale"]
    model.eval().requires_grad_(False)
    return model, tg.ForwardScheduleConfig(**saved["forward_args"])


def first_tokens(tokens: torch.Tensor, bos: int, length: int) -> torch.Tensor:
    """The first `length` tokens of every document at least that long.

    Args:
        tokens: Concatenated documents, each starting with bos.
        bos: The document separator.
        length: Window length.

    Returns:
        (documents, length) int64, each row starting at its bos.
    """
    starts = (tokens == bos).nonzero().squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([len(tokens)])])
    kept = starts[ends - starts >= length]
    return tokens[kept[:, None] + torch.arange(length)].long()


def long_documents(pattern: str) -> torch.Tensor:
    """Windows of SEQ + 1 tokens: the start of every longer document.

    Args:
        pattern: Shard glob in FINEWEB, e.g. "fineweb_val_*.bin".

    Returns:
        (documents, SEQ + 1) int64 on the CPU.
    """
    tg = module()
    shards = sorted(FINEWEB.glob(pattern))
    return torch.cat(
        [first_tokens(tg._load_data_shard(f).int(), tg.BOS_ID, SEQ + 1) for f in shards]
    )


def token_losses(model: nn.Module, config: Any, window: torch.Tensor) -> torch.Tensor:
    """Per-token loss on one document window (SEQ + 1 tokens), as validation computes
    it but without the canonical-token mask."""
    inputs = window[:-1]
    seqlens = torch.tensor([0, len(inputs)], dtype=torch.int32, device="cuda")
    bigrams = module().get_bigram_hash(inputs).cuda()
    return model(inputs.cuda(), window[1:].cuda(), seqlens, bigrams, config)
