"""csa2_ref against DeepSeek's reference and TensorRT-LLM's cache formats:
- the 10-layer stack = model.py's Attention layers (src/deepseek4.1-1017/inference,
  run with torch stand-ins for its tilelang kernels, no FP8/FP4 rounding, fp32):
  every layer's output, picks and candidate blocks, on one sequence;
- two packed documents = each document through model.py alone;
- attend (flex + sink) = dense softmax with the sink, forward and gradients, eager
  and compiled;
- fake_quant = TensorRT-LLM quantization.py's pack_rows then unpack_rows (its file in
  ~/repos/TensorRT-LLM, a read-only sparse clone; skipped when absent);
- a compiled bf16 training step: finite, gradients reach the indexer.
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_ref
"""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nanogpt import csa2_ref, csa2_select

ROOT = Path(__file__).parents[2]
MODEL = ROOT / "src/deepseek4.1-1017/inference/model.py"
TRTLLM = Path.home() / (
    "repos/TensorRT-LLM/tensorrt_llm/_torch/attention/backends/sparse/csa2/"
    "quantization.py"
)
# small sizes; 16 picks and 8 candidate blocks of 4 cut at these lengths
CFG = csa2_ref.Config(
    dim=64, n_heads=2, head_dim=32, rope_head_dim=8, q_lora_rank=32, o_lora_rank=16,
    norm_eps=1e-6, candidate_topk_blocks=8, candidate_block_size=4, index_n_heads=8,
    index_head_dim=16, index_topk=16,
)  # fmt: skip
LENGTH = 512


def sparse_attn(
    q: Tensor, kv: Tensor, attn_sink: Tensor, topk_idxs: Tensor, softmax_scale: float
) -> Tensor:
    """kernel.py's sparse_attn in torch: softmax over the gathered rows (-1 = none)
    with the sink's exp in the sum."""
    rows = torch.arange(kv.shape[0], device=kv.device)[:, None, None]
    keys = kv[rows, topk_idxs.clamp(min=0).long()].float()  # (b, m, k, d)
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), keys) * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0)[:, :, None], float("-inf"))
    sink = attn_sink.float().view(1, 1, -1, 1).expand(*logits.shape[:3], 1)
    prob = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", prob, keys).to(q.dtype)


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference() -> Any:
    """model.py with stand-ins: act_quant and fp4_act_quant return x unrounded,
    sparse_attn above; its other imports unused here. fp32 weights."""

    def same(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return x

    stubs: dict[str, dict[str, Any]] = {
        "kernel": {
            "act_quant": same, "fp4_act_quant": same, "fp4_gemm": None,
            "fp8_gemm": None, "hc_split_sinkhorn": None, "sparse_attn": sparse_attn,
        },
        "engram": {"EngramLayout": object, "NgramHashState": object},
        "image_processor": dict.fromkeys(
            ("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START"), 0
        ),
        "vision": {"Aligner": object, "ViT": object},
    }  # fmt: skip
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    ref = load("deepseek_v41_reference", MODEL)
    ref.default_dtype = torch.float32
    return ref


def layers(ref: Any) -> tuple[list[csa2_ref.Csa2RefAttention], list[Any]]:
    """The port's 10 layers (random weights, positive indexer head weights so that
    relu ties cannot decide picks) and model.py's with the same weights."""
    torch.manual_seed(0)
    port = [csa2_ref.Csa2RefAttention(i, CFG).cuda() for i in range(10)]
    args = ref.ModelArgs(
        n_layers=10, max_batch_size=1, max_seq_len=LENGTH, original_seq_len=0,
        **dataclasses.asdict(CFG),
    )  # fmt: skip
    model = []
    with torch.device("cuda"):
        for i, layer in enumerate(port):
            for name, p in layer.named_parameters():
                p.data.normal_(std=0.2 if p.ndim == 2 else 1)
                if p.ndim == 1 and "sink" not in name:
                    p.data.uniform_(0.5, 1.5)  # norm weights
            if layer.indexer is not None:  # input channel 0 is 1: w ~ 1 + noise
                layer.indexer.weights_proj.weight.data[:, 0] = 2
            # .float(): model.py hard-codes bf16 for some projections
            model.append(ref.Attention(i, args).float())
            model[-1].load_state_dict(layer.state_dict())
    return port, model


def chosen(entry: Tensor, valid: Tensor, n: int) -> Tensor:
    """(T, n) bool of picked entries."""
    at = torch.where(valid, entry.long(), n)
    return entry.new_zeros(entry.shape[0], n + 1, dtype=torch.bool).scatter(
        1, at, True
    )[:, :n]


def run_reference(ref: Any, model: list[Any], xs: Tensor) -> list[dict]:
    """model.py's layers on xs (10, 1, L, dim), one sequence: per layer its output,
    picks (L, N) and candidates (L, N)."""
    out = []
    with torch.device("cuda"):
        for layer, x in zip(model, xs, strict=True):
            y = layer(x.clone(), 0)
            state: dict[str, Tensor] = {"y": y[0]}
            if layer.compress_ratio:
                n = x.shape[1] // layer.compress_ratio
                idx = ref.shared_attn.topk_idxs[0]
                state["picks"] = chosen(idx - x.shape[1], idx >= 0, n)
                if layer.layer_id == CFG.candidate_source_layer:
                    state["cand"] = ref.shared_attn.candidates[0]
            out.append(state)
    return out


def run_port(
    port: list[csa2_ref.Csa2RefAttention], xs: Tensor, cu: Tensor
) -> list[dict]:
    """The port on the packed row xs (10, 1, T, dim): as run_reference, row-wide."""
    shared: dict = {}
    out = []
    for layer, x in zip(port, xs, strict=True):
        state = {"y": layer(x, cu, shared)[0]}
        if layer.ratio:
            n = x.shape[1] // layer.ratio
            state["picks"] = chosen(shared["picks"][0], shared["picks"][1], n)
            if layer.mode == 1:  # blocks numbered from each document's first entry
                size = CFG.candidate_block_size
                blocks = shared["blocks"]
                first = csa2_ref.starts(cu, x.shape[1])  # ratio 1: entry = token
                rel = (torch.arange(n, device=x.device) - first[:, None]) // size
                hit = (rel[..., None] == blocks[:, None]) & (blocks[:, None] >= 0)
                state["cand"] = hit.any(-1) & (rel >= 0)
        out.append(state)
    return out


def inputs(length: int) -> Tensor:
    torch.manual_seed(1)
    xs = torch.randn(10, 1, length, CFG.dim, device="cuda")
    xs[..., 0] = 1
    return xs


def compare(got: dict, want: dict, rows: slice, cols: slice, layer: int) -> None:
    y, w = got["y"][rows], want["y"]
    assert torch.allclose(y, w, atol=1e-4, rtol=1e-4), (layer, (y - w).abs().max())
    for key in ("picks", "cand"):
        if key in want:
            a, b = got[key][rows, cols], want[key]
            assert torch.equal(a, b), (layer, key, (a != b).sum())


def same_as_reference() -> None:
    """One sequence of LENGTH tokens: every layer as model.py's."""
    ref = reference()
    port, model = layers(ref)
    xs = inputs(LENGTH)
    cu = torch.tensor([0, LENGTH], device="cuda")
    with torch.no_grad():
        got, want = run_port(port, xs, cu), run_reference(ref, model, xs)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        compare(a, b, slice(None), slice(None), i)
    print("stack: outputs, picks, candidate blocks = model.py (1 sequence)")


def documents_are_separate() -> None:
    """Documents of 200 and 312 tokens packed: each as model.py alone."""
    ref = reference()
    port, model = layers(ref)
    xs = inputs(LENGTH)
    cut = 200
    cu = torch.tensor([0, cut, LENGTH], device="cuda")
    with torch.no_grad():
        got = run_port(port, xs, cu)
        for a, b in ((0, cut), (cut, LENGTH)):
            want = run_reference(ref, model, xs[:, :, a:b])
            for i, (g, w) in enumerate(zip(got, want, strict=True)):
                m = port[i].ratio or 1
                compare(g, w, slice(a, b), slice(a // m, b // m), i)
    print("documents: packed row = each document through model.py alone")


def attend_is_dense(compiled: bool) -> None:
    """attend = dense softmax over window rows and picks plus the sink: output,
    log-sum-exp and gradients."""
    g = torch.Generator().manual_seed(2)
    docs, heads, dim, m = (200, 312, 128, 128), 4, 64, 2
    length = sum(docs)
    q = torch.randn(length, heads, dim, generator=g).cuda()
    kv = torch.randn(length, dim, generator=g).cuda()
    entries = torch.randn(length // m, dim, generator=g).cuda()
    sink = torch.randn(heads, generator=g).cuda()
    cu = torch.tensor((0, *torch.tensor(docs).cumsum(0).tolist())).cuda()
    begin = csa2_ref.starts(cu, length)
    entry, valid = csa2_select.exact(q, entries, begin, m, 8)
    tensors = q, kv, entries, sink
    for x in tensors:
        x.requires_grad_()
    run = csa2_ref.attend
    if compiled:
        run = torch.compile(run, fullgraph=True, dynamic=False)
    out, lse = run(q, kv, entries, entry, valid, cu, 0.2, m, sink)

    t = torch.arange(length, device="cuda")[:, None]
    near = (t >= t.T) & (t - t.T < 128) & (t.T >= begin[:, None])
    picked = chosen(entry, valid, entries.shape[0])
    keys = torch.cat([kv, entries])
    logits = torch.einsum("thd,kd->thk", q, keys) * 0.2
    logits = logits.masked_fill(~torch.cat([near, picked], 1)[:, None], float("-inf"))
    full = torch.cat([logits, sink[None, :, None].expand(length, -1, 1)], -1)
    want = full.softmax(-1)[..., :-1] @ keys
    assert torch.allclose(out, want, atol=2e-5), (out - want).abs().max()
    assert torch.allclose(lse, logits.logsumexp(-1), atol=2e-5)
    grad = torch.randn_like(want)
    expect = torch.autograd.grad(want, tensors, grad)
    for a, b in zip(torch.autograd.grad(out, tensors, grad), expect, strict=True):
        assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()
    print(
        "attend: = dense softmax with sink, fwd + lse + grads", "(compiled)" * compiled
    )


def fake_quant_is_trtllm() -> None:
    """fake_quant = TensorRT-LLM's pack_rows -> unpack_rows, incl. E2M1 ties."""
    if not TRTLLM.exists():
        print("fake_quant: SKIPPED, no TensorRT-LLM clone at", TRTLLM)
        return
    trt = load("trtllm_csa2_quantization", TRTLLM)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(300, 128, generator=g) * torch.rand(300, 1, generator=g) * 4
    x[0, :32] = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0) * 4)
    x[1] = 0
    for fmt in ("main", "index", "swa"):
        want = trt.unpack_rows(trt.pack_rows(x, fmt), 128, fmt, torch.float32)
        got = csa2_ref.fake_quant(x, fmt)
        assert torch.equal(got, want), (fmt, (got - want).abs().max())
    print("fake_quant: = TensorRT-LLM pack_rows/unpack_rows (main, index, swa)")


def training_step() -> None:
    """Compiled bf16 forward + backward of the SMALL stack on 2 documents: finite
    loss, finite gradients, the indexer's weights moved by the indexer loss."""
    torch.manual_seed(5)
    port = torch.nn.ModuleList(csa2_ref.Csa2RefAttention(i) for i in range(10)).cuda()
    length = 4096
    cu = torch.tensor([0, 1500, 3000, length], device="cuda")
    x = torch.randn(1, length, csa2_ref.SMALL.dim, device="cuda", requires_grad=True)

    def step(x: Tensor) -> Tensor:
        shared: dict = {}
        total = x.new_zeros(())
        with torch.autocast("cuda", torch.bfloat16):
            for layer in port:
                total = total + layer(x, cu, shared).float().square().mean()
        return total

    loss = torch.compile(step, dynamic=False)(x)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in port.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    indexer = port[9].indexer
    assert isinstance(indexer, csa2_ref.Indexer)
    assert indexer.wq_b.weight.grad is not None
    assert indexer.wq_b.weight.grad.abs().sum() > 0
    print("training step: compiled bf16 fwd + bwd finite, indexer trained")


if __name__ == "__main__":
    fake_quant_is_trtllm()
    same_as_reference()
    documents_are_separate()
    for compiled in (False, True):
        attend_is_dense(compiled)
    training_step()
