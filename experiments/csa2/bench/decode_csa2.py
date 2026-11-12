"""Decode speed: one token's attention through all 10 attention layers, batch 1,
bf16, context n tokens of one document. CSA2 (learned indexer) and our
parameter-free picks (exact, tournament) in CSA2's KV layout, in the CSA2 model and
in stripped record 91, vs FlashAttention 2 dense (all n keys; record 91's windows
768 / 2560).
CSA2 layout: entries of 4 tokens, one shared head per group of 4 layers (3 groups);
picks at layers 0, 2, 4, 6, 8, reused by the next layer; attention over 128 window
keys + 64 picked entries. Not timed: projections (CSA2's indexer query too; our
sum of q heads), cache writes, compressing a finished entry.
CSA2 and ours: PyTorch eager, fp32 scores as in training (csa2_select) and HF's
indexer scorer, one CUDA graph per token; and Triton (src/nanogpt/csa2_decode.py,
the same top 64 kernels for both). FlashAttention 2: one layer's graph x 10.
L2 flushed before each replay; the least of 5 medians (triton do_bench).
Fork venv, from the repo root:
    PYTHONPATH=src python decode_csa2.py check      # decode picks = training picks
    PYTHONPATH=src python decode_csa2.py OUT.csv
"""

import csv
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton
from flash_attn import flash_attn_with_kvcache
from torch import Tensor

from nanogpt import csa2_decode, csa2_flex, csa2_select
from nanogpt.tournament_channel_blocks import windows
from nanogpt.tournament_fast import Settings

H, D, W, TOP, M, DIMS = 6, 128, 128, 64, 4, 16
LAYERS, GROUP = 10, 4
IH, ID = 6, 32  # CSA2 indexer heads x dims
ROPE = 16  # CSA2 rotary channels (partial_rotary_factor 0.125)
RR = 96  # record 91 rotary table width
QK = 96  # record 91's real q, k channels (zero-padded to D)
SCALE = D**-0.5
BF = torch.bfloat16


def rand(*shape: int, dtype: torch.dtype = BF) -> Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def exact(q: Tensor, keys: Tensor) -> Tensor:
    """csa2_select.exact for one query q (H, D): top 64 by (sum_h q_h) . kv, fp32."""
    return (keys.float() @ q.float().sum(0)).topk(TOP).indices


def tournament(q: Tensor, keys: Tensor) -> Tensor:
    """csa2_select.tournament for one query: each round adds 16 channels to the
    survivors' scores and keeps the best half; the last round leaves 64. Reads only
    the survivors' round channels."""
    qs = q.float().sum(0)
    rounds = D // DIMS
    values, order = windows(qs[None, None], Settings(TOP << rounds, TOP, DIMS))
    values, order = values[0, 0], order[0, 0].long()
    n = keys.shape[0]
    doubles = sum(n > TOP << r for r in range(rounds))
    alive = torch.arange(n, device=keys.device)
    score = torch.zeros(n, device=keys.device)
    for r in range(doubles):
        w = slice(r * DIMS, (r + 1) * DIMS)
        score = score + keys[alive[:, None], order[w]].float() @ values[w]
        keep = score.topk(TOP << (doubles - r - 1)).indices
        alive, score = alive[keep], score[keep]
    return alive


def indexer(qi: Tensor, weights: Tensor, index_k: Tensor) -> Tensor:
    """HF DeepseekV4IndexerScorer + top 64: sum_h w_h relu(q_h . k) / sqrt(32)."""
    scores = (index_k.float() @ qi.float().T).relu() * ID**-0.5  # (N, IH)
    return (scores @ weights).topk(TOP).indices


def csa2_attend(
    q: Tensor,
    window_kv: Tensor,
    main_kv: Tensor,
    picks: Tensor,
    sinks: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """csa2.sparse_attention for one query (K = V, a sink logit per head), then the
    output's inverse RoPE on the rotary channels."""
    keys = torch.cat([main_kv[picks], window_kv])  # (64 + W, D)
    logits = (q @ keys.T).float() * SCALE
    probs = torch.cat([logits, sinks[:, None]], 1).softmax(-1)[:, :-1]
    out = probs.to(BF) @ keys
    x = out[:, -ROPE:]
    half = torch.cat([-x[:, ROPE // 2 :], x[:, : ROPE // 2]], 1)
    return torch.cat([out[:, :-ROPE], x * cos - half * sin], 1)


def turn(x: Tensor, f1: Tensor, f2: Tensor) -> Tensor:
    """x (N, D) rotated by rows f1, f2 (N, RR) as csa2_flex.rotate; (f1, -f2)
    undoes it."""
    n, r = x.shape[0], f1.shape[1]
    y = x[:, :r].float()
    flip = y.view(n, -1, 2).flip(-1).view(n, r)
    return torch.cat([(f1 * y + f2 * flip).type_as(x), x[:, r:]], 1)


def b_attend(
    q: Tensor,
    k_win: Tensor,
    v_win: Tensor,
    keys: Tensor,
    picks: Tensor,
    f1: Tensor,
    f2: Tensor,
    scale: float = SCALE,
) -> Tensor:
    """csa2_flex.attention for one query: own window k, v (W, H, D) and the picked
    entries. The cache keeps rotated keys only; a pick's value (the unrotated entry)
    is its key turned back. k_win may hold only the QK real channels."""
    k_win = F.pad(k_win, (0, D - k_win.shape[-1]))
    kp = keys[picks]
    vp = turn(kp, f1[picks], -f2[picks])
    k = torch.cat([k_win, kp[:, None].expand(-1, H, -1)]).transpose(0, 1)
    v = torch.cat([v_win, vp[:, None].expand(-1, H, -1)]).transpose(0, 1)
    return F.scaled_dot_product_attention(q[:, None], k, v, scale=scale)[:, 0]


def csa2_layers(s: dict, pick: Callable) -> None:
    for layer in range(LAYERS):
        g = layer // GROUP
        if layer % 2 == 0:
            if pick is indexer:
                picks = indexer(s["qi"][layer], s["w"][layer], s["index_k"][g])
            else:
                picks = pick(s["q"][layer], s["main_kv"][g])
        csa2_attend(
            s["q"][layer],
            s["window_kv"][layer],
            s["main_kv"][g],
            picks,
            s["sinks"][layer],
            s["cos"],
            s["sin"],
        )


def b_layers(s: dict, pick: Callable) -> None:
    for layer in range(LAYERS):
        keys = s["keys"][layer // GROUP]
        if layer % 2 == 0:
            picks = pick(s["q"][layer], keys)
        b_attend(s["q"][layer], *s["win"][layer], keys, picks, s["f1"], s["f2"])


def fused_csa2_layers(s: dict, keep: int | None) -> None:
    """CSA2 with csa2_decode's Triton kernels: top 64 by the indexer (keep None) or
    by ours on the main KV (the CSA2 model with our picks: sum of q heads, all 128
    channels; keep 0: exact, else the prefilter keeping keep), then attention."""
    for layer in range(LAYERS):
        g = layer // GROUP
        if layer % 2 == 0:
            if keep is None:
                picks = csa2_decode.top64(
                    s["qi"][layer], s["index_k"][g], s["w"][layer]
                )
            elif keep:
                picks = csa2_decode.prefilter64(s["qa"][layer], s["main_kv"][g], keep)
            else:
                picks = csa2_decode.top64(s["qa"][layer], s["main_kv"][g])
        csa2_decode.csa2_attend(
            s["q"][layer],
            s["window_kv"][layer],
            s["main_kv"][g],
            picks,
            s["sinks"][layer],
            s["cos"],
            s["sin"],
        )


def fused_b_layers(s: dict, setting: tuple[str, int, int, bool]) -> None:
    """Ours with csa2_decode's Triton kernels: top 64 (keep 0: exact; else the
    prefilter keeping keep) every `every` layers, attention over the window `window`
    (win: 6 heads, mqa: one K and one V head). q's 96 real channels scored (record
    91 pads); routed: csa2_route's q_routed (2 of 6 blocks) picks on its 2 blocks
    and gives the picks' logits."""
    window, keep, every, routed = setting
    for layer in range(LAYERS):
        keys = s["keys"][layer // GROUP]
        if layer % every == 0:
            qs = s["qs"][layer]
            if routed:
                picks = csa2_decode.top64(s["qrs"][layer], keys, blocks=2)
            elif keep:
                picks = csa2_decode.prefilter64(qs, keys, keep, width=RR)
            else:
                picks = csa2_decode.top64(qs, keys, width=RR)
        k, v = s[window][layer]
        q_pick = s["qr"][layer] if routed else None
        csa2_decode.b_attend(
            s["q"][layer], k, v, keys, picks, s["table"], SCALE, q_pick
        )


VERSIONS = {
    "CSA2 stripped, learned indexer, Triton": (fused_csa2_layers, None),
    "CSA2 stripped, exact top 64 entries instead of the indexer, Triton": (
        fused_csa2_layers,
        0,
    ),
    "CSA2 stripped, 16-channel prefilter keeps 512, then exact top 64 instead of the "
    "indexer, Triton": (fused_csa2_layers, 512),
    **{
        f"CSA2 KV layout in stripped record 91{label}, {rule}, Triton": (
            fused_b_layers,
            (window, keep, 2, False),
        )
        for window, label in (
            ("win", ""),
            ("mqa", ", MQA window"),
        )
        for keep, rule in (
            (0, "exact top 64 entries"),
            (512, "16-channel prefilter keeps 512, then exact top 64"),
        )
    },
    "CSA2 KV layout in stripped record 91, exact top 64 entries, picks once per "
    "group of 4 layers, Triton": (fused_b_layers, ("win", 0, GROUP, False)),
    "CSA2 KV layout in stripped record 91, routed channels (2 of 6 blocks), exact "
    "top 64 entries, Triton": (fused_b_layers, ("win", 0, 2, True)),
    "CSA2 KV layout in stripped record 91, routed channels (2 of 6 blocks), exact "
    "top 64 entries, picks once per group of 4 layers, Triton": (
        fused_b_layers,
        ("win", 0, GROUP, True),
    ),
    "CSA2 stripped, learned indexer, PyTorch": (csa2_layers, indexer),
    "CSA2 stripped, exact picks instead of the indexer, PyTorch": (
        csa2_layers,
        exact,
    ),
    "CSA2 stripped, tournament picks instead of the indexer, PyTorch": (
        csa2_layers,
        tournament,
    ),
    "CSA2 KV layout in stripped record 91, exact top 64 entries, PyTorch": (
        b_layers,
        exact,
    ),
    "CSA2 KV layout in stripped record 91, tournament top 64 entries, PyTorch": (
        b_layers,
        tournament,
    ),
}


def state(n: int) -> dict:
    """Random caches at context n: 3 groups of entries, per-layer windows."""
    n4 = n // M  # finished entries
    # record 91's tables (Yarn(96)): the first 48 channels turn, pairs share angles
    angle = F.pad(rand(n4, RR // 4, dtype=torch.float32), (0, RR // 4))
    angle = angle.repeat_interleave(2, 1)
    sign = torch.tensor([1.0, -1.0], device="cuda").repeat(RR // 2)
    f1, f2 = angle.cos().to(BF), (angle.sin() * sign).to(BF)  # bf16 as record 91's
    q = rand(LAYERS, H, D)
    # csa2_route: 2 of q's 6 16-channel blocks kept per token (shared by its heads)
    kept = torch.rand(LAYERS, 6, device="cuda").argsort(1)[:, :2]
    gate = torch.zeros(LAYERS, D // 16, device="cuda").scatter(1, kept, 1.0)
    qr = q * gate.repeat_interleave(16, 1)[:, None].to(BF)
    return {
        "qr": qr,
        "qrs": qr.float().sum(1),
        "q": q,
        "qs": q.float().sum(1) * (torch.arange(D, device="cuda") < RR),
        "qa": q.float().sum(1),
        "main_kv": [rand(n4, D) for _ in range(3)],
        "index_k": [rand(n4, ID) for _ in range(3)],
        "window_kv": [rand(W, D) for _ in range(LAYERS)],
        "qi": rand(LAYERS, IH, ID),
        "w": rand(LAYERS, IH, dtype=torch.float32),
        "sinks": rand(LAYERS, H, dtype=torch.float32),
        "cos": rand(ROPE, dtype=BF),
        "sin": rand(ROPE, dtype=BF),
        "keys": [rand(n4, D) for _ in range(3)],
        "win": [(rand(W, H, QK), rand(W, H, D)) for _ in range(LAYERS)],
        "mqa": [
            (rand(W, 1, QK).expand(W, H, QK), rand(W, 1, D).expand(W, H, D))
            for _ in range(LAYERS)
        ],
        "f1": f1,
        "f2": f2,
        "table": csa2_decode.pair_table(f1, f2),
    }


def graphed_ms(fn: Callable[[], object]) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    # WSL shares the GPU with Windows, whose time slices only add time: the least
    # of 5 medians
    return min(
        triton.testing.do_bench(graph.replay, return_mode="median") for _ in range(5)
    )


def flash_ms(keys: int) -> float:
    """One layer of FlashAttention 2 decode over `keys` cached keys."""
    q = rand(1, 1, H, D)
    k, v = rand(1, keys, H, D), rand(1, keys, H, D)
    lengths = torch.tensor([keys], dtype=torch.int32, device="cuda")
    return graphed_ms(
        lambda: flash_attn_with_kvcache(
            q, k, v, cache_seqlens=lengths, softmax_scale=SCALE
        )
    )


def check() -> None:
    """Decode picks equal csa2_select's for a row's last query (one document, 8192
    tokens); b_attend equals csa2_flex.attention's last row (fp32, 1024 tokens)."""
    torch.manual_seed(0)
    n = 8192
    q, keys = rand(n, H, D), rand(n // M, D)
    starts = torch.zeros(n, dtype=torch.long, device="cuda")
    for rule, fn in (("exact", exact), ("tournament", tournament)):
        want = getattr(csa2_select, rule)(q, keys, starts, M, TOP)[0][-1]
        got = fn(q[-1], keys)
        assert set(got.tolist()) == set(want.tolist()), rule
        print(f"{rule}: decode picks = csa2_select picks", flush=True)
    t, f32 = 1024, torch.float32
    q, k, v = (rand(t, H, D, dtype=f32) for _ in range(3))
    entries = rand(t // M, D, dtype=f32)
    angle = rand(t, RR // 2, dtype=f32).repeat_interleave(2, 1)
    f1 = angle.cos()
    f2 = angle.sin() * torch.tensor([1.0, -1.0], device="cuda").repeat(RR // 2)
    rotated = csa2_flex.rotate(entries, M, f1, f2)
    cu = torch.tensor([0, t], device="cuda")
    entry, valid = csa2_select.exact(q, rotated, starts[:t], M, TOP)
    want = csa2_flex.attention(q, k, v, rotated, entries, entry, valid, cu, 0.1, M)
    got = b_attend(
        q[-1], k[-W:], v[-W:], rotated, entry[-1], f1[::M], f2[::M], scale=0.1
    )
    assert torch.allclose(got, want[-1], atol=1e-4), (got - want[-1]).abs().max()
    print("record 91 layout: decode attention = csa2_flex.attention", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
        sys.exit()
    rows = []
    for n in (8192, 32768, 131072, 524288):
        torch.manual_seed(0)
        full = flash_ms(n)
        rows.append(("Dense, FlashAttention 2, all n keys", n, LAYERS * full))
        print(rows[-1], flush=True)
        windowed = 8 * flash_ms(min(n, 768)) + 2 * flash_ms(min(n, 2560))
        name = "Dense stripped, FlashAttention 2 windows 768 / 2560"
        rows.append((name, n, windowed))
        print(rows[-1], flush=True)
        s = state(n)
        for name, (layers, pick) in VERSIONS.items():
            ms = graphed_ms(lambda: layers(s, pick))  # noqa: B023
            rows.append((name, n, ms))
            print(rows[-1], flush=True)
        del s
        torch.cuda.empty_cache()
    with open(sys.argv[1], "w", newline="") as f:
        csv.writer(f).writerows([("version", "n", "ms_per_token"), *rows])
