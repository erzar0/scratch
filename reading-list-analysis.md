# Analysis of a 63-item research bookmark collection

Compiled 2026-09-08 from a browser bookmark export spanning 2025-08-24 → 2026-09-07.

Titles and abstracts for items that could not be identified from the URL alone were
verified against arXiv/JMLR listings rather than recalled. Nine items were misfiled in
my initial read and corrected: `1609.03890` is Santambrogio's gradient-flows survey (not
an ML paper), `cs/0111050` is Spielman–Teng smoothed analysis, `2402.15391` is Genie,
`1707.05300` is Florensa et al. reverse curriculum, `2608.16884` is the AlphaEvolve
matrix-multiplication-exponent paper, `2608.17981` is Mozer et al. "Recirculation",
`2608.09888` is BDH-CQ, `2509.01533` is FoRo, `2409.19561` unifies backprop and
forward-forward, and JMLR 20-1061 is the t-SNE/UMAP/TriMap/PaCMAP study.

---

## 1. The headline

This is not a general ML reading list. It is two lists layered on top of each other:

- **A dormant coursework layer** (Aug 2025 – Jul 2026): RL courses, NLP courses,
  diffusion courses, systems playbooks. Broad, canonical, mostly *intake*.
- **An active research layer** (last five days) that is extremely narrow and reads like
  a literature sweep for a specific paper.

The dates make this unambiguous. **Twenty of the 63 items were added on 2026-09-07 inside
a two-hour window (21:31 → 21:53), and the item added the evening before was Neel Nanda's
"Highly Opinionated Advice on How to Write ML Papers."** That is the signature of someone
who has decided on a contribution and is now assembling related work.

What that sweep contains is two tightly-scoped literatures, pulled in back to back:

| 21:31–21:32 | depth-recurrence / looping as a scaling axis |
| 21:33       | zeroth-order and memory-constrained training  |
| 21:48–21:53 | more of both, plus algorithmic-limits theory  |

Everything else in the collection is either background for those two, or leftover from
an earlier phase.

---

## 2. Cluster A — "Depth is a resource separate from parameters"

The dominant and most coherent thread. Every one of these attacks the same question:
*how much serial computation does a forward pass actually get, and can you buy more of it
without buying more weights?*

- **Beyond Parameters: Virtual Logic Depth for Scaling Laws** (`2506.18233`, Zhu et al.)
  — the load-bearing claim: parameter reuse holds knowledge capacity roughly constant
  while materially improving reasoning. Parameters ↔ knowledge; depth ↔ reasoning.
- **SMELT** (`2609.01343`) — the rigorous version. Loops the middle half of an MoE while
  matching per-token FLOPs, non-embedding params, *and* KV cache; fits separate
  Chinchilla laws per architecture; 6.8–18.0% FLOP savings on the compute-optimal
  frontier. Crucially it fixes the standard methodological cheat (comparing at fixed
  model size, which conflates architecture with extra FLOPs). This is your
  methodology template.
- **Gated Recurrent Transformers** (`2608.15062`) — prelude/coda blocks bracketing one
  shared core iterated R times, with a GRU-style update gate so the shared block can
  specialize per iteration. Addresses the classic failure of naive depth-sharing
  (uniform transforms collapse representational diversity).
- **Recirculation** (`2608.17981`, Mozer et al., DeepMind) — the most interesting item
  here. Inference-time recurrence on *frozen off-the-shelf* models; explicitly framed as
  letting the model act as a dynamical system tracking belief states, and explicitly
  distinguished from both chain-of-thought and looping. Near-zero added generation
  latency, serial cost in prefill only.
- **BDH-CQ** (`2608.09888`) — recurrent latent reasoning without verbalizing; 150M
  params, 29.5% pass@2 on ARC-AGI-1 at $0.0007/task. The existence proof that latent
  serial compute beats token-serial compute on cost.
- **mHC: Manifold-Constrained Hyper-Connections** (`2512.24880`, DeepSeek) — the
  plumbing. Hyper-Connections widened the residual stream and broke the identity-mapping
  property; mHC projects the connection space onto a manifold to restore it. This is what
  makes very deep / heavily-recirculated stacks trainable at scale.
- **DeepSeek-V4** (`2606.19348`) — mHC shipped in production, alongside hybrid
  Compressed Sparse Attention + Heavily Compressed Attention and Muon. The proof that
  the residual-stream work is not academic.
- **Neuroscience of Transformers** (`2603.15339`) + **Cortical Columns Computing
  Systems** (IntechOpen) — the biological framing of the same idea. A cortical column
  *is* a repeated shared block applied recurrently. These two are the "motivation
  paragraph" for a looping paper, not evidence.

**Reading:** these are not seven papers on a topic; they are seven positions on one
question, and you have the full stack — theory (VLD), scaling law (SMELT), architecture
(GRT), training-free variant (Recirculation), efficiency existence proof (BDH-CQ),
optimization enabler (mHC), production validation (DeepSeek-V4).

---

## 3. Cluster B — Gradient-free and memory-constrained training

Added minutes after cluster A. Also near-complete for its scope.

- **A Primer on Zeroth-Order Optimization** (`2006.06224`) — the entry point.
- **Sparse MeZO** (`2402.15751`) — fewer parameters, better ZO fine-tuning.
- **Variance-reduced Zeroth-Order Methods for Fine-Tuning LMs** (`2404.08080`) — the
  main lever on ZO's central weakness (gradient-estimate variance).
- **MeSP** (`2602.13069`) — the counter-position: don't go gradient-free, go
  *structured*. Hand-derived backward passes exploiting LoRA's low rank; recomputes
  `h = xA` rather than storing it; 49% memory cut vs MeBP with *mathematically identical*
  gradients (361MB → 136MB on Qwen2.5-0.5B). Frames the field as exact-but-heavy (MeBP)
  vs cheap-but-noisy (MeZO) and splits the difference.
- **FoRo** (`2509.01533`) — forward-only, gradient-free continual learning; prompt tuning
  plus a knowledge-encoding mechanism, backbone untouched.
- **Unifying backprop and forward-forward via model predictive control** (`2409.19561`)
  — the theoretical keystone. Puts BP and FF on a single continuum indexed by
  look-forward horizon, with an exact analysis on deep linear nets. If you write in this
  area, this is the paper that gives you a coordinate system.

---

## 4. The intersection nobody in these two piles has taken

Clusters A and B were bookmarked ninety seconds apart, and they compose:

> Depth-recurrent and looped models are **parameter-light and compute-heavy**. That is
> precisely the regime where memory-constrained and gradient-free training methods are
> attractive — and precisely the regime nobody in cluster B is testing on, because they
> all evaluate on standard dense/LoRA fine-tuning of ordinary transformers.

Two specific unclaimed openings sit in the gap:

1. **Recirculation and FoRo are the same move applied to different resources.** Both
   freeze the pretrained weights and add capability — one adds serial computation, one
   adds forward-only adaptation. Nobody has stacked them: *frozen backbone + inference-time
   recurrence + forward-only adaptation of the recurrence controller.* GRT's update gate
   is exactly the kind of tiny, cheap-to-tune module that ZO or forward-only methods can
   handle, where full backprop through R unrolled iterations cannot.
2. **Looping changes the memory profile of backprop non-linearly.** Backprop through R
   iterations of a shared block costs R× activation memory unless you recompute. MeSP's
   argument (recompute the cheap low-rank intermediate rather than store it) generalizes
   directly to looped blocks, and neither paper mentions the other's setting. SMELT's
   compute-matching protocol gives you the fair-comparison machinery to evaluate it.

If a paper is the goal, (1) is the more novel framing and (2) is the more defensible
empirical contribution.

---

## 5. Cluster C — Limits of computation

Small, deliberate, and thematically the parent of cluster A.

- **Ultimate physical limits to computation** (`quant-ph/9908043`, Lloyd) — the ceiling:
  what a kilogram of matter can compute. Bookmarked Sep 2025, nine months before the rest.
- **Smoothed Analysis of Algorithms** (`cs/0111050`, Spielman–Teng) — why worst-case
  bounds mislead about practice. The methodological ancestor of "your benchmark comparison
  is measuring the wrong thing."
- **Improving the matrix multiplication exponent with AlphaEvolve** (`2608.16884`) —
  ω < 2.371177, via ML-designed optimization over the laser method. Note the shape:
  *learned search beating hand-designed search on a theory problem*.
- **Breaking the Sorting Barrier for Directed SSSP** (`2504.17033`) — a decades-old
  barrier falling.

The common shape with cluster A is exact: **an assumed floor on the cost of computation,
and evidence it isn't the real floor.** Lloyd sets the physical version; VLD/SMELT set the
architectural version. This is a genuine intellectual through-line, not a coincidence —
and it's why the working branch for this analysis is named after the Lloyd paper.

---

## 6. Cluster D — Diffusion and flow matching

Entirely pedagogy and surveys. Zero research contributions, zero recent additions
(newest is Jul 2026). This looks like a completed or abandoned learning phase.

- Step-by-Step Diffusion: An Elementary Tutorial (`2406.08929`) — best single entry point
- MIT 6.S184 *Flow Matching and Diffusion Models* (2026) — theory course
- MIT 6.S183 *A Practical Introduction to Diffusion Models* (IAP 2026) — applied course
- Cambridge MLG flow-matching blog post — short-form version of the same
- Efficient Diffusion Models: A Survey (`2502.06805`)
- Efficient Video Diffusion Models (`2604.15911`)

**The one you're underusing:** Santambrogio's *{Euclidean, Metric, Wasserstein} Gradient
Flows: an overview* (`1609.03890`) is filed here as if it were another tutorial. It is the
mathematical substrate — optimal transport, geodesic convexity, the JKO scheme. If the
diffusion interest ever revives, this is the item that would make it *deep* rather than
another course. If it doesn't revive, this is the one to keep and the four courses to drop.

---

## 7. Cluster E — Systems and performance

- Ultra-Scale Playbook (HF/nanotron) — the best single reference here for distributed training
- modded-nanogpt (Keller Jordan) — speedrun as a source of empirically-validated tricks
- CUDA Programming for H100s (video course); Triton intro (bornlex) — kernel-level entry
- DeepWiki: pytorch, nanochat — code navigation aids, not reading
- The Ubiquitous Sparse Matrix-Matrix Products (`2508.04077`) — SpGEMM survey
- Multiplying Matrices Without Multiplying (`2106.10860`, MADDNESS) — approximate matmul
  via learned hashing; the "you don't actually need the multiply" position
- Kimi K3 on vLLM; frontier training methodologies blog — current-practice tracking

MADDNESS + SpGEMM + the AlphaEvolve ω paper form a small sub-theme: *the cost of a matmul
is not fixed.* That rhymes with cluster A more than it looks.

---

## 8. Cluster F — Reinforcement learning

The largest cluster by item count and the weakest by content.

Sutton & Barto (**twice** — `RLbook2020.pdf` and `SuttonBartoIPRLBook2ndEd.pdf` are the
same book), CS 285, CS 234, HF Deep RL Course, "RL by the Book" video series — five
overlapping introductions to the same material, spread across a year, plus exactly one
research paper: **Reverse Curriculum Generation** (`1707.05300`, Florensa et al., 2017).

This is a nine-year-old paper and the cluster contains nothing from the era that made RL
central to LLMs — no RLHF, no PPO/GRPO, no RLVR, no verifier or process-reward work. Either
this is a deliberately closed chapter, or it is the single biggest blind spot in the
collection. Given that the active work is on architecture rather than post-training, I read
it as closed — but see §10.

---

## 9. Singletons

Each is the only representative of its area; none connects to the active thread.

- **Genie** (`2402.15391`) — 11B foundation world model, latent actions from unlabelled video
- **GenRec** (`2608.10257`) — Netflix's LLM ranker; verbalized user histories replacing
  thousands of engineered features, with ~40× less training data. Notable as a rare
  honest industrial A/B result.
- **SynthID-Text** (Nature) — production LLM watermarking
- **Geometric Deep Learning proto-book** (`2104.13478`) — the 5G framework
- **t-SNE/UMAP/TriMap/PaCMAP** (JMLR 22:20-1061) — what dimensionality-reduction plots
  actually preserve. Practically the most useful item in this section; most people
  over-read these plots.
- **SSOG-Attention** (pisoni.ai) — separable sum-of-Gaussians attention for ViTs: content
  never *scores*, only *steers* a learned positional field; factorizes into two 1D passes
  so the N×N matrix never materializes. Content-blind fixed field lands within a point of
  SDPA on ImageNet. Speculative and unrefereed, but it is the only *attention*-side idea
  in a collection otherwise focused on depth — worth keeping for that reason alone.
- lernapparat (nonparametric stats), Bayesian Data Analysis handouts, CS 224n (×2:
  site + playlist), CS 329S (ML systems design), deep-ml (×2: problems + collections),
  alphaXiv, ML-Papers-of-the-Week, Zenodo

---

## 10. Gaps that matter if the goal is a paper

Ranked by how much they'd hurt in review:

1. **No history of depth-sharing failures.** Cluster A contains no Universal Transformers,
   ALBERT, Adaptive Computation Time, or PonderNet. Every one of these tried parameter
   sharing across depth and the field largely moved on. A reviewer will ask what is
   different now, and "we didn't cite them" is the worst available answer. This is the
   most urgent hole.
2. **No negative or null results anywhere.** The collection is entirely papers that
   worked. That biases both the effect sizes you expect and the baselines you'll pick.
3. **No evaluation methodology.** Cluster A's whole premise is that prior comparisons were
   unfair (fixed-size instead of compute-matched). You have SMELT's protocol but nothing on
   benchmark contamination, statistical significance in LM eval, or ablation design — and
   that is exactly the ground a compute-matched-comparison paper will be attacked on.
4. **RL post-training absent** (§8). Defensible if the work is purely architectural;
   indefensible if any claim touches reasoning capability, since the standard rival
   explanation for improved reasoning in 2026 is post-training, not architecture.
5. **No mechanistic interpretability.** The claim "depth buys reasoning, parameters buy
   knowledge" (VLD) is a claim about *what computation happens inside*, and the collection
   has no tooling to look. The neuroscience pair is analogy, not measurement.

---

## 11. Housekeeping

**Exact duplicates — delete one of each:**
- Sutton & Barto: `incompleteideas.net/book/RLbook2020.pdf` = `web.stanford.edu/class/psych209/.../SuttonBartoIPRLBook2ndEd.pdf`
- SMELT: `arxiv.org/abs/2609.01343` = `arxiv.org/pdf/2609.01343`
- CS 224n: course site and YouTube playlist are the same content
- deep-ml: `/problems` and `/collections` are two views of one site

**Redundant coverage — keep one:** of the four diffusion introductions (MIT 6.S184,
MIT 6.S183, Cambridge MLG blog, Step-by-Step tutorial), Step-by-Step is the fastest
complete path and 6.S184 the most rigorous. The other two add nothing.

**Reference tools, not reading** (fine to keep, but they aren't papers): alphaXiv,
ML-Papers-of-the-Week, Zenodo, both DeepWiki pages, deep-ml.

Net effect: 63 bookmarks → roughly 45 distinct items, of which ~15 are load-bearing.

---

## 12. Suggested order

**If the goal is the paper the 2026-09-07 sweep implies:**

1. VLD (`2506.18233`) — the thesis, stated plainly
2. SMELT (`2609.01343`) — the methodology you must match or beat
3. Recirculation (`2608.17981`) — the strongest and most orthogonal recent result
4. GRT (`2608.15062`) — the architectural details you'll build on
5. mHC (`2512.24880`) → DeepSeek-V4 (`2606.19348`) — why it trains, and that it ships
6. **Then go find Universal Transformers / ACT / PonderNet before writing a word of
   related work** (§10.1)
7. BP↔FF via MPC (`2409.19561`) — if the training-side angle in §4 is the direction
8. Nanda's writing advice — last, when you have results, not now

**If the goal is breadth:** Lloyd → Spielman–Teng → MADDNESS → AlphaEvolve ω. Four short
papers, one idea, four decades. Cluster C is the most enjoyable reading in the collection
and the least useful for shipping anything.
