# What's actually in each paper

A ten-minute read. One entry per research contribution in the collection, grouped by
cluster. Courses, textbooks and tools are collected at the end, having no single result
to state.

**On sourcing:** arXiv, JMLR and every mirror are blocked from this session, so nothing
below comes from a PDF I opened. Pre-2025 papers are described from working knowledge;
later ones from their abstracts plus secondary summaries retrieved by search, kept to what
those sources stated. Every number is one a source reported — verify it against the PDF
before it goes into a related-work section.

---

## Depth as a resource separate from parameters

**Beyond Parameters: Virtual Logic Depth** — `2506.18233`
Proposes *virtual logical depth*: reuse the same parameters several times in a forward
pass to raise effective algorithmic depth without raising parameter count. Unlike
test-time scaling, which extends computation token-wise, VLD alters the internal
computation graph. Its controlled experiments yield the collection's most quotable claim —
**knowledge capacity stays almost constant under VLD scaling while reasoning improves
markedly**, so parameters track knowledge but not reasoning. A strong claim resting
entirely on how "knowledge capacity" was operationalised; treat it as a hypothesis with
evidence, not a law.

**SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers** — `2609.01343`
The methodologically serious member of the cluster. Its complaint: looping is usually
evaluated at fixed *model size*, which hands the looped model extra FLOPs and then credits
the architecture. SMELT matches three budgets at once — per-token FLOPs, non-embedding
parameters, and KV cache. The surviving recipe is in the name: *Sparse MoE, middle layers
Loop Twice*. Scaled to four sizes up to 54B non-embedding parameters with a separate
Chinchilla-style law fitted per architecture, loss falls faster with compute, saving
**6.8–18.0% of training FLOPs** on the compute-optimal frontier. Two secondary findings
matter more: the advantage transfers downstream *beyond what validation loss predicts*,
and it grows with sequence length and with the number of in-context examples. Largest on
code.

**Recirculation** — `2608.17981` · Mozer, Siddiqui, Sawyer, Sanyal, Liu (DeepMind)
The best mechanism here, and the simplest. A feedforward transformer can only push state
*upward* through depth, so whatever a deep layer has just disambiguated is invisible to the
shallow layers processing later tokens. Recirculation adds a training-free leak: after each
step, mix a deep-layer residual back down into a shallow destination, sequentially, using
convex, non-convex or adaptive mixers. That one edge turns a frozen network into a
dynamical system able to carry a belief state across tokens. On off-the-shelf models with
**no weight updates**, it lowers perplexity and raises generation and reasoning accuracy at
essentially zero added generation latency — the serial cost falls entirely in prefill. The
authors separate it explicitly from chain-of-thought, from looping, and from training a
recurrent transformer. An adaptive variant tunes a few hyperparameters, weights still
frozen.

**Gated Recurrent Transformers** — `2608.15062` · Hegazy, Alanwar, Elhoushi
States the design tension precisely: unique per-layer weights preserve functional
specialisation (input-grounding low, abstract refinement high) at heavy memory cost, while
naive depth-sharing imposes one uniform transformation that collapses that diversity. Their
answer is a fixed prelude and coda bracketing a single shared core iterated `R` times, with
a lightweight projection and an elementwise update gate — borrowed from gated RNNs —
conditioned on the hidden state, the prelude output, and noise resampled each step. The
gate lets one small core specialise per iteration instead of needing many unique layers.

**BDH-CQ: In-Context Learning with Recurrent Latent Reasoning** — `2608.09888` · Pathway
Extends the Dragon Hatchling architecture: high-dimensional positive activations, low-rank
communication, linear-attention-style updates, a recurrent associative state. Inputs
continuously update a recurrent memory; queries are then solved by iterating in a
high-dimensional latent workspace **without verbalising intermediate steps**. The bet is a
persistent large latent state rather than an ever-growing context window. A 150M
configuration reports **29.5% pass@2 on ARC-AGI-1 public eval at $0.0007 per task**. Note
the scope: a strong efficiency result on one benchmark from one company, not a general
capability claim.

**mHC: Manifold-Constrained Hyper-Connections** — `2512.24880` · DeepSeek
Hyper-Connections widen the residual stream (expansion rate typically 4) and diversify how
streams mix — which helps, but breaks the identity-mapping property that makes residuals
trainable, causing instability, poor scaling and memory-access overhead. mHC constrains the
connection matrices to the **Birkhoff polytope of doubly stochastic matrices** via
Sinkhorn–Knopp projection, conserving feature means and bounding signal propagation, so
identity mapping is restored while the widened stream is kept. At 18B MoE scale: final
training loss **1.776 vs 1.799** for the baseline, better downstream average, modest extra
FLOPs. Paired with real kernel work, which is why it shipped.

**DeepSeek-V4** — `2606.19348`
The production artifact: **V4-Pro at 1.6T parameters (49B active)** and **V4-Flash at 284B
(13B active)**, both at one-million-token context. Three named upgrades — hybrid attention
combining Compressed Sparse Attention and Heavily Compressed Attention, mHC in place of
conventional residuals, and Muon. Evidence that the residual-stream line isn't academic,
rather than a source of method detail.

**The Neuroscience of Transformers** — `2603.15339` · König & Negrello
A perspective piece, not a result: places transformer operations and cortical
architectonics in one descriptive framework to surface functional correspondences. Paired
here with an IntechOpen chapter on cortical-column computing systems, the hardware-oriented
version of the same analogy. Both are motivation, not evidence.

---

## Gradient-free and memory-constrained training

**A Primer on Zeroth-Order Optimization** — `2006.06224`
The entry point: estimating gradients from function evaluations alone via random directions
(one-point and two-point estimators, Gaussian smoothing), convergence rates, and the
applications that motivated the field — black-box adversarial attacks, model-agnostic
explanation, on-chip training. The theme to carry forward is the **dimension-dependence
penalty**: ZO convergence degrades by a factor of `d` or `√d`, which is why naive ZO on a
billion parameters is hopeless and why every paper below attacks that factor.

**Sparse MeZO** — `2402.15751`
MeZO made ZO fine-tuning of LLMs possible at inference-level memory. Sparse MeZO observes
that perturbing every parameter is wasteful — applying ZO to a selected subset improves both
accuracy and convergence over MeZO. The "shrink `d`" attack on the dimension penalty.

**Variance-reduced Zeroth-Order Methods** — `2404.08080`
The other attack: reduce estimator *variance* rather than dimension, with SVRG-style
machinery adapted to ZO. Gains are largest where plain MeZO is weakest — full-parameter and
non-prompt-based fine-tuning — with memory still far below backpropagation.

**MeSP: Memory-Efficient Structured Backpropagation** — `2602.13069`
The dissenting position, and the most concrete. Frames on-device fine-tuning as a forced
choice between exact-but-heavy (MeBP) and cheap-but-noisy (MeZO), then refuses it: instead
of going gradient-free, hand-derive the backward pass to exploit LoRA's low-rank structure.
The intermediate projection `h = xA` needn't be stored, because with rank `r ≪ d_in` it can
be recomputed for almost nothing. Result: **49% average memory reduction versus MeBP on
Qwen2.5 (0.5B–3B) with mathematically identical gradients** — 361 MB → 136 MB on
Qwen2.5-0.5B. No approximation, no noise, just a better-derived backward pass.

**FoRo: Forward-Only Continual Learning** — `2509.01533`
Targets catastrophic forgetting with pre-trained models. Existing methods freeze the
backbone and tune a few parameters but still run iterative backpropagation; FoRo is
forward-only and gradient-free — lightweight prompt tuning plus a knowledge-encoding
mechanism, backbone untouched. Aimed at resource-constrained deployment.

**Unifying backprop and forward-forward through model predictive control** — `2409.19561`
The theoretical keystone. Casts training as model predictive control, making
backpropagation and Hinton's forward-forward the two endpoints of one continuum indexed by
**look-forward horizon**, with a family of intermediate algorithms trading performance
against efficiency. Includes a precise analysis on a deep linear network and a principled
rule for choosing the horizon. This is the paper that gives the area coordinates rather
than a pile of methods.

---

## Limits of computation

**Ultimate physical limits to computation** — `quant-ph/9908043` · Lloyd (1999)
Derives what physics permits of a computer from energy and entropy alone: operation rate is
bounded by available energy through the Margolus–Levitin relation (~`2E/πℏ` ops/second),
memory by the system's entropy. For a one-kilogram, one-litre "ultimate laptop", that is
around **10⁵¹ operations per second on ~10³¹ bits**. The structure matters more than the
numbers — a machine at that limit is enormously *parallel*, because buying serial speed
means concentrating energy, and the end of that road is a black hole, where communication
and computation time coincide. Serial depth is scarce even at the physical limit.

**Smoothed Analysis of Algorithms** — `cs/0111050` · Spielman & Teng
Worst-case analysis says simplex is exponential; average-case says it's fast, under
distributions nobody encounters. Smoothed analysis takes the *maximum over inputs of
expected runtime under a small random perturbation* of that input, under which simplex with
shadow-vertex pivoting is polynomial in the dimensions and `1/σ`. The lasting lesson is
methodological: when practice contradicts your bound, the framework may be what's wrong.

**Improving the matrix multiplication exponent with AlphaEvolve** — `2608.16884`
The best bounds on `ω` come from combination loss analysis, a refinement of the laser
method whose core is a hard optimisation problem. The paper reformulates it, designs a new
optimisation algorithm, and refines that with AlphaEvolve, reaching **ω < 2.371177** from
2.371339. The gain is tiny; the shape is the point — machine-designed search moving a
constant human-designed search had been inching along for decades.

**Breaking the Sorting Barrier for Directed SSSP** — `2504.17033`
Dijkstra is optimal *if* you insist on emitting vertices in sorted distance order — the
`O(m + n log n)` sorting barrier. This gives a deterministic `O(m log^(2/3) n)` algorithm
for directed graphs with non-negative real weights, beating it on sparse graphs by
combining Bellman–Ford-style relaxation with a recursive partition that shrinks the
frontier so the full sort is never needed.

---

## Diffusion and flow matching

**Step-by-Step Diffusion: An Elementary Tutorial** — `2406.08929`
Derives diffusion from scratch with minimal prerequisites and no historical baggage: DDPM
and DDIM built up from reversing a gradual corruption, the role and learnability of the
score, discretisation, and the connection to flow matching and stochastic interpolants.

**{Euclidean, Metric, Wasserstein} Gradient Flows** — `1609.03890` · Santambrogio
Three movements: gradient flows in Euclidean space; their generalisation to metric spaces
(curves of maximal slope); then the Wasserstein space of probability measures under the
optimal-transport metric, with the transport tools and geodesic convexity needed to get
there. Culminates in the Jordan–Kinderlehrer–Otto scheme — the "minimising movement"
construction realising a PDE as a gradient flow on distributions — with a convergence proof
for linear Fokker–Planck. **The substrate under this whole cluster**: a diffusion model's
forward process is a Fokker–Planck evolution, and its reversal is a gradient flow in
exactly this sense.

**Efficient Diffusion Models: A Survey** — `2502.06805` (TMLR 2025)
Organises efficiency work into three levels — algorithm, system, framework. The maintained
GitHub index of covered papers is the more useful artifact.

**Efficient Video Diffusion Models** — `2604.15911`
A deployment-oriented review organising methods into four paradigms: **step distillation,
efficient attention, model compression, and cache/trajectory optimisation**. The framing to
keep: all four target one of two objectives — fewer function evaluations, or lower per-step
overhead — with attention and memory traffic the real bottlenecks once spatial-temporal
token growth compounds with iterative denoising.

---

## Systems and approximate computation

**Multiplying Matrices Without Multiplying** — `2106.10860` · Blalock & Guttag
Exploits the case where one matrix is known in advance. A small learned hash function — a
balanced binary regression tree, sixteen prototypes per subspace — encodes each row, after
which the product is table lookup and averaging, with **no multiply-accumulates in the
inner loop at all**. Roughly 100× over exact matmul and 10× over prior approximate methods,
with guarantees, at an accuracy cost. The position is what earns it a place: the multiply
is not a fixed cost of linear algebra.

**The Ubiquitous Sparse Matrix-Matrix Products** — `2508.04077`
Survey of SpGEMM — where it appears (graph algorithms, algebraic multigrid, triangle
counting, ML), how it's formulated, the data structures, and parallel algorithms across
architectures.

**SSOG-Attention** — pisoni.ai (blog, unrefereed)
Argues scaled dot-product attention is the wrong primitive for vision transformers. Each
head owns a few Gaussians forming a fixed field over relative position — "a learned habit of
where to look" — and content never *scores*, only *steers*, via a small content-conditioned
nudge shifting each token's field. The field is separable, so it factorises into two 1D
passes and the N×N matrix never materialises. The result is the interesting part: the
**content-blind fixed field alone** comes within a point of SDPA on ImageNet, and steering
closes the gap.

---

## Everything else

**Genie: Generative Interactive Environments** — `2402.15391` · DeepMind
An 11B foundation world model trained on unlabelled internet video with **no action
labels**. Three parts: a spatiotemporal video tokenizer; a latent action model inferring a
small discrete action vocabulary from adjacent frame pairs, unsupervised; and an
autoregressive dynamics model predicting the next frame's tokens from past tokens plus a
latent action. Output is a frame-by-frame controllable environment generated from a prompt,
photograph or sketch — and the latent action space transfers, letting agents imitate
behaviour from unseen video.

**GenRec: An LLM-Backed Recommendation Ranker at Netflix** — `2608.10257`
Two phases: adapt an open-source LLM to Netflix's catalogue and member behaviour, then
post-train it for ranking with recommendation-specific labels and rewards. The paper covers
phase two — the move from a discriminative ranker with thousands of engineered features to
one driven by **verbalised user histories** in natural language. A large-scale A/B test
reports significant offline and online gains from substantially fewer labelled examples.

**Scalable watermarking for identifying LLM outputs** — *Nature* (2024) · DeepMind
SynthID-Text. Uses **tournament sampling**: candidate tokens compete in a tournament scored
by pseudorandom functions keyed on preceding context, and the winner is emitted — so the
watermark lives in the sampling procedure, not in perturbed logits. Distortionary and
non-distortionary variants, the latter provably preserving quality; composes with
speculative decoding; validated on live Gemini traffic at production scale.

**Reverse Curriculum Generation for RL** — `1707.05300` · Florensa et al.
For sparse-reward goal-reaching tasks, requires only a **single state in which the task is
solved** — no demonstrations. Train backwards: start from states near the goal, grow the
start distribution outward by Brownian motion, keeping "good starts" whose success rate sits
in a middle band, so the agent always trains at the edge of its competence.

**Deciphering t-SNE, UMAP, TriMap and PaCMAP** — JMLR 22:20-1061 · Wang, Huang, Rudin, Shaposhnik
Decomposes four dimensionality-reduction methods into loss components and ablates them,
showing that local versus global structure preservation is governed mostly by the choice of
neighbour pairs and the **ratio of attractive to repulsive forces**, not the specific loss.
Introduces PaCMAP. Its value is inoculation: it tells you which features of an embedding
plot you are entitled to believe.

**Geometric Deep Learning** — `2104.13478` · Bronstein, Bruna, Cohen, Veličković
A proto-book arguing most successful architectures instantiate one blueprint: symmetry —
invariance or equivariance to a group — plus locality, local pooling and scale separation.
The "5G": grids, groups, graphs, geodesics, gauges. CNNs are the translation case, GNNs the
permutation case, transformers the case of sets. A lens for placing an architecture, not a
method to apply.

**Courses, textbooks and tools.** Sutton & Barto (tabular methods, then function
approximation and policy gradients, then psychology and applications) is the RL reference;
CS 285, CS 234, the Hugging Face course and *RL by the Book* cover its first two-thirds in
different registers. The **Ultra-Scale Playbook** is the distributed-training counterpart —
data, tensor, sequence, pipeline and expert parallelism, ZeRO sharding, activation
recomputation and overlap across 512 GPUs — and **modded-nanogpt** is that knowledge as a
competitive speedrun, where the Muon optimizer DeepSeek-V4 now uses first drew attention.
The rest — CS 224n, CS 329S, the diffusion courses, Triton and CUDA intros, DeepWiki,
lernapparat, deep-ml, alphaXiv, Zenodo — are reference surfaces rather than reading.
