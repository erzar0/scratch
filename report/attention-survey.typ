// Survey of attention / sequence-mixing layers, including fractional Laplacian attention.
// Figures: uv run --with matplotlib --with numpy --with scipy python report/figs.py
// PDF:     uv run --with typst python -c "import typst; typst.compile('report/attention-survey.typ', output='report/attention-survey.pdf')"

#set document(title: "Attention and Its Successors: A Meta-Analysis")
#set page(
  paper: "a4",
  margin: (x: 2.1cm, top: 2.3cm, bottom: 2.2cm),
  header: context if here().page() > 1 [
    #set text(size: 8pt, fill: luma(120))
    Attention and Its Successors #h(1fr) #counter(page).display()
    #v(-4pt) #line(length: 100%, stroke: 0.3pt + luma(190))
  ],
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.6em, spacing: 0.95em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): set block(spacing: 1em)
#show heading.where(level: 1): it => block(above: 1.6em, below: 0.9em, text(size: 14pt, it))
#show heading.where(level: 2): it => block(above: 1.3em, below: 0.7em, text(size: 11.5pt, it))
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(left)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure: set block(breakable: false)
#show table: set text(size: 8.5pt)
#show table: set par(justify: false)
#set table(stroke: none, inset: (x: 4pt, y: 3pt))

// palette shared with figs.py
#let c-softmax = rgb("#4c72b0")
#let c-gated = rgb("#dd8452")
#let c-delta = rgb("#c44e52")
#let c-ours = rgb("#2a9d8f")
#let c-hybrid = rgb("#8172b3")

#let rule = table.hline(stroke: 0.8pt)
#let mid = table.hline(stroke: 0.4pt)
#let note(body) = block(width: 100%, inset: 9pt, radius: 3pt, fill: luma(245), stroke: 0.5pt + luma(200), body)
#let key(body) = block(
  width: 100%, inset: (x: 9pt, y: 6pt), fill: rgb("#eef6f5"), stroke: (left: 2.2pt + c-ours),
  [*Key idea.* #body],
)
#let theorem(title, body) = block(
  width: 100%, inset: 9pt, radius: 2pt, stroke: 0.7pt + c-softmax, fill: rgb("#f1f5fb"),
  [*#title* #body],
)
#let card(..cells) = block(breakable: false, table(
  columns: (2.3cm, 1fr), stroke: 0.4pt + luma(215), inset: (x: 5pt, y: 3.5pt),
  fill: (x, _) => if x == 0 { luma(244) },
  ..cells,
))
// qualitative ratings for the capability matrix
#let yes = table.cell(fill: rgb("#d4ecd9"))[yes]
#let part = table.cell(fill: rgb("#fbeec9"))[partial]
#let no = table.cell(fill: rgb("#f6d5d5"))[no]
#let na = table.cell(fill: luma(240))[—]
#let pred(x) = table.cell(fill: luma(240))[#emph(x)]

// ================================================================ title
#align(center)[
  #v(1.5em)
  #text(size: 20pt, weight: "bold")[Attention and Its Successors]
  #v(0.3em)
  #text(size: 12.5pt)[A meta-analysis of sequence-mixing layers, \ including fractional Laplacian attention]
  #v(0.6em)
  #text(size: 9.5pt, fill: luma(80))[September 2026 · 18 papers in `papers/` · branch `frac-attention` of `transformer.py`]
]

#v(0.8em)
#note[
  *Abstract.* We cast the layers of 18 papers (2020–2025) in one notation. Every causal mixing layer
  computes $y_t = sum_(s <= t) A_(t s) v_s$, and families differ only in how $A$ is built.

  - Recurrent layers differ further in their state transition $M_t$. The literature moves along one axis,
    from identity (linear attention) to fixed decay (RetNet), then data-dependent gates (GLA, Mamba-2),
    then the delta rule (Gated DeltaNet, RWKV-7, Kimi Linear).
  - Long-convolution models (S4, H3, Hyena) trade content-dependent memory for $O(T log T)$ training.
  - Every pure subquadratic model reported here trails softmax attention on recall-heavy tasks. Hybrids
    with 10–25% attention layers close or reverse the gap.

  Our fractional Laplacian attention is a positive, power-law long convolution with two parameters per
  head, inside an H3-style gate. We confirm its tail decays as $m^(-(1+alpha))$. Its kernel has a
  log-concave core, so no positive sum of exponentials fits it. A _signed_ 32-term sum does, with
  $3 dot 10^(-4)$ relative error, which gives an $O(1)$-state decoder. The literature predicts good
  smoothing and extrapolation but weak selective recall. None of that is tested yet.
]

#v(0.4em)
#outline(depth: 2, indent: auto)
#pagebreak()

// ================================================================ 1
= Scope, sources and how to read the numbers

The corpus is the 18 PDFs in `papers/`. The softmax baseline @vaswani2017 is not among them and is cited from
general knowledge. The RWKV survey @datta2024rwkv is a secondary source, used only for lineage.

#figure(
  image("figs/timeline.svg", width: 100%),
  caption: [The corpus on a timeline (arXiv month), one lane per family. Colours are reused in every figure.],
) <fig:timeline>

#note[
  *No pooling.* The papers differ in:
  - data: Pile, SlimPajama, FineWeb-Edu, proprietary mixes;
  - tokenizers;
  - model sizes: 45M to 48B;
  - token budgets: 7B to 5.6T;
  - baseline implementations.

  For example, the same "Transformer++" at 1.3B scores 16.85 Wiki perplexity in GLA @yang2024gla and 18.53 in
  Gated DeltaNet @yang2025gdn. So *every chart panel and every table row block comes from one paper*, and the
  source table is named in the caption. Appendix C maps every number to its origin. Cross-paper statements are
  qualitative agreements between rankings. Speed claims are the authors' own.
]

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    rule,
    table.header([*Section*], [*Question it answers*]),
    mid,
    [§2 Unified view], [What do all these layers have in common, and what exactly differs?],
    [§3 Comparison], [Which family costs what, and can do what?],
    [§4 Families], [How does each model work, one equation per line?],
    [§5 Evidence], [What do the papers actually measure, in charts and tables?],
    [§6 Fractional Laplacian attention], [Where does our layer sit, what does the kernel look like, and what should we test?],
    [§7 Synthesis], [What are the trends, and which layer should one pick?],
    [Appendices], [Notation, one card per paper, provenance of every number.],
    rule,
  ),
  caption: [Reading guide.],
)

// ================================================================ 2
= A unified view

== One equation for every layer

Let $x_t in RR^d$ be the input at position $t$, and $T$ the sequence length. Per head, a layer forms

$
  q_t & = W_q x_t in RR^(d_k) \
  k_t & = W_k x_t in RR^(d_k) \
  v_t & = W_v x_t in RR^(d_v)
$

and mixes values causally:

$
  y_t & = sum_(s <= t) A_(t s) v_s \
    A & = (Q K^top) ∘ L
$ <eq:sma>

This is Mamba-2's _structured masked attention_ (SMA) @dao2024ssd. The families differ in how they fill it:

- *Softmax* replaces $∘ L$ by a row-wise softmax.
- *Linear-family layers* choose a structured mask $L$.
- *Convolutions* drop $Q K^top$ and keep a Toeplitz $L_(t s) = h_(t-s)$, gated elementwise by the input.

@fig:masks shows the resulting matrices.

#figure(
  image("figs/masks.svg", width: 78%),
  caption: [What $A$ or $L$ looks like for each family ($T = 24$, log colour scale, normalized per matrix).
  Panels (a), (d) and (e) use random inputs and are illustrative only. The key contrasts:
  - (b) Linear attention never forgets.
  - (c) Decay makes $L$ Toeplitz.
  - (d) Data-dependent gates can _reset_ memory at chosen tokens.
  - (e) A gated convolution modulates a fixed Toeplitz pattern.
  - (f) The fractional kernel is Toeplitz and row-normalized, with a heavy tail.],
) <fig:masks>

== The recurrent form

When $L$ is generated by a recurrence, the same layer runs as an RNN with a matrix state
$S_t in RR^(d_v times d_k)$:

$
  S_t & = S_(t-1) M_t + beta_t v_t k_t^top \
  y_t & = S_t q_t
$ <eq:rec>

The transition $M_t$ is the whole story of the linear family (@tab:transitions). Read top to bottom, the
table is the history of the field.

#figure(
  placement: auto,
  table(
    columns: (auto, auto, 1fr, auto),
    align: (left, left, left, left),
    rule,
    table.header([*Transition $M_t$*], [*Models*], [*Meaning*], [*Content-dep.*]),
    mid,
    [$I$], [Linear Transformer @katharopoulos2020], [never forget; $S$ only grows], [$q, k$ only],
    [$gamma I$, fixed per head], [RetNet @sun2023retnet], [exponential forgetting, multi-scale $gamma$], [$q, k$ only],
    [$a_t I$, scalar gate], [Mamba-2 @dao2024ssd], [data-dependent forgetting], [yes],
    [$"Diag"(alpha_t)$], [GLA @yang2024gla, RWKV-6 @datta2024rwkv], [per-channel data-dependent forgetting], [yes],
    [$exp(Delta_t A)$, elementwise], [Mamba @gu2023mamba], [selective SSM with $k = B_t$, $q = C_t$], [yes],
    mid,
    [$I - beta_t k_t k_t^top$], [DeltaNet (in @yang2025gdn)], [overwrite the value stored under $k_t$], [yes],
    [$a_t (I - beta_t k_t k_t^top)$], [Gated DeltaNet @yang2025gdn], [forget globally, then overwrite], [yes],
    [$"Diag"(alpha_t)(I - beta_t k_t k_t^top)$], [Kimi Delta Attention @kimi2025linear], [per-channel forget, then overwrite], [yes],
    [$"diag"(w_t) - hat(kappa)_t (a_t ⊙ hat(kappa)_t)^top$], [RWKV-7 @peng2025rwkv7], [diagonal plus rank one, separate removal key], [yes],
    [nonlinear (MLP memory)], [Titans @behrouz2024titans], [gradient step with momentum and decay], [yes],
    rule,
  ),
  caption: [Linear-family layers written as @eq:rec. $S$ uses the $d_v times d_k$ convention, so papers that use
  $S^top$ (Kimi, Mamba) are transposed. Griffin's RG-LRU @de2024griffin is the vector-state special case with
  no $q k^top$ outer product (§4.7).],
) <tab:transitions>

== Why the mask structure decides the decoder

#theorem[Theorem (Dao & Gu, Thm. 5.2, paraphrased @dao2024ssd).][
  An SMA layer $Y = (Q K^top ∘ L) V$ is an autoregressive process of bounded order, i.e. it can be
  decoded with a state of size independent of $T$, *only if* $L$ is semiseparable.
]

Three consequences are used throughout:

- Softmax is not of this form, so its cache is the full $O(T d)$ KV history.
- Decay masks $gamma^(t-s)$ and gate products $a_(s+1) dots.c a_t$ are 1-semiseparable, hence $O(1)$ state per head.
- A Toeplitz mask with a power-law profile is *not* low-order semiseparable. Such a mask, including ours, needs $O(T)$
  work per token, unless the profile is approximated by a finite sum of exponentials (§6.5).

#figure(
  {
    let bx(body, fill: white) = box(
      width: 100%, inset: 5pt, radius: 3pt, stroke: 0.6pt + luma(120), fill: fill,
      align(center, text(size: 8pt, body)),
    )
    let arrow = align(center, text(size: 9pt, fill: luma(110))[↓])
    let hot = rgb("#fde8c8")
    let ours = rgb("#d4ecd9")
    grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: 8pt,
      row-gutter: 3pt,
      grid.cell(colspan: 3, bx(fill: luma(235))[*Causal sequence mixer* \ $y_t = sum_(s <= t) A_(t s) v_s$]),
      arrow, arrow, arrow,
      bx(fill: luma(235))[*Content only* \ $A = "softmax"(Q K^top \/ sqrt(d_k))$],
      bx(fill: luma(235))[*Content ∘ mask* \ $A = (Q K^top) ∘ L$, recurrent state],
      bx(fill: luma(235))[*Mask only (LTI)* \ $A_(t s) = h_(t-s)$, gated by $x$],
      bx[MHA / MQA / GQA \ KV cache $O(T d)$],
      bx[Linear attention: $M = I$],
      bx[SSM kernels: S4, S4D],
      arrow, arrow, arrow,
      bx[MLA: latent KV cache \ (DeepSeek-V2, TransMLA)],
      bx[Fixed decay: RetNet],
      bx[\+ shift and gate: H3],
      [], arrow, arrow,
      [],
      bx[Data gates: GLA, Mamba, \ Mamba-2, RWKV-6, Griffin],
      bx[Explicit / implicit filters: \ SGConv, TNN, Hyena, M2],
      [], arrow, arrow,
      [],
      bx(fill: hot)[Delta rule: DeltaNet, GDN, \ RWKV-7, KDA; Titans (deep)],
      bx(fill: ours)[*Fractional heat kernel* \ (this work)],
      grid.cell(colspan: 3, v(2pt)),
      grid.cell(colspan: 3, bx(fill: hot)[*Hybrids*: interleave a few softmax or sliding-window layers with any
        column (H3-hybrid, Griffin, Samba, GDN-H, Mamba-2-Attention, Kimi Linear 3:1)]),
    )
  },
  caption: [Taxonomy. Arrows mean "adds expressivity" within a column, not chronology. Shaded boxes are where
  the reported evidence currently points.],
) <fig:taxonomy>

// ================================================================ 3
= Comparison at a glance

== Costs and state

@tab:master gives per-layer costs, ignoring projections and MLPs. Here:
- $d$ is the model width;
- $d_k$ is the head dimension;
- $N$ is the SSM state size;
- $r_"kv"$ is the MLA latent width.

#figure(
  placement: auto,
  table(
    columns: (auto, 2.3cm, auto, auto, auto, auto, 1fr),
    align: (left, left, left, left, left, center, left),
    rule,
    table.header([*Family*], [*Examples*], [*State / cache*], [*Train*], [*Decode / token*], [*Selective*], [*Recall evidence (§5)*]),
    mid,
    [Softmax], [@vaswani2017], [$2 T d$ KV], [$O(T^2 d)$], [$O(T d)$], [yes], [reference: exact lookup],
    [Latent KV], [MLA @meng2025transmla], [$T r_"kv"$], [$O(T^2 d)$], [$O(T r_"kv")$], [yes], [≈ softmax (avg 59.82 vs 59.85)],
    mid,
    [Linear attn.], [@katharopoulos2020], [$d_k d_v$], [$O(T d d_k)$], [$O(d d_k)$], [weak], [never forgets; poor],
    [Fixed decay], [RetNet], [$d_k d_v$], [chunkwise], [$O(d d_k)$], [partly], [FDA 14.3 vs 27.4 (GLA paper)],
    [Data gate], [GLA, Mamba-2], [$d_k d_v$], [chunkwise], [$O(d d_k)$], [yes], [MQAR good; real recall lags],
    [Selective SSM], [Mamba], [$d N$, $N = 16$], [scan $O(T d N)$], [$O(d N)$], [yes], [sel. copy 99.8; FDA 6.2],
    [Vector RNN], [Hawk, Griffin], [$d$], [scan $O(T d)$], [$O(d)$], [gates], [Hawk fails phonebook],
    [Delta rule], [GDN, RWKV-7, KDA], [$d_k d_v$], [chunkwise (WY)], [$O(d d_k)$], [yes], [best pure model on S-NIAH],
    [Deep memory], [Titans], [MLP weights], [chunkwise], [$O(|theta|)$], [yes], [S-NIAH-16K 95–98 (MAC)],
    mid,
    [LTI SSM], [S4], [$d N$], [$O(T log T d)$], [$O(d N)$], [no], [sel. copy 18.3],
    [Gated SSM], [H3], [$d N$], [$O(T log T d)$], [$O(d N)$], [gates], [AR 99.8 (short); sel. copy 57.0],
    [Long conv.], [SGConv, TNN, Hyena, M2], [$T d$ history], [$O(T log T d)$], [$O(T d)$], [gates], [Hyena AR 100 at 64K; sel. copy 30.1],
    [*Frac. heat*], [*this work*], [$T d$, or $R d$], [$O(T log T d)$], [$O(T d)$, or $O(R d)$], [gates], [untested; predicted weak],
    rule,
  ),
  caption: [Families at a glance. "Selective" means the layer decides per token what to write or erase.
  "Gates" means content enters only through elementwise gates around a fixed filter. For the fractional
  layer, $R$ is the number of exponentials in the recurrent approximation (§6.5).],
) <tab:master>

#figure(
  image("figs/cost.svg", width: 100%),
  caption: [Analytic decoding cost per layer vs context length ($d = 4096$, 32 heads of 128, $N = 16$;
  mixing only). This is a cost model, not a measurement. Softmax, MLA and the exact fractional layer grow
  linearly. Every recurrent layer, and the fractional layer with its sum-of-exponentials (SoE) decoder, is
  flat.],
) <fig:cost>

== Capability matrix

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    rule,
    table.header([*Family*], [*Parallel training*], [*$O(1)$ decode*], [*Selective memory*], [*Exact recall*], [*Length extrap.*], [*Matmul-friendly*]),
    mid,
    [Softmax], yes, no, yes, yes, no, yes,
    [MLA], yes, no, yes, yes, na, yes,
    [Linear attention], yes, yes, part, no, na, yes,
    [RetNet], yes, yes, part, part, yes, yes,
    [GLA / Mamba-2], yes, yes, yes, part, yes, yes,
    [Mamba], yes, yes, yes, part, part, part,
    [Hawk (pure RG-LRU)], yes, yes, part, no, yes, part,
    [Delta rule (GDN, KDA, RWKV-7)], yes, yes, yes, part, part, yes,
    [Titans], yes, yes, yes, part, part, part,
    [S4], yes, yes, no, no, na, part,
    [H3], yes, yes, part, part, na, part,
    [Hyena / SGConv / TNN], yes, no, part, part, yes, part,
    [M2], yes, no, part, part, na, yes,
    [*Frac. heat (ours)*], yes, pred[with SoE], no, pred[predicted no], pred[predicted yes], part,
    [Hybrids (≈ 25% attn)], yes, part, yes, yes, part, yes,
    rule,
  ),
  caption: [Qualitative capability matrix. *This is our judgment*, condensed from the evidence in §5, and grey
  cells mark missing evidence or predictions. Main sources per column:
  - extrapolation: GLA, TNN, Griffin (Transformers with RoPE fail beyond training length);
  - recall: @tab:gla, @tab:gdn, @fig:niah;
  - hardware: Mamba-2, GLA, M2 (FFT convolution runs at 3% FLOP utilization at 64K).],
) <tab:capability>

// ================================================================ 4
= The families, one equation per line

Each subsection gives the defining equations, a card with the paper's setup and headline, and the key idea.
Symbols follow Appendix A.

== Softmax attention and KV compression

$
  s_(t s) & = q_t^top k_s \/ sqrt(d_k) \
  A_(t s) & = exp(s_(t s)) / (sum_(r <= t) exp(s_(t r))) \
      y_t & = sum_(s <= t) A_(t s) v_s
$

The KV cache is exact but grows as $O(T d)$. Mamba calls this "attention does not compress its context"
@gu2023mamba. *Multi-head latent attention* (MLA) caches one latent vector per token and folds the
up-projections into the query and output ("absorb"):

$
                         c_t & = W^"DKV" x_t in RR^(r_"kv") quad "(cached)" \
                 k_t^((h)) & = W_h^"UK" c_t \
                 v_t^((h)) & = W_h^"UV" c_t \
  q_t^((h) top) k_s^((h)) & = (W_h^("UK" top) q_t^((h)))^top c_s quad "(absorbed)"
$

#card(
  [Paper], [TransMLA @meng2025transmla: converts GQA models (LLaMA-2-7B) into MLA],
  [Theory], [for a fixed cache size, any GQA is an MLA, but not the reverse],
  [Headline], [68.75% less KV cache: avg accuracy 58.20 untrained, 59.82 after 500M tokens (original 59.85)],
  [Speed], [92.97% less KV cache: 10.6× faster at 8K context (vLLM, 24 GB accelerator)],
  [Limitation], [memory is still $O(T)$; only the per-token constant shrinks],
)

#key[Softmax is the only layer here with an uncompressed, exact memory. MLA is orthogonal to everything
else: it keeps exact retrieval and shrinks the constant.]

== Linear attention

$
  S_t & = S_(t-1) + v_t phi(k_t)^top \
  z_t & = z_(t-1) + phi(k_t) \
  y_t & = (S_t phi(q_t)) / (z_t^top phi(q_t)) \
  phi(x) & = "elu"(x) + 1
$

#card(
  [Paper], [Katharopoulos et al. @katharopoulos2020, ICML 2020],
  [Headline], [image generation 317× faster (MNIST) and 4,462× faster (CIFAR-10); bits/dim 0.644 vs 0.621 for softmax],
  [Weakness], [speech phoneme error rate 8.08 vs 5.12 for softmax (WSJ)],
  [Limitation], [$S_t$ only accumulates: no forgetting, so interference grows with $T$],
)

#key[Attention without softmax is an RNN with a matrix state. Every later linear-family model adds a
decay to this recurrence.]

== Fixed decay: RetNet

$
  S_t & = gamma S_(t-1) + v_t k_t^top \
  y_t & = S_t q_t \
  Y & = (Q K^top ∘ D) V, quad D_(t s) = gamma^(t-s) bb(1)[s <= t] \
  gamma_h & = 1 - 2^(-5-h), quad h = 0, dots, H - 1
$

The model adds xPos rotations on $q, k$, GroupNorm per head and a swish output gate. It trains in chunkwise form and decodes recurrently.

#card(
  [Paper], [RetNet @sun2023retnet, 2023],
  [Scale], [up to 6.7B parameters, 100B tokens],
  [Headline], [6.7B zero-shot avg 69.51 vs 66.07 for their Transformer; 8.4× faster decoding and 70% less memory at 8K],
  [Ablation], [200M: removing the decay 26.05 → 27.86 ppl; a single $gamma$ for all heads → 27.02],
  [Limitation], [decay is data-independent: the layer cannot choose what to forget],
)

#key[A fixed exponential decay makes the mask Toeplitz and 1-semiseparable. That is cheap and stable, but blind to content.]

== Data-dependent gates: GLA and the RWKV lineage

$
  alpha_t & = sigma(x_t W_1 W_2 + b)^(1 \/ tau) in (0, 1)^(d_k), quad tau = 16 \
  S_t & = S_(t-1) "Diag"(alpha_t) + v_t k_t^top \
  y_t & = S_t q_t
$

RWKV reached the same place step by step @datta2024rwkv:

- RWKV-4 is a normalized exponential decay with vector state (the bonus term for the current token is omitted here):
  $
    "wkv"_t = (sum_(i <= t) e^(k_i - (t - i) w) v_i) / (sum_(i <= t) e^(k_i - (t - i) w))
  $
- RWKV-5 moved to a matrix state.
- RWKV-6 made the decay data-dependent: $alpha_t = exp(-exp(x_t W))$.

#card(
  [Paper], [GLA @yang2024gla, ICML 2024; SlimPajama, Mistral tokenizer],
  [Kernel], [FlashLinearAttention: chunkwise, faster than FlashAttention-2 even at 1K tokens],
  [Headline], [1.3B / 100B tokens: ppl 17.22 vs 16.85 (Transformer++); avg accuracy 51.0 vs 50.9],
  [Extrapolation], [trained at 2K, GLA and RetNet generalize to 18K; the Transformer fails beyond training length],
  [Limitation], [recall-intensive tasks lag: FDA 19.9 vs 27.4 for Transformer++],
)

#key[Making the decay a function of the input is the single largest win in GLA's ablation: training ppl goes from 23.21 to 14.77 (@fig:ablations).]

== Selective state spaces: Mamba

$
  Delta_t & = "softplus"(p + "Linear"(x_t)) \
      B_t & = "Linear"_N (x_t) \
      C_t & = "Linear"_N (x_t) \
  overline(A)_t & = exp(Delta_t A) \
  overline(B)_t & = Delta_t B_t \
      h_t & = overline(A)_t h_(t-1) + overline(B)_t x_t \
      y_t & = C_t^top h_t
$

This runs per channel, with a diagonal $A in RR^(N times N)$ and $N = 16$. Selection makes the system time-varying,
so it cannot be a convolution. It is computed by a hardware-aware parallel scan in SRAM.

#card(
  [Paper], [Mamba @gu2023mamba, 2023; Pile, 300B tokens],
  [Headline], [Mamba-2.8B zero-shot avg 63.3 vs 59.1 (Pythia-2.8B) and 61.7 (Pythia-6.9B)],
  [Synthetic], [selective copying 99.8%; induction heads extrapolate to $2^20$ tokens (trained at 256)],
  [Speed], [4–5× inference throughput vs a Transformer of similar size],
  [Limitation], [scan is not matmul-based; selection can hurt continuous signals (audio); tested ≤ 2.8B],
)

#key[Mamba's Theorem 1 shows selection generalizes RNN gating: with $N = 1$, $A = -1$, $B = 1$, it reduces to
$h_t = (1 - g_t) h_(t-1) + g_t x_t$. Content decides what enters the state.]

== State space duality: Mamba-2

$
  h_t & = a_t h_(t-1) + B_t x_t^top, quad a_t in (0, 1) "scalar per head" \
  y_t & = h_t^top C_t \
    Y & = (L ∘ C B^top) X \
  L_(t s) & = a_(s+1) a_(s+2) dots.c a_t
$

#card(
  [Paper], [Mamba-2 @dao2024ssd, ICML 2024],
  [Algorithm], [SSD block decomposition with matmuls: 2–8× faster than the Mamba scan; faster than FlashAttention-2 from 2K],
  [Headline], [2.7B / 300B tokens: Mamba-2 ppl 6.09 vs 6.13 (Transformer++); adding 6 attention layers gives 5.95, avg 61.0 vs 60.2],
  [Negative result], [kernel feature maps (Performer, cosFormer) do not beat a plain identity or swish map],
  [Limitation], [a scalar gate per head is less expressive than Mamba's per-channel $Delta$],
)

#key[Restricting the gate to a scalar makes the layer _exactly_ masked linear attention. That unlocks tensor-core training and the SMA theory in §2.]

== Gated vector recurrence: Griffin and Hawk

$
  r_t & = sigma(W_a x_t + b_a) quad "(recurrence gate)" \
  i_t & = sigma(W_x x_t + b_x) quad "(input gate)" \
  a_t & = a^(c r_t), quad a = sigma(Lambda), quad c = 8 \
  h_t & = a_t ⊙ h_(t-1) + sqrt(1 - a_t^2) ⊙ (i_t ⊙ x_t)
$

The state is a _vector_ $h_t in RR^(d_"RNN")$. Hawk stacks RG-LRU blocks, each with a width-4 conv1d. Griffin
interleaves two recurrent blocks with one local MQA layer (window 1024).

#card(
  [Paper], [Griffin / Hawk @de2024griffin, 2024; 300B tokens],
  [Headline], [Griffin-7B avg 65.8 vs 64.2 for their MQA Transformer and 65.3 for Llama-2-7B (2T tokens); Griffin-14B 69.5],
  [Recall], [phonebook lookup: Hawk fails as the book grows; Griffin is perfect up to its 1024 window, then degrades],
  [Limitation], [a vector state has too little capacity for associative recall without attention],
)

#key[Gates without an outer-product state extrapolate well but cannot store key→value pairs. That is why Griffin adds local attention.]

== Delta rule: DeltaNet and Gated DeltaNet

The delta rule treats $S$ as a key→value map trained online. It takes one SGD step, of size $beta_t$, on the
regression loss $1/2 norm(S k_t - v_t)^2$:

$
  S_t & = S_(t-1) - beta_t (S_(t-1) k_t - v_t) k_t^top \
      & = S_(t-1) (I - beta_t k_t k_t^top) + beta_t v_t k_t^top quad "(DeltaNet)" \
  S_t & = S_(t-1) (a_t (I - beta_t k_t k_t^top)) + beta_t v_t k_t^top quad "(Gated DeltaNet)" \
  y_t & = S_t q_t
$

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    rule,
    table.header([*Rule*], [*Online objective* $cal(L)(S)$], [*Update* $S_t$]),
    mid,
    [Linear attention], [$norm(S - S_(t-1))^2 - 2 ⟨S k_t, v_t⟩$], [$S_(t-1) + v_t k_t^top$],
    [Mamba-2], [same, around $a_t S_(t-1)$], [$a_t S_(t-1) + v_t k_t^top$],
    [DeltaNet], [$1/2 norm(S k_t - v_t)^2$, one SGD step], [$S_(t-1)(I - beta_t k_t k_t^top) + beta_t v_t k_t^top$],
    [Gated DeltaNet], [both], [$S_(t-1)(a_t (I - beta_t k_t k_t^top)) + beta_t v_t k_t^top$],
    [Titans], [$norm(cal(M)(k_t) - v_t)^2$, MLP $cal(M)$], [momentum SGD with weight decay (§4.11)],
    rule,
  ),
  caption: [The online-learning view, after Table 1 of @yang2025gdn. Linear attention maximizes $⟨S k, v⟩$
  without bound. The delta rule regresses $S k$ onto $v$, so writing to an existing key _replaces_ its value.],
) <tab:online>

#card(
  [Paper], [Gated DeltaNet @yang2025gdn, ICLR 2025; FineWeb-Edu, 4K context],
  [Training], [chunkwise via the WY / UT transform, so everything is matmuls; $q, k$ get a short conv, SiLU and L2 norm],
  [Headline], [1.3B / 100B tokens: ppl 16.42 and avg 55.32, vs 18.53 / 52.25 (Transformer++) and 16.56 / 54.89 (Mamba-2)],
  [Mechanism], [S-NIAH-2 at 4K: GDN 92.2%, Mamba-2 56.2% (decay forgets), DeltaNet 18.6% (cannot clear space)],
  [Limitation], [recall avg 30.6 vs 37.0 for Transformer++; slightly slower than Mamba-2],
)

#key[Decay and the delta rule are complementary. Decay clears old memory, while the delta rule writes precisely.]

== Generalized delta rule: RWKV-7

$
  w_t & = exp(-e^(-0.5) sigma(d_t)) quad "(vector decay)" \
  S_t & = S_(t-1) ("diag"(w_t) - hat(kappa)_t (a_t ⊙ hat(kappa)_t)^top) + v_t tilde(k)_t^top \
  y_t & = S_t r_t
$

The new pieces:
- $a_t$ is a vector in-context learning rate;
- $hat(kappa)_t$ is an L2-normalized _removal_ key;
- $tilde(k)_t$ is the _replacement_ key;
- $r_t$ is the receptance (query).

Token shift and a value residual are kept.

#card(
  [Paper], [RWKV-7 "Goose" @peng2025rwkv7, 2025],
  [Headline], [2.9B / 5.6T tokens: English avg 71.5 vs 71.4 for Qwen2.5-3B (18T tokens)],
  [Synthetic], [MAD avg 79.3 vs 74.5 (Transformer); pass-key perfect to 35K (2.9B)],
  [Theory], [claims recognition of all regular languages, beyond TC#super[0]],
  [Limitation], [numerically sensitive kernel; prompt sensitivity; trained from earlier checkpoints],
)

#key[Transition eigenvalues outside $[0, 1]$ buy state tracking that diagonal decays provably lack.]

== Kimi Delta Attention

$
  S_t & = S_(t-1) "Diag"(alpha_t) (I - beta_t k_t k_t^top) + beta_t v_t k_t^top \
  y_t & = S_t q_t
$

The decay is per-channel ($alpha_t in (0,1)^(d_k)$), vs a scalar in GDN. Queries and keys get a short conv, Swish
and L2 norm, with a sigmoid output gate. Layers alternate 3 KDA : 1 MLA, and the MLA layers use no positional encoding.

#card(
  [Paper], [Kimi Linear @kimi2025linear, 2025; 48B total / 3B active MoE],
  [Headline], [1.4T tokens, same recipe: MMLU-Pro 51.0 vs 47.2 (MLA) and 47.9 (GDN-H); 128K-context avg 54.5 vs 52.2 (MLA)],
  [Efficiency], [75% less KV cache; decoding at 1M tokens 1.84 vs 11.48 ms per token (6.3×); scaling-law compute efficiency 1.16×],
  [Insight], [the gated delta rule acts as a data-dependent multiplicative positional encoding that generalizes RoPE],
  [Limitation], [still needs one full-attention layer in four],
)

#key[At frontier scale, a delta-rule hybrid beats full attention at equal training recipe.]

== Test-time memory: Titans

$
  ell(cal(M); x_t) & = norm(cal(M)(k_t) - v_t)^2 \
  u_t & = eta_t u_(t-1) - theta_t nabla ell(cal(M)_(t-1); x_t) quad "(momentum: surprise)" \
  cal(M)_t & = (1 - alpha_t) cal(M)_(t-1) + u_t quad "(weight decay: forgetting)" \
  y_t & = cal(M)_t^* (q_t) quad "(read without update)"
$

A linear $cal(M)$ with no momentum is exactly the gated delta rule. Variants:
- *MAC*: memory as context for segment attention;
- *MAG*: memory gated with sliding-window attention;
- *MAL*: memory as a layer before sliding-window attention;
- *LMM*: the memory alone.

#card(
  [Paper], [Titans @behrouz2024titans, 2024],
  [Headline], [760M / 30B tokens: MAG ppl 18.61, avg 52.50 vs 25.21 / 48.69 (Transformer++) and 21.18 / 49.69 (GDN)],
  [Recall], [S-NIAH at 16K: MAC 98.4 / 97.4 / 95.2 (PK / N / W) vs Mamba-2 5.4 / 0.0 / 0.0],
  [Caveat], [baselines differ from the GDN paper (other data and setup); compare within the paper only],
  [Limitation], [slightly slower than Mamba-2 and GDN],
)

#key[Deep, momentum-driven memory extends the delta rule from a linear map to an MLP.]

== LTI state space kernels: S4

$
  x'(t) & = A x(t) + B u(t) \
  y(t) & = C x(t) \
  x_k & = overline(A) x_(k-1) + overline(B) u_k quad "(bilinear discretization)" \
  overline(K) & = (C overline(B), C overline(A) overline(B), dots, C overline(A)^(L-1) overline(B)) \
  y & = overline(K) * u quad "(FFT)"
$

Here $A = Lambda - P Q^*$ has normal plus low-rank (NPLR) structure and HiPPO initialization.

#card(
  [Paper], [S4 @gu2022s4, ICLR 2022],
  [Headline], [Long Range Arena avg 86.09, the first model to solve Path-X (96.35); Transformer 53.66],
  [LM], [WikiText-103: 20.95 vs 20.51 ppl (Transformer), with 60× faster generation],
  [Limitation], [time-invariant: selective copying 18.3% (Mamba T1)],
)

== Long convolutions: SGConv, TNN, Hyena, M2

*SGConv* @li2022sgconv builds the kernel from a few multiscale sub-kernels:

$
  k & = "concat"(k_0, k_1, dots, k_(N-1)) \
  k_i & = alpha^i "Upsample"(w_i), quad alpha = 1/2
$

Sub-kernel $w_i$ has $d$ parameters and is upsampled by a factor $2^(max(i-1, 0))$, so later sub-kernels are longer and smaller.

*TNN* @qin2023tnn uses a Toeplitz matrix generated by a relative-position MLP, with an explicit decay:

$
  y & = T x, quad T_(i j) = t_(i-j) \
  t_(i-j) & = lambda^(|i-j|) "RPE"(i - j), quad lambda = 0.99
$

*Hyena* @poli2023hyena is an order-$N$ recurrence of data-controlled gates and implicit long filters:

$
  z^1_t & = v_t \
  z^(n+1)_t & = x^n_t dot (h^n * z^n)_t, quad n = 1, dots, N \
  y_t & = z^(N+1)_t \
  h^n_t & = "Window"(t) dot "FFN"("PosEnc"(t)), quad "Window"(t) = e^(-alpha t) + b
$

In matrix form this is $y = D_x^N S_h^N dots.c D_x^1 S_h^1 v$, with diagonal $D$ and Toeplitz $S$.
H3 is the case $N = 2$.

*Monarch Mixer* @fu2023monarch replaces the FFT by structured matmuls:

$
  M & = (product_(i=1)^p P_i B_i) P_0 \
  y & = M_2 (k ⊙ M_1 x)
$

Here $P_i$ are permutations and $B_i$ block-diagonal matrices. An order-$p$ Monarch matrix costs $O(p N^((p+1)\/p))$.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, left, left, left),
    rule,
    table.header([*Model*], [*Headline*], [*Key ablation or property*], [*Limitation*]),
    mid,
    [SGConv], [LRA avg 87.17; hybrid WikiText-103 18.70 vs 18.75 (Transformer-XL)], [multiplying the kernel by a power law $1 \/ "pos"^t$: decay is crucial], [LM still needs attention layers],
    [TNN], [WikiText-103 24.67 vs 24.78 (Transformer, ≈ 45M); LRA avg 74.97], [extrapolation to 14K: 23.70 ppl, 672.72 without decay], [content-free token mixing],
    [Hyena], [AR 100% at 64K (vocab 30); WikiText-103 18.6 = Transformer], [20% fewer FLOPs at 355M for equal Pile ppl], [lower hardware utilization than FlashAttention],
    [M2], [Pile ppl 12.9 vs 13.3 (125M, 5B tokens); GLUE 79.9 with 27% fewer params], [FLOP utilization 41.4% vs 3.0% for FFT at 64K], [not optimized for inference],
    rule,
  ),
  caption: [Long-convolution models. Each row is within one paper.],
)

== Data-gated convolution: H3

$
  overline(k) & = "SSM"_("shift")(k) quad "(short convolution)" \
  y & = q ⊙ "SSM"_("diag")(overline(k) ⊙ v)
$

The shift lets a channel remember "the previous token was $a$". The diagonal SSM stores it, and the $q$ gate
reads it when $a$ reappears. That circuit is associative recall.

#card(
  [Paper], [H3 @fu2023h3, ICLR 2023],
  [Synthetic], [2-layer: induction head 100 and associative recall 99.8, vs 35.6 / 86.0 for S4D],
  [Headline], [Pile, 400B tokens: hybrid 8.8 vs 9.4 ppl (GPT-Neo-125M); 5.4 vs 5.7 at 2.7B; up to 2.4× generation throughput],
  [Limitation], [pure H3 trails the Transformer on OpenWebText (21.0 vs 20.6); the best results use 2 attention layers],
)

#key[Our layer is H3 with the shift removed and the diagonal SSM replaced by a fractional heat kernel (§6.2).
The shift is exactly the part that makes H3 recall.]

== Hybrids

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, left, left, center, left),
    rule,
    table.header([*Model*], [*Recurrent / conv layer*], [*Attention layer*], [*Ratio*], [*Result, within paper*]),
    mid,
    [H3-hybrid @fu2023h3], [H3], [full], [2 attn layers], [OpenWebText 19.6 vs 20.6 (Transformer) and 21.0 (H3)],
    [SGConv hybrid @li2022sgconv], [SGConv], [full], [—], [WikiText-103 18.70 vs 18.75],
    [Griffin @de2024griffin], [RG-LRU], [local MQA, window 1024], [2 : 1], [7B avg 65.8 vs 63.8 (Hawk) and 64.2 (MQA)],
    [Mamba-2-Attention @dao2024ssd], [SSD], [full], [58 : 6], [2.7B ppl 5.95 vs 6.09 (Mamba-2) and 6.13 (Transformer++)],
    [GDN-H1 @yang2025gdn], [Gated DeltaNet], [sliding window], [—], [recall avg 39.0 vs 30.6 (GDN) and 37.0 (Transformer++)],
    [GDN-H2 @yang2025gdn], [GDN + Mamba-2], [sliding window], [—], [recall avg 40.1; LongBench 18.4 vs 11.0],
    [Kimi Linear @kimi2025linear], [KDA], [MLA, no positional encoding], [3 : 1], [MMLU-Pro 51.0 vs 47.2 (MLA); 6.3× faster decoding at 1M],
    rule,
  ),
  caption: [Hybrid designs. "—" means the paper does not state the ratio in a form we extracted.],
) <tab:hybrids>

// ================================================================ 5
= Evidence

Each panel and each table below reproduces numbers from a single paper, named in the panel title (T = table).
Bold marks the best entry within a table.

== Selectivity and associative recall (synthetic)

#figure(
  image("figs/synthetic.svg", width: 100%),
  caption: [(a) Selective copying requires a _selective_ inner layer. The gating architecture around it (none,
  H3, Mamba) barely matters: every S4 and Hyena variant scores 18–57%, while S6 scores 97–99.8%.
  (b) Associative recall at long range. Hyena's implicit filters stay near 100% up to 131K tokens, and softmax
  runs out of memory.],
) <fig:synthetic>

== Needle in a haystack

#figure(
  image("figs/niah.svg", width: 100%),
  caption: [Single-needle retrieval vs context length.
  - (a, b) 1.3B models from the Gated DeltaNet paper. Pure decay (Mamba-2) forgets the needle; the pure delta
    rule (DeltaNet) fails when distractors must be erased (S-NIAH-2); gated delta combines both.
  - (c) 760M models from the Titans paper. Deep test-time memory (LMM, MAC) holds to 16K.
  Panels (a–b) and (c) come from different papers and setups, so do not compare them.],
) <fig:niah>

== Recall on real data, and language modelling

#figure(
  image("figs/recall.svg", width: 100%),
  caption: [Real-data recall. (a) GLA paper, 1.3B / 100B tokens. (b, c) Gated DeltaNet paper, 1.3B / 100B tokens;
  the dashed line is Transformer++, and \* marks hybrids. On recall averages, every pure subquadratic model is
  below softmax and every hybrid is above it. On LongBench, every model beats the 4K-context Transformer++.],
) <fig:recall>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    rule,
    table.header([*1.3B, 100B tok.*], [*Wiki ppl ↓*], [*Avg acc ↑*], [*FDA*], [*SWDE*], [*SQuAD*]),
    mid,
    [Transformer++], [*16.85*], [50.9], [*27.4*], [*66.6*], [31.5],
    [RetNet], [18.64], [48.9], [14.3], [42.8], [34.7],
    [Mamba], [17.06], [50.0], [6.2], [41.4], [35.2],
    [GLA], [17.22], [*51.0*], [19.9], [50.6], [*42.6*],
    rule,
  ),
  caption: [GLA paper @yang2024gla, SlimPajama (T2, T3). Language modelling is at parity; extraction is not.],
) <tab:gla>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    rule,
    table.header([*1.3B, 100B tok.*], [*Wiki ppl ↓*], [*Avg acc ↑*], [*Recall avg ↑*], [*LongBench ↑*]),
    mid,
    [Transformer++], [18.53], [52.25], [37.0], [11.0],
    [RetNet], [19.08], [52.02], [22.9], [13.2],
    [Mamba], [17.92], [53.12], [21.0], [14.6],
    [Mamba-2], [16.56], [54.89], [29.8], [13.5],
    [DeltaNet], [17.71], [52.14], [26.2], [13.6],
    [Gated DeltaNet], [16.42], [55.32], [30.6], [16.6],
    mid,
    [Samba (hybrid)], [16.13], [54.00], [37.3], [15.9],
    [GDN-H1 (+SWA)], [16.07], [*56.40*], [39.0], [17.8],
    [GDN-H2 (+Mamba-2, SWA)], [*15.91*], [56.18], [*40.1*], [*18.4*],
    rule,
  ),
  caption: [Gated DeltaNet paper @yang2025gdn, FineWeb-Edu, 4K context (T3–T5). The recall average covers SWDE,
  SQuAD, FDA, TriviaQA, NQ and Drop.],
) <tab:gdn>

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, left, center, center),
    rule,
    table.header([*Paper*], [*Setup*], [*Metric*], [*Proposed*], [*Baseline*]),
    mid,
    [RetNet], [6.7B, 100B tokens], [zero-shot avg ↑], [69.51], [66.07 (Transformer)],
    [Mamba], [2.8B, 300B Pile tokens], [zero-shot avg ↑], [63.3], [59.1 (Pythia-2.8B)],
    [Mamba-2], [2.7B, 300B Pile tokens], [Pile ppl ↓], [6.09], [6.22 (Mamba-2.8B)],
    [Griffin], [7B, 300B tokens], [downstream avg ↑], [65.8], [64.2 (MQA Transformer)],
    [H3], [2.7B, 400B Pile tokens, hybrid], [Pile ppl ↓], [5.4], [5.7 (GPT-Neo)],
    [Hyena], [355M, 15B Pile tokens], [Pile ppl ↓], [9.2], [9.1 (GPT), at 20% more FLOPs],
    [M2], [360M-class, 15B Pile tokens], [Pile ppl ↓], [9.0], [9.1 (Transformer)],
    [Titans (MAG)], [760M, 30B tokens], [Wiki ppl ↓], [18.61], [21.18 (GDN)],
    [RWKV-7], [2.9B, 5.6T tokens], [English avg ↑], [71.5], [71.4 (Qwen2.5-3B, 18T)],
    [Kimi Linear], [48B-A3B, 1.4T tokens], [MMLU-Pro ↑], [51.0], [47.2 (MLA)],
    [TNN], [≈ 45M, WikiText-103], [test ppl ↓], [24.67], [24.78 (Transformer)],
    [S4], [249M, WikiText-103], [test ppl ↓], [20.95], [20.51 (Transformer)],
    rule,
  ),
  caption: [Headline language-modelling comparisons. Each row is one paper's own proposed-vs-baseline pair.
  Rows are *not* comparable with each other.],
) <tab:headlines>

== Hybrid ratio

#figure(
  image("figs/hybrid.svg", width: 100%),
  caption: [Perplexity vs share of attention layers. (a) Mamba-2 paper, 350M, 48 layers, 7B tokens.
  (b) Kimi Linear paper. Both curves are U-shaped, with the optimum (circled) at 12.5–25% attention (shaded).
  Pure attention is worst in (a) and second worst in (b).],
) <fig:hybrid>

== Single-ingredient ablations

#figure(
  image("figs/ablations.svg", width: 100%),
  caption: [Perplexity cost of removing one ingredient, relative to the full model. Every bar is within one
  paper; colours mark the paper. Largest effects: a data-dependent gate (GLA), selection (Mamba), and decay
  for extrapolation (TNN, off-scale). Kernel feature maps, the original linear-attention idea, _hurt_ (Mamba-2).],
) <fig:ablations>

== Efficiency claims

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    rule,
    table.header([*Paper*], [*Authors' efficiency claim*]),
    mid,
    [Linear Transformer], [317× (MNIST) and 4,462× (CIFAR-10) faster image generation than softmax],
    [S4], [60× faster generation on WikiText-103],
    [H3], [up to 2.4× generation throughput (1.3B hybrid)],
    [Hyena], [2× faster than optimized attention at 8K; 100× at 64K],
    [RetNet], [8.4× faster decoding and 70% less memory at 8K (6.7B)],
    [Mamba / Mamba-2], [4–5× inference throughput; SSD 2–8× faster than the Mamba scan],
    [M2], [41.4% FLOP utilization vs 3.0% for FFT convolution at 64K],
    [TransMLA], [10.6× at 8K context with 92.97% less KV cache],
    [Kimi Linear], [6.3× faster decoding at 1M tokens (1.84 vs 11.48 ms per token); 75% less KV cache],
    rule,
  ),
  caption: [Self-reported speedups. Baselines, kernels and hardware differ, so we do not chart them together.],
) <tab:speed>

// ================================================================ 6
= Fractional Laplacian attention (this work)

== Definition

Take the cycle graph with $n = 2 dot "max_len"$ nodes. Its Laplacian is diagonalized by Fourier modes. Per head
$h$, the layer uses the heat kernel of the fractional Laplacian $L^(alpha\/2)$, built in three steps:

$
  lambda_j & = 4 sin^2(pi j \/ n), quad j = 0, dots, n \/ 2 quad "(eigenvalues)" \
  hat(w)_j & = exp(-tau_h lambda_j^(alpha_h \/ 2)) quad "(spectral filter)" \
  w_m & = 1/n sum_(j=0)^(n-1) hat(w)_j cos(2 pi j m \/ n) quad "(kernel, one irfft)"
$ <eq:kernel>

Each head has two parameters:

$
  alpha_h & = 1 + sigma(a_h) in (1, 2) \
  tau_h & = "softplus"(b_h)
$

The layer (`frac_attention` and `causal_mix` in `transformer.py`) is:

$
  u_s & = k_s ⊙ v_s \
  c_t & = sum_(s <= t) w_(t-s) u_s quad "(FFT, zero-padded to " 2T ")" \
  N_t & = sum_(m=0)^t w_m \
  y_t & = W_o (q_t ⊙ c_t \/ N_t)
$ <eq:frac>

Initialization: $alpha = 1.5$, and $tau$ geometrically spaced in $[0.5, 32]$ across heads.

#figure(
  image("figs/kernel.svg", width: 100%),
  caption: [The fractional heat kernel, computed from @eq:kernel with $n = 4096$.
  - (a) The spectrum: smaller $alpha$ lets more high frequencies through.
  - (b) The kernel has a power-law tail for $alpha < 2$ and collapses to a Gaussian-like decay at $alpha = 2$.
    The upturn at the right edge is the cycle wrap-around.
  - (c) Compared at equal mass, it keeps more weight at long lags than RetNet's slowest decay, TNN's
    $0.99^m$ or a Hyena-style window.],
) <fig:kernel>

== Where it sits

In the notation of @eq:sma, the layer is a _row-normalized Toeplitz mask whose $Q K^top$ factor is replaced by
per-channel gates_ (@fig:masks f). Equivalently, it is H3 with head dimension 1, no shift SSM, and the fractional
heat kernel in place of the diagonal SSM.

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, center, center, center, center),
    rule,
    table.header([*Model*], [*Filter family*], [*Params / filter*], [*Tail*], [*Shift / conv*], [*Finite state*]),
    mid,
    [S4 / H3], [SSM kernel $C overline(A)^m overline(B)$], [$O(N)$], [exponential], [H3: yes], [yes],
    [RetNet], [$gamma^m$ times $q^top k$], [0 (fixed)], [exponential], [no], [yes],
    [SGConv], [multiscale sub-kernels, $1 \/ 2^i$], [$O(log T)$], [piecewise], [no], [no],
    [TNN], [RPE-MLP × $0.99^m$], [MLP (shared)], [exponential], [no], [no],
    [Hyena], [FFN(PosEnc) × $e^(-alpha m)$], [FFN (shared)], [exponential], [yes], [no],
    [*Frac. heat*], [$exp(-tau L^(alpha\/2))$ on a cycle], [*2*], [$m^(-(1+alpha))$], [*no*], [with SoE],
    rule,
  ),
  caption: [Filter comparison. "Finite state" means an exact or approximate $O(1)$-per-token recurrent decoder.],
) <tab:frac-vs>

== Properties, checked numerically

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    rule,
    table.header([*Property*], [*Why*], [*Check (`figs.py`)*]),
    mid,
    [Positive], [$exp(-tau L^(alpha\/2))$ with $alpha\/2 in (0, 1]$ is a Markov semigroup (subordinated heat semigroup)], [$min_m w_m > -10^(-12)$ for $alpha in {1.1, 1.5, 1.9}$, $tau in {0.5, 4, 32}$],
    [Mass 1], [$sum_m w_m = hat(w)_0 = 1$, since $lambda_0 = 0$], [within $10^(-4)$; the eigenvalue clamp at $10^(-12)$ causes the gap],
    [Weighted mean], [after dividing by $N_t$, @eq:frac averages $k ⊙ v$], [mask rows sum to 1; flat weights give the running mean (`transformer.py` assert)],
    [Power-law tail], [$alpha$-stable asymptotics: $w_m tilde m^(-(1+alpha))$], [fitted slopes in @tab:slopes],
    rule,
  ),
  caption: [Properties of the fractional kernel.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center,
    rule,
    table.header([*$alpha$*], [*Theory $-(1+alpha)$*], [*Fitted slope*], [*Deviation*]),
    mid,
    [1.1], [−2.10], [−2.102], [0.002],
    [1.5], [−2.50], [−2.520], [0.020],
    [1.9], [−2.90], [−2.915], [0.015],
    rule,
  ),
  caption: [Tail exponent of $w_m$: a log–log fit over $30 < m < 300$, with $tau = 4$ and $n = 4096$. The
  continuum $alpha$-stable prediction holds on the discrete cycle.],
) <tab:slopes>

- *Power-law memory from 2 parameters.* $alpha$ sets the tail exponent in $(2, 3)$ and $tau$ the core width.
  SGConv found a power-law decay crucial @li2022sgconv. TNN found decay essential for extrapolation @qin2023tnn.
  Hyena learns its filters with an FFN, while this kernel comes from two scalars.
- *Content-free in time.* $w$ depends only on $t - s$, so the layer is LTI and belongs in the convolution column of @fig:taxonomy.

== Predicted weaknesses and fixes

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    rule,
    table.header([*Weakness*], [*Evidence from the literature*], [*Candidate fix*]),
    mid,
    [No selection], [LTI inner layers score 18–57% on selective copying (@fig:synthetic a)],
      [data-dependent $tau_t$ or $alpha_t$; the mask becomes non-Toeplitz],
    [No shift], [H3 needs its shift for associative recall; every later model keeps a short conv (@fig:ablations)],
      [depthwise conv of width 3–4 on $k$, as in H3, Hyena, GDN and KDA],
    [Vector state], [Hawk fails the phonebook; matrix-state models beat Hyena and RWKV-4 on MQAR @yang2024gla],
      [multi-channel heads $q ⊙ (w * (k v^top))$, i.e. $d_k > 1$],
    [$O(T)$ decoding], [Theorem 5.2: a power-law Toeplitz mask is not low-order semiseparable],
      [signed sum of $R$ exponentials, $O(R d)$ state (§6.5)],
    [FFT efficiency], [FFT runs at 3.0% FLOP utilization at 64K @fu2023monarch],
      [chunked recurrent form after the exponential fit],
    rule,
  ),
  caption: [What the literature predicts for fractional Laplacian attention. None of this is tested on our model yet.],
) <tab:frac-weak>

== A recurrent decoder: sum of exponentials

Approximate the kernel by $R$ exponentials, then run $R$ scalar recurrences per channel, together with a
recurrent normalizer:

$
  w_m & approx sum_(r=1)^R c_r gamma_r^m \
  h_(r,t) & = gamma_r h_(r,t-1) + u_t \
  n_(r,t) & = gamma_r n_(r,t-1) + 1 \
  c_t & approx sum_(r=1)^R c_r h_(r,t) \
  N_t & approx sum_(r=1)^R c_r n_(r,t)
$ <eq:soe>

This is a diagonal real SSM, i.e. a multi-scale RetNet or S4D, whose kernel is pinned to two learned scalars
$(alpha, tau)$. It satisfies Theorem 5.2 and decodes in $O(R d)$.

*The positive fit fails.* A power law is completely monotone,
$m^(-beta) = 1 / Gamma(beta) integral_0^infinity s^(beta - 1) e^(-s m) dif s$, so one might expect a positive
mixture ($c_r >= 0$) to work. It does not. The discrete kernel has a *log-concave core* at lags 1–5 (for $alpha = 1.5$,
$tau = 4$). There the heat-kernel smoothing dominates the power law, and no positive mixture of decays is
log-concave. Letting $c_r$ take either sign fixes it (@fig:soe, @tab:soe).

#figure(
  image("figs/soe.svg", width: 100%),
  caption: [Sum-of-exponentials fits of the kernel ($alpha = 1.5$, $tau = 4$, $T = 1024$, $n = 2048$ as in
  the model). Rates lie on a geometric grid in $[0.3\/T, 3]$, fitted by relative least squares. "Output" is
  the maximum error of the normalized mixed signal $c_t \/ N_t$ for a random input channel.],
) <fig:soe>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    rule,
    table.header(
      table.cell(rowspan: 2, align: horizon)[*$R$*],
      table.cell(colspan: 2)[*Signed $c_r$*], table.cell(colspan: 2)[*$c_r >= 0$*],
      [kernel], [output], [kernel], [output],
    ),
    mid,
    [2], [1.00], [1.86], [1.00], [1.86],
    [4], [0.856], [0.774], [0.856], [0.774],
    [8], [0.433], [0.221], [0.539], [0.357],
    [16], [$3.1 dot 10^(-2)$], [$1.4 dot 10^(-2)$], [0.400], [0.205],
    [32], [*$3.0 dot 10^(-4)$*], [*$8.0 dot 10^(-5)$*], [0.341], [0.171],
    rule,
  ),
  caption: [Maximum relative error of the fit. One kernel configuration only; other $(alpha, tau, T)$ are
  untested. With signed coefficients, $R = 32$ (32 floats per channel) replaces the FFT with an error of
  $10^(-4)$.],
) <tab:soe>

== Existing result (toy, anecdotal)

The random-step task:
- each sequence counts up mod 16, with a per-sequence step $s in {1, 2, 3}$;
- sequence length 32, batch 32;
- model: 2 layers, $d = 64$, 4 heads, 500 steps at lr 0.1, nested dropout on the MLP.

Without token mixing the loss floor is $ln 3 approx 1.10$. With mixing only the first prediction is ambiguous,
so the floor is $approx ln 3 \/ 32 approx 0.034$.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    rule,
    table.header([*Mixer*], [*Eval loss, 256 MLP units*], [*Eval loss, truncated to 32*], [*Train loss at step 100*]),
    mid,
    [Softmax], [0.0454], [0.0518], [0.314],
    [Fractional heat], [*0.0416*], [*0.0434*], [*0.097*],
    rule,
  ),
  caption: [Our only model result, from `transformer.py` (`__main__`, 48 held-out sequences). Single seed and a
  tiny model. The task needs a running statistic, not recall, so it favours averaging mixers. Treat it as a
  smoke test.],
) <tab:toy>

== Proposed experiments

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr),
    align: (left, left, left, left),
    rule,
    table.header([*Task*], [*Metric*], [*Baselines*], [*Prediction from the literature*]),
    mid,
    [MQAR], [accuracy vs number of KV pairs], [softmax, H3, Mamba-2, GDN], [below H3 without a shift; near H3 with a width-4 conv on $k$],
    [Selective copying], [accuracy], [S4, Mamba], [LTI level (18–57%) unless $tau$ is made data-dependent],
    [Induction heads], [accuracy, train 256, test up to $2^14$], [softmax, Mamba], [fails without a shift],
    [Length extrapolation], [ppl vs length, train 512, test to 8K], [softmax + RoPE, RetNet], [graceful, because of the decay (TNN, GLA evidence)],
    [Small-scale LM], [ppl at 125M], [Transformer++, Hyena, GDN], [near Hyena; below GDN],
    [Decoding], [tokens/s and output error], [exact FFT vs SoE $R in {16, 32}$], [flat cost with SoE; error $< 10^(-3)$ at $R = 32$ (@tab:soe)],
    rule,
  ),
  caption: [Experiment protocol, in priority order.],
) <tab:protocol>

// ================================================================ 7
= Synthesis

== Six trends

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    rule,
    table.header([*Trend*], [*Evidence*], [*Where*]),
    mid,
    [The transition got richer; decoding cost did not], [$I → gamma I → a_t I → "Diag"(alpha_t) →$ delta rule, each step justified by a within-paper ablation, all at $O(d d_k)$ per token], [@tab:transitions, @fig:ablations],
    [Decay is the most robust ingredient], [RetNet, TNN, SGConv and GLA all lose badly without it; every model that extrapolates has one], [@fig:ablations],
    [A short conv is nearly universal], [H3 shift, Hyena, RWKV token shift, Griffin conv1d(4), GDN, KDA, Titans; removing it costs 0.9–6.4% ppl], [@fig:ablations],
    [State size bounds recall], [Mamba $N = 1 → 16$; Mamba-2 better with larger $N$ on MQAR; Hawk fails the phonebook; RWKV-7 needs width 512 for full MQAR], [@fig:niah, @fig:recall],
    [Hybrids win], [U-shaped ratio curves with the optimum at 12.5–25%; every hybrid beats its pure baselines where both are reported], [@fig:hybrid, @tab:hybrids],
    [Hardware decides adoption], [chunkwise matmul forms (SSD, FlashLinearAttention, WY) replaced scans and FFTs, which are memory-bound], [@tab:speed],
    rule,
  ),
  caption: [Trends across the corpus. These are qualitative agreements between papers, not pooled statistics.],
)

== Choosing a layer

#figure(
  {
    let q(body) = box(width: 100%, inset: 6pt, radius: 3pt, stroke: 0.7pt + luma(110), fill: luma(240),
      align(center, text(size: 8.5pt, body)))
    let a(body, c) = box(width: 100%, inset: 6pt, radius: 3pt, stroke: 0.7pt + c, fill: c.lighten(85%),
      align(center, text(size: 8.5pt, body)))
    let right(lbl) = align(center + horizon, text(size: 8pt, fill: luma(90))[#lbl →])
    let down(lbl) = align(center, text(size: 8pt, fill: luma(90))[↓ #lbl])
    grid(
      columns: (1fr, 1.4cm, 1fr),
      row-gutter: 4pt,
      q[Must decoding memory stay constant in $T$?], right[no], a([Softmax, with GQA or MLA to shrink the cache], c-softmax),
      down[yes], [], [],
      q[Does the workload need exact in-context retrieval?], right[yes], a([Hybrid: gated delta rule + 1 softmax or MLA layer in 4 (Kimi Linear, GDN-H)], c-hybrid),
      down[no], [], [],
      q[Need content-dependent forgetting or state tracking?], right[yes], a([Pure delta rule: Gated DeltaNet, KDA, RWKV-7], c-delta),
      down[no], [], [],
      q[Mostly smooth, long-range signals and long extrapolation?], right[yes], a([LTI convolution: S4, Hyena, or fractional heat with the SoE decoder], c-ours),
      down[no], [], [],
      a([Data-gated linear recurrence: Mamba-2, GLA], c-gated), [], [],
    )
  },
  caption: [A decision guide distilled from §5. It is our judgment, not a result from any single paper.],
) <fig:decision>

== Open problems

- *No shared benchmark.* Numbers across these papers cannot be pooled (§1). A common recall and LM suite at fixed compute would settle most rankings.
- *Expressivity claims.* The claims beyond TC#super[0] (RWKV-7, Titans, negative-eigenvalue DeltaNet) need independent tests.
- *Numerics.* RWKV-7 reports kernel precision sensitivity, and deep recurrences accumulate error.
- *Convolution decoders.* Long-convolution models still lack a standard fast decoder. For our kernel, @tab:soe suggests a signed SoE is enough.

#note[
  *Verdict.*
  - For a new model at scale, use a gated delta-rule layer (GDN or KDA) with 1 softmax or MLA layer in 4.
  - Use MLA when the architecture must stay pure softmax.
  - Keep fractional Laplacian attention as a cheap, extrapolating smoother with a proven 32-state decoder. Its
    next test is MQAR with a short conv on $k$, because the literature predicts it fails recall without one.
]

// ================================================================ appendices
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Notation

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    rule,
    table.header([*Symbol*], [*Meaning*]),
    mid,
    [$T$, $t$, $s$, $m$], [sequence length; query position; key position; lag $m = t - s$],
    [$d$, $d_k$, $d_v$, $H$], [model width; key and value head dimension; number of heads],
    [$q_t, k_t, v_t$], [query, key and value at position $t$],
    [$A$, $L$], [mixing matrix $y = A v$; structured mask in $A = (Q K^top) ∘ L$],
    [$S_t in RR^(d_v times d_k)$], [matrix state; $y_t = S_t q_t$],
    [$M_t$], [state transition in $S_t = S_(t-1) M_t + beta_t v_t k_t^top$],
    [$gamma$, $a_t$, $alpha_t$], [fixed decay; scalar data gate; per-channel data gate],
    [$beta_t$], [delta-rule write strength (learning rate)],
    [$N$], [SSM state size per channel],
    [$h$, $w$], [convolution kernel; fractional heat kernel],
    [$alpha_h$, $tau_h$], [fractional order and diffusion time of head $h$],
    [$R$], [number of exponentials in the SoE decoder],
    [$⊙$, $∘$, $*$], [elementwise product of vectors; elementwise product of matrices; causal convolution],
    rule,
  ),
  caption: [Notation.],
)

= One card per paper

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr),
    align: (left, left, left, left),
    rule,
    table.header([*Paper*], [*Largest setup*], [*Headline*], [*Main limitation*]),
    mid,
    [Linear Transformer @katharopoulos2020], [images, speech], [317× / 4,462× faster generation], [worse speech PER (8.08 vs 5.12)],
    [S4 @gu2022s4], [249M, WT103], [LRA 86.09; Path-X solved], [LTI: no selection],
    [SGConv @li2022sgconv], [BERT-size, WT103 hybrid], [LRA 87.17; decay principle], [LM needs attention layers],
    [H3 @fu2023h3], [2.7B, 400B Pile], [hybrid beats GPT-Neo (5.4 vs 5.7)], [pure H3 trails the Transformer],
    [Hyena @poli2023hyena], [355M, 15B Pile], [AR 100% at 64K; 20% fewer FLOPs], [FFT hardware utilization],
    [TNN @qin2023tnn], [≈ 45M, WT103], [24.67 vs 24.78; extrapolates to 14K], [content-free mixing],
    [RetNet @sun2023retnet], [6.7B, 100B], [69.51 vs 66.07 avg; 8.4× decoding], [fixed decay],
    [M2 @fu2023monarch], [360M GPT, 260M BERT], [GLUE 79.9 at −27% params], [not optimized for inference],
    [Mamba @gu2023mamba], [2.8B, 300B Pile], [63.3 vs 59.1 (Pythia-2.8B)], [scan-based; ≤ 2.8B tested],
    [GLA @yang2024gla], [1.3B, 100B SlimPajama], [parity with Transformer++; fast kernel], [recall gap (FDA 19.9 vs 27.4)],
    [Griffin @de2024griffin], [14B, 300B], [7B matches Llama-2-7B with 7× fewer tokens], [recall limited to the local window],
    [Mamba-2 @dao2024ssd], [2.7B, 300B Pile], [SSD duality; 2–8× faster than Mamba], [best results need attention],
    [RWKV survey @datta2024rwkv], [—], [lineage RWKV-4 → 6], [secondary source],
    [Gated DeltaNet @yang2025gdn], [1.3B, 100B FineWeb-Edu], [best pure model: 16.42 ppl, 55.32 avg], [recall 30.6 vs 37.0],
    [Titans @behrouz2024titans], [760M, 30B], [S-NIAH-16K 95–98%], [baselines differ; slower],
    [TransMLA @meng2025transmla], [LLaMA-2-7B conversion], [−68.75% KV at 59.82 vs 59.85], [still $O(T)$ memory],
    [RWKV-7 @peng2025rwkv7], [2.9B, 5.6T], [71.5 vs Qwen2.5-3B 71.4], [precision-sensitive kernel],
    [Kimi Linear @kimi2025linear], [48B-A3B, 1.4T], [beats MLA; 6.3× decoding at 1M], [needs 1 MLA layer in 4],
    rule,
  ),
  caption: [The 18 papers at a glance, plus the softmax baseline @vaswani2017 cited from general knowledge.],
)

= Provenance

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    rule,
    table.header([*Item in this document*], [*Source (paper, table)*]),
    mid,
    [@fig:synthetic], [(a) Mamba T1; (b) Hyena T4.2],
    [@fig:niah], [(a, b) Gated DeltaNet T2; (c) Titans T2],
    [@fig:recall, @tab:gla, @tab:gdn], [GLA T2–T3; Gated DeltaNet T3–T5],
    [@fig:hybrid], [(a) Mamba-2 T2; (b) Kimi Linear T1],
    [@fig:ablations], [GLA T4; Mamba T7, T10; Titans T5; RetNet T6; Mamba-2 T6–T7; Kimi Linear T1; TNN T10],
    [@tab:headlines], [RetNet T3; Mamba T3; Mamba-2 T1; Griffin T1; H3 T4; Hyena and M2 Pile tables (M2 T8); Titans T1; RWKV-7 T3; Kimi Linear T3; TNN T2; S4 T8],
    [@tab:speed and family cards], [abstracts and main text of each paper; H3 T7; M2 T1; TransMLA T1, T4; Kimi Linear T5],
    [@fig:kernel, @fig:soe, @tab:slopes, @tab:soe], [computed by `report/figs.py` from @eq:kernel (no training)],
    [@fig:masks, @fig:cost], [illustrative: random inputs and an analytic cost model],
    [@tab:toy], [`transformer.py`, `__main__`, branch `frac-attention`],
    rule,
  ),
  caption: [Where every number comes from. T = table in the source paper.],
)

#bibliography("refs.bib", style: "ieee", title: "References")
