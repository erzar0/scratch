// Build figures, numbers.json and this PDF: uv run --with typst python -m plots.paper pdf
// Template: bloated-neurips (NeurIPS 2026), preprint mode. Table numbers come from
// numbers.json, never typed by hand.
#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, toprule

#let num = json("numbers.json")

#show: neurips2026.with(
  title: {
    set par(justify: false)
    set text(hyphenate: false)
    [Double Indexing: A Cheaper Entry Selector for Compressed Sparse Attention 2]
  },
  authors: (((name: "Eryk Zarębski", affl: "author"),), (author: (institution: ""),)),
  keywords: ("sparse attention", "compressed sparse attention", "hierarchical selection", "long context"),
  abstract: [
    Compressed sparse attention 2 (CSA2), from DeepSeek-V4.1, attends to only 8 compressed entries per query, but a small _indexer_ still scores every entry to choose them, so entry selection is the part of CSA2 whose cost grows with context. We look for a selector better than this pure CSA2: one that reaches the same loss (within 0.01 nats) from fewer index-key reads, or a lower loss from the same reads. On frozen CSA2 language models we test double indexing, which scores one summary key per superblock of entries and runs the indexer only inside the best superblocks. No selector lowers the loss: even an oracle that picks the entries the attention weights most is no better than pure CSA2, although it often picks different entries. Double indexing with a small router trained on the frozen model matches pure CSA2 from about 3× fewer reads at 2048 tokens and 7–8× fewer at 8192, and at 128k tokens its selection step decodes 14× faster. Summary keys that need no training also match, with smaller savings. These settings still match on a second model never used for tuning; a mixed variant that keeps more superblocks in its most sensitive layers does not, because those layers change between models. On a copy test whose answers lie half a sequence back, every double indexer loses more than on ordinary text, the training-free bound least.
  ],
  bibliography: bibliography("refs.bib"),
  accepted: none,
  aux: (font: (family: ("STIXGeneral",))),
)

// captions at most 85% of the text width; short ones stay centered
#show figure.caption: it => align(center, block(width: 85%, it))
#show table: set text(size: 8pt)
#show table: set par(justify: false)
#show raw.where(block: true): set text(size: 6.5pt)
#show raw.where(block: true): set block(breakable: false)
#show figure.where(kind: image): set figure(placement: auto)
// the template's headings are not blocks, so nothing keeps them with the next line;
// spacing as in its heading rule
#show heading: it => block(
  sticky: true,
  above: if it.level == 1 { 21.5pt } else { 17.9pt },
  below: if it.level == 1 { 15.9pt } else { 13.4pt },
  it,
)

#let booktabs(columns: auto, align: left, header, rows) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: 4pt, y: 2.6pt),
  toprule,
  table.header(..header.map(strong)),
  midrule,
  ..rows,
  botrule,
)
#let find(rows, method) = {
  let hit = rows.filter(r => r.method == method)
  if hit.len() > 0 { hit.first() } else { none }
}
#let nb(text) = text.replace(" ", "\u{a0}") // "+0.0314 ± 0.0018" stays on one line
#let bold-if(flag, body) = if flag { strong(body) } else { body }

= Introduction <sec:intro>

Softmax attention reads every past key for every query @vaswani2017, so its cost grows with context. DeepSeek's compressed sparse attention bounds the main attention instead. Its second version, CSA2 in DeepSeek-V4.1 @deepseek2026v41, compresses every 4 tokens into one main KV _entry_ and lets each query attend to only $k = 8$ entries, chosen by a small scoring network, the _indexer_ @deepseek2025v32, plus the last 32 raw tokens; layers share entries and picks with the layers after them. The indexer, however, still scores every finished entry for every query in each layer that picks: $T\/4$ dot products per decoded token and about $T^2\/8$ per prefilled sequence and layer. It is the last part of CSA2 whose cost grows with context. In our implementation the indexer's decode time grows 30-fold from 4k to 128k tokens, and a CSA2 layer's prefill, which builds the full $(T, T\/4)$ score matrix, no longer fits in our 16 GB GPU from 64k tokens.

This paper looks for an entry selector that is better than _pure CSA2_, the unmodified indexer that scores every entry. We call a selector better if it either (i) reaches the same validation loss, within +0.01 nats per token (about 1% perplexity), from fewer index-key reads, or (ii) reaches a lower loss from the same reads. Every selector is tested zero-shot: the trained CSA2 model stays frozen and only the choice of its 8 entries changes, so every difference in loss is caused by selection. @sec:background defines the terms used in figures and tables, and @sec:tested shows how each tested method works.

Our selector, _double indexing_, adds a coarse step before the indexer: it scores one summary key per _superblock_ of consecutive entries and runs the indexer only inside the few best superblocks. Hierarchical selection is established in sparse attention @tang2024quest @lu2025moba @lee2025hip, two recent works apply it to DeepSeek's token indexer @xu2026hisa @zan2026longcat, and DeepSeek-V4.1 itself narrows later indexers to a candidate pool chosen by a first indexer that still reads every entry @deepseek2026v41. We apply it to that full-range pass in every selecting layer and compare four summary keys, one of them a small router trained on the frozen model.

Before searching, we measure how much there is to find. Replacing the indexer's top-8 picks by the 8 entries the main attention would weight most, an _oracle_ that reads every entry with the full query, changes validation loss by less than 0.004 nats at both context lengths (@sec:headroom). Route (ii) is therefore closed: the trained indexer already picks as well as the attention would, and the search is for route (i), the same loss from fewer reads. This favors summaries: one summary key can stand for many entries, so a coarse pass over summaries can skip most index-key reads.

#figure(
  image("figs/route.svg", width: 100%),
  caption: [Double indexing against pure CSA2 on the frozen model 1. _How to read:_ each faint point is one setting; lines join the best settings of each way to form the summary key: zero-shot mean and bound, the learned router, and mixed (learned plus a wide bound route in needle layers). Left means less work to choose entries (index keys read per sequence and selecting layer, log scale); down means closer to pure CSA2's loss, and the dotted line is the +0.01-nat match threshold. A setting beats pure CSA2's cost where it lies below the dotted line, left of the diamond (pure CSA2 itself; its loss in the panel title).],
) <fig:route>

@fig:route summarizes the study on CSA2 language models trained on WikiText-103. Double indexing with the learned router matches pure CSA2 from about 3× fewer index-key reads at 2048 tokens and 7–8× fewer at 8192, and zero-shot summary keys match with smaller savings. On a second model, where no setting was tuned, the mean, bound and learned settings still match (@sec:seed). On a copy test whose answers lie half a sequence back, every cheaper selector loses more than on text, the zero-shot bound least (@sec:copy). Our contributions are:

- a controlled test of double indexing against pure CSA2: four summary keys and two reference selectors on frozen CSA2 models built on Hugging Face's DeepSeek-V4, with one read-cost model, paired standard errors, a held-out second model and kernel benchmarks;
- evidence that no test-time selector lowers CSA2's loss when it attends to 8 entries, and that double indexing with a router trained in 600 steps on the frozen model reaches about the same loss on text from 3–8× fewer reads, with a selection step whose decode time barely grows with context;
- two checks of why and where this breaks: which entries each selector picks compared with pure CSA2, and a copy test whose answers lie half a sequence back.

= Background and terms <sec:background>

*Softmax attention.* Query $t$ reads keys $k_1, dots, k_t$ and returns $o_t = sum_(j <= t) "softmax"_j (q_t dot k_j \/ sqrt(d)) v_j$. A causal sequence of length $T$ reads $(T+1)\/2$ keys per query on average; FlashAttention @dao2022flash makes this IO-efficient but not cheaper in reads.

*Compressed sparse attention 2.* We follow DeepSeek-V4.1 §2.3 @deepseek2026v41, scaled down:

- _Compression._ Each window of $m = 4$ tokens becomes one main KV entry, a learned per-channel softmax-weighted average of its tokens. Unlike V4's CSA @deepseek2026v4, windows do not overlap and carry no absolute position bias. Only finished entries are visible to a query.
- _Indexer._ Each entry has one index key $k_e$ of $c = 32$ channels, a projection of its main KV entry. The query has $H_I = 4$ indexer heads, $q_(t,h)$, and head weights $w_(t,h)$. The entry's score is
  $ I_(t,e) = sum_(h=1)^(H_I) w_(t,h) "ReLU"(q_(t,h) dot k_e). $ <eq:index>
- _Selection._ The $k = 8$ entries with the highest $I_(t,e)$ are kept.
- _Attention._ One softmax over the $k$ selected entries, a sliding window of $w = 32$ raw tokens and a learned attention sink @xiao2024streaming, with 8 query heads sharing one key–value head of 64 channels.
- _Cross-layer reuse._ Every layer has one of three modes. A _Full_ layer computes entries and index keys and picks its own top $k$; a _Reindex_ layer reuses the latest Full layer's entries and index keys but picks with its own indexer queries; a _Reuse_ layer reuses the entries and the latest picks. Our 8 layers run Full, Reuse, Reindex, Reuse, twice (assumed: V4.1 specifies no 8-layer model), so layers 0, 2, 4 and 6 select entries.
- _Indexer loss._ V4.1 does not say how the indexer is trained. We use DeepSeek-V3.2's loss @deepseek2025v32: cross-entropy from the attention's weight on the picked entries, summed over heads, to the softmax of their indexer scores.

*Cost model.* We count _reads_: dot products spent on selection, in units of one index-key read (4 heads × 32 channels = 128 multiply-adds). In each Full or Reindex layer pure CSA2 reads $n_t = floor(t\/4)$ keys per query; Reuse layers read none. A main-attention key read (8 heads × 64 channels) costs 4 units. Reads are summed over the sequence and averaged over the selecting layers; "times fewer reads" means pure CSA2's reads divided by the method's reads. Reads are not wall time; @sec:cost measures kernels.

*Terms.* @tab:terms explains every term the figures and tables use.

#figure(
  booktabs(
    columns: (auto, 1fr),
    ([Term], [Meaning]),
    (
      [Entry], [4 consecutive tokens compressed into one main KV entry. A CSA2 query attends to 8 entries.],
      [Indexer], [Small scorer (4 heads × 32 channels) that rates every finished entry for the current query, @eq:index.],
      [Index key], [One entry's key in the indexer. Reading one costs 128 multiply-adds and is our unit of work.],
      [Top-$k$, window, sink], [The $k = 8$ best-scored entries; the last $w = 32$ raw tokens, always attended; a learned logit that can absorb attention.],
      [Full, Reindex, Reuse], [CSA2's layer modes: compute entries and pick; reuse entries and pick; reuse entries and picks. Full and Reindex layers are the _selecting_ layers.],
      [Pure CSA2], [Unmodified CSA2: the indexer reads all index keys for every query. The baseline every selector is compared to.],
      [Selection work, reads], [Index keys read to choose entries, summed over a sequence and averaged over selecting layers. One main-attention key counts 4.],
      [Times fewer reads], [Pure CSA2's reads divided by the method's reads. Above 1× is cheaper than pure CSA2; below 1× is more expensive.],
      [Extra loss], [The method's validation loss minus pure CSA2's on the same text. Loss is the mean negative natural-log probability of the correct next token (nats per token); +0.01 nats ≈ 1% higher perplexity. Lower is better; 0 means as good as pure CSA2.],
      [Match, headline], [A method matches pure CSA2 if its extra loss is at most 0.01 nats. Its headline is the matching setting with the fewest reads.],
      [Zero-shot], [Model weights frozen; only the selector is swapped in at test time.],
      [Superblock], [$S$ consecutive entries: the group double indexing ranks first.],
      [Coarse step, fine step], [Score one summary key per superblock and keep the best $s$ superblocks; then run the indexer on their entries.],
      [Router], [A learned pooling that builds a superblock's summary key, trained with the model frozen.],
      [Oracle], [Picks the 8 entries the main attention weights most, reading every entry with the full query: a reference, not a method.],
      [Needle layer], [A selecting layer whose output the learned route changes most (above the mean over selecting layers); double indexing keeps more superblocks there.],
      [Paired SE], [Standard error: how far the extra loss could move on other text of the same kind, written as extra loss ± SE. The true value lies within two standard errors about 95% of the time. It covers text sampling only, not training seed (@sec:setup).],
      [Pick and mass recall], [Share of pure CSA2's 8 picks a method also picks; share of pure CSA2's attention weight on those shared picks.],
      [Model 1, model 2], [Two CSA2 models with 2048-token context that differ only in training seed. Settings are chosen on model 1.],
    ).flatten(),
  ),
  placement: auto,
  caption: [Terms in plain words.],
) <tab:terms>

= What we tested <sec:tested>

We test several ways to choose which entries a CSA2 query reads (@tab:map).

#figure(
  booktabs(
    columns: (auto, 1fr, auto, auto),
    ([Attention], [How entries are chosen], [Weights], [Context (tokens)]),
    (
      [Pure CSA2], [indexer scores every entry, keeps the top 8], [trained], [2048, 8192],
      [\+ double indexing], [summary keys rank superblocks (mean, bound, learned router; mixed per layer), indexer inside the best], [frozen; router trained], [2048, 8192],
      [\+ random / oracle], [random entries / the attention's own top 8], [frozen], [2048, 8192],
    ).flatten(),
  ),
  placement: auto,
  caption: [Every tested model and selector. "+" rows modify pure CSA2 in its selecting layers; frozen rows change only selection, at test time.],
) <tab:map>

*Combinations.* Every selector replaces only the indexer's scan in Full and Reindex layers; compression, cross-layer reuse, attention and every trained weight stay unchanged (@fig:pipeline). _Mixed_ double indexing combines two summary keys: the learned router in most layers and the bound in needle layers. @fig:methods shows pure CSA2 and double indexing on one toy query.

#figure(
  {
    let node(body) = box(
      stroke: 0.6pt + luma(110),
      fill: luma(246),
      inset: 4pt,
      radius: 2pt,
      width: 100%,
      height: 100%, // of the fixed first grid row
      align(center + horizon, text(size: 8pt, body)),
    )
    let to = align(center + horizon, text(size: 11pt, sym.arrow.r))
    let note(body) = align(center, text(size: 7.5pt, fill: luma(60), body))
    let panel(body) = align(left, text(size: 8pt, weight: "bold", body))
    set par(justify: false)
    stack(
      spacing: 7pt,
      panel[(a) CSA2 in a selecting layer, and where each method enters],
      grid(
        columns: (1fr, 0.25fr, 1.1fr, 0.25fr, 1.5fr, 0.25fr, 0.9fr, 0.25fr, 1.6fr),
        rows: (36pt, auto),
        row-gutter: 5pt,
        node[tokens\ $x_1, dots, x_t$], to,
        node[compress\ 4 tokens $->$ 1 entry], to,
        node[indexer, @eq:index\ scores all $n$ entries], to,
        node[top-$k$\ $k = 8$], to,
        node[softmax over $k$ entries\ + $w = 32$ raw tokens + sink],
        [], [], [], [],
        note[*double indexing*: scores fewer entries], [],
        note[*random*, *oracle*], [],
        [],
      ),
      4pt,
      panel[(b) Double indexing, in place of the indexer in (a)],
      grid(
        columns: (1.1fr, 0.25fr, 1.5fr, 0.25fr, 1.2fr, 0.25fr, 1.7fr, 0.25fr, 0.7fr),
        rows: (36pt, auto),
        row-gutter: 5pt,
        node[$n$ index keys,\ $S$ per superblock], to,
        node[pool each superblock\ into $c$ summary keys], to,
        node[coarse scores,\ keep the top $s$], to,
        node[@eq:index on the $s S$ kept entries\ + incomplete superblock], to,
        node[top-$k$],
        [], [],
        note[mean, bound or learned router; cached], [],
        note[$c G$ reads], [],
        note[$s S + (n mod S)$ reads], [], [],
      ),
    )
  },
  caption: [_How to read:_ boxes are the stages of one CSA2 query in a Full or Reindex layer, left to right; grey notes say which stage each method changes. (a) All selectors leave compression, cross-layer reuse and attention unchanged. (b) Double indexing: coarse scores over $G$ cached superblock summary keys, then exact indexer scores inside the $s$ best superblocks and the incomplete one.],
) <fig:pipeline>

#figure(
  image("figs/methods_csa.svg", width: 100%),
  caption: [How pure CSA2 and double indexing choose 8 of 32 entries (illustration on synthetic scores, not model data). _How to read:_ each square is an entry, the most recent on the right; blue squares were read (darker = higher score), white squares were not; triangles mark pure CSA2's top 8 and black frames the method's final picks. Right: reads in index-key units and how many of pure CSA2's 8 picks the method finds.],
) <fig:methods>

Notation: a query sees $n$ finished entries, grouped into superblocks of $S$ consecutive entries; $G = floor(n\/S)$ superblocks are complete and $s$ are kept.

== Double indexing <sec:double>

Pure CSA2 reads all $n$ index keys per query to keep only $k = 8$. If the kept entries fall into a few regions of the context, a coarse pass over regions followed by an exact pass inside the best regions recovers the same picks from far fewer reads. Our coarse-oracle ablation supports this premise: ranking superblocks of 16 entries by their best entry score and keeping 4 stays within 0.001 nats of pure CSA2 at both lengths (@sec:ablations).

Given a query and its $n$ index keys, double indexing first scores $c$ summary keys per complete superblock, then keeps the $s$ superblocks with the highest coarse scores, and finally applies @eq:index to the $s S$ entries they contain and to the $n mod S$ entries of the incomplete superblock (@fig:pipeline b, @fig:methods second row). The top $k$ fine scores are the selected entries, as in pure CSA2. Summary keys are computed once per completed superblock and cached. Reads per query are
$ c G + min(s, G) S + (n mod S), $
so at $n = 2048$ entries ($T = 8192$), $S = 16$, $s = 4$ and $c = 1$ give $128 + 64 = 192$ reads instead of 2048.

The summary key decides which superblocks survive, and we test four ways to form it:

- *Mean* ($c = 1$): @eq:index on the superblock's mean key. Zero-shot, but one sharp entry can be averaged away.
- *Bound* ($c = 2$), as in Quest @tang2024quest: the per-channel midpoint $mu$ and half-range $rho$ of the superblock's keys. Per head, $"ReLU"(q dot mu + |q| dot rho) >= "ReLU"(q dot k)$ for every key $k$ in the superblock, so the coarse score bounds the best entry score from above (heads with negative weight count as 0). Zero-shot, but loose when keys spread.
- *Learned* ($c = 1$): a per-layer router pools the superblock's index keys with per-channel softmax weights that depend on each key and its position, and scores the pooled key with its own copy of the indexer's query projection and head weights. It starts as mean pooling and is trained for 600 Adam steps on the frozen model, with cross-entropy from a target to the softmax over coarse scores. The target is pure CSA2's attention weight on its picked entries inside each superblock, averaged over heads and over the layers that share the picks.
- *Mixed*: learned routes in most layers and a wider bound route in _needle layers_, the selecting layers whose attention output moves more than the mean over selecting layers (relative squared error) under the learned route alone, on 2 training windows. The rule picks layers 2 and 4 on model 1 at $T = 2048$, layer 4 at $T = 8192$ and layer 4 on model 2.

A fifth mode, the *coarse oracle*, scores each superblock by its best entry score and therefore reads every entry. It is V4.1's blockwise candidate selection (§2.3.2) applied in every selecting layer, and it measures how good any summary key could be.

*Why summary keys.* A summary key covers all $S$ entries for the price of one read. Its reads are contiguous, one summary key per superblock and then $s$ runs of $S$ keys, so they map to dense memory reads (@sec:cost). It also leaves the model unchanged: the router adds one cached key per superblock and trains without touching CSA's weights. The price is a short training run for the learned router and a summary key that can miss isolated entries, which the needle-layer bound route addresses.

== Reference selectors <sec:baselines>

*Random entries* selects $k$ uniformly random finished entries and reads nothing; it measures what selection is worth. The *oracle* selects the $k$ entries the main attention would weight most if it read every entry (4 units per entry), with its softmax weights summed over heads. It shows how well any selector could match the attention; it is not a method.

= Experiments <sec:experiments>

== Setup <sec:setup>

*Data and models.* We train CSA2 language models on WikiText-103 @merity2017wikitext with GPT-2 BPE @radford2019gpt2. The model is Hugging Face's DeepSeek-V4 (`DeepseekV4ForCausalLM`, transformers 5.19.0 @wolf2020transformers) with CSA2 written as a diff on its CSA classes: $d = 512$, 8 layers, 8 query heads of 64 sharing one key–value head, query latent 256 and grouped output 2 × 128. Two parts are trimmed so that attention stays the object of study (assumed choices): the mixture of experts becomes one routed plus one shared expert, a dense SwiGLU, and the embeddings are tied. Training uses AdamW ($beta_2 = 0.95$, weight decay 0.1 on matrices), peak learning rate $10^(-3)$, 300 warm-up steps, cosine decay to 10%, bf16 autocast and gradient clipping at 1 that skips the indexer, whose loss has much larger gradients. Every model sees 98M tokens: $T = 2048$ (24000 steps × 2 sequences) or $T = 8192$ (12000 × 1). Loss is nats per BPE token, not word-level perplexity.

#figure(
  booktabs(
    columns: 5,
    align: (left, right, right, right, right),
    ([Model], [Context (tokens)], [Validation loss], [Test loss], [Training minutes]),
    num.training.map(r => (r.model, r.T, r.val, r.test, r.minutes)).flatten(),
  ),
  caption: [Base models: final loss in nats on the full validation and test splits (lower is better). Model 2 repeats model 1's recipe with another training seed.],
) <tab:training>

*Base models.* @tab:training lists the three CSA2 models. Model 2, the same recipe with another seed, ends about 0.01 nats lower than model 1, so base losses vary by about that much across seeds. The 8192-token model ends about 0.1 nats lower, as expected with 4× the context, and trains faster with half the optimizer steps. Training curves are in @fig:training.

*Evaluation.* All CSA2 selector studies are zero-shot: model weights stay frozen and only routers are trained. We evaluate the whole validation split in consecutive non-overlapping windows and report the _extra loss_: the validation loss minus that of pure CSA2 (8 entries, window 32) on the same windows. A method _matches_ pure CSA2 if its extra loss is at most 0.01 nats. A method's _headline_ setting is the one with the fewest reads that matches pure CSA2, or the one with the lowest extra loss if none matches.

*Reading ± SE.* A table entry such as $+0.0094 plus.minus 0.0008$ gives the extra loss and its paired standard error (SE). We compute the loss difference to pure CSA2 on each batch; the extra loss is their mean, and SE is their standard deviation divided by $sqrt(n)$ ($n = 60$ batches of 2 windows at $T = 2048$, $n = 30$ windows at $T = 8192$). _Paired_ means both selectors see exactly the same batches, so batch difficulty cancels out and SE measures only how consistently the method differs from pure CSA2. Treating batches as independent (assumed: neighbouring windows can come from the same article), the true extra loss on more text of the same kind lies within two standard errors of the reported value about 95% of the time, so $+0.0094 plus.minus 0.0008$ means roughly +0.008 to +0.011. Two consequences follow. An extra loss just under 0.01 is not reliably a match, because its range crosses the threshold; we therefore read matches as "about 0.01". And SE covers only which text was sampled, not which model was trained: on model 2 the mean, bound and learned headlines shift by about twice their SE, and the mixed headline by 13 times its SE, because its needle layers change (@sec:seed). Differences between trained models are the larger uncertainty.

*Selection bias and hardware.* Grids (@sec:grids) are searched on model 1's validation split, so each headline is optimistic by its own search: among many settings near the threshold, the one picked as "fewest reads within 0.01 nats" is likely one whose noise happened to fall low, so its true extra loss is probably somewhat higher than reported. Model 2 tests this: we evaluate model 1's headline settings on it without re-tuning, retraining only the routers, which are fitted to a model's weights (@sec:seed). Experiments run on one RTX 4070 Ti SUPER (16 GB) under WSL2 with PyTorch 2.14.1 (CUDA 13); benchmarks time one layer in bf16 with random weights.

== How much can selection gain? <sec:headroom>

#figure(
  booktabs(
    columns: 5,
    align: (left, right, right, right, right),
    ([Selector], [Entries attended], [Raw-token window], [Extra loss, 2048 tokens], [Extra loss, 8192 tokens]),
    num.vanilla.map(r => (r.selector, r.top_k, r.window, r.at("2048"), r.at("8192"))).flatten(),
  ),
  caption: [Can a different selection lower the loss? _How to read:_ extra loss is the validation loss minus pure CSA2's, in nats; positive means worse, lower is better. Pure CSA2 rows change how many entries it attends to, or its window of raw recent tokens, and compare with its training setting (8 entries, window 32). Random and oracle rows replace its picks and compare with pure CSA2 attending to the same number of entries. Weights frozen.],
) <tab:vanilla>

Pure CSA2's trained setting is a broad optimum in the number of entries but not in the window (@tab:vanilla). Attending to 4–16 entries costs less than 0.01 nats; 2 or 32 entries cost more, since extra entries dilute a softmax trained on 8. The window matters far more: halving or doubling it costs several times the match threshold, and a one-token window costs 2–3 nats. A wider window shows the model raw keys at distances it never saw in training (assumed). Test-time selectors therefore keep 8 entries and a window of 32.

Selection matters: random entries cost about 0.16 nats. But the trained indexer already selects as well as the attention would. The oracle, which reads every entry with the main query, is slightly worse than pure CSA2 (by less than 0.004 nats) and does not beat it with 2 entries either. No selector can therefore beat pure CSA2 in loss with 8 entries, and the remaining question is how few reads reach it.

== Main comparison <sec:selectors>

#let sel2 = num.selectors.at("2048")
#let sel8 = num.selectors.at("8192")
#let cells(r) = if r == none { ([—], [—]) } else { (bold-if(r.matched, r.ratio), nb(r.delta)) }
#let metrics = ([Times fewer reads], [Extra loss ± SE])
#figure(
  {
    table(
      columns: 5,
      align: (left, ..range(4).map(_ => right)),
      stroke: none,
      inset: (x: 5pt, y: 2.6pt),
      toprule,
      table.header(
        [],
        table.cell(colspan: 2, align: center)[*2048 tokens*],
        table.cell(colspan: 2, align: center)[*8192 tokens*],
        table.hline(start: 1, end: 3, stroke: 0.05em),
        table.hline(start: 3, end: 5, stroke: 0.05em),
        ..([Method], ..metrics, ..metrics).map(strong),
      ),
      midrule,
      ..sel2.map(r => (r.method, ..cells(r), ..cells(find(sel8, r.method)))).flatten(),
      botrule,
    )
  },
  caption: [Every CSA2 selector on the frozen model 1, at its headline setting (@tab:headline). _How to read:_ _times fewer reads_ is pure CSA2's index-key reads divided by the method's; above 1× is cheaper, and bold marks a match (extra loss at most 0.01 nats). The oracle reads every entry with the full query, so it reads more than pure CSA2. _Extra loss ± SE_ is the method's validation loss minus pure CSA2's, in nats, with its standard error; the true value lies within two standard errors about 95% of the time (@sec:setup). Lower is better.],
) <tab:selectors>

Double indexing matches pure CSA2 from fewer reads at both lengths (@tab:selectors, @fig:route). The learned router saves the most reads at both lengths. The mixed route, which keeps more superblocks in its needle layers, saves almost as many with a lower extra loss, and at 8192 tokens stays within 0.001 nats of pure CSA2 at 4× fewer reads. Among settings that read at least 4× less, the learned router has the lowest extra loss at 2048 tokens (no mixed setting reads that little) and the mixed route at 8192 (@fig:route).

Without any training, double indexing still matches pure CSA2, and the mean summary key saves more reads than the bound. Every summary key saves more at the longer context, where pure CSA2 reads 4× as many keys per query.

These differences are larger than the noise: for double indexing and the oracle, standard errors are about a tenth of the match threshold or less. Matched headlines sit under 0.01 nats by construction, so matched methods differ more in their read savings than in extra loss.

== Do the matches hold on a second model? <sec:seed>

#figure(
  booktabs(
    columns: 4,
    align: (left, left, right, right),
    ([Method], [Setting (chosen on model 1)], [Extra loss ± SE, model 1], [Extra loss ± SE, model 2]),
    num.seeds.map(r => (r.method, text(size: 7pt, r.config), bold-if(r.matched0, nb(r.seed0)), bold-if(r.matched1, nb(r.seed1)))).flatten(),
  ),
  caption: [Model 1's headline settings on model 2 (2048 tokens), with no re-tuning. _How to read:_ each row runs the same selector setting on two CSA2 models that differ only in training seed. Extra loss is in nats against each model's own pure CSA2 (lower is better); bold marks a match (at most 0.01 nats). If a row stays bold, its headline is not an artifact of searching on model 1. Routers are retrained on model 2 with the same recipe; needle layers are re-chosen by the same rule.],
) <tab:seeds>

Model 1's zero-shot and learned matches survive on model 2 (@tab:seeds); all three land slightly lower there, so searching on model 1 did not make them optimistic. The mixed route does not transfer and misses the threshold. On model 2 the rule picks only layer 4 as a needle layer, so layer 2 runs the narrow learned route, keeping one superblock, which model 1's search could afford only because layer 2 kept more. The oracle again equals pure CSA2 and random entries stay far off, so the no-headroom finding replicates. For a fixed route the match threshold thus holds to about ±0.002 nats across models, while a per-model needle choice can move the extra loss by about 0.009.

== What each method picks <sec:picks>

#figure(
  image("figs/picks.svg", width: 100%),
  caption: [Which entries each selector picks, on real text (model 1). _How to read:_ left, one query (the last token of a 2048-token validation window) in layer 4, a needle layer at both lengths and on both models: the grey band on top is pure CSA2's indexer score per entry, grey vertical lines mark pure CSA2's 8 picks, and each row shows a method's 8 picks, blue dots where they agree with pure CSA2 and orange crosses where they differ. Right: over all queries and selecting layers, the share of pure CSA2's picks each method also picks (pick recall).],
) <fig:picks>

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 5pt, y: 2.6pt),
    toprule,
    table.header(
      [],
      table.cell(colspan: 2, align: center)[*2048 tokens*],
      table.cell(colspan: 2, align: center)[*8192 tokens*],
      table.hline(start: 1, end: 3, stroke: 0.05em),
      table.hline(start: 3, end: 5, stroke: 0.05em),
      ..([Method], [Pick recall], [Mass recall], [Pick recall], [Mass recall]).map(strong),
    ),
    midrule,
    ..num.picks.filter(r => r.method != "pure CSA2").map(r => (r.method, r.pick_2048, r.mass_2048, r.pick_8192, r.mass_8192)).flatten(),
    botrule,
  ),
  caption: [Agreement with pure CSA2's picks at each method's headline setting, over all selecting layers and 8 validation windows per length. _How to read:_ pick recall is the share of pure CSA2's 8 picks the method also picks (1 = identical picks); mass recall is the share of pure CSA2's attention weight that falls on those shared picks, so it is high when the missed picks carried little weight. Higher means closer to pure CSA2.],
) <tab:picks>

Matching pure CSA2's loss does not require matching its picks (@tab:picks, @fig:picks). Double indexing with mean, bound and learned summary keys shares about three quarters of pure CSA2's picks at 2048 tokens and fewer at 8192. The oracle shares fewer still, yet its loss is about the same: many different sets of 8 entries serve the attention about equally well. Mass recall is above pick recall for every double indexer and the oracle, so the missed picks carry less attention weight than the shared ones.

Picks matter more in some layers than in others. The mixed route keeps most of pure CSA2's picks in its needle layers but under half in the other selecting layers, and still matches pure CSA2. Random entries share almost none.

== Copy stress test <sec:copy>

#let copy-cell(r, key) = nb(r.at(key))
#figure(
  {
  show table: set text(size: 7pt)
  booktabs(
    columns: 6,
    align: (left, right, right, right, right, right),
    ([Method], [Copies, 2048 tokens], [Copies, 8192 tokens], [Copies, model 2], [Text, 2048 tokens], [Text, 8192 tokens]),
    num.copy.map(r => (r.method, copy-cell(r, "2048_s0"), copy-cell(r, "8192_s0"), copy-cell(r, "2048_s1"), copy-cell(r, "normal_2048"), copy-cell(r, "normal_8192"))).flatten(),
  )
  },
  caption: [Copy stress test: windows whose second half repeats the first. _How to read:_ the pure CSA2 row gives its own loss on the repeated half, in nats (low: the model copies). Every other row gives the method's extra loss ± SE against that, at its headline setting; lower is better. Model 2 runs at 2048 tokens. _Text_ columns repeat the method's extra loss on ordinary validation text from @tab:selectors; a method that loses more on copies than on text misses distant entries.],
) <tab:copy>

Ordinary text rarely needs an entry from far back, so a selector that misses distant entries can still match pure CSA2 on it. The copy test makes distance necessary: each window is the first half of a validation window followed by an exact repeat, and loss is measured on the repeat only, whose every token is predictable from the text $T\/2$ tokens back.

Every cheaper selector loses more on copies than on text (@tab:copy). Pure CSA2's loss on the repeated half is only 0.07–0.10 nats below its loss on ordinary text: the small model copies only partly, and the test measures how much of that copying each selector keeps.

The zero-shot bound is the most consistent double indexer: it loses about the same in every copy run and has the lowest worst case of all cheaper selectors. Its coarse score is never below any of its entries' scores, so a superblock with one sharp entry keeps a high coarse score. The mixed route loses least at 8192 tokens but has its worst case on model 2, whose needle layers differ. The mean and especially the learned router lose most at 2048 tokens, the learned router nearly 10× its loss on text: a summary key pooled or trained on ordinary text misses distant entries that copying needs. The oracle stays at pure CSA2's loss on copies, so the main attention's own logits find the copied entries as well as the trained indexer does.

== Ablations <sec:ablations>

*Summary key.* The summary key causes nearly all of double indexing's error (@fig:route). With the coarse oracle, which ranks superblocks by their best entry score, keeping 4 superblocks of 16 entries stays within 0.001 nats of pure CSA2 at both lengths, so the fine step loses almost nothing. The learned router dominates the zero-shot modes at 2048 tokens and saves the most reads at 8192. Mean beats bound on text at both lengths. The needle-layer bound route lowers the extra loss at a similar read budget, most at 8192 tokens, but its layer choice does not transfer between models (@sec:seed).

== Wall time and memory <sec:cost>

#figure(
  image("figs/cost.svg", width: 100%),
  caption: [Measured kernel cost against context length, bf16, one layer with random weights, decode steps replayed as CUDA graphs. _How to read:_ top, a whole CSA2 layer versus exact softmax attention (FlashAttention): time per new token with 8 sequences, and memory for past tokens per sequence. Bottom, only the stage that picks entries: time per new token with 64 sequences, and time for a whole prompt. Lower is better; flat lines mean the cost does not grow with context.],
) <fig:cost>

#figure(
  grid(
    row-gutter: 12pt,
    booktabs(
      columns: (1fr, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      ([Layer], [Decode, 1\ sequence (µs)], [Decode, 8\ sequences (µs)], [Cache per\ sequence (MiB)], [Prefill at\ 32k (ms)]),
      num.cost.layer.map(r => (r.layer, r.decode1, r.decode8, r.cache, r.prefill32k)).flatten(),
    ),
    booktabs(
      columns: (1fr, auto, auto, auto),
      align: (left, right, right, right),
      ([Selection stage], [Decode, 64\ sequences (µs)], [Prefill\ (ms)], [Index keys read\ per token (KiB)]),
      num.cost.indexer.map(r => (r.method, r.decode, r.prefill, r.kib)).flatten(),
    ),
  ),
  caption: [Benchmarks with 131072 (128k) past tokens, except the prefill at 32k: from 64k, CSA2's $(T, T\/4)$ score matrix does not fit in 16 GB. _How to read:_ _decode_ is the time to process one new token for that many sequences at once; _cache_ is the memory for past tokens; the bottom table times only the stage that picks entries. Lower is better everywhere.],
) <tab:cost>

*CSA2 versus flash attention.* CSA2's decode cost barely grows with context, but its prefill does (@fig:cost, @tab:cost). At 128k tokens one CSA2 layer decodes 5× faster than flash attention for one sequence and 21× faster for eight, with a 43× smaller cache. At short context flash is faster; CSA2 catches up at 16k tokens for one sequence and at 4k for eight. Prefill is the weakness: CSA2 materializes the indexer's $(T, T\/4)$ score matrix, is slower than flash at every length, and from 64k needs more memory than the 16 GB GPU holds.

*Double indexing versus pure CSA2's indexer.* Double indexing removes most of the selection stage's growth in both phases (@tab:cost). At 128k tokens, with superblocks of 16 entries, it decodes 14× faster than pure CSA2's indexer, builds the prefill index 5× faster and reads about a sixteenth of the index keys per token. It decodes faster from 4k tokens on, while its prefill index, which also builds the summary keys, is slower than pure CSA2's up to 16k tokens and faster from 32k. Pure CSA2's prefill in this benchmark processes 1024 queries per step, so it avoids the memory wall; superblocks of 64 entries are timed only, without a trained router of that size.

= Related work <sec:related>

*Block-sparse attention and learned selection.* Most sparse attention methods select keys in blocks and score a summary of every block. Quest keeps per-page channel minima and maxima and selects pages by an upper bound on their attention scores @tang2024quest; MoBA gates blocks by the query's similarity to their mean-pooled keys @lu2025moba; NSA combines compressed, selected and sliding branches, choosing blocks from the compressed branch's attention @yuan2025nsa. DeepSeek's lightning indexer scores every token with a few ReLU heads @deepseek2025v32, CSA applies it to compressed entries @deepseek2026v4, and CSA2 shares entries and picks across layers @deepseek2026v41. These methods shrink the constant of selection, not its growth: each query still scores all entries, blocks or pages in every layer that selects. HiP is an exception, pruning key groups hierarchically without training @lee2025hip.

*Hierarchical indexers.* Several recent works make DeepSeek's indexer hierarchical. HISA replaces the flat token indexer of DeepSeek-V3.2 by a training-free coarse-to-fine search over pooled blocks @xu2026hisa, and LongCat Sparse Attention also indexes coarse-to-fine @zan2026longcat. Closest to us, DeepSeek-V4.1's Hierarchical Sparse Indexer @deepseek2026v41 lets the first Full layer of its decoder score every entry, gives each block of 8 entries its best entry score, and keeps the best blocks (up to 2048) as a candidate pool in which later Reindex layers search; the model is trained with the pool in place. That first layer still reads every index key; the pool needs no extra state because it reuses exact scores. Double indexing removes the full pass instead: summary keys rank superblocks before any entry is scored, in every selecting layer, with the model frozen. Our coarse oracle is V4.1's blockwise selection applied in every selecting layer, so it bounds what any summary key can reach. We do not claim the hierarchy as new; we add a router trained on the frozen model and its comparison with zero-shot mean and bound summary keys under one cost model.

*Bandits.* Multi-armed bandit selectors @auer2002ucb @evendar2006elimination @tiwari2024banditmips that sampled superblocks, indexer channels or raw tokens matched pure CSA at no more than 1.87× fewer reads in our earlier tests on V4-style CSA, so we omit them.

= Discussion: is anything better than pure CSA2? <sec:best>

On ordinary text, yes, by route (i) only: no selector lowers pure CSA2's loss, but double indexing reaches about the same loss from about 3× fewer reads at 2048 tokens and 7–8× fewer at 8192. No selector is better in every respect, though: on copies every cheaper selector loses something. We rank methods by read savings at matched loss (extra loss at most 0.01 nats), then by transfer to model 2, robustness on copies, wall time and the training they need (@tab:best).

#figure(
  placement: auto,
  booktabs(
    columns: (auto, 1fr, 1fr),
    ([Criterion], [Best], [Runner-up]),
    (
      [Fewest reads, 2048 tokens], [learned router: 3.12× fewer], [mixed: 3.03× fewer],
      [Fewest reads, 8192 tokens], [learned router: 7.65× fewer], [mixed: 7.24× fewer, with lower extra loss],
      [Fewest reads, no training], [mean: 2.70× / 6.06× fewer], [bound: 2.03× / 5.08× fewer],
      [Still matches on model 2], [learned router, at 3.12× fewer], [mean, at 2.70× fewer],
      [Copy test, worst extra loss], [bound: +0.034 nats], [mixed: +0.041 nats],
      [Decode time at 128k tokens], [learned router: 109 µs (pure CSA2: 1518 µs)], [bound: 218 µs],
    ).flatten(),
  ),
  caption: [Ranking per criterion; every entry is a double-indexing variant. _How to read:_ "3.12× fewer" means pure CSA2 reads 3.12 times as many index keys while the method's extra loss stays at most 0.01 nats; two numbers give 2048 / 8192 tokens. Decode times use superblocks of 16 entries.],
) <tab:best>

*We recommend double indexing with the learned router* where a short training run is possible. It saves the most reads while matching pure CSA2, keeps matching on model 2, and its selection step decodes 14× faster at 128k tokens. It costs one small router per selecting layer, trained in 600 steps on the frozen model; its weakness is copying. Where training is not possible, or distant entries matter most, the zero-shot bound matches with smaller savings and the lowest worst case on copies. The mixed route lowers the extra loss at 8192 tokens, but its needle layers must be re-chosen per model and its 2048-token headline does not transfer.

Two facts explain the result. With 8 entries the trained indexer ranks entries as well as the oracle (@sec:headroom), so the best a cheaper selector can do is reach pure CSA2's loss with fewer reads; it need not reproduce every pick, since the oracle shares only about 60% of pure CSA2's picks at about the same loss (@sec:picks). And a summary key covers all entries of a superblock in one read, so a coarse pass can skip most of the index keys, more of them at longer context.

Copies expose the weakness of summary keys. A summary pooled or trained on ordinary text can rank a distant superblock low even when copying needs one of its entries; the bound, which never falls below an entry's score, limits this best.

= Limitations <sec:limits>

- Small models (8 layers, width 512): magnitudes may not transfer to large models. Two training seeds at 2048 tokens and one at 8192.
- Settings are chosen on model 1's validation split. Model 2 tests them on a new model, but on the same split; there is no held-out selection set of text.
- Needle layers are chosen per model by a fixed rule on 2 training windows, so the mixed route's layers can differ between models; on model 2 they do, which breaks its 2048-token headline. The rule (error above the mean over selecting layers) and one extra mixed setting at 8192 tokens (needle layers: superblocks of 16, keep 8) were set after a first evaluation on model 1, in which no layer passed our earlier fixed threshold of 5% and the mixed grid stopped well short of the +0.01 line.
- Quality is measured up to 8192 tokens. The 128k results time one layer with random weights; they show cost, not quality. The copy test is synthetic repetition, not a long-context benchmark, and the routers never saw such input in training; training them on it may close part of the gap (unverified).
- CSA2's layer modes (Full, Reuse, Reindex, Reuse, twice), the index-key projection before RoPE and the V3.2 indexer loss are our choices where V4.1 is silent. V4.1's Hierarchical Sparse Indexer and causal encoder–decoder are not implemented.
- Selection reads are counted, not timed, except in the benchmarks. Kernels are plain PyTorch under `torch.compile`, with decode steps replayed as CUDA graphs; our model has no KV cache, so the layer benchmark decodes with a hand-written single-token CSA2 step, checked against the layer's forward.
- No softmax model is trained: CSA2 is compared only with itself.

= Conclusion <sec:conclusion>

This paper tested double indexing as an entry selector for CSA2, whose indexer is the part of CSA2 that grows with context. A selector could be better than pure CSA2 by lowering the loss or by reading less; the oracle shows the first route is closed with 8 entries, because the trained indexer already picks as well as the attention would. Along the second route, double indexing with a router trained in 600 steps matches pure CSA2 from about 3× fewer reads at 2048 tokens and 7–8× fewer at 8192, and zero-shot summary keys from 2–6× fewer; these settings also match on a second model never used for tuning, and the selection step's decode time barely grows up to 128k tokens. On copies that need distant entries, double indexing loses more than on text; the zero-shot bound loses least in the worst case. Next steps are a fused double-indexing decode kernel, a chunked CSA2 prefill without the $(T, T\/4)$ score matrix, routers trained on inputs that need distant entries, V4.1's Hierarchical Sparse Indexer as a further baseline, and the same study on larger models with quality measured beyond 8k tokens.

#show: appendix
#pagebreak()

= Appendix

== Training curves

#figure(
  placement: none,
  image("figs/training.svg", width: 100%),
  caption: [Validation loss during training. _How to read:_ x is training tokens seen, y is validation loss (lower is better); the legend gives the last in-training estimate on 40 batches, and @tab:training has full-split values.],
) <fig:training>

== Searched grids <sec:grids>

#figure(
  booktabs(
    columns: (auto, 1fr),
    ([Study], [Grid]),
    (
      [Pure CSA2], [2, 4, 8, 16 or 32 entries with window 32; window 1, 16 or 64 with 8 entries; 2 or 32 entries with window 1 or 64],
      [Double indexing, 2048 tokens], [superblocks of 8, 16 or 32 entries, keep 1, 2, 4 or 8; learned routers for superblocks of 8 and 16],
      [Double indexing, 8192 tokens], [superblocks of 16, 32 or 64 entries, keep 2, 4 or 8; learned routers for superblocks of 16 and 32],
      [Mixed, 2048 tokens], [learned: superblocks of 8, keep 1 or 2; needle layers, bound: superblocks of 8, keep 8 or 16],
      [Mixed, 8192 tokens], [learned: superblocks of 16, keep 2 or 4; needle layers, bound: superblocks of 16, keep 8, 16 or 32, or superblocks of 32, keep 16],
      [Routers], [600 Adam steps, learning rate $10^(-3)$ with cosine decay, 1 pooled key per superblock],
    ).flatten(),
  ),
  caption: [Searched settings, all on model 1.],
) <tab:grids>

== Headline settings <sec:headline-configs>

#figure(
  booktabs(
    columns: (auto, 1fr, 1fr),
    align: left,
    ([Method], [2048 tokens], [8192 tokens]),
    sel2
      .map(r => {
        let other = find(sel8, r.method)
        (r.method, text(size: 7pt, r.config), text(size: 7pt, if other == none { [—] } else { other.config }))
      })
      .flatten(),
  ),
  caption: [Headline setting of each selector (@tab:selectors). _Superblocks of 16, keep 4_: 16 entries per superblock (`superblock_size`), the 4 best superblocks kept (`kept_superblocks`). Mixed: the needle layers' setting in parentheses; random and oracle pick 8 entries.],
) <tab:headline>

== Reproducing

Code: `src/` (deepseek, model, experiments, plots, bench); results: `runs/`. Regression checks: 40 unit tests pass, among them CSA2's gathered sparse attention against Hugging Face's dense masked softmax, double indexing against a per-query loop and the benchmark's decode steps against the layers' forward.

```sh
uv sync
# train; T = 8192: SEQ=8192 BATCH=1 STEPS=12000; model 2: SEED=1
SEQ=2048 BATCH=2 STEPS=24000 uv run python -m model.train csa2
# selectors at T = 2048; repeat with SEQ=8192 BATCH=1 and routers 16 32
SEQ=2048 BATCH=2 uv run python -m experiments.eval_csa              # vanilla, random, oracle
SEQ=2048 uv run python -m experiments.train_router 8 16
SEQ=2048 BATCH=2 uv run python -m experiments.eval_route learned    # or mean, bound, oracle, mixed
SEQ=2048 BATCH=2 uv run python -m experiments.paired_se             # paired SE; SEED=1: model 2
SEQ=2048 BATCH=2 uv run python -m experiments.pick_recall           # which entries are picked
SEQ=2048 BATCH=2 uv run python -m experiments.copy_test             # copy stress test
# benchmarks; then figures, numbers.json and this PDF
uv run python -m bench.layer && uv run python -m bench.indexer
uv run --with typst python -m plots.paper pdf
```
