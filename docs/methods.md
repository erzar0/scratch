# The attention versions, explained simply

This page explains every benchmarked version and every setting it has, in plain words.
Each row of a figure is one version, named as in `src/attention/registry.py`.
The name in brackets on each row, Triton, PyTorch or transformers, is the code the latency benchmark timed.

## 1. The problem

A language model writes one word (token) at a time.
To pick the next word, it looks back at every earlier word: this is attention.
With 100 000 earlier words, every new word means reading 100 000 stored entries.
That reading is slow.

Sparse attention reads only the few earlier words that matter most.
The number it keeps is the **budget**.
The hard part is finding those words without reading everything first.
Every version here is a different way to guess them cheaply.

## 2. Words used on this page

| Word | Meaning, simply |
|---|---|
| Query (q) | The question the new word asks: "which earlier words matter to me?" A list of 128 numbers. |
| Key (k) | The label each earlier word carries, so queries can find it. Also 128 numbers. |
| Value (v) | The content each earlier word hands over if it is chosen. Also 128 numbers. |
| Channel | One of the 128 numbers in a query or key. Reading 8 channels of a key = reading 8 of its 128 numbers. |
| Score | How well a key matches the query: multiply them channel by channel and add up. High score = relevant. |
| Partial score | The same sum over only some channels. A cheap first guess of the score. |
| Softmax | Turns scores into weights that add up to 1. A key with a much higher score gets almost all the weight. |
| Output | The weighted average of the values. Dense attention's output is the right answer. |
| Head, query group | The model asks several questions at once. Here 2 query heads share one set of keys and values (one KV head). Every version picks one set of keys per KV head, good for both query heads. |
| Budget | How many keys get their exact score and value read. Standard budget: 512. A few versions miss it slightly; their sections say when. One, shuffled blocks with a growing budget, keeps 512 only at 32K keys and more on longer contexts. |
| Recent window | The last few positions, always kept: recent words almost always matter. |
| Losers | Keys not chosen. Most versions ignore them; some give them a rough weight. |
| Emulation | A slow, exact float32 PyTorch copy of a version. The quality benchmark runs it. |
| Fast step | The quick version, timed by the latency benchmark. Tests check it picks the same keys as the emulation. |
| Triton, PyTorch, transformers | Triton = a custom GPU program written for this version. PyTorch = built from standard library calls. Triton is usually faster. transformers = a layer from the Hugging Face library, run as is. |

## 3. What the benchmarks measure

| Measure | Meaning, simply | Better |
|---|---|---|
| Latency | Microseconds to produce one word's attention output. | lower |
| Output error | How far the output is from dense attention's: size of the difference ÷ size of the dense output. 0 = identical; 1 = as wrong as the answer is big. | lower |
| Attention mass kept | Share of dense attention's weight that falls on the chosen keys. 1 = nothing that mattered was missed. | higher |
| Top-128 recall | Share of dense attention's 128 most-weighted keys that were chosen. | higher |
| Share of K and V read | Numbers read from memory ÷ numbers dense attention reads. | lower |

The latency benchmark times one decode step: batch 1, 8 KV heads × 2 query heads, 128 channels, bf16.
The fidelity benchmark checks 128 decode queries per run, over 10 random seeds.
Both sweep context lengths 8K–128K and budgets 128–2048.
A budget runs only where the context is at least 16 times larger, so every version runs the same pairs.

## 4. The test data

All data is generated; no real model is involved.

| Name | What it is, simply | What it tests |
|---|---|---|
| iid | Every number random. No key matters more than another. | Nothing useful: attention is spread evenly, so no small set of keys is enough. Every sparse version fails here. |
| Outlier channels | Random, except 4 channels much larger than the rest, as in real models. | Versions that read the big channels first. |
| LLM-like | Outlier channels, plus 4 "sink" keys at the start that every query likes, plus 4 "needle" keys per query that only that query likes. | Finding a few important keys among many. The most realistic case here. |
| Flat | LLM-like with the query halved. | Softer attention, spread over many keys. |
| Sharp | LLM-like with the query doubled. | Peaked attention on a few keys. |
| Local | LLM-like, but nearby keys look alike: each key's noise is the average of 16 neighbours, and each needle is a run of 16 neighbouring keys (the needle added to each key's noise, so alike but not copies), as a phrase in text is. | Block methods, which assume the keys that matter come in runs. Real text has this; the other distributions do not. |
| Many needles | LLM-like with 64 needles per query instead of 4. | Many relevant keys against a small budget. |

## 5. The versions

Each version has a table: setting, its value at the standard budget of 512, and what it means.
"Budget" always means the same thing: the keys that get their exact score and value read.

### Dense

Id: `dense`.

Read every key and every value. Exact and slow: this is the reference every error is measured against.

| Setting | Value | Meaning |
|---|---|---|
| none | | Reads all keys and values. Kernel: PyTorch. |

### Exact top-k

Id: `exact-topk`.

Score every key exactly, then read the values of only the best k.
Like grading every exam fully, then reading only the top essays.
It is no cheaper on keys, so it is a yardstick: the best any version can do by picking 512 keys and ignoring the rest.
Versions that give the losers a rough weight can beat it.

| Setting | Value | Meaning |
|---|---|---|
| k | 512 (budget) | Keys whose values are read: the k highest exact scores. Kernel: PyTorch. |

### Exact top-k, summed query

Id: `exact-topk/summed-query`.

The pick rule of our record 91 runs that beat CSA2 (src/nanogpt): add the 2 query heads into one query, score every key with it exactly, read the values of the best k.
Exact top-k adds the two heads' attention after the softmax instead, so one head's sharp peak can outvote the other head.
Adding the queries first is the same as adding the heads' raw scores.

| Setting | Value | Meaning |
|---|---|---|
| k | 512 (budget) | Keys whose values are read: the k highest summed scores. |
| Kernel | PyTorch | The nanogpt decode has a fused Triton top-k for this rule, but it takes one KV head per call; at 8 KV heads it is slower than PyTorch (measured 2026-11-12, 104 vs 41 µs at 8K keys and k = 128). |

### SparQ

Id: `sparq`.

Paper: Ribar et al., arXiv 2312.04985.
Look at the query's biggest numbers; score every key on just those channels; read the best keys fully.
Like judging every book by the 32 most important words of your question.

| Setting | Value | Meaning |
|---|---|---|
| r | 32 | Channels used for the quick score: the 32 where the query is largest (summed over the 2 query heads). |
| k | 512 (budget) | Keys with the highest quick score get the exact score and value. |
| local | 16 | The 16 most recent keys are always among them. |
| step 3 | left out | The paper can give the losers' weight to their average value. Like the paper does for shared heads, this is skipped: losers are ignored. |
| Kernel | Triton | |

### Loki

Id: `loki`.

Paper: Singhania et al., arXiv 2406.02542.
Like SparQ, but first turns the keys to face their 32 most informative directions (PCA), then scores on those.
Like squashing a photo to its main shapes before comparing it.

| Setting | Value | Meaning |
|---|---|---|
| d | 32 | Directions used for the quick score: the 32 along which the keys differ most, fitted per KV head. |
| k | 512 (budget) | Keys with the highest quick score get the exact score and value. |
| Kernel | PyTorch | |

### DeepSeek-V4 CSA

Ids: `deepseek-v4/csa`, `deepseek-v4/csa-flash`.

Paper: DeepSeek-V4 (Compressed Sparse Attention). The layer comes from the Hugging Face transformers library, unchanged.
Every 4 neighbouring keys are squeezed into one summary card; neighbouring cards overlap, so each card sums up 8 keys.
A small learned scorer (the indexer) rates every card for the new word and keeps the best 512.
The word then reads those 512 cards plus the last 128 keys as they are.
Like skimming a pile of index cards, each covering a few pages, then reading the best cards and the last few pages.

Latency only: it has no quality rows ("n/a" in those figures).
Its keys, cards and scorer are computed from the words through learned weights, so it cannot run on this page's synthetic keys.
With random weights its speed is real but its picks are not.
The library computes the score of every card, then hides the ones not picked, so its time grows with the number of cards (context ÷ 4).

| Setting | Value | Meaning |
|---|---|---|
| keep | 512 (budget) | Cards read, picked by the indexer (`index_topk`; 512 in DeepSeek-V4-Flash). |
| window | 128 | The 128 most recent keys, read as they are. |
| compression | 4 | Keys squeezed into one card. Cards overlap: each covers 8 keys. |
| heads, dim | 2, 128 | 2 query heads of 128 numbers, as in the other rows; the indexer also has 2 heads of 128. DeepSeek-V4 keeps one key per word that also serves as its value. |
| Projections | sized down | The timed step includes the layer's own query, key and output projections (the other rows start from ready queries and keys). They are made small (256 wide) so their cost stays minor. |
| Kernel | transformers | The library's eager attention, the only kind it has for this layer. Batch = 8 sequences, one per KV head of the other rows. |

DeepSeek-V4 CSA, V4-Flash size: the same layer at the library's default sizes, the attention of DeepSeek-V4-Flash.
It is much bigger than the other rows (64 query heads of 512 numbers instead of 16 of 128), so it is a real model's cost, not a like-for-like comparison.
Its projections alone take about 450 µs per step at every length (the same layer with only the 128-key window, measured once).

| Setting | Value | Meaning |
|---|---|---|
| keep, window, compression | 512, 128, 4 | As above. |
| heads, dim | 64, 512 | 64 query heads of 512 numbers, all sharing one key per word (DeepSeek-V4-Flash). The indexer has 64 heads of 128. |
| Projections | full size | Library defaults: 4096 wide, queries through a 1024-wide bottleneck, output in 8 groups of 1024. |
| Kernel | transformers | As above. Batch = 1 sequence: one new word of one text. |

### DeepSeek-V4.1 CSA2

Id: `deepseek-v4.1/csa2`.

Code: DeepSeek's V4.1-Flash-1017 inference code (`src/deepseek4.1-1017/`); `src/nanogpt/csa2_ref.py` ports it for training.
One layer like V4.1's last layers: every key is its own card (no squeezing), and it serves as its own value.
A small learned scorer (the indexer, 2 heads) rates every key for the new word through keys of its own, and the best k are read with the last 128 keys.
Each head also has a "sink", a score for reading nothing, which lets it put less weight on the keys.

Latency only, for the same reason as DeepSeek-V4 CSA: the scorer and its keys are learned, so with random weights its picks mean nothing on these synthetic keys.
Not timed: the layer's projections, cache writes, V4.1's 4-bit cache, and its candidate blocks (V4.1's later scoring layers search only 256 blocks of 8 keys that an earlier layer chose).

| Setting | Value | Meaning |
|---|---|---|
| keep | 512 (budget) | Keys picked by the indexer. |
| window | 128 | The 128 most recent keys, always read. |
| index heads | 2 × 128 | The indexer's heads and their size, as in the DeepSeek-V4 CSA row. |
| rope | 16 | The output's last 16 numbers are turned back by RoPE (DeepSeek's code). |
| Kernel | PyTorch | Batch = 8 sequences, one per KV head of the other rows, 2 query heads each. Values are not read: each key is its own value. |

### Block top-k

Id: `block-topk/flat`.

Group neighbouring keys into small blocks; judge each block by its average key; read the best blocks fully.
Like choosing chapters by a one-line summary, then reading the chosen chapters.

| Setting | Value | Meaning |
|---|---|---|
| block | 4 | Neighbouring keys per block. A block's summary is the average of its 4 keys. |
| keep | 512 (budget) | Keys read in total: the window plus (512 − 32) / 4 = 120 blocks. Only blocks that start before the window can be chosen, so none is wasted on keys the window already has. One block may straddle the window's edge, so at most 3 fewer keys are read. |
| window | 32 | The 32 most recent keys, always read. |
| Block score | | The summary's score, combined over the 2 query heads. |
| Kernel | PyTorch | |

### Double indexing

Ids: `block-topk/double-indexing-mean`, `block-topk/double-indexing-bound`.

Block top-k with a first, coarser pick: choose sections (superblocks), then blocks inside them only.
Like choosing a book's parts first, then chapters inside those parts.
Fewer summaries are scored, at the risk of a good block hiding in a dull section.

| Setting | Value | Meaning |
|---|---|---|
| block | 4 | Keys per block, as in block top-k. |
| superblock | 16 | Blocks per section: 64 keys. |
| kept superblocks | 32 (budget ÷ 16) | Sections kept in the first pick: 2048 keys to choose blocks from. The section holding the query is always added. |
| coarse = mean | Double indexing: mean | A section's summary is the average of its 64 keys. |
| coarse = bound | Double indexing: bound | A section's summary is the highest score any of its keys could have. It stores each channel's middle and half-range in the section, so a section with one great key is not averaged away (Quest, arXiv 2406.10774). |
| keep, window | 512, 32 | As in block top-k: 120 blocks that start before the window, plus the 32 most recent keys. |
| Kernel | PyTorch | |

### Tournament

A knockout contest between keys.
In round 1 every key is scored on a few channels only.
The weaker half goes home.
Each later round, survivors are scored on more channels, and half go home again.
After enough rounds, about 512 keys remain; they get the exact score and value.
At 32K keys and budget 512 this takes 6 rounds, so a winner has been read on 48 of its 128 channels before the final.
Most keys go home early, after 8 channels, so far less is read than in dense attention.

Settings shared by the tournaments; each version below changes one or two of them.

| Setting | Values | Meaning |
|---|---|---|
| order = query | default | Channels are read in order of the query's largest numbers (summed over the 2 query heads): the most telling channels first. |
| order = random | original | A fixed random order of channels. |
| rounds = halving | default | Each round the better-scoring half of all survivors goes on. |
| rounds = bracket | original | Survivors next to each other in position play each other, one-on-one, like a sports bracket. One key per stretch of positions is left. |
| channels per round | 8 | Channels added to each survivor's score per round. |
| first round | 8 or 16 | Channels read in round 1. 16 makes the first cut, which removes half the keys, more careful. |
| losers = drop | default | Keys that went home are ignored. |
| losers = approximate | | Keys that went home keep a weight from their partial score (capped at the weakest winner's), so their values still count a little. Every value is read. |
| losers = mean value | | The losers' total weight goes to the average of their values, kept as a running sum, so no loser value is read. |
| keep | 512 (budget) | Keys left after the last round. Keys tied exactly at a cut all go on, so exact copies of a key (the synthetic sinks and needles) can leave a few more: up to 514 seen at 32K. |

| Version | Id | What changes | Kernel |
|---|---|---|---|
| Tournament: global halving | `tournament/global-halving` | The default: query order, halving over all keys, 8 channels every round, losers dropped. A GPU kernel must sync all keys once per round to find each cut. | Triton |
| Tournament: blocks | `tournament/blocks` | Halving inside fixed blocks of 1024 neighbouring keys; no sync needed. Every block ends with the same number of winners (16 at 32K), so a block full of important keys loses some. Unless the length is the budget times a power of two, the last round overshoots and fewer than the budget remain (more than half). | PyTorch |
| Tournament: blocks, then global | `tournament/blocks-then-global` | Blocks for the early rounds, then the last 3 rounds over all survivors. The early rounds hold most of the reading; the last rounds fix the equal-quota problem. | Triton |
| Tournament: wide first round | `tournament/wide-first-round` | Round 1 reads 16 channels, then 8 per round. | PyTorch |
| Tournament: sampled cutoffs | `tournament/sampled-cutoffs` | Each early round finds its cut from about 4096 sampled keys, not all keys, so the GPU needs less syncing. The sample is spread over all positions. Room is kept for one-eighth extra survivors. The last 2 rounds use exact cuts. | Triton |
| Tournament: sampled cutoffs, approximate losers | `tournament/sampled-cutoffs-approximate-losers` | Sampled cutoffs, plus losers keep an approximate weight. | Triton |
| Tournament: channel groups | `tournament/channel-groups` | Keys stored in fixed groups of 8 neighbouring channels; the query orders whole groups, not single channels. Round 1 reads 16 channels (2 groups). Groups are friendlier to GPU memory, at the cost of a coarser order. | PyTorch |
| Tournament: mean-value losers | `tournament/mean-value-losers` | Round 1 reads 16 channels; losers' weight goes to their average value. | PyTorch |
| Tournament: shuffled blocks | `tournament/shuffled-blocks` | Keys are stored in a fixed random order, then split into 32 blocks per KV head that run in parallel, each keeping 16 winners. Each block is a random sample of the context, so important keys are not crowded into one block. | Triton |
| Tournament: shuffled blocks, growing budget | `tournament/shuffled-blocks-growing` | Shuffled blocks with fixed blocks of 1024 keys, each keeping 16 winners, so a longer context keeps more keys. See below. | Triton |
| Tournament: original | `tournament/original` | The first design: random channel order, bracket rounds, approximate losers. Like blocks, it ends below the budget unless the length is the budget times a power of two. | PyTorch |

#### Tournament: shuffled blocks, growing budget

Shuffled blocks splits every context into 32 blocks, so on a longer context each block is bigger but still sends 16 winners: the total stays 512.
Here every block holds 1024 keys and sends 16 winners to the final, so every 1024 new keys add 16 more winners.
Like a contest where each school sends 16 pupils: more schools, more pupils in the final.
At 32K keys it picks exactly what shuffled blocks picks; it keeps 128 keys at 8K, 2048 at 128K.
So against the other rows it reads fewer keys below 32K and more above.
On short contexts it is not dense: 1024 keys still keep 16.

| Setting | Value | Meaning |
|---|---|---|
| keep | 512 (budget) | Keys kept at the reference length. |
| block | 1024 | Key slots per block, at every length. |
| reference | 32768 | The length at which it keeps exactly the budget. |
| winners per block | 16 (budget × 1024 ÷ 32768) | Keys each block sends to attention. Keys kept = 16 × blocks. |

### Bandit

Ids: `bandit/exact-bound`, `bandit/gaussian-bound`.

Keep a range for each key: its lowest and highest possible score, given the channels read so far.
Each round reads 8 more channels of every surviving key, narrowing every range.
A key whose highest possible score is below the 512th-best lowest score cannot win, so it is dropped.
Like a race where a runner who can no longer reach the podium leaves the track.
The last round reads every channel, so the final pick among the survivors is exact.

| Setting | Values | Meaning |
|---|---|---|
| order | query | Channels in order of the query's largest numbers, as in the tournament. |
| channels per round | 8 | Channels read per round for every surviving key. |
| keep | 512 (budget) | Keys that get the exact score and value. |
| bound = exact | Bandit: exact bound | The range is the largest the unread channels could possibly add (a guaranteed limit, from the sizes of the unread parts of query and key). |
| bound = gaussian | Bandit: Gaussian bound | The range is a typical size of the unread part, from each channel's spread over the keys. Not guaranteed, so narrower. |
| width | 0.25 (exact), 1.0 (gaussian) | Scales the range. Below 1, keys are dropped sooner, but a true top key may be dropped too. |
| Kernel | PyTorch | Masks over all keys rather than truly skipping them, so its latency is an upper bound. |

## 6. How to read the figures

- One row per version, in the order of this page. Thin lines separate the families.
- One panel per data distribution (or per measure in the budget figure).
- One dot per context length: darker = longer. In the budget figure, one dot per budget: darker = larger.
- The x axis is logarithmic: each grid step multiplies the value.
- Whiskers: latency, fastest to slowest repeat; quality, ± one standard error over seeds.
- Dense has 0 error by definition, so its row says "0 (reference)" instead of a dot.
- "n/a": the benchmark does not run that version (both DeepSeek-V4 CSA rows have latency only).
- Captions with every setting (GPU, shapes, repeats, seeds) are in `runs/benchmarks/<suite>/tables.md`.
