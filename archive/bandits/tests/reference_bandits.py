"""Cut from tests/reference.py when the bandits were archived (frozen, not run).

Used CsaArrays, softmax and logsumexp from tests/reference.py.
"""

import numpy as np


def naive_bandit_candidates(
    a: CsaArrays,
    order: np.ndarray,
    route: tuple[int, int, int, int, int],
    ucb_width: float,
) -> np.ndarray:
    """Superblock bandit candidates (T', n) by definition.

    route = (S, s, first_pulls, survivors, second_pulls); order (G, S) pull order.
    """
    size, keep, first, survivors, second = route
    n_queries, n_blocks = len(a.x), len(a.indexer_keys)
    out = np.zeros((n_queries, n_blocks), bool)
    for t in range(n_queries):
        done = (t // a.block_size) // size

        def pull(g: int, i: int, t: int = t) -> float:
            return a.index_score(t, a.indexer_keys[g * size + order[g, i]])

        pulls = [[pull(g, i) for i in range(first)] for g in range(done)]
        ucb = [-(max(p) + ucb_width * np.std(p)) for p in pulls]
        surv = np.argsort(ucb, kind="stable")[:survivors]
        extra = range(first, first + second)
        best = [max(pulls[g] + [pull(g, i) for i in extra]) for g in surv]
        kept = surv[np.argsort(-np.array(best), kind="stable")[:keep]]
        for g in [*kept, done]:
            out[t, g * size : (g + 1) * size] = True
    return out


def naive_channel_bandit(
    a: CsaArrays, order: np.ndarray, top_k: int, settings: tuple[float, float, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Channel bandit survivors (T', n) and reads (T',) by definition.

    settings = (ucb_width, tolerance, dims_per_round); order: channel read order.
    Per head, after n of the c channels: estimate c·mean(q_i k_i), radius
    ucb_width · c · std(q_i k_i) / sqrt(n).
    """
    ucb_width, tolerance, step = settings
    n_queries, n_blocks = len(a.x), len(a.indexer_keys)
    n_heads, dim = a.indexer_query.shape[1:]
    scale = (n_heads * dim) ** -0.5
    survivors, reads = np.zeros((n_queries, n_blocks), bool), np.zeros(n_queries)
    for t in range(n_queries):
        active = set(range(t // a.block_size))
        w = a.head_weight[t]
        for start in range(0, dim - step, step):
            reads[t] += len(active) * step / dim
            read = order[: start + step]
            bounds = {}
            for b in active:
                upper = lower = 0.0
                for h in range(n_heads):
                    products = a.indexer_query[t, h, read] * a.indexer_keys[b, read]
                    estimate = dim * products.mean()
                    spread = np.sqrt(
                        max((products**2).mean() - products.mean() ** 2, 1e-12)
                    )
                    radius = ucb_width * dim * spread / np.sqrt(len(read))
                    low, high = max(estimate - radius, 0.0), max(estimate + radius, 0.0)
                    upper += w[h] * (high if w[h] >= 0 else low)
                    lower += w[h] * (low if w[h] >= 0 else high)
                bounds[b] = (upper * scale, lower * scale)
            lowers = sorted((low for _, low in bounds.values()), reverse=True)
            kth = lowers[top_k - 1] if len(lowers) >= top_k else -np.inf
            active = {b for b in active if bounds[b][0] >= kth - tolerance}
        reads[t] += len(active) * step / dim
        survivors[t, sorted(active)] = True
    return survivors, reads


def naive_token_bandit(
    a: CsaArrays, order: np.ndarray, top_k: int, settings: tuple[int, int, int, float]
) -> np.ndarray:
    """Token bandit candidates (T', n) by definition.

    settings = (first_pulls, second_pulls, survivors, ucb_width); order (n, block)
    token pull order. A pull: every main head's logit on one raw token key.
    """
    first, second, survivors, ucb_width = settings
    m, c = a.block_size, a.p["query_norm"].shape[0]
    n_heads = a.p["sink"].shape[0]
    n_queries, n_blocks = len(a.x), len(a.compressed_kv)
    out = np.zeros((n_queries, n_blocks), bool)
    for t in range(n_queries):
        queries = [a.main_query(t, h) for h in range(n_heads)]

        def logits(b: int, pulls: range, queries: list = queries) -> np.ndarray:
            tokens = [b * m + order[b, i] for i in pulls]
            keys = [a.main_key(a.raw_kv[j], j) for j in tokens]
            return np.array([[q @ k / np.sqrt(c) for k in keys] for q in queries])

        finished = list(range(t // m))
        first_logits = {b: logits(b, range(first)) for b in finished}
        ucb = {
            b: logsumexp(lg) - np.log(first) + np.log(m) + ucb_width * lg.std()
            for b, lg in first_logits.items()
        }
        surv = sorted(finished, key=lambda b: -ucb[b])[:survivors]
        estimate = {}
        for b in surv:
            pooled = np.concatenate(
                [first_logits[b], logits(b, range(first, first + second))], 1
            )
            estimate[b] = logsumexp(pooled) - np.log(first + second) + np.log(m)
        out[t, sorted(surv, key=lambda b: -estimate[b])[:top_k]] = True
    return out


def naive_exact(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Causal softmax attention of one head, one query at a time."""
    scale = np.sqrt(q.shape[1])
    return np.stack(
        [softmax(k[: i + 1] @ q[i] / scale) @ v[: i + 1] for i in range(len(q))]
    )


def naive_key_bandit(
    qkv: tuple[np.ndarray, np.ndarray, np.ndarray],
    order: np.ndarray,
    settings: tuple[float, float, int],
) -> tuple[np.ndarray, float]:
    """Per-key successive elimination by definition -> (output, mean reads per query).

    settings = (ucb_width, tolerance, dims_per_round); order is the channel order.
    """
    q, k, v = qkv
    ucb_width, tolerance, step = settings
    seq_len, dim = q.shape
    out, cost = np.zeros_like(v), 0.0
    for i in range(seq_len):
        active = np.ones(i + 1, bool)
        total, total_sq = np.zeros(i + 1), np.zeros(i + 1)
        for start in range(0, dim - step, step):
            channels = order[start : start + step]
            cost += active.sum() * step / dim
            products = q[i, channels][None] * k[: i + 1][:, channels] / np.sqrt(dim)
            total += products.sum(1)
            total_sq += (products**2).sum(1)
            n = start + step
            estimate = total * dim / n
            variance = np.maximum(total_sq / n * dim - estimate**2 / dim, 1e-12)
            radius = ucb_width * np.sqrt(variance * dim) / np.sqrt(n)
            best = (estimate - radius)[active].max()
            active &= estimate + radius >= best - tolerance
        cost += active.sum() * step / dim
        weight = softmax(k[: i + 1][active] @ q[i] / np.sqrt(dim))
        out[i] = weight @ v[: i + 1][active]
    return out, cost / seq_len
