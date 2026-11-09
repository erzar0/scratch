"""Dense attention: every key, exactly (the reference)."""

from attention.harness import Chunk, Outcome


def dense(chunk: Chunk) -> Outcome:
    """Every key, exactly."""
    groups, _, queries, dim = chunk.query.shape
    output = chunk.logits.softmax(-1) @ chunk.value[:, None]
    reads = 2 * dim * chunk.valid.sum(-1).expand(groups, queries)
    return Outcome(output, chunk.valid.expand(groups, -1, -1), reads)
