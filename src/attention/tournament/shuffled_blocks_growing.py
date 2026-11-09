"""Shuffled block halving with fixed blocks: the same winners per block at any length.

Tournament: shuffled blocks splits each KV head into 32 blocks, so blocks grow with
the context and the total stays `keep`. Here every block holds `block` key slots and
sends `keep * block / reference` winners to attention, so the total grows with the
context, as a cache grows during decoding: `keep` keys at `reference` length, half
at half the length, twice at twice. At `reference` it picks exactly what Tournament:
shuffled blocks picks (the same kernel split, 32 blocks of 1024 at 32K).
"""

from dataclasses import dataclass

from attention.harness import Chunk, Outcome
from attention.tournament.shuffled_blocks import ShuffledBlocks


@dataclass(frozen=True)
class GrowingShuffledBlocks:
    """ShuffledBlocks with `block`-slot blocks, each keeping winners() keys."""

    keep: int = 512  # keys kept at `reference` length
    block: int = 1024
    reference: int = 32768

    def winners(self) -> int:
        """Keys each block sends to attention."""
        assert self.keep * self.block % self.reference == 0
        return self.keep * self.block // self.reference

    def split(self, n: int) -> tuple[int, int]:
        """(keys kept, blocks) at length n."""
        blocks = -(-n // self.block)
        return self.winners() * blocks, blocks

    def __call__(self, chunk: Chunk) -> Outcome:
        """Shuffled block halving at this chunk's length."""
        keep, _ = self.split(chunk.key.shape[1])
        return ShuffledBlocks(block=self.block, keep=keep)(chunk)
