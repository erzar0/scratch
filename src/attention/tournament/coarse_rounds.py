"""Shuffled blocks with coarser rounds: each round keeps 1 / cut of every block.

ShuffledBlocks halves each block every round and adds `dims` channels each time.
Here a round adds `dims` channels and then halves log2(cut) times on the same
partial logits, which keeps the top 1 / cut: the same budget in log2(cut) times
fewer rounds (one cutoff each in a kernel), every key read on fewer channels.
cut 2 is ShuffledBlocks.
"""

from dataclasses import dataclass

from attention.tournament.shuffled_blocks import ShuffledBlocks


@dataclass(frozen=True)
class CoarseRounds(ShuffledBlocks):
    """ShuffledBlocks keeping 1 / cut of each block per round (cut a power of 2)."""

    cut: int = 4

    def read_by(self, r: int) -> int:
        """Channels read after halving r: dims more every log2(cut) halvings."""
        step = self.cut.bit_length() - 1
        return self.dims * -(-r // step)
