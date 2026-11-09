"""Tournament attention: the first rounds inside fixed blocks, the rest global.

Block-local halving (block_tournament.py) needs no global sync but ends with a fixed
quota per block, which starves the few blocks attention needs. Here only the first
`block_rounds` rounds halve inside blocks; each block still keeps block / 2^r keys,
so a crowded block loses little. The later rounds halve the n / 2^block_rounds
survivors globally. Rounds 1-3 hold about 7/8 of the partial reads, so a kernel
could run them per block and finish the rest on the few survivors.
"""

from dataclasses import dataclass

from attention.tournament.blocks import BlockTournament


@dataclass(frozen=True)
class HybridTournament(BlockTournament):
    """BlockTournament for the first block_rounds rounds, global halving after."""

    block_rounds: int = 2

    def in_blocks(self, r: int) -> bool:
        """The first block_rounds rounds halve inside the blocks."""
        return r <= self.block_rounds
