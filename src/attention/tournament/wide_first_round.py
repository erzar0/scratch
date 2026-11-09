"""Global halving with a wider first round (the user's sort-based variant).

Every key is scored on the top `first` dimensions of the |q| order; the better
half goes on and adds `dims` more per round, until about keep keys remain for
exact attention. Keeping the sorted upper half is the same set as halving's
kthvalue cutoff, so only the first round's width differs from
Tournament("query", "halving", losers, keep). Losers drop or keep approximate
scores, as in attention/tournament/general.py.
"""

from dataclasses import dataclass

from attention.tournament.general import Tournament


@dataclass(frozen=True)
class FirstRound(Tournament):
    """Tournament whose first round reads `first` dimensions, later ones `dims`."""

    order: str = "query"
    rounds: str = "halving"
    losers: str = "drop"
    keep: int = 256
    first: int = 16

    def __post_init__(self) -> None:
        assert self.order == "query" and self.rounds == "halving"

    def read_by(self, r: int) -> int:
        """Dimensions read after round r: first, then dims more per round."""
        return self.first + self.dims * (r - 1) if r else 0
