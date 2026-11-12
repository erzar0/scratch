"""CSA2's window for record 91: per token, G shared K=V vectors instead of each head's
own K and V (DeepSeek-V4's window is one K=V head for all query heads). Each is
made like a CSA2 entry with m = 1: e = rms_norm(x w_g); its key is e rotated by the
token's RoPE row (csa2_flex.rotate), its value e itself, so a decode cache keeps G x
128 channels per window token (record 91: 6 x (96 + 128)). Heads [g H / G,
(g + 1) H / G) read vector g.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def shared_window(
    x: Tensor, w: tuple[Tensor, ...], heads: int, factor1: Tensor, factor2: Tensor
) -> tuple[Tensor, Tensor]:
    """(T, heads, D) window keys and values of x (T, C) from the G weights w (D, C):
    each group's vector rms-normed, keys rotated as csa2_flex.rotate."""
    length, groups = x.shape[0], len(w)
    e = F.rms_norm(F.linear(x, torch.cat(w)).view(length, groups, -1), (w[0].shape[0],))
    width = factor1.shape[1]
    y = e[..., :width].float()
    flip = y.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    f1, f2 = factor1[:length, None].float(), factor2[:length, None].float()
    k = torch.cat([(f1 * y + f2 * flip).type_as(e), e[..., width:]], -1)
    return k.repeat_interleave(heads // groups, 1), e.repeat_interleave(
        heads // groups, 1
    )
