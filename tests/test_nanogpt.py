import torch

from deepseek import csa2
from nanogpt import csa2_varlen as cv
from nanogpt import model as nm
from nanogpt.csa2_library import LibraryCsa2Attention
from tests.test_csa2 import TINY


def test_first_tokens_keeps_only_documents_long_enough():
    bos = 9
    tokens = torch.tensor([bos, 1, 2, 3, bos, 4, bos, 5, 6, 7, 8])
    windows = nm.first_tokens(tokens, bos, 4)
    assert windows.tolist() == [[bos, 1, 2, 3], [bos, 5, 6, 7]]
    assert windows.dtype == torch.int64


def test_varlen_csa2_reads_only_its_own_document():
    # Arrange: two documents in one row, the second starting on an entry boundary
    torch.manual_seed(0)
    layer = cv.VarlenCsa2Attention(csa2.small_config((csa2.FULL,), **TINY), 0)
    layer.sinks.data.normal_()
    first, second = 20, 16
    x = torch.randn(1, first + second, TINY["hidden_size"])
    cu = torch.tensor([0, first, first + second, first + second], dtype=torch.int32)

    # Act
    with torch.no_grad():
        packed = layer(x, cu, csa2.SharedState())
        alone = layer(x[:, first:], cu[:2], csa2.SharedState())

    # Assert: the second document sees nothing of the first
    torch.testing.assert_close(packed[:, first:], alone, rtol=1e-4, atol=1e-5)


def test_library_csa2_matches_sparse_attention():
    # Arrange: two packed documents, sinks and weights shared by both layers
    torch.manual_seed(0)
    config = csa2.small_config((csa2.FULL,), **TINY)
    sparse = cv.VarlenCsa2Attention(config, 0)
    sparse.sinks.data.normal_()
    library = LibraryCsa2Attention(config, 0)
    library.load_state_dict(sparse.state_dict())
    x = torch.randn(1, 36, TINY["hidden_size"])
    cu = torch.tensor([0, 20, 36, 36], dtype=torch.int32)

    # Act: the indexer's own projection learns only through the indexer loss
    results = []
    for layer in (sparse, library):
        inputs = x.clone().requires_grad_()
        out = layer(inputs, cu, csa2.SharedState())
        out.square().sum().backward()
        results.append((out, inputs.grad, layer.indexer.q_b_proj.weight.grad))

    # Assert
    for want, got in zip(*results, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
