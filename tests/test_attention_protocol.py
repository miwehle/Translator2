import torch
import torch.nn as nn

from translator.attention import (
    make_attention_factory,
    AttentionProtocol,
)
from translator.model import Seq2Seq


def _dummy_batch(batch_size: int = 2, src_len: int = 5, tgt_len: int = 6):
    src = torch.tensor(
        [
            [2, 7, 8, 0, 0],
            [2, 5, 6, 9, 0],
        ],
        dtype=torch.long,
    )[:batch_size, :src_len]
    tgt = torch.tensor(
        [
            [2, 4, 3, 1, 0, 0],
            [2, 6, 7, 8, 1, 0],
        ],
        dtype=torch.long,
    )[:batch_size, :tgt_len]
    return src, tgt


def _make_model(attention_factory):
    return Seq2Seq(
        src_vocab_size=32,
        tgt_vocab_size=40,
        d_model=16,
        ff_dim=32,
        num_heads=4,
        num_layers=1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        tgt_sos_idx=2,
        dropout=0.0,
        max_len=32,
        attention_factory=attention_factory,
    )


def test_attention_factories_return_protocol_compatible_modules():
    for factory in (make_attention_factory("torch"), make_attention_factory("simple_sdp")):
        attn = factory(16, 4, 0.0)
        assert isinstance(attn, nn.Module)
        assert isinstance(attn, AttentionProtocol)


def test_seq2seq_forward_works_with_torch_attention_factory():
    model = _make_model(make_attention_factory("torch"))
    src, tgt = _dummy_batch()
    logits = model(src, tgt)
    assert logits.shape == (src.size(0), tgt.size(1) - 1, 40)


def test_seq2seq_forward_works_with_simple_sdp_attention_factory():
    model = _make_model(make_attention_factory("simple_sdp"))
    src, tgt = _dummy_batch()
    logits = model(src, tgt)
    assert logits.shape == (src.size(0), tgt.size(1) - 1, 40)


def test_simple_sdp_attention_applies_masks_and_returns_expected_shapes():
    attn = make_attention_factory("simple_sdp")(16, 4, 0.0)

    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    attn_mask = torch.zeros(4, 5, dtype=torch.bool)
    attn_mask[:, -1] = True
    key_padding_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, False, True, True],
        ],
        dtype=torch.bool,
    )

    out, weights = attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    assert out.shape == (2, 4, 16)
    assert weights is not None
    assert weights.shape == (2, 4, 5)


def test_invalid_attention_factory_raises_clear_error():
    def bad_factory(d_model: int, num_heads: int, dropout: float):
        del d_model, num_heads, dropout
        return object()

    model = _make_model(bad_factory)
    src, tgt = _dummy_batch()
    try:
        model(src, tgt)
        assert False, "expected runtime error"
    except TypeError:
        assert True
