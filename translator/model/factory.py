from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class AttentionProtocol(Protocol):
    """Shared attention contract matching PyTorch MHA, so custom attention can be swapped in safely."""

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...


AttentionFactory = Callable[[int, int, float], AttentionProtocol]
ATTENTION_CHOICES = ("torch", "simple_sdp")


def make_attention_factory(attention: str) -> AttentionFactory:
    from .attention import SimpleMultiheadSDPAttention

    def make_torch_attention_factory() -> AttentionFactory:
        return lambda d_model, num_heads, dropout: nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def make_simple_sdp_attention_factory() -> AttentionFactory:
        return lambda d_model, num_heads, dropout: SimpleMultiheadSDPAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    if attention == "torch":
        return make_torch_attention_factory()
    if attention == "simple_sdp":
        return make_simple_sdp_attention_factory()
    raise ValueError(f"Unknown attention={attention!r}. Allowed values: {ATTENTION_CHOICES}.")


def build_attention(
    attention_factory: AttentionFactory, d_model: int, num_heads: int, dropout: float
) -> AttentionProtocol:
    return attention_factory(d_model, num_heads, dropout)
