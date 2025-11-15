from typing import Optional

import torch
from torch import Tensor, nn

from feedforward import FeedForward
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Single encoder layer from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

    Stacks multi-head self-attention and feed-forward sub-layers with residuals and layer norm.
    Processes source sequences in parallel.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float) -> None:
        """
        Initialize the encoder layer.

        Parameters
        ----------
        d_model : int
            Model/embedding dimension (e.g., 128-512).
        d_ff : int
            Feed-forward hidden dimension (typically 4 * d_model).
        n_heads : int
            Number of attention heads (e.g., 4-8).
        dropout : float
            Dropout probability (in [0, 1]; applied to sublayers).

        Returns
        -------
        None
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, dropout)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the encoder layer.

        Parameters
        ----------
        x : Tensor
            Input embeddings, shape (batch_size, seq_len, d_model).
        src_mask : Tensor, optional
            Padding mask for source, shape (batch_size, 1, 1, seq_len).
            Ignores pad tokens in attention (auto-created if None).

        Returns
        -------
        out : Tensor
            Layer output with residuals and norms, same shape as x.
        """
        # Self-Attention
        mha_out = self.mha(x, x, x, src_mask)
        out = self.mha_norm(x + mha_out)
        # FFN
        out = self.ffn_norm(out + self.ffn(out))
        return out


if __name__ == "__main__":
    # Test
    d_model = 16
    d_ff = 64
    n_heads = 4
    batch = 4
    seq_len = 64
    random_inp = torch.randn((batch, seq_len, d_model))
    encoder_layer = EncoderLayer(d_model, d_ff, n_heads, 0.0)
    print(encoder_layer(random_inp).shape)
