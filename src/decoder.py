from typing import Optional

import torch
from torch import Tensor, nn

from feedforward import FeedForward
from multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """Single decoder layer from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

    Stacks masked self-attention, cross-attention to encoder, and feed-forward sub-layers
    with residuals and layer norm. Ensures autoregression via causal masking.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float) -> None:
        """
        Initialize the decoder layer.

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
        super(DecoderLayer, self).__init__()
        self.masked_mha = MultiHeadAttention(n_heads, d_model, dropout)
        self.masked_mha_norm = nn.LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(n_heads, d_model, dropout)
        self.cross_mha_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the decoder layer.

        Parameters
        ----------
        x : Tensor
            Target input embeddings, shape (batch_size, tgt_seq_len, d_model).
        encoder_out : Tensor
            Encoder representations, shape (batch_size, src_seq_len, d_model).
        tgt_mask : Tensor, optional
            Causal + padding mask for target self-attention,
            shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
        src_mask : Tensor, optional
            Padding mask for encoder keys in cross-attention,
            shape (batch_size, 1, 1, src_seq_len).

        Returns
        -------
        out : Tensor
            Layer output with residuals and norms, same shape as x.
        """
        # 1. Masked self-attention
        masked_mha_out = self.masked_mha(x, x, x, mask=tgt_mask)
        out = self.masked_mha_norm(x + masked_mha_out)
        # 2. Cross-attention
        cross_mha_out = self.cross_mha(out, encoder_out, encoder_out, src_mask)
        out = self.cross_mha_norm(out + cross_mha_out)
        # 3. FFN
        ffn_out = self.ffn(out)
        out = self.ffn_norm(out + ffn_out)
        return out


if __name__ == "__main__":
    # Test
    d_model = 16
    d_ff = 64
    n_heads = 4
    batch = 4
    seq_len = 64
    random_inp = torch.randn((batch, seq_len, d_model))
    random_enc_out = torch.randn((batch, seq_len, d_model))
    decoder_layer = DecoderLayer(d_model, d_ff, n_heads, 0.0)
    print(decoder_layer(random_inp, random_enc_out).shape)
