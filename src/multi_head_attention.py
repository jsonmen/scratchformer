import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self/cross-attention mechanism from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

    Computes scaled dot-product attention in parallel across multiple heads,
    allowing the model to jointly attend to information from different subspaces.
    Supports masking for padding and causality.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float) -> None:
        """
        Initialize the multi-head attention module.

        Parameters
        ----------
        n_heads : int
            Number of attention heads (e.g., 4-8; enables parallel representations).
        d_model : int
            Model/embedding dimension (e.g., 128-512). Must be divisible by `n_heads`
            for even per-head dimensions (d_k = d_v = d_model / n_heads).
        dropout : float
            Dropout probability (in [0, 1]; applied to attention weights and output).

        Returns
        -------
        None
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)  # Query projection
        self.W_K = nn.Linear(d_model, d_model)  # Key projection
        self.W_V = nn.Linear(d_model, d_model)  # Value projection
        self.W_O = nn.Linear(d_model, d_model)  # Output projection
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Splits the last dimension into (n_heads, head_dim) and transposes for multi-head.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (batch_size, seq_len, d_model).

        Returns
        -------
        split_x : Tensor
            Reshaped and transposed tensor, shape (batch_size, n_heads, seq_len, d_k).
        """
        batch, seq_len, _ = x.shape  # (batch, seq_len, d_model)
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (batch, n_heads, seq_len, d_k)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combines multi-head outputs by transposing and reshaping.

        Parameters
        ----------
        x : Tensor
            Multi-head tensor with shape (batch_size, n_heads, seq_len, d_v).

        Returns
        -------
        combined_x : Tensor
            Flattened tensor, shape (batch_size, seq_len, d_model).
        """
        batch, _, seq_len, _ = x.shape
        return (
            x.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.d_k)
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass: Projects Q/K/V, computes scaled dot-product attention, and combines heads.

        Parameters
        ----------
        q : Tensor
            Query tensor, shape (batch_size, q_seq_len, d_model).
        k : Tensor
            Key tensor, shape (batch_size, k_seq_len, d_model).
        v : Tensor
            Value tensor, shape (batch_size, v_seq_len, d_model).
        mask : Tensor, optional
            Attention mask (0=mask, 1=attend), broadcastable to (batch, 1, q_seq_len, k_seq_len).
            Supports padding (src_mask) or causal (tgt_mask).

        Returns
        -------
        attn_out : Tensor
            Attention output, shape (batch_size, q_seq_len, d_model).
        """
        Q = self._split_heads(self.W_Q(q))
        K = self._split_heads(self.W_K(k))
        V = self._split_heads(self.W_V(v))
        scores = (Q @ K.transpose(-2, -1)) * (
            self.d_k**-0.5
        )  # (batch, n_heads, q_seq_len, k_seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = attn_weights @ V  # (batch, n_heads, q_seq_len, d_v)
        concat_out = self._combine_heads(attn_out)
        return self.dropout(self.W_O(concat_out))


if __name__ == "__main__":
    # Test
    torch.manual_seed(42)
    attn = MultiHeadAttention(8, 512, 0.2)
    attn2 = nn.MultiheadAttention(512, 8, batch_first=True)
    x = torch.rand(2, 10, 512)
    mask = torch.zeros(2, 1, 1, 10)
    out = attn(x, x, x)
    out2, _ = attn2(x, x, x)
    print(out.shape, out2.shape)
    print("Custom norm:", torch.norm(out).item())
    print("PyTorch norm:", torch.norm(out2).item())
