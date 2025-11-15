import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    """Positional encoding module from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

    Adds fixed sinusoidal encodings to input embeddings to inject sequence position information,
    enabling the model to handle variable-length sequences without recurrence.
    """

    def __init__(
        self, max_seq_len: int, d_model: int, dropout: float, device: torch.device | str
    ) -> None:
        """
        Initialize the positional encoding module.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length to precompute encodings for (e.g., 512-2048).
        d_model : int
            Model/embedding dimension (e.g., 128-512; encodings match this size).
        dropout : float
            Dropout probability (in [0, 1]; applied after adding encodings).
        device : torch.device | str
            Device for the encodings tensor (e.g., "cuda" for GPU).

        Returns
        -------
        None
        """
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = PositionalEncoding.get_encoding(max_seq_len, d_model, device)

    @staticmethod
    def get_encoding(max_seq_len: int, d_model: int, device: torch.device | str) -> Tensor:
        """
        Computes fixed sinusoidal positional encodings.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length.
        d_model : int
            Embedding dimension.
        device : torch.device | str
            Device to place the tensor on.

        Returns
        -------
        pe : Tensor
            Positional encodings matrix, shape (max_seq_len, d_model).
            Rows: positions; even columns: sin, odd: cos with wavelength 10000.
        """
        pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len)[:, None]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10_000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encodings to input and applies dropout.

        Parameters
        ----------
        x : Tensor
            Input embeddings, shape (batch_size, seq_len, d_model).

        Returns
        -------
        out : Tensor
            Positionally encoded inputs with dropout, same shape as x.
        """
        out = x + self.pe[: x.size(1)]
        return self.dropout(out)

if __name__ == "__main__":
    # Test
    max_seq_len = 10
    d_model = 64
    pe = PositionalEncoding(max_seq_len, d_model, 0.0, "cpu").pe
    print(pe.shape)
