from torch import Tensor, nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network sub-layer from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

    Applies two linear layers with ReLU activation and dropout for non-linear transformation.
    Shared across positions for parallelism.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initialize the feed-forward sub-layer.

        Parameters
        ----------
        d_model : int
            Input/output dimension (e.g., 128-512).
        d_ff : int
            Hidden dimension for expansion (typically 4 * d_model).
        dropout : float
            Dropout probability (in [0, 1]; applied after FFN).

        Returns
        -------
        None
        """
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the FFN.

        Parameters
        ----------
        x : Tensor
            Input, shape (batch_size, seq_len, d_model).

        Returns
        -------
        out : Tensor
            Transformed output with dropout, same shape as x.
        """
        return self.dropout(self.ffn(x))
