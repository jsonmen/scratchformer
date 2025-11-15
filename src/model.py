import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from decoder import DecoderLayer
from encoder import EncoderLayer
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Model that reimplements the Transformer translator from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper."""

    def __init__(
        self,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        max_seq_len: int,
        vocab_size: int,
        dropout: float,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3,
        pe_device: torch.device | str = "cpu",
    ) -> None:
        """
        Initialize the Transformer model.

        Parameters
        ----------
        n_encoder_layers : int
            Number of encoder layers to stack (typically 3-6).
        n_decoder_layers : int
            Number of decoder layers to stack (typically 3-6; often matches encoder).
        d_model : int
            Embedding/model dimension (e.g., 128-512). Must be divisible by `n_heads` for even head sizes.
        d_ff : int
            Feed-forward network hidden dimension (typically 4 * d_model for expansion).
        n_heads : int
            Number of attention heads (e.g., 4-8; enables parallel representations).
        max_seq_len : int
            Maximum sequence length supported (for precomputing positional encodings).
        vocab_size : int
            Size of the vocabulary, including special tokens like <pad>/<bos>/<eos>.
        dropout : float
            Dropout probability (in [0, 1]; applied to sublayers and embeddings for regularization).
        pad_idx : int, default=0
            Token ID for padding (<pad>); used to ignore positions in loss/masking.
        bos_idx : int, default=2
            Token ID for beginning-of-sequence (<bos>); starts decoder inputs.
        eos_idx : int, default=3
            Token ID for end-of-sequence (<eos>); stops generation.
        pe_device : torch.device | str, default="cpu"
            Device for positional encodings tensor (e.g., "cuda" for GPU).

        Returns
        -------
        None
        """
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )  # Shared for src/tgt
        self.out_linear = nn.Linear(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(
            max_seq_len, d_model, dropout, pe_device
        )  # (max_seq, d_model)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(d_model, d_ff, n_heads, dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d_model, d_ff, n_heads, dropout)
                for _ in range(n_decoder_layers)
            ]
        )

    def create_src_mask(self, src: Tensor) -> Tensor:
        """
        Creates a padding mask for source sequences to ignore <pad> tokens during attention.

        Parameters
        ----------
        src : Tensor
            Source sequence token IDs with shape (batch_size, src_seq_len).

        Returns
        -------
        src_mask : Tensor
            Binary mask (1 for attend, 0 for mask) with shape (batch_size, 1, 1, src_seq_len).
            Broadcasts to attention scores for key masking.
        """
        return (src != self.pad_idx).float().unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt: Tensor) -> Tensor:
        """
        Creates a causal (lower-triangular) mask for target sequences to enforce autoregression.

        Parameters
        ----------
        tgt : Tensor
            Target sequence token IDs with shape (batch_size, tgt_seq_len).

        Returns
        -------
        tgt_mask : Tensor
            Causal mask (1 for attend, 0 for future) with shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            Combines with padding mask for full decoder self-attention.
        """
        batch_size, seq_len = tgt.shape
        tgt_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
            .float()
            .expand(batch_size, 1, seq_len, seq_len)
        )
        return tgt_mask

    def compute_encoder(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Computes the full encoder stack, producing contextual representations.

        Parameters
        ----------
        src : Tensor
            Source sequence token IDs with shape (batch_size, src_seq_len).
        src_mask : Tensor
            Padding mask for source with shape (batch_size, 1, 1, src_seq_len).

        Returns
        -------
        encoder_out : Tensor
            Contextual embeddings from final encoder layer, shape (batch_size, src_seq_len, d_model).
        """
        src_emb = self.embedding(src) * math.sqrt(self.d_model)  # Scale
        encoder_out = self.positional_encoding.forward(src_emb)
        for layer in self.encoder:
            encoder_out = layer(encoder_out, src_mask)
        return encoder_out

    def compute_decoder(
        self, tgt: Tensor, encoder_out: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        """
        Computes the full decoder stack, attending to encoder outputs autoregressively.

        Parameters
        ----------
        tgt : Tensor
            Target sequence token IDs with shape (batch_size, tgt_seq_len).
        encoder_out : Tensor
            Encoder contextual representations, shape (batch_size, src_seq_len, d_model).
        src_mask : Tensor
            Source padding mask, shape (batch_size, 1, 1, src_seq_len).
        tgt_mask : Tensor
            Causal + padding mask for target, shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        decoder_out : Tensor
            Decoder representations, shape (batch_size, tgt_seq_len, d_model).
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        decoder_out = self.positional_encoding.forward(tgt_emb)
        for layer in self.decoder:
            decoder_out = layer(decoder_out, encoder_out, tgt_mask, src_mask)
        return decoder_out

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass: Encodes source and decodes target with masking.

        Parameters
        ----------
        src : Tensor
            Source sequence token IDs, shape (batch_size, src_seq_len).
        tgt : Tensor
            Target sequence token IDs (shifted right for teacher forcing), shape (batch_size, tgt_seq_len).
        src_mask : Tensor, optional
            Source padding mask, shape (batch_size, 1, 1, src_seq_len). Auto-created if None.
        tgt_mask : Tensor, optional
            Target causal + padding mask, shape (batch_size, 1, tgt_seq_len, tgt_seq_len). Auto-created if None.

        Returns
        -------
        logits : Tensor
            Vocabulary logits for each target position, shape (batch_size, tgt_seq_len, vocab_size).
            Use for cross-entropy loss (shifted targets).
        """
        # Mask Creation
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)

        # Encoder
        encoder_out = self.compute_encoder(src, src_mask)

        # Decoder
        decoder_out = self.compute_decoder(tgt, encoder_out, src_mask, tgt_mask)

        # Output logits
        logits = self.out_linear(decoder_out)  # (batch, tgt_seq, vocab)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        max_new_tokens: int,
        device: torch.device | str,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> Tensor:
        """
        Autoregressively generates a target sequence from source, starting with <bos>.

        Parameters
        ----------
        src : Tensor
            Source sequence token IDs, shape (batch_size, src_seq_len).
        max_new_tokens : int
            Maximum number of new tokens to generate (beyond <bos>).
        device : torch.device | str
            Computation device (e.g., "cuda").
        temperature : float, default=1.0
            Softens logits for sampling (>1: more random; <1: sharper/greedy).
        do_sample : bool, default=False
            If True, sample from softmax probs; else use argmax (greedy).
        top_k : int | None, default=None
            If set, sample only from top-k most likely tokens (reduces diversity).

        Returns
        -------
        tgt : Tensor
            Generated target sequence, shape (batch_size, 1 + max_new_tokens), including <bos>.
        """
        batch_size, _ = src.shape
        tgt = torch.full((batch_size, 1), self.bos_idx, device=device)

        # Mask Creation
        src_mask = self.create_src_mask(src)
        # Encoder precomputing
        encoder_out = self.compute_encoder(src, src_mask)

        for _ in range(max_new_tokens):
            tgt_mask = self.create_tgt_mask(tgt)
            decoder_out = self.compute_decoder(tgt, encoder_out, src_mask, tgt_mask)
            logits = self.out_linear(decoder_out)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            tgt = torch.cat((tgt, next_token), dim=1)

        return tgt


if __name__ == "__main__":
    # Test
    device = "cuda"
    n_encoder_layers = n_decoder_layers = 6
    d_model = 512
    d_ff = 2048
    n_heads = 8
    max_seq_len = 1024
    vocab_size = 1024
    model = Transformer(
        n_encoder_layers,
        n_decoder_layers,
        d_model,
        d_ff,
        n_heads,
        max_seq_len,
        vocab_size,
        0.0,
        pe_device=device,
    )
    model = model.to(device)
    batch_size = 8
    fake_src_seq = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.long, device=device
    )
    fake_src_seq[:, :5] = torch.arange(
        0, 40, 1, dtype=torch.long, device=device
    ).reshape(batch_size, 5)
    fake_tgt_seq = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.long, device=device
    )
    fake_tgt_seq[:, :5] = torch.arange(
        40, 80, 1, dtype=torch.long, device=device
    ).reshape(batch_size, 5)
    p_out = model.forward(fake_src_seq, fake_tgt_seq)
    model.eval()
    out = model.generate(fake_src_seq, max_seq_len, device, 0.8, True, 5)
    print(out)
