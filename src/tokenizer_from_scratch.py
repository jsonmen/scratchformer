"""
I don't use it in my code, because it's very slow!
"""

import regex as re
import collections
import pandas as pd
import torch


class BPETokenizer:
    """Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization.

    Trains merges on text corpus and handles encoding/decoding with special tokens.
    Supports DataFrame inputs for large datasets like WMT.
    """

    def __init__(self, vocab_size: int) -> None:
        """
        Initialize the BPE tokenizer.

        Parameters
        ----------
        vocab_size : int
            Target vocabulary size (includes specials + bytes + merges; e.g., 8k-32k).

        Returns
        -------
        None
        """
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.special_tokens = {
            0: b"<pad>",
            1: b"<unk>",
            2: b"<bos>",
            3: b"<eos>",
        }
        self.special_strings = {
            v.decode("utf-8"): k for k, v in self.special_tokens.items()
        }
        self.byte_start = len(self.special_tokens)  # 4
        self.merge_start = self.byte_start + 256  # 260
        self.n_merges = vocab_size - self.merge_start
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?<[^>]+> ?| ?[a-zA-Z]+| ?[0-9]+| ?[^\s a-zA-Z0-9]+?|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )
        self.pad_id = 0

    @property
    def is_pretrained(self) -> bool:
        """
        Checks if vocab/merges are loaded (pretrained).

        Returns
        -------
        pretrained : bool
            True if ready for encoding/decoding.
        """
        if len(self.vocab) > 0 and len(self.merges) > 0:
            return True
        else:
            return False

    def _get_stats(self, byte_seqs: list[list[int]]) -> collections.Counter:
        """
        Computes pair frequencies for BPE merging.

        Parameters
        ----------
        byte_seqs : list[list[int]]
            List of byte/token sequences.

        Returns
        -------
        stats : collections.Counter
            Frequency counter of adjacent pairs.
        """
        stats = collections.Counter()
        for byte_seq in byte_seqs:
            for pair in zip(byte_seq, byte_seq[1:]):
                stats[pair] += 1
        return stats

    def _get_pairs(self, byte_seq: list[int]) -> set[tuple[int, int]]:
        """
        Extracts unique adjacent pairs from a sequence.

        Parameters
        ----------
        byte_seq : list[int]
            Byte/token sequence.

        Returns
        -------
        pairs : set[tuple[int, int]]
            Unique pairs.
        """
        return set(zip(byte_seq, byte_seq[1:]))

    def merge_byte_seq(
        self, byte_seq: list[int], pair: tuple[int, int], new_idx: int
    ) -> list[int]:
        """
        Merges a single pair in a byte sequence.

        Parameters
        ----------
        byte_seq : list[int]
            Byte/token sequence.
        pair : tuple[int, int]
            Pair to merge (e.g., (b'a', b'b')).
        new_idx : int
            New token ID for the merged pair.

        Returns
        -------
        new_seq : list[int]
            Merged sequence.
        """
        new_seq = []
        i = 0
        while i < len(byte_seq):
            if (
                i < len(byte_seq) - 1
                and byte_seq[i] == pair[0]
                and byte_seq[i + 1] == pair[1]
            ):
                new_seq.append(new_idx)
                i += 2
            else:
                new_seq.append(byte_seq[i])
                i += 1
        return new_seq

    def merge_byte_seqs(
        self, byte_seqs: list[list[int]], pair: tuple[int, int], new_idx: int
    ) -> list[list[int]]:
        """
        Merges a pair across multiple sequences.

        Parameters
        ----------
        byte_seqs : list[list[int]]
            List of byte/token sequences.
        pair : tuple[int, int]
            Pair to merge.
        new_idx : int
            New token ID.

        Returns
        -------
        merged_seqs : list[list[int]]
            List of updated sequences.
        """
        return [self.merge_byte_seq(byte_seq, pair, new_idx) for byte_seq in byte_seqs]

    def train(self, X) -> "BPETokenizer":
        """
        Trains BPE merges on input text corpus.

        Parameters
        ----------
        X : str | list[str] | pd.DataFrame
            Training texts (str/list for small; DataFrame for large, uses all string cols).

        Returns
        -------
        self : BPETokenizer
            Trained tokenizer instance (in-place).
        """
        if isinstance(X, str):
            texts = [X]
        elif isinstance(X, list) and all(isinstance(s, str) for s in X):
            texts = X
        elif isinstance(X, pd.DataFrame):
            texts = []
            for col in X.columns:
                for text in X[col]:
                    if pd.notna(text) and isinstance(text, str):
                        texts.append(text)
        else:
            raise ValueError("Input must be str, list[str], or pd.DataFrame")

        # Add BOS and EOS if missing for each text
        processed_texts = []
        for text in texts:
            if not text.startswith("<bos>"):
                text = "<bos>" + text
            if not text.endswith("<eos>"):
                text += "<eos>"
            processed_texts.append(text)

        words = []
        for text in processed_texts:
            words.extend(re.findall(self.pat, text))

        byte_seqs = []
        for word in words:
            stripped = word.strip()
            if stripped in self.special_strings:
                byte_seqs.append([self.special_strings[stripped]])
            else:
                byte_seqs.append(
                    [
                        self.byte_start + b
                        for b in word.encode("utf-8", errors="replace")
                    ]
                )

        for i in range(self.n_merges):
            stats = self._get_stats(byte_seqs)
            if not stats:
                break
            top_pair = stats.most_common(1)[0][0]
            new_idx = self.merge_start + i
            byte_seqs = self.merge_byte_seqs(byte_seqs, top_pair, new_idx)
            self.merges[top_pair] = new_idx

        self.vocab = self.special_tokens.copy()
        for i in range(256):
            self.vocab[self.byte_start + i] = bytes([i])
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def decode(self, byte_seq: list[int]) -> str:
        """
        Decodes token IDs to text, skipping pads.

        Parameters
        ----------
        byte_seq : list[int]
            List of token IDs.

        Returns
        -------
        text : str
            Decoded string (UTF-8, with <unk> fallback).
        """
        if not self.vocab:
            raise ValueError("Vocab is empty!")
        tokens = b"".join(
            self.vocab.get(idx, b"<unk>") for idx in byte_seq if idx != self.pad_id
        )
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(
        self,
        text: str,
        max_seq_len: int,
        padding: bool,
        plug_special_tokens: bool = True,
        return_type: str | None = "pt",
    ) -> list[int] | torch.Tensor:
        """
        Encodes text to token IDs, with optional padding/BOS/EOS.

        Parameters
        ----------
        text : str
            Input text to encode.
        max_seq_len : int
            Maximum length; truncates if longer.
        padding : bool
            If True, pad to max_seq_len with <pad>.
        plug_special_tokens : bool, default=True
            If True, add <bos> at start and <eos> at end (adjusts len).
        return_type : str | None, default="pt"
            "pt" for torch.Tensor; None for list[int].

        Returns
        -------
        tokens : list[int] | Tensor
            Encoded sequence (long dtype if tensor).
        """
        if not self.vocab:
            raise ValueError("Vocab is empty!")

        # Optionally add BOS/EOS to raw text if not present
        words = re.findall(self.pat, text)
        tokens = []
        for word in words:
            stripped = word.strip()
            if stripped in self.special_strings:
                tokens.append(self.special_strings[stripped])
            else:
                byte_seq = [
                    self.byte_start + b for b in word.encode("utf-8", errors="replace")
                ]
                while len(byte_seq) >= 2:
                    pairs = self._get_pairs(byte_seq)
                    candidates = [p for p in pairs if p in self.merges]
                    if not candidates:
                        break
                    pair = min(candidates, key=lambda p: self.merges[p])
                    idx = self.merges[pair]
                    byte_seq = self.merge_byte_seq(byte_seq, pair, idx)
                tokens.extend(byte_seq)

        if plug_special_tokens:
            max_seq_len -= 2

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        if padding:
            tokens += [self.pad_id] * (max_seq_len - len(tokens))
        if plug_special_tokens:
            # Add <bos> token at the start and <eos> at the end
            tokens.insert(0, 2)
            tokens.append(3)
        if return_type == "pt":
            return torch.tensor(tokens, dtype=torch.long)
        return tokens


def main():
    text = """Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training.
It centralizes the model definition so that this definition is agreed upon across the ecosystem. transformers is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, …), inference engines (vLLM, SGLang, TGI, …), and adjacent modeling libraries (llama.cpp, mlx, …) which leverage the model definition from transformers.
We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.
There are over 1M+ Transformers model checkpoints on the Hugging Face Hub you can use.
Explore the Hub today to find a model and use Transformers to help you get started right away.
Explore the Models Timeline to discover the latest text, vision, audio and multimodal model architectures in Transformers."""
    max_seq_len = 2048
    pad = False
    tokenizer = BPETokenizer(1500)
    tokenizer.train(text)
    encoded_padded = tokenizer.encode(text, max_seq_len, pad)
    encoded = tokenizer.encode(text, max_seq_len, False)  # For compression ratio
    print("Length of original text:", len(text))
    print("Length of original tokenized text:", len(encoded_padded))
    print(f"Compress ratio: {len(text) / len(encoded):.2f}X")
    print(
        "Does decoded text equal to original:",
        tokenizer.decode(encoded_padded)[5:-5] == text,
    )
    print(tokenizer.decode(encoded_padded)[5:-5])
    # print(tokenizer.vocab)


if __name__ == "__main__":
    main()
