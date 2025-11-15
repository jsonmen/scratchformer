import csv
import os
from subprocess import run
from zipfile import ZipFile

import pandas as pd
import torch
import sentencepiece as sp
from torch.utils.data import Dataset

from config import (
    DATASET_DOWNLOAD_LINK,
    DATASET_TRAIN_FILE,
    MAX_SEQ_LEN,
    SUBSET_SIZE,
    TOKENIZER_OPTIONS,
    TOKENIZER_PATH,
)
from tokenizer import load_tokenizer


class WMTDataset(Dataset):
    """WMT 2014 English-German translation dataset loader.

    Loads sentence pairs from CSV, tokenizes with SentencePiece, and supports subsets for efficient training.
    Handles BOS/EOS addition and truncation to max_seq_len.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: sp.SentencePieceProcessor,
        max_seq_len: int,
        subset_size: int | None = None,
    ) -> None:
        """
        Initialize the WMT dataset.

        Parameters
        ----------
        dataset_path : str
            Path to CSV file (columns: 'en', 'de').
        tokenizer : sp.SentencePieceProcessor
            Trained tokenizer for encoding/decoding.
        max_seq_len : int
            Maximum sequence length; truncates longer pairs.
        subset_size : int | None, default=None
            If set, randomly sample this many pairs (with replacement) for faster training.

        Returns
        -------
        None
        """
        self.dataset = pd.read_csv(
            dataset_path,
            engine="python",
            quoting=csv.QUOTE_ALL,
            on_bad_lines="warn",
            encoding="utf-8-sig",
            sep=",",
        )
        if subset_size is not None:
            self.dataset = self.dataset.sample(n=subset_size, replace=True).reset_index(
                drop=True
            )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def tokenize_text(self, txt: str) -> torch.Tensor:
        """
        Tokenizes text with BOS/EOS and returns tensor.

        Parameters
        ----------
        txt : str
            Raw text to encode.

        Returns
        -------
        tok_seq : Tensor
            Token ID sequence, dtype=torch.long.
        """
        tok_seq = self.tokenizer.Encode(str(txt), int, add_bos=True, add_eos=True)
        return torch.tensor(tok_seq, dtype=torch.long)

    def __len__(self) -> int:
        """
        Returns the number of sentence pairs.

        Returns
        -------
        length : int
            Dataset size (after subset if applied).
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Fetches a single pair and tokenizes.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        item : dict
            {'src': Tensor (en tokens, truncated), 'tgt': Tensor (de tokens, truncated)}.
        """
        en = self.dataset.loc[index, "en"]
        de = self.dataset.loc[index, "de"]
        return {
            "src": self.tokenize_text(en)[: self.max_seq_len],  # Source tensor
            "tgt": self.tokenize_text(de)[: self.max_seq_len],  # Target tensor
        }

    @staticmethod
    def download_dataset(save_dir: str) -> None:
        """
        Downloads and extracts the WMT dataset ZIP.

        Parameters
        ----------
        save_dir : str
            Directory to save extracted CSV files.

        Returns
        -------
        None
        """
        # Dataset: https://www.kaggle.com/api/v1/datasets/download/mohamedlotfy50/wmt-2014-english-german
        zip_file_path = os.path.join(save_dir, "dataset.zip")
        run(["curl", "-L", "-o", zip_file_path, DATASET_DOWNLOAD_LINK])
        with ZipFile(zip_file_path, "r") as f:
            f.extractall(save_dir)
        run(["rm", zip_file_path])


if __name__ == "__main__":
    if not os.path.isfile(DATASET_TRAIN_FILE):
        WMTDataset.download_dataset("data")
    tokenizer = load_tokenizer(TOKENIZER_PATH, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE)
    dataset = WMTDataset(DATASET_TRAIN_FILE, tokenizer, MAX_SEQ_LEN, SUBSET_SIZE)
    print(f"Dataset Length: {len(dataset)}")
    item = dataset[0]
    print(f"English:\n{tokenizer.decode(item['src'].tolist())}\n---")
    print(f"German:\n{tokenizer.decode(item['tgt'].tolist())}")
