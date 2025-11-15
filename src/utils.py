import math
import os

import sentencepiece as sp
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    Collate function for translation batches: Sorts by source length and pads dynamically.

    Groups padding by sorting sequences (bucket-like), reducing waste vs fixed pads.
    Returns masks/lengths for efficient attention.

    Parameters
    ----------
    batch : list[dict[str, Tensor]]
        List of {'src': Tensor, 'tgt': Tensor} items (unpadded).

    Returns
    -------
    collated : dict[str, Tensor]
        {'src': padded_src (batch, src_max), 'tgt': padded_tgt (batch, tgt_max),
         'src_lengths': Tensor (batch,), 'tgt_lengths': Tensor (batch,)}.
    """
    # Sort batch by descending src length
    batch = sorted(batch, key=lambda x: len(x["src"]), reverse=True)
    srcs = [item["src"] for item in batch]
    tgts = [item["tgt"] for item in batch]
    # Compute actual max lengths (src[0] is max due to sorting; tgt needs explicit max)
    src_max_len = len(srcs[0])  # Safe: longest src first
    tgt_max_len = max(
        len(tgt) for tgt in tgts
    )  # Key fix: tgt lengths vary independently
    # Pad src
    padded_srcs = torch.full((len(srcs), src_max_len), 0, dtype=srcs[0].dtype)
    src_lengths = torch.tensor([len(src) for src in srcs])
    for i, src in enumerate(srcs):
        padded_srcs[i, : len(src)] = src
    # Pad tgt
    padded_tgts = torch.full((len(tgts), tgt_max_len), 0, dtype=tgts[0].dtype)
    tgt_lengths = torch.tensor([len(tgt) for tgt in tgts])
    for i, tgt in enumerate(tgts):
        padded_tgts[i, : len(tgt)] = tgt
    return {
        "src": padded_srcs,
        "tgt": padded_tgts,
        "src_lengths": src_lengths,  # For encoder packing/masking
        "tgt_lengths": tgt_lengths,  # For decoder
    }


def get_fname_info(path: str, idx: int) -> int:
    """
    Extracts an indexed integer from filename (e.g., epoch/step in "model_3_500.pt").

    Parameters
    ----------
    path : str
        File path (e.g., "checkpoint_epoch_3_step_500.pt").
    idx : int
        Index in split (e.g., -1 for step, -2 for epoch).

    Returns
    -------
    value : int
        Extracted integer (assumes "_num" format).
    """
    return int(path[:-3].split("_")[idx])


def get_last_checkpoint_fname(checkpoint_paths: list[str]) -> str:
    """
    Finds the latest checkpoint filename by comparing epoch/step.

    Parameters
    ----------
    checkpoint_paths : list[str]
        List of checkpoint file paths (e.g., ["model_1_100.pt", "model_3_500.pt"]).

    Returns
    -------
    latest_fname : str
        Path to the most recent checkpoint (max epoch, then step).
    """
    last_checkpoint_fname = max(
        checkpoint_paths,
        key=lambda x: get_fname_info(x, -1) + 1 * get_fname_info(x, -2),
    )
    return last_checkpoint_fname


def load_last_checkpoint(
    load: bool, model: nn.Module, checkpoints_dir: str
) -> tuple[nn.Module, int, int]:
    """
    Loads the latest checkpoint if available, resuming model/epoch/step.

    Parameters
    ----------
    load : bool
        If True, attempt to load; else return zeros.
    model : nn.Module
        Model to load state_dict into.
    checkpoints_dir : str
        Directory with checkpoint files (e.g., "model/checkpoints").

    Returns
    -------
    model : nn.Module
        Loaded model (or original if no load).
    last_epoch : int
        Resumed epoch (0 if none).
    last_step : int
        Resumed step (0 if none).
    """
    last_epoch, last_step = 0, 0
    if load:
        checkpoint_paths = os.listdir(checkpoints_dir)
        if len(checkpoint_paths) > 0:
            last_checkpoint_fname = get_last_checkpoint_fname(checkpoint_paths)
            checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint_fname)
            checkpoint_info = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint_info["model"])
            last_epoch, last_step = checkpoint_info["epoch"], checkpoint_info["step"]
            print("Checkpoint is loaded correctly!")
    return model, last_epoch, last_step


def val_score(
    val_dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    tokenizer: sp.SentencePieceProcessor,
) -> tuple[float, float]:
    """
    Evaluates model on validation set: Computes average loss and perplexity (teacher-forced).

    No generation/BLEU here (for speed); subsets full val (~3k WMT pairs) if needed.
    Uses cross-entropy on shifted targets, ignoring pads.

    Parameters
    ----------
    val_dataloader : DataLoader
        Validation batches (dicts with 'src'/'tgt' tensors).
    model : Transformer
        Model in eval mode during call.
    device : torch.device
        Computation device (e.g., torch.device("cuda")).
    tokenizer : sp.SentencePieceProcessor
        Tokenizer for pad_id (used in ignore_index).

    Returns
    -------
    avg_loss : float
        Token-averaged cross-entropy loss (lower better).
    avg_ppl : float
        Exponentiated average loss (perplexity; lower better, e.g., <20 for good models).
    """
    model.eval()  # Dropout off
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():  # No grads
        for item in val_dataloader:
            src = item["src"].to(device)
            tgt = item["tgt"].to(device)
            # Forward for loss/PPL (teacher forcing)
            logits = model.forward(src, tgt[:, :-1])
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = tgt[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                logits_flat, targets_flat, ignore_index=tokenizer.pad_id()
            )
            non_pad = (targets_flat != tokenizer.pad_id()).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
        # Averages
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        avg_ppl = math.exp(avg_loss)  # Token-avg PPL
    model.train()  # Back to train mode
    return avg_loss, avg_ppl


def test_translation(
    text: str,
    model: nn.Module,
    tokenizer: sp.SentencePieceProcessor,
    max_seq_len: int,
    device: str | torch.device,
) -> str:
    """
    Generates a demo translation from a fixed English prompt to test model coherence.

    Uses autoregressive generation with sampling; decodes output for quick inspection.
    Prompt: A sample sentence on reading comprehension (replaceable for custom tests).

    Parameters
    ----------
    model : Transformer
        Trained model for generation.
    tokenizer : sp.SentencePieceProcessor
        Tokenizer for encoding prompt and decoding output.
    max_seq_len : int
        Maximum tokens to generate (beyond BOS).
    device : str | torch.device
            Computation device (e.g., "cuda").

    Returns
    -------
    translation : str
        Decoded generated text (skipping BOS, stopping at EOS).
    """
    tokenized_text: Tensor = (
        torch.tensor(tokenizer.Encode(text, int, add_bos=True, add_eos=True))
        .unsqueeze(0)
        .to(device)
    )
    generated = model.generate(
        tokenized_text,
        max_seq_len,
        device,
        temperature=0.7,
        do_sample=True,
        top_k=50,
    ).squeeze(0)
    out = []
    for tok in generated:
        tok = tok.item()
        if tok != tokenizer.eos_id():
            out.append(tok)
        else:
            break
    return tokenizer.decode(out[1:])
