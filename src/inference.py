import argparse

import torch

from config import (
    D_FF,
    D_MODEL,
    DATASET_TRAIN_FILE,
    DROPOUT,
    N_DECODER_LAYERS,
    N_ENCODER_LAYERS,
    N_HEADS,
    TOKENIZER_OPTIONS,
    TOKENIZER_PATH,
    VOCAB_SIZE,
)
from model import Transformer
from tokenizer import load_tokenizer
from utils import test_translation


def main():
    parser = argparse.ArgumentParser(
        description="CLI for Transformer en-de translation inference."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="English input text to translate.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="model/best_model.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=TOKENIZER_PATH,
        help="Path to SentencePiece model file.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Max sequence length for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature."
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Enable sampling (else greedy).",
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE)

    # Load model
    pad_idx = int(tokenizer.pad_id())
    bos_idx = int(tokenizer.bos_id())
    eos_idx = int(tokenizer.eos_id())
    device = torch.device(args.device)

    model = Transformer(
        N_ENCODER_LAYERS,
        N_DECODER_LAYERS,
        D_MODEL,
        D_FF,
        N_HEADS,
        args.max_seq_len,
        VOCAB_SIZE,
        DROPOUT,
        pad_idx,
        bos_idx,
        eos_idx,
        pe_device=device,
    )

    checkpoint_info = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint_info["model"])
    model.to(device)
    model.eval()

    # Generate translation
    with torch.no_grad():
        translation = test_translation(
            args.input,
            model,
            tokenizer,
            args.max_seq_len,
            device,
        )

    print(f"Input: {args.input}")
    print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
