from dataset import WMTDataset
from utils import load_last_checkpoint
from model import Transformer
from tokenizer import load_tokenizer
from config import (
    DATASET_TEST_FILE,
    TOKENIZER_PATH,
    TOKENIZER_OPTIONS,
    DATASET_TRAIN_FILE,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_PATH,
    PIN_MEMORY,
    N_DECODER_LAYERS,
    N_ENCODER_LAYERS,
    N_HEADS,
    D_FF,
    D_MODEL,
    VOCAB_SIZE,
    DROPOUT,
)
from utils import collate_fn
from evaluate import load, EvaluationModule
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sentencepiece as sp
import math


@torch.no_grad()
def test(
    test_dataloader: DataLoader,
    model: Transformer,
    bleu_fn: EvaluationModule,
    loss_fn: torch.nn.Module,
    tokenizer: sp.SentencePieceProcessor,
    device: str | torch.device,
) -> tuple[float, float, float]:
    model.eval()
    model.to(device)
    avg_loss = 0.0
    num_samples = 0
    predictions = []
    references = []
    loop = tqdm(test_dataloader, desc="Test", leave=True)
    for item in loop:
        src = item["src"].to(device)
        tgt = item["tgt"].to(device)

        # Teacher-forced loss and perplexity (standard eval)
        tgt_input = tgt[:, :-1]  # Shifted input
        tgt_labels = tgt[:, 1:]  # Shifted labels
        logits = model.forward(src, tgt_input)  # (batch=1, tgt_seq-1, vocab)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_labels.reshape(-1))

        # Accumulate average loss (weighted by non-pad tokens for accuracy)
        non_pad = (tgt_labels != tokenizer.pad_id()).sum().item()
        avg_loss += loss.item() * non_pad
        num_samples += non_pad

        # Generation for BLEU
        generated = model.generate(
            src,
            MAX_SEQ_LEN,
            device,
            temperature=0.7,
            do_sample=True,
            top_k=50,
        )

        # Process generated: Stop at EOS, include EOS
        for batch_idx in range(generated.size(0)):
            tmp_seq = []
            for tok in generated[batch_idx]:
                tok = tok.item()
                tmp_seq.append(tok)
                if tok == tokenizer.eos_id():
                    break
            predictions.append(tokenizer.decode(tmp_seq))
            references.append([tokenizer.decode(tgt[batch_idx].tolist())])

    # Final averages
    avg_loss = avg_loss / num_samples if num_samples > 0 else float("inf")
    avg_perplexity = math.exp(avg_loss)

    # Corpus BLEU
    bleu_results = bleu_fn.compute(predictions=predictions, references=references)
    avg_bleu_score = bleu_results["bleu"]

    return avg_loss, avg_perplexity, avg_bleu_score


def main():
    tokenizer = load_tokenizer(TOKENIZER_PATH, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE)
    test_dataset = WMTDataset(DATASET_TEST_FILE, tokenizer, MAX_SEQ_LEN)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,  # Deterministic eval
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    pad_idx = int(tokenizer.pad_id())
    bos_idx = int(tokenizer.bos_id())
    eos_idx = int(tokenizer.eos_id())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bleu = load("bleu")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    model = Transformer(
        N_ENCODER_LAYERS,
        N_DECODER_LAYERS,
        D_MODEL,
        D_FF,
        N_HEADS,
        MAX_SEQ_LEN,
        VOCAB_SIZE,
        DROPOUT,
        pad_idx,
        bos_idx,
        eos_idx,
        pe_device=device,
    )
    model, _, _ = load_last_checkpoint(True, model, CHECKPOINT_PATH)
    avg_loss, avg_ppl, avg_bleu = test(
        test_dataloader, model, bleu, loss_fn, tokenizer, device
    )
    print(f"Avg Loss: {avg_loss:.4f}, Avg PPL: {avg_ppl:.2f}, Avg BLEU: {avg_bleu:.2f}")


if __name__ == "__main__":
    main()
