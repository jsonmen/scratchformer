import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_STEPS,
    D_FF,
    D_MODEL,
    DATASET_TRAIN_FILE,
    DATASET_VAL_FILE,
    DROPOUT,
    EPOCH,
    FP16,
    GRAD_CLIP,
    LABEL_SMOOTHING,
    LOAD_CHECKPOINT,
    LR_SCALE,
    MAX_SEQ_LEN,
    N_DECODER_LAYERS,
    N_ENCODER_LAYERS,
    N_HEADS,
    NUM_WORKERS,
    PIN_MEMORY,
    SUBSET_SIZE,
    TENSORBOARD_OUTPUT_DIR,
    TOKENIZER_OPTIONS,
    TOKENIZER_PATH,
    VOCAB_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    ACCUM_STEPS,
    TEST_TRANSLATION_TEXT,
)
from dataset import WMTDataset
from model import Transformer
from tokenizer import load_tokenizer
from utils import collate_fn, load_last_checkpoint, test_translation, val_score

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def lr_lambda(steps: int) -> float:
    """
    Computes the Noam learning rate schedule from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper (Section 5.3).

    Combines linear warmup and inverse square-root decay for stable Transformer training.
    Formula: d_model^{-0.5} * min(LR_SCALE * step^{-0.5}, LR_SCALE * step * warmup^{-1.5}).

    Parameters
    ----------
    steps : int
        Current training step (global batch count; >=1).

    Returns
    -------
    lr_multiplier : float
        Multiplier to apply to base LR (e.g., peaks ~1e-3, then decays).
    """
    steps = max(steps, 1)
    return D_MODEL ** (-0.5) * min(
        LR_SCALE * (steps ** (-0.5)), LR_SCALE * (steps * WARMUP_STEPS ** (-1.5))
    )


def train(num_epochs: int):
    Path("checkpoints/").mkdir(exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_PATH, TOKENIZER_OPTIONS, DATASET_TRAIN_FILE)
    writer = SummaryWriter(TENSORBOARD_OUTPUT_DIR)

    # Train Dataset
    dataset = WMTDataset(DATASET_TRAIN_FILE, tokenizer, MAX_SEQ_LEN, SUBSET_SIZE)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Val Dataset
    val_dataset = WMTDataset(DATASET_VAL_FILE, tokenizer, MAX_SEQ_LEN, SUBSET_SIZE)
    val_dataloader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    pad_idx = int(tokenizer.pad_id())
    bos_idx = int(tokenizer.bos_id())
    eos_idx = int(tokenizer.eos_id())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
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
    model, last_epoch, last_step = load_last_checkpoint(
        LOAD_CHECKPOINT, model, CHECKPOINT_PATH
    )
    model = model.to(device)
    model.compile()

    scaler = GradScaler(enabled=FP16)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler._step_count = last_step
    scheduler.last_epoch = last_step

    print(f"Using {device}")
    print(
        "Num of params:", sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    step = last_step
    train_loss_total = 0.0
    non_pad_total = 0

    for epoch in range(last_epoch + 1, num_epochs):
        translation_txt = test_translation(
            TEST_TRANSLATION_TEXT, model, tokenizer, MAX_SEQ_LEN, device
        )
        print(translation_txt)
        exit()
        writer.add_text("translation", translation_txt, epoch)
        model.train()
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{num_epochs}", leave=True)
        for batch_idx, item in enumerate(loop):
            src = item["src"].to(device)
            tgt = item["tgt"].to(device)
            # forward + loss
            with torch.autocast(device_type="cuda", enabled=FP16):
                logits = model.forward(src, tgt[:, :-1])
                logits = logits.reshape(-1, logits.shape[-1])
                targets = tgt[:, 1:].reshape(-1)
                loss = loss_fn(logits, targets)
            # Token count
            non_pad = (targets != pad_idx).sum().item()
            train_loss_total += loss.item() * non_pad
            non_pad_total += non_pad
            # Backprop with accumulation
            loss = loss / ACCUM_STEPS  # Normalize for avg
            scaler.scale(loss).backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            step += 1
            ppl = torch.exp(loss * ACCUM_STEPS).item()  # De-normalize for display
            loop.set_postfix(
                ppl=ppl,
                lr=scheduler.get_last_lr()[0],
                tgt=tgt.shape,
                src=src.shape,
            )
            writer.add_scalar("train/loss", loss.item() * ACCUM_STEPS, step)
            writer.add_scalar("train/perplexity", ppl, step)
            if batch_idx % CHECKPOINT_STEPS == 0:
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "step": step},
                    os.path.join(
                        CHECKPOINT_PATH, f"model_checkpoint_{epoch}_{batch_idx}.pt"
                    ),
                )
        epoch_ppl = (
            math.exp(train_loss_total / non_pad_total)
            if non_pad_total > 0
            else float("inf")
        )
        writer.add_scalar("train/epoch_perplexity", epoch_ppl, epoch)
        train_loss_total = 0.0
        non_pad_total = 0
        # Val
        avg_val_loss, avg_val_perplexity = val_score(
            val_dataloader, model, device, tokenizer
        )
        writer.add_scalar("val/avg_loss", avg_val_loss, epoch)
        writer.add_scalar("val/avg_perplexity", avg_val_perplexity, epoch)

    writer.close()


if __name__ == "__main__":
    train(EPOCH)
