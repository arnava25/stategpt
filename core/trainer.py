"""
core/trainer.py

Single training loop for TinyGPT, used by both brain/ and conversation/.
Both pipelines call train_on_tokens() with their respective .npy arrays.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.model import TinyGPT, save_checkpoint


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(data: np.ndarray, block_size: int, batch_size: int,
              device: str):
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Token sequence too short ({len(data)}) for block_size={block_size}. "
            "Use a smaller --block value."
        )
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:     i + block_size    ] for i in ix])
    y = np.stack([data[i + 1: i + block_size + 1] for i in ix])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


@torch.no_grad()
def estimate_loss(model: TinyGPT, data: np.ndarray, block_size: int,
                  batch_size: int, vocab_size: int, device: str,
                  iters: int = 40) -> float:
    model.eval()
    losses = []
    safe_iters = min(iters, max(1, len(data) // (block_size + 1)))
    for _ in range(safe_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_on_tokens(
    tokens:       np.ndarray,
    out_path:     str,
    steps:        int   = 2000,
    eval_every:   int   = 50,
    block_size:   int   = 64,
    batch_size:   int   = 64,
    lr:           float = 3e-4,
    weight_decay: float = 0.05,
    dropout:      float = 0.2,
    d_model:      int   = 96,
    n_heads:      int   = 4,
    n_layers:     int   = 2,
    source:       str   = "unknown",   # "brain" or "conversation" — logged only
) -> dict:
    """
    Train TinyGPT on a token sequence and save the best checkpoint.

    Args:
        tokens:    int64 numpy array of state tokens
        out_path:  where to save best_*.pt checkpoint
        source:    label for logging ("brain" or "conversation")

    Returns:
        dict with best_val, best_step, log (list of dicts)
    """
    tokens = tokens.astype(np.int64)
    vocab_size = int(tokens.max() + 1)

    # Auto-shrink block_size for short sequences
    block_size = min(block_size, len(tokens) // 4)
    if block_size < 4:
        raise ValueError(
            f"Token sequence too short ({len(tokens)}) to train meaningfully. "
            "Generate more data or reduce cluster count."
        )

    split = int(0.9 * len(tokens))
    train_data = tokens[:split]
    val_data   = tokens[split:]

    device = get_device()
    print(f"[{source}] Training on {device} | tokens={len(tokens)} "
          f"vocab={vocab_size} block={block_size}")

    model = TinyGPT(vocab_size, block_size, d_model, n_heads, n_layers,
                    dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{source}] Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    best_val  = float("inf")
    best_step = -1
    log       = []

    for step in range(1, steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0:
            tr = estimate_loss(model, train_data, block_size, batch_size,
                               vocab_size, device, iters=20)
            vl = estimate_loss(model, val_data,   block_size, batch_size,
                               vocab_size, device, iters=40)
            print(f"  step {step:5d}  train={tr:.4f}  val={vl:.4f}", end="")
            log.append({"step": step, "train_loss": tr, "val_loss": vl})

            if vl < best_val:
                best_val  = vl
                best_step = step
                save_checkpoint(model, out_path,
                                extra={"source": source, "step": step})
                print(f"  ✅ saved {out_path}", end="")
            print()

    # Write training log next to checkpoint
    log_path = str(out_path).replace(".pt", "_train_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log)

    print(f"[{source}] Best val={best_val:.4f} at step {best_step}")
    return {"best_val": best_val, "best_step": best_step, "log": log}
