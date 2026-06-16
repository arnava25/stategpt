"""
core/scorer.py

GPT log-probability surprisal scoring — shared by brain/ and conversation/.

Uses a batched sliding-window approach: builds all context windows as a
single tensor and runs one forward pass per batch, making full use of
MPS / CUDA instead of the original one-token-at-a-time loop.

Speedup vs naive loop: ~50-100x on MPS for typical sequence lengths.
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.model import load_checkpoint


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def score_tokens(tokens: np.ndarray, ckpt_path: str,
                 device: str = None,
                 batch_size: int = 512,
                 verbose: bool = True) -> np.ndarray:
    """
    Compute per-token GPT surprisal (negative log probability, nats).

    Batched sliding-window: all context windows are stacked into one
    tensor and processed in batches, fully utilizing MPS/CUDA.

    Args:
        tokens:     int64 array of state tokens, shape (N,)
        ckpt_path:  path to checkpoint saved by core/trainer.py
        device:     'cpu', 'cuda', or 'mps' -- auto-detected if None
        batch_size: number of windows per forward pass (tune for VRAM)
        verbose:    print progress

    Returns:
        surprisal: float32 array, shape (N,), in nats
                   surprisal[0] = 0.0 by convention
    """
    if device is None:
        device = get_device()

    model, ckpt = load_checkpoint(ckpt_path, device)
    block_size  = ckpt["block_size"]
    tokens      = tokens.astype(np.int64)
    N           = len(tokens)

    if verbose:
        print(f"  Scoring {N} tokens on {device} "
              f"(block={block_size}, batch={batch_size})...")

    # Pad the token sequence on the left so every position has a full context
    pad    = np.zeros(block_size, dtype=np.int64)
    padded = np.concatenate([pad, tokens])          # length N + block_size

    # Build all context windows at once: shape (N, block_size)
    # windows[i] = padded[i : i+block_size]  (context for predicting tokens[i])
    idx     = np.arange(N)[:, None] + np.arange(block_size)[None, :]
    windows = padded[idx]                           # (N, block_size)
    targets = tokens                                # (N,)

    surprisal = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        ctx  = torch.tensor(windows[start:end], dtype=torch.long,
                            device=device)          # (B, block_size)
        tgt  = torch.tensor(targets[start:end], dtype=torch.long,
                            device=device)          # (B,)

        logits   = model(ctx)                       # (B, block_size, vocab)
        last     = logits[:, -1, :]                 # (B, vocab)
        log_prob = F.log_softmax(last, dim=-1)      # (B, vocab)
        lp       = log_prob[torch.arange(end - start, device=device), tgt]
        surprisal[start:end] = (-lp).cpu().numpy()

        if verbose and (start // batch_size) % 10 == 0:
            pct = 100 * end / N
            print(f"  {end}/{N} ({pct:.0f}%)", end="\r")

    # First token has no real context -- set to 0 by convention
    surprisal[0] = 0.0

    if verbose:
        print(f"  Done. mean={surprisal.mean():.4f}  max={surprisal.max():.4f}")

    return surprisal


def classify_surprisal(surprisal: np.ndarray) -> list:
    """
    Bucket per-token surprisal into levels using percentiles.
    Returns list of strings: 'start', 'low', 'medium', 'high'
    """
    nonzero = surprisal[surprisal > 0]
    if len(nonzero) == 0:
        return ["none"] * len(surprisal)

    low_thresh  = np.percentile(nonzero, 50)
    high_thresh = np.percentile(nonzero, 80)

    labels = []
    for v in surprisal:
        if v == 0.0:
            labels.append("start")
        elif v >= high_thresh:
            labels.append("high")
        elif v >= low_thresh:
            labels.append("medium")
        else:
            labels.append("low")
    return labels


def surprisal_summary(surprisal: np.ndarray, labels: list) -> dict:
    """Return a dict of summary statistics for reporting."""
    from collections import Counter
    nonzero = surprisal[surprisal > 0]
    return {
        "n_tokens":     len(surprisal),
        "mean":         float(surprisal.mean()),
        "std":          float(surprisal.std()),
        "max":          float(surprisal.max()),
        "max_idx":      int(surprisal.argmax()),
        "min_nonzero":  float(nonzero.min()) if len(nonzero) else 0.0,
        "level_counts": dict(Counter(labels)),
    }
