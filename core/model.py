"""
core/model.py

TinyGPT — the single model definition shared by both pipelines.
Both brain/ and conversation/ import from here. Never copy-paste this.
"""

import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    """
    Small causal transformer for learning state token sequences.
    Works identically on brain macro-state tokens and conversation
    latent-state tokens — the two pipelines differ only in how they
    produce tokens, not in how the model processes them.

    Args:
        vocab_size:  number of distinct state tokens
        block_size:  context window length (tokens)
        d_model:     embedding / hidden dimension (default 96)
        n_heads:     attention heads (default 4)
        n_layers:    transformer layers (default 2)
        dropout:     dropout rate (set 0.0 at inference)
    """

    def __init__(self, vocab_size: int, block_size: int,
                 d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok  = nn.Embedding(vocab_size, d_model)
        self.pos  = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm = nn.Linear(d_model, vocab_size)

        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) long tensor of token indices, T <= block_size
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok(idx) + self.pos(pos)[None, :, :]
        h = self.drop(h)
        h = self.tr(h, mask=self.causal_mask[:T, :T])
        return self.lm(h)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: TinyGPT, path: str, extra: dict = None):
    """Save model + architecture config so it can be fully reconstructed."""
    payload = {
        "state_dict": model.state_dict(),
        "vocab_size":  model.vocab_size,
        "block_size":  model.block_size,
        "d_model":     model.tok.embedding_dim,
        "n_heads":     model.tr.layers[0].self_attn.num_heads,
        "n_layers":    len(model.tr.layers),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> TinyGPT:
    """Load a checkpoint saved by save_checkpoint(). Returns eval-mode model."""
    ckpt = torch.load(path, map_location="cpu")
    model = TinyGPT(
        vocab_size  = ckpt["vocab_size"],
        block_size  = ckpt["block_size"],
        d_model     = ckpt.get("d_model",  96),
        n_heads     = ckpt.get("n_heads",   4),
        n_layers    = ckpt.get("n_layers",  2),
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
