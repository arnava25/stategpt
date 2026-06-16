"""
brain/build_tokens.py — fixed for high-dimensional fMRI data.

Uses PCA to reduce to 10 components before discretizing,
avoiding integer overflow with 100-channel data.
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from collections import Counter


def discretize(timeseries: np.ndarray, n_micro: int = 50,
               n_pca: int = 10) -> np.ndarray:
    """
    Discretize a multivariate timeseries into integer micro-state tokens.

    For high-dimensional data (>10 channels), applies PCA first to
    avoid integer overflow when hashing channel bins.

    Args:
        timeseries: (T, N) float array
        n_micro:    number of bins per PCA component
        n_pca:      number of PCA components to retain

    Returns:
        tokens: (T,) int64 array of micro-state indices
    """
    T, N = timeseries.shape

    # Reduce dimensionality if needed
    if N > n_pca:
        pca   = PCA(n_components=n_pca, random_state=42)
        data  = pca.fit_transform(timeseries)   # (T, n_pca)
    else:
        data  = timeseries
        n_pca = N

    # Normalize each component to [0, 1]
    mn = data.min(axis=0, keepdims=True)
    mx = data.max(axis=0, keepdims=True)
    normed = (data - mn) / np.clip(mx - mn, 1e-8, None)

    # Bin each component
    bins = np.floor(normed * n_micro).astype(np.int64).clip(0, n_micro - 1)

    # Hash to single integer using mixed radix encoding
    # n_micro^n_pca must fit in int64: 50^10 = 9.76e16 < 9.2e18 ✓
    tokens = np.zeros(T, dtype=np.int64)
    multiplier = np.int64(1)
    for ch in range(n_pca):
        tokens += bins[:, ch] * multiplier
        multiplier *= np.int64(n_micro)

    # Re-index to contiguous integers starting at 0
    unique_vals = np.unique(tokens)
    remap = {int(v): i for i, v in enumerate(unique_vals)}
    tokens = np.array([remap[int(t)] for t in tokens], dtype=np.int64)

    return tokens


def build_macro_states(tokens: np.ndarray, k: int = 10) -> tuple:
    """
    Spectral clustering on token transition graph → macro tokens.
    Returns (macro_tokens, token_to_macro dict).
    """
    # Compress to runs
    runs = [tokens[0]]
    for t in tokens[1:]:
        if t != runs[-1]:
            runs.append(t)
    runs = np.array(runs, dtype=int)

    states = np.unique(runs)
    idx    = {s: i for i, s in enumerate(states)}
    M      = np.zeros((len(states), len(states)), dtype=np.float32)

    for a, b in zip(runs[:-1], runs[1:]):
        M[idx[a], idx[b]] += 1.0

    A = M + M.T
    A += np.eye(A.shape[0], dtype=np.float32) * 1e-3

    k  = min(k, len(states))
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=0)
    labels = sc.fit_predict(A)

    token_to_macro = {int(s): int(labels[idx[s]]) for s in states}
    macro = np.array([token_to_macro[t] for t in tokens], dtype=np.int64)

    return macro, token_to_macro
