"""
brain/analyze.py

Brain surprisal analysis: event alignment, timeline plots.
Replaces: macro_surprisal_event_alignment.py + plot_event_locked_surprisal_v2.py

Usage:
    python -m brain.analyze
    python -m brain.analyze --events brain/data/task_events.npy
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.scorer import classify_surprisal, surprisal_summary


def align_events_to_tokens(events: np.ndarray, n_tokens: int) -> np.ndarray:
    """
    Downsample a long events timeseries to match the token sequence length.
    Groups events into bins of size len(events)//n_tokens and marks any
    bin that contains at least one event.
    """
    bin_size = max(1, len(events) // n_tokens)
    n_bins   = n_tokens
    aligned  = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        chunk = events[i * bin_size: (i + 1) * bin_size]
        if len(chunk) > 0 and chunk.max() > 0:
            aligned[i] = 1.0
    return aligned


def plot_surprisal_timeline(tokens, surprisal, labels, outpath, events=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    N = len(tokens)
    xs = np.arange(N)

    # ── Top: macro state sequence ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(xs, tokens, linewidth=0.5, alpha=0.7, color="#7F77DD")
    ax.set_ylabel("Macro state")
    ax.set_title("Brain macro-state sequence")
    ax.set_ylim(-0.5, tokens.max() + 0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)

    if events is not None:
        event_xs = np.where(events > 0)[0]
        for ex in event_xs:
            ax.axvline(ex, color="#c85252", alpha=0.4, linewidth=0.8)

    # ── Bottom: surprisal ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(xs, surprisal, linewidth=0.8, alpha=0.8, color="#5260c8")
    ax.fill_between(xs, surprisal, alpha=0.12, color="#5260c8")

    high_mask = np.array(labels) == "high"
    if high_mask.any():
        ax.scatter(xs[high_mask], surprisal[high_mask],
                   color="#c85252", s=8, zorder=5, label="high surprise", alpha=0.7)

    if events is not None:
        event_xs = np.where(events > 0)[0]
        for ex in event_xs:
            ax.axvline(ex, color="#c85252", alpha=0.25, linewidth=0.8)

    ax.set_ylabel("Surprisal (nats)")
    ax.set_xlabel("Token index")
    ax.set_title("GPT surprisal — brain macro states")
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.2)
    if high_mask.any():
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
    print(f"Saved: {outpath}")


def plot_event_locked(surprisal, events, outpath, window=50):
    """Average surprisal locked to event onset (in token space)."""
    event_times = np.where(events > 0)[0]
    event_times = event_times[
        (event_times >= window) & (event_times < len(surprisal) - window)
    ]

    if len(event_times) == 0:
        print("No valid event times for event-locked plot.")
        return

    snippets  = np.stack([surprisal[t - window: t + window] for t in event_times])
    mean_surp = snippets.mean(axis=0)
    sem_surp  = snippets.std(axis=0) / np.sqrt(len(snippets))
    lags      = np.arange(-window, window)

    # Baseline: mean surprisal in pre-event window
    baseline  = mean_surp[:window].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(lags, mean_surp, linewidth=1.5, color="#5260c8", label="mean surprisal")
    plt.fill_between(lags,
                     mean_surp - sem_surp,
                     mean_surp + sem_surp,
                     alpha=0.2, color="#5260c8", label="±1 SEM")
    plt.axhline(baseline, color="#888780", linestyle=":", linewidth=1,
                label=f"pre-event baseline ({baseline:.3f})")
    plt.axvline(0, color="#c85252", linestyle="--", linewidth=1.2,
                label="event onset")

    # Shade post-event window
    plt.axvspan(0, window, alpha=0.05, color="#c85252")

    post_mean = mean_surp[window:].mean()
    delta     = post_mean - baseline
    plt.title(
        f"Event-locked surprisal  (n={len(event_times)} events, ±{window} steps)\n"
        f"post−pre Δ = {delta:+.3f} nats"
    )
    plt.xlabel("Lag (token steps relative to event onset)")
    plt.ylabel("Mean surprisal (nats)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="brain/data")
    parser.add_argument("--surprisal", default=None)
    parser.add_argument("--events",    default=None)
    parser.add_argument("--outdir",    default="outputs/brain")
    args = parser.parse_args()

    outdir   = Path(args.outdir)
    data_dir = Path(args.data)
    outdir.mkdir(parents=True, exist_ok=True)

    tokens = np.load(data_dir / "macro_tokens.npy")
    print(f"Loaded macro tokens: {tokens.shape}  vocab={int(tokens.max()+1)}")

    if args.surprisal:
        surprisal = np.load(args.surprisal)
    else:
        default = outdir / "brain_surprisal.npy"
        if default.exists():
            surprisal = np.load(default)
            print(f"Loaded surprisal from {default}")
        else:
            print("No surprisal file. Run 'python run.py brain --score' first.")
            return

    labels  = classify_surprisal(surprisal)
    summary = surprisal_summary(surprisal, labels)

    print("\nSurprisal summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Align events to token space if provided
    events_aligned = None
    if args.events:
        raw_events = np.load(args.events)
        print(f"\nEvents file: {raw_events.shape}  tokens: {tokens.shape}")
        if len(raw_events) != len(tokens):
            print(f"  Downsampling events {len(raw_events)} → {len(tokens)} bins")
            events_aligned = align_events_to_tokens(raw_events, len(tokens))
        else:
            events_aligned = raw_events.astype(np.float32)
        n_events = int((events_aligned > 0).sum())
        print(f"  Event onsets in token space: {n_events}")

    plot_surprisal_timeline(
        tokens, surprisal, labels,
        outdir / "brain_surprisal_timeline.png",
        events=events_aligned,
    )

    if events_aligned is not None:
        plot_event_locked(
            surprisal, events_aligned,
            outdir / "brain_event_locked_surprisal.png",
        )

    print(f"\nOutputs saved to {outdir}/")


if __name__ == "__main__":
    main()