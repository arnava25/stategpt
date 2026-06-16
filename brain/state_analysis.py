"""
brain/state_analysis.py

Answers three questions:
  Q2. State identity — which macro-states are surprising to transition INTO?
      Are certain states almost always high-surprisal?
  Q3. (placeholder) Task condition A vs B — needs task_gain files, see below.
  Q4. Recovery time — how long does surprisal take to return to baseline
      after an event? Measure of neural resilience/flexibility.

Usage:
    python brain/state_analysis.py
    python brain/state_analysis.py --window 80
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Helpers ───────────────────────────────────────────────────────────────────

def align_events(events: np.ndarray, n_tokens: int) -> np.ndarray:
    bin_size = max(1, len(events) // n_tokens)
    aligned  = np.zeros(n_tokens, dtype=np.float32)
    for i in range(n_tokens):
        chunk = events[i * bin_size: (i + 1) * bin_size]
        if len(chunk) > 0 and chunk.max() > 0:
            aligned[i] = 1.0
    return aligned


def baseline_surprisal(surprisal, events, pre_window=50):
    """Mean surprisal in the pre_window steps before any event."""
    event_times = np.where(events > 0)[0]
    event_times = event_times[event_times >= pre_window]
    if len(event_times) == 0:
        return surprisal.mean()
    pre_vals = np.concatenate([surprisal[t - pre_window: t] for t in event_times])
    return float(pre_vals.mean())


# ── Q2: State identity ────────────────────────────────────────────────────────

def analyze_state_identity(tokens, surprisal, outdir):
    """
    For each macro-state, compute:
      - mean surprisal when transitioning INTO that state
      - frequency (how often it appears)
      - mean dwell time (how long the brain stays in it)
    """
    n_states = int(tokens.max()) + 1

    # Per-state surprisal (surprisal[i] = cost of arriving at tokens[i])
    state_surprisals = defaultdict(list)
    for i in range(1, len(tokens)):
        state_surprisals[tokens[i]].append(surprisal[i])

    # Dwell times — run-length encoding
    state_dwells = defaultdict(list)
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        state_dwells[tokens[i]].append(j - i)
        i = j

    states      = list(range(n_states))
    mean_surp   = [np.mean(state_surprisals[s]) if state_surprisals[s] else 0.0
                   for s in states]
    std_surp    = [np.std(state_surprisals[s])  if state_surprisals[s] else 0.0
                   for s in states]
    frequency   = [len(state_surprisals[s]) / len(tokens) for s in states]
    mean_dwell  = [np.mean(state_dwells[s])  if state_dwells[s] else 0.0
                   for s in states]

    # Sort by mean surprisal for the bar chart
    order = np.argsort(mean_surp)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Q2 — Macro-state identity: which states are surprising?",
                 fontsize=13, y=1.02)

    # Panel 1: mean surprisal per state
    ax = axes[0]
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_states))
    bars = ax.bar([f"S{s}" for s in order],
                  [mean_surp[s] for s in order],
                  yerr=[std_surp[s] for s in order],
                  color=[colors[i] for i in range(n_states)],
                  capsize=4, alpha=0.85)
    ax.set_xlabel("Macro state")
    ax.set_ylabel("Mean surprisal (nats)")
    ax.set_title("Surprisal on entry\n(sorted high → low)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: frequency
    ax = axes[1]
    ax.bar([f"S{s}" for s in order],
           [frequency[s] for s in order],
           color=[colors[i] for i in range(n_states)],
           alpha=0.85)
    ax.set_xlabel("Macro state")
    ax.set_ylabel("Frequency (fraction of time)")
    ax.set_title("How often each state occurs\n(same ordering)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: mean dwell time
    ax = axes[2]
    ax.bar([f"S{s}" for s in order],
           [mean_dwell[s] for s in order],
           color=[colors[i] for i in range(n_states)],
           alpha=0.85)
    ax.set_xlabel("Macro state")
    ax.set_ylabel("Mean dwell time (steps)")
    ax.set_title("How long brain stays in state\n(same ordering)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = outdir / "q2_state_identity.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print table
    print("\nQ2 — State identity table (sorted by mean surprisal):")
    print(f"{'State':>6} {'MeanSurp':>10} {'StdSurp':>9} {'Freq%':>7} {'MeanDwell':>10}")
    print("-" * 48)
    for s in order:
        print(f"  S{s:>2}   {mean_surp[s]:>9.3f}  {std_surp[s]:>8.3f}  "
              f"{frequency[s]*100:>6.1f}%  {mean_dwell[s]:>9.1f}")

    return {s: {"mean_surp": mean_surp[s], "freq": frequency[s],
                "mean_dwell": mean_dwell[s]} for s in states}


# ── Q4: Recovery time ─────────────────────────────────────────────────────────

def analyze_recovery_time(surprisal, events, outdir, window=80, threshold_factor=1.5):
    """
    After each event, measure how many steps it takes for surprisal
    to return to within threshold_factor * baseline.

    threshold_factor=1.5 means "recovered when surprisal < 1.5x baseline"
    """
    base      = baseline_surprisal(surprisal, events)
    threshold = base * threshold_factor
    print(f"\nQ4 — Recovery time")
    print(f"  Baseline surprisal: {base:.4f} nats")
    print(f"  Recovery threshold: {threshold:.4f} nats ({threshold_factor}x baseline)")

    event_times = np.where(events > 0)[0]
    event_times = event_times[
        (event_times >= window) & (event_times < len(surprisal) - window)
    ]

    recovery_times = []
    snippets       = []

    for t in event_times:
        post = surprisal[t: t + window]
        snippets.append(post)

        # Find first position where surprisal drops back below threshold
        recovered = np.where(post <= threshold)[0]
        if len(recovered) > 0:
            recovery_times.append(int(recovered[0]))
        else:
            recovery_times.append(window)  # didn't recover within window

    recovery_times = np.array(recovery_times)
    snippets_arr   = np.array(snippets)
    mean_snippet   = snippets_arr.mean(axis=0)
    sem_snippet    = snippets_arr.std(axis=0) / np.sqrt(len(snippets_arr))
    lags           = np.arange(window)

    print(f"  N events analyzed: {len(recovery_times)}")
    print(f"  Mean recovery time: {recovery_times.mean():.1f} steps")
    print(f"  Median recovery time: {np.median(recovery_times):.1f} steps")
    print(f"  Recovered within window: "
          f"{(recovery_times < window).sum()}/{len(recovery_times)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q4 — Recovery time: how long to return to baseline after event?",
                 fontsize=13, y=1.02)

    # Panel 1: mean post-event surprisal with recovery threshold
    ax = axes[0]
    ax.plot(lags, mean_snippet, linewidth=1.8, color="#5260c8", label="mean surprisal")
    ax.fill_between(lags,
                    mean_snippet - sem_snippet,
                    mean_snippet + sem_snippet,
                    alpha=0.2, color="#5260c8", label="±1 SEM")
    ax.axhline(base,      color="#888780", linestyle=":",  linewidth=1.2,
               label=f"baseline ({base:.3f})")
    ax.axhline(threshold, color="#c85252", linestyle="--", linewidth=1.2,
               label=f"recovery threshold ({threshold:.3f})")
    ax.axvline(recovery_times.mean(), color="#1D9E75", linestyle="-.",
               linewidth=1.5,
               label=f"mean recovery ({recovery_times.mean():.1f} steps)")
    ax.set_xlabel("Steps after event onset")
    ax.set_ylabel("Mean surprisal (nats)")
    ax.set_title("Post-event surprisal decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: distribution of recovery times
    ax = axes[1]
    bins = min(20, len(np.unique(recovery_times)))
    ax.hist(recovery_times, bins=bins, color="#5260c8", alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(recovery_times.mean(),   color="#c85252", linewidth=1.5,
               linestyle="--", label=f"mean={recovery_times.mean():.1f}")
    ax.axvline(np.median(recovery_times), color="#1D9E75", linewidth=1.5,
               linestyle="-.", label=f"median={np.median(recovery_times):.1f}")
    ax.set_xlabel("Recovery time (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of recovery times\nacross events")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = outdir / "q4_recovery_time.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    return recovery_times


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="brain/data")
    parser.add_argument("--surprisal", default="outputs/brain/brain_surprisal.npy")
    parser.add_argument("--events",    default="brain/data/task_events.npy")
    parser.add_argument("--outdir",    default="outputs/brain")
    parser.add_argument("--window",    type=int, default=80,
                        help="Post-event window for recovery analysis (steps)")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Recovery threshold as multiple of baseline")
    args = parser.parse_args()

    outdir   = Path(args.outdir)
    data_dir = Path(args.data)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    tokens    = np.load(data_dir / "macro_tokens.npy")
    surprisal = np.load(args.surprisal)
    raw_events = np.load(args.events)

    print(f"Tokens:    {tokens.shape}  vocab={int(tokens.max()+1)}")
    print(f"Surprisal: {surprisal.shape}  mean={surprisal.mean():.4f}")
    print(f"Events:    {raw_events.shape}")

    # Align events to token space
    if len(raw_events) != len(tokens):
        events = np.zeros(len(tokens), dtype=np.float32)
        bin_size = len(raw_events) // len(tokens)
        for i in range(len(tokens)):
            chunk = raw_events[i * bin_size: (i + 1) * bin_size]
            if len(chunk) > 0 and chunk.max() > 0:
                events[i] = 1.0
    else:
        events = raw_events.astype(np.float32)

    n_events = int((events > 0).sum())
    print(f"Event onsets (token space): {n_events}")

    # Run analyses
    print("\n" + "=" * 60)
    state_info = analyze_state_identity(tokens, surprisal, outdir)

    print("\n" + "=" * 60)
    recovery_times = analyze_recovery_time(
        surprisal, events, outdir,
        window=args.window,
        threshold_factor=args.threshold,
    )

    print(f"\nDone. Plots saved to {outdir}/")
    print("  q2_state_identity.png")
    print("  q4_recovery_time.png")
    print("\nFor Q3 (task condition A vs B), copy task_gain_A.npy and")
    print("task_gain_B.npy into brain/data/ and run:")
    print("  python brain/condition_analysis.py")


if __name__ == "__main__":
    main()
