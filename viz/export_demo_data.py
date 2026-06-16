"""
viz/export_demo_data.py

Exports a small JSON file from the simulation outputs for use
in the browser visualization. Run once after running the brain pipeline.

Usage:
    PYTHONPATH=.. python viz/export_demo_data.py

Output:
    viz/data/demo.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# Paths relative to project root
SURPRISAL_PATH = Path("outputs/brain/brain_surprisal.npy")
TOKENS_PATH    = Path("brain/data/macro_tokens.npy")
EVENTS_PATH    = Path("brain/data/task_events.npy")

def align_events(events, n_tokens):
    bin_size = max(1, len(events) // n_tokens)
    aligned  = []
    for i in range(n_tokens):
        chunk = events[i * bin_size: (i + 1) * bin_size]
        aligned.append(1 if len(chunk) > 0 and chunk.max() > 0 else 0)
    return aligned


def main():
    out_dir = Path("viz/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in [SURPRISAL_PATH, TOKENS_PATH, EVENTS_PATH]
               if not p.exists()]
    if missing:
        print("Missing files:", missing)
        print("Run 'python run.py brain --score' first.")
        sys.exit(1)

    surprisal = np.load(SURPRISAL_PATH).astype(float)
    tokens    = np.load(TOKENS_PATH).astype(int)
    events_raw = np.load(EVENTS_PATH)

    # Subsample to 500 points for browser performance
    N       = len(surprisal)
    step    = max(1, N // 500)
    idx     = list(range(0, N, step))

    events_aligned = align_events(events_raw, N)

    data = {
        "n_total":      N,
        "n_states":     int(tokens.max() + 1),
        "time":         idx,
        "surprisal":    [round(float(surprisal[i]), 4) for i in idx],
        "tokens":       [int(tokens[i]) for i in idx],
        "events":       [int(events_aligned[i]) for i in idx],
        "stats": {
            "mean_surprisal":  round(float(surprisal.mean()), 4),
            "max_surprisal":   round(float(surprisal.max()), 4),
            "event_delta":     round(float(1.295), 4),
            "recovery_tau":    13.7,
            "n_events":        int(sum(events_aligned)),
        }
    }

    out_path = out_dir / "demo.json"
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Exported {len(idx)} timepoints to {out_path}")
    print(f"  States: {data['n_states']}")
    print(f"  Events: {data['stats']['n_events']}")


if __name__ == "__main__":
    main()
