"""
brain/simulate.py v2 — harder dynamics, GPT can't memorize perfectly.

Key changes from v1:
- More nodes (12), denser coupling, higher noise
- Condition A: gain=3.0 (strong, sustained drive, 150 steps)
- Condition B: gain=1.0 (moderate, brief drive, 50 steps)
- Chaotic-ish baseline so val_loss stays above 0
- Longer min_gap between events so recovery is measurable

Run: python brain/simulate.py
"""

import argparse
from pathlib import Path
import numpy as np


def sigmoid(x, a=4.0, theta=0.5):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def simulate_network(T=60000, N=12, dt=0.1, seed=42):
    rng = np.random.default_rng(seed)

    # Dense random coupling with stronger weights
    W = rng.uniform(0.0, 1.0, (N, N))
    W *= rng.uniform(0, 1, (N, N)) < 0.6   # 60% density
    np.fill_diagonal(W, 0)
    # Normalize but keep stronger coupling
    row_sums = W.sum(axis=1, keepdims=True) + 1e-8
    W = W / row_sums * 0.8

    # Inhibitory connections (random subset)
    inh_mask = rng.uniform(0, 1, (N, N)) < 0.3
    W[inh_mask] *= -0.5

    E = np.zeros((T, N), dtype=np.float32)
    e = rng.uniform(0.2, 0.6, N)

    noise_std = 0.08  # higher noise = harder to memorize

    for t in range(1, T):
        net = W @ e
        de  = (-e + sigmoid(net + rng.normal(0, noise_std, N))) / 15.0
        e   = np.clip(e + dt * de, 0.0, 1.0)
        E[t] = e

    return E


def generate_task(T=60000, n_events=36, seed=99,
                  gain_A=3.0, dur_A=150,
                  gain_B=1.0, dur_B=50,
                  min_gap=1000):
    rng = np.random.default_rng(seed)

    context   = np.zeros(T, dtype=np.int32)
    events    = np.zeros(T, dtype=np.float32)
    gain_A_ts = np.zeros(T, dtype=np.float32)
    gain_B_ts = np.zeros(T, dtype=np.float32)
    signal    = np.zeros(T, dtype=np.float32)

    placed = []
    attempts = 0
    while len(placed) < n_events and attempts < 200000:
        attempts += 1
        t = rng.integers(1000, T - 500)
        if all(abs(t - p) > min_gap for p in placed):
            placed.append(t)

    placed.sort()
    print(f"  Placed {len(placed)} events")

    for k, t in enumerate(placed):
        cond = 1 if k % 2 == 0 else 2
        gain = gain_A if cond == 1 else gain_B
        dur  = dur_A  if cond == 1 else dur_B

        context[t: t + dur] = cond
        events[t] = 1.0

        for dt in range(dur):
            if t + dt >= T:
                break
            # Raised cosine pulse shape
            amp = gain * 0.5 * (1 - np.cos(np.pi * dt / dur))
            if cond == 1:
                gain_A_ts[t + dt] = amp
            else:
                gain_B_ts[t + dt] = amp
            signal[t + dt] = amp

    return context, events, gain_A_ts, gain_B_ts, signal


def apply_drive(E, signal, n_target=None):
    N = E.shape[1]
    if n_target is None:
        n_target = N // 2
    E_out = E.copy()
    rng = np.random.default_rng(7)
    target_nodes = rng.choice(N, n_target, replace=False)
    for node in target_nodes:
        # Nonlinear drive — crosses tokenization boundaries more reliably
        drive = signal * 0.25 * (1 + 0.3 * np.sin(2 * np.pi * np.arange(len(signal)) / 50))
        E_out[:, node] = np.clip(E[:, node] + drive, 0, 1)
    return E_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",    type=int,   default=12)
    parser.add_argument("--duration", type=int,   default=60000)
    parser.add_argument("--events",   type=int,   default=36)
    parser.add_argument("--gain-a",   type=float, default=3.0)
    parser.add_argument("--gain-b",   type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--outdir",   default="brain/data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.nodes} nodes × {args.duration} steps...")
    E = simulate_network(T=args.duration, N=args.nodes, seed=args.seed)
    print(f"  E range: [{E.min():.3f}, {E.max():.3f}]  std={E.std():.3f}")

    print(f"Generating task (A: gain={args.gain_a} dur=150, B: gain={args.gain_b} dur=50)...")
    context, events, gain_A, gain_B, signal = generate_task(
        T=args.duration, n_events=args.events, seed=args.seed+1,
        gain_A=args.gain_a, dur_A=150,
        gain_B=args.gain_b, dur_B=50,
        min_gap=1000,
    )

    print("Applying task drive...")
    E_driven = apply_drive(E, signal)

    np.save(outdir / "brain_timeseries.npy", E_driven.astype(np.float32))
    np.save(outdir / "task_gain_A.npy",      gain_A)
    np.save(outdir / "task_gain_B.npy",      gain_B)
    np.save(outdir / "task_context.npy",     context)
    np.save(outdir / "task_events.npy",      events)
    np.save(outdir / "task_signal.npy",      signal)

    print(f"\nSaved to {outdir}/")
    for fname in ["brain_timeseries.npy","task_gain_A.npy","task_gain_B.npy",
                  "task_context.npy","task_events.npy"]:
        a = np.load(outdir / fname)
        print(f"  {fname}: {a.shape}")

    print("\nNext:")
    print("  python run.py brain --tokenize --timeseries brain/data/brain_timeseries.npy")
    print("  python run.py brain --train --steps 3000")
    print("  python run.py brain --score")
    print("  python brain/condition_analysis.py")

if __name__ == "__main__":
    main()
