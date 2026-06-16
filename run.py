#!/usr/bin/env python3
"""
run.py — brain state trajectory pipeline

Stages:
  simulate   → generate neural timeseries + task structure
  tokenize   → discretize timeseries into macro-state tokens
  train      → train TinyGPT on token sequence
  score      → compute per-token GPT surprisal
  analyze    → surprisal timeline + event-locked plots
  q2         → state identity (which states are surprising?)
  q3         → condition A vs B state occupancy
  q4         → recovery time after task events

Usage:
  python run.py simulate
  python run.py tokenize
  python run.py train --steps 3000
  python run.py score
  python run.py analyze --events brain/data/task_events.npy
  python run.py q2
  python run.py q3
  python run.py q4
  python run.py all       # full pipeline from tokenize onward
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def get_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── Stages ────────────────────────────────────────────────────────────────────

def run_simulate(args):
    section("Simulate")
    from brain.simulate import simulate_network, generate_task, apply_drive
    outdir = Path("brain/data")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.nodes} nodes × {args.duration} steps...")
    E = simulate_network(T=args.duration, N=args.nodes, seed=args.seed)

    print("Generating task structure...")
    context, events, gain_A, gain_B, signal = generate_task(
        T=args.duration, n_events=args.events, seed=args.seed+1,
        gain_A=args.gain_a, dur_A=150,
        gain_B=args.gain_b, dur_B=50,
        min_gap=1000,
    )

    E_driven = apply_drive(E, signal)

    np.save(outdir / "brain_timeseries.npy", E_driven.astype(np.float32))
    np.save(outdir / "task_gain_A.npy",      gain_A)
    np.save(outdir / "task_gain_B.npy",      gain_B)
    np.save(outdir / "task_context.npy",     context)
    np.save(outdir / "task_events.npy",      events)
    np.save(outdir / "task_signal.npy",      signal)
    print(f"Saved to {outdir}/")


def run_tokenize(args):
    section("Tokenize")
    from brain.build_tokens import discretize, build_macro_states
    ts_path = Path(args.timeseries)
    if not ts_path.exists():
        sys.exit(f"Error: {ts_path} not found.")

    ts = np.load(ts_path)
    print(f"Timeseries: {ts.shape}")

    tokens = discretize(ts, n_micro=args.n_micro)
    macro, _ = build_macro_states(tokens, k=args.k)

    outdir = Path("brain/data")
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "brain_tokens.npy", tokens)
    np.save(outdir / "macro_tokens.npy", macro)
    print(f"Saved brain_tokens.npy + macro_tokens.npy  vocab={int(macro.max()+1)}")


def run_train(args):
    section("Train GPT")
    from core.trainer import train_on_tokens
    token_path = Path("brain/data/macro_tokens.npy")
    if not token_path.exists():
        sys.exit("Error: macro_tokens.npy not found. Run tokenize first.")

    tokens = np.load(token_path)
    Path("outputs/brain").mkdir(parents=True, exist_ok=True)
    train_on_tokens(
        tokens     = tokens,
        out_path   = "outputs/brain/best_brain_gpt.pt",
        steps      = args.steps,
        block_size = args.block,
        batch_size = args.batch,
        source     = "brain",
    )


def run_score(args):
    section("Score surprisal")
    from core.scorer import score_tokens, classify_surprisal, surprisal_summary
    token_path = Path("brain/data/macro_tokens.npy")
    ckpt_path  = Path("outputs/brain/best_brain_gpt.pt")
    for p in [token_path, ckpt_path]:
        if not p.exists():
            sys.exit(f"Error: {p} not found.")

    tokens    = np.load(token_path)
    surprisal = score_tokens(tokens, str(ckpt_path), device=get_device())
    labels    = classify_surprisal(surprisal)
    summary   = surprisal_summary(surprisal, labels)

    outdir = Path("outputs/brain")
    np.save(outdir / "brain_surprisal.npy", surprisal)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def run_analyze(args):
    section("Analyze + plot")
    from brain.analyze import main as _main
    extra = ["--data", "brain/data", "--outdir", "outputs/brain"]
    surp = Path("outputs/brain/brain_surprisal.npy")
    if surp.exists():
        extra += ["--surprisal", str(surp)]
    if args.events:
        extra += ["--events", args.events]
    sys.argv = ["brain.analyze"] + extra
    _main()


def run_q2(args):
    section("Q2 — State identity")
    sys.argv = ["brain.state_analysis",
                "--data", "brain/data",
                "--surprisal", "outputs/brain/brain_surprisal.npy",
                "--events", "brain/data/task_events.npy",
                "--outdir", "outputs/brain"]
    from brain.state_analysis import main as _main
    _main()


def run_q3(args):
    section("Q3 — Condition A vs B")
    # Run q3_state_occupancy.py directly
    import importlib.util, os
    script = Path("brain/q3_state_occupancy.py")
    if not script.exists():
        sys.exit(f"Error: {script} not found.")
    spec = importlib.util.spec_from_file_location("q3", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def run_q4(args):
    section("Q4 — Recovery time")
    import importlib.util
    script = Path("q4_fix2.py")
    if not script.exists():
        sys.exit("Error: q4_fix2.py not found in project root.")
    spec = importlib.util.spec_from_file_location("q4", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brain state trajectory pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=[
        "simulate", "tokenize", "train", "score",
        "analyze", "q2", "q3", "q4", "all"
    ])

    # Simulate options
    parser.add_argument("--nodes",    type=int,   default=12)
    parser.add_argument("--duration", type=int,   default=60000)
    parser.add_argument("--events",   default=None)
    parser.add_argument("--gain-a",   type=float, default=3.0)
    parser.add_argument("--gain-b",   type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)

    # Tokenize options
    parser.add_argument("--timeseries", default="brain/data/brain_timeseries.npy")
    parser.add_argument("--k",          type=int, default=10)
    parser.add_argument("--n-micro",    type=int, default=50)

    # Train options
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    stages = {
        "simulate": run_simulate,
        "tokenize": run_tokenize,
        "train":    run_train,
        "score":    run_score,
        "analyze":  run_analyze,
        "q2":       run_q2,
        "q3":       run_q3,
        "q4":       run_q4,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            if name == "simulate":
                continue  # simulate separately, it's slow
            fn(args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()