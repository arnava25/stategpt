"""
brain/group_analysis.py

Runs the full unified-mind pipeline across all subjects in ds000115
and compares surprisal dynamics between diagnostic groups.

Scientific questions:
  1. Does mean surprisal differ across SCZ → SCZ-SIB → CON-SIB → CON?
     (Tests whether prediction error is elevated in schizophrenia)
  2. Does surprisal correlate with symptom severity (SAPS/SANS)?
     (Tests whether worse symptoms = more aberrant prediction error)
  3. Does surprisal scale with task load (0-back < 1-back < 2-back)
     and is this scaling disrupted in SCZ?
     (Tests whether SCZ shows abnormal load-dependent prediction error)
  4. Does state occupancy differ between groups?
     (Tests whether SCZ spends more time in rare high-cost states)

Usage:
    python brain/group_analysis.py
    python brain/group_analysis.py --subjects brain/data/subjects --task letter2backtask
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


TASKS   = ["letter0backtask", "letter1backtask", "letter2backtask"]
TASK_LABELS = {"letter0backtask": "0-back", "letter1backtask": "1-back",
               "letter2backtask": "2-back"}
GROUPS  = ["CON", "CON-SIB", "SCZ-SIB", "SCZ"]
COLORS  = {"CON": "#888780", "CON-SIB": "#5260c8",
           "SCZ-SIB": "#E85D24", "SCZ": "#c85252"}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def run_pipeline_on_subject(ts_path, ckpt_path, device, k=10, n_micro=50):
    """
    Run tokenize → score on a single subject timeseries.
    Returns (macro_tokens, surprisal) arrays.
    Uses cached tokens/surprisal if they exist.
    """
    from brain.build_tokens import discretize, build_macro_states
    from core.scorer import score_tokens

    ts_path    = Path(ts_path)
    tok_path   = ts_path.with_suffix("").with_suffix("") \
                        .parent / (ts_path.stem + "_macro_tokens.npy")
    surp_path  = ts_path.with_suffix("").with_suffix("") \
                        .parent / (ts_path.stem + "_surprisal.npy")

    # Load or compute macro tokens
    if tok_path.exists():
        macro = np.load(tok_path)
    else:
        ts    = np.load(ts_path)
        micro = discretize(ts, n_micro=n_micro)
        macro, _ = build_macro_states(micro, k=k)
        np.save(tok_path, macro)

    # Load or compute surprisal
    if surp_path.exists():
        surprisal = np.load(surp_path)
    else:
        surprisal = score_tokens(macro, str(ckpt_path),
                                 device=device, verbose=False)
        np.save(surp_path, surprisal)

    return macro, surprisal


# ── Per-subject metrics ───────────────────────────────────────────────────────

def compute_subject_metrics(macro, surprisal, n_states=10):
    """Extract key metrics from one subject's surprisal + token sequence."""
    nonzero = surprisal[surprisal > 0]
    base    = nonzero.mean() if len(nonzero) > 0 else 0.0

    # State occupancy — fraction of time in each state
    occ = np.zeros(n_states)
    for s in range(n_states):
        occ[s] = (macro == s).mean()

    # Dominant state fraction (how much time in the most common state)
    dominant_frac = occ.max()

    # State entropy — higher = more distributed across states
    occ_nonzero = occ[occ > 0]
    entropy = float(-np.sum(occ_nonzero * np.log(occ_nonzero + 1e-10)))

    # Transition rate — how often does the state change
    transitions = (macro[1:] != macro[:-1]).mean()

    return {
        "mean_surprisal":  float(surprisal.mean()),
        "std_surprisal":   float(surprisal.std()),
        "max_surprisal":   float(surprisal.max()),
        "baseline_surp":   float(base),
        "dominant_frac":   float(dominant_frac),
        "state_entropy":   float(entropy),
        "transition_rate": float(transitions),
    }


# ── Statistical tests ─────────────────────────────────────────────────────────

def group_comparison(data_by_group, metric):
    """One-way ANOVA + pairwise t-tests across groups."""
    groups = [g for g in GROUPS if g in data_by_group
              and len(data_by_group[g]) > 1]
    arrays = [np.array(data_by_group[g]) for g in groups]

    if len(arrays) < 2:
        return None

    f, p_anova = stats.f_oneway(*arrays)

    pairs = {}
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            t, p = stats.ttest_ind(data_by_group[g1], data_by_group[g2])
            pairs[f"{g1}_vs_{g2}"] = {"t": t, "p": p}

    return {"F": f, "p_anova": p_anova, "pairwise": pairs,
            "means": {g: np.mean(data_by_group[g]) for g in groups},
            "sems":  {g: stats.sem(data_by_group[g]) for g in groups}}


def symptom_correlation(subject_df, metric_col, symptom_cols):
    """Correlate a metric with SAPS/SANS scores in SCZ subjects."""
    scz = subject_df[subject_df["condit"] == "SCZ"].copy()
    results = {}
    for col in symptom_cols:
        valid = scz[[metric_col, col]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[metric_col], valid[col])
        results[col] = {"r": r, "p": p, "n": len(valid)}
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_group_comparison(results_by_task, outdir):
    """Q1: Mean surprisal by group, across task conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Q1 — Mean surprisal by diagnostic group and task load",
                 fontsize=13, y=1.02)

    for ax, task in zip(axes, TASKS):
        if task not in results_by_task:
            continue
        res = results_by_task[task]
        stats_res = res.get("group_stats_mean_surprisal")
        if stats_res is None:
            continue

        groups = [g for g in GROUPS if g in stats_res["means"]]
        means  = [stats_res["means"][g] for g in groups]
        sems   = [stats_res["sems"][g]  for g in groups]
        colors = [COLORS[g] for g in groups]

        ax.bar(groups, means, yerr=sems, color=colors, alpha=0.8,
               capsize=5, edgecolor="white")
        ax.set_title(f"{TASK_LABELS[task]}\n"
                     f"F={stats_res['F']:.2f}, p={stats_res['p_anova']:.3f}"
                     f"{'*' if stats_res['p_anova'] < 0.05 else ''}")
        ax.set_ylabel("Mean surprisal (nats)")
        ax.grid(True, alpha=0.3, axis="y")

        # Mark significant pairs
        pairs = stats_res["pairwise"]
        sig   = [(k, v) for k, v in pairs.items() if v["p"] < 0.05]
        if sig:
            y_max = max(means) + max(sems) * 1.5
            for i, (pair, pval) in enumerate(sig[:3]):
                g1, g2 = pair.split("_vs_")
                if g1 in groups and g2 in groups:
                    x1, x2 = groups.index(g1), groups.index(g2)
                    y = y_max + i * 0.05
                    ax.plot([x1, x2], [y, y], "k-", linewidth=0.8)
                    ax.text((x1+x2)/2, y+0.01,
                            f"p={pval['p']:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    out = outdir / "q1_group_surprisal.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_symptom_correlation(subject_df, outdir):
    """Q2: Surprisal vs symptom severity in SCZ subjects."""
    scz = subject_df[subject_df["condit"] == "SCZ"].copy()
    if len(scz) < 5:
        print("Not enough SCZ subjects for symptom correlation plot")
        return

    # Use 2-back (highest load) surprisal
    metric_col = "mean_surprisal_letter2backtask"
    if metric_col not in scz.columns:
        print(f"No {metric_col} column — skipping symptom correlation")
        return

    saps_total = scz[["saps7","saps20","saps25","saps34"]].sum(axis=1)
    sans_total = scz[["sans8","sans13","sans17","sans22","sans25"]].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Q2 — Surprisal vs symptom severity in schizophrenia",
                 fontsize=13, y=1.02)

    for ax, (scores, label) in zip(axes, [
        (saps_total, "SAPS total (positive symptoms)"),
        (sans_total, "SANS total (negative symptoms)"),
    ]):
        valid = pd.concat([scz[metric_col], scores], axis=1).dropna()
        if len(valid) < 4:
            continue
        x = valid.iloc[:, 1].values
        y = valid.iloc[:, 0].values
        r, p = stats.pearsonr(x, y)

        ax.scatter(x, y, color=COLORS["SCZ"], s=60, alpha=0.8)

        # Regression line
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m*xs + b, color="#c85252", linewidth=1.5, linestyle="--")

        ax.set_xlabel(label)
        ax.set_ylabel("Mean surprisal 2-back (nats)")
        ax.set_title(f"r={r:.3f}, p={p:.3f}{'*' if p < 0.05 else ''}\nn={len(valid)}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "q2_symptom_correlation.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_load_scaling(subject_df, outdir):
    """Q3: Surprisal scaling with task load by group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for group in GROUPS:
        gdf = subject_df[subject_df["condit"] == group]
        means, sems = [], []
        for task in TASKS:
            col = f"mean_surprisal_{task}"
            if col not in gdf.columns:
                continue
            vals = gdf[col].dropna().values
            if len(vals) == 0:
                continue
            means.append(vals.mean())
            sems.append(stats.sem(vals))

        if len(means) == len(TASKS):
            xs = np.arange(len(TASKS))
            ax.plot(xs, means, marker="o", linewidth=2,
                    color=COLORS[group], label=group)
            ax.fill_between(xs,
                            np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            alpha=0.15, color=COLORS[group])

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("Task condition (increasing load →)")
    ax.set_ylabel("Mean surprisal (nats)")
    ax.set_title("Q3 — Surprisal scaling with cognitive load by group")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "q3_load_scaling.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_state_occupancy(subject_df, outdir, n_states=10):
    """Q4: State occupancy profiles by group."""
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(16, 5), sharey=True)
    fig.suptitle("Q4 — Brain state occupancy by diagnostic group",
                 fontsize=13, y=1.02)

    # Use 2-back task
    task = "letter2backtask"
    occ_cols = [f"state_{s}_occ_{task}" for s in range(n_states)]

    for ax, group in zip(axes, GROUPS):
        gdf = subject_df[subject_df["condit"] == group]
        avail_cols = [c for c in occ_cols if c in gdf.columns]
        if not avail_cols:
            ax.set_title(f"{group}\n(no data)")
            continue

        mean_occ = gdf[avail_cols].mean().values
        states   = [f"S{s}" for s in range(len(avail_cols))]

        # Color bars by occupancy level
        bar_colors = plt.cm.RdYlBu_r(mean_occ / (mean_occ.max() + 1e-8))
        ax.bar(states, mean_occ * 100, color=bar_colors, alpha=0.85)
        ax.set_title(f"{group}\n(n={len(gdf)})")
        ax.set_xlabel("Macro state")
        if ax == axes[0]:
            ax.set_ylabel("Mean occupancy (%)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = outdir / "q4_state_occupancy.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="brain/data/subjects",
                        help="Dir with preprocessed timeseries + metadata.csv")
    parser.add_argument("--ckpt",     default="outputs/brain/best_brain_gpt.pt",
                        help="Trained GPT checkpoint")
    parser.add_argument("--outdir",   default="outputs/brain/group_analysis")
    parser.add_argument("--k",        type=int, default=10,
                        help="Number of macro states")
    parser.add_argument("--retrain",  action="store_true",
                        help="Retrain GPT on group-averaged timeseries first")
    args = parser.parse_args()

    outdir   = Path(args.outdir)
    sub_dir  = Path(args.subjects)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = sub_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found in {sub_dir}. "
            "Run preprocess_ds000115.py first."
        )
    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta)} subjects")
    print(f"Groups: {meta['condit'].value_counts().to_dict()}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Run 'python run.py train' first."
        )

    device = get_device()
    print(f"Device: {device}")

    # ── Process each subject ──────────────────────────────────────────────────
    subject_rows = []

    for _, row in meta.iterrows():
        sub_id = row["subject_id"]
        condit = row["condit"]
        print(f"\n{sub_id} ({condit})")

        sub_row = row.to_dict()

        for task in TASKS:
            ts_col = f"{task}_path"
            if ts_col not in row or pd.isna(row[ts_col]):
                print(f"  {task}: no timeseries — skipping")
                continue

            ts_path = Path(row[ts_col])
            if not ts_path.exists():
                print(f"  {task}: file not found — skipping")
                continue

            print(f"  {task}...", end=" ", flush=True)
            try:
                macro, surprisal = run_pipeline_on_subject(
                    ts_path, ckpt_path, device, k=args.k
                )
                metrics = compute_subject_metrics(macro, surprisal,
                                                  n_states=args.k)
                for key, val in metrics.items():
                    sub_row[f"{key}_{task}"] = val

                # State occupancy per state
                for s in range(args.k):
                    sub_row[f"state_{s}_occ_{task}"] = float((macro == s).mean())

                print(f"surp_mean={metrics['mean_surprisal']:.4f} ✓")
            except Exception as e:
                print(f"FAILED: {e}")

        subject_rows.append(sub_row)

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(outdir / "subject_metrics.csv", index=False)
    print(f"\nSaved subject metrics: {outdir}/subject_metrics.csv")

    # ── Statistical analysis ──────────────────────────────────────────────────
    results_by_task = {}

    print("\n" + "="*60)
    print("GROUP COMPARISON RESULTS")
    print("="*60)

    for task in TASKS:
        metric_col = f"mean_surprisal_{task}"
        if metric_col not in subject_df.columns:
            continue

        data_by_group = defaultdict(list)
        for _, row in subject_df.iterrows():
            if not pd.isna(row.get(metric_col)):
                data_by_group[row["condit"]].append(row[metric_col])

        stats_res = group_comparison(data_by_group, metric_col)
        results_by_task[task] = {"group_stats_mean_surprisal": stats_res}

        if stats_res:
            print(f"\n{TASK_LABELS[task]}:")
            print(f"  ANOVA: F={stats_res['F']:.2f}, p={stats_res['p_anova']:.4f}"
                  f"  {'*SIGNIFICANT*' if stats_res['p_anova'] < 0.05 else 'n.s.'}")
            for g in GROUPS:
                if g in stats_res["means"]:
                    print(f"  {g}: {stats_res['means'][g]:.4f} ± "
                          f"{stats_res['sems'][g]:.4f}")
            sig_pairs = [(k,v) for k,v in stats_res["pairwise"].items()
                         if v["p"] < 0.05]
            if sig_pairs:
                print(f"  Significant pairs: "
                      + ", ".join(f"{k} (p={v['p']:.3f})"
                                  for k,v in sig_pairs))

    # Symptom correlations
    print("\n" + "="*60)
    print("SYMPTOM CORRELATIONS (SCZ only, 2-back)")
    print("="*60)
    saps_cols = ["saps7", "saps20", "saps25", "saps34"]
    sans_cols = ["sans8", "sans13", "sans17", "sans22", "sans25"]
    metric_col = "mean_surprisal_letter2backtask"
    if metric_col in subject_df.columns:
        corrs = symptom_correlation(subject_df, metric_col,
                                    saps_cols + sans_cols)
        for col, res in corrs.items():
            sig = "*" if res["p"] < 0.05 else ""
            print(f"  {col}: r={res['r']:.3f}, p={res['p']:.3f}{sig}, n={res['n']}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_group_comparison(results_by_task, outdir)
    plot_symptom_correlation(subject_df, outdir)
    plot_load_scaling(subject_df, outdir)
    plot_state_occupancy(subject_df, outdir)

    print(f"\nAll outputs saved to {outdir}/")
    print("  q1_group_surprisal.png")
    print("  q2_symptom_correlation.png")
    print("  q3_load_scaling.png")
    print("  q4_state_occupancy.png")
    print("  subject_metrics.csv")


if __name__ == "__main__":
    main()
