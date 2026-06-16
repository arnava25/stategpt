"""
brain/within_subject_analysis.py  (Markov edition)

Within-subject design using a first-order Markov model instead of GPT.
Appropriate for short fMRI sequences (~137 timepoints).

For each subject:
  1. Build transition probability matrix from 0-back token sequence
  2. Score 1-back and 2-back using -log P(token_t | token_{t-1})
  3. Mean surprisal = measure of how unexpected the dynamics are
     given the subject's own 0-back baseline

Same conceptual logic as GPT surprisal, valid for short sequences.

Usage:
    PYTHONPATH=. python brain/within_subject_analysis.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


TASKS       = ["letter0backtask", "letter1backtask", "letter2backtask"]
TASK_LABELS = {"letter0backtask": "0-back",
               "letter1backtask": "1-back",
               "letter2backtask": "2-back"}
GROUPS  = ["CON", "CON-SIB", "SCZ-SIB", "SCZ"]
COLORS  = {"CON": "#888780", "CON-SIB": "#5260c8",
           "SCZ-SIB": "#E85D24", "SCZ": "#c85252"}


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize(ts_path, k=8):
    """Load timeseries and return macro-state token sequence."""
    from brain.build_tokens import discretize, build_macro_states

    # Check for cached tokens
    tok_path = Path(ts_path).parent / (Path(ts_path).stem + f"_k{k}_tokens.npy")
    if tok_path.exists():
        return np.load(tok_path), tok_path

    ts    = np.load(ts_path)
    micro = discretize(ts)
    macro, _ = build_macro_states(micro, k=k)
    np.save(tok_path, macro)
    return macro, tok_path


# ── Markov model ──────────────────────────────────────────────────────────────

def build_markov(tokens, k, alpha=0.5):
    """
    Build a first-order Markov transition matrix from a token sequence.
    Uses Laplace (add-alpha) smoothing so unseen transitions get small
    but nonzero probability.

    Returns:
        log_trans: (k, k) array of log transition probabilities
                   log_trans[i, j] = log P(j | i)
    """
    counts = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(tokens[:-1], tokens[1:]):
        if a < k and b < k:
            counts[a, b] += 1.0

    # Laplace smoothing
    counts += alpha
    row_sums = counts.sum(axis=1, keepdims=True)
    trans    = counts / row_sums
    log_trans = np.log(trans)
    return log_trans


def score_markov(tokens, log_trans, k):
    """
    Score a token sequence using a Markov model.
    Returns per-token surprisal array (first token = 0).
    Clamps to k-1 to handle out-of-vocab tokens from other conditions.
    """
    tokens    = np.clip(tokens, 0, k - 1).astype(int)
    surprisal = np.zeros(len(tokens), dtype=np.float32)
    for i in range(1, len(tokens)):
        surprisal[i] = float(-log_trans[tokens[i-1], tokens[i]])
    return surprisal


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="brain/data/subjects")
    parser.add_argument("--outdir",   default="outputs/brain/within_subject")
    parser.add_argument("--k",        type=int, default=8,
                        help="Number of macro states")
    parser.add_argument("--alpha",    type=float, default=0.5,
                        help="Laplace smoothing parameter")
    args = parser.parse_args()

    outdir  = Path(args.outdir)
    sub_dir = Path(args.subjects)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(sub_dir / "metadata.csv")
    print(f"Loaded {len(meta)} subjects")
    print(f"Groups: {meta['condit'].value_counts().to_dict()}")
    print(f"Macro states k={args.k}  Laplace alpha={args.alpha}\n")

    results = []

    for _, row in meta.iterrows():
        sub_id = row["subject_id"]
        condit = row["condit"]

        zero_col  = "letter0backtask_path"
        if zero_col not in row or pd.isna(row[zero_col]):
            continue
        zero_path = Path(row[zero_col])
        if not zero_path.exists():
            continue

        # Tokenize 0-back and build Markov model
        try:
            zero_tokens, _ = tokenize(zero_path, k=args.k)
            log_trans       = build_markov(zero_tokens, k=args.k,
                                           alpha=args.alpha)
        except Exception as e:
            print(f"{sub_id}: tokenize/markov failed: {e}")
            continue

        sub_row = {
            "subject_id": sub_id,
            "condit":     condit,
            "age":        row.get("age"),
            "gender":     row.get("gender"),
        }

        # Symptom scores
        for col in ["saps7","saps20","saps25","saps34",
                    "sans8","sans13","sans17","sans22","sans25"]:
            val = row.get(col)
            try:
                sub_row[col] = float(val) if pd.notna(val) else None
            except (ValueError, TypeError):
                sub_row[col] = None

        # Score all three conditions
        for task in TASKS:
            col = f"{task}_path"
            if col not in row or pd.isna(row[col]):
                continue
            ts_path = Path(row[col])
            if not ts_path.exists():
                continue
            try:
                tokens, _ = tokenize(ts_path, k=args.k)
                surp      = score_markov(tokens, log_trans, k=args.k)
                sub_row[f"surp_{task}"] = float(surp.mean())
            except Exception as e:
                print(f"  {sub_id} {task}: {e}")

        # Load effect
        s0 = sub_row.get("surp_letter0backtask")
        s1 = sub_row.get("surp_letter1backtask")
        s2 = sub_row.get("surp_letter2backtask")
        if all(x is not None for x in [s0, s1, s2]):
            sub_row["load_effect"]    = ((s1 + s2) / 2) - s0
            sub_row["load_effect_2b"] = s2 - s0
        else:
            sub_row["load_effect"]    = None
            sub_row["load_effect_2b"] = None

        results.append(sub_row)
        s0s = f"{s0:.3f}" if s0 else "—"
        s1s = f"{s1:.3f}" if s1 else "—"
        s2s = f"{s2:.3f}" if s2 else "—"
        print(f"{sub_id} ({condit})  0={s0s}  1={s1s}  2={s2s}")

    df = pd.DataFrame(results)
    df.to_csv(outdir / "within_subject_results.csv", index=False)
    print(f"\nSaved: {outdir}/within_subject_results.csv")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("WITHIN-SUBJECT MARKOV SURPRISAL RESULTS")
    print("="*60)

    valid = df.dropna(subset=["surp_letter0backtask",
                               "surp_letter1backtask",
                               "surp_letter2backtask"])
    s0 = valid["surp_letter0backtask"].astype(float).values
    s1 = valid["surp_letter1backtask"].astype(float).values
    s2 = valid["surp_letter2backtask"].astype(float).values

    t01, p01 = stats.ttest_rel(s1, s0)
    t02, p02 = stats.ttest_rel(s2, s0)
    t12, p12 = stats.ttest_rel(s2, s1)

    print(f"\n1. Load effect (all subjects, n={len(valid)}):")
    print(f"   0→1-back: Δ={s1.mean()-s0.mean():+.4f}  "
          f"t={t01:.2f}  p={p01:.4f}  {'*' if p01<0.05 else 'n.s.'}")
    print(f"   0→2-back: Δ={s2.mean()-s0.mean():+.4f}  "
          f"t={t02:.2f}  p={p02:.4f}  {'*' if p02<0.05 else 'n.s.'}")
    print(f"   1→2-back: Δ={s2.mean()-s1.mean():+.4f}  "
          f"t={t12:.2f}  p={p12:.4f}  {'*' if p12<0.05 else 'n.s.'}")

    group_effects = {}
    print(f"\n2. Load effect by group (0→2-back Δ):")
    for g in GROUPS:
        gdf = df[df["condit"] == g].dropna(subset=["load_effect_2b"])
        if len(gdf) < 3:
            continue
        eff = gdf["load_effect_2b"].astype(float).values
        group_effects[g] = eff
        print(f"   {g} (n={len(eff)}): "
              f"Δ={eff.mean():+.4f} ± {stats.sem(eff):.4f}")

    if len(group_effects) >= 2:
        f, p_anova = stats.f_oneway(*group_effects.values())
        print(f"   ANOVA: F={f:.2f}  p={p_anova:.4f}  "
              f"{'*' if p_anova<0.05 else 'n.s.'}")
        if "SCZ" in group_effects and "CON" in group_effects:
            t, p = stats.ttest_ind(group_effects["SCZ"],
                                   group_effects["CON"])
            d = ((group_effects["SCZ"].mean() - group_effects["CON"].mean()) /
                 np.sqrt((group_effects["SCZ"].std()**2 +
                          group_effects["CON"].std()**2) / 2))
            print(f"   SCZ vs CON: t={t:.2f}  p={p:.4f}  "
                  f"d={d:.3f}  {'*' if p<0.05 else 'n.s.'}")

    print(f"\n3. Symptom correlations (SCZ, load effect 0→2-back):")
    scz = df[df["condit"] == "SCZ"].dropna(subset=["load_effect_2b"])
    for col in ["saps7","saps20","saps25","saps34",
                "sans8","sans13","sans17","sans22","sans25"]:
        v = scz[["load_effect_2b", col]].copy()
        v[col] = pd.to_numeric(v[col], errors="coerce")
        v = v.dropna()
        if len(v) < 4:
            continue
        r, p = stats.pearsonr(v["load_effect_2b"].astype(float),
                              v[col].astype(float))
        print(f"   {col}: r={r:.3f}  p={p:.3f}  n={len(v)}"
              f"  {'*' if p<0.05 else ''}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Within-subject surprisal (Markov model, personal baseline)",
                 fontsize=13, y=1.02)

    # Load scaling curves
    ax = axes[0]
    for g in GROUPS:
        gdf = df[df["condit"] == g].dropna(
            subset=[f"surp_{t}" for t in TASKS])
        if len(gdf) < 2:
            continue
        means = [gdf[f"surp_{t}"].astype(float).mean() for t in TASKS]
        sems  = [stats.sem(gdf[f"surp_{t}"].astype(float)) for t in TASKS]
        ax.plot(range(3), means, marker="o", linewidth=2,
                color=COLORS[g], label=f"{g} (n={len(gdf)})")
        ax.fill_between(range(3),
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.15, color=COLORS[g])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0-back", "1-back", "2-back"])
    ax.set_xlabel("Task condition")
    ax.set_ylabel("Mean surprisal vs own 0-back (nats)")
    ax.set_title("Surprisal by cognitive load")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Load effect by group
    ax = axes[1]
    grps   = [g for g in GROUPS if g in group_effects]
    means  = [group_effects[g].mean() for g in grps]
    sems   = [stats.sem(group_effects[g]) for g in grps]
    colors = [COLORS[g] for g in grps]
    ax.bar(grps, means, yerr=sems, color=colors,
           alpha=0.8, capsize=5, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ surprisal (2-back − 0-back, nats)")
    ax.set_title("Load-induced surprisal by group\n"
                 "(positive = more surprising under high load)")
    ax.grid(True, alpha=0.3, axis="y")

    # Mark SCZ vs CON significance
    if "SCZ" in group_effects and "CON" in group_effects:
        _, p = stats.ttest_ind(group_effects["SCZ"], group_effects["CON"])
        if p < 0.1:
            y = max(means) + max(sems) * 2
            x1, x2 = grps.index("CON"), grps.index("SCZ")
            ax.plot([x1, x2], [y, y], "k-", linewidth=0.8)
            ax.text((x1+x2)/2, y + abs(y)*0.05,
                    f"p={p:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = outdir / "within_subject_load_effect.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # Symptom scatter
    scz_plot = df[df["condit"] == "SCZ"].dropna(subset=["load_effect_2b"])
    if len(scz_plot) >= 5:
        saps = scz_plot[["saps7","saps20","saps25","saps34"]].apply(
            pd.to_numeric, errors="coerce").sum(axis=1)
        sans = scz_plot[["sans8","sans13","sans17","sans22","sans25"]].apply(
            pd.to_numeric, errors="coerce").sum(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("SCZ: load-induced surprisal vs symptom severity",
                     fontsize=13, y=1.02)
        for ax, (scores, label) in zip(axes, [
            (saps, "SAPS total (positive symptoms)"),
            (sans, "SANS total (negative symptoms)"),
        ]):
            v = pd.concat([scz_plot["load_effect_2b"].reset_index(drop=True),
                           scores.reset_index(drop=True)], axis=1).dropna()
            if len(v) < 4:
                continue
            x = v.iloc[:,1].astype(float).values
            y = v.iloc[:,0].astype(float).values
            r, p = stats.pearsonr(x, y)
            ax.scatter(x, y, color=COLORS["SCZ"], s=60, alpha=0.8)
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m*xs+b, color="#c85252",
                    linewidth=1.5, linestyle="--")
            ax.set_xlabel(label)
            ax.set_ylabel("Load effect Δ (nats)")
            ax.set_title(f"r={r:.3f}, p={p:.3f}{'*' if p<0.05 else ''}"
                         f"\nn={len(v)}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out2 = outdir / "within_subject_symptom_correlation.png"
        plt.savefig(out2, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out2}")

    print(f"\nDone. All outputs in {outdir}/")


if __name__ == "__main__":
    main()