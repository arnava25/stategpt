"""
Q3 — Condition A vs B: state occupancy and transition analysis.
Run from unified-mind/: python3 q3_state_occupancy.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

tokens  = np.load("brain/data/macro_tokens.npy")
context = np.load("brain/data/task_context.npy")
surprisal = np.load("outputs/brain/brain_surprisal.npy")

N = len(tokens)
bin_size = len(context) // N
ctx_tok = np.zeros(N, dtype=int)
for i in range(N):
    chunk = context[i*bin_size:(i+1)*bin_size]
    vals, counts = np.unique(chunk, return_counts=True)
    ctx_tok[i] = vals[counts.argmax()]

n_states = int(tokens.max()) + 1
states   = list(range(n_states))
cond_names = {0: "Baseline", 1: "Condition A\n(high gain)", 2: "Condition B\n(low gain)"}
colors     = {0: "#888780", 1: "#c85252", 2: "#5260c8"}

# ── State occupancy per condition ─────────────────────────────────────────────
occ = {}
for cond in [0, 1, 2]:
    mask  = ctx_tok == cond
    total = mask.sum()
    dist  = np.zeros(n_states)
    if total > 0:
        for s in states:
            dist[s] = (tokens[mask] == s).sum() / total
    occ[cond] = dist

# Chi-squared test: A vs B state distribution
# Build count matrix for A and B
count_A = np.array([(tokens[ctx_tok==1] == s).sum() for s in states])
count_B = np.array([(tokens[ctx_tok==2] == s).sum() for s in states])
# Only include states with nonzero counts
mask_nonzero = (count_A + count_B) > 0
if mask_nonzero.sum() >= 2:
    chi2, p_chi2, dof, _ = chi2_contingency(
        np.stack([count_A[mask_nonzero], count_B[mask_nonzero]])
    )
else:
    chi2, p_chi2, dof = 0, 1, 0

print("Q3 — State occupancy analysis")
print(f"\nChi-squared test (A vs B state distribution):")
print(f"  chi2={chi2:.2f}  dof={dof}  p={p_chi2:.4f}  "
      f"{'*significant*' if p_chi2 < 0.05 else 'n.s.'}")

print(f"\nState occupancy:")
print(f"{'State':>6}  {'Baseline%':>10}  {'Cond A%':>8}  {'Cond B%':>8}  {'SurpOnEntry':>12}")
for s in states:
    surp_on_entry = surprisal[1:][tokens[1:] == s].mean() if (tokens[1:] == s).any() else 0
    print(f"  S{s:>2}    {occ[0][s]*100:>8.1f}%  {occ[1][s]*100:>7.1f}%  "
          f"{occ[2][s]*100:>7.1f}%  {surp_on_entry:>11.3f}")

# States exclusively visited in A
only_A = [s for s in states if occ[1][s] > 0 and occ[2][s] == 0 and occ[0][s] < 0.001]
print(f"\nStates only entered during Condition A: {['S'+str(s) for s in only_A]}")

# ── Surprisal during each condition ──────────────────────────────────────────
surp_by_cond = {}
for cond in [0, 1, 2]:
    mask = ctx_tok == cond
    surp_by_cond[cond] = surprisal[mask]

print(f"\nMean surprisal by condition:")
for cond, name in [(0,"Baseline"),(1,"Cond A"),(2,"Cond B")]:
    s = surp_by_cond[cond]
    print(f"  {name}: mean={s.mean():.5f}  max={s.max():.4f}  "
          f"nonzero={((s>0.01).sum())}/{len(s)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Q3 — How task conditions reshape brain state occupancy",
             fontsize=13, y=1.02)

# Panel 1: state occupancy bars grouped by condition
ax = axes[0]
x     = np.arange(n_states)
width = 0.28
for i, (cond, label) in enumerate([(0,"Baseline"),(1,"Cond A (high)"),(2,"Cond B (low)")]):
    ax.bar(x + (i-1)*width, occ[cond]*100, width,
           color=list(colors.values())[i], alpha=0.8, label=label)
ax.set_xticks(x)
ax.set_xticklabels([f"S{s}" for s in states])
ax.set_xlabel("Macro state")
ax.set_ylabel("% time in state")
ax.set_title("State occupancy by condition")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Panel 2: difference from baseline
ax = axes[1]
diff_A = (occ[1] - occ[0]) * 100
diff_B = (occ[2] - occ[0]) * 100
ax.bar(x - width/2, diff_A, width, color="#c85252", alpha=0.8, label="A − baseline")
ax.bar(x + width/2, diff_B, width, color="#5260c8", alpha=0.8, label="B − baseline")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"S{s}" for s in states])
ax.set_xlabel("Macro state")
ax.set_ylabel("Δ occupancy vs baseline (%)")
ax.set_title(f"Condition effect on state distribution\n"
             f"χ²={chi2:.1f}, p={p_chi2:.4f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: surprisal on entry per state, colored by whether A-exclusive
ax = axes[2]
surp_entry = []
for s in states:
    vals = surprisal[1:][tokens[1:] == s]
    surp_entry.append(vals.mean() if len(vals) > 0 else 0.0)

bar_colors = []
for s in states:
    if s in only_A:
        bar_colors.append("#c85252")
    elif occ[1][s] > occ[0][s] * 1.5:
        bar_colors.append("#E85D24")
    else:
        bar_colors.append("#888780")

ax.bar([f"S{s}" for s in states], surp_entry,
       color=bar_colors, alpha=0.85)
ax.set_xlabel("Macro state")
ax.set_ylabel("Mean surprisal on entry (nats)")
ax.set_title("Surprisal cost per state\n(red = A-exclusive or A-elevated)")
ax.grid(True, alpha=0.3, axis="y")

from matplotlib.patches import Patch
legend_els = [Patch(color="#c85252", label="A-exclusive state"),
              Patch(color="#E85D24", label="A-elevated state"),
              Patch(color="#888780", label="baseline state")]
ax.legend(handles=legend_els, fontsize=8)

plt.tight_layout()
out = Path("outputs/brain/q3_state_occupancy.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
