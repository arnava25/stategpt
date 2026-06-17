# StateGPT

A computational neuroscience framework that applies GPT-based surprisal scoring
to neural state sequences, implementing the core claim of predictive processing
theory: the brain is a prediction machine, and cognition is the process of
minimizing prediction error.

**[Live visualization](viz/index.html)** — open in browser, no setup needed.

---

## What it does

Takes any neural timeseries, discretizes it into macro-state tokens, trains a
small GPT to learn the baseline dynamics, and measures surprisal
(-log P(state | context)) at every timepoint. High surprisal = unexpected state
transition = prediction error.

```
neural timeseries → macro-state tokens → TinyGPT → surprisal signal
```

The same method runs on:
- Simulated Wilson-Cowan neural dynamics
- Real fMRI data (validated on ds000115, 99 subjects, schizophrenia + controls)

---

## Key findings

**Simulation (Wilson-Cowan, 60,000 steps, 36 task events):**

| Result | Value |
|---|---|
| Post-event surprisal Δ | +1.295 nats |
| Recovery time constant τ | 13.7 steps (exponential decay) |
| Condition A vs B state separation | χ²=58.4, p<0.0001 |
| Dominant attractor (S0) occupancy | 32% of time, mean dwell 104 steps |
| High-cost gateway state (S3) entry cost | 5.83 nats |

**Real fMRI (ds000115, working memory in schizophrenia):**

| Result | Value |
|---|---|
| Load-dependent surprisal increase | t=64.82, p<0.0001 (n=98) |
| 0-back to 2-back Δ | +0.050 nats per subject |
| Subjects preprocessed | 99 (SCZ, SCZ-SIB, CON-SIB, CON) |

---

## Structure

```
stategpt/
├── run.py                        single entry point
├── core/
│   ├── model.py                  TinyGPT architecture
│   ├── trainer.py                training loop
│   └── scorer.py                 GPT surprisal scoring
├── brain/
│   ├── simulate.py               Wilson-Cowan neural simulation
│   ├── build_tokens.py           timeseries to macro-state tokens
│   ├── analyze.py                surprisal timeline + event-locked plots
│   ├── state_analysis.py         state identity (Q2) + recovery time (Q4)
│   ├── q3_state_occupancy.py     condition A vs B (Q3)
│   ├── group_analysis.py         group-level fMRI analysis
│   ├── within_subject_analysis.py within-subject Markov analysis
│   ├── preprocess_ds000115.py    fMRI preprocessing for ds000115
│   └── preprocess_fmri.py        generic fMRI preprocessing
├── viz/
│   ├── index.html                interactive browser visualization
│   ├── export_demo_data.py       exports simulation data for viz
│   └── README.md
└── outputs/brain/                plots and results
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run simulation pipeline
python run.py simulate
python run.py tokenize
python run.py train --steps 3000
python run.py score
python run.py analyze --events brain/data/task_events.npy
python run.py q2
python run.py q3
python run.py q4

# Or all at once
python run.py all
```

For real fMRI data (ds000115 from OpenNeuro):

```bash
python -m openneuro download --dataset ds000115 --tag 00001 \
    --target-dir ~/data/ds000115 --max-concurrent-downloads 4

pip install nilearn nibabel

PYTHONPATH=. python brain/preprocess_ds000115.py --data ~/data/ds000115
PYTHONPATH=. python brain/group_analysis.py
PYTHONPATH=. python brain/within_subject_analysis.py
```

---

## Theoretical grounding

StateGPT implements predictive processing (Friston, Clark) computationally:

- **Macro-states** = discrete attractor states of neural dynamics
- **TinyGPT** = the brain's generative model of its own dynamics
- **Surprisal** = -log P(state | context) = prediction error
- **Task events** = perturbations that violate the generative model
- **Recovery τ** = how fast the system updates after prediction error

The schizophrenia application tests the aberrant salience hypothesis
(Kapur, 2003; Fletcher & Frith, 2009): that psychosis involves
dysregulated prediction error signaling.

---

## Limitations

- Individual-difference detection requires longer sequences (400+ timepoints).
  Task fMRI (137 timepoints) is insufficient for within-subject personalization.
  Resting-state fMRI is the appropriate next step.
- The GPT is small (~230k parameters) and trained on short sequences.
  Larger models and longer data would improve sensitivity.
- Group separation (SCZ vs CON) was not achieved in the current dataset.

---

## Citation

If you use this code, please cite the ds000115 dataset:

> Repovs G, Barch DM (2012). Working memory related brain network
> connectivity in individuals with schizophrenia and their siblings.
> Frontiers in Human Neuroscience.

---

## Author

Arnav Amit — independent researcher, UCLA Psychology '24, MD applicant.
Background in acute psychiatric care (PHP/IOP) and early psychosis research
(UCLA Aftercare, California OnTrack).