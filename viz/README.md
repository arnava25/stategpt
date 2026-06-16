# StateGPT Visualizer

Interactive browser visualization of brain state trajectories and GPT surprisal.

## What it shows

- **Surprisal timeline** — prediction error at each timestep, colored by macro-state
- **State sequence bar** — the brain's path through 10 macro-states over time
- **Task events** — red lines mark when task perturbations were applied
- **Cursor** — scrub or play through time, inspect any moment

## Usage

Open `index.html` directly in a browser. No server needed.

```bash
open viz/index.html
```

The default visualization uses generated demo data approximating the
Wilson-Cowan simulation results (12,000 steps, 10 macro-states, 36 events).

## Using real simulation data

After running the brain pipeline:

```bash
python run.py simulate
python run.py tokenize
python run.py train
python run.py score

# Export data for visualization
PYTHONPATH=. python viz/export_demo_data.py
```

This writes `viz/data/demo.json`. The visualizer will load it automatically
if served from a local server:

```bash
cd viz && python -m http.server 8000
# open http://localhost:8000
```

## Key findings shown

| Metric | Value |
|---|---|
| Task event surprisal Δ | +1.295 nats |
| Recovery time constant τ | 13.7 steps |
| Baseline surprisal | 0.134 nats |
| Dominant state S0 occupancy | 32% |
| S3 entry cost | 5.83 nats |
