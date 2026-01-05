# ACCA Reference Implementation (SNJNG + DeepPA/DeepPER + ITDE-CTRL)

This repository contains a **self-contained, end-to-end** Python reference implementation of the ACCA architecture described in **`Adaptive Cyber-Physical Systems Against Coordinated Attacks: Integrating Stochastic Neural Jump Node Graphs and Continuous-Time Reinforcement Learning`**. The code implements:

- **SNJNG** (Stochastic Neural Jump Node Graphs): a neural point-process model of *event-driven* attacker-network evolution.
- **DeepPA**: neural preferential attachment with entanglement/superposition and stochastic connections.
- **DeepPER**: neural percolation with a tunneling term and neural jump intensities.
- **CTRL** in **continuous time**: NODE-style latent dynamics with **ITDE** (integral temporal-difference error) to handle jumps.

> Practical note: The script included here is a *research-grade reference* aimed at clarity and faithful mechanisms, not a production simulator.

---

## Design philosophy

The implementation follows three principles that mirror the paper’s intent:

1. **Event-driven dynamics, not time-step heuristics.**  
   The attacker graph evolves through **stochastic jump events** (node/edge events), and the controller’s learning signal integrates rewards around those discontinuities using ITDE.

2. **Continuous-time RL with jump-aware value learning.**  
   The controller uses a NODE-style latent dynamics model between events, while jumps produce instantaneous state updates, and the critic is trained via **piecewise ITDE** to avoid discretization bias.

3. **Real-time scalability via linear-time surrogates.**  
   Classical PA/PER can be expensive at scale; the paper motivates neural/vectorized approximations and avoiding quadratic scans whenever possible.

---

## What’s in the codebase?

Depending on which version you use locally, filenames may differ. The latest “edge-restoration” variant produced in this workspace is:

- `CTRL_SNJNJ_v6_REVISED_RESTORE.py` — adds **stochastic edge restoration** to prevent degree collapse and to emulate the paper’s “stochastic connections” mechanism.

Other variants in this workspace:

- `CTRL_SNJNJ_v5.py` — CLI-arg version without the edge restoration pool.

---

## Simulation environment (paper-aligned, lightweight)

The environment is a **synthetic attacker-network** proxy intended to reproduce the *structural dynamics* in the paper, not a full SDN/EMANE/CORE testbed.

### Entities
- **Nodes** represent adversarial entities (e.g., compromised IoT/UAV/ground units).
- **Edges** represent communication/collaboration links.

### State (SNJNG → CTRL)
- A sparse GCN computes node embeddings **H** from graph structure and node features.
- A readout (mean + max pooling) yields a compact latent state **s(t)**.

### Actions (3D continuous)
The controller outputs a vector `a ∈ [0,1]^3` interpreted as:
1. `isolate`: isolate top-degree nodes (removes their incident edges)
2. `jam`: jamming strength (attenuates link quality and GCN edge weights)
3. `misinfo`: misinformation (adds noise to node features)

### Attacker events: node/edge jumps
Within each environment step (Δt), a small number of events are sampled:

- **Node event**: updates a chosen node’s degree by `Δk ∈ [-deltaK, +deltaK]`
- **Edge event**: flips an edge (dissolves an existing edge or creates a new one)

Edge creation is approximated by evaluating intensities over:
- existing edges, plus
- a capped set of **sampled non-edges** (`--nonedge_candidates`) to keep runtime linear.

### Reward shaping
The reward is an integral over the time window `[t, t+Δt]` and encourages the controller to:
- **degrade link quality** (RSSI),
- reduce hub dominance (PA proxy),
- increase fragility / percolation threshold (PER proxy).

---

## Stochastic edge restoration (why it exists)

In the paper, DeepPA introduces **stochastic connections** (Bernoulli presence of edges) to preserve volatility and to avoid brittle, deterministic topologies. In practice, if the defender aggressively removes edges (e.g., isolation) and if edge flips often dissolve edges, the graph can collapse toward zero degree unless **some edges reappear**.

The v6 “restore” variant therefore maintains a **removed-edge pool** and restores a random fraction of that pool **each step**:

- `--edge_restore_low 0.10 --edge_restore_high 0.20` restores ~10–20% of the removed pool per step (in expectation).
- Use `--edge_restore_max_per_step` to cap restores when the pool is very large.

This feature is meant to reproduce the “edges can disappear and reappear stochastically” behavior that is central to the volatility assumptions in the draft.

---

## Installation

### Python
Python 3.9+ recommended.

### Dependencies
- `numpy`
- `torch`
- `matplotlib` (optional, only needed for `--plot` / `--show_plots`)

Example:
```bash
pip install numpy torch matplotlib
```

---

## Quick start

### 1) Run unit tests
```bash
python CTRL_SNJNJ_v6.py --run_tests
```

### 2) Run a demo training loop
```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200
```

### 3) Demo + plots + CSV
```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200 \
  --plot --outdir runs/exp_demo --show_plots
```

Outputs in `runs/exp_demo/` typically include:
- `metrics.csv`
- `reward.png`, `nll.png`, `edges.png`, `rssi.png`, `zeta.png`, `gcc.png`, …
- edge restoration diagnostics (if enabled): `restored_edges.png`, `removed_pool.png`

---

## CLI reference (most-used knobs)

> Tip: always check the authoritative help for your local file:
> ```bash
> python CTRL_SNJNJ_v6.py --help
> ```

### Core run modes
- `--run_tests` : run built-in unit tests
- `--demo` : run a short training loop
- `--steps` : training steps for `--demo`
- `--eval_steps` : evaluation steps after training (policy=mode vs policy=random)
- `--plot` : save plots + `metrics.csv` after demo
- `--outdir` : output folder for plots/CSV
- `--show_plots` : display plots interactively

### Graph / model size
- `--N` : number of nodes
- `--F` : raw node feature dimension
- `--H` : GCN embedding dimension (state dimension becomes `2*H`)
- `--gcn_hidden` : hidden width of the GCN
- `--gcn_layers` : number of GCN layers (2–4 is usually a sensible range)

### Initial graph density
Choose **one**:
- `--mean_degree` : target mean degree; sets `p = mean_degree/(N-1)` (recommended for interpretability)
- `--p_init` : Erdos–Renyi edge probability (used only if `mean_degree <= 0`)

### Event / dynamics parameters
- `--deltaK` : max degree change per node jump
- `--xi_d` : DeepPER tunneling distance scale
- `--lambda0_edge` : global scale for edge-event intensity
- `--dt` : environment step size Δt
- `--rk4_steps` : RK4 sub-steps per environment step (ODE integration fidelity)
- `--discount_rate` : continuous-time discount γ in `exp(-γ t)`
- `--max_events_per_step` : cap events per Δt to bound runtime
- `--nonedge_candidates` : candidate non-edges sampled for potential edge creation

### Edge restoration (v6 restore variant)
- `--edge_restore_low` / `--edge_restore_high` : restoration fraction range per step
- `--edge_restore_max_per_step` : optional cap on restored edges per step

### Optimization / runtime
- `--lr` : learning rate
- `--grad_clip` : gradient clipping norm
- `--seed` : RNG seed
- `--device` : `cpu | cuda | mps`
- `--deepper_exact` : compute exact DeepPER diagnostic (small N only; slow)

---

## Experiment recipes

### A. Small correctness check (fast)
```bash
python CTRL_SNJNJ_v6.py --demo --steps 100 --eval_steps 50 \
  --N 50 --F 16 --H 16 --gcn_hidden 32 --gcn_layers 2 --mean_degree 8
```

### B. Dense small graph (stress event logic)
```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200 \
  --N 100 --mean_degree 30 --nonedge_candidates 400
```

### C. Edge restoration ablation (turn it off)
```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200 \
  --N 100 --mean_degree 30 \
  --edge_restore_low 0.0 --edge_restore_high 0.0
```

### D. Paper-aligned restoration rate (10–20%)
```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200 \
  --N 100 --mean_degree 30 \
  --edge_restore_low 0.10 --edge_restore_high 0.20
```

### E. Large-scale stress test (5k+ nodes)
This is the closest analogue to “realistic scale” behavior within this toy simulator.

```bash
python CTRL_SNJNJ_v6.py --demo --steps 300 --eval_steps 200 \
  --N 5000 --F 160 --H 160 --gcn_hidden 64 --gcn_layers 2 \
  --mean_degree 20 \
  --max_events_per_step 3 --nonedge_candidates 500 \
  --edge_restore_low 0.10 --edge_restore_high 0.20 \
  --device cuda
```

Notes:
- Start with **mean_degree 10–30**. Higher values grow `|E|` linearly and increase per-step cost.
- Keep `gcn_layers` small (2–4) to avoid oversmoothing and excessive compute.
- Do **not** enable `--deepper_exact` at this scale.

---

## Interpreting the demo logs

A typical print line looks like:
```
[t=  20] nll=... td_err=... Rint=... edges=... mean_deg=... restored=... pool=... zeta=... gcc=... events=...
```

Glossary (high-level):
- `nll` : SNJNG negative log-likelihood proxy (point-process likelihood terms)
- `td_err` : integral TD error proxy
- `Rint` : discounted reward integral over the step
- `edges`, `mean_deg` : current graph density
- `restored`, `pool` : edge restoration diagnostics (v6 restore)
- `zeta` : percolation threshold proxy (higher ≈ more fragile network)
- `gcc` : giant connected component fraction
- `events` : number of jump events within the step

---

## Paper → code map (conceptual)

The implementation follows the paper’s module decomposition:

- **SNJNG state construction**: GCN embeddings → state readout `s(t)`
- **CTRL action**: continuous action vector (isolation, jamming, misinformation)
- **Reward shaping**: combines link quality (RSSI) + PA + PER terms
- **ITDE**: integrates rewards and value updates across jump times (piecewise)
- **DeepPA**: node jump intensities + entanglement/superposition-inspired target sampling
- **DeepPER**: edge jump intensities using cosine similarity and a tunneling term

---

## Scaling & performance notes

This reference implementation is mostly **O(N + |E| + C)** per segment, where `C` is the number of non-edge candidates evaluated. Two practical considerations:

1. **Initialization cost**: naive Erdos–Renyi initialization uses nested loops over pairs.  
   For very large N, consider replacing the initializer with a faster sampler if startup time matters.

2. **Edge intensity evaluation** scales with `(existing_edges + nonedge_candidates)`.  
   For dense graphs, reduce `--nonedge_candidates` or reduce `--mean_degree`.

---

## Common troubleshooting

### “Edges go to ~0” / mean degree collapses
- Enable restoration: `--edge_restore_low 0.10 --edge_restore_high 0.20`.
- Reduce isolation aggressiveness (currently controlled by the learned policy; for debugging, you can clamp or fix actions).

### “Type accuracy is always ~0”
Event-prediction metrics are *proxies* based on sampled events and candidate sets. They are not supervised labels from a ground-truth dataset.

### macOS MPS quirks
If `--device mps` is unstable, fall back to `--device cpu` for debugging.

---

## Citation / reference

Primary reference: `Adaptive Cyber-Physical Systems Against Coordinated Attacks: Integrating Stochastic Neural Jump Node Graphs and Continuous-Time Reinforcement Learning ` (draft manuscript - paper under-review).
