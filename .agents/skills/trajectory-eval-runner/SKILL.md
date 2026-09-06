---
name: trajectory-eval-runner
description: Automated runbook for executing the 443-segment GNSS blackout evaluation, generating drift CDFs, boxplots, and trajectory overlay plots for SIH reporting.
---

# Trajectory Evaluation & Benchmark Runner Skill

## Purpose
Enforces reproducible statistical validation across all IO-VNBD drives (M, S, Vf, Vta, Vtb, Vw, Y1) under multi-length GNSS outages (100m–1500m).

## Workflow

### 1. Run Complete Blackout Benchmark
```bash
python -m idr.eval.scenarios
```
- Iterates over all 7 IO-VNBD synchronized drives.
- Tests dead reckoning across:
  - 100m, 250m, 500m, 1000m, 1500m blackout segments.
  - Straight motorway, roundabout, curved urban road regimes.

### 2. Compute Navigation Metrics
- **Final Drift %**: $\text{Final Position Error} / \text{Outage Distance} \times 100\%$.
- **CEP50 & CEP95**: Circular Error Probable (50th & 95th percentiles).
- **Maximum Lateral Deviation**: Peak error orthogonal to true road centerline.

### 3. Generate Official Figures
```bash
python -m idr.eval.plotting
```
Generates figures in `reports/figures/`:
- `drift_cdf_comparison.png`
- `scenario_boxplots.png`
- `reacquisition_smoothing.png`
- `velocity_estimate.png`
- `mapmatch_nhc_overlay.png`
