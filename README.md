# models-virtualcell

Curated collection of **custom-built virtual cell** models for the **biosim** platform. This repository contains 5 hand-crafted Python implementations for in silico perturbation experiments, gene regulatory network (GRN) predictions, and ML-powered cell state transitions.

## What's Inside

### Models (5 packages)

Each model is a custom Python implementation designed for composable virtual cell experiments.

**Virtual Cell** — gene regulatory networks, perturbations, and expression monitoring:

| Model | Description |
|-------|-------------|
| `virtualcell-perturbation-source` | Defines gene perturbations (knockout/overexpression) over time |
| `virtualcell-grn-predictor` | Classical GRN-based virtual cell producing expression profiles |
| `virtualcell-arc-state-predictor` | Arc Institute State Transition ML model for expression prediction |
| `virtualcell-expression-translator` | Translates expression profiles into neural input currents |
| `virtualcell-expression-monitor` | Visualizes gene expression fold-changes and timeseries |

## How It Works

These are **native Python models**, not SBML imports. They implement the `biosim.BioModule` interface and are designed to be wired together for virtual cell experiments, enabling:
- Programmatic gene perturbations
- GRN-based or ML-based state predictions
- Cross-domain integration (e.g., gene expression → neural activity)

## Prerequisites
```bash
pip install "biosim @ git+https://github.com/BioSimulant/biosim.git@main"
```

## License
Dual-licensed: Apache-2.0 (code), CC BY 4.0 (content)
