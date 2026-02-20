# models

Public curated monorepo of biological simulation model packs and composed spaces for the **biosim** platform. Models are modular, composable components that can be wired together into full simulation scenarios without writing code — just YAML.

## What's Inside

### Models (20 packages)

Each model is a self-contained simulation component with a `model.yaml` manifest.

**Neuroscience** — spiking neural networks, synaptic dynamics, and neural monitoring:

| Model | Description |
|-------|-------------|
| `neuro-izhikevich-population` | Spiking neuron population (Regular Spiking, Fast Spiking presets) |
| `neuro-hodgkin-huxley-population` | Conductance-based Hodgkin-Huxley neuron population |
| `neuro-hodgkin-huxley-state-monitor` | Detailed HH state monitor (V, gates, ionic currents) |
| `neuro-exp-synapse-current` | Exponential-decay synapses with configurable connectivity |
| `neuro-step-current` | Constant/step current injection into neurons |
| `neuro-poisson-input` | Poisson-distributed spike train generator |
| `neuro-spike-monitor` | Spike raster visualization |
| `neuro-rate-monitor` | Firing rate computation and display |
| `neuro-state-monitor` | Neuron state variable tracking (membrane potential, etc.) |
| `neuro-spike-metrics` | Summary statistics from spike streams |

**Ecology** — population dynamics, environments, and ecosystem interactions:

| Model | Description |
|-------|-------------|
| `ecology-abiotic-environment` | Broadcasts environmental conditions (temperature, water, food, sunlight) |
| `ecology-organism-population` | Population dynamics with birth, death, and predation |
| `ecology-predator-prey-interaction` | Predation rates and functional response |
| `ecology-population-monitor` | Population size tracking over time |
| `ecology-phase-space-monitor` | Predator vs prey phase-space visualization |
| `ecology-population-metrics` | Ecosystem summary statistics |

**Virtual Cell** — gene regulatory networks, perturbations, and expression monitoring:

| Model | Description |
|-------|-------------|
| `virtualcell-perturbation-source` | Defines gene perturbations (knockout/overexpression) over time |
| `virtualcell-grn-predictor` | Classical GRN-based virtual cell producing expression profiles |
| `virtualcell-arc-state-predictor` | Arc Institute State Transition ML model for expression prediction |
| `virtualcell-expression-translator` | Translates expression profiles into neural input currents |
| `virtualcell-expression-monitor` | Visualizes gene expression fold-changes and timeseries |

### Spaces (6 composed simulations)

Spaces wire multiple models into runnable simulation scenarios.

| Space | Models | Description |
|-------|--------|-------------|
| `neuro-single-neuron` | 5 | Single Izhikevich neuron with step current, monitors, and metrics |
| `neuro-microcircuit` | 13 | Balanced E/I microcircuit: 40 excitatory + 10 inhibitory neurons, Poisson input, recurrent synaptic connectivity |
| `ecology-predator-prey` | 7 | Classic predator-prey dynamics with environment broadcast and monitors |
| `ecology-temperature-control` | 7 | Predator-prey ecosystem where environment temperature is an exposed parameter |
| `virtualcell-drug-neural-effect` | 8 | Virtual cell perturbation translated into a neural spiking response |

## Layout

```
models/
├── models/<model-slug>/     # One model package per folder, each with model.yaml
├── spaces/<space-slug>/     # Composed spaces with space.yaml
├── libs/                    # Shared helper code for curated models
├── templates/model-pack/    # Starter template for new model packs
├── scripts/                 # Manifest and entrypoint validation scripts
├── docs/                    # Governance documentation
└── .github/workflows/       # CI/CD pipeline
```

## How It Works

### Model Interface

Every model implements the `biosim.BioModule` interface:

- **`inputs()`** — declares named input signals the module consumes
- **`outputs()`** — declares named output signals the module produces
- **`advance_to(t)`** — advances the model's internal state to time `t`

Most curated models include Python source under `src/` and are wired together via `space.yaml` without additional code.

### Wiring

Spaces connect models by routing outputs to inputs in `space.yaml`:

```yaml
wiring:
  - from: current_source.current
    to: [neuron.input_current]
  - from: neuron.spikes
    to: [spike_monitor.spikes, rate_monitor.spikes]
```

No code changes needed to recombine models into new configurations.

### Running a Space

Spaces are loaded and executed by the `biosim-platform`. The platform reads `space.yaml`, instantiates models from their manifests, wires signals, and runs the simulation loop at the configured `tick_dt` timestep for the specified `duration`.

## Getting Started

### Prerequisites

- Python 3.11+
- `biosim` framework

### Install biosim

```bash
pip install "biosim @ git+https://github.com/BioSimulant/biosim.git@main"
```

### Create a New Model

1. Copy `templates/model-pack/` to `models/<your-model-slug>/`
2. Edit `model.yaml` with metadata, entrypoint, and pinned dependencies
3. Implement your module (subclass `biosim.BioModule` or use a built-in pack)
4. Validate: `python scripts/validate_manifests.py && python scripts/check_entrypoints.py`

### Create a New Space

1. Create `spaces/<your-space-slug>/space.yaml`
2. Reference models by `manifest_path` (e.g., `models/neuro-step-current/model.yaml`)
3. Define wiring between model outputs and inputs
4. Set `runtime.duration` and `runtime.tick_dt`

## Linking in biosim-platform

- Root manifests can be linked with `manifest_path=model.yaml` or `space.yaml`
- Subdirectory manifests require explicit paths:
  - `models/neuro-izhikevich-population/model.yaml`
  - `spaces/neuro-microcircuit/space.yaml`

## External Repos

External authors can keep models in independent repositories and link them directly in `biosim-platform`. This monorepo is curated, not exclusive.

## Validation & CI

Three scripts enforce repository integrity on every push:

| Script | Purpose |
|--------|---------|
| `scripts/validate_manifests.py` | Schema validation for all model.yaml and space.yaml files |
| `scripts/check_entrypoints.py` | Verifies Python entrypoints are importable and callable |
| `scripts/check_public_boundary.sh` | Prevents business-sensitive content in this public repo |

The CI pipeline (`.github/workflows/ci.yml`) runs: **secret scan** → **manifest validation** → **smoke sandbox** (Docker).

## Contributing

- All dependencies must use exact version pinning (`==`)
- Model slugs use kebab-case with domain prefix (`neuro-`, `ecology-`, `virtualcell-`)
- Custom modules must follow the `biosim.BioModule` interface
- Pre-commit hooks enforce trailing whitespace, EOF newlines, YAML syntax, and secret detection
- See [docs/PUBLIC_INTERNAL_BOUNDARY.md](docs/PUBLIC_INTERNAL_BOUNDARY.md) for content policy

## License

This repository is dual-licensed:

- **Code** (scripts, templates, Python modules): Apache-2.0 (`LICENSE-CODE.txt`)
- **Model/content** (manifests, docs, wiring/config): CC BY 4.0 (`LICENSE-CONTENT.txt`)

Attribution guidance: `ATTRIBUTION.md`
