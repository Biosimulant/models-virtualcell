# Model & Space Standards

Acceptance criteria and conventions for every model and space in this repository.
New contributions **must** satisfy all items marked **REQUIRED**. Items marked
**RECOMMENDED** are strongly encouraged and will be enforced over time.

---

## Table of Contents

1. [Model Standards](#model-standards)
   - [Directory Layout](#directory-layout)
   - [Manifest (`model.yaml`)](#manifest-modelyaml)
   - [Source Code](#source-code)
   - [BioModule Interface Contract](#biomodule-interface-contract)
   - [Configuration & Parameters](#configuration--parameters)
   - [Visualization](#visualization)
   - [Tests](#tests)
   - [Naming Conventions](#naming-conventions)
2. [Space Standards](#space-standards)
   - [Directory Layout](#space-directory-layout)
   - [Manifest (`space.yaml`)](#manifest-spaceyaml)
   - [Wiring](#wiring)
   - [Runner Scripts](#runner-scripts)
   - [Space Tests](#space-tests)
   - [Space Naming Conventions](#space-naming-conventions)
3. [Acceptance Checklist](#acceptance-checklist)

---

## Model Standards

### Directory Layout

**REQUIRED** — every model directory must contain at minimum:

```
models/<domain>-<subject>-<role>/
├── model.yaml            # Manifest (metadata + entrypoint)
├── src/
│   └── <module>.py       # Single Python module with the BioModule class
└── tests/
    └── test_<module>.py  # Unit tests for the module
```

| Path | Required | Purpose |
|------|----------|---------|
| `model.yaml` | REQUIRED | Manifest declaring metadata, entrypoint, tags, and dependencies |
| `src/<module>.py` | REQUIRED | Python module containing the main `BioModule` subclass |
| `tests/test_<module>.py` | REQUIRED | pytest test file exercising the module |

---

### Manifest (`model.yaml`)

**REQUIRED** fields:

```yaml
schema_version: "2.0"
title: "<Domain>: <ComponentName>"      # e.g. "Neuro: SpikeMonitor"
description: "<One-line feature description>"
standard: other
tags: [<domain>, <subdomain>, ...]      # At least one domain tag
authors: ["Biosimulant Team"]
bsim:
  entrypoint: "src.<module>:<ClassName>"
```

| Field | Rule |
|-------|------|
| `schema_version` | Must be `"2.0"` |
| `title` | Format: `"<Domain>: <PascalCaseName>"` |
| `description` | Single sentence, starts with a verb or noun |
| `standard` | Must be `"other"` (reserved for future standards) |
| `tags` | List of lowercase strings; first tag must be the domain (`neuroscience`, `ecology`, `brain`, etc.) |
| `authors` | List of strings |
| `bsim.entrypoint` | Format `src.<snake_case_module>:<PascalCaseClass>` — must be importable and callable |

**REQUIRED** — if the model has external dependencies:

```yaml
runtime:
  dependencies:
    packages:
      - numpy==1.26.0       # Exact pin with == only
      - scipy==1.11.0
```

All dependency versions **must** be pinned with `==`. No `>=`, `~=`, or unpinned
specifiers are allowed. This is enforced by CI (`scripts/validate_manifests.py`).

---

### Source Code

**REQUIRED** — every `.py` file in `src/` must include:

1. **SPDX license header** (first two lines):
   ```python
   # SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
   #
   # SPDX-License-Identifier: MIT
   ```

2. **Module docstring** describing the model's purpose.

3. **Future annotations import**:
   ```python
   from __future__ import annotations
   ```

4. **Type annotations** on all public method signatures (parameters and return types).

5. **TYPE_CHECKING guard** for type-only imports to avoid circular dependencies:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from bsim import BioWorld
   ```

---

### BioModule Interface Contract

Every model class **must** inherit from `bsim.BioModule` and implement the
following interface:

#### Required Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `(self, ..., min_dt: float = <default>) -> None` | Constructor. Must set `self.min_dt`. All domain parameters must have sensible defaults. `min_dt` should be the last parameter. |
| `inputs` | `(self) -> Set[str]` | Return the set of input signal names this module accepts. Return `set()` if the module has no inputs. |
| `outputs` | `(self) -> Set[str]` | Return the set of output signal names this module produces. Must not be empty — every model must produce at least one output. |
| `advance_to` | `(self, t: float) -> None` | Advance internal state to simulation time `t`. Must update `self._outputs`. |
| `get_outputs` | `(self) -> Dict[str, BioSignal]` | Return a fresh dict of current output signals. Keys must match the set returned by `outputs()`. |

#### Conditionally Required Methods

| Method | Signature | When Required |
|--------|-----------|---------------|
| `set_inputs` | `(self, signals: Dict[str, BioSignal]) -> None` | **REQUIRED** if `inputs()` returns a non-empty set. Must gracefully handle missing keys. |

#### Optional Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `reset` | `(self) -> None` | Reset module to its initial state. **RECOMMENDED** for testability and replay. |
| `visualize` | `(self) -> Optional[Dict[str, Any]]` | Return a visualization spec. Return `None` when there is no data to display yet. |

#### Signal Construction

All outputs must be `BioSignal` instances:

```python
from bsim.signals import BioSignal, SignalMetadata

BioSignal(
    source=<module_alias>,     # Use getattr(self, "_world_name", self.__class__.__name__)
    name=<signal_name>,        # Must match the key in outputs()
    value=<payload>,           # Domain-specific data (float, dict, list)
    time=t,                    # Current simulation time
    metadata=SignalMetadata(   # RECOMMENDED
        units="mV",
        description="Membrane potential",
        kind="state",          # "state" | "event" | "metric"
    ),
)
```

**RECOMMENDED** — include `SignalMetadata` with `units`, `description`, and `kind`
on every output signal. This enables downstream tools to label axes and validate
wiring compatibility.

---

### Configuration & Parameters

**REQUIRED**:
- All constructor parameters must have sensible defaults so the model can be
  instantiated with zero arguments (except `min_dt` which always has a default).
- `min_dt` must reflect the biological timescale of the model:
  - Fast dynamics (e.g., Hodgkin-Huxley): `0.0001` (0.1 ms)
  - Moderate dynamics (e.g., Izhikevich): `0.001` (1 ms)
  - Slow dynamics (e.g., ecology populations): `1.0`

**RECOMMENDED**:
- Use `@dataclass` presets for named parameter bundles (e.g., `IzhikevichPreset`,
  `SpeciesPreset`). Expose a `preset` string parameter that maps to the dataclass.
- Store presets in a module-level `dict` (e.g., `PRESETS = {"RS": PRESET_RS, ...}`).

---

### Visualization

**RECOMMENDED** — implement `visualize()` returning one of the standard render types:

| Render Type | Data Structure | Use Case |
|-------------|---------------|----------|
| `timeseries` | `{"series": [{"name": str, "points": [[t, v], ...]}], "title": str}` | Quantities that evolve over time |
| `image` | `{"src": "data:image/svg+xml;...", "alt": str, "width": int, "height": int}` | Raster plots, spatial layouts |
| `table` | `{"rows": [[...], ...], "columns": [str, ...]}` | Summary metrics, parameter tables |
| `heatmap` | `{"data": [[...]], "x_labels": [...], "y_labels": [...], "colorscale": str}` | Matrix data |

Rules:
- Return `None` before the first `advance_to` call (no data to show yet).
- Keep history buffers bounded (use `max_points` / `max_neurons` parameters).

---

### Tests

**REQUIRED** — every model must have a `tests/test_<module>.py` file with at
least the following test cases:

#### 1. Instantiation Test

Verify the module can be created with default parameters:

```python
def test_instantiation():
    module = MyModule()
    assert module.min_dt > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0
```

#### 2. Advance & Output Test

Verify `advance_to` produces valid outputs:

```python
def test_advance_produces_outputs():
    module = MyModule(min_dt=0.1)
    module.advance_to(0.1)
    outputs = module.get_outputs()
    for name in module.outputs():
        assert name in outputs
        signal = outputs[name]
        assert signal.source is not None
        assert signal.time == 0.1
```

#### 3. Reset Test (if `reset()` is implemented)

Verify state is fully cleared:

```python
def test_reset():
    module = MyModule(min_dt=0.1)
    module.advance_to(0.1)
    module.advance_to(0.2)
    module.reset()
    module.advance_to(0.1)
    outputs = module.get_outputs()
    # Assert outputs match a fresh module at t=0.1
```

#### 4. Input Handling Test (if `inputs()` is non-empty)

Verify `set_inputs` processes signals correctly:

```python
def test_set_inputs():
    from bsim.signals import BioSignal
    module = MyModule(min_dt=0.1)
    module.set_inputs({
        "input_name": BioSignal(source="test", name="input_name", value=1.0, time=0.1)
    })
    module.advance_to(0.1)
    outputs = module.get_outputs()
    # Assert outputs reflect the injected input
```

#### 5. Visualization Test (if `visualize()` is implemented)

```python
def test_visualize_none_before_advance():
    module = MyModule(min_dt=0.1)
    assert module.visualize() is None

def test_visualize_after_advance():
    module = MyModule(min_dt=0.1)
    module.advance_to(0.1)
    vis = module.visualize()
    assert vis is not None
    assert vis["render"] in ("timeseries", "image", "table", "heatmap")
```

#### 6. Output Signal Consistency Test

Verify that `get_outputs()` keys always match `outputs()`:

```python
def test_output_keys_match():
    module = MyModule(min_dt=0.1)
    module.advance_to(0.1)
    assert set(module.get_outputs().keys()) == module.outputs()
```

**Running tests**:
```bash
# Single model
pytest models/<model-slug>/tests/

# All models
pytest models/models/
```

---

### Naming Conventions

| Element | Pattern | Example |
|---------|---------|---------|
| Model directory (slug) | Native/handcrafted: `<domain>-<subject>-<role>`; Public-catalog faithful: `<science_domain>-<format>-<subject>-<catalog_token>-<role>` (kebab-case) | `neuro-hodgkin-huxley-population`; `epidemiology-sbml-ghanbari2020-second-wave-covid-iran-biomd0000000976-model` |
| Manifest title | `"<Domain>: <PascalCaseName>"` | `"Neuro: HodgkinHuxleyPopulation"` |
| Python module file | `<snake_case>.py` | `hodgkin_huxley_population.py` |
| Python class name | `PascalCase` | `HodgkinHuxleyPopulation` |
| Tags | lowercase, no spaces | `neuroscience`, `monitor`, `ecology` |
| Signal names | `snake_case` | `population_state`, `spikes` |

#### Slug Structure (Native/Handcrafted): `<domain>-<subject>-<role>`

Every model slug has three parts:

| Part | What it is | Examples |
|------|-----------|---------|
| **domain** | Domain prefix (see table below) | `neuro-`, `ecology-`, `virtualcell-` (legacy), `epidemiology-` |
| **subject** | The specific model, algorithm, or biological entity being simulated. Always a **noun or noun phrase**. | `hodgkin-huxley`, `izhikevich`, `lotka-volterra`, `poisson`, `retina`, `abiotic` |
| **role** | What the module does in the simulation. Always a **noun**. | `population`, `monitor`, `input`, `source`, `interaction`, `encoder`, `relay` |

#### Slug Structure (Public-Catalog Faithful): `<domain>-<format>-<subject>-<catalog_token>-<role>`

For models that are faithful wrappers of upstream artifacts (SBML/CellML/NeuroML, etc.) and include `upstream:` metadata in
`model.yaml`, the slug must make the upstream format visible as the 2nd segment:

| Part | What it is | Examples |
|------|-----------|---------|
| **science_domain** | Science-domain prefix (see table below) | `signaling-`, `epidemiology-`, `neuroscience-` |
| **format** | Upstream format token | `sbml`, `cellml`, `neuroml`, `other` |
| **subject** | Short descriptive subject (noun phrase); may be multi-segment | `goldbeter2013-cdk-oscillations`, `noble1962-cardiac-ap` |
| **catalog_token** | Upstream catalog `source_id` normalized (lowercase; strip non-alphanumerics) | `biomd0000000944`, `model2003190005` |
| **role** | What the module does in the simulation (noun) | `model`, `network`, `population` |

Examples:
- Public SBML (epidemiology): `epidemiology-sbml-ghanbari2020-forecasting-the-second-wave-of-covid-19-in-iran-biomd0000000976-model`
- Public SBML (signaling): `signaling-sbml-radulescu2008-nfkb-hierarchy-...-model7743212613-model`
- Public CellML (cardiovascular): `cardiovascular-cellml-noble1962-cardiac-ap-<cellml_id>-model`
- Legacy native: `neuro-hodgkin-huxley-population` (still valid)

**Grandfathered legacy slugs:** existing directories already in `models/models/` may keep older prefixes and the native 3-part form.
New public-catalog models should use science-domain + format-visible slugs.

The subject is the most important part — it is what makes a slug unique and
self-describing. A slug without a clear subject (e.g., `neuro-metrics`,
`ecology-environment`) is ambiguous and will collide when a second
implementation of the same concept is added.

**REQUIRED** — slug naming rules:

1. **Use nouns, not verbs.** The slug names what the model *is*, not what it
   *does*. Use `neuro-poisson-input` (noun: Poisson input source), not
   `neuro-generate-spikes`.

2. **Include the model or algorithm name when one exists.** If the module
   implements a named model (Hodgkin-Huxley, Izhikevich, Lotka-Volterra,
   Poisson, Ricker, Kuramoto), that name must appear in the slug.

3. **Be specific enough to be globally unique.** The slug must distinguish this
   model from any other model that could plausibly fill a similar role. Ask:
   *"If someone contributed a second model with the same role, would the slug
   still be unambiguous?"*

4. **Disambiguate by subject first, then by author/institution if needed.**
   - Same role, different algorithms: `neuro-hodgkin-huxley-population` vs `neuro-izhikevich-population`
   - Same algorithm, different implementations: add the author, institution, or variant — e.g., `neuro-allen-hh-population` vs `neuro-markram-hh-population`
   - Same algorithm, different configurations: add a qualifier — e.g., `neuro-hodgkin-huxley-fast-population` vs `neuro-hodgkin-huxley-detailed-population`

5. **Abbreviations.** Well-known abbreviations within the domain are allowed
   (HH for Hodgkin-Huxley, LGN for Lateral Geniculate Nucleus, GRN for Gene
   Regulatory Network) but must be used consistently across the entire repo.
   When in doubt, spell it out.

6. **Keep it concise but not cryptic.** Three to four segments (domain + 2-3
   words) is typical. Avoid going beyond five segments unless clarity demands it.

#### Good and Bad Examples

| Slug | Verdict | Why |
|------|---------|-----|
| `neuro-hodgkin-huxley-population` | Good | Named algorithm + clear role |
| `neuro-izhikevich-population` | Good | Named algorithm + clear role |
| `neuro-poisson-input` | Good | Named distribution + clear role |
| `ecology-predator-prey-interaction` | Good | Specific subject + role |
| `ecology-phase-space-monitor` | Good | Specific visualization type + role |
| `neuro-metrics` | Bad | Missing subject — *which* metrics? For what? |
| `ecology-environment` | Bad | Too generic — *what kind* of environment? |
| `neuro-monitor` | Bad | Missing subject — could be anything |
| `ecology-model` | Bad | Says nothing useful |
| `neuro-generate-spikes` | Bad | Verb phrase, not a noun |

#### Science Domain Prefixes

For **new** models, the first slug segment should be a **science domain** (single token, no hyphens).

Public-catalog faithful slugs use:

- `<science_domain>-<format>-<subject>-<catalog_token>-<role>`

Example (SBML, epidemiology):

- `epidemiology-sbml-ghanbari2020-forecasting-the-second-wave-of-covid-19-in-iran-biomd0000000976-model`

| Prefix | Domain |
|--------|--------|
| `epidemiology-` | Infectious disease dynamics (SIR/SEIR, outbreaks, transmission) |
| `immunology-` | Immune response (T-cells, cytokines, antibodies, vaccines) |
| `oncology-` | Cancer and tumor dynamics |
| `pharmacology-` | PK/PD, dosing, drug response |
| `metabolism-` | Metabolic networks, genome-scale metabolism, FBA |
| `signaling-` | Signaling pathways (MAPK, NFkB, receptors, phosphorylation) |
| `generegulation-` | Gene regulation, transcription, translation, GRNs |
| `cellcycle-` | Cell cycle checkpoints, cyclins/CDKs, mitosis |
| `cardiovascular-` | Heart and circulation, cardiac electrophysiology |
| `neuroscience-` | Neurons, synapses, spikes, brain regions |
| `microbiology-` | Microbes (bacteria/yeast/pathogens; non-immune focus) |
| `physiology-` | Organ/system homeostasis not covered above |
| `biomechanics-` | Mechanics of tissues/systems (force, stress/strain) |
| `development-` | Developmental biology (morphogens, patterning, differentiation) |
| `ecology-` | Ecosystems and population interactions |
| `systemsbiology-` | Fallback bucket for molecular/cellular models without a clearer category |

**Legacy slugs:** existing `models/models/` directories may keep older prefixes (e.g. `neuro-`, `virtualcell-`, `example-`) without renaming.

New domains can be added as the repository grows. Propose a prefix in your PR
description and get approval before merging.

---

## Space Standards

### Space Directory Layout

**REQUIRED** — every space directory must contain at minimum:

```
spaces/<domain>-<scenario>/
├── space.yaml             # Composition manifest (models + wiring + runtime)
└── wiring.yaml            # Local wiring spec (module classes + connections)
```

**RECOMMENDED** — include runner scripts for local development:

```
spaces/<domain>-<scenario>/
├── space.yaml
├── wiring.yaml
├── run_local.py           # Python runner for CLI testing
├── simui_local.py         # SimUI web dashboard runner (if interactive)
└── tests/
    └── test_space.py      # Integration tests for the composed simulation
```

| Path | Required | Purpose |
|------|----------|---------|
| `space.yaml` | REQUIRED | Declares models, wiring, and runtime parameters |
| `wiring.yaml` | REQUIRED | Local-equivalent wiring with `class:` references and `args:` |
| `run_local.py` | RECOMMENDED | Standalone Python script to run the simulation from CLI |
| `simui_local.py` | RECOMMENDED | SimUI launcher for interactive web-based exploration |
| `tests/test_space.py` | RECOMMENDED | Integration tests verifying the full wired simulation |

---

### Manifest (`space.yaml`)

**REQUIRED** fields:

```yaml
schema_version: "2.0"
title: "<Domain>: <Scenario Name>"
description: "<One-line description of what this space demonstrates>"
models:
  - repo: Biosimulant/models
    alias: <unique_alias>
    manifest_path: models/<model-slug>/model.yaml
    parameters:
      <key>: <value>
runtime:
  duration: <float>          # Simulation end time
  tick_dt: <float>           # Global simulation timestep
  initial_inputs: {}         # Optional initial signal overrides
wiring:
  - from: <alias>.<output>
    to: [<alias>.<input>, ...]
```

| Field | Rule |
|-------|------|
| `schema_version` | Must be `"2.0"` |
| `title` | Format: `"<Domain>: <Descriptive Name>"` |
| `description` | Single sentence describing the scenario |
| `models` | Non-empty list of model references |
| `models[].repo` | Repository identifier (e.g., `Biosimulant/models`) |
| `models[].alias` | Unique short name for this model instance within the space |
| `models[].manifest_path` | Path to the model's `model.yaml` relative to repo root |
| `models[].parameters` | Dict of constructor kwargs overriding model defaults |
| `runtime.duration` | Positive float — total simulation time |
| `runtime.tick_dt` | Positive float — must be ≤ the smallest `min_dt` among all models |
| `wiring` | List of `{from, to}` signal routes |

**Validation rules**:
- When the same `repo` appears multiple times, every entry **must** include
  `manifest_path` to disambiguate.
- All `from` and `to` references must use the format `<alias>.<signal_name>`.
- Signal names in wiring must match the `inputs()` and `outputs()` declared by
  the referenced models.

---

### Wiring

**REQUIRED** — `wiring.yaml` provides a local-equivalent wiring spec:

```yaml
modules:
  <alias>:
    class: src.<module>:<ClassName>
    args: { <param>: <value>, ... }
wiring:
  - { from: "<alias>.<output>", to: ["<alias>.<input>", ...] }
```

Rules:
- Every model alias in `space.yaml` must have a corresponding entry in
  `wiring.yaml`.
- `class` must match the `bsim.entrypoint` from the model's `model.yaml`.
- `args` must be consistent with `parameters` in `space.yaml` (same keys, same
  values).

---

### Runner Scripts

#### `run_local.py` (RECOMMENDED)

A standalone Python script that wires and runs the simulation without the
platform. Must follow this structure:

```python
#!/usr/bin/env python3
"""<Space name> demo.

Demonstrates:
- <bullet point 1>
- <bullet point 2>

Run:
    pip install "bsim @ git+https://github.com/BioSimulant/bsim.git@<ref>"
    python spaces/<space-slug>/run_local.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import bsim

# Add model source directories to sys.path
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
for _model in [<list of model slugs used>]:
    sys.path.insert(0, str(MODELS_DIR / _model))

# Import model classes
from src.<module> import <ClassName>
# ...

def main() -> None:
    world = bsim.BioWorld()
    # Instantiate modules
    # Wire with bsim.WiringBuilder
    # Run simulation
    # Print/collect results

if __name__ == "__main__":
    main()
```

Requirements:
- Must include a docstring with run instructions.
- Must use `bsim.WiringBuilder` for wiring (not manual signal passing).
- Must print meaningful output (metrics, visual summaries).
- Must be runnable with only `bsim` installed (plus model dependencies).

#### `simui_local.py` (RECOMMENDED for interactive spaces)

Same structure as `run_local.py` but launches a SimUI web dashboard:

```python
from bsim.simui import Interface, Number, Button, EventLog, VisualsPanel

ui = Interface(
    world,
    title="<Space Title>",
    description="<Markdown description>",
    controls=[...],
    outputs=[...],
)
ui.launch(host="127.0.0.1", port=8765, open_browser=True)
```

Requirements:
- Must include `--port` and `--duration` CLI arguments via `argparse`.
- Must include a multi-line markdown description explaining what to observe.
- Must gracefully handle missing `bsim[ui]` dependency with a clear error message.

---

### Space Tests

**RECOMMENDED** — `tests/test_space.py` should verify the composed simulation:

#### 1. Space Loads Test

```python
def test_space_loads():
    """All models instantiate and wire without errors."""
    # Instantiate all models
    # Wire with WiringBuilder
    # Assert world has expected number of modules
```

#### 2. Simulation Runs Test

```python
def test_simulation_runs():
    """Simulation completes without crashing."""
    # Set up world (same as run_local.py)
    world.run(duration=<short_duration>, tick_dt=<tick_dt>)
    # Assert no exceptions
```

#### 3. Signal Flow Test

```python
def test_signals_flow():
    """Outputs propagate through the wiring graph."""
    # Run a few steps
    # Verify downstream modules received inputs
    # Verify monitor modules have data
```

#### 4. Visualization Collection Test

```python
def test_visuals_collected():
    """Visualization specs are returned after simulation."""
    world.run(duration=<short_duration>, tick_dt=<tick_dt>)
    visuals = world.collect_visuals()
    assert len(visuals) > 0
```

---

### Space Naming Conventions

| Element | Pattern | Example |
|---------|---------|---------|
| Space directory (slug) | `<domain>-<scenario>` (kebab-case) | `ecology-predator-prey` |
| Manifest title | `"<Domain>: <Descriptive Name>"` | `"Ecology: Predator-Prey"` |
| Model aliases | `snake_case`, short and descriptive | `neuron`, `spike_mon`, `rabbits` |
| Runner scripts | `run_local.py`, `simui_local.py` | — |

---

## Acceptance Checklist

Use this checklist before submitting a new model or space.

### New Model Checklist

- [ ] Directory follows a valid model slug structure (see [Naming Conventions](#naming-conventions)):
      - Native/handcrafted: `models/<domain>-<subject>-<role>/`
      - Public-catalog faithful (has `upstream:` in `model.yaml`): `models/<domain>-<format>-<subject>-<catalog_token>-<role>/`
- [ ] `model.yaml` contains all required fields (`schema_version`, `title`,
      `description`, `standard`, `tags`, `authors`, `bsim.entrypoint`)
- [ ] `src/<module>.py` exists with SPDX header, module docstring, and
      `from __future__ import annotations`
- [ ] Class inherits from `bsim.BioModule`
- [ ] `__init__` accepts `min_dt` as the last parameter with a sensible default
- [ ] `__init__` sets `self.min_dt`
- [ ] `inputs()` returns `Set[str]`
- [ ] `outputs()` returns non-empty `Set[str]`
- [ ] `advance_to(t)` updates internal state and populates `self._outputs`
- [ ] `get_outputs()` returns `Dict[str, BioSignal]` with keys matching `outputs()`
- [ ] `set_inputs()` implemented if `inputs()` is non-empty
- [ ] `reset()` implemented (RECOMMENDED)
- [ ] `visualize()` implemented with a standard render type (RECOMMENDED)
- [ ] All method signatures have type annotations
- [ ] All constructor parameters have sensible defaults
- [ ] External dependencies pinned with `==` in `model.yaml`
- [ ] `tests/test_<module>.py` exists with:
  - [ ] Instantiation test
  - [ ] Advance & output test
  - [ ] Output key consistency test
  - [ ] Reset test (if `reset()` exists)
  - [ ] Input handling test (if `inputs()` is non-empty)
  - [ ] Visualization test (if `visualize()` exists)
- [ ] `python scripts/validate_manifests.py` passes
- [ ] `python scripts/check_entrypoints.py` passes

### New Space Checklist

- [ ] Directory follows `spaces/<domain>-<scenario>/` structure
- [ ] `space.yaml` contains all required fields (`schema_version`, `title`,
      `description`, `models`, `runtime`, `wiring`)
- [ ] Every model entry has `repo`, `alias`, and `manifest_path`
- [ ] All model aliases are unique within the space
- [ ] `runtime.tick_dt` ≤ smallest `min_dt` of any model in the space
- [ ] All `wiring` entries use valid `<alias>.<signal>` references
- [ ] Signal names in `from`/`to` match model `inputs()`/`outputs()`
- [ ] No dangling inputs (every model input is wired or has a default)
- [ ] `wiring.yaml` exists with matching module classes and args
- [ ] `run_local.py` exists and is runnable (RECOMMENDED)
- [ ] `simui_local.py` exists for interactive spaces (RECOMMENDED)
- [ ] `tests/test_space.py` exists with:
  - [ ] Space loads test
  - [ ] Simulation runs test
  - [ ] Signal flow test (RECOMMENDED)
  - [ ] Visualization collection test (RECOMMENDED)
- [ ] `python scripts/validate_manifests.py` passes
