# biosim-my-pack

A template for creating custom biosim module packs.

## Installation

```bash
# Development install
pip install -e .

# Or install from PyPI (after publishing)
pip install biosim-my-pack
```

## Usage

### In Python

```python
import biosim
from my_pack import Counter

world = biosim.BioWorld()
world.add_biomodule("counter", Counter(name="my_counter", min_dt=0.1))
world.run(duration=10.0, tick_dt=0.1)
```

### In YAML Configs

```yaml
modules:
  counter:
    class: my_pack.Counter
    args:
      name: "my_counter"
    min_dt: 0.1
```

Run with:
```bash
python -m biosim config.yaml --simui
```

## Included Components

### Modules

| Module | Description | Inputs | Outputs |
|--------|-------------|--------|---------|
| `Counter` | Counts simulation steps | - | `count` |
| `Accumulator` | Accumulates values | `value` | `total` |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Creating Your Own Pack

1. Copy this template
2. Rename `my_pack` to your package name
3. Update `pyproject.toml` with your details
4. Implement your modules in `modules.py`
5. Add example configs in `examples/`
6. Publish to PyPI

## License

MIT
