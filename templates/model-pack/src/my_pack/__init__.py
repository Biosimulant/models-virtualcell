"""My custom bsim pack.

This pack provides custom modules for bsim simulations.

Usage in YAML configs:
    modules:
      counter:
        class: my_pack.Counter
        args:
          name: "my_counter"
        min_dt: 0.1
"""
from .modules import Counter, Accumulator

__version__ = "0.1.0"

__all__ = [
    "Counter",
    "Accumulator",
]
