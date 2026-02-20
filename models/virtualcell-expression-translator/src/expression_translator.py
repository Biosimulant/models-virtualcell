# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Expression-to-biophysics translator for multi-scale simulations.

Bridges the molecular scale (gene expression from VirtualCell) to the
circuit scale (biophysical currents for neuron models). Maps fold-changes
in ion channel and receptor gene expression to equivalent changes in
injected current, enabling a complete perturbation-to-firing pipeline.

Default gene-to-parameter mappings reflect known neuroscience:
- SCN1A/SCN2A (Na+ channels): upregulation increases excitability
- KCNA1/KCNB1 (K+ channels): upregulation decreases excitability
- GRIA1/GRIA2 (AMPA receptors): upregulation increases excitatory drive
- GAD1/GAD2 (GABA synthesis): upregulation increases inhibition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


@dataclass
class GeneMapping:
    """Maps a gene's fold-change to a current contribution.

    Attributes:
        gene: Gene name (must match VirtualCell gene_names).
        weight: Scaling factor. Positive = excitatory contribution,
            negative = inhibitory contribution.
        baseline_current: Current contribution at fold-change = 1.0.
    """
    gene: str
    weight: float
    baseline_current: float = 0.0


# Default mappings for neuroscience-relevant genes
DEFAULT_GENE_MAPPINGS: List[GeneMapping] = [
    # Na+ channels: more expression -> more excitability -> positive current
    GeneMapping(gene="SCN1A", weight=3.0, baseline_current=0.0),
    GeneMapping(gene="SCN2A", weight=2.0, baseline_current=0.0),
    # K+ channels: more expression -> more repolarization -> negative current
    GeneMapping(gene="KCNA1", weight=-2.5, baseline_current=0.0),
    GeneMapping(gene="KCNB1", weight=-1.5, baseline_current=0.0),
    # AMPA receptors: more expression -> more excitatory drive
    GeneMapping(gene="GRIA1", weight=2.0, baseline_current=0.0),
    GeneMapping(gene="GRIA2", weight=1.0, baseline_current=0.0),
    # GABA synthesis: more expression -> more inhibition
    GeneMapping(gene="GAD1", weight=-2.0, baseline_current=0.0),
    GeneMapping(gene="GAD2", weight=-1.5, baseline_current=0.0),
]


class ExpressionTranslator(BioModule):
    """Translates gene expression fold-changes to biophysical current.

    Receives an expression_profile signal from VirtualCell and computes
    a scalar current value by summing weighted fold-change contributions
    from mapped genes. The output current can drive Izhikevich or
    Hodgkin-Huxley neuron populations.

    The translation formula for each mapped gene g:
        I_g = weight_g * (fold_change_g - 1.0) + baseline_current_g

    Total output current:
        I_total = base_current + sum(I_g for all mapped genes)

    Parameters:
        base_current: Constant baseline current added to the output.
        gene_mappings: List of [gene, weight, baseline_current] triples
            overriding the defaults. None uses DEFAULT_GENE_MAPPINGS.
        clamp_min: Minimum output current (prevents extreme inhibition).
        clamp_max: Maximum output current (prevents runaway excitation).
        min_dt: Minimum time step.

    Emits:
        current: float — total biophysical current for neuron models
    """

    def __init__(
        self,
        base_current: float = 5.0,
        gene_mappings: Optional[List[List[Any]]] = None,
        clamp_min: float = -20.0,
        clamp_max: float = 50.0,
        min_dt: float = 0.01,
    ) -> None:
        self.min_dt = min_dt
        self.base_current = base_current
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if gene_mappings is not None:
            self._mappings = [
                GeneMapping(gene=m[0], weight=m[1], baseline_current=m[2] if len(m) > 2 else 0.0)
                for m in gene_mappings
            ]
        else:
            self._mappings = list(DEFAULT_GENE_MAPPINGS)

        self._current: float = base_current
        self._fold_changes: Dict[str, float] = {}
        self._time: float = 0.0
        self._current_history: List[List[float]] = []
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"expression_profile"}

    def outputs(self) -> Set[str]:
        return {"current"}

    def reset(self) -> None:
        """Reset to baseline current."""
        self._current = self.base_current
        self._fold_changes = {}
        self._time = 0.0
        self._current_history = []
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        sig = signals.get("expression_profile")
        if sig is None:
            return
        profile = sig.value
        if not isinstance(profile, dict):
            return

        gene_names = profile.get("gene_names", [])
        fold_changes = profile.get("fold_change", [])
        if len(gene_names) != len(fold_changes):
            return

        self._fold_changes = dict(zip(gene_names, fold_changes))

    def advance_to(self, t: float) -> None:
        """Compute current from fold-changes."""
        self._time = t

        total = self.base_current
        for mapping in self._mappings:
            fc = self._fold_changes.get(mapping.gene, 1.0)
            contribution = mapping.weight * (fc - 1.0) + mapping.baseline_current
            total += contribution

        self._current = max(self.clamp_min, min(self.clamp_max, total))
        self._current_history.append([t, self._current])

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "current": BioSignal(
                source=source,
                name="current",
                value=self._current,
                time=t,
                metadata=SignalMetadata(
                    units="uA/cm^2",
                    description="Biophysical current derived from gene expression",
                    kind="state",
                ),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self._time,
            "current": self._current,
            "fold_changes": dict(self._fold_changes),
        }

    def visualize(self) -> Optional[Dict[str, Any]]:
        """Return timeseries of the translated current."""
        if not self._current_history:
            return None

        return {
            "render": "timeseries",
            "data": {
                "series": [{"name": "I_translated", "points": self._current_history}],
                "title": "Expression-Derived Current",
            },
            "description": (
                "Biophysical current computed from gene expression fold-changes. "
                f"Base current: {self.base_current} uA/cm^2. "
                f"Mapped genes: {', '.join(m.gene for m in self._mappings)}."
            ),
        }
