# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Perturbation source module for virtual cell simulations.

Emits gene perturbation signals that drive virtual cell models. Supports
knockout, overexpression, and drug perturbation types with configurable
timing and magnitude.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


class PerturbationSource(BioModule):
    """Emits perturbation signals on a configurable schedule.

    Outputs a perturbation dict describing which gene is perturbed, the
    perturbation type, and its magnitude. Downstream virtual cell modules
    consume this to compute transcriptional responses.

    Parameters:
        gene: Target gene name (e.g. "SCN1A", "BDNF").
        pert_type: One of "knockout", "overexpression", or "drug".
        magnitude: Perturbation strength. For knockout: fraction remaining
            (0.0 = full knockout). For overexpression: fold-change (2.0 = 2x).
            For drug: dose in arbitrary units.
        apply_at: Simulation time when the perturbation is applied.
        schedule: Optional list of (start, end, gene, pert_type, magnitude)
            tuples for multi-phase perturbation protocols. Overrides the
            single-perturbation parameters when provided.
        min_dt: Minimum time step.

    Emits:
        perturbation: {"gene": str, "type": str, "magnitude": float, "active": bool}
    """

    def __init__(
        self,
        gene: str = "SCN1A",
        pert_type: str = "knockout",
        magnitude: float = 0.0,
        apply_at: float = 0.0,
        schedule: Optional[List[Tuple[float, float, str, str, float]]] = None,
        min_dt: float = 0.01,
    ) -> None:
        self.min_dt = min_dt
        self.gene = gene
        self.pert_type = pert_type
        self.magnitude = magnitude
        self.apply_at = apply_at
        self.schedule = schedule
        self._time: float = 0.0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return set()

    def outputs(self) -> Set[str]:
        return {"perturbation"}

    def reset(self) -> None:
        """Reset to initial state."""
        self._time = 0.0
        self._outputs = {}

    def _active_perturbation(self, t: float) -> Dict[str, Any]:
        """Determine the active perturbation at time t."""
        if self.schedule is not None:
            for start, end, gene, pert_type, mag in self.schedule:
                if start <= t < end:
                    return {
                        "gene": gene,
                        "type": pert_type,
                        "magnitude": mag,
                        "active": True,
                    }
            return {"gene": "", "type": "none", "magnitude": 0.0, "active": False}

        if t >= self.apply_at:
            return {
                "gene": self.gene,
                "type": self.pert_type,
                "magnitude": self.magnitude,
                "active": True,
            }
        return {"gene": "", "type": "none", "magnitude": 0.0, "active": False}

    def advance_to(self, t: float) -> None:
        """Advance and emit the current perturbation state."""
        self._time = t
        pert = self._active_perturbation(t)
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "perturbation": BioSignal(
                source=source,
                name="perturbation",
                value=pert,
                time=t,
                metadata=SignalMetadata(
                    description="Gene perturbation specification",
                    kind="state",
                ),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {"time": self._time}
