# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Expression monitor for virtual cell simulations.

Collects and visualizes gene expression profiles over time. Produces
bar charts of current fold-changes and timeseries of tracked genes,
enabling visual inspection of the transcriptional response dynamics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


class ExpressionMonitor(BioModule):
    """Monitors gene expression profiles over time.

    Receives expression_profile signals from VirtualCell and accumulates
    history for visualization. Tracks a configurable set of genes and
    provides bar chart and timeseries visual outputs.

    Parameters:
        tracked_genes: List of gene names to track in detail. None tracks
            the five genes with the largest fold-change deviation.
        max_points: Maximum history points to retain per gene.
        min_dt: Minimum time step.

    Emits:
        expression_state: {"gene_names": [...], "fold_change": [...],
            "n_upregulated": int, "n_downregulated": int}
    """

    def __init__(
        self,
        tracked_genes: Optional[List[str]] = None,
        max_points: int = 5000,
        min_dt: float = 0.01,
    ) -> None:
        self.min_dt = min_dt
        self.tracked_genes = tracked_genes
        self.max_points = max_points

        self._gene_names: List[str] = []
        self._fold_change: List[float] = []
        self._history: Dict[str, List[List[float]]] = {}  # gene -> [[t, expr], ...]
        self._time: float = 0.0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"expression_profile"}

    def outputs(self) -> Set[str]:
        return {"expression_state"}

    def reset(self) -> None:
        """Clear accumulated history."""
        self._gene_names = []
        self._fold_change = []
        self._history = {}
        self._time = 0.0
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        sig = signals.get("expression_profile")
        if sig is None:
            return
        profile = sig.value
        if not isinstance(profile, dict):
            return

        self._gene_names = profile.get("gene_names", [])
        expression = profile.get("expression", [])
        self._fold_change = profile.get("fold_change", [])

        # Determine which genes to track
        if self.tracked_genes is not None:
            genes_to_track = self.tracked_genes
        else:
            # Auto-select: top 5 by fold-change deviation
            if self._fold_change:
                deviations = [
                    (abs(fc - 1.0), name)
                    for name, fc in zip(self._gene_names, self._fold_change)
                ]
                deviations.sort(reverse=True)
                genes_to_track = [name for _, name in deviations[:5]]
            else:
                genes_to_track = self._gene_names[:5]

        # Record history for tracked genes
        for i, name in enumerate(self._gene_names):
            if name in genes_to_track and i < len(expression):
                if name not in self._history:
                    self._history[name] = []
                self._history[name].append([sig.time, expression[i]])
                if len(self._history[name]) > self.max_points:
                    self._history[name] = self._history[name][-self.max_points:]

    def advance_to(self, t: float) -> None:
        """Update outputs with current expression state."""
        self._time = t

        n_up = sum(1 for fc in self._fold_change if fc > 1.1)
        n_down = sum(1 for fc in self._fold_change if fc < 0.9)

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "expression_state": BioSignal(
                source=source,
                name="expression_state",
                value={
                    "gene_names": list(self._gene_names),
                    "fold_change": list(self._fold_change),
                    "n_upregulated": n_up,
                    "n_downregulated": n_down,
                },
                time=t,
                metadata=SignalMetadata(
                    description="Current expression state summary",
                    kind="state",
                ),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {"time": self._time, "n_tracked": len(self._history)}

    def visualize(self) -> Optional[List[Dict[str, Any]]]:
        """Return fold-change bar chart and gene expression timeseries."""
        if not self._fold_change:
            return None

        # Bar chart of current fold-changes
        bar_items = [
            {"label": name, "value": round(float(fc), 3)}
            for name, fc in zip(self._gene_names, self._fold_change)
        ]

        panels: List[Dict[str, Any]] = [
            {
                "render": "bar",
                "data": {
                    "items": bar_items,
                    "title": "Expression Fold-Change (current)",
                },
                "description": (
                    "Current fold-change relative to baseline for all genes. "
                    "Values near 1.0 are unaffected; deviations show perturbation effects."
                ),
            },
        ]

        # Timeseries of tracked genes
        if self._history:
            series = [
                {"name": gene, "points": points}
                for gene, points in self._history.items()
            ]
            panels.append({
                "render": "timeseries",
                "data": {
                    "series": series,
                    "title": "Tracked Gene Expression Over Time",
                },
                "description": (
                    f"Expression levels of {len(self._history)} tracked genes "
                    "over simulation time."
                ),
            })

        return panels
