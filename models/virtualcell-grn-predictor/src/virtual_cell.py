# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Classical gene regulatory network predictor.

Predicts how gene expression changes in response to perturbations (gene
knockouts, overexpression, drug effects) using coupled ODEs with a
biologically-motivated interaction matrix.

The GRN dynamics follow:
    dx/dt = W @ x + b - decay * x + perturbation_effect

where x is the expression vector, W is the interaction matrix, b is the
basal transcription rate, and decay controls turnover.

Reference genes include ion channels, receptors, and signaling molecules
relevant to neuroscience, enabling downstream translation to biophysical
parameters via ExpressionTranslator.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bsim import BioWorld

from bsim import BioModule
from bsim.signals import BioSignal, SignalMetadata


# -- Default gene set: neuroscience-relevant genes --------------------------

DEFAULT_GENE_NAMES: List[str] = [
    "SCN1A",   # Nav1.1 sodium channel
    "SCN2A",   # Nav1.2 sodium channel
    "KCNA1",   # Kv1.1 potassium channel
    "KCNB1",   # Kv2.1 potassium channel
    "GRIA1",   # GluA1 AMPA receptor subunit
    "GRIA2",   # GluA2 AMPA receptor subunit
    "GRIN1",   # NMDA receptor subunit NR1
    "GRIN2A",  # NMDA receptor subunit NR2A
    "GAD1",    # GAD67 — GABA synthesis enzyme
    "GAD2",    # GAD65 — GABA synthesis enzyme
    "SLC6A1",  # GAT-1 GABA transporter
    "BDNF",    # Brain-derived neurotrophic factor
    "NTRK2",   # TrkB receptor (BDNF receptor)
    "DLG4",    # PSD-95 scaffold protein
    "SYP",     # Synaptophysin (vesicle marker)
    "SNAP25",  # Synaptic vesicle fusion
    "SYN1",    # Synapsin I (vesicle trafficking)
    "CAMK2A",  # CaMKII alpha (Ca2+ signaling)
    "CREB1",   # CREB transcription factor
    "FOS",     # c-Fos immediate early gene
]


def _build_default_interaction_matrix(n: int, seed: int) -> np.ndarray:
    """Build a biologically-motivated gene regulatory network.

    Encodes known regulatory relationships between neuroscience genes:
    - CREB1 activates BDNF, FOS, SYN1
    - BDNF (via NTRK2) upregulates ion channels and synaptic genes
    - CAMK2A activates CREB1
    - GRIN1/2A (NMDA) activate CAMK2A (Ca2+ influx)
    - GAD1/2 have cross-regulatory relationships
    - SCN genes have mutual compensation
    - All genes have self-decay (diagonal)
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.02, size=(n, n))

    # Self-regulation (negative feedback / decay)
    np.fill_diagonal(W, -0.5)

    if n < 20:
        return W

    # Gene indices (matching DEFAULT_GENE_NAMES order)
    SCN1A, SCN2A, KCNA1, KCNB1 = 0, 1, 2, 3
    GRIA1, GRIA2, GRIN1, GRIN2A = 4, 5, 6, 7
    GAD1, GAD2, SLC6A1 = 8, 9, 10
    BDNF, NTRK2, DLG4, SYP = 11, 12, 13, 14
    SNAP25, SYN1, CAMK2A, CREB1, FOS = 15, 16, 17, 18, 19

    # CREB1 -> activates BDNF, FOS, SYN1, NTRK2
    W[BDNF, CREB1] = 0.4
    W[FOS, CREB1] = 0.5
    W[SYN1, CREB1] = 0.3
    W[NTRK2, CREB1] = 0.2

    # CAMK2A -> activates CREB1
    W[CREB1, CAMK2A] = 0.4

    # GRIN1/2A (NMDA) -> activates CAMK2A (calcium signaling)
    W[CAMK2A, GRIN1] = 0.3
    W[CAMK2A, GRIN2A] = 0.3

    # BDNF -> upregulates synaptic and receptor genes
    W[GRIA1, BDNF] = 0.2
    W[GRIA2, BDNF] = 0.15
    W[DLG4, BDNF] = 0.2
    W[SYP, BDNF] = 0.15
    W[SNAP25, BDNF] = 0.15
    W[SCN1A, BDNF] = 0.1
    W[KCNA1, BDNF] = 0.1

    # NTRK2 mediates BDNF signaling
    W[BDNF, NTRK2] = 0.1

    # GAD1/GAD2 co-regulation
    W[GAD2, GAD1] = 0.2
    W[GAD1, GAD2] = 0.2
    W[SLC6A1, GAD1] = 0.15
    W[SLC6A1, GAD2] = 0.1

    # Sodium channel compensation
    W[SCN2A, SCN1A] = 0.15
    W[SCN1A, SCN2A] = 0.1

    # Potassium channel co-regulation
    W[KCNB1, KCNA1] = 0.1

    # Activity-dependent: FOS is a transient response gene
    W[FOS, FOS] = -0.8
    W[FOS, CAMK2A] = 0.3

    # Excitatory-inhibitory balance: GRIA activates GAD (homeostatic)
    W[GAD1, GRIA1] = 0.1
    W[GAD1, GRIN1] = 0.1

    return W


def _build_default_baseline(n: int, seed: int) -> np.ndarray:
    """Build a realistic baseline expression profile."""
    rng = np.random.default_rng(seed + 100)
    baseline = rng.uniform(1.0, 5.0, size=n)

    if n >= 20:
        baseline[0] = 3.5   # SCN1A
        baseline[1] = 2.8   # SCN2A
        baseline[2] = 3.0   # KCNA1
        baseline[3] = 2.5   # KCNB1
        baseline[4] = 4.0   # GRIA1
        baseline[5] = 3.8   # GRIA2
        baseline[6] = 3.2   # GRIN1
        baseline[7] = 2.5   # GRIN2A
        baseline[8] = 3.5   # GAD1
        baseline[9] = 3.0   # GAD2
        baseline[10] = 2.0  # SLC6A1
        baseline[11] = 2.5  # BDNF
        baseline[12] = 2.0  # NTRK2
        baseline[13] = 3.5  # DLG4
        baseline[14] = 4.0  # SYP
        baseline[15] = 3.8  # SNAP25
        baseline[16] = 3.5  # SYN1
        baseline[17] = 3.0  # CAMK2A
        baseline[18] = 2.5  # CREB1
        baseline[19] = 1.0  # FOS

    return baseline


class VirtualCell(BioModule):
    """GRN-based virtual cell predicting gene expression responses.

    Models a cell's transcriptional state as a vector of gene expression
    levels governed by a gene regulatory network (GRN).  Perturbations
    modify expression directly (knockout sets to zero, overexpression
    multiplies) and the network relaxes toward a new steady state.

    Parameters:
        n_genes: Number of genes in the model.
        gene_names: Custom gene name list (defaults to 20 neuro genes).
        interaction_matrix: Custom N x N GRN weight matrix (flat or nested).
        baseline_expression: Initial steady-state expression levels.
        decay_rate: Global mRNA decay rate constant.
        response_speed: How quickly expression responds to perturbation.
        seed: Random seed for reproducible default matrices.
        min_dt: Minimum time step.

    Emits:
        expression_profile: {gene_names, expression, baseline, fold_change}
        expression_summary: {top_up, top_down, mean_fold_change}
    """

    def __init__(
        self,
        n_genes: int = 20,
        gene_names: Optional[List[str]] = None,
        interaction_matrix: Optional[List[Any]] = None,
        baseline_expression: Optional[List[float]] = None,
        decay_rate: float = 0.3,
        response_speed: float = 5.0,
        seed: int = 42,
        min_dt: float = 0.01,
    ) -> None:
        self.min_dt = min_dt
        self.n_genes = n_genes
        self.decay_rate = decay_rate
        self.response_speed = response_speed
        self.seed = seed

        # Gene names
        if gene_names is not None:
            self.gene_names = list(gene_names)
        elif n_genes <= 20:
            self.gene_names = DEFAULT_GENE_NAMES[:n_genes]
        else:
            self.gene_names = DEFAULT_GENE_NAMES + [
                f"GENE_{i}" for i in range(20, n_genes)
            ]

        # Interaction matrix
        if interaction_matrix is not None:
            self._W = np.array(interaction_matrix, dtype=np.float64).reshape(
                n_genes, n_genes
            )
        else:
            self._W = _build_default_interaction_matrix(n_genes, seed)

        # Baseline expression
        if baseline_expression is not None:
            self._baseline = np.array(baseline_expression, dtype=np.float64)
        else:
            self._baseline = _build_default_baseline(n_genes, seed)

        # Basal transcription rate (maintains baseline at steady state)
        self._basal = self.decay_rate * self._baseline - self._W @ self._baseline

        # Runtime state
        self._expression = self._baseline.copy()
        self._perturbation_mask = np.ones(n_genes, dtype=np.float64)
        self._perturbation_active = False
        self._time: float = 0.0

        # History for visualization
        self._expression_history: List[List[float]] = []
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"perturbation"}

    def outputs(self) -> Set[str]:
        return {"expression_profile", "expression_summary"}

    def reset(self) -> None:
        """Reset expression to baseline."""
        self._expression = self._baseline.copy()
        self._perturbation_mask = np.ones(self.n_genes, dtype=np.float64)
        self._perturbation_active = False
        self._time = 0.0
        self._expression_history = []
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        sig = signals.get("perturbation")
        if sig is None:
            return
        pert = sig.value
        if not isinstance(pert, dict) or not pert.get("active", False):
            return

        gene = pert.get("gene", "")
        pert_type = pert.get("type", "none")
        magnitude = pert.get("magnitude", 0.0)

        if gene not in self.gene_names:
            return

        idx = self.gene_names.index(gene)

        if pert_type == "knockout":
            self._perturbation_mask[idx] = float(magnitude)
        elif pert_type == "overexpression":
            self._perturbation_mask[idx] = float(magnitude)
        elif pert_type == "drug":
            self._perturbation_mask[idx] = max(0.0, 1.0 - float(magnitude))

        self._perturbation_active = True

    def advance_to(self, t: float) -> None:
        """Advance the GRN dynamics to time t."""
        dt = t - self._time if t > self._time else self.min_dt
        self._time = t

        if self._perturbation_active:
            target = self._baseline * self._perturbation_mask

            n_substeps = max(1, int(dt / 0.001))
            sub_dt = dt / n_substeps

            for _ in range(n_substeps):
                interaction = self._W @ self._expression
                dx = (
                    interaction
                    + self._basal
                    - self.decay_rate * self._expression
                ) * sub_dt

                # Perturbation forcing: push perturbed genes toward target
                # Use abs(1 - mask) so both knockout (mask<1) and
                # overexpression (mask>1) produce positive forcing strength
                forcing = (
                    self.response_speed
                    * (target - self._expression)
                    * (np.abs(1.0 - self._perturbation_mask) + 0.1)
                    * sub_dt
                )

                self._expression = self._expression + dx + forcing
                self._expression = np.maximum(self._expression, 0.0)
        else:
            interaction = self._W @ self._expression
            dx = (
                interaction + self._basal - self.decay_rate * self._expression
            ) * dt
            self._expression = np.maximum(self._expression + dx, 0.0)

        # Record history
        self._expression_history.append(
            [t] + self._expression.tolist()
        )

        # Compute fold-change relative to baseline
        fold_change = self._expression / (self._baseline + 1e-8)

        # Find top up/down regulated genes
        fc_sorted = np.argsort(fold_change)
        top_down = [
            {"gene": self.gene_names[i], "fold_change": float(fold_change[i])}
            for i in fc_sorted[:3]
            if fold_change[i] < 0.9
        ]
        top_up = [
            {"gene": self.gene_names[i], "fold_change": float(fold_change[i])}
            for i in fc_sorted[-3:][::-1]
            if fold_change[i] > 1.1
        ]

        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "expression_profile": BioSignal(
                source=source,
                name="expression_profile",
                value={
                    "gene_names": list(self.gene_names),
                    "expression": self._expression.tolist(),
                    "baseline": self._baseline.tolist(),
                    "fold_change": fold_change.tolist(),
                },
                time=t,
                metadata=SignalMetadata(
                    units="log_counts",
                    description="Gene expression profile with fold-changes",
                    kind="state",
                ),
            ),
            "expression_summary": BioSignal(
                source=source,
                name="expression_summary",
                value={
                    "top_up": top_up,
                    "top_down": top_down,
                    "mean_fold_change": float(np.mean(fold_change)),
                },
                time=t,
                metadata=SignalMetadata(
                    description="Summary of expression changes",
                    kind="state",
                ),
            ),
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self._time,
            "expression": self._expression.tolist(),
            "perturbation_mask": self._perturbation_mask.tolist(),
        }

    def visualize(self) -> Optional[List[Dict[str, Any]]]:
        """Return bar chart of fold-change and timeseries of top changers."""
        if not self._expression_history:
            return None

        fold_change = self._expression / (self._baseline + 1e-8)
        bar_items = [
            {"label": name, "value": round(float(fc), 3)}
            for name, fc in zip(self.gene_names, fold_change)
        ]

        # Timeseries of the five most-changed genes
        fc_deviation = np.abs(fold_change - 1.0)
        top_indices = np.argsort(fc_deviation)[-5:][::-1]

        series = []
        for idx in top_indices:
            points = [
                [row[0], row[1 + idx]] for row in self._expression_history
            ]
            series.append({"name": self.gene_names[idx], "points": points})

        return [
            {
                "render": "bar",
                "data": {"items": bar_items, "title": "Gene Expression Fold-Change"},
                "description": (
                    f"Fold-change of {self.n_genes} genes relative to baseline. "
                    "Values < 1.0 indicate downregulation; > 1.0 indicate upregulation."
                ),
            },
            {
                "render": "timeseries",
                "data": {
                    "series": series,
                    "title": "Top Changing Genes Over Time",
                },
                "description": (
                    "Expression levels of the five most affected genes over "
                    "simulation time, showing transcriptional response dynamics."
                ),
            },
        ]
