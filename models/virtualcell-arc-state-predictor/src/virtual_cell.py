# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Arc Institute State Transition ML predictor.

Predicts how gene expression changes in response to perturbations using the
Arc Institute's State Transition model — a GPT-2-based transformer trained
on 270M+ single-cell transcriptomic profiles.

The model is loaded from a HuggingFace checkpoint (e.g.
``arcinstitute/ST-Tahoe``) and runs inference once per perturbation change.
The predicted steady-state expression profile is cached and re-emitted each
simulation tick.

Requires the ``arc-state`` package and its dependencies:
    uv pip install arc-state

Both this module and ``virtualcell-grn-predictor`` emit identical BioSignal
output formats, so they are drop-in replaceable in any space.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim import BioWorld

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata

logger = logging.getLogger(__name__)


# -- Default gene set (shared with virtualcell-grn-predictor) ---------------

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
    """Arc State Transition virtual cell predictor.

    Uses the Arc Institute State Transition model to predict post-
    perturbation gene expression profiles.  Inference runs once per
    perturbation change and the result is cached between ticks.

    Parameters:
        model_name: HuggingFace repo for the Arc ST checkpoint.
        checkpoint_filename: Path within the HF repo to the ``.ckpt`` file.
        baseline_adata_path: Path to a baseline ``.h5ad`` AnnData file.
            If *None*, a synthetic baseline is constructed from the default
            gene set (approximate; real data recommended for production).
        pert_col: AnnData ``obs`` column name for perturbation labels.
        embed_key: AnnData ``obsm`` key for cell embeddings / HVG features.
        control_label: Label used for control (non-perturbed) cells.
        device: PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
        n_genes: Number of genes to track in output signals.
        gene_names: Custom gene name list (defaults to 20 neuro genes).
        seed: Random seed for reproducible synthetic baseline.
        min_dt: Minimum time step.

    Emits:
        expression_profile: {gene_names, expression, baseline, fold_change}
        expression_summary: {top_up, top_down, mean_fold_change}
    """

    def __init__(
        self,
        model_name: str = "arcinstitute/ST-Tahoe",
        checkpoint_filename: str = "checkpoints/final.ckpt",
        baseline_adata_path: Optional[str] = None,
        pert_col: str = "target_gene",
        embed_key: str = "X_hvg",
        control_label: str = "non-targeting",
        device: str = "cpu",
        n_genes: int = 20,
        gene_names: Optional[List[str]] = None,
        seed: int = 42,
        min_dt: float = 0.01,
    ) -> None:
        self.min_dt = min_dt
        self.n_genes = n_genes
        self.seed = seed

        # Arc State configuration
        self._model_name = model_name
        self._checkpoint_filename = checkpoint_filename
        self._baseline_adata_path = baseline_adata_path
        self._pert_col = pert_col
        self._embed_key = embed_key
        self._control_label = control_label
        self._device = device

        # Lazy-loaded runtime objects
        self._st_model = None
        self._baseline_adata = None
        self._arc_gene_names: Optional[List[str]] = None

        # Gene names
        if gene_names is not None:
            self.gene_names = list(gene_names)
        elif n_genes <= 20:
            self.gene_names = DEFAULT_GENE_NAMES[:n_genes]
        else:
            self.gene_names = DEFAULT_GENE_NAMES + [
                f"GENE_{i}" for i in range(20, n_genes)
            ]

        # Baseline expression (used for fold-change computation)
        self._baseline = _build_default_baseline(n_genes, seed)

        # Runtime state
        self._expression = self._baseline.copy()
        self._perturbation_active = False
        self._needs_inference = False
        self._current_gene: Optional[str] = None
        self._current_pert_type: Optional[str] = None
        self._time: float = 0.0

        # History for visualization
        self._expression_history: List[List[float]] = []
        self._outputs: Dict[str, BioSignal] = {}

    # -- BioModule interface --------------------------------------------------

    def inputs(self) -> Set[str]:
        return {"perturbation"}

    def outputs(self) -> Set[str]:
        return {"expression_profile", "expression_summary"}

    def reset(self) -> None:
        """Reset expression to baseline."""
        self._expression = self._baseline.copy()
        self._perturbation_active = False
        self._needs_inference = False
        self._current_gene = None
        self._current_pert_type = None
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

        if gene not in self.gene_names:
            return

        # Check if the perturbation changed (invalidate cache)
        if gene != self._current_gene or pert_type != self._current_pert_type:
            self._needs_inference = True
            self._current_gene = gene
            self._current_pert_type = pert_type

        self._perturbation_active = True

    def advance_to(self, t: float) -> None:
        """Advance to time *t*.

        The Arc model produces a steady-state prediction, so there are no
        per-tick dynamics.  Inference runs once when the perturbation changes;
        subsequent ticks re-emit the cached prediction.
        """
        self._time = t

        if self._needs_inference and self._perturbation_active:
            self._run_arc_inference()

        self._emit_outputs(t)

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self._time,
            "expression": self._expression.tolist(),
            "current_gene": self._current_gene,
            "current_pert_type": self._current_pert_type,
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

    # -----------------------------------------------------------------------
    # Arc State Transition inference
    # -----------------------------------------------------------------------

    def _ensure_arc_model(self) -> None:
        """Lazy-load the Arc ST model on first use."""
        if self._st_model is not None:
            return

        from huggingface_hub import hf_hub_download
        from state.tx.models import StateTransitionPerturbationModel
        import torch

        logger.info(
            "Loading Arc ST model from %s (%s)...",
            self._model_name, self._checkpoint_filename,
        )
        ckpt_path = hf_hub_download(
            repo_id=self._model_name,
            filename=self._checkpoint_filename,
        )
        self._st_model = StateTransitionPerturbationModel.load_from_checkpoint(
            ckpt_path, map_location=self._device,
        )
        self._st_model.eval()
        self._st_model.to(torch.device(self._device))

        # Store the model's gene names if available
        if hasattr(self._st_model, "gene_names") and self._st_model.gene_names:
            self._arc_gene_names = list(self._st_model.gene_names)

        # Load or construct baseline AnnData
        if self._baseline_adata_path:
            import scanpy as sc

            self._baseline_adata = sc.read_h5ad(self._baseline_adata_path)
            logger.info(
                "Loaded baseline AnnData: %d cells x %d genes",
                self._baseline_adata.n_obs, self._baseline_adata.n_vars,
            )
        else:
            self._build_synthetic_baseline()

    def _build_synthetic_baseline(self) -> None:
        """Construct a minimal AnnData from the default gene set.

        Enables Arc inference without a real scRNA-seq dataset.
        Results are approximate — provide a real ``.h5ad`` via
        ``baseline_adata_path`` for production use.
        """
        import anndata as ad
        import pandas as pd

        n_cells = 50
        rng = np.random.default_rng(self.seed)
        X = np.tile(self._baseline, (n_cells, 1))
        X += rng.normal(0, 0.1, size=X.shape)
        X = np.maximum(X, 0.0)

        obs = pd.DataFrame({
            self._pert_col: [self._control_label] * n_cells,
        })
        var = pd.DataFrame(index=self.gene_names)

        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
        adata.obsm[self._embed_key] = X.astype(np.float32)

        self._baseline_adata = adata
        logger.info(
            "Built synthetic baseline AnnData: %d cells x %d genes",
            n_cells, len(self.gene_names),
        )

    def _encode_perturbation(self, gene: str):
        """Encode a perturbation as a one-hot tensor for the Arc model."""
        import torch

        if self._arc_gene_names and gene in self._arc_gene_names:
            idx = self._arc_gene_names.index(gene)
            pert_dim = len(self._arc_gene_names)
        elif hasattr(self._st_model, "hparams") and "pert_dim" in self._st_model.hparams:
            pert_dim = self._st_model.hparams["pert_dim"]
            idx = self.gene_names.index(gene) if gene in self.gene_names else 0
            idx = min(idx, pert_dim - 1)
        else:
            pert_dim = self.n_genes
            idx = self.gene_names.index(gene) if gene in self.gene_names else 0

        n_cells = self._baseline_adata.n_obs
        pert_emb = torch.zeros(n_cells, pert_dim, dtype=torch.float32)
        pert_emb[:, idx] = 1.0

        return pert_emb.to(self._device)

    def _run_arc_inference(self) -> None:
        """Run Arc State Transition model prediction."""
        import torch

        self._ensure_arc_model()

        if self._current_gene is None:
            return

        logger.info(
            "Running Arc ST inference for %s %s...",
            self._current_pert_type, self._current_gene,
        )

        ctrl_emb = torch.tensor(
            self._baseline_adata.obsm[self._embed_key],
            dtype=torch.float32,
        ).to(self._device)

        pert_emb = self._encode_perturbation(self._current_gene)
        n_cells = ctrl_emb.shape[0]

        batch: Dict[str, Any] = {
            "ctrl_cell_emb": ctrl_emb,
            "pert_emb": pert_emb,
            "pert_name": [self._current_gene] * n_cells,
        }

        with torch.no_grad():
            output = self._st_model.predict_step(batch, batch_idx=0)

        # Prefer gene-space predictions if decoder is available
        if (
            "pert_cell_counts_preds" in output
            and output["pert_cell_counts_preds"] is not None
        ):
            predicted = output["pert_cell_counts_preds"].cpu().numpy()
        else:
            predicted = output["preds"].cpu().numpy()

        # Average across cells
        predicted_mean = predicted.mean(axis=0)

        # Map to our gene set
        self._map_predictions_to_genes(predicted_mean)
        self._needs_inference = False

        logger.info(
            "Arc inference complete. Mean fold-change: %.3f",
            float(np.mean(self._expression / (self._baseline + 1e-8))),
        )

    def _map_predictions_to_genes(self, predicted: np.ndarray) -> None:
        """Map Arc model predictions to the VirtualCell gene set."""
        if len(predicted) == self.n_genes:
            self._expression = np.maximum(predicted.astype(np.float64), 0.0)
            return

        if self._arc_gene_names:
            arc_name_to_idx = {
                g: i for i, g in enumerate(self._arc_gene_names)
            }
            new_expr = self._baseline.copy()
            for i, gene in enumerate(self.gene_names):
                if gene in arc_name_to_idx:
                    arc_idx = arc_name_to_idx[gene]
                    if arc_idx < len(predicted):
                        new_expr[i] = max(0.0, float(predicted[arc_idx]))
            self._expression = new_expr
        else:
            n = min(len(predicted), self.n_genes)
            self._expression = self._baseline.copy()
            self._expression[:n] = np.maximum(
                predicted[:n].astype(np.float64), 0.0
            )

    # -----------------------------------------------------------------------
    # Shared output emission
    # -----------------------------------------------------------------------

    def _emit_outputs(self, t: float) -> None:
        """Build and store BioSignal outputs."""
        self._expression_history.append(
            [t] + self._expression.tolist()
        )

        fold_change = self._expression / (self._baseline + 1e-8)

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
