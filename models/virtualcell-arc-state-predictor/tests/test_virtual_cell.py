"""Tests for VirtualCell Arc State predictor."""
from __future__ import annotations

import numpy as np
import pytest


def test_instantiation(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell()
    assert module.min_dt > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0
    assert module.n_genes == 20
    assert len(module.gene_names) == 20


def test_advance_produces_outputs(biosim):
    """Advance without perturbation should emit baseline expression."""
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.01)
    outputs = module.get_outputs()
    for name in module.outputs():
        assert name in outputs
        signal = outputs[name]
        assert signal.source is not None
        assert signal.time == 0.01


def test_output_keys_match(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.01)
    assert set(module.get_outputs().keys()) == module.outputs()


def test_reset(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.1)
    module.reset()
    assert module._time == 0.0
    assert len(module._expression_history) == 0
    assert module._current_gene is None
    assert module._needs_inference is False
    np.testing.assert_array_almost_equal(module._expression, module._baseline)


def test_expression_profile_structure(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.1)
    profile = module.get_outputs()["expression_profile"].value

    assert "gene_names" in profile
    assert "expression" in profile
    assert "baseline" in profile
    assert "fold_change" in profile
    assert len(profile["gene_names"]) == 20
    assert len(profile["expression"]) == 20
    assert len(profile["fold_change"]) == 20


def test_expression_summary_structure(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.1)
    summary = module.get_outputs()["expression_summary"].value

    assert "top_up" in summary
    assert "top_down" in summary
    assert "mean_fold_change" in summary
    assert isinstance(summary["mean_fold_change"], float)


def test_unknown_gene_ignored(biosim):
    """Perturbation of unknown gene should be silently ignored."""
    from src.virtual_cell import VirtualCell
    from biosim.signals import BioSignal

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.01)
    baseline_fc = list(module.get_outputs()["expression_profile"].value["fold_change"])

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "NONEXISTENT", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.02,
        ),
    })
    module.advance_to(0.02)
    fc_after = module.get_outputs()["expression_profile"].value["fold_change"]

    for i in range(len(baseline_fc)):
        assert abs(fc_after[i] - baseline_fc[i]) < 0.1


def test_perturbation_sets_needs_inference(biosim):
    """Setting a perturbation should flag that inference is needed."""
    from src.virtual_cell import VirtualCell
    from biosim.signals import BioSignal

    module = VirtualCell(min_dt=0.01)
    assert module._needs_inference is False

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "SCN1A", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.0,
        ),
    })
    assert module._needs_inference is True
    assert module._current_gene == "SCN1A"


def test_same_perturbation_no_retrigger(biosim):
    """Repeating the same perturbation should not re-trigger inference."""
    from src.virtual_cell import VirtualCell
    from biosim.signals import BioSignal

    module = VirtualCell(min_dt=0.01)

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "SCN1A", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.0,
        ),
    })
    module._needs_inference = False  # simulate inference completed

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "SCN1A", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.01,
        ),
    })
    assert module._needs_inference is False


def test_different_perturbation_retriggers(biosim):
    """Changing the perturbation gene should re-trigger inference."""
    from src.virtual_cell import VirtualCell
    from biosim.signals import BioSignal

    module = VirtualCell(min_dt=0.01)

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "SCN1A", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.0,
        ),
    })
    module._needs_inference = False

    module.set_inputs({
        "perturbation": BioSignal(
            source="test", name="perturbation",
            value={"gene": "BDNF", "type": "knockout", "magnitude": 0.0, "active": True},
            time=0.01,
        ),
    })
    assert module._needs_inference is True
    assert module._current_gene == "BDNF"


def test_visualize_none_before_advance(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    assert module.visualize() is None


def test_visualize_after_advance(biosim):
    from src.virtual_cell import VirtualCell

    module = VirtualCell(min_dt=0.01)
    module.advance_to(0.01)
    vis = module.visualize()
    assert vis is not None
    assert isinstance(vis, list)
    assert len(vis) == 2
    assert vis[0]["render"] == "bar"
    assert vis[1]["render"] == "timeseries"


def test_custom_gene_set(biosim):
    """Module should work with a custom gene set."""
    from src.virtual_cell import VirtualCell

    names = ["A", "B", "C"]
    module = VirtualCell(
        n_genes=3, gene_names=names, min_dt=0.01,
    )
    assert module.gene_names == names
    module.advance_to(0.01)
    profile = module.get_outputs()["expression_profile"].value
    assert profile["gene_names"] == names
