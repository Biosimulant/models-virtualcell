"""Tests for PerturbationSource model."""
from __future__ import annotations


def test_instantiation(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource()
    assert module.min_dt > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0


def test_advance_produces_outputs(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(min_dt=0.01)
    module.advance_to(0.01)
    outputs = module.get_outputs()
    for name in module.outputs():
        assert name in outputs
        signal = outputs[name]
        assert signal.source is not None
        assert signal.time == 0.01


def test_output_keys_match(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(min_dt=0.01)
    module.advance_to(0.01)
    assert set(module.get_outputs().keys()) == module.outputs()


def test_reset(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(min_dt=0.01)
    module.advance_to(0.1)
    module.advance_to(0.2)
    module.reset()
    module.advance_to(0.01)
    outputs = module.get_outputs()
    assert "perturbation" in outputs
    assert outputs["perturbation"].time == 0.01


def test_perturbation_inactive_before_apply_at(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(gene="SCN1A", apply_at=0.5, min_dt=0.01)
    module.advance_to(0.1)
    pert = module.get_outputs()["perturbation"].value
    assert pert["active"] is False
    assert pert["gene"] == ""


def test_perturbation_active_after_apply_at(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(
        gene="BDNF", pert_type="overexpression", magnitude=2.0, apply_at=0.1,
        min_dt=0.01,
    )
    module.advance_to(0.2)
    pert = module.get_outputs()["perturbation"].value
    assert pert["active"] is True
    assert pert["gene"] == "BDNF"
    assert pert["type"] == "overexpression"
    assert pert["magnitude"] == 2.0


def test_schedule_overrides_single(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(
        gene="SCN1A",
        schedule=[
            (0.0, 0.5, "KCNA1", "knockout", 0.0),
            (0.5, 1.0, "GAD1", "drug", 0.8),
        ],
        min_dt=0.01,
    )

    module.advance_to(0.3)
    pert = module.get_outputs()["perturbation"].value
    assert pert["active"] is True
    assert pert["gene"] == "KCNA1"

    module.advance_to(0.7)
    pert = module.get_outputs()["perturbation"].value
    assert pert["active"] is True
    assert pert["gene"] == "GAD1"
    assert pert["type"] == "drug"


def test_schedule_inactive_outside_windows(bsim):
    from src.perturbation_source import PerturbationSource

    module = PerturbationSource(
        schedule=[(0.2, 0.4, "SCN1A", "knockout", 0.0)],
        min_dt=0.01,
    )

    module.advance_to(0.1)
    assert module.get_outputs()["perturbation"].value["active"] is False

    module.advance_to(0.5)
    assert module.get_outputs()["perturbation"].value["active"] is False
