"""Tests for ExpressionTranslator model."""
from __future__ import annotations


def test_instantiation(bsim):
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator()
    assert module.min_dt > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0


def test_advance_produces_outputs(bsim):
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator(min_dt=0.01)
    module.advance_to(0.01)
    outputs = module.get_outputs()
    for name in module.outputs():
        assert name in outputs
        signal = outputs[name]
        assert signal.source is not None
        assert signal.time == 0.01


def test_output_keys_match(bsim):
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator(min_dt=0.01)
    module.advance_to(0.01)
    assert set(module.get_outputs().keys()) == module.outputs()


def test_reset(bsim):
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(base_current=5.0, min_dt=0.01)

    # Inject expression data to change state
    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={
                "gene_names": ["SCN1A", "KCNA1"],
                "fold_change": [0.1, 1.0],
            },
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    module.reset()
    module.advance_to(0.01)

    # After reset, should output base_current (no fold-change data)
    current = module.get_outputs()["current"].value
    assert current == 5.0


def test_baseline_current_with_no_expression(bsim):
    """Without expression data, output should be base_current."""
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator(base_current=10.0, min_dt=0.01)
    module.advance_to(0.01)
    current = module.get_outputs()["current"].value
    assert current == 10.0


def test_set_inputs_processes_expression(bsim):
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(base_current=5.0, min_dt=0.01)

    # SCN1A at 2x fold-change -> +3.0 * (2.0 - 1.0) = +3.0
    # All other mapped genes at 1.0 -> no contribution
    gene_names = ["SCN1A", "SCN2A", "KCNA1", "KCNB1", "GRIA1", "GRIA2", "GAD1", "GAD2"]
    fold_changes = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={"gene_names": gene_names, "fold_change": fold_changes},
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    current = module.get_outputs()["current"].value
    # base(5.0) + SCN1A(3.0 * 1.0) = 8.0
    assert abs(current - 8.0) < 0.01, f"Expected ~8.0, got {current}"


def test_knockout_reduces_current(bsim):
    """SCN1A knockout (fc=0) should reduce excitatory current."""
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(base_current=5.0, min_dt=0.01)

    gene_names = ["SCN1A", "SCN2A", "KCNA1", "KCNB1", "GRIA1", "GRIA2", "GAD1", "GAD2"]
    fold_changes = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={"gene_names": gene_names, "fold_change": fold_changes},
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    current = module.get_outputs()["current"].value
    # base(5.0) + SCN1A(3.0 * -1.0) = 2.0
    assert abs(current - 2.0) < 0.01, f"Expected ~2.0, got {current}"


def test_kcna1_knockout_increases_current(bsim):
    """KCNA1 knockout (fc=0) should increase current (less K+ repolarization)."""
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(base_current=5.0, min_dt=0.01)

    gene_names = ["SCN1A", "SCN2A", "KCNA1", "KCNB1", "GRIA1", "GRIA2", "GAD1", "GAD2"]
    fold_changes = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={"gene_names": gene_names, "fold_change": fold_changes},
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    current = module.get_outputs()["current"].value
    # base(5.0) + KCNA1(-2.5 * -1.0) = 7.5
    assert abs(current - 7.5) < 0.01, f"Expected ~7.5, got {current}"


def test_current_clamped(bsim):
    """Output should be clamped within [clamp_min, clamp_max]."""
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(
        base_current=5.0, clamp_min=-10.0, clamp_max=15.0, min_dt=0.01,
    )

    # Extreme upregulation of all excitatory genes
    gene_names = ["SCN1A", "SCN2A", "GRIA1", "GRIA2"]
    fold_changes = [10.0, 10.0, 10.0, 10.0]

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={"gene_names": gene_names, "fold_change": fold_changes},
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    current = module.get_outputs()["current"].value
    assert current <= 15.0, f"Current {current} exceeds clamp_max"


def test_custom_gene_mappings(bsim):
    """Custom gene mappings should override defaults."""
    from src.expression_translator import ExpressionTranslator
    from bsim.signals import BioSignal

    module = ExpressionTranslator(
        base_current=0.0,
        gene_mappings=[["MY_GENE", 10.0, 0.0]],
        min_dt=0.01,
    )

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={"gene_names": ["MY_GENE"], "fold_change": [2.0]},
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    current = module.get_outputs()["current"].value
    # 0.0 + 10.0 * (2.0 - 1.0) = 10.0
    assert abs(current - 10.0) < 0.01


def test_visualize_none_before_advance(bsim):
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator(min_dt=0.01)
    assert module.visualize() is None


def test_visualize_after_advance(bsim):
    from src.expression_translator import ExpressionTranslator

    module = ExpressionTranslator(min_dt=0.01)
    module.advance_to(0.01)
    vis = module.visualize()
    assert vis is not None
    assert vis["render"] == "timeseries"
