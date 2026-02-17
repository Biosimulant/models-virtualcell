"""Tests for ExpressionMonitor model."""
from __future__ import annotations


def test_instantiation(bsim):
    from src.expression_monitor import ExpressionMonitor

    module = ExpressionMonitor()
    assert module.min_dt > 0
    assert isinstance(module.inputs(), set)
    assert isinstance(module.outputs(), set)
    assert len(module.outputs()) > 0


def test_advance_produces_outputs(bsim):
    from src.expression_monitor import ExpressionMonitor

    module = ExpressionMonitor(min_dt=0.01)
    module.advance_to(0.01)
    outputs = module.get_outputs()
    for name in module.outputs():
        assert name in outputs
        signal = outputs[name]
        assert signal.source is not None
        assert signal.time == 0.01


def test_output_keys_match(bsim):
    from src.expression_monitor import ExpressionMonitor

    module = ExpressionMonitor(min_dt=0.01)
    module.advance_to(0.01)
    assert set(module.get_outputs().keys()) == module.outputs()


def test_reset(bsim):
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(min_dt=0.01)

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={
                "gene_names": ["A", "B"],
                "expression": [1.0, 2.0],
                "fold_change": [1.0, 0.5],
            },
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    module.reset()
    assert module._history == {}
    assert module._fold_change == []


def test_set_inputs_records_history(bsim):
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(tracked_genes=["A"], min_dt=0.01)

    for i in range(1, 4):
        module.set_inputs({
            "expression_profile": BioSignal(
                source="test", name="expression_profile",
                value={
                    "gene_names": ["A", "B"],
                    "expression": [float(i), 2.0],
                    "fold_change": [float(i), 1.0],
                },
                time=i * 0.01,
            ),
        })
        module.advance_to(i * 0.01)

    assert "A" in module._history
    assert len(module._history["A"]) == 3
    assert "B" not in module._history


def test_expression_state_counts(bsim):
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(min_dt=0.01)

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={
                "gene_names": ["A", "B", "C", "D"],
                "expression": [1.0, 2.0, 3.0, 4.0],
                "fold_change": [0.5, 1.0, 1.5, 2.0],  # 1 down, 2 up
            },
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    state = module.get_outputs()["expression_state"].value
    assert state["n_downregulated"] == 1  # A at 0.5
    assert state["n_upregulated"] == 2    # C at 1.5, D at 2.0


def test_max_points_respected(bsim):
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(tracked_genes=["A"], max_points=5, min_dt=0.01)

    for i in range(1, 11):
        module.set_inputs({
            "expression_profile": BioSignal(
                source="test", name="expression_profile",
                value={
                    "gene_names": ["A"],
                    "expression": [float(i)],
                    "fold_change": [float(i)],
                },
                time=i * 0.01,
            ),
        })
        module.advance_to(i * 0.01)

    assert len(module._history["A"]) == 5


def test_visualize_none_before_advance(bsim):
    from src.expression_monitor import ExpressionMonitor

    module = ExpressionMonitor(min_dt=0.01)
    assert module.visualize() is None


def test_visualize_after_input(bsim):
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(tracked_genes=["A"], min_dt=0.01)

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={
                "gene_names": ["A", "B"],
                "expression": [1.0, 2.0],
                "fold_change": [0.5, 1.5],
            },
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    vis = module.visualize()
    assert vis is not None
    assert isinstance(vis, list)
    assert vis[0]["render"] == "bar"
    assert vis[1]["render"] == "timeseries"


def test_auto_select_tracked_genes(bsim):
    """Without explicit tracked_genes, should auto-select top deviators."""
    from src.expression_monitor import ExpressionMonitor
    from bsim.signals import BioSignal

    module = ExpressionMonitor(min_dt=0.01)

    module.set_inputs({
        "expression_profile": BioSignal(
            source="test", name="expression_profile",
            value={
                "gene_names": ["A", "B", "C", "D", "E", "F", "G"],
                "expression": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "fold_change": [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
            },
            time=0.01,
        ),
    })
    module.advance_to(0.01)

    # A (0.1, deviation 0.9) and G (5.0, deviation 4.0) should be tracked
    assert "A" in module._history
    assert "G" in module._history
