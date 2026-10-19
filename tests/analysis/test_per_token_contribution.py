import importlib.util
from pathlib import Path

import pytest

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts" / "analysis" / "per_token_contribution.py"
)

spec = importlib.util.spec_from_file_location("per_token", SCRIPT)
per_token = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_token)


def test_uniform_contributions_are_diffuse():
    """Check equal per-token contributions report no localization."""
    stats = per_token.summarize_concentration(np.ones(100))

    assert stats["effective_fraction"] == pytest.approx(1.0)


def test_single_spike_is_maximally_local():
    """Check one dominant position collapses the effective count to one."""
    deviations = np.zeros(100)
    deviations[42] = 5.0

    stats = per_token.summarize_concentration(deviations)

    assert stats["effective_positions"] == pytest.approx(1.0)
    assert stats["effective_fraction"] == pytest.approx(0.01)
    assert stats["top1pct_mass"] == pytest.approx(1.0)


def test_effective_fraction_tracks_spike_count():
    """Check the effective count matches the number of active positions."""
    deviations = np.zeros(200)
    deviations[:8] = 1.0

    stats = per_token.summarize_concentration(deviations)

    assert stats["effective_positions"] == pytest.approx(8.0)


def test_sign_does_not_change_concentration():
    """Check concentration uses magnitude, not signed contribution."""
    deviations = np.zeros(100)
    deviations[10] = 3.0
    deviations[20] = -3.0

    stats = per_token.summarize_concentration(deviations)

    assert stats["effective_positions"] == pytest.approx(2.0)


def test_zero_contributions_do_not_divide_by_zero():
    """Check an all-zero profile reports as diffuse rather than failing."""
    stats = per_token.summarize_concentration(np.zeros(50))

    assert stats["effective_fraction"] == 1.0
    assert stats["top1pct_mass"] == 0.0


def test_regression_weights_are_the_coefficient_vector():
    """Check a regression probe yields a flat weight vector."""
    model = Ridge().fit(np.random.rand(20, 6), np.random.rand(20))

    weights = per_token.get_probe_weights(model, "regression", 0)

    assert weights.shape == (6,)


def test_binary_classifier_weights_are_squeezed():
    """Check a binary probe's (1, H) coef_ is reduced to (H,)."""
    model = LogisticRegression().fit(
        np.random.rand(20, 6), [0, 1] * 10
    )

    weights = per_token.get_probe_weights(model, "classification", 0)

    assert weights.shape == (6,)


def test_multiclass_selects_requested_class():
    """Check the class index picks the matching coef_ row."""
    model = LogisticRegression().fit(
        np.random.rand(30, 6), [0, 1, 2] * 10
    )

    weights = per_token.get_probe_weights(model, "classification", 2)

    assert np.array_equal(weights, model.coef_[2])


def test_rejects_estimator_without_coefficients():
    """Check a non-linear estimator is refused rather than approximated."""
    class NoCoef:
        pass

    with pytest.raises(NotImplementedError, match="no coef_"):
        per_token.get_probe_weights(NoCoef(), "multilabel", 0)


def test_decomposition_identity_holds():
    """Check mean of per-token contributions equals the probe output."""
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(37, 12))
    weights = rng.normal(size=12)
    intercept = 0.7

    pooled_prediction = tokens.mean(axis=0) @ weights + intercept
    per_token_mean = (tokens @ weights).mean() + intercept

    assert pooled_prediction == pytest.approx(per_token_mean, abs=1e-10)


def test_mean_pooling_dilutes_a_local_motif_as_one_over_length():
    """Check a fixed motif's pooled contribution decays like 1 / length."""
    rng = np.random.default_rng(0)
    hidden = 12
    weights = rng.normal(size=hidden)
    motif = rng.normal(size=hidden) * 10.0

    pooled = {}
    for length in [100, 1000]:
        tokens = np.zeros((length, hidden))
        tokens[:8] = motif
        pooled[length] = abs(tokens.mean(axis=0) @ weights)

    # Ten times the background length, one tenth the pooled signal.
    assert pooled[1000] == pytest.approx(pooled[100] / 10, rel=1e-6)
