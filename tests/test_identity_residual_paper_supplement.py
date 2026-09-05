from __future__ import annotations

import numpy as np

from singing_identity.methods.identity_residual import fit_oas_dual, speaker_interval
from singing_identity.evaluation.metrics import cosine_scores


def test_x1_identical_layers_have_identical_oas_rankings() -> None:
    rng = np.random.default_rng(71601)
    speakers = rng.normal(size=(24, 12))
    offset = np.full(12, 0.4)
    speech = speakers + rng.normal(scale=0.01, size=speakers.shape)
    singing = speakers + offset + rng.normal(scale=0.01, size=speakers.shape)
    train = np.arange(16)
    test = np.arange(16, 24)
    z_train = np.vstack([speech[train], singing[train]])
    midpoint = z_train.mean(axis=0)
    backend_a = fit_oas_dual(z_train)
    backend_b = fit_oas_dual(z_train.copy())
    scores_a = cosine_scores(
        backend_a.whiten(speech[test] - midpoint), backend_a.whiten(singing[test] - midpoint)
    )
    scores_b = cosine_scores(
        backend_b.whiten(speech[test] - midpoint), backend_b.whiten(singing[test] - midpoint)
    )
    assert np.array_equal(np.argsort(-scores_a, axis=1), np.argsort(-scores_b, axis=1))


def test_x2_known_speaker_advantage_is_positive() -> None:
    values = {f"s{i:03d}": [1.0 if i % 5 else 0.0] for i in range(100)}
    mean, low, _high, _p, n = speaker_interval(values, seed=71602, samples=10_000)
    assert n == 100
    assert mean > 0
    assert low > 0


def test_x2_speaker_permuted_advantage_is_not_positive() -> None:
    # A deterministic speaker permutation pairs equal positive and negative effects.
    values = {f"s{i:03d}": [1.0 if i % 2 == 0 else -1.0] for i in range(100)}
    mean, low, high, _p, _n = speaker_interval(values, seed=71603, samples=10_000)
    assert abs(mean) < 1e-12
    assert low < 0 < high
