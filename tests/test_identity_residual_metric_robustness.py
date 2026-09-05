from __future__ import annotations

import numpy as np

from singing_identity.methods.identity_residual import fit_oas_dual
from singing_identity.evaluation.metrics import (
    cosine_scores,
    euclidean_scores,
    ranks_from_scores,
)


def synthetic_centroids(seed: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    identity = rng.normal(scale=2.0, size=(24, 16))
    offset = np.linspace(-0.8, 1.1, 16)
    speech = identity + rng.normal(scale=0.05, size=identity.shape)
    singing = identity + offset + rng.normal(scale=0.05, size=identity.shape)
    return speech, singing, offset


def test_common_centering_changes_cosine_not_euclidean() -> None:
    speech, singing, _ = synthetic_centroids()
    common = np.arange(speech.shape[1], dtype=float) * 3.0
    raw_cos = cosine_scores(speech, singing)
    shifted_cos = cosine_scores(speech + common, singing + common)
    assert not np.allclose(raw_cos, shifted_cos)
    np.testing.assert_allclose(
        euclidean_scores(speech, singing),
        euclidean_scores(speech + common, singing + common),
        rtol=1e-10,
        atol=1e-10,
    )


def test_correct_offset_improves_matching_controls_do_not_systematically() -> None:
    speech, singing, offset = synthetic_centroids()
    raw = cosine_scores(speech, singing)
    corrected = cosine_scores(speech + offset, singing)
    wrong = cosine_scores(speech - offset, singing)
    rng = np.random.default_rng(29)
    random = rng.normal(size=offset.shape)
    random *= np.linalg.norm(offset) / np.linalg.norm(random)
    random_scores = cosine_scores(speech + random, singing)
    assert np.diag(corrected).mean() > np.diag(raw).mean()
    assert np.diag(corrected).mean() > np.diag(wrong).mean()
    assert np.diag(corrected).mean() > np.diag(random_scores).mean()


def test_heldout_energy_positive_and_deliberately_negative() -> None:
    speech, singing, offset = synthetic_centroids()
    delta = singing[12:] - speech[12:]
    correct = 1.0 - np.sum((delta - offset) ** 2) / np.sum(delta**2)
    misspecified = 1.0 - np.sum((delta + offset) ** 2) / np.sum(delta**2)
    assert correct > 0
    assert misspecified < 0


def test_oas_dual_is_positive_finite_and_lambda_one_identities_hold() -> None:
    speech, singing, offset = synthetic_centroids()
    z = np.vstack([speech[:12], singing[:12]])
    oas = fit_oas_dual(z)
    assert 0 <= oas.shrinkage <= 1
    assert oas.lambda0 > 0
    assert np.all(np.isfinite(oas.whiten(speech)))

    spherical = fit_oas_dual(z, 1.0)
    midpoint = z.mean(axis=0)
    whitened = cosine_scores(
        spherical.whiten(speech[12:] - midpoint),
        spherical.whiten(singing[12:] - midpoint),
    )
    centered = cosine_scores(speech[12:] - midpoint, singing[12:] - midpoint)
    assert np.array_equal(np.argsort(-whitened, axis=1), np.argsort(-centered, axis=1))
    mahal = spherical.precision_scores(speech[12:] + offset, singing[12:])
    euclid = euclidean_scores(speech[12:] + offset, singing[12:])
    assert np.array_equal(np.argsort(-mahal, axis=1), np.argsort(-euclid, axis=1))


def test_distance_correction_placement_and_normalized_cosine_redundancy() -> None:
    speech, singing, offset = synthetic_centroids()
    query = euclidean_scores(speech + offset, singing)
    symmetric = euclidean_scores(speech + offset / 2.0, singing - offset / 2.0)
    np.testing.assert_allclose(query, symmetric, rtol=1e-10, atol=1e-10)

    sn = speech / np.linalg.norm(speech, axis=1, keepdims=True)
    gn = singing / np.linalg.norm(singing, axis=1, keepdims=True)
    assert np.array_equal(
        np.argsort(-euclidean_scores(sn, gn), axis=1),
        np.argsort(-cosine_scores(speech, singing), axis=1),
    )
