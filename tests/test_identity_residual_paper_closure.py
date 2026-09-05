from __future__ import annotations

from legacy.probing.run_identity_residual_paper_closure import (
    BASELINE_CONDITIONS,
    HEADLINE_MODELS,
    cosine_scores,
    fit_diagonal_whitener,
    remove_subspace,
    summarize_baseline,
)


def baseline_rows(r1_shift: float = 0.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in HEADLINE_MODELS:
        for condition in BASELINE_CONDITIONS:
            for seed in range(20):
                rows.append(
                    {
                        "model": model,
                        "condition_id": condition,
                        "split_seed": seed,
                        "R1": 0.8 + r1_shift,
                        "EER": 0.1,
                    }
                )
    return rows


def expected_rows() -> dict[tuple[str, str], dict[str, float]]:
    return {
        (model, condition): {"R1": 0.8, "EER": 0.1}
        for model in HEADLINE_MODELS
        for condition in BASELINE_CONDITIONS
    }


def test_baseline_summary_passes_within_half_point() -> None:
    summary = summarize_baseline(baseline_rows(r1_shift=0.0049), expected_rows())
    assert summary
    assert all(row["status"] == "PASS" for row in summary)


def test_baseline_summary_requires_audit_beyond_half_point() -> None:
    summary = summarize_baseline(baseline_rows(r1_shift=0.0051), expected_rows())
    assert all(row["status"] == "FAIL_AUDIT_REQUIRED" for row in summary)


def test_diagonal_whitening_equal_variance_limit_matches_centered_cosine() -> None:
    import numpy as np

    rng = np.random.default_rng(715)
    z_train = np.vstack([np.eye(8), -np.eye(8)])
    scale, _epsilon, _stats, _transform_hash = fit_diagonal_whitener(z_train)
    assert np.allclose(scale, scale[0])
    query = rng.normal(size=(12, 8))
    gallery = rng.normal(size=(12, 8))
    midpoint = rng.normal(size=8)
    raw = cosine_scores(query - midpoint, gallery - midpoint)
    whitened = cosine_scores((query - midpoint) / scale, (gallery - midpoint) / scale)
    assert np.array_equal(np.argsort(-raw, axis=1), np.argsort(-whitened, axis=1))


def test_abtt_projection_is_idempotent_and_removes_k_dimensions() -> None:
    import numpy as np

    rng = np.random.default_rng(716)
    train = rng.normal(size=(30, 12))
    train -= train.mean(axis=0)
    _u, _s, vt = np.linalg.svd(train, full_matrices=False)
    basis = vt[:4].T
    projected = remove_subspace(train, basis)
    np.testing.assert_allclose(remove_subspace(projected, basis), projected, atol=1e-10)
    np.testing.assert_allclose(projected @ basis, 0.0, atol=1e-10)
    assert np.linalg.matrix_rank(train) - np.linalg.matrix_rank(projected) == 4
