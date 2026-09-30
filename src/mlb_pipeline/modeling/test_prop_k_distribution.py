from __future__ import annotations

from .prop_k_distribution import score_k_rate


def test_k_rate_scores_serialized_nonlinear_terms() -> None:
    artifact = {
        "rate_model": {
            "intercept": 0.0,
            "numeric_features": ["a", "b"],
            "numeric_means": {"a": 0.0, "b": 0.0},
            "numeric_scales": {"a": 1.0, "b": 1.0},
            "derived_terms": [
                {"name": "square:a", "kind": "square", "left": "a"},
                {"name": "interaction:a:b", "kind": "interaction", "left": "a", "right": "b"},
            ],
            "coef": {"a": 0.0, "b": 0.0, "square:a": 0.5, "interaction:a:b": 0.5},
        }
    }
    neutral = score_k_rate({"a": 0.0, "b": 0.0}, artifact)
    nonlinear = score_k_rate({"a": 2.0, "b": 1.0}, artifact)
    assert neutral == 0.5
    assert nonlinear is not None and nonlinear > 0.9
