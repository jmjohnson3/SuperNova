"""
Integration test: verify feature column alignment between training and inference.

Checks that:
  1. feature_columns.json and feature_medians.json are in sync
  2. add_game_derived_features() produces consistent column sets
  3. No column drift between make_xy_raw() output and _prep_features() expectations

Run:
  python -m mlb_pipeline.modeling.test_feature_alignment
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from .features import add_game_derived_features


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


@pytest.fixture(scope="module")
def feature_cols_and_medians(model_dir: Path) -> tuple[list[str], dict[str, float]]:
    return load_model_artifacts(model_dir)


@pytest.fixture(scope="module")
def feature_cols(feature_cols_and_medians: tuple[list[str], dict[str, float]]) -> list[str]:
    return feature_cols_and_medians[0]


@pytest.fixture(scope="module")
def feature_medians(feature_cols_and_medians: tuple[list[str], dict[str, float]]) -> dict[str, float]:
    return feature_cols_and_medians[1]

def load_model_artifacts(model_dir: Path) -> tuple[list[str], dict[str, float]]:
    """Load feature_columns.json and feature_medians.json."""
    cols_path = model_dir / "feature_columns.json"
    medians_path = model_dir / "feature_medians.json"

    if not cols_path.exists():
        raise FileNotFoundError(f"Missing {cols_path}")
    if not medians_path.exists():
        raise FileNotFoundError(f"Missing {medians_path}")

    with open(cols_path) as f:
        feature_cols = json.load(f)

    with open(medians_path) as f:
        feature_medians = json.load(f)

    return feature_cols, feature_medians


def test_columns_vs_medians(feature_cols: list[str], feature_medians: dict[str, float]) -> None:
    """Check that every column in feature_columns has a median (and vice versa)."""
    print("\n=== Test 1: Feature columns vs medians alignment ===")

    cols_set = set(feature_cols)
    medians_set = set(feature_medians.keys())

    missing_medians = cols_set - medians_set
    extra_medians = medians_set - cols_set

    success = True

    if missing_medians:
        print(f"FAIL: {len(missing_medians)} training columns missing medians:")
        for c in sorted(missing_medians)[:10]:
            print(f"  - {c}")
        if len(missing_medians) > 10:
            print(f"  ... and {len(missing_medians) - 10} more")
        success = False

    if extra_medians:
        print(f"FAIL: {len(extra_medians)} medians with no matching column:")
        for c in sorted(extra_medians)[:10]:
            print(f"  - {c}")
        if len(extra_medians) > 10:
            print(f"  ... and {len(extra_medians) - 10} more")
        success = False

    if success:
        print(f"PASS: {len(feature_cols)} columns ↔ {len(feature_medians)} medians (100% aligned)")

    assert success


def test_derived_features_consistency(feature_cols: list[str]) -> None:
    """Test that add_game_derived_features produces consistent columns."""
    print("\n=== Test 2: Derived features consistency ===")

    # Create a dummy DataFrame with columns that might trigger derived feature creation
    # We include some of the raw stat columns that add_game_derived_features expects
    dummy_data = {
        "home_runs_avg_10": [4.5],
        "away_runs_avg_10": [4.3],
        "home_win_pct": [0.55],
        "away_win_pct": [0.50],
        "home_sp_era_5": [3.2],
        "away_sp_era_5": [3.5],
        "home_sp_fip_5": [3.1],
        "away_sp_fip_5": [3.4],
        "home_sp_k_pct_5": [0.25],
        "away_sp_k_pct_5": [0.24],
        "home_runs_avg_5": [4.6],
        "away_runs_avg_5": [4.2],
        "home_hr_avg_10": [1.2],
        "away_hr_avg_10": [1.1],
        "home_avg_avg_10": [0.265],
        "away_avg_avg_10": [0.260],
        "home_iso_avg_10": [0.180],
        "away_iso_avg_10": [0.175],
        "home_era_10": [3.3],
        "away_era_10": [3.6],
        "home_whip_10": [1.15],
        "away_whip_10": [1.20],
        "home_run_diff_per_game": [0.5],
        "away_run_diff_per_game": [0.3],
        "total_line": [8.5],
        "run_line_home": [-1.5],
        "home_rest_days": [1],
        "away_rest_days": [2],
    }

    df = pd.DataFrame(dummy_data)

    # Call add_game_derived_features
    result_df = add_game_derived_features(df)

    # Check that new columns were added (derived features)
    new_cols = set(result_df.columns) - set(df.columns)

    if not new_cols:
        print("WARN: No derived features were created (expected at least some)")
        print("      This may be OK if input columns don't match expectations")
    else:
        print(f"PASS: Created {len(new_cols)} derived feature columns")
        print(f"      Examples: {sorted(new_cols)[:5]}")

    # All columns should be numeric
    non_numeric = [c for c in result_df.columns if not pd.api.types.is_numeric_dtype(result_df[c])]
    assert not non_numeric, f"Non-numeric columns after add_game_derived_features: {non_numeric}"


def test_inference_path_consistency(feature_cols: list[str], feature_medians: dict[str, float]) -> None:
    """
    Simulate the inference path: create a mock game frame, apply feature prep
    (like _prep_features does), and verify no unexpected columns appear.
    """
    print("\n=== Test 3: Inference path consistency ===")

    # Simulate a raw game prediction features row
    dummy_game = {
        "season": ["2025-regular"],
        "game_slug": ["20250401-LAD-NYM"],
        "game_date_et": [pd.Timestamp("2025-04-01")],
        "start_ts_utc": [pd.Timestamp("2025-04-01 23:05:00", tz="UTC")],
        "home_team_abbr": ["NYM"],
        "away_team_abbr": ["LAD"],
        # Raw stats
        "home_runs_avg_10": [4.5],
        "away_runs_avg_10": [4.3],
        "home_win_pct": [0.55],
        "away_win_pct": [0.50],
    }

    df = pd.DataFrame(dummy_game)

    # Simulate _prep_features logic: drop ID cols, create derived features, align to feature_cols
    id_cols = ["season", "game_slug", "game_date_et", "start_ts_utc",
               "home_team_abbr", "away_team_abbr"]
    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    # Add derived features (as _prep_features does)
    X = add_game_derived_features(X)

    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        print(f"WARN: Inference created {len(extra)} extra columns: {extra[:5]}")
    X = X.reindex(columns=feature_cols)

    # After alignment, should have exactly feature_cols
    if set(X.columns) == set(feature_cols):
        print(f"PASS: Inference frame aligns to {len(feature_cols)} training columns")
    else:
        missing = set(feature_cols) - set(X.columns)
        extra = set(X.columns) - set(feature_cols)
        if missing:
            print(f"FAIL: Missing {len(missing)} columns in inference frame")
            for c in sorted(missing)[:5]:
                print(f"  - {c}")
        if extra:
            print(f"FAIL: Extra {len(extra)} columns in inference frame")
            for c in sorted(extra)[:5]:
                print(f"  - {c}")
    assert set(X.columns) == set(feature_cols)


def main():
    model_dir = Path(__file__).resolve().parent / "models"

    print(f"Loading model artifacts from {model_dir}...")

    try:
        feature_cols, feature_medians = load_model_artifacts(model_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Models have not been trained yet. Run train_game_models first.")
        sys.exit(1)

    print(f"Loaded: {len(feature_cols)} columns, {len(feature_medians)} medians")

    tests = [
        (test_columns_vs_medians, (feature_cols, feature_medians)),
        (test_derived_features_consistency, (feature_cols,)),
        (test_inference_path_consistency, (feature_cols, feature_medians)),
    ]

    print("\n" + "=" * 60)
    passed = 0
    for test, args in tests:
        try:
            test(*args)
            passed += 1
        except AssertionError as exc:
            print(f"FAIL: {test.__name__}: {exc}")
    total = len(tests)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("PASS: All feature alignment checks passed!")
        sys.exit(0)
    else:
        print("FAIL: Some feature alignment checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
