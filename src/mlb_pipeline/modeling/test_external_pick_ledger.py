from __future__ import annotations

from datetime import date

from mlb_pipeline.modeling.external_pick_ledger import (
    discover_external_pick_files,
    external_agreement_for_row,
    normalize_external_pick,
)
from mlb_pipeline.modeling.prop_shadow_selector import (
    ShadowSelectorConfig,
    _external_agreement_micro_lane,
)


def test_normalize_external_pick_accepts_common_export_aliases() -> None:
    row = normalize_external_pick({
        "source": "Outlier",
        "date": "2026-07-27",
        "player": "A.J. Ewing",
        "prop": "total bases",
        "selection": "Over",
        "book": "DK",
        "line": "1.5",
        "odds": "+120",
        "prob": "57.5%",
        "grade": "A",
    })

    assert row is not None
    assert row["platform"] == "Outlier"
    assert row["player_name_norm"] == "aj ewing"
    assert row["market"] == "batter_total_bases"
    assert row["side"] == "over"
    assert row["bookmaker_key"] == "draftkings"
    assert row["market_line"] == 1.5
    assert row["external_probability"] == 0.575


def test_discover_external_pick_files_scans_import_dir_and_dedupes(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    outlier = import_dir / "outlier_2026-07-27.csv"
    edge = import_dir / "edge_terminal_2026-07-27.csv"
    explicit = tmp_path / "manual.csv"
    import_dir.mkdir()
    outlier.write_text("date,player,prop,selection\n2026-07-27,A.J. Ewing,tb,over\n", encoding="utf-8")
    edge.write_text("date,player,prop,selection\n2026-07-27,A.J. Ewing,tb,over\n", encoding="utf-8")
    explicit.write_text("date,player,prop,selection\n2026-07-27,A.J. Ewing,tb,over\n", encoding="utf-8")

    paths = discover_external_pick_files(
        explicit_paths=[explicit, explicit],
        import_dir=import_dir,
    )

    assert set(paths) == {explicit, edge, outlier}
    assert len(paths) == 3


def test_external_agreement_prefers_exact_same_book_line() -> None:
    prediction = {
        "game_date_et": date(2026, 7, 27),
        "game_slug": "20260727-LAD-SD",
        "player_name": "A.J. Ewing",
        "team_abbr": "LAD",
        "stat": "batter_total_bases",
        "bet_side": "over",
        "bookmaker_key": "draftkings",
        "book_line": 1.5,
    }
    external = [{
        "id": 7,
        "platform": "edge_terminal",
        "game_slug": "20260727-LAD-SD",
        "player_name_norm": "aj ewing",
        "team_abbr": "LAD",
        "market": "batter_total_bases",
        "side": "over",
        "bookmaker_key": "draftkings",
        "market_line": 1.5,
        "external_grade": "A",
        "external_ev": 0.08,
        "external_probability": 0.58,
    }]

    agreement = external_agreement_for_row(prediction, external)

    assert agreement["external_agreement"] is True
    assert agreement["external_agreement_count"] == 1
    assert agreement["external_match_level"] == "exact_line_same_book"
    assert agreement["external_agreement_strength"] == 1.0


def test_external_agreement_tracks_opposite_side_disagreement() -> None:
    prediction = {
        "player_name": "A.J. Ewing",
        "stat": "batter_total_bases",
        "bet_side": "over",
        "bookmaker_key": "draftkings",
        "book_line": 1.5,
    }
    external = [{
        "platform": "edge_ai",
        "player_name_norm": "aj ewing",
        "market": "batter_total_bases",
        "side": "under",
        "bookmaker_key": "draftkings",
        "market_line": 1.5,
    }]

    agreement = external_agreement_for_row(prediction, external)

    assert agreement["external_agreement"] is False
    assert agreement["external_disagreement_count"] == 1


def test_external_agreement_micro_lane_requires_true_market_evidence() -> None:
    row = {
        "external_agreement_count": 1,
        "external_agreement_strength": 1.0,
        "external_match_level": "exact_line_same_book",
        "external_platforms": ["outlier"],
    }

    allowed, reason, blockers = _external_agreement_micro_lane(
        row,
        cfg=ShadowSelectorConfig(),
        market_evidence_confirms=True,
        synthetic_fanduel_evidence=False,
        tail_alt=False,
        line_cap_exceeded=False,
    )
    assert allowed is True
    assert reason == "external_model_agreement_micro_lane"
    assert blockers == []

    blocked, _reason, blockers = _external_agreement_micro_lane(
        row,
        cfg=ShadowSelectorConfig(),
        market_evidence_confirms=False,
        synthetic_fanduel_evidence=False,
        tail_alt=False,
        line_cap_exceeded=False,
    )
    assert blocked is False
    assert "external_agreement_needs_true_pair_market" in blockers
