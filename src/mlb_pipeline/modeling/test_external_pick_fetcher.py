from __future__ import annotations

import csv
import json
from datetime import date

from mlb_pipeline.modeling.external_pick_fetcher import (
    ExternalPickSource,
    _normalized_rows,
    _rows_from_response,
    sources_from_env_text,
    write_canonical_csv,
)


def test_sources_from_env_text_accepts_direct_specs(monkeypatch) -> None:
    monkeypatch.delenv("MLB_EXTERNAL_PICK_FETCH_HEADERS_JSON", raising=False)

    sources = sources_from_env_text(
        "outlier|csv|https://example.com/outlier.csv;"
        "edge_terminal|https://example.com/edge.json"
    )

    assert [source.platform for source in sources] == ["outlier", "edge_terminal"]
    assert sources[0].format == "csv"
    assert sources[1].format == "auto"


def test_json_response_path_extracts_pick_rows() -> None:
    source = ExternalPickSource(
        platform="edge_ai",
        url="https://example.com/feed",
        format="json",
        json_path="data.picks",
    )
    payload = {
        "data": {
            "picks": [
                {"date": "2026-07-27", "player": "A.J. Ewing", "prop": "tb", "pick": "over"}
            ]
        }
    }

    fmt, rows = _rows_from_response(source, json.dumps(payload), "application/json")

    assert fmt == "json"
    assert rows == payload["data"]["picks"]


def test_normalized_rows_write_canonical_csv(tmp_path) -> None:
    source = ExternalPickSource(
        platform="outlier",
        url="https://example.com/outlier.csv",
        format="csv",
    )
    raw_rows = [
        {
            "date": "2026-07-27",
            "player": "A.J. Ewing",
            "prop": "total bases",
            "selection": "Over",
            "book": "DK",
            "line": "1.5",
            "odds": "+120",
            "prob": "57.5%",
        },
        {
            "date": "2026-07-28",
            "player": "A.J. Ewing",
            "prop": "total bases",
            "selection": "Over",
        },
    ]

    normalized, skipped = _normalized_rows(raw_rows, source=source, game_date=date(2026, 7, 27))
    path = write_canonical_csv(tmp_path / "outlier.csv", normalized)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert skipped == 1
    assert len(rows) == 1
    assert rows[0]["platform"] == "outlier"
    assert rows[0]["market"] == "batter_total_bases"
    assert rows[0]["side"] == "over"
    assert rows[0]["bookmaker_key"] == "draftkings"
    assert rows[0]["external_probability"] == "0.575"
