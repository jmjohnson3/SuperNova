import pandas as pd
import pytest

from nfl_pipeline.modeling import sharp_alert_report as r


def alert(**kw):
    base = dict(season=2026, week=4, stat="receiving_yards", side="over", fd_line=45.5, fd_price=-110, ev=0.05,
                close_price=-130, sharp_close_book="pinnacle", sharp_close_line=45.5, sharp_close_over=-125,
                sharp_close_under=105, status="final", home_score=24, away_score=20, actual=60.0)
    return dict(base, **kw)


def test_grade_clv_sharp_close_and_result():
    df = r.grade(pd.DataFrame([alert(), alert(side="under", close_price=-105, actual=40.0),
                               alert(stat="total", fd_line=44.5, actual=None, sharp_close_line=44.5)]))
    over, under, total = df.to_dict("records")
    assert over["clv"] > 0 and under["clv"] < 0  # FD moved to -130 on the over; under got cheaper
    assert over["ev_at_sharp_close"] > 0 and under["ev_at_sharp_close"] < 0
    assert over["result"] == "win" and under["result"] == "win" and total["result"] == "loss"  # 44 < 44.5
    assert over["units"] == pytest.approx(100 / 110)


def test_verdict_needs_volume_then_evidence():
    small = r.summarize(r.grade(pd.DataFrame([alert()])))
    assert small["verdict"] == "keep collecting"
    rows = [alert(week=w) for w in (4, 5, 6) for _ in range(40)]
    assert r.summarize(r.grade(pd.DataFrame(rows)))["verdict"].startswith("PASS")
    bad = [alert(week=w, close_price=-105, sharp_close_over=105, sharp_close_under=-125) for w in (4, 5, 6) for _ in range(40)]
    assert r.summarize(r.grade(pd.DataFrame(bad)))["verdict"].startswith("FAIL")
