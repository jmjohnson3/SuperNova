"""The Predictions tab's reader: stored forecasts beside the latest line, with the sign conventions checked."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sports_sniper import predictions, predictions_view
from sports_sniper.test_app import dashboard, env  # noqa: F401

GAMES, PROPS, RECORD = predictions.games, predictions.props, predictions.ncaaf_record     # the shared setup blanks these
NOW = datetime.now(timezone.utc).replace(microsecond=0)
TIP = NOW + timedelta(hours=2)
NAMES = {"IND": "Indiana Pacers", "WAS": "Washington Wizards", "BOS": "Boston Celtics", "TOR": "Toronto Raptors"}


def event(home, away, spread, total, book="fanduel"):
    return dict(home_team=home, away_team=away, bookmakers=[
        dict(key="pinnacle", markets=[dict(key="spreads", outcomes=[dict(name=home, point=spread - 1), dict(name=away, point=1 - spread)])]),
        dict(key=book, markets=[dict(key="spreads", outcomes=[dict(name=away, point=-spread), dict(name=home, point=spread)]),
                                dict(key="totals", outcomes=[dict(name="Over", point=total), dict(name="Under", point=total)])])])


TABLES = {
    "ncaaf.projections": (["start_date", "away_team", "home_team", "pred_margin", "pred_total", "market_provider", "market_spread",
                           "market_total", "generated_at_utc"],
                          [(TIP, "Buffalo", "Ball State", -6.08, 54.42, "DraftKings", 1.5, 50.5, NOW - timedelta(hours=8))]),
    "nbaprops.game_projections": (["tip_utc", "away_team", "home_team", "pred_margin", "pred_total", "home_out", "away_out", "run_at_utc",
                                   "model_version"],
                                  [(TIP, "WAS", "IND", 0.1, 223.0, "Tyrese Haliburton", None, NOW - timedelta(hours=1), "nba-v2"),
                                   (TIP, "BOS", "TOR", 2.7, 222.9, None, None, NOW - timedelta(hours=1), "nba-v2")]),
    "bets.sharp_multi_quotes": (["observed_at_utc", "payload"],
                                [(NOW - timedelta(minutes=5), event("Indiana Pacers", "Washington Wizards", -3.5, 227.5))]),
    "bets.nfl_game_predictions": (["game_id", "away_team_abbr", "home_team_abbr", "market", "side", "book", "line", "predicted_home_margin",
                                   "predicted_total_points", "start", "updated_at_utc", "model_version"],
                                  [("g1", "WAS", "NYG", "spread", "away", "fanduel", 1.5, 1.5, 41.5, str(TIP), NOW, "nfl-1"),
                                   ("g1", "WAS", "NYG", "total", "over", "fanduel", 41.5, 1.5, 41.5, str(TIP), NOW, "nfl-1"),
                                   ("old", "BUF", "MIN", "spread", "home", "fanduel", 3.0, -3.0, 45.5, str(NOW - timedelta(days=1)), NOW, "nfl-1")]),
    "nbaprops.projections": (["tip_utc", "player_name", "team", "opponent", "stat", "mu", "disp_a", "disp_b", "run_at_utc", "model_version"],
                             [(TIP, "Derrick White", "BOS", "TOR", "pts", 12.4, 1.0, 0.0, NOW - timedelta(hours=1), "nba-v2"),
                              (TIP, "Derrick White", "BOS", "TOR", "ast", 4.1, 1.0, 0.0, NOW - timedelta(hours=1), "nba-v2")]),
    "bets.nfl_player_prop_predictions": (["player", "team_abbr", "opponent_abbr", "stat", "projection", "raw", "projection_p10",
                                          "projection_p90", "line", "side", "price", "book", "raw_over_probability",
                                          "market_no_vig_probability", "start", "updated_at_utc", "model_version"],
                                         [("Isaiah Likely", "NYG", "WAS", "receptions", 4.83, "5.44", 2.0, 8.0, 4.5, "under", -125, "fanduel",
                                           0.53, 0.52, str(TIP), NOW, "nfl-1"),
                                          ("Malik Nabers", "NYG", "WAS", "receiving_tds", 0.25, None, 0.0, 1.0, None, None, None, None, None,
                                           None, str(TIP), NOW, "nfl-1")]),
}


class Cursor:
    def __init__(self, tables):
        self.tables, self.rows, self.description = tables, [], []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        if "to_regclass" in sql:
            self.rows = [(params[0] if params[0] in self.tables else None,)]
            return
        table = next(t for t in self.tables if f"FROM {t} " in " ".join(sql.split()) + " ")
        columns, self.rows = self.tables[table]
        self.description = [(c,) for c in columns]

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, tables):
        self.tables = tables

    def set_session(self, readonly=True):
        assert readonly

    def cursor(self):
        return Cursor(self.tables)

    def close(self):
        pass


def connect(tables=TABLES):
    return lambda dsn: Connection(tables)


def prices():
    rows = []
    for book, line, over, under, minutes in (("draftkings", 13.5, -110, -110, 1), ("fanduel", 14.5, -115, -111, 5), ("fanduel", 15.5, 100, -130, 30)):
        for side, price in (("over", over), ("under", under)):
            rows.append(dict(sport="nba", player_key="derrick white", market="player_points", book=book, line=line, side=side, price=price,
                             observed_at=pd.Timestamp(NOW - timedelta(minutes=minutes)), commence_time=pd.Timestamp(TIP)))
    return pd.DataFrame(rows)


def test_games_come_from_all_three_sports_with_the_home_spread_convention():
    found = GAMES("dsn", NOW, connect(), NAMES).set_index("home")
    assert sorted(found.sport) == ["nba", "nba", "ncaaf", "nfl"] and "MIN" not in found.index                # a finished game is left out
    ball = found.loc["Ball State"]
    assert ball.spread == 1.5 and ball.spread_gap == pytest.approx(-4.58) and ball.total_gap == pytest.approx(3.92) and ball.book == "draftkings"
    pacers = found.loc["Indiana Pacers"]
    assert (pacers.spread, pacers.total, pacers.book) == (-3.5, 227.5, "fanduel")                             # the bet book before Pinnacle
    assert pacers.spread_gap == pytest.approx(-3.4) and pacers.total_gap == pytest.approx(-4.5) and "Haliburton" in pacers.note
    assert pd.isna(found.loc["Toronto Raptors"].spread) and pd.isna(found.loc["Toronto Raptors"].spread_gap)   # no stored line yet
    giants = found.loc["NYG"]
    assert giants.spread == -1.5 and giants.spread_gap == 0 and giants.total_gap == 0                          # away +1.5 is home -1.5
    table = predictions_view.games_table(found.reset_index()).set_index("Game")
    wizards = table.loc["Washington Wizards at Indiana Pacers"]
    assert wizards["Line"] == "Pacers -3.5" and wizards["Model spread"] == "Pacers -0.1"
    assert wizards["Spread lean"] == "Wizards +3.5 (by 3.4)" and wizards["Total lean"] == "Under 227.5 (by 4.5)"
    buffalo = table.loc["Buffalo at Ball State"]
    assert buffalo["Line"] == "Buffalo -1.5" and buffalo["Model spread"] == "Buffalo -6.1"
    assert buffalo["Spread lean"] == "Buffalo -1.5 (by 4.6)" and buffalo["Total lean"] == "Over 50.5 (by 3.9)"
    assert table.loc["WAS at NYG", "Spread lean"] == "-" and table.loc["Boston Celtics at Toronto Raptors", "Line"] == "-"


def test_props_take_the_newest_line_from_the_first_bet_book():
    pytest.importorskip("nba_props.model")
    found = PROPS("dsn", NOW, prices(), connect())
    points = found[(found.player == "Derrick White") & (found.stat == "Points")].iloc[0]
    assert (points.line, points.book, points.over_price, points.under_price) == (14.5, "fanduel", -115, -111)   # FanDuel's latest, not its older 15.5
    assert points.gap == pytest.approx(-2.1) and 0 < points.model_over < 0.5 and points.market_over == pytest.approx(0.504, abs=0.002)
    assert points.low < 12.4 < points.high
    assists = found[(found.player == "Derrick White") & (found.stat == "Assists")].iloc[0]
    assert pd.isna(assists.line) and pd.isna(assists.model_over) and pd.isna(assists.gap)                    # no line stored: projection only
    likely = found[found.player == "Isaiah Likely"].iloc[0]
    assert likely.stat == "Receptions" and likely.raw == 5.44 and likely.under_price == -125 and pd.isna(likely.over_price)
    assert likely.model_over == 0.53 and likely.market_over == pytest.approx(0.48)                            # the stored market chance was the under's
    assert found[found.player == "Malik Nabers"].stat.iloc[0] == "Receiving TDs"
    table = predictions_view.props_table(found[found.player == "Derrick White"])
    assert table.Line.tolist() == ["-", "14.5"] and table.Gap.tolist() == ["-", "-2.1"]


def test_nothing_is_read_when_the_tables_are_not_there():
    empty = connect({})
    assert GAMES("dsn", NOW, empty, NAMES).empty and PROPS("dsn", NOW, None, empty).empty
    assert RECORD("dsn", empty) is None and predictions.prop_lines(None) == {}

    def refused(dsn):
        raise OSError("no database")
    assert GAMES("dsn", NOW, refused, NAMES).empty


def test_the_record_compares_the_model_and_the_line_on_finished_games():
    done = pd.DataFrame([dict(pred_margin=7.0, pred_total=50.0, spread=-3.0, total=45.0, home_points=30, away_points=20),    # home covered, over
                         dict(pred_margin=7.0, pred_total=50.0, spread=-3.0, total=45.0, home_points=21, away_points=20),    # home did not, under
                         dict(pred_margin=3.0, pred_total=45.0, spread=-3.0, total=45.0, home_points=24, away_points=21),    # no lean, a push
                         dict(pred_margin=None, pred_total=45.0, spread=-3.0, total=45.0, home_points=24, away_points=21)])
    record = predictions.score_games(done)
    assert record["games"] == 3 and record["spread_record"] == (1, 1) and record["total_record"] == (1, 1)
    assert record["margin_model"] == pytest.approx((3 + 6 + 0) / 3) and record["margin_line"] == pytest.approx((7 + 2 + 0) / 3)
    assert predictions.score_games(done.iloc[0:0]) is None
    assert "measured here on 3 finished games" in predictions_view.record_lines(record)[0]
    assert "no finished games" in predictions_view.record_lines(None)[0]


def test_dashboard_predictions_tab(env, monkeypatch):  # noqa: F811
    pytest.importorskip("nba_props.model")
    games = GAMES("dsn", NOW, connect(), NAMES)
    props = PROPS("dsn", NOW, prices(), connect())
    app = dashboard().run()
    app.button[[b.label for b in app.button].index("Scan sample data")].click().run()
    assert not app.exception and any("No stored forecasts for coming games" in i.value for i in app.info)     # nothing stored: says so
    monkeypatch.setattr(predictions, "games", lambda dsn, now, connect=None, names=None: games)
    monkeypatch.setattr(predictions, "props", lambda dsn, now, prices=None, connect=None: props)
    import streamlit as st
    st.cache_data.clear()
    app.run()
    assert not app.exception and any(h.value == "Predictions" for h in app.subheader)
    shown = next(d.value for d in app.dataframe if "Spread lean" in d.value.columns)
    assert len(shown) == 4 and set(shown.Sport) == {"NBA", "NFL", "NCAAF"}
    table = next(d.value for d in app.dataframe if "Model: over" in d.value.columns)
    assert table.Player.tolist() == ["Derrick White", "Isaiah Likely"]                    # only those with a line, biggest disagreement first
    next(t for t in app.toggle if t.label == "Lines only").set_value(False).run()
    assert not app.exception and len(next(d.value for d in app.dataframe if "Model: over" in d.value.columns)) == 4
    next(m for m in app.multiselect if m.label == "Sports").set_value(["nfl"]).run()
    assert not app.exception and next(d.value for d in app.dataframe if "Spread lean" in d.value.columns).Game.tolist() == ["WAS at NYG"]
