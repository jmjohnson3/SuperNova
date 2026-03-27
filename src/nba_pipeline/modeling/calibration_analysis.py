"""
Calibration analysis for the game spread model.

Re-runs the walk-forward using saved Optuna hyperparameters (no new tuning),
collects per-prediction detail, and prints four analysis tables:

  1. Calibration by edge bucket (is 7pt edge worth more than 5pt?)
  2. ATS by game context (B2B, rest, home/away, spread size)
  3. Rolling 30-game ATS trend (chronological)
  4. Live CLV breakdown (from bets.game_predictions)
  5. Prediction magnitude vs actual outcome

Usage:
    python -m nba_pipeline.modeling.calibration_analysis
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

from .train_game_models import (
    TrainConfig,
    load_training_frame,
    make_xy_raw,
    fit_fill_stats,
    apply_fill,
    walk_forward_folds,
    temporal_eval_split,
    build_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TrainConfig()

    # ── Load training data ──────────────────────────────────────────────────
    conn = psycopg2.connect(cfg.pg_dsn)
    df = load_training_frame(conn)
    conn.close()
    log.info(
        "Loaded %d rows  %s → %s",
        len(df),
        df["game_date_et"].min().date(),
        df["game_date_et"].max().date(),
    )

    # ── Build feature matrix ────────────────────────────────────────────────
    X_raw, y_spread, _ = make_xy_raw(df)

    # ── Load saved Optuna params and feature schema ─────────────────────────
    params_path = MODEL_DIR / "optuna_best_params.json"
    feat_path   = MODEL_DIR / "feature_columns.json"

    if not params_path.exists():
        log.error("optuna_best_params.json not found — run train_game_models first")
        sys.exit(1)

    saved_params  = json.loads(params_path.read_text(encoding="utf-8"))
    spread_params = saved_params.get("spread", {})

    all_feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_cols  = [c for c in all_feat_cols if c in X_raw.columns]
    log.info("Feature schema: %d cols loaded, %d present in X_raw", len(all_feat_cols), len(feature_cols))

    # ── Walk-forward with fixed params ──────────────────────────────────────
    folds = walk_forward_folds(df, min_train_days=cfg.min_train_days, test_window_days=cfg.test_window_days, step_days=cfg.step_days)
    log.info("Running %d walk-forward folds (no Optuna re-tuning)…", len(folds))

    records = []

    for i, (train_end, test_end) in enumerate(folds, start=1):
        train_mask = df["game_date_et"] < train_end
        test_mask  = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())
        if n_train < 50 or n_test == 0:
            continue

        X_tr_raw = X_raw.loc[train_mask]
        X_te_raw = X_raw.loc[test_mask]

        medians = fit_fill_stats(X_tr_raw)
        X_tr    = apply_fill(X_tr_raw, medians, feature_cols)
        X_te    = apply_fill(X_te_raw, medians, feature_cols)

        y_tr = y_spread.loc[train_mask]
        y_te = y_spread.loc[test_mask]

        # Temporal eval split for early stopping (same as production training)
        fit_rel, eval_rel = temporal_eval_split(df.loc[train_mask, "game_date_et"])

        model = build_model(cfg, params_override=spread_params, n_estimators=2000, use_early_stopping=True)
        model.fit(
            X_tr.iloc[fit_rel], y_tr.iloc[fit_rel],
            eval_set=[(X_tr.iloc[eval_rel], y_tr.iloc[eval_rel])],
            verbose=False,
        )

        preds = model.predict(X_te)

        # Metadata for each test row
        mkt_spread = df.loc[test_mask, "market_spread_home"].to_numpy(dtype=float)

        for j, idx in enumerate(df.index[test_mask]):
            mkt = mkt_spread[j]
            if np.isnan(mkt):
                continue

            pred   = float(preds[j])
            actual = float(y_te.iloc[j])
            edge   = pred + mkt          # >0 = model favors home
            covers = actual > -mkt       # True if home covered
            bet_home = edge > 0
            bet_won  = bet_home == covers

            row = df.loc[idx]
            records.append({
                "fold":           i,
                "game_date_et":   row["game_date_et"],
                "home_team":      row.get("home_team_abbr", ""),
                "away_team":      row.get("away_team_abbr", ""),
                "pred_margin":    pred,
                "actual_margin":  actual,
                "market_spread":  mkt,
                "edge":           edge,
                "abs_edge":       abs(edge),
                "bet_side":       "home" if bet_home else "away",
                "covers":         covers,
                "bet_won":        bet_won,
                "home_is_b2b":    int(row.get("home_is_b2b", 0) or 0),
                "away_is_b2b":    int(row.get("away_is_b2b", 0) or 0),
                "home_rest_days": float(row.get("home_rest_days") or 2.0),
                "away_rest_days": float(row.get("away_rest_days") or 2.0),
            })

        if i % 10 == 0:
            log.info("  Fold %d/%d done", i, len(folds))

    pred_df = pd.DataFrame(records)
    log.info("Collected %d OOF predictions with market data", len(pred_df))

    # ── Print all tables ────────────────────────────────────────────────────
    _print_calibration(pred_df)
    _print_context_breakdown(pred_df)
    _print_rolling_ats(pred_df)
    _print_clv_analysis(cfg.pg_dsn)
    _print_prediction_magnitude(pred_df)


# ---------------------------------------------------------------------------
# Table 1: Calibration by edge bucket
# ---------------------------------------------------------------------------

def _print_calibration(df: pd.DataFrame) -> None:
    bins   = [0, 3, 5, 6, 7, 8, 9, 10, 12, 99]
    labels = ["0-3", "3-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10-12", "12+"]
    df = df.copy()
    df["bucket"] = pd.cut(df["abs_edge"], bins=bins, labels=labels)

    print("\n" + "="*76)
    print("TABLE 1: SPREAD CALIBRATION BY EDGE BUCKET  (walk-forward OOF, all seasons)")
    print("="*76)
    print(f"  {'Edge':>8}  {'N':>5}  {'Wins':>5}  {'Win%':>6}  {'ROI':>7}  {'p-val':>7}  {'sig':3}")
    print("  " + "-"*68)

    above5_n = above5_w = 0
    for lbl in labels:
        sub = df[df["bucket"] == lbl]
        n   = len(sub)
        if n < 3:
            continue
        w   = int(sub["bet_won"].sum())
        pct = w / n
        roi = (w * 100 - (n - w) * 110) / (n * 110)
        # one-sided binomial test: H1 = win% > 50%
        p   = float(stats.binomtest(w, n, 0.5, alternative="greater").pvalue)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {lbl:>8}  {n:>5}  {w:>5}  {pct:>5.1%}  {roi:>+7.1%}  {p:>7.4f}  {sig}")
        if lbl not in ("0-3", "3-5"):
            above5_n += n
            above5_w += w

    print("  " + "-"*68)
    if above5_n > 0:
        pct = above5_w / above5_n
        roi = (above5_w * 100 - (above5_n - above5_w) * 110) / (above5_n * 110)
        p   = float(stats.binomtest(above5_w, above5_n, 0.5, alternative="greater").pvalue)
        print(f"  {'>=5pt':>8}  {above5_n:>5}  {above5_w:>5}  {pct:>5.1%}  {roi:>+7.1%}  {p:>7.4f}")

    total_n = len(df)
    total_w = int(df["bet_won"].sum())
    print(f"\n  All {total_n} predictions (no edge filter): {total_w/total_n:.1%} ATS")
    print(f"  Note: win% is vs 50% (pick-em); -4.5% ROI breakeven at standard -110 juice")


# ---------------------------------------------------------------------------
# Table 2: ATS by game context
# ---------------------------------------------------------------------------

def _print_context_breakdown(df: pd.DataFrame) -> None:
    bets = df[df["abs_edge"] >= 5.0].copy()

    # Derived rest-asymmetry fields
    bets["bet_team_rest"] = np.where(
        bets["bet_side"] == "home", bets["home_rest_days"], bets["away_rest_days"]
    )
    bets["opp_team_rest"] = np.where(
        bets["bet_side"] == "home", bets["away_rest_days"], bets["home_rest_days"]
    )
    bets["bet_team_b2b"] = np.where(
        bets["bet_side"] == "home", bets["home_is_b2b"], bets["away_is_b2b"]
    )
    bets["opp_team_b2b"] = np.where(
        bets["bet_side"] == "home", bets["away_is_b2b"], bets["home_is_b2b"]
    )

    print("\n" + "="*76)
    print("TABLE 2: ATS BY GAME CONTEXT  (edge >= 5pt bets only)")
    print("="*76)
    print(f"  {'Context':36s}  {'N':>5}  {'Win%':>6}  {'ROI':>7}")
    print("  " + "-"*62)

    def _row(label: str, mask) -> None:
        sub = bets[mask]
        n   = len(sub)
        if n < 3:
            print(f"  {label:36s}  {'<3 samples':>14}")
            return
        w   = int(sub["bet_won"].sum())
        pct = w / n
        roi = (w * 100 - (n - w) * 110) / (n * 110)
        print(f"  {label:36s}  {n:>5}  {pct:>5.1%}  {roi:>+7.1%}")

    _row("ALL bets >=5pt",              pd.Series([True] * len(bets), index=bets.index))
    print()
    _row("  Bet on home team",          bets["bet_side"] == "home")
    _row("  Bet on away team",          bets["bet_side"] == "away")
    print()
    _row("  Market spread |mkt| <= 3",  bets["market_spread"].abs() <= 3)
    _row("  Market spread 3-6",         bets["market_spread"].abs().between(3, 6))
    _row("  Market spread 6-10",        bets["market_spread"].abs().between(6, 10))
    _row("  Market spread > 10",        bets["market_spread"].abs() > 10)
    print()
    _row("  Our team on B2B",           bets["bet_team_b2b"] == 1)
    _row("  Our team NOT on B2B",       bets["bet_team_b2b"] == 0)
    _row("  Opponent on B2B",           bets["opp_team_b2b"] == 1)
    _row("  Rest advantage >=2 days",   bets["bet_team_rest"] - bets["opp_team_rest"] >= 2)
    _row("  Rest disadvantage >=2",     bets["opp_team_rest"] - bets["bet_team_rest"] >= 2)
    _row("  Equal rest",                (bets["bet_team_rest"] - bets["opp_team_rest"]).abs() < 2)


# ---------------------------------------------------------------------------
# Table 3: Rolling 30-game ATS
# ---------------------------------------------------------------------------

def _print_rolling_ats(df: pd.DataFrame) -> None:
    bets = df[df["abs_edge"] >= 5.0].sort_values("game_date_et").reset_index(drop=True)

    print("\n" + "="*76)
    print("TABLE 3: ROLLING 30-GAME ATS (edge >= 5pt, chronological)")
    print("="*76)

    W = 30
    if len(bets) < W:
        print(f"  Only {len(bets)} bets — need {W} for rolling window")
        return

    rows_out = []
    for start in range(0, len(bets), W):
        chunk = bets.iloc[start : start + W]
        if len(chunk) < 10:
            continue
        n   = len(chunk)
        w   = int(chunk["bet_won"].sum())
        pct = w / n
        d0  = chunk["game_date_et"].min().strftime("%b %d")
        d1  = chunk["game_date_et"].max().strftime("%b %d, %Y")
        bar = "#" * round(pct * 20)
        rows_out.append((f"{d0} – {d1}", n, w, pct, bar))

    for date_rng, n, w, pct, bar in rows_out:
        print(f"  {date_rng:<28}  {w:>2}/{n:<2}  {pct:>5.1%}  {bar}")

    # Cumulative
    all_n = len(bets)
    all_w = int(bets["bet_won"].sum())
    print(f"\n  Cumulative: {all_w}/{all_n} = {all_w/all_n:.1%}")


# ---------------------------------------------------------------------------
# Table 4: Live CLV analysis
# ---------------------------------------------------------------------------

def _print_clv_analysis(pg_dsn: str) -> None:
    print("\n" + "="*76)
    print("TABLE 4: LIVE CLV ANALYSIS (bets.game_predictions)")
    print("="*76)

    try:
        conn = psycopg2.connect(pg_dsn)
        cur  = conn.cursor()
        cur.execute("""
            SELECT game_date_et, edge_spread, spread_bet_side,
                   spread_covered, clv_spread,
                   market_spread_home, closing_spread_home
            FROM bets.game_predictions
            WHERE spread_bet_side IS NOT NULL
              AND edge_spread IS NOT NULL
            ORDER BY game_date_et
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"  (unavailable: {e})")
        return

    if not rows:
        print("  No live bets found.")
        return

    live = pd.DataFrame(rows, columns=[
        "game_date_et", "edge_spread", "spread_bet_side",
        "spread_covered", "clv_spread", "market_spread_home", "closing_spread_home"
    ])
    live["abs_edge"] = live["edge_spread"].abs()

    print(f"\n  Bets logged:  {len(live)}  ({live['game_date_et'].min()} – {live['game_date_et'].max()})")
    graded = live.dropna(subset=["spread_covered"])
    print(f"  Graded bets:  {len(graded)}")
    if len(graded) > 0:
        w = int(graded["spread_covered"].sum())
        print(f"  ATS record:   {w}/{len(graded)} = {w/len(graded):.1%}")

    clv_data = live.dropna(subset=["clv_spread"])
    if len(clv_data) > 0:
        n_pos = int((clv_data["clv_spread"] > 0).sum())
        print(f"\n  CLV computed: {len(clv_data)} bets")
        print(f"  Mean CLV:     {clv_data['clv_spread'].mean():+.3f} pts")
        print(f"  Median CLV:   {clv_data['clv_spread'].median():+.3f} pts")
        print(f"  Beat close:   {n_pos}/{len(clv_data)} = {n_pos/len(clv_data):.1%}")

        # By edge bucket
        print(f"\n  {'Edge':>6}  {'N':>4}  {'CLV mean':>9}  {'Beat%':>6}  {'ATS':>6}")
        print("  " + "-"*42)
        clv_data2 = clv_data.copy()
        clv_data2["bucket"] = pd.cut(clv_data2["abs_edge"], bins=[0,5,7,9,99], labels=["<5","5-7","7-9","9+"])
        for lbl in ["<5", "5-7", "7-9", "9+"]:
            sub = clv_data2[clv_data2["bucket"] == lbl]
            if len(sub) < 2:
                continue
            mean_clv  = sub["clv_spread"].mean()
            beat_pct  = (sub["clv_spread"] > 0).mean()
            g = sub.dropna(subset=["spread_covered"])
            ats_str = f"{int(g['spread_covered'].sum())}/{len(g)}" if len(g) >= 2 else "n/a"
            print(f"  {lbl:>6}  {len(sub):>4}  {mean_clv:>+9.3f}  {beat_pct:>5.1%}  {ats_str:>6}")
    else:
        print("\n  No CLV data available (need 6:30 PM close-only crawl to be running).")


# ---------------------------------------------------------------------------
# Table 5: Prediction magnitude vs actual outcome
# ---------------------------------------------------------------------------

def _print_prediction_magnitude(df: pd.DataFrame) -> None:
    print("\n" + "="*76)
    print("TABLE 5: PREDICTION MAGNITUDE vs ACTUAL OUTCOME")
    print("="*76)
    print(f"  {'|Pred margin|':>14}  {'N':>5}  {'Avg actual':>11}  {'Dir correct':>12}  {'Std actual':>11}")
    print("  " + "-"*60)

    bins   = [0, 3, 5, 7, 9, 12, 20, 99]
    labels = ["0-3", "3-5", "5-7", "7-9", "9-12", "12-20", "20+"]
    df2 = df.copy()
    df2["mag_bucket"] = pd.cut(df2["pred_margin"].abs(), bins=bins, labels=labels)

    for lbl in labels:
        sub = df2[df2["mag_bucket"] == lbl]
        n   = len(sub)
        if n < 5:
            continue
        avg_actual = sub["actual_margin"].mean()
        std_actual = sub["actual_margin"].std()
        dir_ok     = (np.sign(sub["pred_margin"]) == np.sign(sub["actual_margin"])).mean()
        print(
            f"  {lbl:>14}  {n:>5}  {avg_actual:>+11.2f}  {dir_ok:>11.1%}  {std_actual:>11.2f}"
        )

    print(
        f"\n  A well-calibrated model: avg actual ~= avg predicted within each bucket."
        f"\n  Low dir-correct% for high-magnitude preds = model overconfident on big edges."
    )


if __name__ == "__main__":
    main()
