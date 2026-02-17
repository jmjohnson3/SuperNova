from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd


def _fmt_float(x: float | int | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def _safe_str(x) -> str:
    return "-" if x is None else str(x)


@dataclass(frozen=True)
class DiscordReportConfig:
    max_players_per_game: int = 18
    max_alt_lines_per_stat: int = 25


def build_discord_report(
    et_day: date,
    game_preds: pd.DataFrame | None,
    player_preds: pd.DataFrame | None,
    alt_points: pd.DataFrame | None,
    alt_rebounds: pd.DataFrame | None,
    alt_assists: pd.DataFrame | None,
    cfg: DiscordReportConfig | None = None,
) -> str:
    cfg = cfg or DiscordReportConfig()

    lines: list[str] = []
    lines.append(f"**🏀 SuperNovaBets — NBA Slate (ET {et_day})**")
    lines.append("")

    # --- Game predictions (optional) ---
    if game_preds is not None and not game_preds.empty:
        lines.append("**Game Model (spread/total)**")
        lines.append("```")
        # expects columns like: game_slug, away_team_abbr, home_team_abbr, market_spread_home, market_total, start_ts_utc
        gp = game_preds.copy()
        cols = [c for c in ["start_ts_utc","away_team_abbr","home_team_abbr","market_spread_home","market_total"] if c in gp.columns]
        gp = gp[cols].head(20)
        for _, r in gp.iterrows():
            a = _safe_str(r.get("away_team_abbr"))
            h = _safe_str(r.get("home_team_abbr"))
            sp = _fmt_float(r.get("market_spread_home"), 1)
            tot = _fmt_float(r.get("market_total"), 1)
            lines.append(f"{a} @ {h} | spread_home={sp} | total={tot}")
        lines.append("```")
        lines.append("")

    # --- Player props predictions ---
    if player_preds is not None and not player_preds.empty:
        lines.append("**Player Props (model projections)**")

        # expected columns: start_ts_utc, game_slug, team_abbr, opponent_abbr, is_home, player_name, pred_points, pred_rebounds, pred_assists
        pp = player_preds.copy()
        if "start_ts_utc" in pp.columns:
            pp = pp.sort_values(["start_ts_utc","game_slug","team_abbr"], ascending=[True, True, True])

        group_cols = [c for c in ["start_ts_utc", "game_slug"] if c in pp.columns]
        if not group_cols:
            group_cols = ["game_slug"] if "game_slug" in pp.columns else []

        for keys, g in pp.groupby(group_cols, sort=False) if group_cols else [((None,), pp)]:
            slug = g["game_slug"].iloc[0] if "game_slug" in g.columns else "GAME"
            lines.append(f"__{slug}__")
            lines.append("```")
            g2 = g.sort_values(["pred_points"], ascending=False).head(cfg.max_players_per_game)

            for _, r in g2.iterrows():
                name = _safe_str(r.get("player_name"))
                team = _safe_str(r.get("team_abbr"))
                opp = _safe_str(r.get("opponent_abbr"))
                p = _fmt_float(r.get("pred_points"), 1)
                reb = _fmt_float(r.get("pred_rebounds"), 1)
                ast = _fmt_float(r.get("pred_assists"), 1)
                lines.append(f"{name} ({team} vs {opp}) | PTS {p} | REB {reb} | AST {ast}")
            lines.append("```")
            lines.append("")

    # --- Alt line scanners (100% last N) ---
    def add_alt_block(title: str, df: pd.DataFrame | None) -> None:
        if df is None or df.empty:
            return
        lines.append(f"**{title} (best line per player)**")
        lines.append("```")
        # expects: player_name, team_abbr, line, hits, games_count, hit_rate, avg_value
        d = df.sort_values(["line","hits","avg_value"], ascending=[False, False, False]).head(cfg.max_alt_lines_per_stat)
        for _, r in d.iterrows():
            name = _safe_str(r.get("player_name"))
            team = _safe_str(r.get("team_abbr"))
            linev = _fmt_float(r.get("line"), 0) if float(r.get("line", 0)).is_integer() else _fmt_float(r.get("line"), 1)
            hits = int(r.get("hits", 0))
            gc = int(r.get("games_count", 0))
            hr = float(r.get("hit_rate", 0.0)) * 100.0
            avg = _fmt_float(r.get("avg_value"), 1)
            lines.append(f"{name} ({team}) | best_line={linev} | {hits}/{gc} ({hr:.0f}%) | avg={avg}")
        lines.append("```")
        lines.append("")

    add_alt_block("✅ POINTS OVER — 100% last sample", alt_points)
    add_alt_block("✅ REBOUNDS OVER — 100% last sample", alt_rebounds)
    add_alt_block("✅ ASSISTS OVER — 100% last sample", alt_assists)

    return "\n".join(lines).strip()
