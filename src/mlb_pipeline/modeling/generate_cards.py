"""
mlb_pipeline.modeling.generate_cards
=====================================
Generate styled PNG prediction cards and upload them to a Discord webhook.

Cards produced (one POST each):
  1. Game bets card  — top run-line / total edge bets for the day
  2. HR card         — top 1 home run hitter
  3. TB card         — top 2 total bases hitters
  4. H  card         — top 2 hits hitters

Usage (standalone):
    python -m mlb_pipeline.modeling.generate_cards [--date YYYY-MM-DD]

Used by run_daily_and_notify.py after predict_today / predict_player_props complete.
"""
from __future__ import annotations

import asyncio
import io
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import psycopg2
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("mlb_pipeline.modeling.generate_cards")

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"

# ─────────────────────────────────────────────────────────────────────────────
# Color palette  (GitHub-dark inspired)
# ─────────────────────────────────────────────────────────────────────────────
BG            = (13,  17,  23)      # #0d1117  card background
PANEL         = (22,  27,  34)      # #161b22  inner panels
PANEL_BORDER  = (48,  54,  61)      # #30363d
GOLD          = (210, 153,  34)     # #d29922  header accent
WHITE         = (230, 237, 243)     # #e6edf3  primary text
GRAY          = (125, 133, 144)     # #7d8590  secondary text
GREEN         = ( 46, 160,  67)     # #2ea043  positive edge
GREEN_BRIGHT  = ( 63, 185,  80)     # #3fb950
RED           = (218,  54,  51)     # #da3633  negative / UNDER
BLUE          = ( 88, 166, 255)     # #58a6ff  win-prob accent
DIVIDER       = ( 33,  38,  45)     # #21262d

# ─────────────────────────────────────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────────────────────────────────────
_FONT_DIR = Path("C:/Windows/Fonts")

def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()

def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    return {
        "title":    _font("bahnschrift.ttf", 26),
        "date":     _font("segoeuil.ttf",    18),
        "section":  _font("arialbd.ttf",     16),
        "team":     _font("arialbd.ttf",     38),
        "line":     _font("arialbd.ttf",     28),
        "label":    _font("arial.ttf",       16),
        "matchup":  _font("arial.ttf",       15),
        "badge":    _font("arialbd.ttf",     17),
        "player":   _font("arialbd.ttf",     24),
        "pred":     _font("arialbd.ttf",     20),
        "small":    _font("arial.ttf",       13),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
W = 700  # card width

def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def _draw_header(draw: ImageDraw.ImageDraw, f: dict, subtitle: str, et_date: date, y: int = 0) -> int:
    """Draw the gold 'SUPERNOVA MLB' header bar. Returns new y."""
    day_str = et_date.strftime("%a %b %d · %Y").replace(" 0", " ") if hasattr(et_date, "strftime") else str(et_date)

    # Logo text
    draw.text((24, y + 14), "SUPERNOVA", font=f["title"], fill=GOLD)
    tw = _text_w(draw, "SUPERNOVA", f["title"])
    draw.text((24 + tw + 10, y + 18), "MLB", font=f["section"], fill=WHITE)

    # Subtitle (stat type or "Game Bets")
    if subtitle:
        sub_w = _text_w(draw, subtitle, f["section"])
        draw.text((W - sub_w - 24, y + 18), subtitle, font=f["section"], fill=GRAY)

    # Date top-right
    date_str = day_str
    dw = _text_w(draw, date_str, f["date"])
    draw.text((W - dw - 24, y + 44), date_str, font=f["date"], fill=GRAY)

    y += 72
    # Divider
    draw.rectangle([(0, y), (W, y + 2)], fill=GOLD)
    return y + 2


def _draw_bet_row(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    f: dict,
    y: int,
    team: str,
    line_str: str,
    edge: float,
    win_pct: float,
    matchup: str,
    sp_info: str = "",
) -> int:
    """Draw one run-line or total bet row. Returns new y."""
    ROW_H = 88
    PAD = 14

    # Panel
    draw.rounded_rectangle(
        [(PAD, y + 6), (W - PAD, y + ROW_H)],
        radius=8, fill=PANEL, outline=PANEL_BORDER, width=1,
    )

    # Team + line (left)
    team_x = PAD + 16
    draw.text((team_x, y + 12), team, font=f["team"], fill=WHITE)
    team_w = _text_w(draw, team, f["team"])
    draw.text((team_x + team_w + 10, y + 22), line_str, font=f["line"], fill=GRAY)

    # Edge badge (centre-right)
    edge_color = GREEN_BRIGHT if edge >= 0 else RED
    edge_txt = f"+{edge:.2f}" if edge >= 0 else f"{edge:.2f}"
    badge_x = W - PAD - 180
    draw.rounded_rectangle(
        [(badge_x, y + 16), (badge_x + 86, y + 44)],
        radius=6, fill=edge_color,
    )
    bw = _text_w(draw, edge_txt, f["badge"])
    draw.text((badge_x + (86 - bw) // 2, y + 20), edge_txt, font=f["badge"], fill=BG)
    draw.text((badge_x + 28, y + 48), "EDGE", font=f["small"], fill=GRAY)

    # Win % (far right)
    wp_txt = f"{win_pct:.0%}"
    wp_x = W - PAD - 72
    draw.text((wp_x, y + 14), wp_txt, font=f["pred"], fill=BLUE)
    draw.text((wp_x + 4, y + 48), "WIN", font=f["small"], fill=GRAY)

    # Matchup (bottom-left)
    draw.text((team_x, y + 58), matchup, font=f["matchup"], fill=GRAY)
    if sp_info:
        sp_w = _text_w(draw, sp_info, f["small"])
        draw.text((W - PAD - 16 - sp_w, y + 62), sp_info, font=f["small"], fill=GRAY)

    return y + ROW_H + 8


def _draw_player_row(
    draw: ImageDraw.ImageDraw,
    f: dict,
    y: int,
    player: str,
    team_opp: str,
    pred: float,
    line: Optional[float],
    edge: Optional[float],
    stat_fmt: str = "{:.2f}",
    dot_color: tuple = GREEN,
) -> int:
    """Draw one player prop row. Returns new y."""
    ROW_H = 82
    PAD = 14

    draw.rounded_rectangle(
        [(PAD, y + 6), (W - PAD, y + ROW_H)],
        radius=8, fill=PANEL, outline=PANEL_BORDER, width=1,
    )

    # Colored dot indicator + player name
    name_x = PAD + 16
    dot_r = 7
    draw.ellipse([(name_x, y + 22), (name_x + dot_r * 2, y + 22 + dot_r * 2)], fill=dot_color)
    name_x += dot_r * 2 + 10
    draw.text((name_x, y + 12), player, font=f["player"], fill=WHITE)

    # Team / opponent (right of name, smaller)
    team_w = _text_w(draw, player, f["player"])
    draw.text((name_x + team_w + 12, y + 20), team_opp, font=f["label"], fill=GRAY)

    # Pred value (bottom-left area)
    pred_txt = f"PRED  {stat_fmt.format(pred)}"
    draw.text((PAD + 16, y + 50), pred_txt, font=f["pred"], fill=WHITE)

    # Line + edge (right side)
    if line is not None and edge is not None:
        dir_ch = "O" if edge >= 0 else "U"
        edge_col = GREEN_BRIGHT if edge >= 0 else RED
        line_txt = f"{dir_ch}{line:.1f}"
        edge_txt = f"edge {edge:+.2f}"
        lw = _text_w(draw, line_txt, f["badge"])
        draw.rounded_rectangle(
            [(W - PAD - 170, y + 12), (W - PAD - 170 + lw + 20, y + 38)],
            radius=5, fill=edge_col,
        )
        draw.text((W - PAD - 170 + 10, y + 16), line_txt, font=f["badge"], fill=BG)
        draw.text((W - PAD - 130, y + 48), edge_txt, font=f["small"], fill=GRAY)
    else:
        draw.text((W - PAD - 100, y + 24), "no line", font=f["small"], fill=GRAY)

    return y + ROW_H + 8


def _finalize(img: Image.Image, y: int, total_h: int) -> Image.Image:
    """Crop to actual used height with a small bottom pad."""
    return img.crop((0, 0, W, min(y + 20, total_h)))


# ─────────────────────────────────────────────────────────────────────────────
# Card builders
# ─────────────────────────────────────────────────────────────────────────────

def make_game_bets_card(bets: list[dict], et_date: date) -> Image.Image:
    """
    bets: list of dicts with keys:
      home_team_abbr, away_team_abbr, run_line_bet_side, edge_run_line, win_prob_rl,
      market_run_line, total_bet_side, edge_total, win_prob_total, market_total,
      home_sp_name, away_sp_name
    """
    n = max(len(bets), 1)
    total_h = 80 + 100 + n * 100 + 30
    img  = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)
    f    = _fonts()

    y = _draw_header(draw, f, "Game Bets", et_date)
    y += 8

    # Section label
    draw.text((24, y), "TOP BETS TODAY", font=f["section"], fill=GRAY)
    y += 28

    if not bets:
        draw.text((24, y + 10), "No edge bets for today's slate.", font=f["label"], fill=GRAY)
        y += 50
    else:
        for b in bets:
            home = b["home_team_abbr"]
            away = b["away_team_abbr"]
            mrl  = float(b.get("market_run_line") or -1.5)
            mtt  = b.get("market_total")

            # Run-line bet
            if b.get("run_line_bet_side"):
                side = b["run_line_bet_side"]
                edge = float(b.get("edge_run_line") or 0)
                wp   = float(b.get("win_prob_rl") or 0.5)
                if side == "home":
                    team, line_str, disp_edge = home, f"{mrl:+.1f}", edge
                else:
                    team, line_str, disp_edge = away, f"{-mrl:+.1f}", abs(edge)
                matchup = f"{away} @ {home}"
                home_sp = (b.get("home_sp_name") or "").split()[-1] if b.get("home_sp_name") else ""
                away_sp = (b.get("away_sp_name") or "").split()[-1] if b.get("away_sp_name") else ""
                sp_info = f"{away_sp} vs {home_sp}" if home_sp and away_sp else ""
                y = _draw_bet_row(img, draw, f, y, team, f"RL {line_str}", disp_edge, wp, matchup, sp_info)

            # Total bet
            if b.get("total_bet_side") and mtt is not None:
                side  = b["total_bet_side"]
                edge  = float(b.get("edge_total") or 0)
                wp    = float(b.get("win_prob_total") or 0.5)
                disp  = abs(edge) if side == "under" else edge
                team  = f"{'OVER' if side == 'over' else 'UNDER'} {float(mtt):.1f}"
                matchup = f"{away} @ {home}"
                y = _draw_bet_row(img, draw, f, y, team, "", disp, wp, matchup)

    return _finalize(img, y, total_h)


def make_prop_card(
    players: list[dict],
    stat_label: str,
    dot_color: tuple,
    stat_fmt: str,
    et_date: date,
) -> Image.Image:
    """
    players: list of dicts with keys:
      player_name, team_abbr, pred_value, book_line, edge, opponent_abbr
    """
    n = max(len(players), 1)
    total_h = 80 + 50 + n * 98 + 30
    img  = Image.new("RGB", (W, total_h), BG)
    draw = ImageDraw.Draw(img)
    f    = _fonts()

    y = _draw_header(draw, f, stat_label, et_date)
    y += 8

    section_txt = f"TOP {stat_label.upper()} HITTER{'S' if len(players) > 1 else ''}"
    draw.text((24, y), section_txt, font=f["section"], fill=GRAY)
    y += 28

    for p in players:
        name  = p.get("player_name") or "Unknown"
        team  = p.get("team_abbr", "?")
        opp   = p.get("opponent_abbr", "?")
        pred  = float(p.get("pred_value") or 0)
        line  = float(p["book_line"]) if p.get("book_line") is not None else None
        edge  = float(p["edge"])      if p.get("edge")      is not None else None

        # Shorten name: "Fernando Tatis Jr." → "Tatis Jr."
        parts = name.split()
        if len(parts) >= 2 and parts[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
            short = f"{parts[-2]} {parts[-1]}"
        elif len(parts) >= 2:
            short = parts[-1]
        else:
            short = name

        team_opp = f"{team} vs {opp}"
        y = _draw_player_row(draw, f, y, short, team_opp, pred, line, edge, stat_fmt, dot_color)

    return _finalize(img, y, total_h)


# ─────────────────────────────────────────────────────────────────────────────
# DB queries
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_game_bets(conn, et_date: date) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT home_team_abbr, away_team_abbr,
                   run_line_bet_side, edge_run_line, win_prob_rl, market_run_line,
                   total_bet_side, edge_total, win_prob_total, market_total,
                   home_sp_name, away_sp_name
            FROM bets.mlb_game_predictions
            WHERE game_date_et = %s
              AND (run_line_bet_side IS NOT NULL OR total_bet_side IS NOT NULL)
            ORDER BY GREATEST(
                ABS(COALESCE(edge_run_line, 0)),
                ABS(COALESCE(edge_total, 0))
            ) DESC
            LIMIT 6
        """, (et_date,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_top_players(conn, et_date: date, stat: str, limit: int) -> list[dict]:
    """Fetch top players for a given stat sorted by pred_value desc.
    Opponent is parsed from game_slug (format: YYYYMMDD-AWAY-HOME).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT player_name, team_abbr, pred_value, book_line, edge, game_slug
            FROM bets.mlb_prop_predictions
            WHERE game_date_et = %s
              AND stat = %s
            ORDER BY pred_value DESC
            LIMIT %s
        """, (et_date, stat, limit))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    # Parse opponent from game_slug: "20260416-BAL-CLE" → away=BAL, home=CLE
    for r in rows:
        slug = r.get("game_slug") or ""
        parts = slug.split("-")
        if len(parts) >= 3:
            away, home = parts[-2], parts[-1]
            r["opponent_abbr"] = home if r["team_abbr"] == away else away
        else:
            r["opponent_abbr"] = "?"
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Discord upload
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


async def _post_image(client: httpx.AsyncClient, webhook_url: str, img_bytes: bytes, filename: str) -> None:
    for attempt in range(4):
        try:
            r = await client.post(
                webhook_url,
                files={"file": (filename, img_bytes, "image/png")},
                timeout=30,
            )
            if r.status_code in (200, 204):
                return
            if r.status_code == 429 and attempt < 3:
                retry_after = float(r.json().get("retry_after", 2.0))
                log.warning("Discord rate-limited — retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                continue
            log.warning("Discord upload failed: %d %s", r.status_code, r.text[:200])
            return
        except httpx.TimeoutException:
            if attempt >= 3:
                raise
            await asyncio.sleep(2.0 * (attempt + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def generate_and_post(
    webhook_url: str,
    et_date: date,
    pg_dsn: str = _PG_DSN,
    save_dir: Optional[Path] = None,
) -> None:
    """Generate all 4 cards and post them to Discord. Optionally save PNGs locally."""
    conn = psycopg2.connect(pg_dsn)
    try:
        game_bets   = _fetch_game_bets(conn, et_date)
        hr_players  = _fetch_top_players(conn, et_date, "batter_home_runs",   1)
        tb_players  = _fetch_top_players(conn, et_date, "batter_total_bases", 2)
        h_players   = _fetch_top_players(conn, et_date, "batter_hits",        2)
    finally:
        conn.close()

    # Dot colors per stat: red for HR (power), gold for TB (extra bases), blue for H (contact)
    cards = [
        (make_game_bets_card(game_bets, et_date),                                            "mlb_game_bets.png"),
        (make_prop_card(hr_players, "Home Runs",   RED,         "{:.3f}", et_date),          "mlb_hr.png"),
        (make_prop_card(tb_players, "Total Bases", GOLD,        "{:.2f}", et_date),          "mlb_tb.png"),
        (make_prop_card(h_players,  "Hits",        BLUE,        "{:.2f}", et_date),          "mlb_hits.png"),
    ]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for img, fname in cards:
            p = save_dir / fname
            img.save(str(p))
            log.info("Saved %s", p)

    if not webhook_url:
        log.warning("No webhook URL — images saved locally only.")
        return

    async with httpx.AsyncClient() as client:
        for img, fname in cards:
            img_bytes = _img_to_bytes(img)
            log.info("Posting %s (%d bytes) to Discord...", fname, len(img_bytes))
            await _post_image(client, webhook_url, img_bytes, fname)
            await asyncio.sleep(0.6)  # avoid rate limits between images


def main() -> None:
    import argparse, os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate MLB prediction cards")
    parser.add_argument("--date",     default=None, help="YYYY-MM-DD (ET), defaults to today")
    parser.add_argument("--save-dir", default=None, help="Directory to save PNG files locally")
    parser.add_argument("--no-post",  action="store_true", help="Skip Discord upload")
    args = parser.parse_args()

    et_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()

    webhook_url = "" if args.no_post else os.getenv(
        "MLB_DISCORD_WEBHOOK_URL",
        "https://discord.com/api/webhooks/1487880251886403596/fB9WT_Krl2QdOV8MD6o0Pzdp-BgnsJ8wISAJ6-Xi0wMVQfViVjbKU2wV4VC9f52Iwo9n",
    )
    save_dir = Path(args.save_dir) if args.save_dir else None

    asyncio.run(generate_and_post(webhook_url, et_date, save_dir=save_dir))


if __name__ == "__main__":
    main()
