"""NFL Report tab: the weekly matchup data packets (`nfl_ai_report.packets`) with a link to every line and prop they show.

Information only. A packet holds statistics, injuries, lines and props, not picks; nothing here flags or paper-bets.
Links are shown for games that have not started: FanDuel's betslip resolved for the bettor's state when
NFL_FANDUEL_STATE is set (FanDuel's public event feed, no credits), otherwise the stored provider link, and the
book's NFL page when neither carries the selection. The price at the book may have moved since the packet.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from sports_sniper import phone_view

BOOK_NAME = {"fanduel": "FanDuel", "draftkings": "DraftKings"}
BOOK_PAGE = {"fanduel": "https://sportsbook.fanduel.com/navigation/nfl",
             "draftkings": "https://sportsbook.draftkings.com/leagues/football/nfl"}
STAT_NAME = {"receiving_yards": "receiving yards", "receptions": "receptions", "rushing_yards": "rushing yards",
             "passing_yards": "passing yards"}
OPENS = {"betslip": "Adds it to the betslip", "selection": "Book's link for this bet", "page": "NFL page: find it there"}
PBP_MAX_AGE = timedelta(hours=12)


def packet_folder() -> Path:
    from nfl_ai_report.packets import OUT
    return OUT


def slate_of(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def list_packets(folder: Path) -> list[Path]:
    """Newest slate first."""
    return sorted((p for p in folder.glob("*.json") if slate_of(p)), key=slate_of, reverse=True)


def default_index(paths: list[Path], today: date) -> int:
    """The next slate from today on (the one to bet), else the latest."""
    ahead = [i for i, p in enumerate(paths) if slate_of(p) >= today]
    return ahead[-1] if ahead else 0


def started(game: dict, slate: date, now: datetime) -> bool:
    kickoff = game.get("kickoff_utc")
    if kickoff:
        return pd.Timestamp(kickoff) <= now
    return slate < now.astimezone(timezone.utc).date()


def _price(v) -> str:
    return "-" if v is None else f"{int(v):+d}"


def bet_rows(game: dict) -> list[dict]:
    """Every bet the packet shows for a game: both sides of each book's latest spread and total, and FanDuel's
    priced side of each prop. 'edge' holds what the link lookup needs."""
    away, home = game.get("away") or game["matchup"].split(" @ ")[0], game.get("home") or game["matchup"].split(" @ ")[-1]
    rows = []
    for book, b in (game.get("lines") or {}).items():
        line, links = b["latest"], b["latest"].get("links") or {}
        spread, total = line.get("home_spread"), line.get("total")
        sides = []
        if spread is not None:
            sides += [("spread", "home", spread, f"{home} {spread:+g}", line.get("home_price"), links.get("spread_home")),
                      ("spread", "away", -spread, f"{away} {-spread:+g}", line.get("away_price"), links.get("spread_away"))]
        if total is not None:
            sides += [("total", "over", total, f"Over {total:g}", line.get("over_price"), links.get("total_over")),
                      ("total", "under", total, f"Under {total:g}", line.get("under_price"), links.get("total_under"))]
        for stat, side, value, label, price, link in sides:
            rows.append({"Book": BOOK_NAME.get(book, book), "Bet": label, "Market": stat.title(), "Odds": _price(price),
                         "Seen": line.get("seen"), "edge": {"book": book, "stat": stat, "side": side, "line": value, "link": link,
                                                            "player": None}})
    for p in game.get("props_fanduel") or []:
        rows.append({"Book": "FanDuel", "Bet": f"{p['player']} {p['side_priced']} {p['line']:g} {STAT_NAME.get(p['stat'], p['stat'])}",
                     "Market": "Prop", "Odds": _price(p.get("price")), "Seen": None,
                     "Model projection": p.get("model_projection"),
                     "edge": {"book": "fanduel", "stat": p["stat"], "side": p["side_priced"], "line": p["line"], "link": p.get("link"),
                              "player": p.get("book_player") or p["player"]}})
    return rows


def resolve(edge: dict, away: str, home: str) -> tuple[str | None, str | None]:
    """(url, kind) for one bet; the book's NFL page when no link carries the selection."""
    try:
        from nfl_pipeline.sharp_links import book_link
        url, kind = book_link(edge, {"away": away, "home": home})
    except Exception:
        url, kind = None, None
    if not url:
        url, kind = BOOK_PAGE.get(edge["book"]), "page"
    return url, kind


@st.cache_data(ttl=600, show_spinner="Looking up FanDuel betslip links...")
def links_for(path: str, mtime: float, now_minute: str) -> dict:
    """{(game_id, row number): (url, kind)} for the games of a packet that have not started."""
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    slate, now = date.fromisoformat(packet["slate"]), pd.Timestamp(now_minute).to_pydatetime()
    out = {}
    for g in packet["games"]:
        if started(g, slate, now):
            continue
        away, home = g.get("away") or g["matchup"].split(" @ ")[0], g.get("home") or g["matchup"].split(" @ ")[-1]
        for i, row in enumerate(bet_rows(g)):
            out[f"{g.get('game_id', g['matchup'])}|{i}"] = resolve(row["edge"], away, home)
    return out


def game_text(packet: dict, game: dict) -> str:
    """The game's part of the packet text, without the slate header and the game heading."""
    from nfl_ai_report.packets import to_text
    lines = to_text({**packet, "games": [game]}).splitlines()[3:]
    return "\n".join(l for l in lines if not l.startswith("## "))


def rebuild(season: int, week: int) -> None:
    from nfl_ai_report.packets import build
    build(season, [week], pbp_max_age=PBP_MAX_AGE)


def render(now: datetime) -> None:
    st.caption("The data packet for each NFL slate: what our AI matchup report will be shown, using only what was stored by the "
               "report time (11 AM ET, or 90 minutes before an early kickoff). It holds no picks yet, and the model's prop projections "
               "have not beaten the lines. Links open the bet at the book; the price there may have moved.")
    folder = packet_folder()
    paths = list_packets(folder)
    if not paths:
        st.info("No packets yet. Build them with `python -m nfl_ai_report.packets --season 2026 --weeks 1-11`.")
        return
    today = now.astimezone().date()
    pick, act = st.columns([3, 1])
    path = pick.selectbox("Slate", paths, index=default_index(paths, today), key="nfl_report_slate",
                          format_func=lambda p: f"Week {int(p.stem.split('_')[1][1:])} · {slate_of(p):%a %b %d, %Y}".replace(" 0", " "))
    packet = json.loads(path.read_text(encoding="utf-8"))
    built = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    act.caption(f"Built {built:%a %I:%M %p}".replace(" 0", " "))
    if act.button("Rebuild this week", key="nfl_report_rebuild",
                  help="Reads the latest stored lines, props and injuries (no API credits) and today's play-by-play. "
                       "Takes a few minutes."):
        with st.spinner("Building this week's packets..."):
            try:
                rebuild(int(packet["season"]), int(packet["week"]))
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:                     # a database or download failure: say so, keep the old packet
                st.error(f"The rebuild failed: {exc}")
    report = pd.Timestamp(packet["report_time_utc"]).to_pydatetime()
    note = "" if packet.get("injury_snapshots") else " No injury report is stored for this date."
    if report > now:
        note += (f" The report time is still ahead ({report.astimezone():%a %I:%M %p}); rebuild after it for the final packet."
                 .replace(" 0", " "))
    n = len(packet["games"])
    st.markdown(f"**{n} game{'' if n == 1 else 's'}.**{note}")
    md = path.with_suffix(".md")
    if md.exists():
        st.download_button("Download the packet (Markdown)", md.read_bytes(), file_name=md.name, mime="text/markdown",
                           key=f"nfl_report_dl_{md.name}")
    links = links_for(str(path), path.stat().st_mtime, pd.Timestamp(now).floor("10min").isoformat())
    slate = date.fromisoformat(packet["slate"])
    for g in packet["games"]:
        live = not started(g, slate, now)
        with st.expander(f"{g['matchup']} · {g['kickoff_et']} ET" + ("" if live else " · started"), expanded=False):
            rows = bet_rows(g)
            if rows:
                frame = pd.DataFrame([{k: v for k, v in r.items() if k != "edge"} for r in rows])
                if live:
                    found = [links.get(f"{g.get('game_id', g['matchup'])}|{i}", (None, None)) for i in range(len(rows))]
                    frame["Opens"] = [OPENS.get(kind, "-") for _, kind in found]
                    frame["Link"] = [url for url, _ in found]
                st.markdown("**Bets in this packet**" + ("" if live else " (the game has started, so no links)"))
                frame = frame.dropna(axis=1, how="all")
                order = [c for c in ("Book", "Bet", "Odds", "Link", "Opens", "Market", "Seen", "Model projection") if c in frame]
                frame = frame[order].astype(object).where(frame[order].notna(), "")
                phone_view.table(frame, title="Bet", sub=["Book", "Odds"], link="Link" if live else None,
                                 column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")})
            st.markdown(game_text(packet, g))
