"""Book links and Discord text for sharp-gap alerts.

Display only: nothing here prices, selects or stakes a bet, so it is deliberately not pinned by the
strategy registration. A sportsbook renaming its domain or changing its link format can be fixed
here without stopping the scanner.

Every link is host-checked against the book's own domains, so a foreign host can never be rendered
as the book's link, whatever parameters it carries.
"""
from __future__ import annotations

from urllib.parse import urlsplit

BOOK_LABEL = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM",
              "williamhill_us": "Caesars", "espnbet": "theScore Bet (ESPN Bet)", "fanatics": "Fanatics"}
# The Odds API still keys theScore Bet as `espnbet`; since the rebrand every link it returns is on
# sportsbook.thescore.bet, which the old espnbet.com-only check rejected (all 22,282 in October).
BOOK_DOMAINS = {"fanduel": ("fanduel.com",), "draftkings": ("draftkings.com",), "betmgm": ("betmgm.com",),
                "williamhill_us": ("caesars.com",), "espnbet": ("thescore.bet", "espnbet.com"),
                "fanatics": ("fanatics.com",)}
# Query parameters a book uses to carry one specific selection into its betslip.
SELECTION_PARAMS = ("outcomes", "selectionids", "selectionid", "marketid", "addtobetslip", "bet",
                    "market_selection_id")


def _carries_a_selection(url) -> bool:
    query = (url.query or "").lower()
    return any(p in query for p in SELECTION_PARAMS) or "addtobetslip" in (url.path or "").lower()


def _on_domain(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def book_link(edge: dict, game: dict) -> tuple[str | None, str | None]:
    """(url, kind) where kind is:
       'betslip'   - resolved to the bettor's own state, one tap adds the exact selection (FanDuel)
       'selection' - the book's own deep link carrying this selection
       'page'      - a valid link to the book that may not add the bet; find it manually
    """
    book = edge.get("book") or "fanduel"
    if book == "fanduel":
        from nfl_pipeline import fanduel_state
        from nfl_pipeline.fanduel_links import betslip_for_row, provider_link
        row = dict(edge, away=game["away"], home=game["home"],
                   market=edge["stat"] if edge["stat"] in ("spread", "total") else None)
        slip = betslip_for_row(row)
        if slip:
            return slip, "betslip"
        fallback = provider_link(edge.get("link"))
        if not fallback:
            return None, None
        # With a state set, the provider's market IDs belong to another state, and FanDuel answers
        # "Selection not added" (docs/nfl_fanduel_links.md); it is only a way into the sportsbook.
        if fanduel_state.state():
            return fallback, "page"
        return fallback, ("selection" if _carries_a_selection(urlsplit(fallback)) else "page")
    link = str(edge.get("link") or "").strip()
    domains = BOOK_DOMAINS.get(book)
    if not link or not domains or any(c.isspace() or ord(c) < 32 or c in "<>\\" for c in link):
        return None, None
    try:
        url = urlsplit(link)
        host = url.hostname or ""
    except ValueError:
        return None, None
    if url.scheme != "https" or url.username or url.password or url.port is not None:
        return None, None
    if not _on_domain(host, domains):
        return None, None
    return link, ("selection" if _carries_a_selection(url) else "page")


def format_alert(edge: dict, game: dict, stake: float = 0.0) -> str:
    who = edge["player"] or f"{game['away']} @ {game['home']}"
    label = edge["stat"].replace("_", " ").title()
    book = edge.get("book") or "fanduel"
    book_name = BOOK_LABEL.get(book, book.title())
    gap = f" ({edge['sharp_book'].title()} {edge['sharp_line']:g})" if edge["line_gap"] else f" ({edge['sharp_book'].title()})"
    link, kind = book_link(edge, game)
    kickoff = int(game["start"].timestamp())
    heading = f"**SHARP EDGE - bet ${stake:g} at {book_name}**" if stake > 0 else "**SHARP EDGE - research, not a bet**"
    return (f"{heading}\n{who} {edge['side'].upper()} {edge['line']:g} {label} "
            f"({game['away']} @ {game['home']}, <t:{kickoff}:R>)\n{book_name} {edge['price']:+d} | "
            f"fair {edge['fair_probability']:.1%}{gap} | "
            f"EV {edge['ev']:+.1%} | take at {edge['minimum_price']:+d} or better"
            + (f"\n[Add to slip](<{link}>)" if kind == "betslip"
               else f"\n[Open this bet on {book_name}](<{link}>)" if kind == "selection"
               else f"\n[Open {book_name} - find it manually](<{link}>)" if link
               else f"\nNo {book_name} link captured; search manually."))
