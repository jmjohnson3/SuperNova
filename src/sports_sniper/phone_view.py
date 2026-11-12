"""Phone layout for the dashboard: narrow-screen styling for every visitor, and wide tables shown as a list of
cards when the visitor is on a phone (or turns "Phone layout" on in the sidebar). Display only."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

CARD_LIMIT = 30              # cards are tall; a phone list longer than this is better narrowed with the filters
PHONE_AGENTS = ("iphone", "android", "mobile", "ipod")
STATE_KEY = "phone_layout"
WIDE_VALUE = 11              # a longer value gets a row of its own, after the short ones

CSS = """<style>
.sn-card {border: 1px solid rgba(128,128,128,.35); border-radius: .6rem; padding: .6rem .75rem; margin: 0 0 .6rem 0;}
.sn-title {font-weight: 600; line-height: 1.3;}
.sn-sub {opacity: .7; font-size: .85rem; margin-top: .1rem;}
.sn-grid {display: grid; grid-template-columns: 1fr 1fr; gap: .2rem .9rem; margin-top: .45rem; font-size: .9rem;}
.sn-grid div {display: flex; justify-content: space-between; gap: .5rem; border-bottom: 1px dotted rgba(128,128,128,.3);}
.sn-grid span {opacity: .7;}
.sn-grid b {font-weight: 600; text-align: right;}
.sn-grid div.sn-wide {grid-column: 1 / -1; display: block;}
.sn-grid div.sn-wide span {display: block; font-size: .8rem;}
.sn-grid div.sn-wide b {display: block; text-align: left; font-weight: 500;}
.sn-open {display: block; text-align: center; margin-top: .55rem; padding: .55rem; border-radius: .5rem; font-weight: 600;
          border: 1px solid #ff4b4b; color: #ff4b4b !important; text-decoration: none !important;}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] {padding: 3.9rem .75rem 4rem .75rem !important;}
  h1 {font-size: 1.45rem !important; padding: .2rem 0 !important;}
  h2 {font-size: 1.25rem !important;}
  h3 {font-size: 1.1rem !important;}
  h4 {font-size: 1rem !important;}
  .st-key-summary [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important; gap: .4rem .75rem !important;}
  .st-key-summary [data-testid="stColumn"] {min-width: calc(50% - .75rem) !important; flex: 1 1 calc(50% - .75rem) !important;}
  .st-key-summary [data-testid="stMetricValue"] {font-size: 1.25rem !important;}
  [class*="st-key-betnow_"] {border-top: 1px solid rgba(128,128,128,.35); padding-top: .6rem;}
  [class*="st-key-betnow_"] [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important; gap: .4rem .5rem !important;}
  [class*="st-key-betnow_"] [data-testid="stColumn"] {min-width: 100% !important;}
  [class*="st-key-betnow_"] [data-testid="stColumn"]:nth-child(n+3) {min-width: 0 !important; flex: 1 1 28% !important;}
  [class*="st-key-betnow_"] button, [class*="st-key-betnow_"] a {width: 100% !important;}
  [data-testid="stTabs"] [role="tablist"] {flex-wrap: wrap !important; gap: .4rem !important; overflow: visible !important;
                                          height: auto !important; padding-bottom: .5rem; box-shadow: none !important;}
  [data-testid="stTabs"] [role="tablist"]::before, [data-testid="stTabs"] [role="tablist"]::after,
  [data-testid="stTab"]::before, [data-testid="stTab"]::after {display: none !important;}
  [data-testid="stTab"] {border: 1px solid rgba(128,128,128,.4) !important; border-radius: 999px; padding: .3rem .75rem !important;
                         height: auto !important; margin: 0 !important; box-shadow: none !important;}
  [data-testid="stTab"][aria-selected="true"] {border-color: #ff4b4b !important; background: rgba(255,75,75,.12);}
  [data-testid="stTabsScrollLeft"], [data-testid="stTabsScrollRight"] {display: none !important;}
  .stButton button, .stLinkButton a {min-height: 2.75rem;}
}
</style>"""


def detected() -> bool:
    """Whether the visitor's browser says it is a phone. False when that cannot be read."""
    try:
        agent = (st.context.headers.get("User-Agent") or "").lower()
    except Exception:
        return False
    return any(word in agent for word in PHONE_AGENTS) and "ipad" not in agent


def setup() -> bool:
    """Adds the narrow-screen styling and the sidebar switch; True when tables should be shown as cards."""
    st.html(CSS)
    with st.sidebar:
        return bool(st.toggle("Phone layout", value=detected(), key=STATE_KEY,
                              help="Shows the main tables as a list of cards that fit a narrow screen. On by default "
                                   "when the page is opened on a phone."))


def active() -> bool:
    try:
        return bool(st.session_state.get(STATE_KEY, False))
    except Exception:
        return False


def _text(value) -> str:
    return "-" if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "" else str(value)


def cards_html(frame: pd.DataFrame, title: str | None = None, sub: list[str] | None = None, keep: list[str] | None = None,
               link: str | None = "Bet", limit: int = CARD_LIMIT) -> str:
    """One card per row: the title column, a line of context, the remaining columns as label and value, and the
    row's sportsbook link as a button. Everything from the data is escaped; only https links are used."""
    columns = list(frame.columns)
    title = title or columns[0]
    sub = [c for c in (sub or []) if c in columns]
    has_link = link in columns
    fields = [c for c in (keep or columns) if c in columns and c != title and c not in sub and c != link]
    out = []
    for row in frame.head(limit).to_dict("records"):
        parts = [f'<div class="sn-title">{escape(_text(row[title]))}</div>']
        context = [escape(_text(row[c])) for c in sub if _text(row[c]) != "-"]
        if context:
            parts.append(f'<div class="sn-sub">{" · ".join(context)}</div>')
        pairs = [(c, _text(row[c])) for c in fields]
        pairs = [(c, v) for c, v in pairs if v != "-"]
        if pairs:
            parts.append('<div class="sn-grid">' + "".join(f"<div{' class=\"sn-wide\"' if len(v) > WIDE_VALUE else ''}><span>{escape(c)}</span><b>{escape(v)}</b></div>"
                                                     for c, v in sorted(pairs, key=lambda pair: len(pair[1]) > WIDE_VALUE))
                         + "</div>")
        address = row[link] if has_link else None
        if isinstance(address, str) and address.startswith("https://"):
            parts.append(f'<a class="sn-open" href="{escape(address, quote=True)}" target="_blank" rel="noopener noreferrer">Open</a>')
        out.append('<div class="sn-card">' + "".join(parts) + "</div>")
    return "".join(out)


def table(frame: pd.DataFrame, title: str | None = None, sub: list[str] | None = None, keep: list[str] | None = None,
          link: str | None = "Bet", column_config: dict | None = None) -> None:
    """The usual table on a wide screen; cards in the phone layout."""
    if not active():
        st.dataframe(frame, hide_index=True, width="stretch", column_config=column_config)
        return
    st.markdown(cards_html(frame, title, sub, keep, link), unsafe_allow_html=True)
    if len(frame) > CARD_LIMIT:
        st.caption(f"Showing the first {CARD_LIMIT} of {len(frame)}. Narrow the filters to see the rest, or turn off "
                   "**Phone layout** in the sidebar for the full table.")
