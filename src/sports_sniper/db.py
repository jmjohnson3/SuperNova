"""Persistence (SQLAlchemy 2): price snapshots, detected opportunities and paper bets. SQLite or PostgreSQL."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine,
                        func, select)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

STRATEGY = "market-consensus-v1"


class Base(DeclarativeBase):
    pass


def line_key(line: float | None) -> str:
    """A text form of the line for unique keys (a NULL line would never collide with itself)."""
    return "" if line is None else f"{float(line):g}"


class Scan(Base):
    __tablename__ = "sniper_scans"
    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(20))               # sample | saved | live
    sports: Mapped[str] = mapped_column(String(80))
    events: Mapped[int] = mapped_column(Integer, default=0)
    quotes: Mapped[int] = mapped_column(Integer, default=0)
    new_quotes: Mapped[int] = mapped_column(Integer, default=0)
    opportunities: Mapped[int] = mapped_column(Integer, default=0)
    credits_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    credits_remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class EventSeen(Base):
    """When each game was last looked at, so an older look at its prices can be told from the current one."""
    __tablename__ = "sniper_events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sport: Mapped[str] = mapped_column(String(10))
    last_observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class QuoteRow(Base):
    """One observed price. A book's unchanged market keeps its timestamp, so it is stored once."""
    __tablename__ = "sniper_quotes"
    __table_args__ = (UniqueConstraint("event_id", "book", "market", "player_key", "line_key", "side", "last_update",
                                       name="sniper_quote_identity"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("sniper_scans.id"), index=True)
    sport: Mapped[str] = mapped_column(String(10))
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    event_name: Mapped[str] = mapped_column(String(160))
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    book: Mapped[str] = mapped_column(String(40))
    market: Mapped[str] = mapped_column(String(60))
    player: Mapped[str] = mapped_column(String(120))
    player_key: Mapped[str] = mapped_column(String(120))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_key: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[int] = mapped_column(Integer)
    last_update: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OpportunityRow(Base):
    __tablename__ = "sniper_opportunities"
    __table_args__ = (UniqueConstraint("event_id", "book", "market", "player_key", "line_key", "side", "observed_at",
                                       name="sniper_opportunity_identity"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("sniper_scans.id"), index=True)
    data_source: Mapped[str] = mapped_column(String(20))          # sample | saved | live
    sport: Mapped[str] = mapped_column(String(10))
    event_id: Mapped[str] = mapped_column(String(64))
    event_name: Mapped[str] = mapped_column(String(160))
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    book: Mapped[str] = mapped_column(String(40))
    market: Mapped[str] = mapped_column(String(60))
    player: Mapped[str] = mapped_column(String(120))
    player_key: Mapped[str] = mapped_column(String(120))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_key: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[int] = mapped_column(Integer)
    decimal: Mapped[float] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_update: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    link: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20))
    source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    standard_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    break_even: Mapped[float] = mapped_column(Float)
    fair_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    conservative_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_books: Mapped[str] = mapped_column(String(200), default="")
    has_sharp: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence: Mapped[str] = mapped_column(String(16))
    is_longshot: Mapped[bool] = mapped_column(Boolean, default=False)
    is_best_price: Mapped[bool] = mapped_column(Boolean, default=False)
    books_quoting: Mapped[int] = mapped_column(Integer, default=1)
    premium_vs_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    minutes_to_start: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)


class PaperBet(Base):
    """A simulated entry at a price that was actually observed. Never a real wager."""
    __tablename__ = "sniper_paper_bets"
    __table_args__ = (UniqueConstraint("strategy", "event_id", "market", "player_key", "line_key", "side", "entry_no",
                                       name="sniper_paper_bet_identity"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    strategy: Mapped[str] = mapped_column(String(60))
    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    entry_no: Mapped[int] = mapped_column(Integer, default=1)
    opportunity_id: Mapped[int | None] = mapped_column(ForeignKey("sniper_opportunities.id"), nullable=True)
    data_source: Mapped[str] = mapped_column(String(20))
    sport: Mapped[str] = mapped_column(String(10))
    event_id: Mapped[str] = mapped_column(String(64))
    event_name: Mapped[str] = mapped_column(String(160))
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    book: Mapped[str] = mapped_column(String(40))
    market: Mapped[str] = mapped_column(String(60))
    player: Mapped[str] = mapped_column(String(120))
    player_key: Mapped[str] = mapped_column(String(120))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_key: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[int] = mapped_column(Integer)
    decimal: Mapped[float] = mapped_column(Float)
    price_observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    standard_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    conservative_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[str] = mapped_column(String(16))
    stake: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10), default="open")   # open | won | lost | push | void
    result_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    closing_line_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def insert_new(session: Session, model, rows: list[dict], conflict: list[str]) -> int:
    """Insert rows, skipping any that already exist on `conflict`. Returns how many were new."""
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise RuntimeError(f"unsupported database: {dialect}")
    added = 0
    for start in range(0, len(rows), 400):
        chunk = rows[start:start + 400]
        added += session.execute(insert(model).values(chunk).on_conflict_do_nothing(index_elements=conflict)).rowcount or 0
    return added


def as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def touch_events(session: Session, latest: dict[str, tuple[str, datetime]]) -> None:
    """Record the newest observation of each game: {event_id: (sport, observed_at)}."""
    for event_id, (sport, observed) in latest.items():
        row = session.get(EventSeen, event_id)
        if row is None:
            session.add(EventSeen(event_id=event_id, sport=sport, last_observed_at=observed))
        elif as_utc(row.last_observed_at) < observed:
            row.last_observed_at = observed


def quote_rows(quotes, scan_id: int) -> list[dict]:
    return [dict(scan_id=scan_id, sport=q.sport, event_id=q.event_id, event_name=q.event_name, commence_time=q.commence_time,
                 book=q.book, market=q.market, player=q.player, player_key=q.player_key, line=q.line, line_key=line_key(q.line),
                 side=q.side, price=q.price, last_update=q.last_update, observed_at=q.observed_at) for q in quotes]


def opportunity_rows(found, scan_id: int, data_source: str) -> list[dict]:
    return [dict(o.record(), scan_id=scan_id, data_source=data_source, line_key=line_key(o.line)) for o in found]


def place_paper_bet(session: Session, opportunity: OpportunityRow, *, stake: float, strategy: str = STRATEGY,
                    model_version: str | None = None, allow_duplicate: bool = False,
                    now: datetime | None = None) -> tuple[PaperBet, bool]:
    """(bet, created). One simulated entry per strategy, game, player, market, line and side, at whichever book,
    unless `allow_duplicate` asks for a separate entry."""
    same = select(PaperBet).where(PaperBet.strategy == strategy, PaperBet.event_id == opportunity.event_id,
                                  PaperBet.market == opportunity.market, PaperBet.player_key == opportunity.player_key,
                                  PaperBet.line_key == opportunity.line_key, PaperBet.side == opportunity.side)
    existing = session.scalars(same.order_by(PaperBet.entry_no.desc())).first()
    if existing is not None and not allow_duplicate:
        return existing, False
    if stake <= 0:
        raise ValueError("a paper stake must be positive")
    bet = PaperBet(created_at=now or datetime.now(timezone.utc), strategy=strategy, model_version=model_version,
                   entry_no=1 if existing is None else existing.entry_no + 1, opportunity_id=opportunity.id,
                   data_source=opportunity.data_source, sport=opportunity.sport, event_id=opportunity.event_id,
                   event_name=opportunity.event_name, commence_time=opportunity.commence_time, book=opportunity.book,
                   market=opportunity.market, player=opportunity.player, player_key=opportunity.player_key,
                   line=opportunity.line, line_key=opportunity.line_key, side=opportunity.side, price=opportunity.price,
                   decimal=opportunity.decimal, price_observed_at=opportunity.observed_at, probability=opportunity.probability,
                   standard_error=opportunity.standard_error, ev=opportunity.ev, conservative_ev=opportunity.conservative_ev,
                   evidence=opportunity.evidence, stake=float(stake))
    session.add(bet)
    session.flush()
    return bet, True


def utc(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """SQLite returns naive timestamps; everything stored here is UTC."""
    for col in columns:
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def latest_scan(session: Session) -> Scan | None:
    return session.scalars(select(Scan).order_by(Scan.id.desc())).first()


BET = ["event_id", "book", "market", "player_key", "line_key", "side"]


def load_opportunities(engine: Engine, since: datetime | None = None, latest_only: bool = True) -> pd.DataFrame:
    """Stored opportunities observed since `since`. `latest_only` keeps each book's most recent look at a bet:
    an earlier snapshot of the same price is history, not a second opportunity. The superseded column marks a
    row whose game has been looked at again since: its price is no longer current, flagged again or not."""
    query = select(OpportunityRow)
    if since is not None:
        query = query.where(OpportunityRow.observed_at >= since)
    with engine.connect() as conn:
        frame = pd.read_sql(query, conn)
        seen = pd.read_sql(select(EventSeen.event_id, EventSeen.last_observed_at), conn)
    frame = utc(frame, ["commence_time", "observed_at", "last_update"])
    newest = frame.event_id.map(utc(seen, ["last_observed_at"]).set_index("event_id").last_observed_at)
    frame["superseded"] = [bool(pd.notna(n) and o < n) for o, n in zip(frame.observed_at, newest)]
    if latest_only and not frame.empty:
        frame = frame.sort_values("observed_at").drop_duplicates(BET, keep="last")
    return frame.reset_index(drop=True)


def load_quotes(engine: Engine, since: datetime | None = None) -> pd.DataFrame:
    query = select(QuoteRow)
    if since is not None:
        query = query.where(QuoteRow.observed_at >= since)
    with engine.connect() as conn:
        frame = pd.read_sql(query, conn)
    return utc(frame, ["commence_time", "observed_at", "last_update"])


def load_paper_bets(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        frame = pd.read_sql(select(PaperBet).order_by(PaperBet.created_at.desc()), conn)
    return utc(frame, ["created_at", "commence_time", "price_observed_at", "settled_at"])


def counts(session: Session) -> dict:
    return {m.__tablename__: session.scalar(select(func.count()).select_from(m)) for m in (Scan, EventSeen, QuoteRow, OpportunityRow, PaperBet)}
