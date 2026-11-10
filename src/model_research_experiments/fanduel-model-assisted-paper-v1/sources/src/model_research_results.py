"""Official NBA final-score snapshots for the prospective model experiment.

The legacy raw NBA schedule stopped in April 2026. Do not use it as evidence
that current games are final. Reuse the official NBA endpoint already used by
the live projection job, on dates with prospective decisions needing results.
"""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import psycopg2.extras
from model_research import NAME,rows,number,safe_time,clean,registry

SCHEMA="""CREATE TABLE IF NOT EXISTS bets.model_research_nba_results(
    snapshot_id TEXT PRIMARY KEY,game_id TEXT NOT NULL,game_date DATE NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,payload JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS model_research_nba_result_game ON bets.model_research_nba_results(game_id,observed_at);
"""


def final_game(game):
    if game.get('gameStatus')!=3:return None
    home,away=game.get('homeTeam') or {},game.get('awayTeam') or {}
    hs,avs=number(home.get('score')),number(away.get('score'))
    tip=safe_time(game.get('gameTimeUTC'))
    if (not game.get('gameId') or not tip or not home.get('teamTricode') or not away.get('teamTricode')
            or hs is None or avs is None or hs<=0 or avs<=0 or hs!=int(hs) or avs!=int(avs) or hs==avs):return None
    return dict(game_id=str(game['gameId']),game_date=tip.astimezone(ZoneInfo('America/New_York')).date().isoformat(),
        tip_utc=tip.isoformat(),home_team=home['teamTricode'],away_team=away['teamTricode'],
        home_score=hs,away_score=avs,status='final',source='stats.nba.com/scoreboardv3',raw=game)


def fetch(day):
    from nba_api.stats.endpoints import scoreboardv3
    return scoreboardv3.ScoreboardV3(game_date=day.isoformat(),timeout=30).get_dict()['scoreboard']['games']


def refresh(conn,doc,fetcher=fetch,now=None):
    now=now or datetime.now(timezone.utc)
    with conn.cursor() as cur:cur.execute(SCHEMA)
    # Once a final snapshot exists, do not repeatedly poll it on news runs.
    pending=rows(conn,"""SELECT d.payload->'offer' AS offer FROM bets.model_research_decisions d
        WHERE experiment_id=%s AND experiment_sha256=%s AND sport='nba'
        AND (payload->'offer'->>'commence_time_utc')::timestamptz<%s""",(NAME,doc['sha256'],now))
    saved=rows(conn,'SELECT payload FROM bets.model_research_nba_results')
    dates=set()
    from model_research import nba_team
    for row in pending:
        o=row['offer'];day=safe_time(o['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date()
        if not any(r['payload']['game_date']==day.isoformat() and r['payload']['home_team']==nba_team(o['home_team'])
                   and r['payload']['away_team']==nba_team(o['away_team']) for r in saved):dates.add(day)
    count=0;errors=[]
    for day in sorted(dates):
        try:games=fetcher(day)
        except Exception as exc:errors.append(dict(day=str(day),error=type(exc).__name__));continue
        for g in games:
            result=final_game(g)
            if not result or result['game_date']!=day.isoformat():continue
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO bets.model_research_nba_results(snapshot_id,game_id,game_date,observed_at,payload)
                    VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (registry._digest(result),result['game_id'],day,now,psycopg2.extras.Json(clean(result))))
                count+=cur.rowcount
        conn.commit()
    return dict(status='errors' if errors else 'ok',new_final_snapshots=count,dates_requested=len(dates),errors=errors)


def lookup(conn,offer):
    from model_research import nba_team
    start=safe_time(offer['commence_time_utc'])
    found=rows(conn,"SELECT payload FROM bets.model_research_nba_results WHERE game_date=%s ORDER BY observed_at DESC",
               (start.astimezone(ZoneInfo('America/New_York')).date(),))
    matched=[r['payload'] for r in found if r['payload'].get('status')=='final'
        and r['payload']['home_team']==nba_team(offer['home_team']) and r['payload']['away_team']==nba_team(offer['away_team'])
        and abs((safe_time(r['payload']['tip_utc'])-start).total_seconds())<=900]
    return matched[0] if matched and len({r['game_id'] for r in matched})==1 else None
