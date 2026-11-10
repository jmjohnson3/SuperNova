"""Exact-line consensus excluding the execution book, and constrained model blending."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math

import numpy as np
import pandas as pd
from scipy.special import expit,logit

from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline.markets import normalize_name
from nfl_pipeline import sharp_math

RULES=dict(execution_book='fanduel',minimum_reference_operators=2,max_source_age_seconds=300,
    max_source_skew_seconds=120,minimum_ev=.03,decision_window_minutes=[15,45],
    consensus='equal operator weight in log-odds; excludes FanDuel; exact half-lines only',
    primary_line='closest FanDuel no-vig probability to one half, then absolute line, then line',
    references=['pinnacle','draftkings','betmgm','fanatics','espnbet'],
    model_weight_grid=[0.,.05,.1,.2,.3,.5],minimum_fit_games=100,minimum_fit_days=21,
    minimum_calibration_games=50,minimum_calibration_days=7,
    selection='one first paired offer per sport/event/player/market; evaluate both offered sides',
    automatic_betting_authorization=False)
MARKETS={'spreads':'spread','alternate_spreads':'spread','totals':'total','alternate_totals':'total',
    'player_points':'points','player_rebounds':'rebounds','player_assists':'assists','player_threes':'threes',
    'player_pass_yds':'passing_yards','player_rush_yds':'rushing_yards','player_reception_yds':'receiving_yards',
    'player_receptions':'receptions','player_pass_tds':'passing_tds',
    'batter_hits':'hits','batter_total_bases':'total_bases','batter_home_runs':'home_runs',
    'pitcher_strikeouts':'strikeouts','pitcher_outs':'outs'}


def source_stamp(book_stamp,market_stamp,observed):
    """A market clock cannot be refreshed by an unrelated book update."""
    book=safe_time(book_stamp);market=safe_time(market_stamp)
    if book_stamp is not None and (book is None or book>observed):return None
    source=market if market_stamp is not None else book
    if source is None or not 0<=(observed-source).total_seconds()<=RULES['max_source_age_seconds']:return None
    return source


def event_pairs(event,observed):
    """Canonical side a=home/over. Conflicting duplicate outcomes invalidate a pair."""
    records=defaultdict(list)
    for book in event.get('bookmakers',[]):
        name=book.get('key')
        if name not in [RULES['execution_book'],*RULES['references']]:continue
        for market in book.get('markets',[]):
            stat=MARKETS.get(market.get('key'))
            stamp=source_stamp(book.get('last_update'),market.get('last_update'),observed)
            if stat is None or stamp is None:continue
            for outcome in market.get('outcomes',[]):
                line=number(outcome.get('point'));price=number(outcome.get('price'))
                if line is None or line%1!=.5 or price is None or price!=int(price) or not sharp_math.implied(price):continue
                side=None;player=outcome.get('description')
                if stat=='spread':
                    if outcome.get('name')==event.get('home_team'):side='a'
                    elif outcome.get('name')==event.get('away_team'):side='b';line=-line
                    who=''
                else:
                    side={'over':'a','under':'b'}.get(str(outcome.get('name','')).lower())
                    who='' if stat=='total' else normalize_name(player or '')
                    if stat!='total' and not who:continue
                if side:
                    records[(name,stat,who,line,side)].append(dict(price=int(price),updated=stamp,
                        book_updated=book.get('last_update'),market_updated=market.get('last_update'),player=player))
    pairs={}
    for book,stat,who,line,_ in records:
        key=(book,stat,who,line)
        if key in pairs:continue
        sides=[]
        for side in ('a','b'):
            values=records.get((*key,side),[])
            unique={(v['price'],v['updated']) for v in values}
            sides.append(values[0] if len(unique)==1 else None)
        if all(sides) and sides[0]['updated']==sides[1]['updated']:
            a,b=sides
            pairs[key]=dict(a=a['price'],b=b['price'],updated=a['updated'],player=a['player'],
                book_updated=a['book_updated'],market_updated=a['market_updated'])
    return pairs


def consensus(pair_map,key,execution=None):
    """Each distinct operator supplies one two-way probability, never a shifted line."""
    sources=[]
    for book in RULES['references']:
        q=pair_map.get((book,*key))
        if not q:continue
        if execution and abs((q['updated']-execution['updated']).total_seconds())>RULES['max_source_skew_seconds']:continue
        p=sharp_math.no_vig_over(q['a'],q['b'])
        if p is not None and 0<p<1:sources.append(dict(book=book,probability=p,**q))
    if len(sources)<RULES['minimum_reference_operators']:return None
    stamps=[s['updated'] for s in sources]+([execution['updated']] if execution else [])
    if (max(stamps)-min(stamps)).total_seconds()>RULES['max_source_skew_seconds']:return None
    logits=[logit(s['probability']) for s in sources]
    return dict(probability=float(expit(np.mean(logits))),sources=sources,
        range=float(max(s['probability'] for s in sources)-min(s['probability'] for s in sources)),
        method='observed_exact_line',operators=len(sources))


def offers(event,sport,observed,pair_map=None):
    tip=safe_time(event.get('commence_time'))
    if not tip or not RULES['decision_window_minutes'][0]<=(tip-observed).total_seconds()/60<=RULES['decision_window_minutes'][1]:return []
    pairs=event_pairs(event,observed) if pair_map is None else pair_map
    groups=defaultdict(list)
    for (book,stat,who,line),q in pairs.items():
        if book!='fanduel':continue
        p=sharp_math.no_vig_over(q['a'],q['b'])
        if p is not None:groups[(stat,who)].append((abs(p-.5),abs(line),line,q,p))
    output=[]
    for (stat,who),options in groups.items():
        _,_,line,q,fdp=min(options,key=lambda r:r[:3])
        ref=consensus(pairs,(stat,who,line),q)
        output.append(dict(sport=sport,event_id=event['id'],game_id=event.get('game_id'),
            home_team=event['home_team'],away_team=event['away_team'],commence_time_utc=tip,
            decided_at=observed,book='fanduel',market=stat,player_norm=who,player=q.get('player'),
            line=line,price_a=q['a'],price_b=q['b'],fanduel_probability=fdp,execution_quote=q,
            consensus=ref,reference_status='ok' if ref else 'insufficient_fresh_exact_references'))
    return output


def blend(reference,model,weight):
    if model is None or not math.isfinite(float(model)):return float(reference)
    return float(expit((1-weight)*logit(np.clip(reference,1e-6,1-1e-6))+weight*logit(np.clip(model,1e-6,1-1e-6))))


def selection(offer,probability,eligible=True):
    if number(probability) is None or not 0<=probability<=1:return dict(selected=False,reason='probability_unavailable')
    a,b=sharp_math.ev(probability,offer['price_a']),sharp_math.ev(1-probability,offer['price_b'])
    is_a=a>=b;side=('home' if is_a else 'away') if offer['market']=='spread' else ('over' if is_a else 'under')
    return dict(selected=bool(eligible and max(a,b)>=RULES['minimum_ev']),side=side,
        line=offer['line'] if is_a or offer['market']!='spread' else -offer['line'],
        price=offer['price_a'] if is_a else offer['price_b'],probability=probability if is_a else 1-probability,
        ev=max(a,b),reason='eligible' if eligible else 'model_context_unverified')


def fit_weight(frame):
    """Development-only weight selection, with zero an explicit competitive candidate."""
    f=frame.dropna(subset=['reference','model','outcome']).copy() if len(frame) else frame
    games=f.event_id.nunique() if len(f) else 0
    days=(pd.to_datetime(f.day).max()-pd.to_datetime(f.day).min()).days+1 if len(f) else 0
    if games<RULES['minimum_fit_games'] or days<RULES['minimum_fit_days']:
        return dict(weight=0.,status='insufficient_earlier_evidence',rows=len(f),games=int(games),days=int(days))
    # Give each game equal total weight; multiple player props are correlated.
    sample_weight=1/f.groupby('event_id').event_id.transform('size')
    scored=[]
    for weight in RULES['model_weight_grid']:
        p=expit((1-weight)*logit(f.reference.clip(1e-6,1-1e-6))+weight*logit(f.model.clip(1e-6,1-1e-6)))
        scored.append((float(np.average((p-f.outcome)**2,weights=sample_weight)),weight))
    loss,weight=min(scored)  # exact ties choose the smaller model weight
    return dict(weight=weight,status='fit_on_earlier_game_weighted_brier',rows=len(f),games=int(games),days=int(days),
        scores=[dict(weight=w,brier=l) for l,w in scored])
