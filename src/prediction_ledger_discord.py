"""Publish/update the authorized NBA/NFL/NCAA ledger in the existing #record-ledger.

One persistent message; no mentions, no sportsbook requests, no changes to MLB posts.
python -m prediction_ledger_discord [--dry-run]
"""
from contextlib import closing
from datetime import datetime,timedelta,timezone
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import psycopg2
import requests

from prediction_ledger import ROOT,ET,summary
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from supernovabets_config import _saved_windows_env

CHANNEL_ID='1520157669153964152'
GUILD_ID='1380268843821170688'
ENV='MLB_RECORD_LEDGER_DISCORD_WEBHOOK_URL'
STATE=ROOT/'reports/predictions_ledger_discord_state.json'
TITLE='**Predictions ledger — NBA / NFL / NCAA**'


def record_text(rows):
    s=summary(rows);rate=f"{s['accuracy']:.1%}" if s['accuracy'] is not None else 'not graded yet'
    return f"{s['win']} right / {s['loss']} wrong | {rate} | {s['push']} push, {s['void']} void, {s['pending']} pending"


def latest_group(rows):
    """Choose the latest saved book/version by forecast time, never by its results."""
    if not rows:return [],None,None
    def order(r):
        at=datetime.fromisoformat(r['forecast_at']) if r.get('forecast_at') else datetime.fromisoformat(r['date']).replace(tzinfo=timezone.utc)
        return at.timestamp(),r['book'],r['version']
    latest=max(rows,key=order);book,version=latest['book'],latest['version']
    return [r for r in rows if r['book']==book and r['version']==version],book,version


def message(doc):
    # Every sport is visible. Different books, strategies and versions remain
    # separate; these are descriptive paper records, never a pooled betting claim.
    nba=[r for r in doc['records'] if r['cohort']=='Delivered price alerts' and r['sport']=='NBA'
         and r['book']=='fanduel' and r['version']=='nba-prop-price-check-v1']
    yesterday=(datetime.fromisoformat(doc['built_at']).astimezone(ET).date()-timedelta(days=1)).isoformat()
    daily=[r for r in nba if r['date']==yesterday]
    nfl,_,nfl_version=latest_group([r for r in doc['records'] if r['sport']=='NFL'
        and r['cohort']=='Locked model picks / paper' and r['book']=='fanduel' and r['result']!='excluded'])
    ncaa,ncaa_book,ncaa_version=latest_group([r for r in doc['records'] if r['sport']=='NCAAF'
        and r['cohort']=='Game model vs saved line' and r['result']!='excluded'])
    text=[TITLE,f"Games: {doc['since']} through {doc['through']} (ET)",
        f"Updated: <t:{int(datetime.fromisoformat(doc['built_at']).timestamp())}:f>",
        '', '**FanDuel NBA delivered paper alerts**', 'All recorded: '+record_text(nba),
        f"{yesterday}: "+record_text(daily), '',
        '**NFL — FanDuel model paper picks**',
        f"Latest saved version: `{nfl_version}`" if nfl_version else 'No saved model picks.',
        'Player props: '+record_text([r for r in nfl if r['market'] not in ('spread','total')]),
        'Spreads/totals: '+record_text([r for r in nfl if r['market'] in ('spread','total')]), '',
        '**NCAA football — model vs saved lines**',
        f"Reference book: {ncaa_book}; version: {ncaa_version}" if ncaa_book else 'No saved game predictions.',
        'Spreads: '+record_text([r for r in ncaa if r['market']=='spread']),
        'Totals: '+record_text([r for r in ncaa if r['market']=='total']), '',
        '**Full records attached**',
        '`prediction-summary.txt` + `predictions-ledger.zip` (extract and open `predictions_ledger.html`).',
        '', 'Repeated picks count once. Pushes/voids/pending are excluded from hit rate. Other versions and books stay separate in the attachments.',
        '**Paper/research results — not actual wagers or net profit.**',
        'This message updates when saved predictions or results change.']
    content='\n'.join(text)
    if len(content)>2000:raise ValueError('Ledger message exceeds Discord content limit')
    return content


def package(folder):
    doc=json.loads((folder/'predictions_ledger_latest.json').read_text(encoding='utf-8'))
    html=(folder/'predictions_ledger_latest.html').read_bytes()
    md=(folder/'predictions_ledger_latest.md').read_text(encoding='utf-8')
    text=md.replace('[Open the filterable ledger](predictions_ledger_latest.html)',
        'Interactive ledger: extract predictions-ledger.zip and open predictions_ledger.html.').encode('utf-8')
    # Ignore the refresh heartbeat; changed rows/grades, scope and reporting code count.
    stable={k:v for k,v in doc.items() if k!='built_at'}
    # A new message layout must publish even when prediction data is unchanged.
    stable['publisher_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('predictions_ledger.html',html)
        z.writestr('predictions_ledger.json',json.dumps(doc,ensure_ascii=False,indent=2))
        z.writestr('prediction-summary.txt',text)
    files=[('prediction-summary.txt',text,'text/plain; charset=utf-8'),('predictions-ledger.zip',archive.getvalue(),'application/zip')]
    if any(len(body)>8_000_000 for _,body,_ in files):raise ValueError('Ledger attachment too large; split explicitly rather than truncating rows')
    return doc,digest,files


def check_response(response,expected=(200,)):
    if response.status_code not in expected:raise RuntimeError(f'Discord ledger request failed: HTTP {response.status_code}')
    return response.json()


def sync(folder,state_path,url,session=requests):
    doc,digest,files=package(folder)
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    if state.get('status')=='pending_create' and not state.get('message_id'):
        raise RuntimeError('Previous Discord creation has an unknown outcome; reconcile it before creating another message')
    meta=check_response(session.get(url,timeout=20))
    if str(meta.get('channel_id'))!=CHANNEL_ID or str(meta.get('guild_id'))!=GUILD_ID:
        raise RuntimeError('Ledger webhook no longer points to the authorized record-ledger channel')
    message_id=state.get('message_id')
    if message_id and (state.get('channel_id')!=CHANNEL_ID or state.get('webhook_id')!=meta['id']):
        raise RuntimeError('Saved ledger message belongs to a different route')
    if message_id and state.get('fingerprint')==digest:
        return dict(status='unchanged',message_url=state['message_url'])
    payload=dict(content=message(doc),allowed_mentions={'parse':[]},
        attachments=[dict(id=i,filename=name) for i,(name,_,_) in enumerate(files)])
    multipart={f'files[{i}]':(name,body,mime) for i,(name,body,mime) in enumerate(files)}
    base=dict(channel_id=CHANNEL_ID,guild_id=GUILD_ID,webhook_id=meta['id'],message_id=message_id,
        attempted_at=datetime.now(timezone.utc).isoformat(),status='updating' if message_id else 'pending_create')
    atomic_json(state_path,base)
    if message_id:
        response=session.patch(url.rstrip('/')+'/messages/'+message_id,data={'payload_json':json.dumps(payload)},files=multipart,timeout=45)
    else:
        response=session.post(url,params={'wait':'true'},data={'payload_json':json.dumps(payload)},files=multipart,timeout=45)
    if 400<=response.status_code<500:
        atomic_json(state_path,dict(base,status='rejected',http_status=response.status_code))
    sent=check_response(response)
    if str(sent.get('channel_id'))!=CHANNEL_ID or not sent.get('id'):raise RuntimeError('Discord ledger response route/identity mismatch')
    mid=sent['id'];link=f'https://discord.com/channels/{GUILD_ID}/{CHANNEL_ID}/{mid}'
    # Save the ID before verification, so a read timeout cannot cause another create.
    saved=dict(base,message_id=mid,message_url=link,status='awaiting_verification')
    atomic_json(state_path,saved)
    verified=check_response(session.get(url.rstrip('/')+'/messages/'+mid,timeout=20))
    names={a.get('filename') for a in verified.get('attachments',[])}
    if verified.get('content')!=payload['content'] or names!={f[0] for f in files}:
        raise RuntimeError('Discord ledger content/attachments did not verify')
    atomic_json(state_path,dict(saved,status='published',fingerprint=digest,report_built_at=doc['built_at'],
        verified_at=datetime.now(timezone.utc).isoformat(),attachments=sorted(names)))
    return dict(status='updated' if message_id else 'posted',message_url=link,attachments=sorted(names))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(86731445)')
            if not cur.fetchone()[0]:print('Ledger build/post already running; retry next cycle');return
        if args.dry_run:
            doc,digest,files=package(ROOT/'reports')
            print(message(doc));print(json.dumps(dict(fingerprint=digest,files=[dict(name=n,bytes=len(b)) for n,b,_ in files])));return
        url=_saved_windows_env(ENV) or os.getenv(ENV,'')
        if not url:raise RuntimeError('record-ledger webhook is not configured')
        try:result=sync(ROOT/'reports',STATE,url)
        except requests.RequestException as exc:
            raise RuntimeError('Discord ledger network failure: '+type(exc).__name__) from None
    print(json.dumps(result))


if __name__=='__main__':main()
