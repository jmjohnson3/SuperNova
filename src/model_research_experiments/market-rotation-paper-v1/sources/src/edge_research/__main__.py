"""Run fixed market/rotation paper experiments; no alerts, bets or paid API requests."""
from contextlib import closing
import argparse
import json
import time

import psycopg2

from edge_research import trial,feeds
from nfl_pipeline.db import PG_DSN
from nba_props.research_trial import clean


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['train-rotation','finish-mean','train-weights','register','verify','project','capture','report','availability'])
    ap.add_argument('--sport',choices=feeds.SUPPORTED,nargs='+',default=list(feeds.SUPPORTED))
    args=ap.parse_args()
    if args.command in ('register','verify'):
        doc=trial.register() if args.command=='register' else trial.active()
        print(json.dumps(dict(experiment_id=trial.NAME,sha256=doc['sha256'],registered_at=doc['registered_at'])));return
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        deadline=time.monotonic()+30
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='90s'")
            while True:
                cur.execute('SELECT pg_try_advisory_lock(%s)',(trial.LOCK,))
                if cur.fetchone()[0]:break
                if time.monotonic()>=deadline:raise TimeoutError('Market/rotation task still running')
                time.sleep(.25)
        trial.init(conn)
        if args.command in ('train-rotation','finish-mean'):
            from edge_research import rotation_study
            result=rotation_study.train(conn) if args.command=='train-rotation' else rotation_study.finish_mean_ablation(conn)
        elif args.command=='train-weights':
            from edge_research import study
            result=study.train(conn)
        elif args.command=='availability':
            from edge_research import reporting
            result=reporting.availability_audit(conn)
        else:
            doc=trial.active()
            if args.command=='report':
                from edge_research import reporting
                result=reporting.build(conn,doc)
            else:
                art=trial.artifact()
                result=trial.project(conn,doc,art) if args.command=='project' else trial.capture(conn,doc,art,args.sport)
        if args.command in ('report','availability','train-weights','train-rotation','finish-mean'):
            result=dict(status='ok',step=args.command,report_saved=True)
        print(json.dumps(clean(result),indent=2))


if __name__=='__main__':main()
