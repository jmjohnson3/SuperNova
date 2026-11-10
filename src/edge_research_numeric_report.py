"""Read-only numeric storage adapter for the unchanged frozen rotation report.

PostgreSQL NUMERIC results are decoded as floats, like the model's saved JSON
numbers. This repairs float-minus-Decimal grading without changing any scorer,
decision, outcome rule, close rule or betting authorization.
"""
from contextlib import closing
import argparse

import psycopg2
from psycopg2.extensions import DECIMAL, new_type, register_type

import edge_research_report as original
from edge_research import trial
from nfl_pipeline import experiment_registry as registry
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

NAME='market-rotation-numeric-report-v1'


def registration(create=False):
    parent=original.registration()
    config=dict(parent_reader=original.NAME,parent_sha256=parent['sha256'],
                repair='connection-local NUMERIC-to-float decoding for report arithmetic',
                rescoring=False,decision_mutations=False,betting_authorized=False)
    if create:
        return registry.register_experiment(NAME,config,
            dict(adapter=trial.ROOT/'src/edge_research_numeric_report.py',
                 reader=trial.ROOT/'src/edge_research_report.py',
                 parent=trial.STORE/original.NAME/'registration.json'),root=trial.ROOT,store=trial.STORE)
    doc=registry.check_experiment(NAME,root=trial.ROOT,store=trial.STORE)
    if doc['config']!=config:raise ValueError('Numeric report adapter drift')
    return doc


def configure(conn):
    # Scope the decoder to this read-only connection; collectors/training are unaffected.
    conn.set_session(readonly=True)
    register_type(new_type(DECIMAL.values,'REPORT_NUMERIC',lambda value,cur:float(value) if value is not None else None),conn)


def run():
    adapter=registration();reader=original.registration();doc=trial.active()
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        configure(conn)
        with conn.cursor() as cur:cur.execute("SET statement_timeout='90s'")
        result=original.build(conn,doc,reader)
    result['numeric_storage_adapter']=dict(name=NAME,sha256=adapter['sha256'])
    atomic_json(trial.ROOT/'reports/market_rotation_trial_latest.json',result)
    path=trial.ROOT/'reports/market_rotation_trial_latest.md'
    path.write_text(path.read_text(encoding='utf-8')+f'\nNumeric storage adapter: {NAME}; connection-local decoding only.\n',encoding='utf-8')
    return dict(status='ok',report_reader=original.NAME,numeric_storage_adapter=NAME,
                pairs=len(result['records']),betting_authorized=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--register',action='store_true')
    if ap.parse_args().register:
        d=registration(create=True);print(NAME,d['sha256'])
    else:print(run())


if __name__=='__main__':main()
