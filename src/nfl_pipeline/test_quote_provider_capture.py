"""Exercise the real quote SELECT and immutable capture, using a temporary table."""
from copy import deepcopy
from datetime import date
import json

import pytest

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.modeling import predict_player_props as predictor, scoring_capture
from nfl_pipeline.modeling import receiving_paper_trial as trial
from nfl_pipeline.test_receiving_paper_trial import NOW, manifest, record


def test_loaded_provider_survives_capture_without_changing_side_probability_or_ev(monkeypatch):
    import psycopg2
    from nfl_pipeline.db import PG_DSN
    from nfl_pipeline import cash_freeze

    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
    except psycopg2.OperationalError:
        pytest.skip('Local PostgreSQL unavailable for temporary-table quote integration test')
    original = record()
    offer = deepcopy(original['forecast_payload']['scoring_replay']['offer'])
    reference, issue = trial.price_record(original, manifest(), NOW)
    assert issue is None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='10s'")
            cursor.execute('''CREATE TEMP TABLE paper_provider_quote (
                id bigint,provider text,as_of_date date,fetched_at_utc timestamptz,
                event_id text,commence_time_utc timestamptz,bookmaker_key text,bookmaker_title text,
                home_team text,away_team text,player_name text,player_name_norm text,
                market_key text,stat text,line numeric,over_price int,under_price int,
                over_link text,under_link text,snapshot_role text) ON COMMIT DROP''')
            cursor.execute('''INSERT INTO paper_provider_quote VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                (offer['offer_id'],offer['provider'],date(2026,9,22),offer['fetched_at_utc'],
                 offer['event_id'],offer['commence_time_utc'],offer['bookmaker_key'],'FanDuel',
                 offer['home_team'],offer['away_team'],'Test Player',offer['player_name_norm'],
                 offer['market_key'],offer['stat'],offer['line'],offer['over_price'],offer['under_price'],
                 offer['over_link'],offer['under_link'],'lock'))

        class Cursor:
            def __init__(self, **kwargs): self.cursor = conn.cursor(**kwargs)
            def __enter__(self): return self
            def __exit__(self, *args): self.cursor.close()
            def execute(self, sql, params):
                # Use the production projection/filter/order against an isolated
                # fixture; no persistent table or original quote is changed.
                self.cursor.execute(sql.replace('FROM odds.nfl_player_prop_lines',
                                                'FROM pg_temp.paper_provider_quote'), params)
            def fetchall(self): return self.cursor.fetchall()

        class Connection:
            def cursor(self, **kwargs): return Cursor(**kwargs)

        loaded = predictor._load_prop_lines(Connection(), date(2026,9,22),
                                            safe_time(original['forecast_payload']['prediction_context_cutoff_utc']))
        assert len(loaded) == 1 and loaded[0]['provider'] == offer['provider']
        fingerprint = manifest()['config']['scoring_fingerprint']
        monkeypatch.setattr(scoring_capture, 'scoring_fingerprint', lambda: fingerprint)
        monkeypatch.setattr(cash_freeze, 'capture_provenance', lambda: None)
        captured = scoring_capture.capture({},'receiving_yards',40,40,{},loaded[0],{}, {},{},0,{})
        changed = deepcopy(original)
        # forecast_store.save_forecasts serializes datetime/date quote fields
        # this way before the trial reads the immutable JSONB record.
        changed['forecast_payload']['scoring_replay'] = json.loads(json.dumps(captured,default=str,allow_nan=False))
        decision, issue = trial.price_record(changed, manifest(), NOW)
        assert issue is None
        assert decision['quote']['provider'] == offer['provider']
        for field in ('side','price','probability','push_probability','market_probability','ev'):
            assert decision[field] == reference[field]
        changed['forecast_payload']['scoring_replay']['offer'].pop('provider')
        assert trial.price_record(changed, manifest(), NOW)[1] == 'missing_quote_provider'
    finally:
        conn.rollback()
        conn.close()
