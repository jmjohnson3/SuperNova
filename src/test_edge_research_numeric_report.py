from decimal import Decimal
import os

import psycopg2
import pytest

from edge_research_numeric_report import configure
from nfl_pipeline.db import PG_DSN

pytestmark=pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in local database test')


def test_numeric_decoding_is_read_only_and_connection_local():
    with psycopg2.connect(PG_DSN,connect_timeout=5) as adapted, psycopg2.connect(PG_DSN,connect_timeout=5) as normal:
        configure(adapted)
        with adapted.cursor() as x,normal.cursor() as y:
            x.execute('SELECT 2.5::numeric,NULL::numeric,current_setting(\'transaction_read_only\')')
            value,missing,readonly=x.fetchone()
            assert isinstance(value,float) and 3.0-value==.5 and missing is None and readonly=='on'
            y.execute('SELECT 2.5::numeric')
            assert isinstance(y.fetchone()[0],Decimal)
