"""Render frozen sharp evidence without labeling paper reservations as actual money.

The frozen scanner, selection and statistical calculations remain in their pinned
modules. This reporting layer exposes monetary totals only in the explicit legacy
confirmed-wager section. Receipt P&L is owned by wager_reconciliation.
"""
import json
from datetime import datetime, timezone

import psycopg2

from nfl_pipeline.modeling import sharp_alert_report as frozen


def separate_money(summary):
    """Suggested stakes on legacy alerts are not confirmed executed wagers."""
    result = dict(summary)
    def paper(metrics):
        return {key: value for key, value in metrics.items()
                if key not in {'staked', 'realized_profit', 'mean_slippage'}}
    result = paper(result)
    for key in ('logged', 'early'):
        if key in result:
            result[key] = paper(result[key])
    for key in ('by_book', 'model'):
        if key in result:
            result[key] = {name: paper(metrics) for name, metrics in result[key].items()}
    result['execution_accounting'] = 'Paper stakes excluded; placed is legacy confirmation only; receipt net profit is separate.'
    return result


def main():
    with psycopg2.connect(frozen.PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = frozen.grade(frozen.load(conn))
    summary = separate_money(frozen.summarize(frame))
    document = dict(built_at=datetime.now(timezone.utc).isoformat(), pass_bar=frozen.PASS, summary=summary,
                    reporting_contract='nfl-sharp-paper-money-separation-v1')
    frozen.atomic_json(frozen.ROOT/'reports/nfl_sharp_alerts_latest.json', document)
    (frozen.ROOT/'reports/nfl_sharp_alerts_latest.md').write_text(frozen.markdown(summary, frame), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
