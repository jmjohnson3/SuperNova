from copy import deepcopy

from nfl_pipeline.modeling.sharp_evidence_report import separate_money


def test_paper_suggestions_cannot_be_reported_as_actual_money():
    metrics = dict(alerts=3, roi=.15, staked=12, realized_profit=2, mean_slippage=.01)
    source = dict(metrics, logged=dict(metrics), early=dict(metrics),
                  by_book={'fanduel':dict(metrics), 'draftkings':dict(metrics)},
                  model={'agrees':dict(metrics)}, placed={'alerts':0})
    original = deepcopy(source)
    report = separate_money(source)
    assert source == original
    for scope in [report, report['logged'], report['early'], *report['by_book'].values(), *report['model'].values()]:
        assert 'staked' not in scope and 'realized_profit' not in scope and 'mean_slippage' not in scope
        assert scope['roi'] == .15
    assert report['placed'] == {'alerts':0}


def test_explicit_legacy_confirmation_section_is_kept_separate():
    placed = dict(alerts=1, staked=1., realized_profit=-1., mean_slippage=.02)
    report = separate_money({'alerts':2, 'placed':placed})
    assert report['placed'] == placed
    assert 'receipt net profit is separate' in report['execution_accounting']


def test_indexed_close_reader_equals_frozen_reader_with_mixed_identity_and_ties():
    import pandas as pd
    from nfl_pipeline.modeling import sharp_alert_report as frozen
    from nfl_pipeline.modeling.test_sharp_alert_report import alert, closes
    from nfl_pipeline.modeling.sharp_evidence_report import attach_indexed
    rows = [alert(), alert(game_id='other', event_id='other'), alert(stat='receptions', fd_line=4.5)]
    rows += [dict(rows[0], initial_provenance={}), dict(rows[1], player_name_norm=None)]
    quotes = [q for row in rows[:3] for q in closes(row)]
    quotes += [dict(quotes[0], event_id='wrong'), dict(quotes[0], stat='rushing_yards'),
               dict(quotes[0], player_name_norm='another'), dict(quotes[0], line=49.5)]
    frame = pd.DataFrame(rows)
    expected = frozen.attach_exact_evidence(frame, quotes)
    actual = attach_indexed(frame, quotes)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frozen.grade(actual), frozen.grade(expected))


def test_indexed_loader_retains_frozen_sql_and_observation_sequence():
    import inspect
    import re
    from nfl_pipeline.modeling.sharp_evidence_report import load_indexed
    from nfl_pipeline.modeling.sharp_alert_report import load
    # The optimization must never silently change result participation or source rows.
    # Main query starts on its own line; compare the two multi-line SELECTs directly.
    def selects(fn):
        return [re.sub(r'\s+', ' ', s).strip() for s in re.findall(r'"""\s*(SELECT.*?)"""', inspect.getsource(fn), re.S)]
    assert selects(load_indexed) == selects(load)


def test_refreshed_legacy_trial_report_cannot_restart_cash():
    from nfl_pipeline.modeling.sharp_evidence_report import paper_only
    source = dict(alerts=26, readiness=dict(status='trial', pause_reasons=[]))
    result = paper_only(source)
    assert result['readiness']['status'] == 'paused'
    assert result['betting_authorized'] is False
    assert result['frozen_readiness_diagnostic'] == source['readiness']
    assert source['readiness']['status'] == 'trial'
    paused = paper_only(dict(readiness=dict(status='paused', pause_reasons=['established loss pause'])))
    assert 'established loss pause' in paused['readiness']['pause_reasons']


def test_estimated_closes_are_not_presented_as_observed_reference_closes():
    from nfl_pipeline.modeling.sharp_evidence_report import paper_only
    result=paper_only(dict(readiness={},by_book=dict(fanduel=dict(alerts=7,with_sharp_close=7,shifted_sharp_closes=4))))
    assert result['exact_reference_by_book']['fanduel']==dict(decisions=7,observed_exact=3,estimated=4)
