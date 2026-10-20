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
