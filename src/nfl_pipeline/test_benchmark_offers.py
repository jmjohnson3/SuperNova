from datetime import datetime, timezone

import pytest
import pandas as pd

from nfl_pipeline.modeling import benchmark_offers as module

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)


def record():
    return dict(id=1, game_id='game', player_id='player', stat='receiving_yards', season=2026,
        week=3, created_at_utc='2026-09-22T10:01:00Z', start_ts_utc='2026-09-22T20:00:00Z',
        offer_fetched_at='2026-09-22T09:59:00Z', actual=20, status='final',
        forecast_payload=dict(game_id='game', player_id='player', team_abbr='MIA', position='WR',
            game_date_et='2026-09-22', stat='receiving_yards', model_version='production',
            prediction_context_cutoff_utc='2026-09-22T10:00:00Z', book='fanduel', line=30.5,
            side='over', price=-114, probability=.65, projection=40,
            forecast_features={'receiving_yards_avg_5': 40}, scoring_replay={'version': 'test'}))


def artifact():
    return dict(created_at='2026-09-21T12:00:00Z', training_end='2026-09-17', production_release='production')


def ledger():
    return dict(ledger_id=9, prediction_id=1, ledger_book='fanduel', ledger_side='over',
        ledger_model_version='production', ledger_line=30.5, ledger_price=-114,
        locked_at_utc='2026-09-22T11:00:00Z')


def document():
    r = record(); p = r['forecast_payload']
    row = dict(forecast_id=1, variant='selected_policy', stat='receiving_yards', production_release='production',
        source_context_cutoff=p['prediction_context_cutoff_utc'], source_book='fanduel', source_side='over',
        source_line=30.5, source_price=-114, production_probability=.65, same_side_probability=.55,
        raw_over_probability=.6, ledger_ids=[9], expected_yards=38, median_yards=32,
        distribution_mean=38, live_p10=1, live_p90=80, candidate_side='under')
    return dict(contract=module.CONTRACT, run_id='benchmark-test', scored_at=NOW.isoformat(),
                artifact_created_at=artifact()['created_at'], training_end='2026-09-17', rows=[row])


def test_prospective_locks_require_artifact_before_context_and_scoring_before_start():
    assert module.prospective_error(record(), artifact(), NOW) is None
    a = artifact(); a['created_at'] = '2026-09-22T10:30:00Z'
    assert module.prospective_error(record(), a, NOW) == 'artifact_or_scoring_not_prospective_at_original_lock'
    assert module.prospective_error(record(), artifact(), datetime(2026, 9, 23, tzinfo=timezone.utc))
    a = artifact(); a['training_end'] = '2026-09-22'
    assert module.prospective_error(record(), a, NOW) == 'training_includes_forecast_date'


def test_original_micro_identity_not_latest_forecast():
    assert module.ledger_matches(ledger(), record(), NOW)
    for key, value in [('ledger_price', 110), ('ledger_line', 40.5), ('prediction_id', 2), ('ledger_book', 'draftkings')]:
        l = ledger(); l[key] = value
        assert not module.ledger_matches(l, record(), NOW)


def test_projection_only_is_not_a_missing_capture_failure():
    r=record()
    r['forecast_payload'].update(line=None,side=None,scoring_replay=None)
    assert module.prospective_error(r,artifact(),NOW)=='not_an_offered_line'
    r=record(); r['forecast_payload']['scoring_replay']=None
    assert module.prospective_error(r,artifact(),NOW)=='missing_exact_lock_inputs'


def test_scoring_after_outcome_cannot_count_as_prospective():
    d = document(); d['scored_at'] = '2026-09-23T12:00:00Z'
    rows, issues, pending = module.validate_documents([record()], [ledger()], [d])
    assert not rows and issues['invalid_original_offer_or_timing'] == 1


def test_exact_micro_fixed_original_side_even_if_challenger_flips():
    rows, issues, pending = module.validate_documents([record()], [ledger()], [document(), document()])
    assert len(rows) == 1 and not issues and pending == 0
    assert rows[0]['outcome'] == 0
    scores = module.offered_metrics(rows)
    micro = next(iter(scores.values()))['exact_micro']
    assert micro['rows'] == 1
    assert micro['final']['brier'] == pytest.approx(.55**2)
    assert micro['hypothetical_side_flips'] == 1
    assert micro['confirmation'] == 'insufficient_prospective_evidence'


def test_mismatched_ledger_is_excluded_not_replaced_by_paper_selection():
    l = ledger(); l['prediction_id'] = 2
    rows, _, _ = module.validate_documents([record()], [l], [document()])
    scores = module.offered_metrics(rows)
    scopes = next(iter(scores.values()))
    assert scopes['real_offers']['rows'] == 1
    assert scopes['exact_micro']['rows'] == 0


def test_pending_and_push_do_not_create_losses():
    r = record(); r['actual'] = None
    rows, _, pending = module.validate_documents([r], [ledger()], [document()])
    assert not rows and pending == 1
    r['actual'] = 30.5
    rows, issues, pending = module.validate_documents([r], [ledger()], [document()])
    assert not rows and issues['push_not_binary'] == 1


def test_each_offer_keeps_its_own_lock_context(monkeypatch):
    first = record(); second = record()
    first['forecast_payload']['context_evidence'] = {'test_out': 0.}
    second['forecast_payload']['context_evidence'] = {'test_out': 1.}
    monkeypatch.setattr(module, 'historical_features', lambda h, r:
                        pd.DataFrame({'wd_injury_out': [0.], 'wd_history': [5.]}, index=r.index))
    monkeypatch.setattr(module, 'context_inputs', lambda r: {'rr_injury_out': r['context_evidence']['test_out']})
    frame = module.scoring_frame([first, second], {'history': pd.DataFrame()})
    assert frame.wd_injury_out.tolist() == [0., 1.]
    assert frame.wd_history.tolist() == [5., 5.]


def test_v1_shadow_evidence_survives_benchmark_upgrade():
    doc = document(); doc['contract'] = 'nfl-shared-yardage-benchmark-v1'
    rows, issues, _ = module.validate_documents([record()], [ledger()], [doc])
    assert len(rows) == 1 and not issues


def test_matched_variants_use_same_ids_and_verified_micro_on_both_sides():
    doc = document()
    extra = dict(doc['rows'][0], variant='stable_ensemble', same_side_probability=.4)
    doc['rows'].append(extra)
    rows, _, _ = module.validate_documents([record()], [ledger()], [doc])
    scores = module.matched_variant_comparisons(rows)
    assert len(scores) == 2
    for result in scores.values():
        assert result['rows'] == 1
        assert result['reference']['brier'] == pytest.approx(.55**2)
        assert result['challenger']['brier'] == pytest.approx(.4**2)
    rows[0]['micro_ids'] = []
    assert len(module.matched_variant_comparisons(rows)) == 1


def test_intermediate_probabilities_use_original_side_not_candidate_side():
    doc=document()
    doc['rows'][0]['probability_trace']=dict(raw_over=.6,heuristic_over=.58,
        post_exact_side=.42,final_side=.45,side='under')
    rows,_,_=module.validate_documents([record()],[ledger()],[doc])
    metrics=next(iter(module.offered_metrics(rows).values()))['exact_micro']['scoring_stages']
    assert metrics['post_exact_side']['brier']==pytest.approx(.58**2)
    assert metrics['final_side']['brier']==pytest.approx(.55**2)
