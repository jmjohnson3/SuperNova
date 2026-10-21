"""Independent v2 PAPER decisions, immutable populations and own-side closes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from nfl_pipeline.modeling import receiving_paper_trial as trial
from nfl_pipeline.test_prospective_market_trial import paired_record

NOW = datetime(2026, 9, 22, 10, 10, tzinfo=timezone.utc)
COMPONENT = "receiving-oe-v2-test"


def manifest():
    artifact = dict(run_id=COMPONENT, production_release="production", training_end="2026-09-17")
    return dict(sha256="registration", registered_at="2026-09-22T09:50:00Z",
                config=trial.registration_config(artifact, "artifact-checksum"))


def record(index=1, game="game", weights=(.7,.3), line=30.5):
    r=paired_record(); r.update(id=index, game_id=game, player_id="player"+str(index), status="scheduled", actual=None)
    p=r["forecast_payload"]
    p.update(game_id=game, player_id=r["player_id"], player_name="Test Player", opponent_abbr="BUF",
             roster_status="ACT", usage_context_quality=.80, limited_usage_risk=.10,
             drift_guard_pass=False, probability=.5, side="over", price=-114, line=line,
             offer_player_name_norm="test player")
    p["forecast_features"].update(player_name="Test Player", team_abbr="MIA", opponent_abbr="BUF",
        same_week_injury_context_known=1., starter_role_stability_score=.8, route_data_source="proxy_history")
    p["context_evidence"] = dict(cutoff=p["prediction_context_cutoff_utc"],
        roster_observed_at="2026-09-21T10:00:00Z", depth_observed_at="2026-09-21T10:00:00Z",
        injury_status="healthy", injury_observed_at="2026-09-21T10:00:00Z")
    p["scoring_replay"]["offer"].update(provider="sportsgameodds",home_team="MIA",away_team="BUF",line=line,
        player_name_norm="test player",
        under_link="https://sportsbook.fanduel.com/addToBetslip?selection=2")
    p["stat_model_challengers"]={COMPONENT:dict(component_id=COMPONENT,outputs=["probability"],
        artifact_sha256="artifact-checksum",experiment_generation=dict(
            artifact_sha256="artifact-checksum",registration_sha256="registration",
            scoring_fingerprint=manifest()["config"]["scoring_fingerprint"],
            started_at="2026-09-22T10:00:20Z",verified_at="2026-09-22T10:00:30Z"),
        same_offer=True,line=line,curve=dict(values=[10.,80.],weights=list(weights)))}
    p["scoring_replay"]["scoring_fingerprint"]=manifest()["config"]["scoring_fingerprint"]
    return r


def test_paper_side_and_ev_are_independent_of_production_side_drift_and_market_weight():
    r=record(); row, issue=trial.price_record(r,manifest(),NOW)
    assert issue is None
    assert row["side"] == "under" and row["production_side"] == "over"
    assert row["production_drift_guard_pass"] is False
    assert row["price"] == -106
    assert row["probability"] == pytest.approx(.7)
    assert row["ev"] == pytest.approx(.7*(100/106)-.3)
    assert row["production_probability"] == .5
    assert row["market_probability"] == pytest.approx((106/206)/((114/214)+(106/206)))
    rows,_=trial.choose([r],manifest(),NOW)
    assert rows[0]["selected"] and rows[0]["betting_approved"] is False
    r["forecast_payload"].update(drift_guard_pass=True,probability=.99,ev=100,model_probability_weight=1)
    other,_=trial.price_record(r,manifest(),NOW)
    assert (other["side"],other["probability"],other["ev"]) == (row["side"],row["probability"],row["ev"])


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r["forecast_payload"].update(player_id="wrong"),"forecast_identity_mismatch"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(player_name_norm="wrong"),"captured_quote_identity_mismatch"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(away_team="NE"),"quote_team_identity_mismatch"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(under_price=None),"missing_or_invalid_paired_price"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(pair_quality="synthetic"),"not_true_same_book_pair"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(is_alt_line=True),"not_common_full_game_line"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(provider=None),"missing_quote_provider"),
    (lambda r: r["forecast_payload"]["scoring_replay"]["offer"].update(fetched_at_utc="2026-09-22T09:00:00Z"),"stale_or_future_quote"),
    (lambda r: r["forecast_payload"].update(roster_status="IR"),"inactive_or_unknown_roster_status"),
    (lambda r: r["forecast_payload"]["context_evidence"].update(injury_status="Out"),"inactive_or_unknown_roster_status"),
    (lambda r: r["forecast_payload"]["context_evidence"].update(depth_observed_at="2026-09-23T10:00:00Z"),"missing_or_future_context_evidence"),
    (lambda r: r["forecast_payload"]["forecast_features"].update(same_week_injury_context_known=None),"same_week_injury_context_unknown"),
    (lambda r: r["forecast_payload"].update(usage_context_quality=.3),"usage_context_quality_insufficient"),
    (lambda r: r["forecast_payload"]["stat_model_challengers"][COMPONENT].update(line=50.5),"missing_pinned_same_offer_curve"),
    (lambda r: r["forecast_payload"]["stat_model_challengers"][COMPONENT].update(artifact_sha256="temporary-other-artifact"),"unverified_frozen_shadow_generation"),
    (lambda r: r["forecast_payload"]["stat_model_challengers"][COMPONENT].update(experiment_generation=None),"unverified_frozen_shadow_generation"),
    (lambda r: r["forecast_payload"]["scoring_replay"].update(scoring_fingerprint="temporary-other-code"),"unverified_frozen_shadow_generation"),
    (lambda r: r["forecast_payload"]["stat_model_challengers"][COMPONENT]["curve"].update(weights=[.7,.4]),"invalid_pinned_distribution"),
])
def test_independent_path_retains_information_guards(mutate,reason):
    r=record(); mutate(r)
    assert trial.price_record(r,manifest(),NOW)==(None,reason)


def test_side_link_is_verified_without_borrowing_production_link():
    r=record();q=r["forecast_payload"]["scoring_replay"]["offer"]
    q["under_link"]="https://fanduel.com.evil.invalid/test"
    rows,_=trial.choose([r],manifest(),NOW)
    assert rows[0]["side"]=="over" and not rows[0]["selected"]
    q["over_link"]=None
    assert trial.price_record(r,manifest(),NOW)[1]=="missing_valid_fanduel_link"


def test_post_registration_and_post_training_and_pregame_only():
    m=manifest();m["registered_at"]=NOW.isoformat()
    assert trial.price_record(record(),m,NOW)[1]=="not_a_new_post_registration_pregame_lock"
    assert trial.price_record(record(),manifest(),NOW+timedelta(days=1))[1]=="not_a_new_post_registration_pregame_lock"
    m=manifest();m["config"]["training_end"]="2026-09-22"
    assert trial.price_record(record(),m,NOW)[1]=="training_includes_forecast_day"


def test_caps_are_frozen_and_no_quota_or_outcome_reranking():
    records=[record(i,"game"+str(i//3),weights=(.6+.02*i,.4-.02*i)) for i in range(1,7)]
    rows,_=trial.choose(records,manifest(),NOW)
    selected=[r["forecast_id"] for r in rows if r["selected"]]
    assert selected==[4,5,6]
    assert trial.choose(records,manifest(),NOW,rows)[0]==[]
    # A newly reviewed result has no role in candidate selection.
    for r in records:
        r.update(actual=999,status="final",graded_result="loss")
    assert trial.choose(records,manifest(),NOW)[0]==rows
    negative=record(weights=(.5,.5))
    rows,_=trial.choose([negative],manifest(),NOW)
    assert len(rows)==1 and not rows[0]["selected"]
    assert rows[0]["selection_reason"]=="no_positive_own_edge"


def test_first_offer_cannot_be_replaced_by_later_better_price_or_outcome():
    first=record();later=deepcopy(first);later["id"]=99;later["created_at_utc"]="2026-09-22T10:02:00Z"
    later["forecast_payload"]["stat_model_challengers"][COMPONENT]["curve"]["weights"]=[.9,.1]
    rows,_=trial.choose([later,first],manifest(),NOW)
    assert len(rows)==1 and rows[0]["forecast_id"]==1 and rows[0]["probability"]==pytest.approx(.7)


def test_integer_push_mass_prices_ev_and_grades_without_binary_loss():
    r=record(line=30)
    r["forecast_payload"]["stat_model_challengers"][COMPONENT]["curve"]={"values":[10,30,80],"weights":[.6,.2,.2]}
    rows,_=trial.choose([r],manifest(),NOW);row=rows[0]
    assert row["push_probability"]==pytest.approx(.2)
    assert row["probability"]==pytest.approx(.75)
    assert row["ev"]==pytest.approx(.6*(100/106)-.2)
    r.update(status="final",actual=30,graded_result="push")
    result=trial.build_report(manifest(),[{"rows":rows}],[r],now=NOW)
    assert result["frozen_selected"]["accounting"]=={"push":1}
    assert result["frozen_selected"]["settled_binary"]==0
    assert result["frozen_selected"]["hypothetical_profit_units"]==0
    assert result["betting_approved"] is False


def test_report_uses_own_side_and_same_comparison_rows_with_pending_void_separate():
    records=[record(i,"game"+str(i)) for i in range(1,5)]
    rows,_=trial.choose(records,manifest(),NOW)
    records[0].update(status="final",actual=10,graded_result="loss") # source over lost; own under wins
    records[1].update(status="final",actual=None,graded_result="pending_participation_review")
    records[2].update(status="final",actual=None,graded_result="void_nonparticipant")
    result=trial.build_report(manifest(),[{"rows":rows}],records,now=NOW)
    scope=result["all_offers"]
    assert scope["accounting"]==dict(win=1,pending_participation_or_result=1,verified_nonparticipant_void=1,pending_game=1)
    assert scope["hypothetical_profit_units"]==pytest.approx(100/106)
    assert all(scope[name]["rows"]==1 for name in ("challenger","production","market"))
    assert scope["challenger"]["brier"]==pytest.approx(.09)
    assert scope["production"]["brier"]==pytest.approx(.25)
    assert scope["brier_gain_vs_market"]["lower_95"] is None
    assert result["frozen_selected"]["rows"]==3
    assert len(result["selected_decision_ids"])==3


def test_empty_cohort_never_counts_as_pass_and_duplicate_offers_cannot_inflate_sample():
    result=trial.build_report(manifest(),[],[],now=NOW)
    assert result["all_offers"]["rows"]==0 and result["all_offers"]["hypothetical_roi"] is None
    assert result["blockers"] and result["betting_approved"] is False
    rows,_=trial.choose([record()],manifest(),NOW)
    with pytest.raises(ValueError,match="Duplicate"):
        trial.build_report(manifest(),[{"rows":rows*2}],[],now=NOW)


def test_actual_without_verified_grade_is_still_pending():
    r=record();rows,_=trial.choose([r],manifest(),NOW)
    r.update(status="final",actual=0,graded_result=None)
    result=trial.build_report(manifest(),[{"rows":rows}],[r],now=NOW)
    assert result["all_offers"]["accounting"]=={"pending_participation_or_result":1}
    assert result["all_offers"]["settled_binary"]==0


def test_inconsistent_push_grade_does_not_create_a_win():
    r=record();rows,_=trial.choose([r],manifest(),NOW)
    r.update(status="final",actual=10,graded_result="push")
    result=trial.build_report(manifest(),[{"rows":rows}],[r],now=NOW)
    assert result["all_offers"]["accounting"]=={"inconsistent_verified_result":1}
    assert result["all_offers"]["settled_binary"]==0


def close_quote(row):
    return dict(row["quote"],id=2,snapshot_role="close",fetched_at_utc="2026-09-22T19:55:00Z",
                under_price=-150,over_price=125)


def test_exact_close_uses_independent_side_and_locked_price():
    rows,_=trial.choose([record()],manifest(),NOW);row=rows[0];q=close_quote(row)
    end=NOW+timedelta(days=1)
    close=trial.exact_close(row,[q],end)
    p=(150/250)/((150/250)+(100/225))
    assert close["valid"] and close["side"]=="under" and close["close_price"]==-150
    assert close["close_probability"]==pytest.approx(p)
    assert close["closing_fair_conditional_ev"]==pytest.approx(p*(1+100/106)-1)
    assert close["probability_delta"]==pytest.approx(p-row["market_probability"])
    assert trial.exact_close(row,[q],NOW)["reason"]=="awaiting_kickoff"


@pytest.mark.parametrize("field,value",[("provider","another"),("event_id","another"),("line",31.5),
    ("player_name_norm","another"),("bookmaker_key","draftkings"),("stat","rushing_yards"),
    ("under_price",None),("snapshot_role","live"),("pair_quality","synthetic"),
    ("fetched_at_utc","2026-09-22T20:00:00Z"),("fetched_at_utc","2026-09-22T19:00:00Z")])
def test_invalid_closes_stay_unknown(field,value):
    rows,_=trial.choose([record()],manifest(),NOW);row=rows[0];q=close_quote(row);q[field]=value
    close=trial.exact_close(row,[q],NOW+timedelta(days=1))
    assert close["valid"] is False and "probability_delta" not in close


def test_capture_reruns_keep_original_decisions_and_checksums(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"STORE",tmp_path)
    monkeypatch.setattr(trial.experiments,"load_experiment",lambda name:manifest())
    monkeypatch.setattr(trial.experiments,"check_experiment",lambda name:manifest())
    monkeypatch.setattr(trial,"_check_candidate",lambda m:None)
    assert trial.capture(records=[record()],now=NOW)["selected"]==1
    assert trial.capture(records=[record()],now=NOW+timedelta(seconds=1))["selected"]==0
    docs=trial.captures(manifest())
    assert sum(len(d["rows"]) for d in docs)==1
    path=sorted((tmp_path/"captures").glob("*.json"))[0]
    modified=json.loads(path.read_text());modified["rows"][0]["side"]="over";path.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match="checksum"):
        trial.captures(manifest())


def test_capture_registrations_remain_separate_immutable_cohorts(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"STORE",tmp_path)
    capture_dir=tmp_path/"captures";capture_dir.mkdir()
    current=manifest();current["sha256"]="current-registration"
    old_body=dict(contract=trial.CONTRACT,registration_sha256="old-registration",
        captured_at=NOW.isoformat(),rows=[{"cohort":"old"}],exclusions={},
        research_only=True,betting_approved=False)
    current_body=dict(contract=trial.CONTRACT,registration_sha256=current["sha256"],
        captured_at=(NOW+timedelta(seconds=1)).isoformat(),rows=[{"cohort":"current"}],exclusions={},
        research_only=True,betting_approved=False)
    old_path=capture_dir/"old.json"
    old_path.write_text(json.dumps(dict(old_body,sha256=trial.digest(old_body))))
    (capture_dir/"current.json").write_text(json.dumps(dict(current_body,sha256=trial.digest(current_body))))
    old_bytes=old_path.read_bytes()

    docs=trial.captures(current)
    assert [row["cohort"] for doc in docs for row in doc["rows"]]==["current"]
    assert old_path.read_bytes()==old_bytes
    modified=json.loads(old_path.read_text());modified["rows"][0]["cohort"]="tampered"
    old_path.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match="checksum"):
        trial.captures(current)


def test_code_or_artifact_drift_cannot_create_capture(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"STORE",tmp_path)
    monkeypatch.setattr(trial.experiments,"load_experiment",lambda name:manifest())
    monkeypatch.setattr(trial.experiments,"check_experiment",lambda name:(_ for _ in ()).throw(ValueError("code drift")))
    with pytest.raises(ValueError,match="code drift"):
        trial.capture(records=[record()],now=NOW)
    assert not list(tmp_path.rglob("*.json"))


def test_close_capture_is_immutable_and_coverage_denominator_is_all_matured(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"STORE",tmp_path)
    records=[record(i,"game"+str(i)) for i in range(1,3)]
    rows,_=trial.choose(records,manifest(),NOW);end=NOW+timedelta(days=1)
    q=close_quote(rows[0]);lookup={rows[0]["decision_id"]:[q]}
    closes=trial.capture_closes(manifest(),rows,end,quote_lookup=lookup)
    assert len(list((tmp_path/"closes").glob("*.json")))==1
    q["under_price"]=-1000
    later=trial.capture_closes(manifest(),rows,end+timedelta(hours=1),quote_lookup=lookup)
    assert later[rows[0]["decision_id"]]==closes[rows[0]["decision_id"]]
    report=trial.build_report(manifest(),[{"rows":rows}],records,closes=closes,now=end)
    assert report["valid_exact_close_coverage"]==.5
    assert report["exact_closes"]["matured"]==2 and report["exact_closes"]["valid"]==1


def test_report_exposes_candidate_registry_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"STORE",tmp_path/"store")
    monkeypatch.setattr(trial,"ROOT",tmp_path)
    monkeypatch.setattr(trial.experiments,"load_experiment",lambda name:manifest())
    monkeypatch.setattr(trial.experiments,"check_experiment",lambda name:manifest())
    monkeypatch.setattr(trial,"_check_candidate",lambda m:(_ for _ in ()).throw(ValueError("candidate changed")))
    result=trial.report(records=[])
    assert result["dependency_drift"]=="candidate changed"
    assert "experiment_dependency_drift" in result["blockers"]
    assert json.loads((tmp_path/"reports/nfl_receiving_paper_v2_latest.json").read_text())["betting_approved"] is False
    assert "Paper only" in (tmp_path/"reports/nfl_receiving_paper_v2_latest.md").read_text()


def test_generation_attestation_refuses_artifact_or_dependency_drift(monkeypatch):
    m=manifest()
    monkeypatch.setattr(trial.experiments,"load_experiment",lambda name:m)
    monkeypatch.setattr(trial.experiments,"check_experiment",lambda name:m)
    monkeypatch.setattr(trial,"_check_candidate",lambda m:None)
    proof=trial.generation_attestation(COMPONENT,"artifact-checksum")
    assert proof["registration_sha256"]==m["sha256"]
    assert proof["scoring_fingerprint"]==m["config"]["scoring_fingerprint"]
    with pytest.raises(ValueError,match="artifact"):
        trial.generation_attestation(COMPONENT,"other")
    monkeypatch.setattr(trial.experiments,"check_experiment",lambda name:(_ for _ in ()).throw(ValueError("temporary code drift")))
    with pytest.raises(ValueError,match="temporary code drift"):
        trial.generation_attestation(COMPONENT,"artifact-checksum")


def test_score_component_attaches_before_and_after_generation_proof(monkeypatch):
    import numpy as np
    import pandas as pd
    from nfl_pipeline.modeling import stat_model_registry as registry
    class Runtime:
        def score(self, frame, projection):
            return dict(values=np.array([[10.,80.]]),weights=np.array([[.7,.3]]),
                mean=np.array([31.]),median=np.array([10.]),p10=np.array([10.]),p90=np.array([80.]))
    proof=dict(registration_sha256="registration",artifact_sha256="artifact-checksum",scoring_fingerprint="code")
    calls=iter([dict(proof,verified_at="2026-09-22T10:00:20Z"),dict(proof,verified_at="2026-09-22T10:00:30Z")])
    monkeypatch.setattr(trial,"generation_attestation",lambda *a:next(calls))
    entry=dict(component_id=COMPONENT,sha256="artifact-checksum",outputs=["probability"])
    score=registry.score_component((entry,dict(runtime=Runtime())),pd.DataFrame([{}]),[40])
    assert score["artifact_sha256"]=="artifact-checksum"
    assert score["experiment_generation"]["started_at"]=="2026-09-22T10:00:20Z"
    assert score["experiment_generation"]["verified_at"]=="2026-09-22T10:00:30Z"


def test_registration_pins_scorer_artifact_production_and_absent_overlays(tmp_path,monkeypatch):
    import hashlib
    import joblib
    from nfl_pipeline.modeling import stat_model_registry as registry
    modelroot=tmp_path/"models"; release=modelroot/"releases/production";release.mkdir(parents=True)
    (release/"players.joblib").write_bytes(b"players")
    (release/"games.joblib").write_bytes(b"games")
    (modelroot/"active_release.json").write_text(json.dumps(dict(release_id="production")))
    artifact_path=modelroot/"candidate.joblib";artifact_path.write_bytes(b"candidate")
    sha=hashlib.sha256(b"candidate").hexdigest()
    artifact=dict(contract="nfl-receiving-opportunity-efficiency-v2",stat="receiving_yards",
                  run_id=COMPONENT,production_release="production",training_end="2026-09-17")
    monkeypatch.setattr(trial,"MODEL_ROOT",modelroot)
    monkeypatch.setattr(joblib,"load",lambda path:artifact)
    monkeypatch.setattr(registry,"load_registry",lambda:dict(active={},candidates={"receiving_yards":{
        COMPONENT:dict(component_id=COMPONENT,sha256=sha,artifact=str(artifact_path),outputs=["probability"])}}))
    monkeypatch.setattr(trial.experiments,"source_dependencies",lambda *a:{"scorer":__file__})
    captured={}
    def register(name,config,files,**kwargs):
        captured.update(name=name,config=config,files=files,**kwargs);return captured
    monkeypatch.setattr(trial.experiments,"register_experiment",register)
    result=trial.register()
    assert result["files"]["oe_probability_artifact"]==artifact_path
    assert result["files"]["production_release_players.joblib"]==release/"players.joblib"
    assert set(result["absent_paths"])=={"market_calibration","interval_calibration","production_distribution_policy"}
    assert result["config"]["production_stat_active"]=={}
    (modelroot/"active_release.json").write_text(json.dumps(dict(release_id="another")))
    with pytest.raises(ValueError,match="Active production pointer"):
        trial.register()
