from nfl_pipeline import live_injuries

GAME = dict(game_id="2026_04_PIT_CLE", season=2026, week=4, home="CLE", away="PIT")
SUMMARY = {"injuries": [
    {"team": {"abbreviation": "PIT"}, "injuries": [
        {"athlete": {"id": "4038815", "displayName": "Rico Dowdle", "position": {"abbreviation": "RB"}},
         "status": "Out", "date": "2026-09-30T19:05Z", "details": {"type": "Toe"}}]},
    {"team": {"abbreviation": "CLE"}, "injuries": [
        {"athlete": {"id": "999", "displayName": "Unknown Guy", "position": {"abbreviation": "WR"}},
         "status": "Questionable", "date": "2026-10-01T18:00Z", "details": {"type": "Ankle"}}]}]}


def test_parses_espn_report_and_maps_to_nflverse_ids():
    entries = live_injuries.parse_summary(SUMMARY)
    rows = live_injuries._rows(GAME, entries, {"4038815": "00-0035xxx"}, {}, "url")
    dowdle = next(r for r in rows if r[7] == "Rico Dowdle")
    assert dowdle[1:7] == (2026, "REG", "REG", 4, "PIT", "00-0035xxx") and dowdle[12] == "Out" and dowdle[16] == "espn_live"
    unknown = next(r for r in rows if r[7] == "Unknown Guy")
    assert unknown[6] is None and unknown[8] == "unknown guy"  # unmapped players still match by name


def test_player_dropped_from_report_gets_cleared_row_and_status_changes_get_new_rows():
    previous = {("PIT", "00-0035xxx"): dict(player_id="00-0035xxx", player_name="Rico Dowdle", position="RB", report_status="Out"),
                ("PIT", "00-0011111"): dict(player_id="00-0011111", player_name="Was Questionable", position="WR",
                                             report_status="Questionable")}
    rows = live_injuries._rows(GAME, live_injuries.parse_summary(SUMMARY), {"4038815": "00-0035xxx"}, previous, "url")
    cleared = [r for r in rows if r[12] == live_injuries.CLEARED]
    assert [r[7] for r in cleared] == ["Was Questionable"]  # still-listed Dowdle is not cleared
    out_again = live_injuries._rows(GAME, live_injuries.parse_summary(SUMMARY), {"4038815": "00-0035xxx"}, {}, "url")
    assert {r[0] for r in out_again if r[7] == "Rico Dowdle"} == {next(r[0] for r in rows if r[7] == "Rico Dowdle")}


def test_live_injuries_run_before_scoring_in_daily_and_pregame(monkeypatch, tmp_path):
    import asyncio
    from nfl_pipeline import run_daily_and_notify as daily, game_scope
    monkeypatch.setenv(game_scope.ENV, '[]')  # restored after the test; main() overwrites it for pregame
    for argv in (['daily', '--date', '2099-09-27', '--skip-train', '--run-id', 'a'],
                 ['daily', '--pregame', '--date', '2099-09-27', '--game-id', 'early', '--run-id', 'b']):
        monkeypatch.setattr('sys.argv', argv)
        monkeypatch.setattr(daily, '_repo_root', lambda: tmp_path)
        monkeypatch.setattr(daily, 'active_release', lambda: {'release_id': 'frozen'})
        monkeypatch.setenv('NFL_MODEL_RELEASE_ID', 'frozen')
        steps = []
        monkeypatch.setattr(daily, '_run', lambda step: steps.append(step.module) or (0, '{}', ''))
        async def post(*a):
            pass
        monkeypatch.setattr(daily, '_post_matchups', post)
        asyncio.run(daily.main())
        assert steps.index('nfl_pipeline.live_injuries') < steps.index('nfl_pipeline.modeling.predict_player_props')
