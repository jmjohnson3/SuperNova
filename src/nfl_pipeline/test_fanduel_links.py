from copy import deepcopy
import re
from urllib.parse import parse_qs, urlsplit

import pytest

from nfl_pipeline import fanduel_links as links
from nfl_pipeline.betting_preferences import display_rows
from nfl_pipeline.discord_matchups import build_bundle, DESCRIPTION_LIMIT
from nfl_pipeline.test_discord_matchups import prop, game_pick, schedule, NOW, DAY, body


def source(market='717.188177722', selection='59996483'):
    return f'https://sportsbook.fanduel.com/addToBetslip?marketId={market}&selectionId={selection}'


@pytest.mark.parametrize('url', [
    source(), source().replace('&', '&amp;'),
    source().replace('marketId=', 'marketId[0]=').replace('selectionId=', 'selectionId[0]='),
    source().replace('marketId=', 'marketId%5B0%5D=').replace('selectionId=', 'selectionId%5B0%5D='),
    source().replace('sportsbook.fanduel.com/addToBetslip', 'account.sportsbook.fanduel.com/sportsbook/addToBetslip'),
    source().replace('sportsbook.fanduel.com', 'co.sportsbook.fanduel.com'),
])
def test_single_link_keeps_the_exact_provider_ids(url):
    normalized = links.single_betslip_url(url)
    assert normalized.startswith(links.BETSLIP_BASE+'?')
    assert parse_qs(urlsplit(normalized).query) == {
        'marketId[0]': ['717.188177722'], 'selectionId[0]': ['59996483']}
    assert links.single_betslip_url(normalized) == normalized
    assert 'shareBetId' not in normalized and 'shareCode' not in normalized


@pytest.mark.parametrize('url', [
    None, '', 'https://fanduel.com/', source().replace('addToBetslip', 'home'),
    source().replace('&selectionId=59996483', ''),
    source().replace('717.188177722', 'not-an-id'),
    source().replace('59996483', ''),
    source().replace('fanduel.com', 'fanduel.com.evil.test'),
    source().replace('https:', 'http:'),
    source().replace('sportsbook.fanduel.com', 'user@sportsbook.fanduel.com'),
    source().replace('sportsbook.fanduel.com', 'sportsbook.fanduel.com:123'),
    source()+'&selectionId=7',
    source()+'&marketId[0]=42.1&selectionId[0]=7',
    source().replace('marketId=', 'marketId[0]=').replace('selectionId=', 'selectionId[1]='),
    source().replace('marketId=', 'marketId[999]=').replace('selectionId=', 'selectionId[999]='),
    source()+'#other', source()+'\n[Fake](<https://example.com>)',
])
def test_invalid_or_ambiguous_selection_cannot_be_an_add_to_slip(url):
    assert links.single_betslip_url(url) is None


def test_parlay_supports_both_formats_and_keeps_same_selection_on_different_markets():
    one = source('42.100', '17')
    two = links.single_betslip_url(source('42.101', '17'))
    result = links.parlay_betslip_url([one, two, one])
    assert links.selections(result) == (('42.100', '17'), ('42.101', '17'))
    assert links.single_betslip_url(result) is None
    assert links.parlay_betslip_url([one, two, None]) is None
    assert links.parlay_betslip_url([one, source('42.100', '18')]) is None
    assert links.parlay_betslip_url([one, one]) is None
    assert links.parlay_betslip_url([one, result]) is None


def test_presentation_does_not_mutate_forecast_or_its_probability():
    row = prop(link=source(), tier='micro_projection')
    before = deepcopy(row)
    rendered = links.format_prop_row(row, action='BET $1')
    assert row == before
    assert 'Add to slip' in rendered and 'Provider link' in rendered
    assert f'(<{source()}>)' in rendered
    assert 'OVER45.5 -110' in rendered
    result = display_rows([row])[0]
    assert result['tier'] == 'micro_projection' and result['link'] == source()


@pytest.mark.parametrize('tier', ['micro_projection', 'bankroll', 'cash_trial'])
def test_homepage_is_manual_only_and_cannot_look_executable(tier):
    row = prop(link='https://sportsbook.fanduel.com/', tier=tier)
    result = display_rows([row])[0]
    assert row['tier'] == tier and result['tier'] == 'paper'
    text = links.row_link(result)
    assert 'manual selection' in text and 'Add to slip' not in text


def test_every_matchup_section_and_manifest_use_new_links_without_changing_prices():
    rows = [prop(player='Research1', link=source(), forecast_id=1),
            prop(player='Research2', link=source('42.9', '8'), forecast_id=2),
            prop(player='Micro', tier='micro_projection', link=source('42.7', '6'), forecast_id=3),
            prop(player='Bankroll', tier='bankroll', link=source('42.5', '4'), forecast_id=4)]
    bundle = build_bundle(DAY, schedule(), [game_pick(link=source('42.3', '2'), forecast_id=5)],
                          rows, ['frozen'], now=NOW)
    text = body(bundle, 'early')
    assert bundle['link_contract'] == links.CONTRACT
    assert text.count('[Add to slip]') == 5 and 'Research parlay' in text
    for card in bundle['cards']:
        assert len(card['payload']['embeds'][0]['description']) <= DESCRIPTION_LIMIT
        for row in card['forecast_manifest']:
            assert row['price'] == -110
            assert links.selections(row['provider_link']) == links.selections(row['betslip_link'])
    rendered_urls = re.findall(r'\]\(<([^>]+)>\)', text)
    assert all(links.selections(url) for url in rendered_urls)


def test_cash_reservation_rejects_a_landing_page_before_spending_capacity(monkeypatch):
    from nfl_pipeline import cash_execution
    monkeypatch.setattr(cash_execution, 'forecast_quote_error', lambda row, now: None)
    assert cash_execution.candidate_error(dict(book='fanduel', link='https://sportsbook.fanduel.com/'), NOW) == 'missing_fanduel_link'


def test_model_projection_shown_beside_line_anchored_projection():
    trace = {'model_projection': 31.24, 'market_calibration': {'projection_trust': 0.0}}
    row = dict(book='fanduel', player_name='D.Metcalf', team_abbr='PIT', opponent_abbr='CLE', stat='receiving_yards',
               side='over', line=45.5, price=-113, projection=45.5, probability_trace=trace)
    text = links._with_model_projection('x proj=45.50 range=1-2', row)
    assert text == 'x model=31.2 | line-anchored=45.5 (0% model) range=1-2'
    full = dict(row, probability_trace=dict(trace, market_calibration={'projection_trust': 1.0}))
    assert links._with_model_projection('x proj=45.50', full) == 'x proj=45.50'  # unanchored: unchanged
    assert links._with_model_projection('x proj=45.50', dict(row, probability_trace=None)) == 'x proj=45.50'
