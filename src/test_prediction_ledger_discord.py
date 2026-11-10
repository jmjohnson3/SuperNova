from copy import deepcopy
import json

import pytest
import requests

import prediction_ledger_discord as publisher


DOC=dict(built_at='2026-11-05T18:00:00+00:00',since='2026-09-01',through='2026-11-05',records=[])
FILES=[('prediction-summary.txt',b'summary','text/plain'),('predictions-ledger.zip',b'zip','application/zip')]


class Response:
    def __init__(self,body,status=200):self.body=body;self.status_code=status
    def json(self):return self.body


class Discord:
    def __init__(self,channel=publisher.CHANNEL_ID):
        self.meta=dict(id='webhook',guild_id=publisher.GUILD_ID,channel_id=channel)
        self.posts=0;self.edits=0;self.current=None
    def get(self,url,**kwargs):return Response(self.current if '/messages/' in url else self.meta)
    def send(self,kwargs):
        payload=json.loads(kwargs['data']['payload_json'])
        assert payload['allowed_mentions']=={'parse':[]}
        assert len(kwargs['files'])==2
        self.current=dict(id='message',channel_id=publisher.CHANNEL_ID,content=payload['content'],
            attachments=[dict(filename=f[0]) for f in FILES])
        return Response(self.current)
    def post(self,url,**kwargs):self.posts+=1;return self.send(kwargs)
    def patch(self,url,**kwargs):self.edits+=1;return self.send(kwargs)


def test_post_once_then_edit_changed_ledger_and_keep_same_message(tmp_path,monkeypatch):
    fingerprint=['one'];monkeypatch.setattr(publisher,'package',lambda folder:(DOC,fingerprint[0],FILES))
    state=tmp_path/'state.json';session=Discord()
    assert publisher.sync(tmp_path,state,'https://example.invalid/webhook',session)['status']=='posted'
    assert publisher.sync(tmp_path,state,'https://example.invalid/webhook',session)['status']=='unchanged'
    fingerprint[0]='two'
    assert publisher.sync(tmp_path,state,'https://example.invalid/webhook',session)['status']=='updated'
    assert session.posts==1 and session.edits==1
    saved=json.loads(state.read_text());assert saved['status']=='published' and saved['message_id']=='message'


def test_wrong_route_is_never_posted(tmp_path,monkeypatch):
    monkeypatch.setattr(publisher,'package',lambda folder:(DOC,'one',FILES));session=Discord('wrong_channel')
    with pytest.raises(RuntimeError,match='authorized'):publisher.sync(tmp_path,tmp_path/'state.json','https://example.invalid',session)
    assert session.posts==0 and session.edits==0


def test_unknown_create_outcome_blocks_duplicate_post(tmp_path,monkeypatch):
    monkeypatch.setattr(publisher,'package',lambda folder:(DOC,'one',FILES));session=Discord();state=tmp_path/'state.json'
    def timeout(*args,**kwargs):raise requests.Timeout()
    session.post=timeout
    with pytest.raises(requests.Timeout):publisher.sync(tmp_path,state,'https://example.invalid',session)
    assert json.loads(state.read_text())['status']=='pending_create'
    with pytest.raises(RuntimeError,match='unknown outcome'):publisher.sync(tmp_path,state,'https://example.invalid',session)


def test_unverified_attachment_response_is_not_marked_published(tmp_path,monkeypatch):
    monkeypatch.setattr(publisher,'package',lambda folder:(DOC,'one',FILES));session=Discord();state=tmp_path/'state.json'
    getter=session.get
    def get(url,**kw):
        result=getter(url,**kw)
        if '/messages/' in url:result.body=dict(result.body,attachments=[])
        return result
    session.get=get
    with pytest.raises(RuntimeError,match='did not verify'):publisher.sync(tmp_path,state,'https://example.invalid',session)
    assert json.loads(state.read_text())['message_id']=='message'
    assert json.loads(state.read_text())['status']=='awaiting_verification'


def test_message_uses_only_original_fanduel_nba_alert_cohort():
    base=dict(cohort='Delivered price alerts',sport='NBA',book='fanduel',version='nba-prop-price-check-v1',date='2026-11-04',
        result='win',prediction=None,actual=None,paper_units=1,repeated_observations=0)
    doc=dict(DOC,records=[base,dict(base,book='draftkings'),dict(base,cohort='Consensus trial / consensus')])
    content=publisher.message(doc)
    assert 'All recorded: 1 right / 0 wrong' in content
    assert len(content)<2000 and 'not actual wagers' in content


def test_football_summaries_split_markets_books_and_versions():
    base=dict(cohort='Locked model picks / paper',sport='NFL',book='fanduel',version='current',date='2026-11-04',
        forecast_at='2026-11-04T20:00:00+00:00',market='receiving_yards',result='loss',prediction=None,actual=None,
        paper_units=-1,repeated_observations=0)
    ncaa=dict(base,cohort='Game model vs saved line',sport='NCAAF',book='DraftKings',version='ncaa',market='spread',result='win')
    rows=[base,dict(base,market='spread',result='win'),
        dict(base,version='older',forecast_at='2026-10-01T20:00:00+00:00',result='win'),
        dict(base,book='draftkings',result='win'),dict(base,result='excluded'),
        ncaa,dict(ncaa,market='total',result='loss'),dict(ncaa,book='Other Book',forecast_at='2026-10-01T20:00:00+00:00')]
    content=publisher.message(dict(DOC,records=rows))
    assert '**NFL — FanDuel model paper picks**' in content
    assert 'Player props: 0 right / 1 wrong' in content
    assert 'Spreads/totals: 1 right / 0 wrong' in content
    assert '**NCAA football — model vs saved lines**' in content
    assert 'Reference book: DraftKings' in content
    assert 'Spreads: 1 right / 0 wrong' in content and 'Totals: 0 right / 1 wrong' in content
    assert len(content)<2000


def test_package_fingerprint_ignores_build_time_and_zip_has_all_rows(tmp_path):
    import io,zipfile
    doc=deepcopy(DOC)
    (tmp_path/'predictions_ledger_latest.json').write_text(json.dumps(doc))
    (tmp_path/'predictions_ledger_latest.html').write_text('<html>ledger</html>')
    (tmp_path/'predictions_ledger_latest.md').write_text('summary')
    _,before,files=publisher.package(tmp_path)
    doc['built_at']='2026-11-05T19:00:00+00:00'
    (tmp_path/'predictions_ledger_latest.json').write_text(json.dumps(doc))
    assert publisher.package(tmp_path)[1]==before
    with zipfile.ZipFile(io.BytesIO(files[1][1])) as z:
        assert json.loads(z.read('predictions_ledger.json'))['records']==DOC['records']
        assert set(z.namelist())=={'predictions_ledger.html','predictions_ledger.json','prediction-summary.txt'}
