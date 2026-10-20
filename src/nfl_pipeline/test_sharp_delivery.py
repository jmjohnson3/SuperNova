import asyncio

import httpx
import pytest

from nfl_pipeline import sharp_delivery as delivery
from nfl_pipeline.experiment_registry import source_dependencies


def fake_transport(monkeypatch, responses):
    calls = []
    sleeps = []

    class Client:
        def __init__(self, *, timeout):
            assert timeout == 20

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            response = responses[len(calls) - 1]
            if isinstance(response, Exception):
                raise response
            return response

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(delivery, '_webhook_url', lambda: 'https://example.invalid/webhook')
    monkeypatch.setattr(delivery.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(delivery.asyncio, 'sleep', sleep)
    return calls, sleeps


def test_confirmed_rate_limit_retries_preserve_payload(monkeypatch):
    calls, sleeps = fake_transport(monkeypatch, [
        httpx.Response(429, json={'retry_after': .25}), httpx.Response(204),
    ])
    payload = {'content': 'research alert', 'allowed_mentions': {'parse': []}}
    asyncio.run(delivery._post_payload(payload))
    assert len(calls) == 2
    assert sleeps == [.25]
    assert all(kwargs == {'params': {'wait': 'true'}, 'json': payload} for _, kwargs in calls)


@pytest.mark.parametrize('error', [httpx.ReadTimeout('unknown send'), httpx.ConnectError('connection failed')])
def test_uncertain_delivery_is_never_retried(monkeypatch, error):
    calls, sleeps = fake_transport(monkeypatch, [error])
    with pytest.raises(RuntimeError, match='not retrying an uncertain send'):
        asyncio.run(delivery._post_payload({'content': 'research alert'}))
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize('delay', [-1, 61, float('inf'), float('nan')])
def test_rate_limit_wait_is_bounded(monkeypatch, delay):
    class Response:
        status_code = 429

        def json(self):
            return {'retry_after': delay}

    calls, sleeps = fake_transport(monkeypatch, [Response()])
    with pytest.raises(RuntimeError, match='bounded retry window'):
        asyncio.run(delivery._post_payload({'content': 'research alert'}))
    assert len(calls) == 1 and sleeps == []


def test_rate_limit_attempts_are_bounded(monkeypatch):
    calls, sleeps = fake_transport(monkeypatch, [httpx.Response(429, json={'retry_after': 0})] * 4)
    with pytest.raises(RuntimeError, match='HTTP 429'):
        asyncio.run(delivery._post_payload({'content': 'research alert'}))
    assert len(calls) == 4 and sleeps == [.1] * 3


def test_other_http_failures_are_not_retried(monkeypatch):
    calls, sleeps = fake_transport(monkeypatch, [httpx.Response(500)])
    with pytest.raises(RuntimeError, match='HTTP 500'):
        asyncio.run(delivery._post_payload({'content': 'research alert'}))
    assert len(calls) == 1 and sleeps == []


def test_sharp_source_closure_does_not_pin_daily_orchestration():
    files = source_dependencies('nfl_pipeline.sharp_watch', 'nfl_pipeline.modeling.sharp_alert_report')
    assert 'nfl_pipeline.sharp_delivery' in files
    assert 'nfl_pipeline.run_daily_and_notify' not in files
