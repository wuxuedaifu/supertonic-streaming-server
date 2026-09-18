"""Tests for raw request-body logging on the POST endpoints.

The structured per-request line in the speech handler only reports fields
the pydantic model kept. Operators debugging a misbehaving client need to
see what actually arrived on the wire — including fields the model drops
(``extra="ignore"``) and bodies that never reach the handler because
validation rejected them first.

    pytest app/tests/test_request_logging.py -q
"""

from __future__ import annotations

import base64
import json
import logging
import types

import pytest

from test_speech_endpoint import FakeSynth, _install_stubs  # noqa: E402

_install_stubs()

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from server import State, create_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SUPERTONIC_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", "1")
    monkeypatch.setattr(State, "synth", FakeSynth())
    monkeypatch.setattr(State, "ready", True)
    monkeypatch.setattr(server, "detect_language", lambda text, hint=None: None)
    with TestClient(create_app()) as c:
        yield c


def _recv_lines(caplog) -> list[str]:
    """Just the body lines — the rejection lines also start with "recv "."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("recv ") and " POST " in r.getMessage()
    ]


# ===================== the body reaches the log =====================

def test_body_is_logged_and_request_still_succeeds(client, caplog):
    """Consuming the body in middleware must not starve the handler: the
    request has to work exactly as before AND show up in the log."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    r = client.post(
        "/v1/audio/speech",
        json={"input": "Hello there.", "voice": "M1", "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")

    lines = _recv_lines(caplog)
    assert len(lines) == 1
    assert "Hello there." in lines[0]
    assert '"voice":"M1"' in lines[0]


def test_fields_the_model_drops_are_still_logged(client, caplog):
    """SpeechRequest is extra="ignore", so unknown keys vanish before the
    handler sees them. The raw-body line is the only place they show up."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    client.post(
        "/v1/audio/speech",
        json={"input": "Hi.", "voice": "M1", "some_unknown_client_field": "surprise"},
    )
    assert "some_unknown_client_field" in _recv_lines(caplog)[0]


def test_conditioning_endpoint_is_logged_too(client, caplog, monkeypatch):
    # The conditioning handler serialises style.ttl/.dp; FakeSynth hands
    # back a plain string, so give it something with those attributes.
    import numpy as np

    monkeypatch.setattr(
        server,
        "_resolve_speaker_lenient",
        lambda synth, ref: types.SimpleNamespace(
            ttl=np.zeros(4, dtype="float32"), dp=np.zeros(4, dtype="float32")
        ),
    )
    caplog.set_level(logging.INFO, logger="supertonic.server")
    r = client.post("/v1/tts/conditioning", json={"speaker_files": ["M1"]})
    assert r.status_code == 200
    assert "/v1/tts/conditioning" in _recv_lines(caplog)[0]


# ===================== blobs must not flood the log =====================

def test_long_strings_are_redacted_to_a_length_marker(client, caplog):
    """XTTS clients pass 60 KB base64 latents. Logging those verbatim makes
    the log unreadable and can outweigh the audio itself."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    blob = base64.b64encode(b"\x00" * 45000).decode()
    client.post(
        "/v1/audio/speech",
        json={"input": "Hi.", "voice": "M1", "gpt_cond_latent_b64": blob},
    )
    line = _recv_lines(caplog)[0]
    assert blob not in line
    assert "60000B" in line, "redaction should report the original length"
    assert len(line) < 3000


def test_whole_line_is_capped(client, caplog, monkeypatch):
    monkeypatch.setenv("SUPERTONIC_LOG_BODY_MAX", "120")
    caplog.set_level(logging.INFO, logger="supertonic.server")
    client.post(
        "/v1/audio/speech",
        json={"input": "Hi.", "voice": "M1", **{f"k{i}": i for i in range(200)}},
    )
    line = _recv_lines(caplog)[0]
    assert "…+" in line
    assert len(line) < 500


# ===================== failures are visible =====================

def test_rejected_body_is_logged_even_though_the_handler_never_runs(client, caplog):
    """A 422 is exactly the case where you most want to see the payload."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    r = client.post("/v1/audio/speech", json={"voice": "M1"})  # no `input`
    assert r.status_code == 422
    lines = _recv_lines(caplog)
    assert len(lines) == 1
    assert '"voice":"M1"' in lines[0]
    assert any("422" in m for m in [rec.getMessage() for rec in caplog.records])


def test_non_json_body_does_not_blow_up_the_middleware(client, caplog):
    caplog.set_level(logging.INFO, logger="supertonic.server")
    r = client.post(
        "/v1/audio/speech",
        content=b"\xff\xfe not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert len(_recv_lines(caplog)) == 1


# ===================== the off switch =====================

def test_logging_can_be_turned_off(client, caplog, monkeypatch):
    """Toggleable at request time so operators can silence it without a
    restart on a busy box."""
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", "0")
    caplog.set_level(logging.INFO, logger="supertonic.server")
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "M1"})
    assert r.status_code == 200
    assert _recv_lines(caplog) == []


def test_req_id_is_shared_with_the_handler_line(client, caplog):
    """Both lines must carry the same req= token or they can't be joined."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    client.post(
        "/v1/audio/speech",
        json={"input": "Hi.", "voice": "M1"},
        headers={"X-Request-Id": "deadbeef99"},
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("recv req=deadbeef99 ") for m in msgs)
    assert any(m.startswith("speech req=deadbeef99 ") for m in msgs)
