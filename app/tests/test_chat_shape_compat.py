"""Accept the chat-completions body shape on /v1/audio/speech.

Real clients probe this endpoint with a chat-completions body:

    {"model": "xtts", "messages": [{"role": "user", "content": "hi"}],
     "max_tokens": 1}

— a gateway liveness check, pointed at the TTS route. Rejecting it with
422 "body.input Field required" makes the gateway mark the model dead, so
the text is taken from `messages` when `input` is absent.

    pytest app/tests/test_chat_shape_compat.py -q
"""

from __future__ import annotations

import logging

import pytest

from test_speech_endpoint import FakeSynth, _install_stubs  # noqa: E402

_install_stubs()

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from server import State, create_app  # noqa: E402


@pytest.fixture
def synth():
    return FakeSynth()


@pytest.fixture
def client(monkeypatch, synth):
    monkeypatch.delenv("SUPERTONIC_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.setattr(State, "synth", synth)
    monkeypatch.setattr(State, "ready", True)
    monkeypatch.setattr(server, "detect_language", lambda text, hint=None: None)
    with TestClient(create_app()) as c:
        yield c


# ===================== the probe that started this =====================

def test_gateway_liveness_probe_is_answered_with_audio(client, synth):
    """Verbatim body from the failing client."""
    r = client.post(
        "/v1/audio/speech",
        json={"model": "xtts", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 1},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/pcm")
    assert len(r.content) > 0
    assert synth.calls[0]["text"] == "hi"


def test_last_user_message_is_the_one_synthesised(client, synth):
    client.post(
        "/v1/audio/speech",
        json={"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": "second"},
        ]},
    )
    assert synth.calls[0]["text"] == "second"


def test_content_parts_are_flattened(client, synth):
    """OpenAI's multimodal shape: content is a list of typed parts."""
    r = client.post(
        "/v1/audio/speech",
        json={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]}]},
    )
    assert r.status_code == 200, r.text
    assert synth.calls[0]["text"] == "hello world"


def test_message_without_a_user_role_still_works(client, synth):
    """Some probes send a lone system or bare message."""
    r = client.post(
        "/v1/audio/speech",
        json={"messages": [{"role": "system", "content": "ping"}]},
    )
    assert r.status_code == 200, r.text
    assert synth.calls[0]["text"] == "ping"


# ===================== input still wins =====================

def test_explicit_input_takes_precedence_over_messages(client, synth):
    client.post(
        "/v1/audio/speech",
        json={"input": "the real text", "messages": [{"role": "user", "content": "ignore me"}]},
    )
    assert synth.calls[0]["text"] == "the real text"


def test_normal_openai_speech_body_is_unaffected(client, synth):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "supertonic-3", "input": "Hello.", "voice": "M1",
              "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")


# ===================== genuinely empty bodies still fail =====================

@pytest.mark.parametrize("body", [
    {},
    {"model": "xtts"},
    {"messages": []},
    {"messages": [{"role": "user", "content": ""}]},
    {"messages": [{"role": "user"}]},
    {"messages": "not a list"},
])
def test_bodies_with_no_text_are_still_rejected(client, body):
    """Leniency must not become "synthesise silence for anything"."""
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 422, f"{body} -> {r.status_code}"


def test_the_422_names_both_accepted_fields(client):
    r = client.post("/v1/audio/speech", json={"model": "xtts"})
    assert r.status_code == 422
    assert "messages" in r.text, "the error should mention the alternative"


# ===================== it is visible in the log =====================

def test_a_normal_body_is_logged_as_such(client, caplog):
    caplog.set_level(logging.INFO, logger="supertonic.server")
    client.post("/v1/audio/speech", json={"input": "Hello.", "voice": "M1"})
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "text_from=input" in msgs


def test_the_fallback_is_logged(client, caplog):
    """An operator seeing TTS of 'hi' every 30s should be able to tell it
    came from a chat-shaped probe, not a real caller."""
    caplog.set_level(logging.INFO, logger="supertonic.server")
    client.post(
        "/v1/audio/speech",
        json={"model": "xtts", "messages": [{"role": "user", "content": "hi"}]},
    )
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "text_from=messages" in msgs
