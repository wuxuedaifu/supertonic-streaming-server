"""Regression tests for the pre-existing XTTS / pipecat contract.

Other services already depend on the legacy shape of this API, and two
recent changes sit directly in its path: OpenAI's ``response_format`` and
the request-body logging middleware. A middleware in particular is easy to
get wrong — it drains the request body and wraps the response, so it can
silently turn a chunked stream into a buffered one, which would wreck TTFB
for every existing client without failing any status-code assertion.

Everything here describes behaviour that predates both changes.

    pytest app/tests/test_legacy_api_compat.py -q
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pytest

from test_speech_endpoint import (  # noqa: E402
    NATIVE_SR,
    TONE_SECONDS,
    FakeSynth,
    _install_stubs,
    _tone,
)

_install_stubs()

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from server import State, create_app  # noqa: E402


LEGACY_BODY = {
    "input": "Hello from the legacy client.",
    "voice": ["M1"],
    "language": "en",
    "stream": True,
    # XTTS knobs supertonic can't honour. They have always been accepted
    # and ignored; a 422 here would break every existing caller.
    "temperature": 0.75,
    "top_k": 50,
    "top_p": 0.85,
    "do_sample": True,
    "enhance_speech": False,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "gpt_cond_len": 30,
    "gpt_cond_chunk_len": 4,
    "max_ref_length": 60,
    "sound_norm_refs": False,
    "model": "xtts_v2",
}


def _make_client(monkeypatch, synth=None):
    monkeypatch.delenv("SUPERTONIC_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.setattr(State, "synth", synth or FakeSynth())
    monkeypatch.setattr(State, "ready", True)
    monkeypatch.setattr(server, "detect_language", lambda text, hint=None: None)
    return TestClient(create_app())


@pytest.fixture
def client(monkeypatch):
    with _make_client(monkeypatch) as c:
        yield c


# ===================== the legacy request shape =====================

def test_full_xtts_legacy_body_is_accepted(client):
    """Every field an XTTS client sends, at once. None may 422."""
    r = client.post("/v1/audio/speech", json=LEGACY_BODY)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-sample-rate"] == "24000"
    assert not r.content.startswith(b"RIFF")
    assert abs(len(r.content) // 2 - int(TONE_SECONDS * 24000)) < 200


def test_two_call_conditioning_flow_still_round_trips(client, monkeypatch):
    """Step 1 precompute, step 2 synthesise with the returned blobs — the
    documented flow other services were built against."""
    monkeypatch.setattr(
        server,
        "_resolve_speaker_lenient",
        lambda synth, ref: types.SimpleNamespace(
            ttl=np.ones(8, dtype="float32"), dp=np.ones(4, dtype="float32")
        ),
    )
    cond = client.post("/v1/tts/conditioning", json={"speaker_files": ["M1"]})
    assert cond.status_code == 200, cond.text
    blob = cond.json()
    assert blob["gpt_cond_latent_b64"] and blob["speaker_embeddings_b64"]

    r = client.post(
        "/v1/audio/speech",
        json={
            "input": "Round trip.",
            "gpt_cond_latent_b64": blob["gpt_cond_latent_b64"],
            "speaker_embeddings_b64": blob["speaker_embeddings_b64"],
            "stream": True,
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.content) > 0


def test_voice_as_base64_audio_is_tolerated_not_rejected(client):
    """XTTS clients put reference WAV bytes in `voice`. Supertonic can't
    clone from audio, but the request must still return audio."""
    wav = io.BytesIO()
    import wave

    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(NATIVE_SR)
        w.writeframes(_tone(NATIVE_SR, 0.1).tobytes())
    r = client.post(
        "/v1/audio/speech",
        json={"input": "Hi.", "voice": [base64.b64encode(wav.getvalue()).decode()]},
    )
    assert r.status_code == 200
    assert len(r.content) > 0


# ===================== TTFB: the stream must stay chunked =====================

# A real uvicorn subprocess, because TestClient cannot prove this.
# httpx's ASGI transport drains the response before handing it back, so an
# in-process `client.stream(...)` reports the same timings whether the app
# streams or buffers — verified by running this assertion against the
# pre-middleware server, where it also "failed". The header contract
# (test_speech_endpoint.test_pcm_is_streamed_not_buffered) is all TestClient
# can check; incremental delivery needs a socket.

_LIVE_APP = """
import sys, time, types
# Two directories, not one: `server` lives in app/src/, the stub helpers
# imported just below live in app/tests/. conftest.py does the same for the
# parent pytest process, but this subprocess never loads it.
sys.path.insert(0, {src_dir!r})
sys.path.insert(0, {tests_dir!r})
from test_speech_endpoint import FakeSynth, _install_stubs, _tone, NATIVE_SR
_install_stubs()
import server
from server import State, create_app

CHUNKS, GAP = 5, 0.3

class SlowSynth(FakeSynth):
    def stream_pcm(self, *, text, voice_style, **kwargs):
        tone = _tone(NATIVE_SR, 2.5).tobytes()
        step = (len(tone) // CHUNKS) & ~1
        for i in range(CHUNKS):
            if i:
                time.sleep(GAP)      # stand in for per-chunk inference
            yield tone[i * step:(i + 1) * step] if i < CHUNKS - 1 else tone[i * step:]

State.synth = SlowSynth()
State.ready = True
server.detect_language = lambda text, hint=None: None
app = create_app()
"""


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Serve the real app over a real socket for the duration of the module."""
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("uvicorn")
    import socket
    import subprocess

    tmp = tmp_path_factory.mktemp("live")
    (tmp / "live_app.py").write_text(
        _LIVE_APP.format(
            src_dir=str(Path(__file__).resolve().parent.parent / "src"),
            tests_dir=str(Path(__file__).resolve().parent),
        )
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "live_app:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server died:\n{proc.stdout.read().decode()}")
            try:
                httpx.get(f"{base}/health", timeout=1).raise_for_status()
                break
            except Exception:
                time.sleep(0.2)
        else:
            pytest.fail("server never became healthy")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_pcm_arrives_incrementally_over_a_real_socket(live_server):
    """TTFB is the entire reason the PCM path exists. A client must get
    the first chunk long before the utterance finishes synthesising."""
    import httpx

    t0 = time.perf_counter()
    arrivals = []
    with httpx.Client(timeout=60) as c:
        with c.stream(
            "POST", f"{live_server}/v1/audio/speech",
            json={"input": "Hi.", "voice": "M1"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("audio/pcm")
            assert r.headers.get("transfer-encoding") == "chunked"
            assert "content-length" not in r.headers
            for chunk in r.iter_bytes():
                if chunk:
                    arrivals.append(time.perf_counter() - t0)

    assert len(arrivals) >= 3, f"delivered in {len(arrivals)} piece(s), not streamed"
    # 5 chunks, 0.3 s apart -> the last lands ~1.2 s after the first.
    assert arrivals[-1] - arrivals[0] > 0.6, "all chunks landed at once"
    assert arrivals[0] < arrivals[-1] / 2, f"TTFB {arrivals[0]:.2f}s is not early"


def test_encoded_format_is_buffered_over_a_real_socket(live_server):
    """The contrast case: wav must arrive as one sized file, not chunked."""
    import httpx

    with httpx.Client(timeout=60) as c:
        r = c.post(
            f"{live_server}/v1/audio/speech",
            json={"input": "Hi.", "voice": "M1", "response_format": "wav"},
        )
    assert r.status_code == 200
    assert r.headers["content-length"] == str(len(r.content))
    assert r.content.startswith(b"RIFF")


def test_mid_stream_error_marker_is_unchanged(monkeypatch):
    """Legacy clients parse the literal b"[error] " tail. Headers are long
    gone by then, so this is the only failure signal they get."""
    with _make_client(monkeypatch, FakeSynth(fail_after=1)) as client:
        r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "M1"})
        assert r.status_code == 200
        assert b"\n[error] " in r.content


# ===================== logging must be side-effect free =====================

@pytest.mark.parametrize("flag", ["1", "0"])
def test_body_logging_does_not_alter_the_audio(monkeypatch, flag):
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", flag)
    with _make_client(monkeypatch) as client:
        r = client.post("/v1/audio/speech", json=LEGACY_BODY)
    assert r.status_code == 200
    expected = int(TONE_SECONDS * 24000) * 2
    assert abs(len(r.content) - expected) < 400


def test_logging_on_and_off_produce_identical_bytes(monkeypatch):
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", "1")
    with _make_client(monkeypatch) as client:
        on = client.post("/v1/audio/speech", json=LEGACY_BODY).content
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", "0")
    with _make_client(monkeypatch) as client:
        off = client.post("/v1/audio/speech", json=LEGACY_BODY).content
    assert on == off


def test_handler_still_sees_every_field_after_the_body_was_logged(monkeypatch):
    """The middleware drains the request body. If the replay ever breaks,
    the handler gets an empty body — assert the synth actually received
    the text and voice the client sent."""
    monkeypatch.setenv("SUPERTONIC_LOG_REQUEST_BODY", "1")
    synth = FakeSynth()
    with _make_client(monkeypatch, synth) as client:
        r = client.post("/v1/audio/speech", json=LEGACY_BODY)
    assert r.status_code == 200
    assert synth.calls, "the handler never reached the synth"
    assert synth.calls[0]["text"] == LEGACY_BODY["input"]


def test_unparseable_body_still_422s_the_same_way(client):
    """The middleware must not convert a validation failure into a 500."""
    r = client.post(
        "/v1/audio/speech",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
