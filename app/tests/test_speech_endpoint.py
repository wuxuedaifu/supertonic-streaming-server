"""HTTP-level tests for POST /v1/audio/speech.

Covers the two things that matter after adding OpenAI's ``response_format``:

  1. the pre-existing raw-PCM contract is byte-for-byte unchanged, and
  2. wav / flac / opus / mp3 come back as complete, decodable files.

No GPU, no model weights: ``supertonic`` and ``onnxruntime`` are stubbed
and ``State.synth`` is replaced with a fake that emits a known tone, so
this runs anywhere numpy + soundfile + fastapi are installed.

    pytest app/tests/test_speech_endpoint.py -q
"""

from __future__ import annotations

import io
import math
import sys
import types
import wave

import numpy as np
import pytest
import soundfile as sf


# ===================== stub the heavy deps =====================
# server.py -> streaming_synth.py -> {onnxruntime, supertonic}. We only
# need the module graph to import; nothing below is exercised, because the
# tests swap State.synth for a fake.

def _install_stubs() -> None:
    if "onnxruntime" not in sys.modules:
        ort = types.ModuleType("onnxruntime")
        ort.get_available_providers = lambda: []
        ort.set_default_logger_severity = lambda level: None
        ort.InferenceSession = object
        sys.modules["onnxruntime"] = ort

    if "supertonic" in sys.modules:
        return

    st = types.ModuleType("supertonic")

    class Style:  # noqa: D401 - stand-in for supertonic.Style
        def __init__(self, ttl, dp):
            self.ttl, self.dp = ttl, dp

    st.Style = Style
    st.TTS = object
    st.__version__ = "stub"

    cfg = types.ModuleType("supertonic.config")
    cfg.DEFAULT_MAX_CHUNK_LENGTH = 135
    cfg.DEFAULT_MAX_CHUNK_LENGTH_KO = 90
    cfg.UNKNOWN_LANGUAGE = "na"

    utils = types.ModuleType("supertonic.utils")
    utils.chunk_text = lambda text, **kw: [text]

    st.config, st.utils = cfg, utils
    sys.modules.update(
        {"supertonic": st, "supertonic.config": cfg, "supertonic.utils": utils}
    )


_install_stubs()

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from server import State, create_app  # noqa: E402


NATIVE_SR = 44100
TONE_HZ = 440.0
TONE_SECONDS = 0.5


def _tone(sample_rate: int, seconds: float, hz: float = TONE_HZ) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (np.sin(2 * math.pi * hz * t) * 12000).astype("<i2")


class FakeSynth:
    """Stand-in for StreamingSynthesize: emits a tone in three chunks."""

    def __init__(self, fail_after: int | None = None):
        self.sample_rate = NATIVE_SR
        self.model_name = "supertonic-3-fake"
        self.tts = types.SimpleNamespace(
            voice_style_names=["M1", "M2", "F1"],
            get_voice_style_from_path=lambda p: object(),
        )
        self.fail_after = fail_after
        self.calls: list[dict] = []

    def get_voice_style(self, name):
        return f"style:{name}"

    def stream_pcm(self, *, text, voice_style, **kwargs):
        self.calls.append({"text": text, "voice_style": voice_style, **kwargs})
        tone = _tone(NATIVE_SR, TONE_SECONDS).tobytes()
        third = (len(tone) // 3) & ~1
        chunks = [tone[:third], tone[third: 2 * third], tone[2 * third:]]
        for i, chunk in enumerate(chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("boom: inference exploded")
            yield chunk


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SUPERTONIC_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.setattr(State, "synth", FakeSynth())
    monkeypatch.setattr(State, "ready", True)
    # The LID path is orthogonal to output formatting; pin it off so a
    # short test phrase can't trip the unsupported-language refusal.
    monkeypatch.setattr(server, "detect_language", lambda text, hint=None: None)
    with TestClient(create_app()) as c:
        yield c


def _speak(client, **body):
    payload = {"input": "Hello from the format test.", "voice": "M1"}
    payload.update(body)
    return client.post("/v1/audio/speech", json=payload)


def _expected_pcm_frames(out_sr: int) -> int:
    return int(TONE_SECONDS * out_sr)


# ===================== back-compat: raw PCM =====================

def test_omitted_response_format_still_returns_raw_pcm(client):
    """The existing XTTS/pipecat contract: no response_format field, get
    headerless int16 LE mono at 24 kHz with no container bytes at all."""
    r = _speak(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-sample-rate"] == "24000"
    assert not r.content.startswith(b"RIFF")
    frames = len(r.content) // 2
    assert abs(frames - _expected_pcm_frames(24000)) < 200


def test_explicit_pcm_matches_omitted_pcm_byte_for_byte(client):
    assert _speak(client).content == _speak(client, response_format="pcm").content


def test_pcm_is_streamed_not_buffered(client):
    """TTFB matters for the PCM path: it must stay a chunked transfer with
    no Content-Length, so the client can start playing on the first chunk.
    (TestClient reassembles the body, so the header is the observable.)"""
    with client.stream(
        "POST", "/v1/audio/speech",
        json={"input": "Hello from the format test.", "voice": "M1"},
    ) as r:
        assert "content-length" not in r.headers, "PCM must not be buffered"
        assert sum(len(c) for c in r.iter_bytes()) > 0
    # Contrast: the encoded formats are whole files and do carry a length.
    assert "content-length" in _speak(client, response_format="wav").headers


def test_custom_sample_rate_is_honoured_for_pcm(client):
    r = _speak(client, sample_rate=16000)
    assert r.headers["x-sample-rate"] == "16000"
    assert abs(len(r.content) // 2 - _expected_pcm_frames(16000)) < 200


# ===================== OpenAI containers =====================

@pytest.mark.parametrize(
    "fmt,media_type",
    [("wav", "audio/wav"), ("flac", "audio/flac"), ("opus", "audio/ogg"),
     ("mp3", "audio/mpeg")],
)
def test_encoded_formats_return_a_decodable_file(client, fmt, media_type):
    r = _speak(client, response_format=fmt)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(media_type)
    assert r.headers["x-audio-format"] == fmt
    # A complete file, not a chunked stream.
    assert r.headers["content-length"] == str(len(r.content))

    data, sr = sf.read(io.BytesIO(r.content), dtype="int16", always_2d=True)
    assert sr == int(r.headers["x-sample-rate"])
    assert data.shape[1] == 1
    assert abs(data.shape[0] - _expected_pcm_frames(sr)) < sr * 0.1
    assert np.abs(data).max() > 1000, "decoded audio is silent"


def test_wav_is_readable_by_the_stdlib_wave_module(client):
    """The strictest common parser — it rejects the placeholder-size headers
    a naive streaming WAV implementation would emit."""
    r = _speak(client, response_format="wav")
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.getnframes() == len(r.content[44:]) // 2


def test_wav_payload_matches_the_pcm_response(client):
    """wav must be the pcm bytes plus a header — same audio, same rate."""
    pcm = _speak(client, response_format="pcm").content
    wav = _speak(client, response_format="wav").content
    decoded, sr = sf.read(io.BytesIO(wav), dtype="int16")
    assert sr == 24000
    assert decoded.tobytes() == pcm


def test_response_format_is_case_insensitive(client):
    assert _speak(client, response_format="WAV").content[:4] == b"RIFF"


def test_opus_snaps_an_unencodable_sample_rate(client):
    """44100 isn't an Opus rate; the server picks 48000 and says so."""
    r = _speak(client, response_format="opus", sample_rate=44100)
    assert r.status_code == 200
    assert r.headers["x-sample-rate"] == "48000"
    _, sr = sf.read(io.BytesIO(r.content), dtype="int16")
    assert sr == 48000


def test_pcm_does_not_snap_sample_rate(client):
    """Snapping is an Opus-only concession; raw PCM keeps what was asked."""
    assert _speak(client, sample_rate=44100).headers["x-sample-rate"] == "44100"


# ===================== rejections =====================

@pytest.mark.parametrize("fmt", ["aac", "ogg", "flacc", "nonsense"])
def test_unsupported_response_format_is_a_400(client, fmt):
    r = _speak(client, response_format=fmt)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_response_format"
    assert "wav" in err["message"] and "flac" in err["message"]


def test_sse_stream_format_is_rejected_not_silently_ignored(client):
    """Answering an SSE request with raw bytes would look like corruption
    on the client side."""
    r = _speak(client, response_format="pcm", stream_format="sse")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_stream_format"


@pytest.mark.parametrize("stream_format", [None, "audio"])
def test_audio_stream_format_is_accepted(client, stream_format):
    assert _speak(client, stream_format=stream_format).status_code == 200


# ===================== OpenAI SDK-shaped bodies =====================

def test_openai_sdk_body_is_accepted(client):
    """What openai-python's audio.speech.create() puts on the wire."""
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "gpt-4o-mini-tts",
            "input": "Hello from the OpenAI SDK.",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.0,
            "instructions": "Speak in a cheerful tone.",
        },
    )
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_unknown_openai_voice_falls_back_instead_of_erroring(client):
    """OpenAI's voice names (alloy, nova, ...) have no supertonic
    equivalent; the server's lenient resolver must still answer."""
    r = _speak(client, voice="nova", response_format="wav")
    assert r.status_code == 200
    assert State.synth.calls[-1]["voice_style"] == "style:M1"


def test_instructions_field_is_ignored_not_synthesised(client):
    _speak(client, instructions="Whisper this.", response_format="wav")
    assert "Whisper" not in State.synth.calls[-1]["text"]


# ===================== failure paths =====================

def test_pcm_failure_still_appends_the_legacy_error_marker(client, monkeypatch):
    """Once bytes are on the wire there's no status code left, so the
    long-standing in-band marker has to stay."""
    monkeypatch.setattr(State, "synth", FakeSynth(fail_after=1))
    r = _speak(client)
    assert r.status_code == 200
    assert b"[error]" in r.content
    assert b"boom: inference exploded" in r.content


def test_encoded_failure_is_a_clean_500_not_a_corrupt_file(client, monkeypatch):
    """Nothing has been sent yet for buffered formats — so report properly
    instead of handing back an audio file with error text glued on."""
    monkeypatch.setattr(State, "synth", FakeSynth(fail_after=1))
    r = _speak(client, response_format="wav")
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["code"] == "synthesis_failed"
    assert "boom" in err["message"]


def test_failed_encoded_request_is_counted_once(client, monkeypatch):
    monkeypatch.setattr(State, "synth", FakeSynth(fail_after=0))
    before_failed, before_inflight = State.req_failed, State.req_inflight
    _speak(client, response_format="flac")
    assert State.req_failed == before_failed + 1
    assert State.req_inflight == before_inflight, "in-flight slot leaked"


def test_successful_encoded_request_releases_the_inflight_slot(client):
    before = State.req_inflight
    assert _speak(client, response_format="flac").status_code == 200
    assert State.req_inflight == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
