"""Unit tests for audio_format.py.

Run with the service venv (needs numpy + soundfile, no GPU / no model):

    pytest app/tests/test_audio_format.py -q
    # or, without pytest:
    python app/tests/test_audio_format.py
"""

from __future__ import annotations

import io
import math
import struct

import numpy as np
import pytest
import soundfile as sf

from audio_format import (  # noqa: E402
    encode_pcm,
    negotiate_sample_rate,
    resolve_format,
)
from models import (  # noqa: E402
    OPUS_SAMPLE_RATES,
    SUPPORTED_RESPONSE_FORMATS,
    UnsupportedAudioFormat,
)


def _tone_pcm(sample_rate: int, seconds: float = 0.5, hz: float = 440.0) -> bytes:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (np.sin(2 * math.pi * hz * t) * 12000).astype("<i2").tobytes()


# ===================== resolve_format =====================

def test_default_is_pcm_and_streams():
    """An omitted response_format must behave exactly like today: raw PCM,
    streamed, media type audio/pcm. This is the back-compat contract for
    existing XTTS / pipecat clients."""
    fmt = resolve_format(None)
    assert fmt.name == "pcm"
    assert fmt.streaming is True
    assert fmt.media_type == "audio/pcm"


@pytest.mark.parametrize("name", sorted(SUPPORTED_RESPONSE_FORMATS))
def test_every_supported_format_resolves(name):
    assert resolve_format(name).name == name


def test_format_matching_is_case_and_space_insensitive():
    assert resolve_format("  WAV ").name == "wav"


@pytest.mark.parametrize("name", ["aac", "", "ogg", "flacc", "nonsense"])
def test_unsupported_formats_raise(name):
    with pytest.raises(UnsupportedAudioFormat):
        resolve_format(name)


def test_unsupported_error_lists_what_is_supported():
    """The 400 body is the only place a client learns the menu, so the
    message has to carry it."""
    with pytest.raises(UnsupportedAudioFormat) as exc:
        resolve_format("aac")
    msg = str(exc.value)
    assert "aac" in msg
    for name in SUPPORTED_RESPONSE_FORMATS:
        assert name in msg


def test_only_pcm_streams():
    assert resolve_format("pcm").streaming is True
    for name in ("wav", "flac", "opus", "mp3"):
        assert resolve_format(name).streaming is False


def test_media_types():
    assert resolve_format("wav").media_type == "audio/wav"
    assert resolve_format("flac").media_type == "audio/flac"
    assert resolve_format("opus").media_type.startswith("audio/ogg")
    # audio/mpeg, not audio/mp3 — the latter is not a registered media type.
    assert resolve_format("mp3").media_type == "audio/mpeg"


# ===================== negotiate_sample_rate =====================

@pytest.mark.parametrize("name", ["pcm", "wav", "flac", "mp3"])
@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_non_opus_keeps_requested_rate(name, rate):
    assert negotiate_sample_rate(resolve_format(name), rate) == rate


@pytest.mark.parametrize("rate", sorted(OPUS_SAMPLE_RATES))
def test_opus_keeps_natively_supported_rates(rate):
    assert negotiate_sample_rate(resolve_format("opus"), rate) == rate


@pytest.mark.parametrize(
    "requested,expected",
    [
        (22050, 24000),   # nearest of 16000/24000 by ratio
        (44100, 48000),
        (11025, 12000),
        (6000, 8000),     # below the floor
        (96000, 48000),   # above the ceiling
    ],
)
def test_opus_snaps_unsupported_rates(requested, expected):
    """libsndfile refuses Opus at anything but 8/12/16/24/48 kHz. Snapping
    keeps the request working instead of 500-ing deep in the encoder."""
    assert negotiate_sample_rate(resolve_format("opus"), requested) == expected


# ===================== encode_pcm =====================

def test_pcm_encode_is_byte_identical_passthrough():
    pcm = _tone_pcm(24000)
    assert encode_pcm(resolve_format("pcm"), pcm, 24000) is pcm


@pytest.mark.parametrize("name", ["wav", "flac", "opus", "mp3"])
def test_encoded_output_round_trips(name):
    sr = 24000
    pcm = _tone_pcm(sr)
    blob = encode_pcm(resolve_format(name), pcm, sr)
    assert blob, "encoder produced no bytes"
    data, out_sr = sf.read(io.BytesIO(blob), dtype="int16", always_2d=True)
    assert out_sr == sr
    assert data.shape[1] == 1, "output must be mono"
    # Opus and MP3 are lossy and frame-padded; WAV/FLAC must be
    # sample-exact in length.
    expected_frames = len(pcm) // 2
    if name in ("opus", "mp3"):
        assert data.shape[0] >= expected_frames
    else:
        assert data.shape[0] == expected_frames


def test_wav_is_lossless_round_trip():
    sr = 24000
    pcm = _tone_pcm(sr)
    blob = encode_pcm(resolve_format("wav"), pcm, sr)
    data, _ = sf.read(io.BytesIO(blob), dtype="int16")
    assert data.tobytes() == pcm


def test_flac_is_lossless_round_trip():
    sr = 24000
    pcm = _tone_pcm(sr)
    blob = encode_pcm(resolve_format("flac"), pcm, sr)
    data, _ = sf.read(io.BytesIO(blob), dtype="int16")
    assert data.tobytes() == pcm


def test_wav_header_declares_real_sizes():
    """A streamed WAV with placeholder sizes trips strict parsers (python's
    stdlib `wave` among them). We buffer precisely so the header is exact."""
    sr = 24000
    pcm = _tone_pcm(sr)
    blob = encode_pcm(resolve_format("wav"), pcm, sr)
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"
    riff_size = struct.unpack("<I", blob[4:8])[0]
    assert riff_size == len(blob) - 8

    import wave

    with wave.open(io.BytesIO(blob)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == sr
        assert w.getnframes() == len(pcm) // 2
        assert w.readframes(w.getnframes()) == pcm


@pytest.mark.parametrize("name", ["wav", "flac", "opus"])
def test_empty_pcm_still_produces_a_valid_container(name):
    """Synthesis can legitimately yield zero bytes (empty-ish input). The
    client should get a decodable, near-silent file rather than a blob
    libsndfile itself refuses to reopen."""
    blob = encode_pcm(resolve_format(name), b"", 24000)
    assert blob
    data, out_sr = sf.read(io.BytesIO(blob), dtype="int16")
    assert out_sr == 24000
    assert len(data) < 24000, "empty input must not produce audible audio"
    assert not np.any(data), "empty input must encode as silence"


def test_odd_length_pcm_drops_the_trailing_half_sample():
    """int16 frames are 2 bytes; a stray odd byte must not blow up the
    encoder."""
    sr = 24000
    pcm = _tone_pcm(sr) + b"\x01"
    blob = encode_pcm(resolve_format("wav"), pcm, sr)
    data, _ = sf.read(io.BytesIO(blob), dtype="int16")
    assert len(data) == len(pcm) // 2


def test_opus_encode_rejects_unsupported_rate():
    """encode_pcm trusts the caller to have run negotiate_sample_rate first;
    if it didn't, fail loudly rather than emitting a corrupt container."""
    with pytest.raises(Exception):
        encode_pcm(resolve_format("opus"), _tone_pcm(44100), 44100)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
