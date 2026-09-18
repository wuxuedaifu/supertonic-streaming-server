"""Output container/codec handling for /v1/audio/speech `response_format`.

OpenAI's speech API lets the client pick the wire format. This module holds
the *behaviour* of that support: it maps a `response_format` string onto a
container spec, reconciles the requested sample rate with what the codec
can actually do, and encodes a finished PCM buffer into that container.

The declarative half — the :class:`~models.AudioFormat` record, the
``FORMATS`` registry of the five containers, ``OPUS_SAMPLE_RATES`` and
:class:`~models.UnsupportedAudioFormat` — lives in ``models.py`` with the
rest of the service's data models.

Two classes of format, and the difference is deliberate:

  * ``pcm``  — headerless int16 LE mono, **streamed** chunk-by-chunk as
    synthesis produces it. This is the pre-existing behaviour of this
    server and the default when the client omits ``response_format``, so
    XTTS / pipecat clients keep getting byte-identical responses with the
    same low TTFB.

  * ``wav`` / ``flac`` / ``opus`` / ``mp3`` — the server buffers the full
    PCM, encodes once, and returns a complete file with an exact
    Content-Length. Compressed containers can't be streamed usefully
    without a header patch at close time (a streamed WAV would have to
    carry placeholder RIFF sizes, which strict parsers — python's stdlib
    ``wave`` among them — reject), and OpenAI's own API only streams
    ``pcm``/``wav``. Clients that need first-byte latency should ask for
    ``pcm``.

Encoding rides on ``soundfile``, which is already a dependency (and whose
wheels bundle libsndfile ≥ 1.2, so FLAC and Ogg Opus are available without
touching the image). No ffmpeg, no subprocesses.

``aac`` is deliberately *not* supported: libsndfile cannot write it, so it
would mean adding an encoder (ffmpeg) for one format. It raises
:class:`UnsupportedAudioFormat`, which the server turns into a 400 listing
what it does support.

``mp3`` was refused on the same grounds until that was checked rather than
assumed — libsndfile has written MPEG Layer III since 1.1, and the
``soundfile`` wheel bundles 1.2.2, so it costs nothing and is now
supported.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import soundfile as sf

from models import (
    DEFAULT_RESPONSE_FORMAT,
    FORMATS,
    OPUS_SAMPLE_RATES,
    SUPPORTED_RESPONSE_FORMATS,
    AudioFormat,
    UnsupportedAudioFormat,
)


def resolve_format(response_format: Optional[str]) -> AudioFormat:
    """Map a client-supplied ``response_format`` onto an :class:`AudioFormat`.

    ``None`` resolves to the back-compatible default (raw PCM). Unknown
    values raise :class:`UnsupportedAudioFormat` — unlike the speaker and
    language fields, we don't silently fall back here: handing a client
    PCM when it asked for MP3 produces a file it can't play, and a loud
    400 is far easier to debug than white noise.
    """
    if response_format is None:
        return FORMATS[DEFAULT_RESPONSE_FORMAT]
    key = response_format.strip().lower()
    fmt = FORMATS.get(key)
    if fmt is None:
        raise UnsupportedAudioFormat(
            f"unsupported response_format {response_format!r}; "
            f"supported: {', '.join(SUPPORTED_RESPONSE_FORMATS)}"
        )
    return fmt


def negotiate_sample_rate(fmt: AudioFormat, sample_rate: int) -> int:
    """Return the rate we can actually encode ``fmt`` at.

    Only Opus constrains this. Rather than 400-ing on
    ``response_format=opus`` + ``sample_rate=44100``, snap to the nearest
    supported rate (by frequency ratio, so 44100 → 48000 rather than
    24000) and let the caller's resampler target that instead. The
    response advertises the rate actually used via ``X-Sample-Rate``.
    """
    rate = int(sample_rate)
    if fmt.name != "opus" or rate in OPUS_SAMPLE_RATES:
        return rate
    return min(OPUS_SAMPLE_RATES, key=lambda r: abs(np.log(r / rate)))


def encode_pcm(fmt: AudioFormat, pcm: bytes, sample_rate: int) -> bytes:
    """Encode int16 LE mono ``pcm`` into ``fmt``'s container.

    For ``pcm`` this is the identity (the exact object is returned, so the
    streaming path stays allocation-free). ``sample_rate`` must already
    have been through :func:`negotiate_sample_rate`; if it hasn't,
    libsndfile raises and the request fails loudly rather than producing
    an unplayable file.

    A trailing odd byte (half a frame) is dropped — synthesis never emits
    one, but a truncated upstream chunk shouldn't crash the encoder.
    """
    if fmt.sf_format is None:
        return pcm

    frames = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if frames.size == 0:
        # libsndfile emits a header-only FLAC/Ogg stream for zero frames
        # that it then refuses to reopen ("file is malformed"), so a
        # silent client would get an undecodable blob. One sample of
        # silence (~0.04 ms) makes the container well-formed.
        frames = np.zeros(1, dtype="<i2")
    buf = io.BytesIO()
    with sf.SoundFile(
        buf,
        mode="w",
        samplerate=int(sample_rate),
        channels=1,
        format=fmt.sf_format,
        subtype=fmt.sf_subtype,
    ) as out:
        out.write(frames)
    # SoundFile.close() patches the header sizes in place before we read
    # the buffer back, which is exactly why this path buffers instead of
    # streaming.
    return buf.getvalue()
