"""Data models for the supertonic streaming TTS service.

Everything here is declarative — what a request looks like, what an output
format is. No I/O, no GPU, no synthesis. The modules that *act* on these
live next door:

    audio_format.py   resolve / negotiate / encode, using AudioFormat
    server.py         the HTTP routes, using the pydantic models
    streaming_synth.py  synthesis

This module imports nothing from its siblings, which keeps it a leaf of the
import graph and makes it safe to import from any of them.

Two groups live here:

1. **Audio output formats** — the AudioFormat record and the registry of
   the five containers this service emits.
2. **HTTP request/response bodies** — the OpenAI/XTTS-compatible pydantic
   models for /v1/tts/conditioning and /v1/audio/speech.

They share a file because SpeechRequest's own ``response_format`` field
documents itself from SUPPORTED_RESPONSE_FORMATS; splitting them would put
an import edge between two halves of one request contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ===================== audio output formats =====================

class UnsupportedAudioFormat(ValueError):
    """Raised by :func:`resolve_format` for a response_format we can't emit.

    The server maps this to a 400 with code ``unsupported_response_format``.
    """


@dataclass(frozen=True)
class AudioFormat:
    """A resolved output format.

    ``sf_format``/``sf_subtype`` are the libsndfile names passed to
    ``soundfile``; both are None for raw PCM, which needs no encoder.
    """

    name: str
    media_type: str
    streaming: bool
    sf_format: Optional[str] = None
    sf_subtype: Optional[str] = None


# Ogg Opus is fixed-rate by design: libsndfile hard-errors on anything
# else ("Opus only supports sample rates of 8000, 12000, 16000, 24000,
# and 48000"). The server's default output rate (24000) is in the set, so
# this only bites clients that ask for an odd rate *and* Opus.
OPUS_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)

FORMATS = {
    "pcm": AudioFormat("pcm", "audio/pcm", streaming=True),
    "wav": AudioFormat("wav", "audio/wav", streaming=False,
                       sf_format="WAV", sf_subtype="PCM_16"),
    "flac": AudioFormat("flac", "audio/flac", streaming=False,
                        sf_format="FLAC", sf_subtype="PCM_16"),
    # Opus always lives in an Ogg container here; `audio/opus` is a raw
    # packet stream, which is not what these bytes are.
    "opus": AudioFormat("opus", "audio/ogg; codecs=opus", streaming=False,
                        sf_format="OGG", sf_subtype="OPUS"),
    # libsndfile >= 1.1 writes MPEG Layer III, and the soundfile wheel
    # bundles 1.2.2 — so this costs no extra dependency and no ffmpeg,
    # despite mp3's reputation. It was refused here until that was actually
    # checked.
    #
    # `audio/mpeg`, not `audio/mp3`: the latter is not a registered media
    # type, and some clients sniff on it and fail.
    "mp3": AudioFormat("mp3", "audio/mpeg", streaming=False,
                       sf_format="MP3", sf_subtype="MPEG_LAYER_III"),
}

SUPPORTED_RESPONSE_FORMATS = tuple(FORMATS)

DEFAULT_RESPONSE_FORMAT = "pcm"


# ===================== HTTP request / response bodies =====================

class ConditioningRequest(BaseModel):
    """XTTS-compatible conditioning request.

    Only ``speaker_files[0]`` is consumed (Supertonic is a single-speaker
    model). ``speaker_files`` is *optional* — if omitted (or pointing at
    something Supertonic can't honor like a raw audio file), we resolve to
    the default built-in voice. All other XTTS fields are accepted for API
    compatibility and ignored.
    """
    # extra="ignore" is what keeps the XTTS legacy fields working
    # (gpt_cond_len, gpt_cond_chunk_len, load_sample_rate, max_ref_length,
    # sound_norm_refs, ...). They used to be declared here purely so the
    # generated OpenAPI schema would list them, which sold the docs reader a
    # lie: none of them has ever been read. Undeclared, they are still
    # accepted and still dropped — identical behaviour, honest schema.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"example": {"speaker_files": ["M1"]}},
    )

    speaker_files: Optional[List[str]] = Field(
        None,
        description=(
            "Voice to precompute. Only the first entry is used (Supertonic is "
            "single-speaker). Omit to get the default built-in voice."
        ),
    )
    # XTTS may pass `voice` here too, as a list or string. _coerce_voice
    # normalises both shapes at use time.
    voice: Optional[Union[str, List[str]]] = Field(
        None, description="Alternative to speaker_files. Accepts a string or a list."
    )


class ConditioningResponse(BaseModel):
    gpt_cond_latent_b64: str
    speaker_embeddings_b64: str
    sample_rate: int
    model: str


# Fields SpeechRequest carries for internal use and hides from the published
# schema. `input_from` is written by the chat-shape validator, never by a
# caller. Module-level rather than a class attribute: a leading underscore
# inside a BaseModel becomes a pydantic private attr, not a constant.
_SPEECH_INTERNAL_FIELDS = ("input_from",)


class SpeechRequest(BaseModel):
    """OpenAI/XTTS-compatible /v1/audio/speech request.

    Speaker selection is, in order of precedence:
      1. ``gpt_cond_latent_b64`` + ``speaker_embeddings_b64`` (precomputed)
      2. ``voice`` (built-in name)
      3. error
    """
    model_config = ConfigDict(
        extra="ignore",
        # What Swagger's "Try it out" prefills. Without it, FastAPI renders
        # every field of the model, which reads as a menu of things the
        # caller is expected to fill in. Four fields cover the overwhelming
        # majority of real requests; the rest are documented individually
        # and stay optional.
        json_schema_extra={
            "example": {
                "input": "Hello! Today is a good day to ship something.",
                "model": "supertonic-3",
                "voice": "M1",
                "response_format": "pcm",
            }
        },
    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Drop internal fields from the published schema.

        Pydantic's `Field(exclude=True)` only affects serialisation — the
        field still shows up in the JSON schema, and therefore in Swagger's
        request body. Editing the generated schema is the supported way to
        hide one.
        """
        schema = handler(core_schema)
        props = schema.get("properties", {})
        for name in _SPEECH_INTERNAL_FIELDS:
            props.pop(name, None)
        return schema

    # --- core ---
    input: str = Field(
        ...,
        min_length=1,
        description=(
            "Text to synthesise. Required — except that a chat-completions "
            "body (`messages`) is also accepted, and the text is then taken "
            "from the last user turn. See the endpoint description."
        ),
    )
    # Set by the validator below, not by the client: "input" or "messages".
    # Hidden from the schema via _SPEECH_INTERNAL_FIELDS.
    input_from: str = "input"
    model: str = Field("supertonic-3", description="Loaded model name (informational)")
    response_format: str = Field(
        "pcm",
        description=(
            "Output container: " + ", ".join(SUPPORTED_RESPONSE_FORMATS) + ". "
            "Defaults to 'pcm' (headerless int16 LE mono, streamed) so existing "
            "XTTS/pipecat clients are unaffected. Every other format returns one "
            "complete buffered file. 'aac' is rejected with a 400 — libsndfile "
            "cannot write it."
        ),
    )
    stream: bool = Field(
        True,
        description=(
            "Informational. Chunked transfer happens for response_format='pcm'; "
            "the encoded containers are always sent whole."
        ),
    )
    language: Optional[str] = Field(
        "auto",
        description="ISO code, 'na', or 'auto' for the multilingual fallback",
    )
    # XTTS clients send `voice` as a list of base64 reference audios;
    # OpenAI clients send a plain string. Accept either; _coerce_voice()
    # normalises both at use time.
    voice: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Built-in voice name (M1..M5 / F1..F5) or path to a Voice "
            "Builder .json. Also accepts a list (XTTS shape) — first item "
            "is used. Raw audio bytes in this field are detected, logged, "
            "and ignored (Supertonic open-weight does not clone from audio)."
        ),
    )

    # `instructions` — OpenAI's voice-steering prompt — is deliberately not
    # declared. Supertonic has no such control surface (voice character comes
    # entirely from the style blob), so declaring it advertised a knob that
    # does nothing. extra="ignore" still accepts it from OpenAI SDK clients.
    # OpenAI's SSE audio streaming selector. We don't implement SSE, and
    # answering an "sse" request with raw bytes would hand the client a
    # stream it cannot parse — so this one is rejected loudly rather than
    # ignored. Absent / "audio" behaves exactly as before.
    stream_format: Optional[str] = None

    # --- conditioning (precomputed) ---
    gpt_cond_latent_b64: Optional[str] = None
    speaker_embeddings_b64: Optional[str] = None

    # --- supertonic streaming/quality controls ---
    # No strict bounds: XTTS clients sometimes pass values outside our
    # preferred ranges. Accept anything that parses, then clamp internally
    # at use time so the request never 422s on a numeric edge case.
    speed: Optional[float] = 1.0
    chunk_size: Optional[int] = Field(
        None,
        description=(
            "Maps to supertonic's max_chunk_length (text-chunk pack target in chars). "
            "If omitted, the server picks a sensible default."
        ),
    )
    total_steps: Optional[int] = Field(6, description="Diffusion steps (clamped to [1, 16])")
    # Output PCM sample rate. supertonic synthesises at 44100; we resample
    # server-side. Default = 24000 to match XTTS / pipecat clients.
    sample_rate: Optional[int] = Field(
        None,
        description=(
            "Output sample rate in Hz (default 24000 unless overridden "
            "by SUPERTONIC_OUTPUT_SAMPLE_RATE). Supertonic synthesises at "
            "44100; the server resamples on the way out using audioop.ratecv "
            "(stdlib, sub-millisecond per chunk on CPU). With "
            "response_format='opus' the rate is snapped to the nearest one "
            "Opus supports (8/12/16/24/48 kHz); the response's X-Sample-Rate "
            "header always reports the rate actually used."
        ),
    )

    # XTTS legacy fields (temperature, top_k, top_p, repetition_penalty,
    # do_sample, enhance_speech, length_penalty, gpt_cond_len,
    # gpt_cond_chunk_len, max_ref_length) are NOT declared here.
    #
    # Supertonic is diffusion-based, not autoregressive, so not one of them
    # has ever affected a single sample of output. Declaring them put ten
    # dead knobs in the OpenAPI schema and in Swagger's "Try it out" body,
    # where they read as things a caller can tune. extra="ignore" accepts
    # and drops them exactly as before — see
    # tests/test_legacy_api_compat.py, which posts all ten.
    #
    # Four of them used to be echoed in the per-request log line. The raw
    # request-body logging (SUPERTONIC_LOG_REQUEST_BODY, on by default for
    # this route) already records whatever the client actually sent, so
    # nothing was lost by dropping them from that line either.

    @model_validator(mode="before")
    @classmethod
    def _accept_chat_shape(cls, data: Any) -> Any:
        """Fill `input` from chat `messages` when the caller sent that shape.

        A gateway health check POSTs a chat-completions body here and reads
        a 422 as "the model is down", so the whole route gets marked dead
        over a field name. Synthesising the probe's text answers it for the
        cost of one short utterance.

        Only fires when `input` is absent — an explicit `input` always wins,
        and a body carrying neither still fails validation rather than
        quietly synthesising nothing.
        """
        if not isinstance(data, dict):
            return data
        if data.get("input"):
            return data
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return data

        def _text(content: Any) -> str:
            # Content is a plain string, or OpenAI's list of typed parts.
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return ""

        # Last user turn first — that's the prompt. Fall back to the last
        # message of any role, so a lone system-role probe still works.
        for pick in (
            lambda m: m.get("role") == "user",
            lambda m: True,
        ):
            for msg in reversed(msgs):
                if not isinstance(msg, dict) or not pick(msg):
                    continue
                text = _text(msg.get("content")).strip()
                if text:
                    data = dict(data)
                    data["input"] = text
                    data["input_from"] = "messages"
                    return data
        return data
