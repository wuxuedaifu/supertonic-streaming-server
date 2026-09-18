"""FastAPI streaming TTS server (OpenAI / XTTS-compatible).

Two-call flow mirrors the upstream XTTS / Auralis API so existing clients
work without code changes:

    POST /v1/tts/conditioning         { speaker_files: [...], ... }
       -> { gpt_cond_latent_b64, speaker_embeddings_b64 }

    POST /v1/audio/speech             OpenAI/XTTS-shaped body
       -> chunked transfer of int16 LE mono PCM (default), or a
          complete wav/flac/opus file when the client asks for one
          via OpenAI's ``response_format``

The conditioning step is conceptually unnecessary for Supertonic (voice
styles are tiny precomputed JSON blobs, not slow GPT pre-fills), but the
endpoint is kept so client code that already does the precompute-then-
synthesize dance can stay the way it is. Supertonic's open-weight model
does **not** clone voices from raw audio at request time — pass a
built-in voice name (M1..M5, F1..F5) or a Voice Builder JSON path as
``speaker_files[0]``. Raw audio paths are rejected with a 400.

XTTS legacy fields the client may send (``temperature``, ``top_k``,
``top_p``, ``do_sample``, ``enhance_speech``, ``length_penalty``,
``repetition_penalty``, ``gpt_cond_*``, ``max_ref_length``,
``sound_norm_refs``, ``load_sample_rate``) are accepted and silently
ignored — Supertonic is diffusion-based, not autoregressive, so those
knobs do not apply. They are deliberately **not declared** on the request
models and so do not appear in the OpenAPI schema: listing ten fields that
cannot change the output made Swagger read like a tuning surface. Accepting
them is a compatibility obligation; advertising them was a lie. See
models.py.

Output format follows OpenAI's ``response_format`` field. ``pcm`` is the
default and is unchanged from previous releases: headerless int16 LE mono,
streamed chunk-by-chunk for minimum TTFB. ``wav`` / ``flac`` / ``opus``
buffer the full utterance and return one complete, correctly-sized file
(see audio_format.py for why compressed containers can't stream here;
the request/response models themselves live in models.py).
``aac`` is rejected with a 400 rather than silently substituted.

If a request carries no ``input`` but does carry chat-completions
``messages``, the text is taken from the last user turn. Gateways probe a
model's liveness with ``{"model": ..., "messages": [...], "max_tokens": 1}``
aimed at whatever URL they hold for it, and read a 422 as "model down".

Every POST to /v1/audio/speech and /v1/tts/conditioning logs its raw body
(redacted and truncated) before validation runs, so operators can see what
a client actually sent — including fields the request model discards and
bodies that pydantic rejected. Toggle with SUPERTONIC_LOG_REQUEST_BODY.

Environment configuration is otherwise identical to the previous release;
see the SUPERTONIC_* env vars documented inline below and in
deploy/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from supertonic import Style  # type: ignore[import-not-found]

from audio_format import encode_pcm, negotiate_sample_rate, resolve_format
from models import (
    ConditioningRequest,
    ConditioningResponse,
    SpeechRequest,
    UnsupportedAudioFormat,
)
from streaming_synth import PcmResampler, StreamingSynthesize
from text_preprocess import detect_language, is_supertonic_lang, warmup_detector


logger = logging.getLogger("supertonic.server")


# ----- audio extensions we explicitly refuse for speaker_files -----
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus"}


# ===================== in-process state =====================

class State:
    synth: Optional[StreamingSynthesize] = None
    ready: bool = False
    model_name: str = "supertonic-3"
    started_at: float = time.time()
    req_total: int = 0
    req_inflight: int = 0
    req_failed: int = 0
    bytes_streamed: int = 0
    _lock = Lock()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ===================== base64 codec for Style =====================
# Style consists of two float32 numpy arrays (style_ttl, style_dp). We
# serialise each as an .npy buffer + base64. The client treats both
# strings as opaque blobs — it never parses them, only hands them back to
# /v1/audio/speech in the next call.

def _encode_array_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr.astype(np.float32)), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_array_b64(s: str) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return arr.astype(np.float32, copy=False)


# numpy .npy magic — what our /v1/tts/conditioning produces. Anything else
# (e.g. torch.save / PKZip from an XTTS server) is a foreign blob and we
# cannot use it as supertonic conditioning (different latent dims and
# semantics).
_NPY_MAGIC = b"\x93NUMPY"


def _try_decode_b64_npy(s: Optional[str]) -> Optional[np.ndarray]:
    """Best-effort decode of a base64 .npy blob. Returns None if the blob is
    absent, empty, in a foreign format (e.g. torch.save / PKZip), or otherwise
    not loadable as a numpy float32 array. Never raises."""
    if not s:
        return None
    try:
        raw = base64.b64decode(s.encode("ascii"), validate=False)
    except Exception:
        return None
    if not raw.startswith(_NPY_MAGIC):
        return None
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _default_voice(synth: StreamingSynthesize) -> str:
    """Fallback voice when the request can't be resolved otherwise. Operator
    can override with SUPERTONIC_DEFAULT_VOICE; we sanity-check it against the
    actual built-in list."""
    env = os.getenv("SUPERTONIC_DEFAULT_VOICE", "M1")
    if env in synth.tts.voice_style_names:
        return env
    return synth.tts.voice_style_names[0]


def _short(s: Optional[str], n: int = 24) -> str:
    """Truncate long strings for log lines. Clients sometimes stuff giant
    base64 blobs into `voice` (or anywhere else), and dumping the whole
    thing into INFO logs makes them unusable."""
    if s is None:
        return "None"
    if len(s) <= n:
        return repr(s)
    return f"{s[:n]!r}…+{len(s) - n}B"


def _redact_body(raw: bytes, max_chars: int, max_str: int = 96) -> str:
    """Render a request body for the log: readable, bounded, no blobs.

    Clients routinely put 60 KB of base64 conditioning latents in the body.
    Echoing that verbatim costs more log volume than the audio costs
    bandwidth and buries the fields you actually wanted to read, so every
    over-long string collapses to a head + length marker. Non-JSON bodies
    (or malformed ones, which are exactly the interesting case) fall back
    to a truncated text preview rather than raising."""
    def _cap(s: str) -> str:
        if len(s) <= max_chars:
            return s
        return f"{s[:max_chars]}\u2026+{len(s) - max_chars}B"

    text = raw.decode("utf-8", "replace")
    try:
        obj = json.loads(raw)
    except Exception:
        return _cap(text)

    def _walk(v: Any) -> Any:
        if isinstance(v, str) and len(v) > max_str:
            return f"<{len(v)}B str:{v[:24]}\u2026>"
        if isinstance(v, list):
            return [_walk(i) for i in v]
        if isinstance(v, dict):
            return {k: _walk(i) for k, i in v.items()}
        return v

    return _cap(json.dumps(_walk(obj), ensure_ascii=False, separators=(",", ":")))


def _coerce_voice(v: Any) -> Optional[str]:
    """Normalize the voice field across the wire-format zoo.

    XTTS clients send ``voice`` as a list of base64-encoded reference audio
    files (e.g. ``voice: ["UklGRg..."]``). OpenAI's API spec uses a plain
    string. Both should work — supertonic only cares about the first usable
    element. Accepts None, a string, or a list of strings.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    if isinstance(v, list):
        for item in v:
            if isinstance(item, str) and item:
                return item
        return None
    return str(v)


def _detect_b64_audio(s: Optional[str]) -> Optional[str]:
    """Return the audio format name if ``s`` looks like a base64-encoded
    audio file (WAV/MP3/OGG/FLAC), else None. Used so we can give a clear
    log message when XTTS clients pass raw audio in a field that
    supertonic open-weight can't honor (it doesn't clone from audio at
    request time)."""
    if not s or len(s) < 24:
        return None
    try:
        head = base64.b64decode(s[:48], validate=False)
    except Exception:
        return None
    if head.startswith(b"RIFF") and b"WAVE" in head[:16]:
        return "WAV"
    if head.startswith(b"ID3") or head[:2] == b"\xff\xfb" or head[:2] == b"\xff\xf3":
        return "MP3"
    if head.startswith(b"OggS"):
        return "OGG"
    if head.startswith(b"fLaC"):
        return "FLAC"
    return None


def _resolve_speaker_lenient(synth: StreamingSynthesize, speaker_ref: Optional[str]) -> Style:
    """Resolve a speaker reference *without* raising.

    Accepts a built-in voice name, a Voice Builder JSON path, or the stem of
    such a path. For anything else (unknown name, audio file, missing file,
    `None`), logs a note and falls back to the default voice. The point is to
    never reject the request just because the client passed something
    Supertonic can't honor verbatim — operators with XTTS clients sending
    arbitrary ``voice`` strings get a working response, not a 400.
    """
    builtin = set(synth.tts.voice_style_names)
    default = _default_voice(synth)

    if not speaker_ref:
        return synth.get_voice_style(default)
    if speaker_ref in builtin:
        return synth.get_voice_style(speaker_ref)

    p = Path(speaker_ref)
    if p.suffix.lower() == ".json" and p.exists():
        try:
            return synth.tts.get_voice_style_from_path(p)
        except Exception:
            logger.info("voice builder JSON %s failed to load, using %s", speaker_ref, default)
            return synth.get_voice_style(default)
    if p.stem in builtin:
        return synth.get_voice_style(p.stem)

    audio_kind = _detect_b64_audio(speaker_ref)
    if audio_kind:
        logger.info(
            "received base64 %s audio in voice/speaker_files field — Supertonic "
            "open-weight does NOT clone from raw audio at request time. Falling "
            "back to %r. To pick a voice, use one of the built-in names (M1..M5 "
            "/ F1..F5), a Voice Builder .json path, or call /v1/tts/conditioning "
            "first and pass its returned blobs to /v1/audio/speech.",
            audio_kind,
            default,
        )
        return synth.get_voice_style(default)

    logger.info(
        "unknown voice ref %s, falling back to %r", _short(speaker_ref), default
    )
    return synth.get_voice_style(default)


def _resolve_lang(lang: Optional[str]) -> Optional[str]:
    """`auto` / empty / None → let supertonic resolve to its multilingual fallback."""
    if lang is None:
        return None
    l = lang.strip().lower()
    if l in ("", "auto", "na"):
        return None
    return l


# ===================== app =====================

def _resp_400(message: str, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status_code=400,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("shutting down")


# Endpoints whose bodies are worth logging. GETs carry nothing and the
# metrics scrape would otherwise dominate the log.
_BODY_LOG_PATHS = ("/v1/audio/speech", "/v1/tts/conditioning")


# Path prefix this service is reached under when it sits behind a gateway
# that does NOT strip the prefix from the upstream request (the ELB in front
# of genai.waha.ai is one). Swagger UI needs it to build the "Try it out"
# URLs and to fetch the OpenAPI document; without it the docs page loads but
# every request from it 404s against the gateway.
#
# Empty by default — a direct hit on the pod, or a gateway that strips the
# prefix, needs no rewriting at all.
ROOT_PATH = os.getenv("SUPERTONIC_ROOT_PATH", os.getenv("ROOT_PATH", ""))

# Interactive docs. Set SUPERTONIC_DOCS=0 to serve the API with no docs
# surface at all (openapi.json included).
DOCS_ENABLED = _env_bool("SUPERTONIC_DOCS", True)


class DocsRootPathMiddleware:
    """Set ``scope["root_path"]`` only for the docs/openapi routes.

    Passing ``root_path=`` to ``FastAPI(...)`` applies it to every request,
    which changes how Starlette matches paths for the *whole* app — the TTS
    routes then only answer on ``{prefix}/v1/audio/speech``. That breaks every
    existing client the moment ROOT_PATH is set, and the legacy XTTS/pipecat
    contract (see tests/test_legacy_api_compat.py) is exactly what must not
    move.

    Scoping it to the four docs paths gets the Swagger URLs right while
    leaving the API routes matched on their real, unprefixed paths.
    """

    def __init__(self, asgi_app, root_path: str, docs_paths: set[str]):
        self.app = asgi_app
        self.root_path = root_path
        self.docs_paths = docs_paths

    async def __call__(self, scope, receive, send):
        if self.root_path and scope["type"] == "http" and scope["path"] in self.docs_paths:
            scope = {**scope, "root_path": self.root_path}
        await self.app(scope, receive, send)


def build_asgi_app() -> "DocsRootPathMiddleware | FastAPI":
    """The callable uvicorn actually serves.

    Separate from ``create_app()`` on purpose: the tests drive the bare
    FastAPI instance through TestClient and need routes, exception handlers
    and the lifespan to stay reachable as attributes. Only the process that
    binds a socket wants the docs wrapper.
    """
    app = create_app()
    if not ROOT_PATH:
        return app
    docs_paths = {
        p
        for p in (app.docs_url, app.redoc_url, app.openapi_url,
                  app.swagger_ui_oauth2_redirect_url)
        if p is not None
    }
    logger.info("serving docs under root_path=%s (paths: %s)",
                ROOT_PATH, ", ".join(sorted(docs_paths)) or "none")
    return DocsRootPathMiddleware(app, ROOT_PATH, docs_paths)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Supertonic Streaming TTS (OpenAI/XTTS compatible)",
        lifespan=lifespan,
        # Explicit rather than relying on FastAPI's defaults, so
        # SUPERTONIC_DOCS=0 has a single place to turn all of it off and
        # DocsRootPathMiddleware has a stable set of paths to match on.
        docs_url="/docs" if DOCS_ENABLED else None,
        redoc_url="/redoc" if DOCS_ENABLED else None,
        openapi_url="/openapi.json" if DOCS_ENABLED else None,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        """Name `messages` when `input` is what's missing.

        Clients reach this endpoint with a chat-completions body; the stock
        "body.input Field required" tells them nothing about the field this
        server actually wants, or that their shape is supported when it
        carries text."""
        errors = exc.errors()
        hint = None
        if any(e.get("type") == "missing" and "input" in e.get("loc", ()) for e in errors):
            hint = (
                "no text to synthesise: send `input` (OpenAI speech shape), or "
                "`messages` with non-empty content (chat shape, accepted for "
                "gateway probes)"
            )
        return JSONResponse(
            {"error": {
                "message": hint or "invalid request body",
                "type": "invalid_request_error",
                "code": "missing_input" if hint else "invalid_body",
                # The raw pydantic errors stay available for anyone debugging
                # a field this message doesn't cover.
                "errors": jsonable_encoder(errors),
            }},
            status_code=422,
        )

    @app.middleware("http")
    async def log_request_body(request: Request, call_next):
        """Log the body as it arrived on the wire, before validation.

        The handler's own line only reports fields SpeechRequest kept —
        it is ``extra="ignore"``, so anything the client invented is gone
        by then, and a body that fails validation never reaches the
        handler at all. Those are the two cases where you most need to
        see the payload, which is why this sits in middleware.

        The env vars are read per request on purpose: it makes the switch
        usable on a box that is already misbehaving, without a restart.
        """
        if request.method != "POST" or request.url.path not in _BODY_LOG_PATHS:
            return await call_next(request)
        if not _env_bool("SUPERTONIC_LOG_REQUEST_BODY", True):
            return await call_next(request)

        # Safe to drain here: BaseHTTPMiddleware hands us a _CachedRequest,
        # which replays a body() we consumed to the downstream app. Doing
        # the usual request._receive override on top of that is what breaks
        # it — the app then gets a second http.request where Starlette
        # expects only a disconnect.
        raw = await request.body()

        # Minted here rather than in the handler so the body line and the
        # handler's line share a req= token and can be joined.
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.req_id = req_id
        logger.info(
            "recv req=%s POST %s client=%s bytes=%d body=%s",
            req_id,
            request.url.path,
            request.client.host if request.client else "?",
            len(raw),
            _redact_body(raw, _env_int("SUPERTONIC_LOG_BODY_MAX", 2000)),
        )

        response = await call_next(request)
        if response.status_code >= 400:
            # Pair the body above with its rejection; a 422 raised by
            # pydantic is otherwise invisible server-side.
            logger.warning(
                "recv req=%s rejected with HTTP %d", req_id, response.status_code
            )
        return response

    @app.get("/health")
    def health():
        """Liveness — process is up. Returns 200 even during warm-up."""
        return {"status": "alive", "uptime_s": round(time.time() - State.started_at, 1)}

    @app.get("/ready")
    def ready():
        """Readiness — model loaded and warmup synthesis completed."""
        if not State.ready or State.synth is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {
            "status": "ok",
            "model": State.synth.model_name,
            "sample_rate": State.synth.sample_rate,
            "uptime_s": round(time.time() - State.started_at, 1),
        }

    @app.get("/metrics")
    def metrics():
        out = [
            "# HELP supertonic_requests_total Total /v1/audio/speech requests",
            "# TYPE supertonic_requests_total counter",
            f"supertonic_requests_total {State.req_total}",
            "# HELP supertonic_requests_failed Total errored requests",
            "# TYPE supertonic_requests_failed counter",
            f"supertonic_requests_failed {State.req_failed}",
            "# HELP supertonic_requests_inflight In-flight /v1/audio/speech requests",
            "# TYPE supertonic_requests_inflight gauge",
            f"supertonic_requests_inflight {State.req_inflight}",
            "# HELP supertonic_bytes_streamed_total PCM bytes streamed to clients",
            "# TYPE supertonic_bytes_streamed_total counter",
            f"supertonic_bytes_streamed_total {State.bytes_streamed}",
            "# HELP supertonic_uptime_seconds Process uptime in seconds",
            "# TYPE supertonic_uptime_seconds gauge",
            f"supertonic_uptime_seconds {time.time() - State.started_at:.3f}",
        ]
        return PlainTextResponse("\n".join(out) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/v1/tts/conditioning", response_model=ConditioningResponse)
    def conditioning(req: ConditioningRequest):
        if not State.ready or State.synth is None:
            raise HTTPException(status_code=503, detail="server not ready")
        ref = (req.speaker_files[0] if req.speaker_files else None) or _coerce_voice(req.voice)
        style = _resolve_speaker_lenient(State.synth, ref)
        return ConditioningResponse(
            gpt_cond_latent_b64=_encode_array_b64(style.ttl),
            speaker_embeddings_b64=_encode_array_b64(style.dp),
            sample_rate=State.synth.sample_rate,
            model=State.synth.model_name,
        )

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, request: Request):
        if not State.ready or State.synth is None:
            raise HTTPException(status_code=503, detail="server not ready")
        try:
            fmt = resolve_format(req.response_format)
        except UnsupportedAudioFormat as exc:
            return _resp_400(str(exc), "unsupported_response_format")
        if req.stream_format and req.stream_format.strip().lower() not in ("audio", ""):
            return _resp_400(
                f"unsupported stream_format {req.stream_format!r}; this server "
                "emits audio bytes only (use response_format='pcm' for chunked "
                "streaming)",
                "unsupported_stream_format",
            )

        # ---- per-request structured log line ----
        # Operators need to see what clients actually send (model name,
        # voice ref, conditioning blob sizes, sampling fields the client
        # set). Blob *contents* would flood the log — XTTS clients pass
        # 60 KB conditioning latents — so we log lengths only.
        # Unique per request — id(req) collides after GC reuses memory,
        # which made multiple unrelated requests share log lines.
        req_id = getattr(request.state, "req_id", None) or (
            request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        )
        client = request.client.host if request.client else "?"
        peer_ua = request.headers.get("user-agent", "?")
        voice = _coerce_voice(req.voice)
        voice_kind = "audio:" + _detect_b64_audio(voice) if _detect_b64_audio(voice) else "name"
        logger.info(
            "speech req=%s client=%s model=%r voice=%s voice_kind=%s lang=%r "
            "len(text)=%d text=%s text_from=%s gpt_cond=%dB spk_emb=%dB "
            "stream=%s response_format=%r chunk_size=%s total_steps=%s "
            "ua=%s",
            req_id,
            client,
            req.model,
            _short(voice),
            voice_kind,
            req.language,
            len(req.input or ""),
            # Truncated input preview. 200 chars is enough to identify
            # the request when debugging without flooding INFO logs;
            # the full text is still recoverable from the client-side
            # logs or by reproducing with the same payload.
            _short(req.input, 200),
            # "messages" means a chat-completions body was translated — most
            # likely a gateway liveness probe, not a real caller.
            req.input_from,
            len(req.gpt_cond_latent_b64 or ""),
            len(req.speaker_embeddings_b64 or ""),
            req.stream,
            fmt.name,
            req.chunk_size,
            req.total_steps,
            # The XTTS sampling knobs used to be echoed here. They are no
            # longer model fields (they never did anything), and the raw
            # request-body log already shows whatever the client sent.
            _short(peer_ua, 64),
        )

        # ---- resolve speaker (max-lenient) ----
        # 1) Try the conditioning blobs. Only blobs we produced (numpy .npy)
        #    are usable; XTTS torch-save blobs are silently rejected and we
        #    fall through to the voice field.
        ttl = _try_decode_b64_npy(req.gpt_cond_latent_b64)
        dp = _try_decode_b64_npy(req.speaker_embeddings_b64)
        if ttl is not None and dp is not None:
            try:
                style = Style(ttl, dp)
            except Exception:
                logger.info(
                    "req=%s Style() rejected the decoded arrays; falling back to voice",
                    req_id,
                )
                style = _resolve_speaker_lenient(State.synth, voice)
        else:
            if req.gpt_cond_latent_b64 or req.speaker_embeddings_b64:
                logger.info(
                    "req=%s conditioning blobs not in supertonic format "
                    "(likely torch/XTTS); ignoring and resolving voice=%s",
                    req_id,
                    _short(voice),
                )
            style = _resolve_speaker_lenient(State.synth, voice)

        lang = _resolve_lang(req.language)

        # Language identification + refusal dispatch.
        #
        # Anything outside supertonic-3's 31 official languages gets
        # a single fixed English apology. Chinese is included in this
        # bucket: the model accepts Han characters but pronounces
        # them with Japanese kanji readings, useless to a Chinese
        # listener. An English fallback is the most universally
        # understandable response across all out-of-scope callers.
        #
        # Detection runs via lingua-py; if it's offline or has no
        # confident answer (very short text), the request passes
        # through unchanged.
        detected = detect_language(req.input, req.language)
        text_to_synth = req.input
        if detected is not None and not is_supertonic_lang(detected):
            text_to_synth = "This language is not supported yet, coming soon."
            logger.info(
                "req=%s detected=%s (lang_hint=%r) outside supertonic-31 — substituting English refusal",
                req_id, detected, req.language,
            )
        # Clamp client-supplied numerics into safe ranges instead of 422'ing
        # on edge cases. The request always succeeds for any parseable input.
        max_chunk_length = req.chunk_size if (req.chunk_size and req.chunk_size > 0) else None
        total_steps = max(1, min(16, req.total_steps or 6))
        speed = max(0.5, min(2.0, req.speed or 1.0))

        # Resample on the way out. supertonic-3 synthesises at 44100 Hz;
        # XTTS / pipecat-style clients expect 24000 Hz. The output rate is
        # picked in priority: per-request `sample_rate` > env var
        # SUPERTONIC_OUTPUT_SAMPLE_RATE > 24000 (default).
        requested_sr = (
            int(req.sample_rate) if req.sample_rate
            else _env_int("SUPERTONIC_OUTPUT_SAMPLE_RATE", 24000)
        )
        # Opus only encodes at 8/12/16/24/48 kHz; snap rather than fail, and
        # resample straight to the snapped rate so there's no second pass.
        out_sr = negotiate_sample_rate(fmt, requested_sr)
        if out_sr != requested_sr:
            logger.info(
                "req=%s sample_rate %d not encodable as %s — using %d",
                req_id, requested_sr, fmt.name, out_sr,
            )
        resampler = PcmResampler(in_rate=State.synth.sample_rate, out_rate=out_sr)

        with State._lock:
            State.req_total += 1
            State.req_inflight += 1

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)

        def _producer():
            try:
                for pcm in State.synth.stream_pcm(
                    text=text_to_synth,
                    voice_style=style,
                    lang=lang,
                    total_steps=total_steps,
                    speed=speed,
                    max_chunk_length=max_chunk_length,
                    req_id=req_id,
                ):
                    pcm_out = resampler.process(pcm)
                    asyncio.run_coroutine_threadsafe(queue.put(pcm_out), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
            except Exception as exc:
                logger.exception("synthesis failed")
                asyncio.run_coroutine_threadsafe(
                    queue.put(("__error__", repr(exc))), loop
                ).result()

        # Shared bookkeeping for both response paths. A dict (not locals)
        # because the async generator below and the finaliser live in
        # different scopes.
        t_start = time.perf_counter()
        stats: dict = {"sent": 0, "t_first_byte": None, "err": None}

        async def _pcm_chunks():
            """Drain the producer thread, yielding resampled PCM chunks.

            Records the first failure in ``stats["err"]`` and stops; it
            never raises, so each caller decides how to surface the error
            (mid-stream marker vs. clean 500)."""
            # Per-chunk inactivity timeout — if the producer doesn't put
            # anything on the queue for this long, treat the request as
            # hung instead of streaming bytes=0 silently. 120s covers a
            # cold TRT engine build but exposes inference-thread deadlocks.
            chunk_timeout_s = _env_int("SUPERTONIC_CHUNK_TIMEOUT_S", 120)
            loop.run_in_executor(None, _producer)
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=chunk_timeout_s)
                except asyncio.TimeoutError:
                    stats["err"] = f"chunk timeout after {chunk_timeout_s}s — inference hung"
                    logger.error("req=%s %s", req_id, stats["err"])
                    return
                if item is None:
                    return
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    stats["err"] = item[1]
                    return
                if stats["t_first_byte"] is None:
                    stats["t_first_byte"] = time.perf_counter()
                stats["sent"] += len(item)
                yield item

        def _finish(encoded_bytes: Optional[int] = None) -> None:
            """Release the in-flight slot and emit the per-request summary.

            ``sent``/``audio_s`` always describe the PCM produced, so the
            numbers stay comparable across formats; ``bytes`` is what the
            client actually received."""
            with State._lock:
                State.req_inflight -= 1
                State.bytes_streamed += stats["sent"]
                if stats["err"]:
                    State.req_failed += 1
            t_first_byte = stats["t_first_byte"]
            ttfb_ms = (t_first_byte - t_start) * 1000 if t_first_byte else None
            total_ms = (time.perf_counter() - t_start) * 1000
            audio_s = stats["sent"] / (2 * out_sr) if stats["sent"] else 0
            logger.info(
                "speech req=%s done fmt=%s bytes=%d pcm_bytes=%d sr_out=%d "
                "audio_s=%.2f ttfb_ms=%s total_ms=%.1f err=%s",
                req_id,
                fmt.name,
                stats["sent"] if encoded_bytes is None else encoded_bytes,
                stats["sent"],
                out_sr,
                audio_s,
                f"{ttfb_ms:.1f}" if ttfb_ms is not None else "-",
                total_ms,
                stats["err"],
            )

        headers = {
            "X-Sample-Rate": str(out_sr),
            "X-Model": State.synth.model_name,
            "X-Audio-Format": fmt.name,
            "Cache-Control": "no-store",
        }

        # ---- raw PCM: stream it, exactly as before ----
        # Mid-stream failures still append a literal "[error] ..." marker
        # to the body; the response has already started by then, so a
        # status code is no longer available. Long-standing behaviour that
        # existing clients parse — don't change it here.
        if fmt.streaming:
            async def _streamer():
                try:
                    async for chunk in _pcm_chunks():
                        yield chunk
                    if stats["err"]:
                        yield f"\n[error] {stats['err']}\n".encode()
                finally:
                    _finish()

            return StreamingResponse(
                _streamer(), media_type=fmt.media_type, headers=headers
            )

        # ---- encoded container: buffer, encode once, send whole ----
        # Nothing has been written to the wire yet, so a failure here can
        # still be a clean JSON 500 instead of a truncated audio file.
        pcm = bytearray()
        try:
            async for chunk in _pcm_chunks():
                pcm.extend(chunk)
            if stats["err"]:
                _finish(encoded_bytes=0)
                return JSONResponse(
                    {"error": {"message": stats["err"], "type": "server_error",
                               "code": "synthesis_failed"}},
                    status_code=500,
                )
            body = await asyncio.get_running_loop().run_in_executor(
                None, encode_pcm, fmt, bytes(pcm), out_sr
            )
        except Exception as exc:  # encoder failure — still no bytes sent
            stats["err"] = repr(exc)
            logger.exception("req=%s encoding to %s failed", req_id, fmt.name)
            _finish(encoded_bytes=0)
            return JSONResponse(
                {"error": {"message": f"failed to encode audio as {fmt.name}: {exc}",
                           "type": "server_error", "code": "encoding_failed"}},
                status_code=500,
            )

        _finish(encoded_bytes=len(body))
        return Response(content=body, media_type=fmt.media_type, headers=headers)

    return app


# ===================== CLI / boot =====================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.getenv("SUPERTONIC_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=_env_int("SUPERTONIC_PORT", 8000))
    p.add_argument("--model-name", default=os.getenv("SUPERTONIC_MODEL_NAME", "supertonic-3"))
    p.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("SUPERTONIC_MODEL_PATH"),
        help="Path to the model directory. If unset, supertonic uses its cache dir. "
             "Auto-download is OFF by default (set SUPERTONIC_AUTO_DOWNLOAD=1 for dev).",
    )
    p.add_argument("--device-id", type=int, default=_env_int("SUPERTONIC_DEVICE_ID", 0))
    p.add_argument("--cpu", action="store_true", default=_env_bool("SUPERTONIC_CPU", False))
    p.add_argument("--trt", action="store_true", default=_env_bool("SUPERTONIC_TRT", True))
    p.add_argument("--no-trt", dest="trt", action="store_false")
    p.add_argument("--trt-cache-dir", type=str,
                   default=os.getenv("SUPERTONIC_TRT_CACHE", "/var/cache/supertonic/trt"))
    p.add_argument("--no-trt-fp16", action="store_true",
                   default=not _env_bool("SUPERTONIC_TRT_FP16", True))
    # Default ON: concurrent supertonic.TTS.__call__ has been observed to
    # silently deadlock in prod (GPU goes 0% util, the second request never
    # logs "inference done"). Per-process mutex is the cheap robust fix
    # until the root cause in supertonic is identified. To get the
    # throughput back, set SUPERTONIC_SERIALIZE=0 and run multiple replicas
    # behind the k8s Service.
    p.add_argument("--serialize", action="store_true",
                   default=_env_bool("SUPERTONIC_SERIALIZE", True))
    p.add_argument("--warmup", action="store_true",
                   default=_env_bool("SUPERTONIC_WARMUP", True))
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--log-level", default=os.getenv("SUPERTONIC_LOG_LEVEL", "INFO"))
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info(
        "loading model=%s path=%s device=%s trt=%s",
        args.model_name,
        args.model_path or "(cache)",
        "cpu" if args.cpu else f"cuda:{args.device_id}",
        args.trt,
    )
    # Dump GPU diagnostics BEFORE model load. If the model load itself
    # hangs (long TRT engine build the first time), the operator at
    # least sees in the log that nvidia-smi works / providers list is OK
    # and can confirm "yes the pod has a GPU, the hang is in inference".
    from streaming_synth import _log_startup_diagnostics

    _log_startup_diagnostics()

    # Operators can enable ORT verbose logging to watch the TRT engine
    # build progress at runtime: SUPERTONIC_ORT_VERBOSE=1
    if _env_bool("SUPERTONIC_ORT_VERBOSE", False):
        import onnxruntime as _ort

        _ort.set_default_logger_severity(0)  # 0=VERBOSE, 1=INFO, 2=WARNING
        logger.info("ORT verbose logging enabled (SUPERTONIC_ORT_VERBOSE=1)")

    synth = StreamingSynthesize(
        model_path=Path(args.model_path) if args.model_path else None,
        model_name=args.model_name,
        use_gpu=not args.cpu,
        device_id=args.device_id,
        serialize=args.serialize,
        use_trt=args.trt and not args.cpu,
        trt_cache_dir=args.trt_cache_dir,
        trt_fp16=not args.no_trt_fp16,
    )
    State.synth = synth
    State.model_name = args.model_name

    # Verdict thresholds derived from the host pressure-test:
    #   TRT FP16 on A100  : ~0.05-0.1s/chunk after engines cached
    #   plain CUDA EP     : ~1.5-2.0s/chunk
    #   CPU fallback      : 5-10s/chunk and pegs one core at 100%
    # On a fresh pod TRT pays a 5-15 min engine build for the first
    # input shape it sees, so we run the diagnostic warmup TWICE: the
    # first call builds engines (slow, expected), the second is the
    # true steady-state measurement we judge against.
    if args.warmup:
        from streaming_synth import _nvidia_smi_snapshot, _start_gpu_heartbeat

        _start_gpu_heartbeat(interval_s=30)
        logger.info("=" * 60)
        logger.info(
            "diagnostic warmup — verifying execution actually runs on GPU + TRT"
        )
        logger.info("GPU pre-warmup:  %s", _nvidia_smi_snapshot() or "<empty>")

        warmup_text = (
            "Warm up the GPU so the first real request is not penalised."
        )
        t0 = time.perf_counter()
        nb1 = synth.speak_streaming(warmup_text, voice="M1", lang="en")
        dt1 = time.perf_counter() - t0
        logger.info(
            "warmup #1 (cold, may include TRT engine build): %d bytes in %.2fs",
            nb1, dt1,
        )
        logger.info("GPU mid-warmup:  %s", _nvidia_smi_snapshot() or "<empty>")

        t0 = time.perf_counter()
        nb2 = synth.speak_streaming(warmup_text, voice="M1", lang="en")
        dt2 = time.perf_counter() - t0
        logger.info("warmup #2 (steady-state): %d bytes in %.2fs", nb2, dt2)
        logger.info("GPU post-warmup: %s", _nvidia_smi_snapshot() or "<empty>")

        # Steady-state verdict on dt2 (cold engines are excluded by warmup #1).
        per_chunk_s = dt2  # the warmup text is ~one chunk
        if per_chunk_s < 0.5:
            verdict = "TRT (GPU + FP16) — fast path confirmed"
        elif per_chunk_s < 3.0:
            verdict = (
                "GPU but no TRT speedup — TRT likely fell back to CUDA EP. "
                "Check: provider list, libnvinfer.so.10 loaded, ORT TRT cache writable."
            )
        else:
            verdict = (
                "CPU fallback suspected — synthesis is too slow for GPU. "
                "Check: nvidia.com/gpu in pod spec, NVIDIA device plugin on node, "
                "CUDA_VISIBLE_DEVICES, /dev/nvidia* devices visible."
            )
        logger.info("WARMUP VERDICT: %s", verdict)
        logger.info("=" * 60)

    # Eagerly build the lingua language detector so the 2-3 s model
    # decompression cost is paid here (before /ready=True), not on
    # the first inbound request. Failure here is non-fatal — the
    # server still serves, language detection just becomes a no-op.
    logger.info("warming up lingua language detector (~2-3 s)")
    t_lid = time.perf_counter()
    lid_ok = warmup_detector()
    logger.info(
        "lingua warmup %s in %.2f s",
        "OK" if lid_ok else "FAILED (LID disabled)",
        time.perf_counter() - t_lid,
    )

    State.ready = True
    app = build_asgi_app()
    if DOCS_ENABLED:
        logger.info("interactive docs at http://%s:%d%s/docs",
                    args.host, args.port, ROOT_PATH)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
