"""Chunk-streaming wrapper around supertonic.TTS.

The upstream `supertonic` PyPI package only exposes an offline
`TTS.synthesize` that returns a single concatenated waveform. To get a
meaningful Time-to-First-Byte for an HTTP streaming endpoint we synthesise
the long text chunk-by-chunk (Supertonic already splits long text on
sentence boundaries internally — we hoist that chunker here and yield each
chunk's PCM as it becomes available).

Production notes:
    * GPU is the only supported deployment target. CPU mode is exposed for
      local debugging but its TTFB is ~4-5× the GPU number — do not ship.
    * Use the TensorRT execution provider when possible (about 47× lower
      TTFB at low concurrency, ~5× higher peak rps on A100). Engine cache
      *must* persist across restarts — the first-time build is 5-15 min.
"""

from __future__ import annotations

import audioop
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import numpy as np
import onnxruntime as ort

import supertonic
from supertonic import TTS, Style
from supertonic.config import (
    DEFAULT_MAX_CHUNK_LENGTH,
    DEFAULT_MAX_CHUNK_LENGTH_KO,
    UNKNOWN_LANGUAGE,
)
from supertonic.utils import chunk_text

from text_preprocess import preprocess_text

logger = logging.getLogger(__name__)


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] waveform → signed 16-bit little-endian PCM bytes."""
    w = np.clip(wav.reshape(-1), -1.0, 1.0)
    return (w * 32767.0).astype("<i2").tobytes()


def _set_trt_shape_profiles() -> None:
    """Export ORT_TENSORRT_PROFILE_{MIN,OPT,MAX}_SHAPES env vars covering
    every input across supertonic-3's four ONNX files.

    ORT TRT EP reads these globally — a single profile spec applies to
    every InferenceSession in the process. We list the union of all
    inputs; ORT ignores names that don't appear in a given graph.

    Operators can override individual bounds by setting the env vars
    BEFORE process start; this function only sets them if they're not
    already set (setdefault).
    """
    # Per-dim bounds. MIN/MAX define the *supported* shape range; OPT is
    # the shape TRT auto-tunes its kernels around. Any actual input
    # inside [MIN, MAX] runs on the same engine, but speed is best near
    # OPT and degrades gracefully as you move away.
    #
    # The defaults below are tuned for server-side sentence-packed
    # chunking — i.e. clients hitting /v1/audio/speech with long
    # inputs that get split by supertonic.utils.chunk_text at the
    # default max_chunk_length=300. That produces chunks averaging
    # 200-280 chars, with corresponding latent_length around 250-400
    # frames (audio runs ~50 latent frames per second).
    #   text_length:   typical 200-280, OPT=256 lands in the middle
    #   latent_length: typical 250-400 for 5-8 s of audio, OPT=512
    #                  covers a 10 s sentence comfortably.
    # If your workload is *streaming* (pipecat / XTTS-style with
    # chunk_size=20, i.e. tiny ~20-char chunks), drop OPT via env:
    #   SUPERTONIC_TRT_TEXT_OPT=32 SUPERTONIC_TRT_LATENT_OPT=128
    # Bump TEXT_MAX/LAT_MAX only if a single chunk really needs more
    # (rare — most clients split text long before hitting these).
    TEXT_MIN, TEXT_OPT, TEXT_MAX = 1, 256, 512
    LAT_MIN, LAT_OPT, LAT_MAX = 1, 512, 4096
    # Operator overrides — tune to the real workload without re-coding.
    TEXT_MIN = int(os.environ.get("SUPERTONIC_TRT_TEXT_MIN", TEXT_MIN))
    TEXT_OPT = int(os.environ.get("SUPERTONIC_TRT_TEXT_OPT", TEXT_OPT))
    TEXT_MAX = int(os.environ.get("SUPERTONIC_TRT_TEXT_MAX", TEXT_MAX))
    LAT_MIN = int(os.environ.get("SUPERTONIC_TRT_LATENT_MIN", LAT_MIN))
    LAT_OPT = int(os.environ.get("SUPERTONIC_TRT_LATENT_OPT", LAT_OPT))
    LAT_MAX = int(os.environ.get("SUPERTONIC_TRT_LATENT_MAX", LAT_MAX))
    logger.info(
        "TRT profile bounds: text_length=[%d,%d,%d] latent_length=[%d,%d,%d]",
        TEXT_MIN, TEXT_OPT, TEXT_MAX, LAT_MIN, LAT_OPT, LAT_MAX,
    )

    def spec_for(level: str) -> str:
        # level ∈ {"min", "opt", "max"}
        t = {"min": TEXT_MIN, "opt": TEXT_OPT, "max": TEXT_MAX}[level]
        l = {"min": LAT_MIN, "opt": LAT_OPT, "max": LAT_MAX}[level]
        return ",".join([
            f"text_ids:1x{t}",
            f"style_dp:1x8x16",
            f"style_ttl:1x50x256",
            f"text_mask:1x1x{t}",
            f"noisy_latent:1x144x{l}",
            f"text_emb:1x256x{t}",
            f"latent_mask:1x1x{l}",
            f"current_step:1",
            f"total_step:1",
            f"latent:1x144x{l}",
        ])

    os.environ.setdefault("ORT_TENSORRT_PROFILE_MIN_SHAPES", spec_for("min"))
    os.environ.setdefault("ORT_TENSORRT_PROFILE_OPT_SHAPES", spec_for("opt"))
    os.environ.setdefault("ORT_TENSORRT_PROFILE_MAX_SHAPES", spec_for("max"))
    logger.info("TRT shape profiles set (one engine covers text_length [1, 512], latent_length [1, 4096])")


class PcmResampler:
    """Stateful PCM 16-bit mono resampler using audioop.ratecv (stdlib, no
    extra deps).

    Supertonic synthesises at 44100 Hz. Most XTTS-style clients (pipecat,
    Coqui XTTS streaming server, ...) expect 24000 Hz on the wire and do
    a second resample on the client side to whatever their downstream
    pipeline needs. To make this server a drop-in replacement, we
    resample 44100 -> 24000 (or any other operator-chosen rate) inside
    the server and stream the result.

    State is carried across chunks so sentence-boundary samples don't
    glitch. One instance per request.
    """

    def __init__(self, in_rate: int, out_rate: int):
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._state = None

    @property
    def passthrough(self) -> bool:
        return self.in_rate == self.out_rate

    def process(self, pcm: bytes) -> bytes:
        if self.passthrough or not pcm:
            return pcm
        # ratecv signature: (fragment, width, nchannels, inrate, outrate, state)
        out, self._state = audioop.ratecv(
            pcm, 2, 1, self.in_rate, self.out_rate, self._state
        )
        return out


def _nvidia_smi_snapshot() -> str:
    """One-line GPU status from nvidia-smi. Empty string if nvidia-smi is
    unavailable (e.g., CPU-only pod). Cheap enough to log every 30s."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return f"nvidia-smi rc={out.returncode}: {out.stderr.strip()[:100]}"
        return " | ".join(line.strip() for line in out.stdout.splitlines() if line.strip())
    except FileNotFoundError:
        return "nvidia-smi NOT FOUND — pod likely has no GPU access"
    except Exception as exc:
        return f"nvidia-smi error: {exc!r}"


def _log_startup_diagnostics() -> None:
    """Dump enough info at server start to tell whether we're on GPU at all.

    If a junior engineer sees the pod hanging, the very first thing they
    should look for in the log is this block. The most common failure
    modes and what they look like here:

      * `nvidia-smi NOT FOUND` → no NVIDIA container runtime / no GPU bits
        injected. Pod spec is wrong (missing `nvidia.com/gpu: 1` in resources)
        OR the node has no NVIDIA device plugin OR runtimeClass is wrong.
      * `/dev/nvidia* devices visible: <none>` → same root cause as above.
      * `NVIDIA_VISIBLE_DEVICES=void` or empty → device plugin didn't bind
        any GPU to this pod (cluster out of GPUs, or wrong scheduler).
      * `CUDA_VISIBLE_DEVICES=""` (empty) → CUDA is explicitly hidden.
        Some chart templates set this; either drop it or set to "0".
      * ORT providers list does NOT include TensorrtExecutionProvider →
        wrong onnxruntime wheel (CPU instead of -gpu) OR libnvinfer.so.10
        not loadable at import time.
    """
    logger.info("=" * 60)
    logger.info("startup diagnostics")
    logger.info("-" * 60)
    # The env vars that actually matter for GPU access, in failure-mode
    # priority order. Empty/missing values are highlighted.
    for k in (
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ):
        v = os.environ.get(k)
        logger.info("  %s=%s", k, "<UNSET>" if v is None else repr(v))
    logger.info("ORT version=%s", ort.__version__)
    logger.info("ORT available providers=%s", ort.get_available_providers())
    logger.info("GPU (nvidia-smi): %s", _nvidia_smi_snapshot() or "<empty>")
    devs = sorted(Path("/dev").glob("nvidia*"))
    logger.info("/dev/nvidia* devices visible: %s", [d.name for d in devs] or "<none>")
    logger.info("=" * 60)


def _start_gpu_heartbeat(interval_s: int = 30) -> None:
    """Background thread that logs GPU status every interval_s seconds.

    Crucial when debugging "the pod is stuck" — if GPU utilisation is
    > 0% the workload is busy on GPU (likely a long TRT engine build for
    a new input shape, which can take 5-15 minutes on first sight). If
    it's 0%, the hang is in our Python code or the kernel scheduler,
    not in actual GPU work."""

    def _tick() -> None:
        while True:
            time.sleep(interval_s)
            try:
                logger.info("gpu heartbeat: %s", _nvidia_smi_snapshot() or "<empty>")
            except Exception:
                logger.exception("gpu heartbeat failed")

    t = threading.Thread(target=_tick, name="gpu-heartbeat", daemon=True)
    t.start()


def _ensure_shape_inferred(orig_model_path: Path, cache_root: Path) -> Path:
    """Mirror the supertonic model tree into a writable cache directory,
    running ONNX shape inference on every .onnx file along the way.

    Why: ORT's TensorRT EP needs intermediate tensor shapes to build
    subgraphs. The released supertonic-3 ONNX files ship without those
    shapes, so a fresh ``InferenceSession`` crashes with
    ``TensorRT input: ... has no shape specified``. Doing the inference
    in-process at startup avoids any out-of-band tooling step and
    survives the model PVC being mounted read-only.

    The mirror is a *full* copy of the model directory — onnx/, the
    voice_styles/ directory, the json sidecar files, anything else.
    Without that, supertonic.loader.list_available_voice_style_paths
    raises FileNotFoundError on the inferred dir.

    Output lives at ``cache_root/inferred/<key>/``. The key is derived
    from every .onnx file's name+size, so model upgrades invalidate the
    cache automatically. A ``.done`` sentinel marks completion so we
    don't re-do the work on every pod start. Override the cache root
    with $SUPERTONIC_TRT_CACHE — set that to /app/models/cache if you
    prefer co-locating cache artefacts with the model PVC (and that
    PVC is mounted RW).
    """
    onnx_dir = orig_model_path / "onnx"
    if not onnx_dir.is_dir():
        # supertonic will raise later with a clearer message
        return orig_model_path

    # Cache key includes a tag for the shape-inference settings so any
    # change to those settings (e.g., toggling auto_merge) doesn't read
    # back stale inferred files.
    _INFER_TAG = "v2-am-false"  # bump when changing inference settings

    h = hashlib.sha256()
    h.update(_INFER_TAG.encode())
    h.update(b"\n")
    for f in sorted(onnx_dir.iterdir()):
        h.update(f.name.encode())
        h.update(b":")
        h.update(str(f.stat().st_size).encode())
        h.update(b"\n")
    key = h.hexdigest()[:16]

    out = cache_root / "inferred" / key
    sentinel = out / ".done"
    if sentinel.is_file():
        logger.info("using cached shape-inferred model at %s", out)
        return out

    logger.info(
        "mirroring %s -> %s with shape inference on .onnx files (one-time, ~10-30s)",
        orig_model_path,
        out,
    )

    import onnx  # local — heavy import deferred to first call
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    def _copy_or_infer(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
        if src.endswith(".onnx"):
            name = Path(src).name
            logger.info("  inferring shapes on %s", name)
            try:
                model = onnx.load(src)
                # auto_merge=False is critical: with auto_merge=True
                # symbolic dimensions get aggressively unified into static
                # values, which then cause TRT to compile engines with
                # hardcoded shapes and fail at runtime when the inputs
                # vary in those dims (e.g.,
                #   "Static dimension mismatch ... Set [1,1,137,32].
                #    Expected [1,2,137,32]").
                # Leaving the dims symbolic lets ORT's TRT EP build per-
                # shape engines (cached separately) instead of locking
                # one shape into the compiled engine.
                inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=False)
            except Exception:
                logger.exception("  shape inference raised on %s; copying as-is", name)
                inferred = None
            if inferred is None:
                shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
            else:
                onnx.save(inferred, dst)
            return dst
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)

    # In case a previous interrupted run left a partial directory.
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        str(orig_model_path),
        str(out),
        copy_function=_copy_or_infer,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    sentinel.touch()
    logger.info("shape-inferred model ready at %s", out)
    return out


class StreamingSynthesize:
    """Streaming wrapper around `supertonic.TTS` with optional TensorRT EP.

    Concurrency model: a single TTS instance holds one set of ONNX
    sessions. ONNX Runtime sessions are documented as thread-safe for
    concurrent `run()` calls (the kernels release the GIL), so by default
    we hand out reads of the underlying model with no application-level
    lock. Set ``serialize=True`` if you observe corruption under load.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        model_name: str = "supertonic-3",
        use_gpu: bool = True,
        device_id: int = 0,
        serialize: bool = False,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
        use_trt: bool = False,
        trt_cache_dir: Optional[Union[Path, str]] = None,
        trt_fp16: bool = True,
    ):
        # supertonic.loader filters providers by string-name membership in
        # ort.get_available_providers(), which strips out tuple-form provider
        # configs. We pass plain strings and tune TRT via env vars.
        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        # First attempt: honour the operator's TRT choice. If TRT EP
        # refuses to load the model — supertonic-3's ONNX graphs hit a
        # known TRT static-shape limitation that requires per-input shape
        # profiles to work around — fall back to CUDA EP. We log loudly
        # so the operator can decide whether to set
        # ORT_TENSORRT_PROFILE_{MIN,OPT,MAX}_SHAPES manually and try again.
        attempted_trt = bool(use_gpu and use_trt)
        self.tts, providers, model_path = self._load_tts(
            model_path=model_path,
            model_name=model_name,
            use_gpu=use_gpu,
            use_trt=attempted_trt,
            trt_cache_dir=trt_cache_dir,
            trt_fp16=trt_fp16,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

        self.model_name = model_name
        self.sample_rate = self.tts.sample_rate
        self._lock = threading.Lock() if serialize else None

        sess_providers = self.tts.model.vocoder_ort.get_providers()
        logger.info(
            "StreamingSynthesize ready: providers=%s sample_rate=%d",
            sess_providers,
            self.sample_rate,
        )

    def _load_tts(
        self,
        *,
        model_path,
        model_name,
        use_gpu,
        use_trt,
        trt_cache_dir,
        trt_fp16,
        intra_op_num_threads,
        inter_op_num_threads,
    ):
        """Try TRT first if asked, fall back to CUDA EP on TRT failure.

        Returns (tts, providers_used, effective_model_path). The
        effective_model_path may differ from the caller's input when the
        TRT path was taken — it's then the shape-inferred mirror dir.
        """

        def _configure(providers):
            supertonic.config.DEFAULT_ONNX_PROVIDERS = providers
            try:
                from supertonic import loader as _loader

                _loader.DEFAULT_ONNX_PROVIDERS = providers
            except Exception:
                pass

        # --- TRT path ---
        # supertonic-3 needs TRT 10.13+ (build-time smoke check enforces
        # the image is on a base that ships that or newer). With 10.13+
        # the ONNX graphs load natively — no offline shape inference, no
        # shape-profile env vars. We just enable the EP, point ORT at a
        # persistent engine cache, and hand the original model_path to
        # supertonic.TTS verbatim.
        if use_gpu and use_trt:
            if trt_cache_dir is None:
                trt_cache_dir = Path("/var/cache/supertonic/trt")
            trt_cache_dir = Path(trt_cache_dir)
            trt_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_CACHE_PATH"] = str(trt_cache_dir)
            os.environ["ORT_TENSORRT_TIMING_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_TIMING_CACHE_PATH"] = str(trt_cache_dir)
            if trt_fp16:
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
            os.environ.setdefault(
                "ORT_TENSORRT_MAX_WORKSPACE_SIZE", str(4 * 1024 * 1024 * 1024)
            )
            # ----- TRT dynamic shape profiles -----
            # Without these, ORT's TRT EP builds a separate engine for
            # every distinct value of text_length / latent_length the
            # session sees at run time. That's why we observed 120s
            # hangs whenever a new text length showed up — the warmup
            # only covered the lengths in the warmup texts, and any
            # other length triggered a fresh 1-2 min engine compile.
            #
            # Setting MIN/OPT/MAX shape profiles tells TRT to build ONE
            # engine that covers the whole text_length range [1, 512]
            # and the whole latent_length range [1, 4096] (audio frames,
            # ~22 s @ 44.1 kHz native). After this lands, the engine
            # cache fills once at first boot and never rebuilds.
            #
            # Input names + dims probed from supertonic-3's ONNX files:
            #   duration_predictor.onnx:
            #     text_ids       [batch, text_length]
            #     style_dp       [batch, 8, 16]
            #     text_mask      [batch, 1, text_length]
            #   text_encoder.onnx:
            #     text_ids       [batch, text_length]
            #     style_ttl      [batch, 50, 256]
            #     text_mask      [batch, 1, text_length]
            #   vector_estimator.onnx:
            #     noisy_latent   [batch, 144, latent_length]
            #     text_emb       [batch, 256, text_length]
            #     style_ttl      [batch, 50, 256]
            #     latent_mask    [batch, 1, latent_length]
            #     text_mask      [batch, 1, text_length]
            #     current_step   [batch]
            #     total_step     [batch]
            #   vocoder.onnx:
            #     latent         [batch, 144, latent_length]
            _set_trt_shape_profiles()
            logger.info(
                "TensorRT EP enabled (cache=%s, fp16=%s)", trt_cache_dir, trt_fp16
            )
            trt_providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            _configure(trt_providers)
            try:
                tts = TTS(
                    model=model_name,
                    model_dir=model_path,
                    auto_download=os.getenv("SUPERTONIC_AUTO_DOWNLOAD", "0") == "1",
                    intra_op_num_threads=intra_op_num_threads,
                    inter_op_num_threads=inter_op_num_threads,
                )
                return tts, trt_providers, model_path
            except Exception:
                logger.exception(
                    "TensorRT EP failed to initialise the supertonic model. "
                    "Falling back to CUDA EP — synthesis will still run on "
                    "GPU, just without TRT engine optimisation. This usually "
                    "means the image was built on a base older than TRT 10.13 "
                    "(check the Dockerfile build smoke check); bump TENSORRT_BASE."
                )

        # --- CUDA / CPU fallback path ---
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        _configure(providers)
        tts = TTS(
            model=model_name,
            model_dir=model_path,
            auto_download=os.getenv("SUPERTONIC_AUTO_DOWNLOAD", "0") == "1",
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )
        return tts, providers, model_path

    def get_voice_style(self, voice_name: str) -> Style:
        return self.tts.get_voice_style(voice_name)

    def _resolve_lang(self, lang: Optional[str]) -> Optional[str]:
        if not self.tts.is_multilingual:
            return None
        return lang or UNKNOWN_LANGUAGE

    def stream_pcm(
        self,
        text: str,
        voice_style: Style,
        lang: Optional[str] = None,
        total_steps: int = 6,
        speed: float = 1.05,
        max_chunk_length: Optional[int] = None,
        req_id: Optional[str] = None,
    ) -> Iterator[bytes]:
        """Yield PCM bytes one synthesised chunk at a time.

        Heavy DEBUG logging on by default — toggle off with
        SUPERTONIC_LOG_LEVEL=INFO. Every chunk gets:
          - the chunk text preview (40 chars)
          - the inference wallclock
          - the resulting wave length in samples / seconds
        so an operator stuck on "why is the pod hanging?" can see
        exactly which chunk is mid-synthesis and how long it's taken.
        """
        tag = f"req={req_id}" if req_id else "stream_pcm"
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        effective_lang = self._resolve_lang(lang)
        if max_chunk_length is None:
            max_chunk_length = (
                DEFAULT_MAX_CHUNK_LENGTH_KO
                if effective_lang == "ko"
                else DEFAULT_MAX_CHUNK_LENGTH
            )
        # Run number / currency / symbol / abbreviation expansion BEFORE
        # chunk_text so "200万" / "$10" / "Dr." are spelled out as whole
        # tokens (chunking them mid-number would break expansion).
        raw_text = text
        if os.environ.get("SUPERTONIC_PREPROCESS", "1") == "1":
            preprocess_lang = effective_lang if effective_lang != UNKNOWN_LANGUAGE else lang
            text = preprocess_text(text, preprocess_lang)
            if text != raw_text:
                logger.info(
                    "%s preprocess(lang=%s): %r -> %r (len %d->%d)",
                    tag, preprocess_lang,
                    raw_text[:80], text[:80], len(raw_text), len(text),
                )
        chunks = list(chunk_text(text, max_chunk_length))
        logger.info(
            "%s chunker produced %d chunk(s) (max_chunk_length=%d, lang=%s, "
            "steps=%d, speed=%.2f, text_len=%d)",
            tag, len(chunks), max_chunk_length, effective_lang,
            total_steps, speed, len(text),
        )
        for i, chunk in enumerate(chunks):
            preview = chunk[:40].replace("\n", " ")
            logger.info(
                "%s chunk %d/%d: text=%r... (%d chars) starting inference",
                tag, i + 1, len(chunks), preview, len(chunk),
            )
            t0 = time.perf_counter()
            if self._lock is not None:
                with self._lock:
                    wav, _ = self.tts.model(
                        [chunk], voice_style, total_steps, speed, effective_lang
                    )
            else:
                wav, _ = self.tts.model(
                    [chunk], voice_style, total_steps, speed, effective_lang
                )
            dt = time.perf_counter() - t0
            n_samples = int(np.asarray(wav).size)
            logger.info(
                "%s chunk %d/%d: inference done in %.2fs → %d samples (%.2fs audio)",
                tag, i + 1, len(chunks), dt, n_samples,
                n_samples / self.sample_rate,
            )
            yield _pcm16_bytes(wav)
        logger.info("%s stream_pcm exiting after %d chunks", tag, len(chunks))

    def speak_streaming(
        self,
        text: str,
        voice: str = "M1",
        lang: Optional[str] = None,
        callback: Optional[Callable[[bytes], None]] = None,
        total_steps: int = 6,
        speed: float = 1.05,
        max_chunk_length: Optional[int] = None,
    ) -> int:
        """Synthesise `text` and push PCM chunks to `callback`. Returns bytes pushed."""
        style = self.get_voice_style(voice)
        total = 0
        for pcm in self.stream_pcm(
            text,
            voice_style=style,
            lang=lang,
            total_steps=total_steps,
            speed=speed,
            max_chunk_length=max_chunk_length,
        ):
            total += len(pcm)
            if callback is not None:
                callback(pcm)
        return total
