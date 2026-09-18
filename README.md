# Supertonic Streaming TTS Service

Production HTTP streaming TTS server built on **[Supertonic](https://huggingface.co/Supertone/supertonic-3)** (ONNX Runtime + TensorRT FP16 on NVIDIA GPUs), with a request shape that is **drop-in compatible with the OpenAI / XTTS audio.speech API**. Point your existing XTTS client at this server and it will work — the conditioning + speech two-call flow is preserved, and XTTS-only sampling fields (`temperature`, `top_k`, `top_p`, `repetition_penalty`, `length_penalty`, `do_sample`, `enhance_speech`, `gpt_cond_*`, `max_ref_length`, ...) are accepted and silently ignored. They are not listed in the OpenAPI schema — none of them can affect a diffusion model's output, so advertising them would misrepresent what the API does.

> The upstream multi-language SDK examples (cpp/csharp/flutter/go/ios/java/nodejs/py/rust/swift/web) live on the [`main`](https://github.com/supertone-inc/supertonic) branch of the public repo.

## Folder layout

```
app/                          service source + image build (the docker build context)
├── src/                      production modules — everything the image ships
│   ├── server.py             /v1/audio/speech + /v1/tts/conditioning + /health /ready /metrics /docs
│   ├── models.py             all data models: request/response bodies + the AudioFormat registry
│   ├── streaming_synth.py    chunk-streaming wrapper over supertonic.TTS, TensorRT EP setup
│   ├── audio_format.py       pcm / wav / flac / opus / mp3 encoding + sample-rate negotiation
│   ├── text_preprocess.py    number/abbreviation expansion + language detection
│   └── zh_num2words.py       Chinese numeral expansion, used by text_preprocess
├── tests/                    pytest suite — no GPU, no weights (supertonic + ORT are stubbed)
│   ├── conftest.py               puts src/ and tests/ on sys.path
│   ├── test_tts.py               end-to-end request/response behaviour
│   ├── test_speech_endpoint.py   /v1/audio/speech, incl. every response_format
│   ├── test_audio_format.py      container encoding + sample-rate negotiation
│   ├── test_request_logging.py   raw-body logging, redaction, req= join key
│   ├── test_chat_shape_compat.py chat-completions bodies used as liveness probes
│   └── test_legacy_api_compat.py XTTS-shaped bodies from pre-existing clients
├── pyproject.toml            the only dependency manifest — prod deps + `dev` / `bench` groups
├── uv.lock                   locked resolution; regenerate with `uv lock`
├── Dockerfile                single stage, deps installed by uv from pyproject.toml + uv.lock
├── Dockerfile.base           the apt/TensorRT layer, pre-baked for runners that cannot apt
└── entrypoint.sh             validates the mounted model, then runs src/server.py
```

Run the tests with no GPU and no model weights — `supertonic` and ONNX Runtime are stubbed in `conftest.py`:

```bash
cd app && uv run --group dev pytest
```

## HTTP API

| Method | Path                  | Purpose                                                                                                       |
|--------|-----------------------|---------------------------------------------------------------------------------------------------------------|
| GET    | `/health`             | Liveness. Always 200 once the process is up (even during warm-up).                                            |
| GET    | `/ready`              | Readiness. 200 only after the model is loaded and the warm-up synthesis succeeded; 503 otherwise.             |
| GET    | `/metrics`            | Prometheus text exposition (`supertonic_requests_total`, `_inflight`, `_failed`, `_bytes_streamed_total`, ...). |
| POST   | `/v1/tts/conditioning`| Precompute the speaker style. Returns `gpt_cond_latent_b64` + `speaker_embeddings_b64`.                       |
| POST   | `/v1/audio/speech`    | Synthesise. `response_format` picks the output: `pcm` (default, streamed) / `wav` / `flac` / `opus` / `mp3`.   |

### Two-call flow (XTTS-compatible)

**Step 1 — get conditioning** (precompute the voice style once):
```bash
curl -sS http://localhost:8000/v1/tts/conditioning \
  -H 'Content-Type: application/json' \
  -d '{ "speaker_files": ["M1"] }'
```
```json
{ "gpt_cond_latent_b64": "<...>", "speaker_embeddings_b64": "<...>",
  "sample_rate": 44100, "model": "supertonic-3" }
```

`speaker_files[0]` accepts:
| Value                          | Meaning                                                                                                        |
|--------------------------------|----------------------------------------------------------------------------------------------------------------|
| `M1`..`M5`, `F1`..`F5`         | One of the 10 built-in voice styles bundled with supertonic-3.                                                 |
| Path ending in `.json`         | A [Voice Builder](https://supertonic.supertone.ai/voice-builder) style file — load it from disk.               |
| Path ending in `.wav` / `.mp3` | **Rejected with 400.** The open-weight Supertonic model does not clone voices from raw audio at request time.  |

**Step 2 — synthesise** (re-use the blobs from step 1):
```bash
curl -sS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello from the streaming TTS server.",
    "model": "xttsv2",
    "response_format": "pcm",
    "stream": true,
    "language": "auto",
    "gpt_cond_latent_b64": "<...>",
    "speaker_embeddings_b64": "<...>",
    "speed": 1.0
  }' --output out.pcm

# Raw PCM is headerless — the default output rate is 24000 Hz, not
# supertonic's native 44100. Pass `"sample_rate": N` to change it; the
# response's X-Sample-Rate header always reports the rate actually used.
ffplay -f s16le -ar 24000 -ac 1 out.pcm
```

### One-call shortcut (skip conditioning)

If you don't need to cache the conditioning, you can pass `voice` directly:
```bash
curl -sS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{ "input": "Hello.", "voice": "M1", "language": "en", "stream": true }' \
  --output out.pcm
```

### OpenAI output formats

`response_format` follows OpenAI's speech API:

| Value             | Content-Type             | Transfer          |
|-------------------|--------------------------|-------------------|
| `pcm` *(default)* | `audio/pcm`              | chunked, streamed |
| `wav`             | `audio/wav`              | one complete file |
| `flac`            | `audio/flac`             | one complete file |
| `opus`            | `audio/ogg; codecs=opus` | one complete file |
| `mp3`             | `audio/mpeg`             | one complete file |

`pcm` is the default, so clients written against earlier releases get
byte-identical responses. Only `pcm` streams — the encoded containers are
buffered and sent whole, so ask for `pcm` when time-to-first-byte matters.
`aac` and `stream_format: "sse"` are rejected with a 400 instead of being
silently substituted — libsndfile cannot write AAC, and answering an SSE
request with raw bytes would hand the client a stream it cannot parse.
`mp3` costs nothing (libsndfile has written MPEG Layer III since 1.1 and
the `soundfile` wheel bundles 1.2.2), so it is supported.

```bash
curl -sS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{ "input": "Hello.", "voice": "M1", "response_format": "wav" }' \
  --output out.wav
```

An OpenAI SDK client works unchanged:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
client.audio.speech.create(
    model="supertonic-3", voice="M1", input="Hello.", response_format="wav"
).stream_to_file("out.wav")
```

### Languages

`language` accepts any of the 31 ISO codes supported by supertonic-3 (`en`, `ko`, `ja`, `ar`, `hi`, `de`, `es`, `fr`, ...) plus `na` or `auto` for the language-agnostic fallback.

Text outside those 31 languages is detected with `lingua-py` and answered with a
fixed English notice rather than mispronounced audio. Chinese is in that bucket
on purpose: the model accepts Han characters but reads them with Japanese kanji
readings, which is useless to a Chinese listener.

## How the streaming is emulated

**Supertonic has no streaming API.** The upstream `supertonic` package exposes
exactly one synthesis entry point — `TTS.synthesize(text, ...)` — and it is
offline: it splits long text internally, synthesises *every* piece, concatenates
the results, and returns a single finished waveform. Nothing reaches the caller
until the last piece is done. A diffusion vocoder also has no autoregressive
token loop to tap, so there is no "partial frame" to emit mid-inference either.

This server therefore **simulates** a stream rather than exposing a real one.
The unit of streaming is one *text chunk*, not a frame and not a diffusion step.
Everything below is the trick, end to end.

### 1. Hoist supertonic's own chunker up to the server

`streaming_synth.StreamingSynthesize.stream_pcm` does not call
`TTS.synthesize`. It reuses supertonic's internal sentence-packing chunker and
then drives the model one chunk at a time:

```python
# app/src/streaming_synth.py
from supertonic.utils import chunk_text

chunks = list(chunk_text(text, max_chunk_length))
for chunk in chunks:
    wav, _ = self.tts.model([chunk], voice_style, total_steps, speed, lang)
    yield _pcm16_bytes(wav)            # <- one chunk of audio, the moment it exists
```

`self.tts.model(...)` is the layer *underneath* `synthesize` — the same call
`synthesize` makes in its own loop. Hoisting the loop into a generator is the
entire difference between the offline API and this one. Output is identical
audio; only the delivery schedule changes.

Before chunking, `text_preprocess.preprocess_text` expands numbers, currency
and abbreviations (`"200万"`, `"$10"`, `"Dr."`). That has to happen *first* —
chunking mid-number would split a token the expander needs whole.

### 2. Convert and resample, statefully

Each chunk's float32 waveform is clipped to [-1, 1] and packed to signed 16-bit
little-endian PCM, then passed through a `PcmResampler` — a thin wrapper over
stdlib `audioop.ratecv` that converts supertonic's native 44100 Hz to the
24000 Hz that XTTS / pipecat clients expect.

The resampler is **one instance per request and carries its filter state across
chunks**. Resampling each chunk independently would leave an audible click at
every sentence boundary; threading the state through removes it.

### 3. Bridge the blocking generator onto the event loop

`stream_pcm` is a blocking generator running ONNX inference. FastAPI needs an
async one. `server.speech` bridges them with a worker thread and a **bounded**
queue:

```python
queue = asyncio.Queue(maxsize=8)          # bounded on purpose

def _producer():                          # runs on a worker thread
    for pcm in State.synth.stream_pcm(...):
        asyncio.run_coroutine_threadsafe(queue.put(resampler.process(pcm)), loop).result()
    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

async def _pcm_chunks():                  # drains it for StreamingResponse
    loop.run_in_executor(None, _producer)
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=chunk_timeout_s)
        ...
        yield item
```

`maxsize=8` is the backpressure valve: a slow client stops draining, the queue
fills, and the producer thread blocks on `put` instead of synthesising the whole
utterance into RAM. The `asyncio.wait_for` gives every chunk an inactivity
timeout (`SUPERTONIC_CHUNK_TIMEOUT_S`, default 120 s) so a hung inference
surfaces as an error instead of a connection that streams zero bytes forever.

### 4. Ship it as HTTP chunked transfer

For `response_format: "pcm"`, each queue item is yielded straight out of a
FastAPI `StreamingResponse`, which writes it as one HTTP chunk. For every other
format the same generator is drained into a buffer, encoded once by
`soundfile`, and returned whole with an exact `Content-Length`.

That split is why the format table above marks only `pcm` as streamed. A
streamed WAV would need placeholder RIFF sizes patched at close time, which
strict parsers (Python's own `wave` module included) reject.

### What this buys, and what it costs

```
Offline (TTS.synthesize):   [chunk1][chunk2][chunk3] ────────────────▶ all bytes at the end
                            └──────── client waits ────────┘

This server (stream_pcm):   [chunk1]──▶bytes  [chunk2]──▶bytes  [chunk3]──▶bytes
                            └─ TTFB ─┘
```

* **TTFB is the cost of the *first* chunk, not the whole utterance.** That is
  the entire reason the numbers below are in milliseconds — a long paragraph
  and a short sentence have almost the same TTFB, because both start emitting
  after one chunk.
* **`chunk_size` is the TTFB knob.** It maps to supertonic's
  `max_chunk_length` (a character target, default ~300). Lower it and the first
  chunk lands sooner; lower it too far and you pay per-call overhead on every
  chunk and coarsen prosody across the extra boundaries. If your client is
  pipecat-style with tiny chunks, also drop the TensorRT shape-profile
  optimisation point to match: `SUPERTONIC_TRT_TEXT_OPT=32
  SUPERTONIC_TRT_LATENT_OPT=128`.
* **It is not real-time-per-token.** Within a chunk you still wait for the full
  diffusion + vocoder pass. If a single chunk is 8 s of audio, the second byte
  is a chunk-latency behind the first, not a frame behind.
* **`stream: true` is informational.** Chunked transfer is decided by
  `response_format`, not by this field; it is accepted for XTTS compatibility.
  SSE (`stream_format: "sse"`) is not implemented and is rejected with a 400.
* **Mid-stream failures cannot change the status code.** By then the 200 and
  its headers are already on the wire, so the server appends a literal
  `\n[error] ...\n` marker to the body — long-standing behaviour existing
  clients parse. The buffered formats have not written anything yet, so they
  still return a clean JSON 500.

### The conditioning endpoint is a compatibility shim too

`POST /v1/tts/conditioning` exists only so XTTS clients keep their two-call
shape. Supertonic has no GPT conditioning pre-fill to precompute — a voice style
is a small pair of arrays that loads instantly. The endpoint resolves the style
and returns its two arrays, `.npy`-serialised and base64'd, under XTTS's field
names:

| XTTS field                | Actually carries                |
|---------------------------|---------------------------------|
| `gpt_cond_latent_b64`     | `Style.ttl` (text-to-latent)    |
| `speaker_embeddings_b64`  | `Style.dp` (duration predictor) |

On the way back in, `/v1/audio/speech` decodes them and reconstitutes
`Style(ttl, dp)`. Blobs that are *not* in that format — a real XTTS client
sending torch-pickled latents, for instance — are silently rejected and the
server falls back to the `voice` field, so such a client gets working audio in
a built-in voice instead of a 400.

## Build & deploy

Minimal local run (Docker, bind-mount weights, GPU 0):

```bash
docker build -t supertonic-streaming:latest app
docker run --rm --gpus '"device=0"' -p 8000:8000 \
  -v "$HOME/.cache/supertonic3:/models/supertonic-3:ro" \
  -v supertonic-trt-cache:/var/cache/supertonic/trt \
  -e SUPERTONIC_MODEL_PATH=/models/supertonic-3 \
  supertonic-streaming:latest
```

The build defaults to the public `nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04`
and installs TensorRT 10.13 on top. On a runner that cannot reach Docker Hub or
apt, pre-bake that layer once with `Dockerfile.base` and point the service build
at it:

```bash
docker build --build-arg CUDA_BASE=<registry>/supertonic-tts:base -t supertonic-streaming:latest app
```

Two mounts matter in production:

* **the weights**, read-only at `SUPERTONIC_MODEL_PATH`. `entrypoint.sh`
  verifies all six required files are present and refuses to start otherwise,
  rather than failing deep inside model load.
* **the TensorRT engine cache**, read-write at `SUPERTONIC_TRT_CACHE`. It
  **must** survive restarts — the first-time engine build takes 5–15 minutes,
  and a pod that loses this volume pays that on every start. Size any startup
  probe accordingly.

### Configuration

Everything is env-vars (and matching `--flags`). The ones you are most likely to touch:

| Variable                        | Default                      | Purpose                                                        |
|---------------------------------|------------------------------|----------------------------------------------------------------|
| `SUPERTONIC_MODEL_PATH`         | *(unset)*                    | Mounted model dir. Required unless `SUPERTONIC_AUTO_DOWNLOAD=1`. |
| `SUPERTONIC_AUTO_DOWNLOAD`      | `0`                          | Fetch weights from HuggingFace. Dev only.                       |
| `SUPERTONIC_TRT`                | `1`                          | Use the TensorRT EP. Falls back to CUDA EP if it won't load.    |
| `SUPERTONIC_TRT_CACHE`          | `/var/cache/supertonic/trt`  | Persistent engine cache. Mount a volume here.                   |
| `SUPERTONIC_TRT_FP16`           | `1`                          | FP16 engines.                                                   |
| `SUPERTONIC_OUTPUT_SAMPLE_RATE` | `24000`                      | Default wire rate; per-request `sample_rate` overrides it.      |
| `SUPERTONIC_CHUNK_TIMEOUT_S`    | `120`                        | Per-chunk inactivity timeout. Sized to cover a cold engine build. |
| `SUPERTONIC_SERIALIZE`          | `1`                          | Per-process mutex around inference — see the note below.        |
| `SUPERTONIC_CPU`                | `0`                          | CPU-only. Debugging only; TTFB is ~4–5× the GPU number.         |
| `SUPERTONIC_PREPROCESS`         | `1`                          | Number/abbreviation expansion before chunking.                  |
| `SUPERTONIC_LOG_REQUEST_BODY`   | `1`                          | Log the raw POST body (redacted, truncated) before validation.  |
| `SUPERTONIC_LOG_LEVEL`          | `INFO`                       | `DEBUG` adds per-chunk text previews and inference wallclocks.  |
| `SUPERTONIC_TRT_TEXT_OPT`       | `256`                        | TRT shape-profile tuning point for `text_length`.               |
| `SUPERTONIC_TRT_LATENT_OPT`     | `512`                        | TRT shape-profile tuning point for `latent_length`.             |

> **`SUPERTONIC_SERIALIZE` defaults to on.** Concurrent
> `supertonic.TTS.__call__` has been observed to deadlock in production (GPU
> drops to 0% util and the second request never logs "inference done"), so each
> process serialises inference behind a mutex. To recover per-process
> throughput you must either set `SUPERTONIC_SERIALIZE=0` — accepting that
> risk — or, preferably, run more replicas behind a load balancer. The
> concurrency figures below were measured without the mutex.

## Performance

Measured on a single **NVIDIA A100 80GB**, TensorRT FP16 execution provider, diffusion `total_steps=6`, FP32 input PCM streamed at 44.1 kHz.

### Headline numbers (TensorRT FP16)

| Concurrency | TTFB p50 | TTFB p95 | TTFB max | Total p95 | Throughput | Success |
|---:|---:|---:|---:|---:|---:|---:|
| **1**  | **35.4 ms** | **36.0 ms** | 36.1 ms | 0.065 s | 15.5 rps | 100% |
| 2  |  56.6 ms |  66.2 ms |  66.8 ms | 0.110 s | 19.3 rps | 100% |
| 4  |  99.5 ms | 111.3 ms | 112.1 ms | 0.182 s | 22.0 rps | 100% |
| 8  | 192.7 ms | 212.6 ms | 214.5 ms | 0.352 s | 22.7 rps | 100% |
| 10 | 242.3 ms | 264.6 ms | 267.7 ms | 0.433 s | 23.0 rps | 100% |
| 12 | 288.3 ms | 322.4 ms | 329.9 ms | 0.516 s | 23.2 rps | 100% |
| 14 | 340.3 ms | 372.6 ms |   ~376 ms | 0.609 s | 23.1 rps | 100% |
| 16 | 393.7 ms | 439.6 ms |   ~442 ms | 0.680 s | 23.5 rps | 100% |
| 18 | 443.6 ms | 483.4 ms |   ~486 ms | 0.759 s | 23.6 rps | 100% |
| 20 | 482.0 ms | 563.6 ms |   ~568 ms | 0.847 s | 23.7 rps | 100% |
| 22 | 536.5 ms | 603.2 ms |   ~608 ms | 0.935 s | 23.6 rps | 100% |
| 24 | 575.2 ms | 641.3 ms |   ~648 ms | 0.992 s | 24.1 rps | 100% |
| **26** | 623.8 ms | 696.6 ms | 700.4 ms | 1.062 s | **24.3 rps** | 100% |

TTFB stays predictable; success rate stays at 100% across the whole sweep; throughput plateaus around 23–24 rps as the GPU saturates.

Read these as *first-chunk* latencies — see [How the streaming is emulated](#how-the-streaming-is-emulated). They are close to constant in input length, because the client starts receiving audio after one chunk regardless of how long the full utterance is.

### Side-by-side with CUDA-only (no TensorRT)

This is what the same workload looks like if you stop at the CUDA execution provider — the state of things before the TensorRT EP rollout.

| Concurrency | CUDA TTFB p50 | TRT TTFB p50 | TTFB speed-up | CUDA rps | TRT rps | rps speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 1659.4 ms | **35.4 ms** | **46.8×** |  0.30 | 15.54 | **51.1×** |
| 2  | 1682.7 ms |  56.6 ms | 29.7× |  0.60 | 19.30 | 32.2× |
| 4  | 1731.5 ms |  99.5 ms | 17.4× |  1.16 | 22.05 | 19.0× |
| 8  | 1897.5 ms | 192.7 ms |  9.8× |  2.15 | 22.65 | 10.5× |
| 16 | 2214.5 ms | 393.7 ms |  5.6× |  3.73 | 23.50 |  6.3× |
| 26 | 2686.8 ms | 623.8 ms |  4.3× |  5.00 | 24.34 |  4.9× |

**47× lower TTFB** at concurrency=1, **~5× higher peak throughput** at concurrency=26. The relative speed-up narrows as load rises because the GPU compute itself becomes the bottleneck — but even at 26 concurrent streams TRT still wins by 4–5× on TTFB.

### Methodology

* Client harness: concurrent HTTP/1.1 streaming requests, per-request CSV row, summary CSV, success/error counts, real percentiles (not interpolated).
* GPU monitoring: `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw -l 0.5`.
* Concurrency sweep: 1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26. Each level runs 3× concurrency requests to compute meaningful percentiles.
* TensorRT engine cache warmed once before the sweep so the headline numbers reflect steady-state, not the 5–15 min first-time engine build.

> Two things that did *not* move the metric, recorded so they don't get retried:
> lowering the diffusion step count and shrinking the text chunks each changed
> TTFB by less than 5%. The TensorRT execution provider is what moved it.

## License

| Component        | License        | Reference                                                                                      |
|------------------|----------------|------------------------------------------------------------------------------------------------|
| Code in this repo | MIT           | [`LICENSE`](LICENSE)                                                                           |
| Model weights     | OpenRAIL-M    | <https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE>                              |

Commercial use is permitted under both licenses. The OpenRAIL-M model license requires you to **propagate the use-based restrictions in Attachment A** (no deepfake-without-consent, no automated-justice/immigration decisions, no medical advice, no impersonation, no targeted harassment, no undisclosed AI-generated content in bot contexts, ...) into your downstream ToS / EULA. This is paperwork, not a code change.
