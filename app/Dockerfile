# Supertonic Streaming TTS server image (CUDA 12.9 + cuDNN 9 + TensorRT 10.13).
#
# Single-stage build. Dependencies are managed entirely by uv: pyproject.toml
# declares them, uv.lock pins them, and both are the only inputs — there is no
# requirements.txt in the repo any more.
#
# Base image:
#   Defaults to the public nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04
#   (Ubuntu 22.04 → Python 3.10 in apt, matching the pyproject pin). The
#   [base-probe] layer below then installs TensorRT 10.13.0.35 from NVIDIA's
#   apt repo plus python3-venv, libsndfile1, tini and curl.
#
#   Build runners that cannot reach Docker Hub or apt should instead point
#   CUDA_BASE at a pre-baked base built once from Dockerfile.base on a host
#   with egress — that image already carries all of the above, so the probe
#   installs nothing. Both paths end at the same layer, and this file needs
#   no other change to switch between them.
#
# Why not nvcr.io/nvidia/tensorrt:<tag>: NGC's year-month tags don't map
# linearly to TRT versions — 25.08-py3 still ships TRT 10.5, which fails
# supertonic-3's dynamic-shape graphs with "TensorRT input ... has no shape
# specified". The host where pressure-tests pass runs TRT 10.13.0.35-1+cuda12.9
# installed from the NVIDIA apt repo, so we install the exact same version
# here to guarantee matching behaviour.

# The base image is an ARG so a build can be redirected to a registry mirror
# or to a pre-baked base. It must sit here, before the first FROM: an ARG
# after a FROM belongs to that stage and can no longer parameterise a later
# one.
#
#   docker build --build-arg CUDA_BASE=<registry>/supertonic-tts:base app
#
# There is deliberately no second base image. Do NOT reintroduce
# `COPY --from=ghcr.io/astral-sh/uv` to get the uv binary: an air-gapped
# runner cannot reach ghcr.io, and the handshake fails with "x509:
# certificate signed by unknown authority", which looks like a CA problem and
# is not. uv is installed from the same PyPI index as everything else instead
# (see the uv bootstrap below), so the build needs exactly one registry and
# one index.
ARG CUDA_BASE=nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

FROM ${CUDA_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Exact host pin: dpkg -l | grep libnvinfer-bin → 10.13.0.35-1+cuda12.9
ARG TRT_VERSION=10.13.0.35-1+cuda12.9

# Optional package index, for runners that can't reach pythonhosted.org and
# have to go through an internal PyPI proxy. Passed in at build time:
#
#   docker build --build-arg PIP_INDEX_URL="https://user:pass@host/repo/simple" .
#
# ARG, deliberately not ENV: the URL usually carries credentials, and an ENV
# would bake them into the image config where `docker inspect` shows them and
# every container inherits them. An ARG is build-time only, and BuildKit does
# not record build-args in the image metadata. Empty default = go straight to
# the public index, which is what a developer build should do.
#
# Still named PIP_INDEX_URL now that uv does the installing: CI forwards it
# under that name from a masked variable, and it feeds both the one-shot
# `pip install uv` and every `uv pip install` after it. Renaming it means
# renaming the CI variable too.
ARG PIP_INDEX_URL=

# uv version used to resolve and install. Pinned like everything else: a uv
# bump can change resolution behaviour, and this build has no way to test that
# on the runner. Bump deliberately, not incidentally.
ARG UV_VERSION=0.9.8

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Nothing is installed at run time, so a populated cache is dead weight
    # in the image layer.
    UV_NO_CACHE=1 \
    # The venv is on the same filesystem as the build, but hardlinking out of
    # a temp dir across BuildKit layers is what produces uv's "failed to
    # hardlink" warnings; copying is the predictable choice in a container.
    UV_LINK_MODE=copy \
    # uv must never try to download a CPython of its own — the runner has no
    # egress to python-build-standalone, and the image deliberately pins the
    # apt python3.10 that pyproject requires.
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/supertonic/.venv \
    PATH=/opt/supertonic/.venv/bin:/usr/local/bin:/usr/bin:/bin

# System deps + Python 3.10 + TensorRT 10.13 runtime.
#
# Ubuntu 22.04's apt ships Python 3.10 (matches pyproject's >=3.10,<3.12).
# The CUDA base image already has the NVIDIA CUDA apt repo configured, and
# that repo also serves the libnvinfer* packages — no extra repo setup
# needed. Pinning the exact host version (10.13.0.35-1+cuda12.9) so the
# image reproduces the host pressure-test config byte-for-byte at the TRT
# layer. libnvinfer-plugin10 is required by ORT's TensorRT EP at session
# create time.
#
# Whether this runs at all is decided by probing the base image, not by a
# flag the caller has to get right. $CUDA_BASE normally already carries
# every one of these — that is the entire point of Dockerfile.base — and
# re-running apt on a runner that cannot reach apt is what used to fail the
# build three layers in.
#
# So: check each feature, and skip the install when the base already has
# them all. This replaces a hand-set SKIP_APT=true whose failure mode was
# silent — a stock CUDA image plus SKIP_APT=true produced an image with no
# python3 in it, and the build then died much later with an error naming
# neither the flag nor the base.
#
# SKIP_APT remains as an override:
#   auto  (default) probe, and install only if something is missing
#   true            never install; FAIL if the probe says something is missing
#   false           always install, whatever the probe says
ARG SKIP_APT=auto
RUN set -eu; \
    missing=''; \
    have_lib() { \
        ldconfig -p 2>/dev/null | grep -q "$1" \
            || ls /usr/lib/*/"$1"* >/dev/null 2>&1; \
    }; \
    command -v python3 >/dev/null 2>&1             || missing="$missing python3"; \
    python3 -c 'import ensurepip' >/dev/null 2>&1  || missing="$missing python3-venv"; \
    command -v curl >/dev/null 2>&1                || missing="$missing curl"; \
    [ -x /usr/bin/tini ]                           || missing="$missing tini"; \
    [ -e /etc/ssl/certs/ca-certificates.crt ]      || missing="$missing ca-certificates"; \
    have_lib 'libsndfile.so.1'                     || missing="$missing libsndfile1"; \
    have_lib 'libnvinfer_plugin.so.10'             || missing="$missing libnvinfer-plugin10"; \
    have_lib 'libnvonnxparser.so.10'               || missing="$missing libnvonnxparsers10"; \
    trt_so=$(ls /usr/lib/*/libnvinfer.so.10.* 2>/dev/null | sort -V | tail -n1 || true); \
    if [ -z "$trt_so" ]; then \
        missing="$missing libnvinfer10"; \
    else \
        trt_minor=$(echo "$trt_so" | sed -E 's/.*libnvinfer\.so\.10\.([0-9]+).*/\1/'); \
        case "$trt_minor" in \
            ''|*[!0-9]*) missing="$missing libnvinfer10(unparsable:$trt_so)" ;; \
            *) [ "$trt_minor" -ge 13 ] \
                 || missing="$missing libnvinfer10(have-10.$trt_minor-need-10.13+)" ;; \
        esac; \
    fi; \
    if [ -n "$missing" ]; then \
        echo "[base-probe] base image is missing:$missing"; \
    else \
        echo "[base-probe] base image already provides every system dependency"; \
    fi; \
    case "$SKIP_APT" in \
        true) \
            if [ -n "$missing" ]; then \
                echo "ERROR: SKIP_APT=true, but the base image is missing:$missing" >&2; \
                echo "       Nothing later in this build installs them, so the image" >&2; \
                echo "       would be broken. Point CUDA_BASE at the pre-baked base" >&2; \
                echo "       (scripts/build-base-image.sh prints its name), or drop" >&2; \
                echo "       SKIP_APT and let the probe decide." >&2; \
                exit 1; \
            fi; \
            do_apt=no ;; \
        false) do_apt=yes; echo "[base-probe] SKIP_APT=false — installing anyway" ;; \
        *)     if [ -n "$missing" ]; then do_apt=yes; else do_apt=no; fi ;; \
    esac; \
    if [ "$do_apt" = yes ]; then \
        echo "[base-probe] running apt"; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            ca-certificates curl libsndfile1 tini \
            python3 python3-venv \
            libnvinfer10=${TRT_VERSION} \
            libnvinfer-plugin10=${TRT_VERSION} \
            libnvonnxparsers10=${TRT_VERSION}; \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "[base-probe] skipping apt — nothing to install"; \
    fi; \
    ln -sf /usr/bin/python3 /usr/local/bin/python3

WORKDIR /opt/supertonic

# --- uv bootstrap ------------------------------------------------------
#
# uv comes from PyPI, not from ghcr.io (see the note above the first ARG).
# It is installed into a throwaway venv built by ensurepip — python3-venv is
# already in the image, so this needs no apt and no second registry — and
# that venv is deleted in the same layer once uv has done its work.
#
# This one `pip install` is the only pip in the build, and it exists solely
# to obtain uv. Every dependency of the application is resolved, exported and
# installed by uv below.
#
# The ${VAR:+...} form expands to nothing when PIP_INDEX_URL is unset or
# empty, so an unproxied developer build goes to the public index instead of
# passing a broken `--index-url ` with no argument.
# Dependency manifests first, so this layer caches independently of source
# changes.
COPY pyproject.toml uv.lock ./

# Resolve → export → install, all with uv.
#
# Why `uv export | uv pip install` and not `uv sync --frozen`: uv.lock records
# every wheel by absolute files.pythonhosted.org URL, and `uv sync` fetches
# exactly those URLs without consulting any index — verified, a sync with a
# dead --index-url and --no-cache still succeeds. On a build runner restricted
# to a private registry and an internal PyPI proxy, those URLs are unreachable
# and no index override can redirect them. `uv export` emits the same resolution as
# plain name==version, which uv pip *will* fetch from whatever index it is
# given. uv.lock stays the single source of truth either way.
#
# --no-dev drops the `dev` and `bench` dependency-groups (pytest, httpx,
# aiohttp, matplotlib): the image runs the service, it does not test it.
#
# --no-deps on the install is load-bearing. The exported set is already the
# complete resolution, and it deliberately omits the CPU `onnxruntime` that
# supertonic declares (see tool.uv.dependency-metadata in pyproject.toml).
# Letting uv re-resolve pulls that back in, and it overwrites onnxruntime-gpu
# in the same site-packages directory — the TensorRT EP then vanishes and the
# assertion below catches it.
# Bootstrap, install and tear down in a single layer so the throwaway uv venv
# never lands in a published layer.
RUN PI="${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL}" && \
    python3 -m venv /tmp/uv-bootstrap && \
    /tmp/uv-bootstrap/bin/pip install --quiet --no-cache-dir --disable-pip-version-check \
        $PI "uv==${UV_VERSION}" && \
    UV=/tmp/uv-bootstrap/bin/uv && \
    "$UV" --version && \
    "$UV" venv --python /usr/bin/python3 /opt/supertonic/.venv && \
    "$UV" export --frozen --no-dev --no-hashes --no-emit-project \
        -o /tmp/requirements.txt && \
    echo "--- resolved $(grep -cE '^[a-z0-9]' /tmp/requirements.txt) packages from uv.lock ---" && \
    "$UV" pip install --python /opt/supertonic/.venv/bin/python \
        $PI --no-deps -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt /tmp/uv-bootstrap && \
    # Fail the build right here if anything is wrong: missing deps,
    # CPU onnxruntime sneaking in instead of -gpu, or — critically —
    # TRT < 10.13 (older minor versions don't handle supertonic-3's
    # dynamic-shape graphs).
    /opt/supertonic/.venv/bin/python -c '\
import sys, ctypes, glob, re, time, numpy, fastapi, soundfile, supertonic, onnxruntime, num2words, lingua;\
ctypes.CDLL("libnvinfer.so.10");\
so = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libnvinfer.so.10.*"))[-1];\
m = re.search(r"libnvinfer\.so\.(\d+)\.(\d+)\.(\d+)$", so);\
trt = tuple(int(x) for x in m.groups());\
assert trt >= (10, 13, 0), f"TRT {trt} < 10.13 — supertonic-3 needs 10.13+ for dynamic-shape graphs (host=10.13.0). Bump TRT_VERSION.";\
eps = onnxruntime.get_available_providers();\
assert "TensorrtExecutionProvider" in eps, f"TRT EP missing! available={eps} — this image needs onnxruntime-gpu, not onnxruntime";\
fmts = soundfile.available_formats();\
assert {"WAV", "FLAC", "OGG", "MP3"} <= set(fmts), f"libsndfile {soundfile.__libsndfile_version__} lacks a response_format container we serve: have={sorted(fmts)}";\
assert "OPUS" in soundfile.available_subtypes("OGG"), f"libsndfile {soundfile.__libsndfile_version__} cannot encode Ogg Opus (needs >= 1.0.29)";\
assert "MPEG_LAYER_III" in soundfile.available_subtypes("MP3"), f"libsndfile {soundfile.__libsndfile_version__} cannot encode MP3 (needs >= 1.1)";\
t0 = time.perf_counter();\
det = lingua.LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build();\
assert det.detect_language_of("Hello world").iso_code_639_1.name.lower() == "en", "lingua sanity-check failed";\
print("venv OK, py=" + sys.version.split()[0], "ort=" + onnxruntime.__version__, f"TRT {trt[0]}.{trt[1]}.{trt[2]} loadable", "providers=" + ",".join(eps), f"lingua build={time.perf_counter()-t0:.2f}s")'

# Application code. The whole of src/ goes in as a directory rather than a
# hand-maintained file list — that list used to drift, and a missing module is
# a ModuleNotFoundError at container start and a crash-looping pod, because
# server.py imports audio_format / streaming_synth / text_preprocess and
# streaming_synth imports text_preprocess → zh_num2words, all at module load.
#
# tests/, benchmarks/ and archive/ are deliberately absent: they are not
# needed to serve, and archive/ in particular holds prototype modules that
# would shadow the real ones if they ever shared a directory.
COPY src/ ./src/
COPY entrypoint.sh ./

# Operator mounts the supertonic weights at /app/models/<model-dir> via a
# PVC; the TRT engine cache must also be persisted. Pre-create both paths
# so the entrypoint's mkdir is a no-op in the common case.
RUN mkdir -p /var/cache/supertonic/trt /app/models \
 && chmod +x /opt/supertonic/entrypoint.sh

# Defaults — override via env or k8s ConfigMap.
ENV SUPERTONIC_HOST=0.0.0.0 \
    SUPERTONIC_PORT=8000 \
    SUPERTONIC_MODEL_NAME=supertonic-3 \
    SUPERTONIC_TRT=1 \
    SUPERTONIC_TRT_FP16=1 \
    SUPERTONIC_TRT_CACHE=/var/cache/supertonic/trt \
    SUPERTONIC_WARMUP=1 \
    SUPERTONIC_OUTPUT_SAMPLE_RATE=24000 \
    SUPERTONIC_LOG_LEVEL=INFO

EXPOSE 8000

# Kubernetes readiness probes hit /ready (NOT /health) — /ready waits for
# the model to be loaded. /health returns 200 even during warm-up so the
# liveness probe doesn't kill the pod mid-engine-build.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/supertonic/entrypoint.sh"]
