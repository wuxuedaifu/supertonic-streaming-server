"""Pressure test for the Supertonic streaming TTS HTTP server.

Targets the supertonic streaming endpoint /v1/audio/speech exposed in
py/pressure_test/server.py (a thin FastAPI wrapper over
supertonic.StreamingSynthesize) and answers two questions:

    1. How many concurrent in-flight requests can the server support
       before requests start failing or queueing collapses?
    2. How does Time-to-First-Byte (TTFB) change as concurrency grows?

The script:
    * fires a wave of N concurrent streaming requests against
      /v1/audio/speech for each concurrency level in CONCURRENCY_LEVELS;
    * measures TTFB (first streamed chunk), total time, byte count, and
      success/failure per request;
    * writes a per-request CSV and an aggregated CSV (column schema is
      preserved from the upstream Auralis pressure-test script — only the
      request-side function call changes);
    * renders TTFB-vs-concurrency and throughput-vs-concurrency plots.

Outputs land in: pressure_test/

Example:
    python test_tts.py \
        --url http://localhost:3000 \
        --concurrency 1,2,4,8,10,12,14,16,18,20,22,24,26 \
        --waves 2

Notes:
    Unlike the original Auralis target, the supertonic server takes a
    built-in voice name (M1..M5, F1..F5) — there is no reference-audio
    upload and no /v1/tts/conditioning step. ``--speaker-file`` is
    accepted for backwards compatibility but ignored; pass ``--voice`` to
    pick a voice. ``--skip-conditioning`` is likewise accepted as a no-op.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import aiohttp


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEAKER_FILE = REPO_ROOT / "examples" / "hello.wav"  # unused; kept for CLI parity
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pressure_test"
DEFAULT_URL = "http://localhost:3000"
DEFAULT_TEXT = (
    "In this stress test we want to measure how the time to first byte "
    "of the streaming TTS endpoint changes as the number of concurrent "
    "callers grows. This sentence is intentionally long enough to make "
    "the model generate several streamed audio chunks per request."
)
DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_WAVES = 3
DEFAULT_TIMEOUT_S = 300.0


@dataclass
class RequestSample:
    concurrency: int
    wave: int
    request_idx: int
    status: int
    ok: bool
    ttfb_s: float | None
    total_s: float
    bytes_received: int
    error: str = ""


@dataclass
class ConcurrencySummary:
    concurrency: int
    n_requests: int
    n_ok: int
    success_rate: float
    ttfb_mean_s: float | None
    ttfb_p50_s: float | None
    ttfb_p95_s: float | None
    ttfb_p99_s: float | None
    ttfb_max_s: float | None
    total_mean_s: float | None
    total_p95_s: float | None
    throughput_rps: float
    wall_clock_s: float
    errors: list[str] = field(default_factory=list)


async def _await_with_heartbeat(task: "asyncio.Task[RequestSample]", label: str, interval_s: float):
    start = time.perf_counter()
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=interval_s)
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            print(f"  [{label}] still waiting... elapsed={elapsed:.1f}s", flush=True)


async def _check_health(session: aiohttp.ClientSession, base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/health"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = (await resp.text())[:200]
            print(f"  /health -> status={resp.status} body={body!r}")
            return resp.status == 200
    except Exception as exc:
        print(f"  /health request failed: {exc!r}", file=sys.stderr)
        return False


def _build_speech_payload(
    text: str,
    voice: str,
    language: Optional[str],
    response_format: str,
    chunk_size: int | None,
    total_steps: int,
    speed: float,
) -> dict:
    """Build the supertonic /v1/audio/speech request body.

    ``chunk_size`` from the upstream Auralis CLI is mapped to
    ``max_chunk_length`` on the supertonic server — both control the
    text-chunk-boundary granularity at which streamed audio chunks are
    emitted (smaller = more chunks = more responsive TTFB). The CLI keeps
    the original flag name so the operator's invocation does not need to
    change.
    """
    payload: dict = {
        "input": text,
        "voice": voice,
        "model": "supertonic-3",
        "response_format": response_format,
        "stream": True,
        "language": language,
        "total_steps": total_steps,
        "speed": speed,
    }
    if chunk_size is not None and chunk_size > 0:
        payload["max_chunk_length"] = chunk_size
    return payload


async def _run_single_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    concurrency: int,
    wave: int,
    request_idx: int,
    timeout_s: float,
) -> RequestSample:
    start = time.perf_counter()
    ttfb: float | None = None
    bytes_received = 0
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            status = resp.status
            ok = status == 200
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                if ttfb is None:
                    ttfb = time.perf_counter() - start
                bytes_received += len(chunk)
            total = time.perf_counter() - start
            err = "" if ok else (await resp.text())[:200]
            return RequestSample(
                concurrency=concurrency,
                wave=wave,
                request_idx=request_idx,
                status=status,
                ok=ok and bytes_received > 0,
                ttfb_s=ttfb,
                total_s=total,
                bytes_received=bytes_received,
                error=err,
            )
    except asyncio.TimeoutError:
        return RequestSample(
            concurrency=concurrency,
            wave=wave,
            request_idx=request_idx,
            status=0,
            ok=False,
            ttfb_s=None,
            total_s=time.perf_counter() - start,
            bytes_received=bytes_received,
            error="timeout",
        )
    except Exception as exc:
        return RequestSample(
            concurrency=concurrency,
            wave=wave,
            request_idx=request_idx,
            status=0,
            ok=False,
            ttfb_s=None,
            total_s=time.perf_counter() - start,
            bytes_received=bytes_received,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_wave(
    session: aiohttp.ClientSession,
    speech_url: str,
    payload: dict,
    concurrency: int,
    wave: int,
    timeout_s: float,
) -> tuple[list[RequestSample], float]:
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(_run_single_request(
            session=session,
            url=speech_url,
            payload=payload,
            concurrency=concurrency,
            wave=wave,
            request_idx=i,
            timeout_s=timeout_s,
        ))
        for i in range(concurrency)
    ]
    samples: list[RequestSample] = []
    completed = 0
    last_heartbeat = start
    for fut in asyncio.as_completed(tasks):
        sample = await fut
        samples.append(sample)
        completed += 1
        now = time.perf_counter()
        if now - last_heartbeat >= 5.0 or completed == len(tasks):
            print(
                f"    wave {wave}: {completed}/{concurrency} done "
                f"(t={now - start:.1f}s)",
                flush=True,
            )
            last_heartbeat = now
    wall = time.perf_counter() - start
    samples.sort(key=lambda s: s.request_idx)
    return samples, wall


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summarize(level: int, samples: list[RequestSample], wall_clock_s: float) -> ConcurrencySummary:
    ok_samples = [s for s in samples if s.ok]
    ttfbs = [s.ttfb_s for s in ok_samples if s.ttfb_s is not None]
    totals = [s.total_s for s in ok_samples]
    errors = sorted({s.error for s in samples if not s.ok and s.error})
    return ConcurrencySummary(
        concurrency=level,
        n_requests=len(samples),
        n_ok=len(ok_samples),
        success_rate=(len(ok_samples) / len(samples)) if samples else 0.0,
        ttfb_mean_s=statistics.fmean(ttfbs) if ttfbs else None,
        ttfb_p50_s=_percentile(ttfbs, 0.50),
        ttfb_p95_s=_percentile(ttfbs, 0.95),
        ttfb_p99_s=_percentile(ttfbs, 0.99),
        ttfb_max_s=max(ttfbs) if ttfbs else None,
        total_mean_s=statistics.fmean(totals) if totals else None,
        total_p95_s=_percentile(totals, 0.95),
        throughput_rps=(len(ok_samples) / wall_clock_s) if wall_clock_s > 0 else 0.0,
        wall_clock_s=wall_clock_s,
        errors=errors,
    )


def _write_per_request_csv(path: Path, samples: list[RequestSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "concurrency", "wave", "request_idx", "status", "ok",
            "ttfb_s", "total_s", "bytes_received", "error",
        ])
        for s in samples:
            writer.writerow([
                s.concurrency, s.wave, s.request_idx, s.status, int(s.ok),
                f"{s.ttfb_s:.6f}" if s.ttfb_s is not None else "",
                f"{s.total_s:.6f}",
                s.bytes_received,
                s.error,
            ])


def _write_summary_csv(path: Path, summaries: list[ConcurrencySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "concurrency", "n_requests", "n_ok", "success_rate",
            "ttfb_mean_s", "ttfb_p50_s", "ttfb_p95_s", "ttfb_p99_s", "ttfb_max_s",
            "total_mean_s", "total_p95_s",
            "throughput_rps", "wall_clock_s",
            "errors",
        ])
        for s in summaries:
            writer.writerow([
                s.concurrency, s.n_requests, s.n_ok, f"{s.success_rate:.4f}",
                _fmt(s.ttfb_mean_s), _fmt(s.ttfb_p50_s), _fmt(s.ttfb_p95_s),
                _fmt(s.ttfb_p99_s), _fmt(s.ttfb_max_s),
                _fmt(s.total_mean_s), _fmt(s.total_p95_s),
                f"{s.throughput_rps:.4f}", f"{s.wall_clock_s:.4f}",
                " | ".join(s.errors),
            ])


def _fmt(v: float | None) -> str:
    return f"{v:.6f}" if v is not None else ""


def _print_summary_table(summaries: list[ConcurrencySummary]) -> None:
    header = (
        f"{'conc':>5} {'reqs':>5} {'ok':>4} {'succ':>6} "
        f"{'ttfb_p50':>9} {'ttfb_p95':>9} {'ttfb_p99':>9} {'ttfb_max':>9} "
        f"{'tot_p95':>9} {'rps':>7}"
    )
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.concurrency:>5d} {s.n_requests:>5d} {s.n_ok:>4d} "
            f"{s.success_rate:>5.1%} "
            f"{_cell(s.ttfb_p50_s):>9} {_cell(s.ttfb_p95_s):>9} "
            f"{_cell(s.ttfb_p99_s):>9} {_cell(s.ttfb_max_s):>9} "
            f"{_cell(s.total_p95_s):>9} {s.throughput_rps:>7.2f}"
        )
    print("=" * len(header))


def _cell(v: float | None) -> str:
    return f"{v:.3f}s" if v is not None else "  n/a  "


def _plot(summaries: list[ConcurrencySummary], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[warn] matplotlib is not installed; skipping plots. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    concurrencies = [s.concurrency for s in summaries]
    # Convert seconds → milliseconds for the TTFB plot (more readable units).
    def to_ms(values):
        return [v * 1000.0 if v is not None else None for v in values]

    p50 = to_ms([s.ttfb_p50_s for s in summaries])
    p95 = to_ms([s.ttfb_p95_s for s in summaries])
    p99 = to_ms([s.ttfb_p99_s for s in summaries])
    mean = to_ms([s.ttfb_mean_s for s in summaries])
    success = [s.success_rate * 100 for s in summaries]
    rps = [s.throughput_rps for s in summaries]

    # ----- TTFB plot -----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, ys in (("mean", mean), ("p50", p50), ("p95", p95), ("p99", p99)):
        xs = [c for c, y in zip(concurrencies, ys) if y is not None]
        ys_ = [y for y in ys if y is not None]
        if ys_:
            ax.plot(xs, ys_, marker="o", label=label)
            for x, y in zip(xs, ys_):
                ax.annotate(
                    f"{y:.0f} ms",
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Time-to-First-Byte (milliseconds)")
    ax.set_title("TTFB vs Concurrency — /v1/audio/speech (streaming)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pressure_test_ttfb.png", dpi=140)
    plt.close(fig)

    # ----- Success rate / throughput plot -----
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(concurrencies, success, marker="o", color="tab:green", label="success rate (%)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(concurrencies)
    ax1.set_xticklabels([str(c) for c in concurrencies])
    ax1.set_xlabel("Concurrency (in-flight requests)")
    ax1.set_ylabel("Success rate (%)", color="tab:green")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(concurrencies, rps, marker="s", color="tab:blue", label="throughput (rps)")
    ax2.set_ylabel("Throughput (requests/sec)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Throughput & Success Rate vs Concurrency")
    fig.tight_layout()
    fig.savefig(out_dir / "pressure_test_throughput.png", dpi=140)
    plt.close(fig)

    print(f"Saved plots to {out_dir}")


def _parse_concurrency(arg: str) -> tuple[int, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("must contain at least one concurrency value")
    try:
        levels = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid concurrency list: {arg!r}") from exc
    if any(c < 1 for c in levels):
        raise argparse.ArgumentTypeError("concurrency values must be >= 1")
    return levels


async def run_pressure_test(
    base_url: str,
    speaker_file: Path,
    text: str,
    voice: str,
    language: Optional[str],
    total_steps: int,
    speed: float,
    concurrency_levels: Iterable[int],
    waves: int,
    response_format: str,
    timeout_s: float,
    skip_conditioning: bool,
    skip_warmup: bool,
    chunk_size: int | None,
    output_dir: Path,
) -> tuple[list[RequestSample], list[ConcurrencySummary]]:
    speech_url = f"{base_url.rstrip('/')}/v1/audio/speech"

    # speaker_file / skip_conditioning are accepted but unused for the
    # supertonic backend — surface that so the operator is not confused.
    if speaker_file and Path(str(speaker_file)).exists():
        print(f"Speaker file: {speaker_file} (ignored — supertonic uses built-in voice {voice!r})")
    else:
        print(f"Voice: {voice} (supertonic built-in; --speaker-file/--skip-conditioning are no-ops)")

    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Health check: {base_url}/health")
        if not await _check_health(session, base_url):
            print(
                "[error] Server /health did not return 200. Aborting before "
                "the pressure run — the URL is probably wrong or the service is down.",
                file=sys.stderr,
            )
            return [], []

        payload = _build_speech_payload(
            text=text,
            voice=voice,
            language=language,
            response_format=response_format,
            chunk_size=chunk_size,
            total_steps=total_steps,
            speed=speed,
        )
        print(
            f"Request payload: stream=True max_chunk_length={chunk_size} "
            f"response_format={response_format} voice={voice} language={language}"
        )

        if not skip_warmup:
            print(
                "Warmup request (cold-start streaming TTS — this can take 20-60s; "
                "use --skip-warmup to skip) ..."
            )
            warm_task = asyncio.create_task(_run_single_request(
                session, speech_url, payload,
                concurrency=0, wave=-1, request_idx=0, timeout_s=timeout_s,
            ))
            warm = await _await_with_heartbeat(warm_task, label="warmup", interval_s=5.0)
            if warm.ok:
                print(
                    f"  warmup ok status={warm.status} "
                    f"ttfb={warm.ttfb_s:.3f}s total={warm.total_s:.3f}s "
                    f"bytes={warm.bytes_received}"
                )
            else:
                print(
                    f"  warmup FAILED status={warm.status} error={warm.error!r} "
                    f"(continuing anyway — failures will be captured per-level)",
                    file=sys.stderr,
                )

        all_samples: list[RequestSample] = []
        summaries: list[ConcurrencySummary] = []
        for level in concurrency_levels:
            print(f"\n--- concurrency={level} ({waves} waves) ---")
            level_samples: list[RequestSample] = []
            total_wall = 0.0
            for w in range(waves):
                samples, wall = await _run_wave(
                    session=session,
                    speech_url=speech_url,
                    payload=payload,
                    concurrency=level,
                    wave=w,
                    timeout_s=timeout_s,
                )
                level_samples.extend(samples)
                total_wall += wall
                ok = sum(1 for s in samples if s.ok)
                ttfb_ok = [s.ttfb_s for s in samples if s.ok and s.ttfb_s is not None]
                ttfb_avg = statistics.fmean(ttfb_ok) if ttfb_ok else float("nan")
                print(
                    f"  wave {w}: ok={ok}/{level} wall={wall:.2f}s "
                    f"avg_ttfb={ttfb_avg:.3f}s"
                )
            all_samples.extend(level_samples)
            summary = _summarize(level, level_samples, total_wall)
            summaries.append(summary)

        per_req_csv = output_dir / "pressure_test_per_request.csv"
        summary_csv = output_dir / "pressure_test_summary.csv"
        _write_per_request_csv(per_req_csv, all_samples)
        _write_summary_csv(summary_csv, summaries)
        print(f"\nWrote {per_req_csv}")
        print(f"Wrote {summary_csv}")

        _print_summary_table(summaries)
        _plot(summaries, output_dir)

        # Also dump a json blob for downstream tooling
        (output_dir / "pressure_test_summary.json").write_text(
            json.dumps([asdict(s) for s in summaries], indent=2)
        )

        return all_samples, summaries


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help="Base URL of the running TTS server")
    p.add_argument("--speaker-file", type=Path, default=DEFAULT_SPEAKER_FILE,
                   help="(Ignored for supertonic — kept for CLI parity with the Auralis variant.)")
    p.add_argument("--voice", default="M1",
                   help="Supertonic built-in voice name: M1..M5 or F1..F5 (default M1)")
    p.add_argument("--language", default="en",
                   help="ISO language code accepted by supertonic, or 'na' for the multilingual fallback")
    p.add_argument("--total-steps", type=int, default=8,
                   help="Number of diffusion steps (supertonic default 8)")
    p.add_argument("--speed", type=float, default=1.05,
                   help="Speech speed multiplier (default 1.05)")
    p.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    p.add_argument("--concurrency", type=_parse_concurrency, default=DEFAULT_CONCURRENCY,
                   help='Comma-separated concurrency levels (default "1,2,4,8,16,32,64")')
    p.add_argument("--waves", type=int, default=DEFAULT_WAVES,
                   help="How many waves per concurrency level (each wave fires N concurrent requests)")
    p.add_argument("--response-format", default="pcm",
                   choices=["pcm"],
                   help="Audio response format (supertonic streaming server emits raw int16 PCM)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="Per-request timeout in seconds")
    p.add_argument("--skip-conditioning", action="store_true",
                   help="(Accepted for CLI parity; supertonic has no conditioning step.)")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip the single warmup request before the pressure run")
    p.add_argument("--chunk-size", type=int, default=80,
                   help="Mapped to supertonic's max_chunk_length. Smaller means more "
                        "audio chunks per request and snappier TTFB (default: 80). "
                        "Set to 0 or negative to let the server pick.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write CSVs and PNG plots")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_pressure_test(
        base_url=args.url,
        speaker_file=args.speaker_file,
        text=args.text,
        voice=args.voice,
        language=args.language,
        total_steps=args.total_steps,
        speed=args.speed,
        concurrency_levels=args.concurrency,
        waves=args.waves,
        response_format=args.response_format,
        timeout_s=args.timeout,
        skip_conditioning=args.skip_conditioning,
        skip_warmup=args.skip_warmup,
        chunk_size=args.chunk_size if args.chunk_size and args.chunk_size > 0 else None,
        output_dir=args.output_dir,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
