"""Lightweight per-section wall-time profiler for the training loop.

Enable by setting ``VAD_GS_PERF=1`` in the environment. When disabled the
context manager is a true no-op (Python-level only, no CUDA sync) so it
adds nothing to production training.

Env vars:
    VAD_GS_PERF         "1" to enable, anything else disables. Default off.
    VAD_GS_PERF_SYNC    "1" (default) inserts ``torch.cuda.synchronize()``
                        around each section to measure the *true* wall
                        time including GPU work. Set to "0" to measure
                        Python-side time only (cheaper, but mixes in
                        kernel-launch async time).
    VAD_GS_PERF_EVERY   Print averaged stats every N iterations.
                        Default 50.
    VAD_GS_PERF_FILE    Optional path; when set, also append a CSV row
                        per print.

Typical use::

    from lib.utils.perf_timer import perf_section, perf_iter_end

    for it in range(N):
        with perf_section("render"):
            ...
        with perf_section("loss"):
            ...
        perf_iter_end()
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("VAD_GS_PERF", "0") == "1"
_SYNC = os.environ.get("VAD_GS_PERF_SYNC", "1") == "1"
_PRINT_EVERY = max(1, int(os.environ.get("VAD_GS_PERF_EVERY", "50")))
_CSV_FILE = os.environ.get("VAD_GS_PERF_FILE", "")

# Use lists rather than running sums so we can summarise the last
# _PRINT_EVERY samples (i.e. ignore warm-up iterations).
_timings: "defaultdict[str, list[float]]" = defaultdict(list)
_order: list[str] = []  # preserve first-seen order for stable printing
_iter_count = 0
_csv_header_written = False


def is_enabled() -> bool:
    return _ENABLED


def _maybe_sync() -> None:
    if _SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def perf_section(name: str):
    """Context manager that records wall time for ``name``.

    No-op when ``VAD_GS_PERF`` is not set, so callers can sprinkle these
    over the hot path without worrying about production overhead.
    """
    if not _ENABLED:
        yield
        return

    _maybe_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_sync()
        dt = time.perf_counter() - t0
        bucket = _timings[name]
        if not bucket:
            _order.append(name)
        bucket.append(dt)


def _emit_csv(rows: list[tuple[str, float]]) -> None:
    global _csv_header_written
    if not _CSV_FILE:
        return
    try:
        new_file = not os.path.exists(_CSV_FILE)
        with open(_CSV_FILE, "a") as f:
            if new_file or not _csv_header_written:
                names = [name for name, _ in rows]
                f.write("iter," + ",".join(names) + ",total\n")
                _csv_header_written = True
            vals = [f"{ms:.3f}" for _, ms in rows]
            total = sum(ms for _, ms in rows)
            f.write(f"{_iter_count}," + ",".join(vals) + f",{total:.3f}\n")
    except OSError as exc:
        print(f"[perf] CSV write failed: {exc}")


def perf_iter_end() -> None:
    """Call at the end of each training iteration. No-op when disabled."""
    if not _ENABLED:
        return

    global _iter_count
    _iter_count += 1
    if _iter_count % _PRINT_EVERY != 0:
        return

    rows: list[tuple[str, float]] = []
    for name in _order:
        samples = _timings[name][-_PRINT_EVERY:]
        if not samples:
            continue
        avg_ms = sum(samples) / len(samples) * 1000.0
        rows.append((name, avg_ms))

    if not rows:
        return

    total = sum(ms for _, ms in rows)
    # Sort by time descending for readability; preserve original list too.
    sorted_rows = sorted(rows, key=lambda r: -r[1])
    print(
        f"\n[perf] iter {_iter_count}  avg over last {_PRINT_EVERY} iters  "
        f"(sync={_SYNC})"
    )
    for name, ms in sorted_rows:
        pct = 100.0 * ms / total if total > 0 else 0.0
        print(f"  {name:30s} {ms:9.3f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s} {total:9.3f} ms")

    _emit_csv(rows)


def perf_reset() -> None:
    """Drop all collected samples (useful around densify / opacity_reset)."""
    if not _ENABLED:
        return
    _timings.clear()
    _order.clear()
