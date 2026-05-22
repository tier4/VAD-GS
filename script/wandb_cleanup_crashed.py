"""Delete wandb runs that crashed with zero runtime.

By default this is a dry-run that just lists matching runs. Pass `--apply`
to actually delete them.

Usage:
    uv run python script/wandb_cleanup_crashed.py --entity <ent> --project <proj>
    uv run python script/wandb_cleanup_crashed.py --entity <ent> --project <proj> --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import wandb
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wandb_cleanup")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(REPO_ROOT / ".env")


def _runtime_seconds(run) -> float:
    # Read `_runtime` straight out of the already-loaded `summaryMetrics`
    # blob on the run. Going through `run.summary` would trigger a per-run
    # GraphQL fetch and tank throughput when there are thousands of runs.
    attrs = getattr(run, "_attrs", None) or {}
    summary = attrs.get("summaryMetrics") or attrs.get("summary_metrics") or {}
    if isinstance(summary, str):
        # Older API versions hand back JSON-encoded summaries.
        import json
        try:
            summary = json.loads(summary)
        except Exception:
            summary = {}
    raw = summary.get("_runtime") if isinstance(summary, dict) else None
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="wandb entity (team/user).")
    parser.add_argument("--project", required=True, help="wandb project name.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matching runs. Without this flag, the script "
             "only lists them (dry-run).",
    )
    parser.add_argument(
        "--runtime-threshold",
        type=float,
        default=0.0,
        help="Treat any run with runtime <= this many seconds as zero-runtime "
             "(default: 0.0, i.e. exact zero).",
    )
    args = parser.parse_args()

    log.info("Connecting to wandb API (entity=%s, project=%s)",
             args.entity, args.project)
    api = wandb.Api(timeout=60)

    # Keep the server-side filter minimal — `state` is indexed and fast.
    # Filtering on `summary_metrics._runtime` (especially with `$exists`)
    # makes wandb's count/query path slow, so we do runtime filtering
    # client-side instead.
    filters = {"state": "crashed"}

    log.info("Querying crashed runs (per_page=500, runtime filtered client-side)...")
    runs = api.runs(
        path=f"{args.entity}/{args.project}",
        filters=filters,
        per_page=500,
    )

    # `len(runs)` triggers a server-side count which can take a while; do it
    # in the background-ish way by simply trying once with a short note.
    log.info("Fetching total count...")
    try:
        total = len(runs)
        log.info("Total crashed runs on server: %d.", total)
    except Exception as exc:
        total = None
        log.warning("Could not get total count (%s); progress bar will be open-ended.", exc)

    log.info("Scanning runs and filtering by runtime <= %.1fs ...",
             args.runtime_threshold)
    matched = []
    skipped = 0
    pbar = tqdm(runs, total=total, desc="scan", unit="run")
    for run in pbar:
        runtime = _runtime_seconds(run)
        if runtime <= args.runtime_threshold:
            matched.append((run, runtime))
            pbar.set_postfix(matched=len(matched), skipped=skipped)
        else:
            skipped += 1
    pbar.close()
    log.info("Scan done: %d matched, %d skipped.", len(matched), skipped)

    if not matched:
        log.info("No crashed runs with runtime <= %.1fs found in %s/%s.",
                 args.runtime_threshold, args.entity, args.project)
        return

    action = "Deleting" if args.apply else "[dry-run] Would delete"
    log.info("%s %d crashed run(s) in %s/%s:",
             action, len(matched), args.entity, args.project)
    for run, runtime in matched:
        created = getattr(run, "created_at", "?")
        log.info("  - %s  name=%r  runtime=%.1fs  created_at=%s",
                 run.id, run.name, runtime, created)

    if not args.apply:
        log.info("Re-run with --apply to actually delete these runs.")
        return

    log.info("Deleting %d run(s)...", len(matched))
    failed = 0
    pbar = tqdm(matched, desc="delete", unit="run")
    for run, _ in pbar:
        pbar.set_postfix_str(run.id)
        try:
            run.delete()
        except Exception as exc:
            failed += 1
            log.error("failed to delete %s: %s", run.id, exc)

    deleted = len(matched) - failed
    log.info("Deleted %d/%d run(s).", deleted, len(matched))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
