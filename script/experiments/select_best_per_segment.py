"""Pick the best segmented-sweep run per `data.selected_frames` group.

Reads finished runs from a wandb sweep, groups them by their
`data.selected_frames` config value (= one of the 3 segments), then
prints the best-PSNR run per segment along with the path to its
final checkpoint on local disk.

Output format (one line per segment) is consumable by
script/experiments/merge_bg_checkpoints.py:

    output/t4_exp/seg_<run_id>/trained_model/iteration_30000.pth

Usage:
    # by sweep id (from `python script/sweep/sweep_init.py ...`):
    python script/experiments/select_best_per_segment.py \\
        --sweep-id advanced-technology-department/VAD-GS/abc1234

    # print + a one-liner shell command that calls merge_bg_checkpoints:
    python script/experiments/select_best_per_segment.py \\
        --sweep-id <id> --emit-merge-cmd
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Allow `from script.sweep...` imports if needed
REPO_ROOT = Path(__file__).resolve().parents[2]


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
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="wandb sweep id, in the form `entity/project/sweepid` or just `sweepid` "
             "(entity/project filled from WANDB_ENTITY / WANDB_PROJECT in .env).",
    )
    parser.add_argument(
        "--metric",
        default="test/test_view/best_psnr",
        help="wandb summary key to rank by. Defaults to the best test PSNR "
             "tracked across the sweep (set by train.py:training_report). "
             "Use `test/test_view/best_psnr_bkgd_near` to bias toward static "
             "FG quality (the merge step only carries BG forward, so this "
             "is often the better selector).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="Checkpoint iteration to point at on disk (seg_base.yaml saves "
             "[4000, 8000, 16000, 24000, 30000]).",
    )
    parser.add_argument(
        "--workspace",
        default=str(REPO_ROOT),
        help="Repo root (used to resolve output/<task>/<exp_name>/trained_model).",
    )
    parser.add_argument(
        "--emit-merge-cmd",
        action="store_true",
        help="Append a ready-to-run `merge_bg_checkpoints.py` command line "
             "after the per-segment listing.",
    )
    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed — `uv pip install wandb` or activate the venv.")

    # Allow short sweep id (just 'abc123') if entity/project are set in env.
    sweep_full = args.sweep_id
    if sweep_full.count("/") == 0:
        entity = os.environ.get("WANDB_ENTITY")
        project = os.environ.get("WANDB_PROJECT")
        if not (entity and project):
            sys.exit("Short sweep id given but WANDB_ENTITY / WANDB_PROJECT not set in env.")
        sweep_full = f"{entity}/{project}/{sweep_full}"

    api = wandb.Api()
    sweep = api.sweep(sweep_full)

    groups: dict[tuple, list] = defaultdict(list)
    for run in sweep.runs:
        if run.state != "finished":
            continue
        sf = run.config.get("data.selected_frames")
        if sf is None:
            sf = run.config.get("data", {}).get("selected_frames")  # nested form fallback
        if sf is None:
            continue
        # Make hashable
        sf_key = tuple(sf) if isinstance(sf, (list, tuple)) else sf
        metric = run.summary.get(args.metric)
        if metric is None:
            continue
        groups[sf_key].append((float(metric), run))

    if not groups:
        sys.exit(f"No finished runs found in sweep {sweep_full} with metric {args.metric}.")

    best_ckpts: list[str] = []
    print(f"# Per-segment best runs from sweep {sweep_full}")
    print(f"# Ranking metric: {args.metric}")
    for sf_key in sorted(groups):
        runs = groups[sf_key]
        runs.sort(key=lambda x: x[0], reverse=True)
        top_metric, top_run = runs[0]
        exp_name = top_run.config.get("exp_name") or f"seg_{top_run.id}"
        task = top_run.config.get("task", "t4_exp")
        ckpt = os.path.join(args.workspace, "output", task, exp_name, "trained_model", f"iteration_{args.iteration}.pth")
        exists_tag = "" if os.path.isfile(ckpt) else "  (MISSING on disk)"
        print(f"# segment frames={list(sf_key)}  n_finished={len(runs)}  "
              f"top {args.metric}={top_metric:.4f}  run={top_run.id}")
        print(f"{ckpt}{exists_tag}")
        best_ckpts.append(ckpt)

    if args.emit_merge_cmd:
        out_path = os.path.join(args.workspace, "output", "t4_exp", "merged_bg", "iteration_0.pth")
        print()
        print("# Suggested merge command:")
        print(
            "python script/experiments/merge_bg_checkpoints.py \\\n"
            f"    --ckpts {' '.join(best_ckpts)} \\\n"
            f"    --out {out_path} \\\n"
            "    --dedup-voxel-size 0.15"
        )


if __name__ == "__main__":
    main()
