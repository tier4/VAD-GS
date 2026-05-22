"""Pick the best per-actor state across all segment checkpoints.

For each T4 original track_id present in any segment, find which
segments contain it and pick the segment whose test/test_view obj PSNR
at the final iteration is highest. Extract that segment's actor state
(from state_dict[`obj_<seq_id>`]) and stash it under the T4 original
track_id in the output.

The receiving fine-tune (train.py with `train.obj_init_from`) reads
this file, looks up the full-sequence build's own seq_id → T4 mapping,
and loads the matching state per actor.

Inputs:
  * Segment model dirs (each containing trained_model/iteration_<N>.pth
    and track_id_map.json — the latter built by either
    street_gaussian_model.setup_functions [forward-going] or
    build_track_id_maps.py [backfill]).
  * Per-segment log file (under output/segmented_logs/) used to scrape
    the psnr_obj metric. Falls back to overall best PSNR if obj-line
    is missing.

Output:
  * <out_path>: a torch.save dict of form:
        {
            "objs_by_t4_track_id": {
                <t4_track_id>: {
                    "state": <actor state_dict (xyz, scaling, ...)>,
                    "source_seg": "t4_seg_05_f200_260",
                    "source_seq_id": 7,
                    "source_metric": 17.42,
                },
                ...
            },
            "iter": 0,
        }

Usage:
    python script/experiments/merge_obj_checkpoints.py \\
        --seg-dir-glob 'output/t4_exp/t4_seg_*' \\
        --iteration 30000 \\
        --out output/t4_exp/merged_obj/iteration_0.pth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

# Regex for the per-iter test eval line that train.py prints (training_report).
# Matches: "[ITER 30000] Evaluating test/test_view: L1 0.0849 PSNR 18.959 | obj 16.046(157) near 18.743(175) ..."
_RE_EVAL = re.compile(
    r"\[ITER (\d+)\] Evaluating test/test_view: "
    r"L1 [\d.]+ PSNR ([\d.]+) "
    r"\| obj ([\d.]+)\([\d]+\)"
)


def _scrape_psnr_obj(seg_dir: Path, log_dir: Path) -> tuple[float | None, float | None, int | None]:
    """Return (psnr_obj, psnr_overall, iter) for the latest eval in this segment's log.

    Searches log_dir for files matching `<seg_name>_gpu*.log`. Uses the
    LAST (= highest-iter) eval line. Returns (None, None, None) on miss.
    """
    seg_name = seg_dir.name
    candidates = sorted(log_dir.glob(f"{seg_name}_gpu*.log"))
    if not candidates:
        # Some setups name the log by short seg index — try seg_NN_*.log instead.
        # e.g. seg_name == "t4_seg_03_f120_180" → look for "seg_03_*.log"
        m = re.search(r"seg_(\d{1,2})", seg_name)
        if m:
            short = f"seg_{int(m.group(1)):02d}"
            candidates = sorted(log_dir.glob(f"{short}_gpu*.log"))
    if not candidates:
        return None, None, None

    best = (None, None, -1)
    for log_path in candidates:
        with open(log_path, "rb") as f:
            # Reading tail is enough for the latest eval; check last 1 MB.
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 1_000_000))
            tail = f.read().decode("utf-8", errors="ignore")
        for m in _RE_EVAL.finditer(tail):
            it = int(m.group(1))
            if it > best[2]:
                best = (float(m.group(3)), float(m.group(2)), it)
    return best


def _resolve_ckpt(seg_dir: Path, iteration: int) -> Path | None:
    p = seg_dir / "trained_model" / f"iteration_{iteration}.pth"
    if p.exists():
        return p
    # Fall back to the highest available iteration if the exact one is missing.
    avail = sorted(seg_dir.glob("trained_model/iteration_*.pth"),
                   key=lambda x: int(re.search(r"iteration_(\d+)", x.name).group(1)))
    return avail[-1] if avail else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg-dir-glob", default="output/t4_exp/t4_seg_*",
                        help="Glob (relative to repo root) for segment model dirs.")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Preferred checkpoint iteration (falls back to highest available).")
    parser.add_argument("--log-dir", default="output/segmented_logs",
                        help="Directory holding per-segment stdout logs (used for metric scrape).")
    parser.add_argument("--out", required=True,
                        help="Output path for the merged-obj torch.save file.")
    parser.add_argument("--metric", choices=["psnr_obj", "psnr"], default="psnr_obj",
                        help="Which scalar to maximise when picking the best segment per actor.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    seg_dirs = sorted(Path(p) for p in glob(args.seg_dir_glob))
    if not seg_dirs:
        sys.exit(f"[merge_obj] no segment dirs matched {args.seg_dir_glob}")

    log_dir = Path(args.log_dir)

    # --- Step 1: gather (track_id_map, metric, ckpt path) per segment -----
    per_seg: dict[str, dict] = {}
    for seg_dir in seg_dirs:
        map_path = seg_dir / "track_id_map.json"
        if not map_path.exists():
            print(f"[skip] {seg_dir.name}: no track_id_map.json — run build_track_id_maps.py first")
            continue
        with open(map_path) as f:
            tid_map_raw = json.load(f)
        tid_map = {int(k): int(v) for k, v in tid_map_raw["seq_id_to_t4_track_id"].items()}

        ckpt = _resolve_ckpt(seg_dir, args.iteration)
        if ckpt is None:
            print(f"[skip] {seg_dir.name}: no checkpoint found")
            continue

        psnr_obj, psnr_overall, it = _scrape_psnr_obj(seg_dir, log_dir)
        metric = psnr_obj if args.metric == "psnr_obj" else psnr_overall
        if metric is None:
            print(f"[warn] {seg_dir.name}: could not scrape {args.metric} from logs — using metric=0")
            metric = 0.0
            it = -1

        per_seg[seg_dir.name] = {
            "seg_dir": seg_dir,
            "ckpt": ckpt,
            "tid_map": tid_map,           # {seq_id: t4_track_id}
            "metric": float(metric),
            "metric_iter": it,
        }
        print(f"[seg] {seg_dir.name}: {len(tid_map):2d} actors, "
              f"{args.metric}={metric:.3f} @ iter {it}, ckpt={ckpt.name}")

    if not per_seg:
        sys.exit("[merge_obj] no usable segments found — aborting")

    # --- Step 2: for each T4 track_id, find best segment ------------------
    # candidates[t4_id] = list of (metric, seg_name, seq_id)
    candidates: dict[int, list] = defaultdict(list)
    for seg_name, info in per_seg.items():
        for seq_id, t4_id in info["tid_map"].items():
            candidates[t4_id].append((info["metric"], seg_name, seq_id))

    # Per-T4-id: pick max-metric source
    picks: dict[int, dict] = {}
    for t4_id, opts in candidates.items():
        opts.sort(reverse=True)            # highest metric first
        metric, seg_name, seq_id = opts[0]
        picks[t4_id] = {
            "source_seg": seg_name,
            "source_seq_id": int(seq_id),
            "source_metric": float(metric),
        }

    # --- Step 3: extract actor states from the chosen ckpts ---------------
    # Group picks by source segment to amortise checkpoint loading.
    picks_by_seg: dict[str, list] = defaultdict(list)
    for t4_id, p in picks.items():
        picks_by_seg[p["source_seg"]].append((t4_id, p["source_seq_id"]))

    out_dict: dict = {"objs_by_t4_track_id": {}, "iter": 0, "picks_summary": picks}

    for seg_name, t4_seq_pairs in picks_by_seg.items():
        info = per_seg[seg_name]
        print(f"[load] {seg_name}: loading {len(t4_seq_pairs)} actor states from {info['ckpt'].name}")
        state = torch.load(info["ckpt"], map_location="cpu")
        for t4_id, seq_id in t4_seq_pairs:
            key = f"obj_{seq_id:03d}"
            if key not in state:
                print(f"  [warn] {seg_name}: ckpt has no '{key}' — skipping T4 {t4_id}")
                continue
            out_dict["objs_by_t4_track_id"][int(t4_id)] = {
                "state": state[key],
                "source_seg": seg_name,
                "source_seq_id": int(seq_id),
                "source_metric": float(info["metric"]),
            }
        del state

    # --- Step 4: save -----------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    print()
    print(f"[merge_obj] saved {len(out_dict['objs_by_t4_track_id'])} actor states → {out_path}")
    print(f"[merge_obj] picks per source segment:")
    counts: dict[str, int] = defaultdict(int)
    for p in out_dict["objs_by_t4_track_id"].values():
        counts[p["source_seg"]] += 1
    for seg_name in sorted(counts):
        print(f"  {seg_name}: {counts[seg_name]:2d} actors")


if __name__ == "__main__":
    main()
