"""Create a wandb sweep from a YAML file and print its full ID.

The output is one line of the form `entity/project/sweep_id` — pipe it
straight into `xargs wandb agent` or capture it in a shell variable.

Usage:
    python script/sweep/sweep_init.py configs/sweep/t4_sweep.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import wandb
import yaml

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
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(REPO_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_yaml", help="Path to the wandb sweep yaml.")
    parser.add_argument(
        "--project",
        default=None,
        help="Override the project field in the YAML (optional).",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Override the entity field in the YAML (optional).",
    )
    args = parser.parse_args()

    with open(args.sweep_yaml, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    project = args.project or sweep_cfg.pop("project", None) or os.environ.get("WANDB_PROJECT")
    entity = args.entity or sweep_cfg.pop("entity", None) or os.environ.get("WANDB_ENTITY")
    if not project:
        raise SystemExit(
            "No wandb project specified. Set WANDB_PROJECT in .env, or pass --project, "
            "or add `project:` to the sweep yaml."
        )

    sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)

    # `wandb.sweep` returns just the sweep_id (e.g. "abc123"); for use with
    # `wandb agent` we want the fully-qualified `entity/project/sweep_id`.
    api = wandb.Api()
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}")
        full_id = f"{sweep.entity}/{sweep.project}/{sweep.id}"
    except Exception:
        full_id = f"{project}/{sweep_id}" if project else sweep_id

    print(full_id)


if __name__ == "__main__":
    main()
