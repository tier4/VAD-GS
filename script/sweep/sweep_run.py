"""Wandb sweep wrapper for VAD-GS.

A wandb agent invokes this script with the sampled hparams as
CLI args (`--optim.densify_grad_threshold_bkgd=0.0001` ...). We:

1. Call `wandb.init()` so the agent's config is materialised.
2. Read every key from `wandb.config` and emit it as a yacs
   `key value` pair (the format `cfg.merge_from_list` expects).
3. Rewrite `sys.argv` so that importing/running `train.py` sees the
   command line `train.py --config <base> key1 v1 key2 v2 ...`.
4. Use `runpy` to execute `train.py` as `__main__` in this process —
   that way the wandb run created in step (1) is still the live run
   when train.py logs to it.

This script is also runnable manually (without wandb) for smoke
testing the harness; pass overrides as `--set key=value` repeatedly.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(env_path: Path) -> None:
    """Minimal .env loader — no dependency on python-dotenv.

    Existing process env vars take precedence (so callers can still override
    via the shell). Lines that aren't `KEY=VALUE` (after stripping `export `,
    leading whitespace, and `#...` comments) are ignored.
    """
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
        # Strip matching surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(REPO_ROOT / ".env")


def _parse_set_overrides(set_args: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _wandb_config_to_overrides() -> dict[str, str]:
    """Read wandb.config keys into a flat dict of `dotted.key -> str(value)`."""
    import wandb  # local import — keeps the manual-mode path lightweight
    cfg_keys = list(wandb.config.keys())
    overrides: dict[str, str] = {}
    for key in cfg_keys:
        if key.startswith("_"):  # wandb-internal keys (e.g. "_wandb")
            continue
        value = wandb.config[key]
        if isinstance(value, (list, tuple)):
            # yacs accepts python literals for list values.
            overrides[key] = str(list(value))
        else:
            overrides[key] = str(value)
    return overrides


def _build_yacs_opts(overrides: dict[str, str]) -> list[str]:
    opts: list[str] = []
    for key, value in overrides.items():
        opts.extend([key, value])
    return opts


def _resolve_cache_dir(cache_from: str) -> Path:
    p = Path(cache_from)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"--cache-from dir does not exist: {p}")
    return p


def _compute_model_path(base_config: Path, overrides: dict[str, str]) -> Path:
    """Replicate cfg_utils.parse_cfg's model_path computation without loading lib.config.

    model_path = cfg.model_path if set, else <workspace>/output/<task>/<exp_name>.
    cfg.workspace defaults to the repo root.
    """
    import yaml
    with open(base_config, "r") as f:
        base = yaml.safe_load(f) or {}

    task = overrides.get("task", base.get("task", ""))
    exp_name = overrides["exp_name"]  # always set by main()
    explicit = overrides.get("model_path", base.get("model_path", ""))
    workspace = Path(overrides.get("workspace", base.get("workspace", str(REPO_ROOT))))
    if not workspace.is_absolute():
        workspace = REPO_ROOT / workspace

    if explicit:
        mp = Path(explicit)
        if not mp.is_absolute():
            mp = workspace / mp
        return mp.resolve()
    return (workspace / "output" / task / exp_name).resolve()


def _link_preprocess_cache(base_config: Path, overrides: dict[str, str], cache_from: str) -> None:
    """Symlink <model_path>/{input_ply,colmap} -> <cache>/{input_ply,colmap}.

    Skips a name if the cache doesn't have it. Idempotent: if a link already
    points at the right target, leaves it alone; if a real directory exists,
    refuses to clobber it (prints a warning instead).
    """
    cache_dir = _resolve_cache_dir(cache_from)
    model_path = _compute_model_path(base_config, overrides)
    model_path.mkdir(parents=True, exist_ok=True)

    linked = []
    for name in ("input_ply", "colmap"):
        src = cache_dir / name
        if not src.is_dir():
            print(f"[sweep_run] cache missing {name}/ at {src} — skipping")
            continue
        dst = model_path / name
        if dst.is_symlink():
            if dst.resolve() == src:
                linked.append(name)
                continue
            dst.unlink()
        elif dst.exists():
            print(f"[sweep_run] {dst} already exists as a real directory — leaving as-is")
            continue
        os.symlink(src, dst, target_is_directory=True)
        linked.append(name)

    if linked:
        print(f"[sweep_run] preprocess cache wired: {model_path} -> {cache_dir} ({', '.join(linked)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="wandb sweep wrapper for VAD-GS")
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base yaml that the sweep overrides on top of.",
    )
    parser.add_argument(
        "--exp-prefix",
        default="sweep",
        help="Prefix for the per-run exp_name (final form: <prefix>_<wandb-run-id>).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip wandb.init — useful for smoke-testing without a sweep.",
    )
    parser.add_argument(
        "--set",
        dest="manual_overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Manual key=value override (repeatable). Used when --no-wandb is set.",
    )
    parser.add_argument(
        "--cache-from",
        default=os.environ.get("VAD_GS_SWEEP_CACHE_FROM", ""),
        help=(
            "Path (relative to repo root or absolute) of a *pre-built* model dir "
            "whose `input_ply/` and `colmap/` subdirs should be symlinked into "
            "this run's model_path, so COLMAP + bkgd PLY are not rebuilt for every "
            "sweep run. Defaults to env VAD_GS_SWEEP_CACHE_FROM."
        ),
    )
    # wandb agent appends sampled hparams as `--dotted.key=value` flags.
    # We don't need to parse them ourselves — wandb.init() reads them into
    # wandb.config — so use parse_known_args and ignore the rest.
    args, _unknown = parser.parse_known_args()

    base_config = (REPO_ROOT / args.base_config).resolve() if not os.path.isabs(args.base_config) else Path(args.base_config)
    if not base_config.is_file():
        raise SystemExit(f"--base-config not found: {base_config}")

    if args.no_wandb:
        overrides = _parse_set_overrides(args.manual_overrides)
        run_id = os.environ.get("VAD_GS_RUN_ID", "manual")
    else:
        import wandb
        # Entity / project are read from env vars (loaded from .env above)
        # so they stay out of the repo. WANDB_ENTITY / WANDB_PROJECT are the
        # names the wandb SDK itself reads — passing them explicitly is just
        # a safety net for when the SDK env-var lookup is disabled.
        wandb_entity = os.environ.get("WANDB_ENTITY") or None
        wandb_project = os.environ.get("WANDB_PROJECT") or None
        run = wandb.init(entity=wandb_entity, project=wandb_project)
        if run is None:
            raise SystemExit("wandb.init() returned None — is the agent invoking us?")
        overrides = _wandb_config_to_overrides()
        run_id = run.id

    # Sanity-check that the sweep didn't accidentally re-enable distributed —
    # this wrapper assumes single-GPU agents.
    overrides.setdefault("dist.enabled", "False")

    # Force a unique exp_name per run so parallel agents do not clobber each
    # other's output dir / record dir / checkpoints.
    overrides["exp_name"] = f"{args.exp_prefix}_{run_id}"

    # Reuse heavy preprocessing (COLMAP + bkgd PLY) by symlinking from a
    # pre-built model dir. Without this every sweep run rebuilds them.
    if args.cache_from:
        try:
            _link_preprocess_cache(base_config, overrides, args.cache_from)
        except Exception as exc:
            print(f"[sweep_run] WARNING: failed to wire preprocess cache: {exc}")

    yacs_opts = _build_yacs_opts(overrides)

    # Rewrite sys.argv so train.py's import-time argparse picks up the
    # config + opts. `train.py` is the program name; argparse skips argv[0].
    train_script = REPO_ROOT / "train.py"
    sys.argv = [str(train_script), "--config", str(base_config), *yacs_opts]

    # Make sure REPO_ROOT is importable as a package root (train.py uses
    # `from lib.config import cfg`).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    print(f"[sweep_run] base_config = {base_config}")
    print(f"[sweep_run] run_id      = {run_id}")
    print(f"[sweep_run] overrides   = {overrides}")

    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
