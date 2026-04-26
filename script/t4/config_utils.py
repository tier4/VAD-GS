"""Shared utilities for T4 preprocessing scripts.

Provides YAML config loading and dataroot resolution so that
preprocessing scripts can be driven by the same config file used
for training (e.g. configs/example/t4_train_example.yaml).
"""

import os
from pathlib import Path

import yaml

_ANNOTATION_DATASET_BASE = os.path.expanduser("~/.webauto/data/data/annotation_dataset")


def resolve_dataroot(dataset_id_or_path, revision=0, version=0):
    """Resolve a dataset UUID or path to a concrete directory."""
    candidate = os.path.expanduser(str(dataset_id_or_path))
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "annotation")):
        return Path(candidate)
    id_path = os.path.join(_ANNOTATION_DATASET_BASE, str(dataset_id_or_path))
    if os.path.isdir(id_path):
        ver_path = os.path.join(id_path, str(version))
        if os.path.isdir(ver_path):
            return Path(ver_path)
        rev_path = os.path.join(id_path, str(revision))
        if os.path.isdir(rev_path):
            return Path(rev_path)
        return Path(id_path)
    return Path(candidate)


def load_train_config(config_path):
    """Load a training YAML config and extract preprocessing-relevant params.

    Returns a dict with keys:
        dataroot, version, revision, scene_index, camera_channels, lidar_channel, box_scale
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    data = raw.get("data", {})

    return {
        "dataroot": raw.get("source_path", None),
        "version": data.get("version", 0),
        "revision": data.get("revision", 0),
        "scene_index": data.get("scene_index", 0),
        "camera_channels": data.get("camera_channels", None),
        "lidar_channel": data.get("lidar_channel", "LIDAR_CONCAT"),
        "box_scale": data.get("box_scale", 1.5),
    }


def add_config_arg(parser):
    """Add --config argument to an argparse parser."""
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training YAML config (e.g. configs/example/t4_train_example.yaml). "
             "Provides defaults for --dataroot, --revision, --scene-index, etc.",
    )


def apply_config_defaults(args, config_keys=None):
    """Apply config file defaults to argparse Namespace.

    Values from --config are used only when the corresponding CLI arg
    was not explicitly provided (i.e. is still at its default / None).

    *config_keys* lists which keys from load_train_config() to apply.
    If None, all keys are applied.
    """
    if args.config is None:
        return

    cfg = load_train_config(args.config)

    # Mapping: config dict key -> argparse attribute name
    key_map = {
        "dataroot": "dataroot",
        "version": "version",
        "revision": "revision",
        "scene_index": "scene_index",
        "camera_channels": "camera_channels",
        "lidar_channel": "lidar_channel",
        "box_scale": "box_scale",
    }

    for cfg_key, attr in key_map.items():
        if config_keys and cfg_key not in config_keys:
            continue
        if not hasattr(args, attr):
            continue
        cfg_val = cfg.get(cfg_key)
        if cfg_val is None:
            continue
        current = getattr(args, attr)
        # Only override if the CLI arg was not explicitly set
        if current is None:
            setattr(args, attr, cfg_val)
