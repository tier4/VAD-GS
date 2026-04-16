#!/bin/bash
# Export a VAD-GS checkpoint (.pth) to Cesium 3D Tiles in one shot.
#
# Usage:
#   ./export_cesium.sh [extra args for export_cesium.py]
#
# Examples:
#   # Config only — checkpoint auto-detected:
#   ./export_cesium.sh --config configs/example/t4_train_example.yaml --background-only
#
#   # Config + explicit checkpoint:
#   ./export_cesium.sh output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
#       --config configs/example/t4_train_example.yaml --background-only
#
#   # Manual coordinates:
#   ./export_cesium.sh output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
#       --lat 35.6812 --lon 139.7671 --output my_tiles
#
# Notes:
#   - Uses a dedicated Python 3.13 venv at .venv_export, isolated from the
#     main VAD-GS environment (which pins numpy==1.23.5 and conflicts with
#     3dgs-io's numpy>=2.1 requirement).
#   - The venv is bootstrapped automatically on first run via `uv`.
#   - export_cesium.py only uses torch.load + numpy; no VAD-GS module imports,
#     so isolation is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_export"
PY_VERSION="3.13"

# --- Bootstrap .venv_export if missing -------------------------------------
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "=== Creating export venv at ${VENV_DIR} (Python ${PY_VERSION}) ==="
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: 'uv' not found. Install from https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    uv venv --python "${PY_VERSION}" "${VENV_DIR}"
fi

# --- Ensure dependencies are installed --------------------------------------
PY="${VENV_DIR}/bin/python"

if ! "${PY}" -c "import torch, numpy, yaml; import importlib; importlib.import_module('3dgs_io'); import spz" >/dev/null 2>&1; then
    echo "=== Installing export dependencies into ${VENV_DIR} ==="
    VIRTUAL_ENV="${VENV_DIR}" uv pip install \
        --python "${PY}" \
        torch \
        "numpy>=2.1,<2.5" \
        pyyaml
fi

# Always update 3dgs-io to latest (picks up bug fixes for SPZ encoding etc.)
echo "=== Updating 3dgs-io to latest ==="
VIRTUAL_ENV="${VENV_DIR}" uv pip install \
    --python "${PY}" \
    --reinstall --no-cache \
    "3dgs-io @ git+https://github.com/tier4/3dgs_io.git"

# --- Run export -------------------------------------------------------------
echo "=== Exporting to Cesium 3D Tiles ==="
echo "Started at: $(date)"
cd "${REPO_ROOT}"
"${PY}" export_cesium.py "$@"
echo "Completed at: $(date)"
