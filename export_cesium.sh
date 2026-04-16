#!/bin/bash
# Export a VAD-GS checkpoint (.pth) to Cesium 3D Tiles in one shot.
#
# Usage:
#   ./export_cesium.sh <checkpoint.pth> [extra args for export_cesium.py]
#
# Examples:
#   ./export_cesium.sh output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
#       --t4-dataset 835afe23-ff50-4883-a0b2-421e101a124b --background-only
#
#   ./export_cesium.sh output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
#       --lat 35.6812 --lon 139.7671 --output my_tiles
#
#   ./export_cesium.sh output/t4_exp/t4_scene_000/trained_model/iteration_30000.pth \
#       --background-only --spz-compression
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

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint.pth> [extra args...]" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  $0 output/nuscenes_exp/nuscenes_val_000_3cam/trained_model/iteration_30000.pth" >&2
    exit 1
fi

CHECKPOINT="$1"
shift

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

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

if ! "${PY}" -c "import torch, numpy; import importlib; importlib.import_module('3dgs_io'); import spz" >/dev/null 2>&1; then
    echo "=== Installing export dependencies into ${VENV_DIR} ==="
    # Use uv pip with the venv's python. 3dgs-io brings in spz as a git dep.
    VIRTUAL_ENV="${VENV_DIR}" uv pip install \
        --python "${PY}" \
        torch \
        "numpy>=2.1,<2.5" \
        "3dgs-io @ git+https://github.com/tier4/3dgs_io.git"
fi

# --- Run export -------------------------------------------------------------
echo "=== Exporting ${CHECKPOINT} to Cesium 3D Tiles ==="
echo "Started at: $(date)"
cd "${REPO_ROOT}"
"${PY}" export_cesium.py "${CHECKPOINT}" "$@"
echo "Completed at: $(date)"
