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
#   # Export and open the Cesium viewer:
#   ./export_cesium.sh --config configs/example/t4_train_example.yaml --background-only --view
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

# --- Parse shell-level options (--view, --port) before passing rest to Python -
VIEW=false
VIEW_PORT=8080
EXPORT_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --view)
            VIEW=true
            shift
            ;;
        --port)
            VIEW_PORT="$2"
            shift 2
            ;;
        *)
            EXPORT_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}"

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

if ! "${PY}" -c "import torch, numpy, yaml; import importlib; m = importlib.import_module('3dgs_io'); m.save_tileset; m.TilesetSaveOptions; import spz" >/dev/null 2>&1; then
    echo "=== Installing export dependencies into ${VENV_DIR} ==="
    VIRTUAL_ENV="${VENV_DIR}" uv pip install \
        --python "${PY}" \
        torch \
        "numpy>=2.1,<2.5" \
        pyyaml
fi

# Always update 3dgs-io to latest (picks up bug fixes for SPZ encoding,
# 3D Tiles writer, etc.). The rewritten 3dgs-io exposes save_tileset() /
# TilesetSaveOptions, which export_cesium.py now uses to emit chunked
# GLBs + tileset.json in one call.
echo "=== Updating 3dgs-io to latest ==="
VIRTUAL_ENV="${VENV_DIR}" uv pip install \
    --python "${PY}" \
    --reinstall --no-cache \
    "3dgs-io @ git+https://github.com/tier4/3dgs_io.git"

# --- Run export -------------------------------------------------------------
echo "=== Exporting to Cesium 3D Tiles ==="
echo "Started at: $(date)"
cd "${REPO_ROOT}"

# Capture output to extract the output directory for --view
EXPORT_OUTPUT=$("${PY}" export_cesium.py "$@" 2>&1 | tee /dev/stderr)
echo "Completed at: $(date)"

# --- Launch viewer if --view was specified ----------------------------------
if $VIEW; then
    OUTPUT_DIR=$(echo "$EXPORT_OUTPUT" | grep -oP '(?<=Output directory: ).*')
    if [ -z "$OUTPUT_DIR" ]; then
        echo "Error: could not detect output directory from export output" >&2
        exit 1
    fi

    echo ""
    echo "=== Starting Cesium viewer ==="
    echo "  Tiles: ${OUTPUT_DIR}"
    echo "  URL:   http://localhost:${VIEW_PORT}/"
    echo "  Press Ctrl+C to stop."

    # Open browser (best-effort, non-blocking)
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:${VIEW_PORT}/" 2>/dev/null &
    elif command -v open >/dev/null 2>&1; then
        open "http://localhost:${VIEW_PORT}/" &
    fi

    python3 "${REPO_ROOT}/viewer/serve.py" --tiles "${OUTPUT_DIR}" --port "${VIEW_PORT}"
fi
