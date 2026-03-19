#!/usr/bin/env bash
# scripts/conda_build.sh — Local conda package build helper
#
# Usage:
#   bash scripts/conda_build.sh          # Build only
#   bash scripts/conda_build.sh --test   # Build + run tests
#   bash scripts/conda_build.sh --install # Build + install into current env
#
# Prerequisites:
#   conda install conda-build conda-verify

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/conda-bld"

echo "=== UMCP Conda Package Builder ==="
echo "Repository: ${REPO_ROOT}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

# Check conda-build is available
if ! command -v conda-build &>/dev/null; then
    echo "ERROR: conda-build not found. Install with:"
    echo "  conda install conda-build conda-verify"
    exit 1
fi

# Build
echo "--- Building conda package ---"
conda build "${REPO_ROOT}/conda.recipe" \
    --output-folder "${OUTPUT_DIR}" \
    --no-anaconda-upload

echo ""
echo "--- Built packages ---"
find "${OUTPUT_DIR}" -name "*.tar.bz2" -o -name "*.conda" | sort

# Optional --test flag
if [[ "${1:-}" == "--test" ]]; then
    echo ""
    echo "--- Running package tests ---"
    conda build "${REPO_ROOT}/conda.recipe" --test
fi

# Optional --install flag
if [[ "${1:-}" == "--install" ]]; then
    PKG=$(conda build "${REPO_ROOT}/conda.recipe" --output 2>/dev/null | head -1)
    if [[ -f "${PKG}" ]]; then
        echo ""
        echo "--- Installing ${PKG} ---"
        conda install --use-local "${PKG}" -y
    else
        echo "ERROR: Could not find built package at ${PKG}"
        exit 1
    fi
fi

echo ""
echo "=== Done ==="
