#!/usr/bin/env bash
# Build a small zip for this personal short test project (no venv, no datasets).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/submission.zip}"
cd "$ROOT"
zip -r "$OUT" . \
  -x "*.git/*" \
  -x "*/.git/*" \
  -x ".venv/*" \
  -x "*/.venv/*" \
  -x "data/*" \
  -x "*/data/*" \
  -x "torchgeo_loveda_inspect/*" \
  -x ".cursor/*" \
  -x "*__pycache__/*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "*.zip"
echo "Created: $OUT"
echo "Size: $(du -h "$OUT" | cut -f1)"
