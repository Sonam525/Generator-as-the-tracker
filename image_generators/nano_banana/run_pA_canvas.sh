#!/usr/bin/env bash
# P-A Option-A memory-canvas baseline — PAID NB2 (working-agreement rule 6, README.md).
# Confirm spend before running; the driver also gates batches over ~$10 (--yes).
#
# On Databricks set the key from the secret scope, e.g. cluster env:
#   GEMINI_API_KEY={{secrets/gentrack/GEMINI_API_KEY}}
#
# Usage: bash image_generators/nano_banana/run_pA_canvas.sh <seq_dir> <out_dir> [--yes] [--max-frames N] ...
set -euo pipefail

: "${GEMINI_API_KEY:?set GEMINI_API_KEY (Databricks: GEMINI_API_KEY={{secrets/gentrack/GEMINI_API_KEY}})}"
SEQ="${1:?usage: run_pA_canvas.sh <seq_dir> <out_dir> [extra args...]}"
OUT="${2:?usage: run_pA_canvas.sh <seq_dir> <out_dir> [extra args...]}"
shift 2

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${HERE}/../../core:${PYTHONPATH:-}"

python -m gentrack.eval.run_pA --seq "$SEQ" --out "$OUT" "$@"
