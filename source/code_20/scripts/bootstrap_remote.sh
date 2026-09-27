#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -i "$PIP_INDEX_URL" --upgrade pip

python -m pip install -i "$PIP_INDEX_URL" "numpy>=1.26,<2"

if [[ -d "$ROOT_DIR/vendor/transformers" ]]; then
  python -m pip install -i "$PIP_INDEX_URL" -e "$ROOT_DIR/vendor/transformers"
else
  python -m pip install -i "$PIP_INDEX_URL" "transformers>=4.41,<5"
fi

python -m pip install -i "$PIP_INDEX_URL" -e .

echo "[ok] environment ready in $ROOT_DIR/.venv"
echo "[ok] pip index: $PIP_INDEX_URL"
if [[ -d "$ROOT_DIR/vendor/transformers" ]]; then
  echo "[ok] using editable transformers at $ROOT_DIR/vendor/transformers"
else
  echo "[ok] using pip transformers"
fi
