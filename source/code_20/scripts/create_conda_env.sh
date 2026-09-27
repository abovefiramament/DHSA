#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-screscomp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cat > "$HOME/.condarc" <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
show_channel_urls: true
channel_priority: strict
EOF

conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" pip -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip config set global.index-url "$PIP_INDEX_URL"
python -m pip install -i "$PIP_INDEX_URL" -U "pip" "setuptools" "wheel"

python -m pip install -i "$PIP_INDEX_URL" \
  "numpy>=1.26,<2" \
  "torch>=2.3" \
  "accelerate>=0.30" \
  "safetensors>=0.4" \
  "sentencepiece>=0.2" \
  "protobuf>=3.20,<6" \
  "huggingface_hub>=0.34,<1"

if [[ -d "$ROOT_DIR/vendor/transformers" ]]; then
  python -m pip install -i "$PIP_INDEX_URL" -e "$ROOT_DIR/vendor/transformers"
else
  python -m pip install -i "$PIP_INDEX_URL" "transformers>=4.41,<5"
fi

python -m pip install -i "$PIP_INDEX_URL" -e "$ROOT_DIR"

python - <<'PY'
import torch
import transformers
import numpy
print("python env OK")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, transformers.__file__)
print("numpy", numpy.__version__)
PY
