#!/bin/bash
set -euxo pipefail

python -m pip install --upgrade \
    onnx==1.12.0 \
    onnxruntime==1.11.0

