#!/bin/bash
set -euxo pipefail

python -m pip install --upgrade \
    black \
    ipykernel \
    isort \
    jupyterlab \
    netron \
    papermill[s3] \
    pyproject-flake8 \
    scipy \
    tqdm

