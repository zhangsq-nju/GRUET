#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

conda create -n gruet python=3.11.14 pip -y
conda run --no-capture-output -n gruet python -m pip install -r requirements-qwen.txt
conda run --no-capture-output -n gruet python -m pip install -e . --no-deps
conda create -n gruet-gemma python=3.11.14 pip -y
conda run --no-capture-output -n gruet-gemma python -m pip install -r requirements-qwen.txt
conda run --no-capture-output -n gruet-gemma python -m pip install --no-deps -r requirements-gemma.txt
conda run --no-capture-output -n gruet-gemma python -m pip install -e . --no-deps
docker build -t intercode-python -f vendor/intercode/docker/python.Dockerfile vendor/intercode
docker build -t intercode-nl2bash --build-arg file_system_version=1 -f vendor/intercode/docker/nl2bash.Dockerfile vendor/intercode
docker pull mysql:latest
