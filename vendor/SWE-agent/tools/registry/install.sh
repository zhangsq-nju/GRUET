#!/usr/bin/env bash

bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH="$bundle_dir/lib":$PYTHONPATH