#!/usr/bin/env bash

bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH="$bundle_dir/lib":$PYTHONPATH

_write_env "WINDOW" "${WINDOW:-100}"
_write_env "OVERLAP" "${OVERLAP:-2}"
_write_env "FIRST_LINE" "${FIRST_LINE:-0}"
_write_env "CURRENT_FILE" "${CURRENT_FILE:-}"
