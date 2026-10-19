_write_env "CURRENT_FILE" "${CURRENT_FILE:-}"
_write_env "CURRENT_LINE" "${CURRENT_LINE:-0}"
_write_env "WINDOW" "$WINDOW"

pip install \
  --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --trusted-host "${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}" \
  flake8
