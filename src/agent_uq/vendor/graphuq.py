from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GRAPHUQ_ROOT: pathlib.Path = _REPO_ROOT / "vendor" / "GUT-Q"


def ensure() -> pathlib.Path:
    if not GRAPHUQ_ROOT.is_dir():
        raise FileNotFoundError(
            f"Vendored graphUQ not found at {GRAPHUQ_ROOT}. "
            "Expected vendor/GUT-Q/ next to src/."
        )
    src_dir = GRAPHUQ_ROOT / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"graphUQ vendored tree is missing its src/ directory: {src_dir}"
        )
    path_str = str(GRAPHUQ_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return GRAPHUQ_ROOT
