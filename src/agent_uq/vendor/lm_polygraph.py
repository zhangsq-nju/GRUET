from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LM_POLYGRAPH_ROOT: pathlib.Path = _REPO_ROOT / "vendor" / "lm-polygraph"
LM_POLYGRAPH_SRC: pathlib.Path = LM_POLYGRAPH_ROOT / "src"


def _module_from_vendored_tree(module: object) -> bool:
    path = getattr(module, "__file__", None)
    if not path:
        return False
    try:
        pathlib.Path(str(path)).resolve().relative_to(LM_POLYGRAPH_SRC.resolve())
    except ValueError:
        return False
    return True


def ensure() -> pathlib.Path:

    if not LM_POLYGRAPH_SRC.is_dir():
        raise FileNotFoundError(
            f"lm-polygraph vendored tree is missing its src directory: {LM_POLYGRAPH_SRC}"
        )
    package_dir = LM_POLYGRAPH_SRC / "lm_polygraph"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"lm-polygraph vendored tree is missing its package directory: {package_dir}"
        )

    path_str = str(LM_POLYGRAPH_SRC)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    loaded = sys.modules.get("lm_polygraph")
    if loaded is not None and not _module_from_vendored_tree(loaded):
        for name in list(sys.modules):
            if name == "lm_polygraph" or name.startswith("lm_polygraph."):
                sys.modules.pop(name, None)

    return LM_POLYGRAPH_SRC
