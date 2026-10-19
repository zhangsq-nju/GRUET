from __future__ import annotations

import pathlib


def resolve_graph_artifact_dir(
    out_dir: pathlib.Path,
    *,
    use_graph_artifact_cache: bool,
    configured_graph_artifact_dir: str | pathlib.Path | None = None,
) -> pathlib.Path | None:

    if not use_graph_artifact_cache:
        return None
    if configured_graph_artifact_dir:
        return pathlib.Path(str(configured_graph_artifact_dir))
    return out_dir / "graphs"


def resolve_pairwise_nli_artifact_dir(
    out_dir: pathlib.Path,
    *,
    use_pairwise_nli_artifact_cache: bool,
) -> pathlib.Path | None:

    if not use_pairwise_nli_artifact_cache:
        return None
    return out_dir / "nli_pairs"
