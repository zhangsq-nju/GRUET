from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class HFLocalFallbackLoadError(RuntimeError):
    pass


def _extra_special_token_name(value: Any, index: int) -> str:
    if isinstance(value, str) and value.startswith("<|") and value.endswith("|>"):
        name = value[2:-2].strip().replace("-", "_").replace(" ", "_")
        if name:
            return f"{name}_token"
    return f"extra_special_token_{index}"


def install_transformers_extra_special_tokens_list_compat() -> None:

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    original = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(original, "_agent_uq_extra_special_tokens_list_compat", False):
        return

    def patched_set_model_specific_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {
                _extra_special_token_name(value, index): value
                for index, value in enumerate(special_tokens)
            }
        return original(self, special_tokens)

    patched_set_model_specific_special_tokens._agent_uq_extra_special_tokens_list_compat = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = (
        patched_set_model_specific_special_tokens
    )


def _is_existing_local_ref(model_ref: str) -> bool:
    return pathlib.Path(model_ref).expanduser().exists()


def configure_hf_home_cache_env(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    hf_home = target.get("HF_HOME")
    if not hf_home:
        return target
    hub_cache = target.get("HF_HUB_CACHE") or target.get("HUGGINGFACE_HUB_CACHE")
    if not hub_cache:
        hub_cache = str(pathlib.Path(hf_home).expanduser() / "hub")
    target.setdefault("HF_HUB_CACHE", hub_cache)
    target.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    return target


def _hf_home_cache_dirs(model_ref: str) -> list[pathlib.Path]:
    if _is_existing_local_ref(model_ref):
        return []
    repo_cache_name = f"models--{model_ref.replace('/', '--')}"
    candidates: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    configured = configure_hf_home_cache_env()
    roots: list[str | None] = []
    hf_home = configured.get("HF_HOME")
    if hf_home:
        roots.extend([str(pathlib.Path(hf_home).expanduser() / "hub"), hf_home])
    roots.extend(
        [
            configured.get("HF_HUB_CACHE"),
            configured.get("HUGGINGFACE_HUB_CACHE"),
        ]
    )
    for raw_root in roots:
        if not raw_root:
            continue
        cache_dir = pathlib.Path(raw_root).expanduser()
        if cache_dir in seen:
            continue
        seen.add(cache_dir)
        if (cache_dir / repo_cache_name).is_dir():
            candidates.append(cache_dir)
    return candidates


def _snapshot_has_required_files(
    snapshot: pathlib.Path,
    required_files: Sequence[str],
) -> bool:
    return all((snapshot / required_file).is_file() for required_file in required_files)


def resolve_hf_home_snapshot_path(
    model_ref: str,
    *,
    required_files: Sequence[str] = (),
) -> str | None:
    if _is_existing_local_ref(model_ref):
        return str(pathlib.Path(model_ref).expanduser())
    repo_cache_name = f"models--{model_ref.replace('/', '--')}"
    for cache_dir in _hf_home_cache_dirs(model_ref):
        repo_dir = cache_dir / repo_cache_name
        refs_main = repo_dir / "refs" / "main"
        if refs_main.is_file():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.is_dir() and _snapshot_has_required_files(snapshot, required_files):
                return str(snapshot)
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                for snapshot in snapshots:
                    if _snapshot_has_required_files(snapshot, required_files):
                        return str(snapshot)
    return None


def resolve_hf_home_model_ref(
    model_ref: str,
    *,
    required_files: Sequence[str] = (),
) -> str:
    return resolve_hf_home_snapshot_path(model_ref, required_files=required_files) or model_ref


def load_with_hf_home_fallback(
    loader: Callable[..., T],
    model_ref: str,
    *args: Any,
    cache_kwarg: str = "cache_dir",
    **kwargs: Any,
) -> T:

    install_transformers_extra_special_tokens_list_compat()
    local_ref = resolve_hf_home_snapshot_path(model_ref)
    if local_ref is not None:
        local_kwargs = dict(kwargs)
        local_kwargs.setdefault("local_files_only", True)
        try:
            return loader(local_ref, *args, **local_kwargs)
        except Exception as local_exc:
            try:
                return loader(model_ref, *args, **kwargs)
            except Exception as remote_exc:
                raise HFLocalFallbackLoadError(
                    f"failed to load Hugging Face model {model_ref!r} "
                    f"from local HF_HOME snapshot {local_ref!r}: {local_exc}; "
                    f"remote load also failed: {remote_exc}"
                ) from local_exc
    try:
        return loader(model_ref, *args, **kwargs)
    except Exception as remote_exc:
        local_errors: list[str] = []
        fallback_kwargs: list[dict[str, Any]] = []
        if cache_kwarg in kwargs:
            fallback_kwargs.append({"local_files_only": True})
        for cache_dir in _hf_home_cache_dirs(model_ref):
            fallback_kwargs.append(
                {
                    cache_kwarg: str(cache_dir),
                    "local_files_only": True,
                }
            )
        if not fallback_kwargs and os.environ.get("HF_HOME"):
            fallback_kwargs.append({"local_files_only": True})
        for local_kwargs in fallback_kwargs:
            merged_kwargs = dict(kwargs)
            merged_kwargs.update(local_kwargs)
            try:
                return loader(model_ref, *args, **merged_kwargs)
            except Exception as local_exc:
                local_errors.append(f"{local_kwargs}: {local_exc}")
        local_detail = (
            "; local HF_HOME fallback errors: " + " | ".join(local_errors)
            if local_errors
            else "; no HF_HOME local cache fallback was available"
        )
        raise HFLocalFallbackLoadError(
            f"failed to load Hugging Face model {model_ref!r}: {remote_exc}{local_detail}"
        ) from remote_exc
