from __future__ import annotations

import os


def validate_tensor_parallel_size(tensor_parallel_size: int) -> int:

    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int):
        raise ValueError(
            "model.tensor_parallel_size must be a positive integer; "
            f"got {type(tensor_parallel_size).__name__} {tensor_parallel_size!r}"
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            "model.tensor_parallel_size must be >= 1; "
            f"got {tensor_parallel_size}"
        )
    if tensor_parallel_size == 1:
        return 1

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "model.tensor_parallel_size > 1 requires torch CUDA inspection, "
            f"but importing torch failed: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "model.tensor_parallel_size > 1 requires CUDA, but torch.cuda.is_available() is false"
        )
    visible_devices = int(torch.cuda.device_count())
    if visible_devices < tensor_parallel_size:
        raise RuntimeError(
            "model.tensor_parallel_size exceeds the number of visible CUDA devices: "
            f"requested {tensor_parallel_size}, visible {visible_devices}"
        )
    return int(tensor_parallel_size)


def configure_vllm_worker_multiproc_method(tensor_parallel_size: int) -> None:

    if int(tensor_parallel_size) <= 1:
        return
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
