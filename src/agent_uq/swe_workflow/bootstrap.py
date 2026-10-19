from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from agent_uq.swe_workflow.common import require_path


def materialize_bootstrap_rewritten_instances(
    *,
    selected_instances: Sequence[Mapping[str, Any]],
    bootstrap_cfg: Any,
    launch_python_executable: str,
    runtime_dir: str | pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    runtime_dir = pathlib.Path(runtime_dir)
    if shutil.which("docker") is None:
        raise FileNotFoundError("docker is required for bootstrap strategy 'local_swe_rex_rewrite'")

    wheelhouse_dir = require_path(bootstrap_cfg.wheelhouse_dir, field_name="bootstrap.wheelhouse_dir")
    swerex_version = ensure_swe_rex_wheelhouse(
        python_executable=launch_python_executable,
        wheelhouse_dir=wheelhouse_dir,
        host_pip_index_url=str(bootstrap_cfg.host_pip_index_url),
        host_pip_trusted_host=str(bootstrap_cfg.host_pip_trusted_host),
    )

    unique_base_images: list[str] = []
    for instance in selected_instances:
        image_name = str(instance["image_name"])
        if image_name not in unique_base_images:
            unique_base_images.append(image_name)

    image_map: dict[str, str] = {}
    for base_image in unique_base_images:
        suffix = hashlib.sha256(base_image.encode("utf-8")).hexdigest()[:16]
        derived_image = f"{str(bootstrap_cfg.local_image_prefix)}:{suffix}"
        local_base_image = f"{str(bootstrap_cfg.local_image_prefix)}-base:{suffix}"
        image_map[base_image] = derived_image
        if _docker_image_exists(derived_image):
            continue
        _build_bootstrap_image(
            base_image=base_image,
            local_base_image=local_base_image,
            derived_image=derived_image,
            swerex_version=swerex_version,
            wheelhouse_dir=wheelhouse_dir,
            docker_platform=str(bootstrap_cfg.docker_platform),
        )

    patched_instances: list[dict[str, Any]] = []
    for instance in selected_instances:
        patched = dict(instance)
        patched["source_image_name"] = str(instance["image_name"])
        patched["image_name"] = image_map[str(instance["image_name"])]
        patched_instances.append(patched)

    patched_instances_path = runtime_dir / "patched_instances.json"
    image_map_path = runtime_dir / "image_map.json"
    patched_instances_path.write_text(
        json.dumps(patched_instances, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    image_map_path.write_text(
        json.dumps(image_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return patched_instances_path, image_map_path


def ensure_swe_rex_wheelhouse(
    *,
    python_executable: str,
    wheelhouse_dir: str | pathlib.Path,
    host_pip_index_url: str,
    host_pip_trusted_host: str,
) -> str:
    wheelhouse_dir = pathlib.Path(wheelhouse_dir)
    version_cmd = [
        str(python_executable),
        "-c",
        "from importlib.metadata import version; print(version('swe-rex'))",
    ]
    version_result = subprocess.run(version_cmd, capture_output=True, text=True, check=True)
    swerex_version = version_result.stdout.strip()
    ready_marker = wheelhouse_dir / f".ready-{swerex_version}"
    if ready_marker.is_file():
        return swerex_version

    if wheelhouse_dir.exists():
        shutil.rmtree(wheelhouse_dir)
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)

    download_cmd = [
        str(python_executable),
        "-m",
        "pip",
        "download",
        "--index-url",
        host_pip_index_url,
    ]
    if host_pip_trusted_host:
        download_cmd.extend(["--trusted-host", host_pip_trusted_host])
    download_cmd.extend(
        [
            "--dest",
            str(wheelhouse_dir),
            f"swe-rex=={swerex_version}",
        ]
    )
    subprocess.run(download_cmd, check=True)
    ready_marker.touch()
    return swerex_version


def _docker_image_exists(image_name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _run_subprocess_with_retries(
    cmd: Sequence[str],
    *,
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(list(cmd), check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(retry_delay_seconds * attempt)


def _ensure_local_tagged_base_image(
    *,
    base_image: str,
    local_base_image: str,
    docker_platform: str,
) -> None:
    if _docker_image_exists(local_base_image):
        return

    if not _docker_image_exists(base_image):
        pull_cmd = ["docker", "pull"]
        if docker_platform:
            pull_cmd.extend(["--platform", docker_platform])
        pull_cmd.append(base_image)
        _run_subprocess_with_retries(pull_cmd)

    subprocess.run(["docker", "tag", base_image, local_base_image], check=True)


def _build_bootstrap_image(
    *,
    base_image: str,
    local_base_image: str,
    derived_image: str,
    swerex_version: str,
    wheelhouse_dir: str | pathlib.Path,
    docker_platform: str,
) -> None:
    _ensure_local_tagged_base_image(
        base_image=base_image,
        local_base_image=local_base_image,
        docker_platform=docker_platform,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = pathlib.Path(tmpdir)
        shutil.copytree(pathlib.Path(wheelhouse_dir), build_dir / "wheelhouse")
        dockerfile = "\n".join(
            [

                f"FROM {local_base_image}",
                "COPY wheelhouse /tmp/wheelhouse",
                f"RUN python3 -m pip install --no-index --find-links=/tmp/wheelhouse swe-rex=={swerex_version}",
                "RUN command -v swerex-remote",
                "RUN swerex-remote --version",
                "",
            ]
        )
        (build_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        cmd = ["docker", "build", "--pull=false"]
        if docker_platform:
            cmd.extend(["--platform", docker_platform])
        cmd.extend(["-t", derived_image, str(build_dir)])
        subprocess.run(cmd, check=True)


__all__ = [
    "materialize_bootstrap_rewritten_instances",
    "ensure_swe_rex_wheelhouse",
    "_docker_image_exists",
    "_run_subprocess_with_retries",
    "_ensure_local_tagged_base_image",
    "_build_bootstrap_image",
]
