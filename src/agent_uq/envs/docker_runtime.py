from __future__ import annotations

import json
import pathlib
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field


class DockerRuntimeError(RuntimeError):
    pass


def _ensure_docker_cli() -> None:
    if shutil.which("docker") is None:
        raise DockerRuntimeError(
            "Docker CLI not found on PATH; install Docker before using the "
            "docker-backed env adapters"
        )


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ContainerState:
    status: str
    running: bool
    exit_code: int | None
    error: str = ""


def run_command(
    args: list[str],
    *,
    timeout_s: float,
    input_text: str | None = None,
    check: bool = False,
) -> CommandResult:

    completed = subprocess.run(
        args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    result = CommandResult(
        args=list(args),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and result.returncode != 0:
        rendered = " ".join(args)
        stderr = result.stderr.strip() or result.stdout.strip() or "<no stderr>"
        raise DockerRuntimeError(f"Command failed ({rendered}): {stderr}")
    return result


def docker_image_exists(image_name: str) -> bool:

    _ensure_docker_cli()
    result = run_command(
        ["docker", "image", "inspect", image_name],
        timeout_s=10.0,
        check=False,
    )
    return result.returncode == 0


def remove_container(container_name: str) -> None:

    _ensure_docker_cli()
    run_command(
        ["docker", "rm", "-f", container_name],
        timeout_s=15.0,
        check=False,
    )


def run_ephemeral_container(
    image_name: str,
    *,
    command: list[str],
    timeout_s: float,
    input_text: str | None = None,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
) -> CommandResult:

    _ensure_docker_cli()
    args = ["docker", "run", "--rm"]
    if input_text is not None:
        args.append("-i")
    if workdir:
        args.extend(["-w", workdir])
    for key, value in sorted((environment or {}).items()):
        args.extend(["-e", f"{key}={value}"])
    args.append(image_name)
    args.extend(command)
    return run_command(args, timeout_s=timeout_s, input_text=input_text)


@dataclass
class PersistentDockerContainer:


    image_name: str
    name_prefix: str
    create_timeout_s: float = 60.0
    environment: dict[str, str] = field(default_factory=dict)
    volumes: list[tuple[str, str, bool]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    extra_run_args: list[str] = field(default_factory=list)
    container_name: str | None = None

    def start(self) -> str:

        _ensure_docker_cli()
        if self.container_name is not None:
            return self.container_name

        container_name = f"{self.name_prefix}-{uuid.uuid4().hex[:8]}"
        args = ["docker", "run", "-d", "--name", container_name]
        for key, value in sorted(self.environment.items()):
            args.extend(["-e", f"{key}={value}"])
        for host_path, container_path, read_only in self.volumes:
            volume = f"{host_path}:{container_path}"
            if read_only:
                volume += ":ro"
            args.extend(["-v", volume])
        args.extend(self.extra_run_args)
        args.append(self.image_name)
        args.extend(self.command)
        run_command(args, timeout_s=self.create_timeout_s, check=True)
        self.container_name = container_name
        return container_name

    def inspect_state(self) -> ContainerState:

        if self.container_name is None:
            raise DockerRuntimeError("docker inspect requested before container start")
        _ensure_docker_cli()
        result = run_command(
            ["docker", "inspect", "--format", "{{json .State}}", self.container_name],
            timeout_s=10.0,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return ContainerState(
                status="missing",
                running=False,
                exit_code=None,
                error=detail or "docker inspect failed",
            )
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise DockerRuntimeError(
                f"Failed to parse docker inspect state for {self.container_name}: {exc}"
            ) from exc
        status = str(payload.get("Status") or "unknown")
        exit_code_raw = payload.get("ExitCode")
        exit_code = int(exit_code_raw) if isinstance(exit_code_raw, int) else None
        return ContainerState(
            status=status,
            running=bool(payload.get("Running")),
            exit_code=exit_code,
            error=str(payload.get("Error") or ""),
        )

    def logs_tail(self, *, lines: int) -> str:

        if self.container_name is None:
            raise DockerRuntimeError("docker logs requested before container start")
        _ensure_docker_cli()
        result = run_command(
            ["docker", "logs", "--tail", str(int(lines)), self.container_name],
            timeout_s=10.0,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return f"<docker logs unavailable: {detail or 'unknown error'}>"
        combined = "".join(part for part in [result.stdout, result.stderr] if part)
        return combined.strip()

    def exec(
        self,
        command: list[str],
        *,
        timeout_s: float,
        input_text: str | None = None,
    ) -> CommandResult:

        if self.container_name is None:
            raise DockerRuntimeError("docker exec requested before container start")
        args = ["docker", "exec"]
        if input_text is not None:
            args.append("-i")
        args.append(self.container_name)
        args.extend(command)
        return run_command(args, timeout_s=timeout_s, input_text=input_text)

    def stop(self) -> None:

        if self.container_name is None:
            return
        remove_container(self.container_name)
        self.container_name = None


def wait_for_port(host: str, port: int, *, timeout_s: float) -> None:

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise DockerRuntimeError(
        f"Timed out after {timeout_s:.1f}s waiting for TCP {host}:{port}"
    )


def resolve_path(path: str) -> str:

    return str(pathlib.Path(path).resolve())
