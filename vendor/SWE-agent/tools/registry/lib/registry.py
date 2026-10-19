import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


class EnvRegistry:


    def __init__(self, env_file: Optional[Path] = None):
        self._env_file = env_file

    @property
    def env_file(self) -> Path:
        if self._env_file is None:
            env_file = Path(os.environ.get("SWE_AGENT_ENV_FILE", "/root/.swe-agent-env"))
        else:
            env_file = self._env_file
        if not env_file.exists():
            env_file.write_text("{}")
        return env_file

    def __getitem__(self, key: str) -> str:
        return json.loads(self.env_file.read_text())[key]

    def get(self, key: str, default_value: Any = None, fallback_to_env: bool = True) -> Any:

        if fallback_to_env and key in os.environ:
            default_value = os.environ[key]
        return json.loads(self.env_file.read_text()).get(key, default_value)

    def get_if_none(self, value: Any, key: str, default_value: Any = None) -> Any:
        if value is not None:
            return value
        return self.get(key, default_value)

    def __setitem__(self, key: str, value: Any):
        env = json.loads(self.env_file.read_text())
        env[key] = value
        self.env_file.write_text(json.dumps(env))


registry = EnvRegistry()
