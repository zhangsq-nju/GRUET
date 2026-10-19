from sweagent.environment.repo import Repo, RepoConfig


class EnvHook:


    def on_init(self, *, env) -> None:
        pass

    def on_copy_repo_started(self, repo: RepoConfig | Repo) -> None:
        pass

    def on_start_deployment(self) -> None:
        pass

    def on_install_env_started(self) -> None:
        pass

    def on_close(self):
        pass

    def on_environment_startup(self) -> None:
        pass


class CombinedEnvHooks(EnvHook):
    def __init__(self):
        self._hooks = []

    def add_hook(self, hook: EnvHook) -> None:
        self._hooks.append(hook)

    def on_init(self, *, env) -> None:
        for hook in self._hooks:
            hook.on_init(env=env)

    def on_copy_repo_started(self, repo: RepoConfig | Repo) -> None:
        for hook in self._hooks:
            hook.on_copy_repo_started(repo=repo)

    def on_start_deployment(self) -> None:
        for hook in self._hooks:
            hook.on_start_deployment()

    def on_install_env_started(self) -> None:
        for hook in self._hooks:
            hook.on_install_env_started()

    def on_close(self):
        for hook in self._hooks:
            hook.on_close()

    def on_environment_startup(self) -> None:
        for hook in self._hooks:
            hook.on_environment_startup()
