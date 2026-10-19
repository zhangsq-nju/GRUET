from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.types import AgentRunResult


class RunHook:


    def on_init(self, *, run):
        pass

    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_instance_start(
        self, *, index: int, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig
    ):
        pass

    def on_instance_skipped(
        self,
    ):
        pass

    def on_instance_completed(self, *, result: AgentRunResult):
        pass


class CombinedRunHooks(RunHook):
    def __init__(self):
        self._hooks = []

    def add_hook(self, hook: RunHook) -> None:
        self._hooks.append(hook)

    @property
    def hooks(self) -> list[RunHook]:
        return self._hooks

    def on_init(self, *, run):
        for hook in self._hooks:
            hook.on_init(run=run)

    def on_start(self):
        for hook in self._hooks:
            hook.on_start()

    def on_end(self):
        for hook in self._hooks:
            hook.on_end()

    def on_instance_start(
        self, *, index: int, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig
    ):
        for hook in self._hooks:
            hook.on_instance_start(index=index, env=env, problem_statement=problem_statement)

    def on_instance_skipped(self):
        for hook in self._hooks:
            hook.on_instance_skipped()

    def on_instance_completed(self, *, result: AgentRunResult):
        for hook in self._hooks:
            hook.on_instance_completed(result=result)
