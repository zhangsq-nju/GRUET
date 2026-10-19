from pathlib import Path
from typing import Self

from sweagent.agent.agents import DefaultAgent, ShellAgentConfig
from sweagent.agent.models import HumanModel, HumanModelConfig, get_model
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.parsing import ActionOnlyParser
from sweagent.tools.tools import ToolHandler
from sweagent.types import AgentRunResult, StepOutput


class ShellAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: ShellAgentConfig) -> Self:


        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            max_steps=config.max_steps,
        )

    def human_step_in(self) -> None:

        self._original_model = self.model
        self._original_parser = self.tools.config.parse_function

        human_config = HumanModelConfig(name="human", catch_eof=False)
        self.model = get_model(human_config, self.tools.config)
        self.tools.config.parse_function = ActionOnlyParser()

        self.logger.info("Switched to human mode. Agent will now accept human input. Press ^D to switch back.")

    def human_step_out(self) -> None:

        if not hasattr(self, "_original_model") or self._original_model is None:
            self.logger.info("No previous model to switch back to. Remaining in current mode.")
            return

        self.model = self._original_model
        self.tools.config.parse_function = self._original_parser
        self._original_model = None
        self._original_parser = None

        self.logger.info("Switched back to AI model mode.")

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        *,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:

        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)


        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            try:
                step_output = self.step()
                self.save_trajectory()
            except KeyboardInterrupt:
                if not isinstance(self.model, HumanModel):
                    self.human_step_in()
                    continue
                raise
            except EOFError:

                self.logger.info("Detected ^D - switching back to AI mode")
                self.human_step_out()
                continue
            if step_output.done and not isinstance(self.model, HumanModel):
                if "exit_max_steps" in str(step_output.exit_status or ""):
                    continue

                self.logger.info("Robot is done! Please submit the solution.")
                self.human_step_in()
                step_output.done = False
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)


        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])
