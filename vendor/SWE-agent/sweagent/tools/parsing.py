import json
import re
import textwrap
from abc import ABC, abstractmethod
from shlex import quote
from textwrap import dedent
from typing import Any, Literal

from jinja2 import Template
from pydantic import BaseModel

from sweagent.exceptions import FormatError, FunctionCallingFormatError
from sweagent.tools.commands import Command
from sweagent.tools.utils import _should_quote


class AbstractParseFunction(ABC):


    error_message: str

    @abstractmethod
    def __call__(self, model_response, commands: list[Command], strict=False) -> tuple[str, str]:
        raise NotImplementedError

    @property
    def format_error_template(self):
        return textwrap.dedent(self.error_message)


class ActionParser(AbstractParseFunction, BaseModel):


    error_message: str = """\
    The command you provided was not recognized. Please specify one of the commands (+ any necessary arguments) from the following list in your response. Do not include any other text.

    COMMANDS:
    {command_docs}
    """

    type: Literal["action"] = "action"


    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        if model_response["message"].split():
            action = model_response["message"].strip().split()[0]
            if action in {command.name for command in commands}:
                return model_response["message"], model_response["message"]
        msg = "First word in model response is not a valid command."
        raise FormatError(msg)


class ActionOnlyParser(AbstractParseFunction, BaseModel):


    error_message: str = "No message found in model response."

    type: Literal["action_only"] = "action_only"


    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        return "", model_response["message"]


class ThoughtActionParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    Your output was not formatted correctly. You must always include one reasoning and one command as part of your response. Make sure you do not have multiple reasoning/command tags.
    Please make sure your output precisely matches the following format:
    REASONING
    Discuss here with yourself about what your planning and what you're going to do in this step.

    ```
    command(s) that you're going to run
    ```
    """)

    type: Literal["thought_action"] = "thought_action"


    def __call__(self, model_response: dict, commands: list[Command], strict=False):

        last_valid_block = find_last_valid_code_block(model_response["message"])
        if last_valid_block:
            start, end = last_valid_block
            thought = model_response["message"][: start.start()] + model_response["message"][end.end() :]
            return thought, model_response["message"][start.end() : end.start()]
        msg = "No action found in model response."
        raise FormatError(msg)


_STEP_MARKER_RE = re.compile(r"(?m)^\s*Step\s+\d+:")


def has_step_marker(text: str | None) -> bool:
    if not text:
        return False
    return _STEP_MARKER_RE.search(text) is not None


def find_last_valid_code_block(message: str):
    code_block_pat = re.compile(r"^```(\S*)[ \t]*\n|^```[ \t]*$", re.MULTILINE)
    stack = []
    last_valid_block = None
    for match in code_block_pat.finditer(message):
        if stack and not match.group(1):
            start = stack.pop()
            if not stack:
                last_valid_block = (start, match)
        elif match.group(1) is not None:
            stack.append(match)
    return last_valid_block


def extract_thought_action_spans(message: str) -> dict[str, int]:
    last_valid_block = find_last_valid_code_block(message)
    if last_valid_block is None:
        msg = "No action found in model response."
        raise FormatError(msg)
    start, end = last_valid_block
    accepted_end = end.end()
    if accepted_end > end.start() and message[end.start() : accepted_end].endswith("\n"):
        accepted_end -= 1
    return {
        "accepted_start": 0,
        "accepted_end": accepted_end,
        "thought_start": 0,
        "thought_end": start.start(),
        "action_start": start.end(),
        "action_end": end.start(),
    }


class XMLThoughtActionParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    Your output was not formatted correctly. You must always include one reasoning and one command as part of your response. Make sure you do not have multiple reasoning/command tags.
    Please make sure your output precisely matches the following format:
    """)

    type: Literal["xml_thought_action"] = "xml_thought_action"


    def __call__(self, model_response: dict, commands: list[Command], strict=False) -> tuple[str, str]:

        if "<command>" not in model_response["message"] or "</command>" not in model_response["message"]:
            msg = "No action found in model response."
            raise FormatError(msg)

        start_action = model_response["message"].rfind("<command>") + len(
            "<command>"
        )
        end_thought = model_response["message"].rfind("<command>")
        end_action = model_response["message"].rfind("</command>")
        restart_thought = model_response["message"].rfind("</command>") + len(
            "</command>"
        )

        action = model_response["message"][start_action:end_action]
        thought = model_response["message"][:end_thought] + model_response["message"][restart_thought:]

        return thought.strip(), action.strip()


FN_REGEX_PATTERN = r"<function=([^>]+)>\n(.*?)</function>"
FN_PARAM_REGEX_PATTERN = r"<parameter=([^>]+)>(.*?)</parameter>"


class XMLFunctionCallingParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    {%- if error_code == "missing" -%}
    Your last output did not use any tool calls!
    Please make sure your output includes exactly _ONE_ function call!
    If you think you have already resolved the issue, please submit your changes by running the `submit` command.
    If you think you cannot solve the problem, please run `submit`.
    Else, please continue with a new tool call!
    {%- elif error_code == "multiple" -%}
    Your last output included multiple tool calls!
    Please make sure your output includes a thought and exactly _ONE_ function call.
    {%- elif error_code == "unexpected_arg" -%}
    Your action could not be parsed properly: {{exception_message}}.
    Make sure your function call doesn't include any extra arguments that are not in the allowed arguments, and only use the allowed commands.
    {%- else -%}
    Your action could not be parsed properly: {{exception_message}}.
    {% endif %}
    """)

    type: Literal["xml_function_calling"] = "xml_function_calling"

    def __call__(self, model_response: dict, commands: list[Command], strict=False) -> tuple[str, str]:
        fn_match = re.search(FN_REGEX_PATTERN, model_response["message"], re.DOTALL)
        if not fn_match:
            msg = "No function found in model response."
            raise FormatError(msg)
        fn_name = fn_match.group(1).strip()


        if fn_name == "execute_bash":
            fn_name = "bash"
        if fn_name == "finish":
            fn_name = "submit"

        fn_body = fn_match.group(2)
        thought = model_response["message"][: fn_match.start()] + model_response["message"][fn_match.end() :]
        thought = thought.strip()

        commands_dict = {c.name: c for c in commands}
        command = commands_dict.get(fn_name)
        if not command:
            msg = f"Command '{fn_name}' not found in list of available commands."
            raise FormatError(msg)

        params_dict = {
            param[0]: re.sub(r"^\n|\n$", "", param[1])
            for param in re.findall(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
        }

        if "view_range" in params_dict:

            v = params_dict["view_range"]
            if isinstance(v, str):
                if not re.match(r"\[\d+,\s*\d+\]", v):
                    msg = f"view_range must be in the format [<start>, <end>], got {v}."
                    raise FormatError(msg)
                params_dict["view_range"] = json.loads(v)


        required_args = {arg.name for arg in command.arguments if arg.required}
        missing_args = required_args - params_dict.keys()
        if missing_args:
            msg = f"Required argument(s) missing: {', '.join(missing_args)}"
            raise FormatError(msg)


        valid_args = {arg.name for arg in command.arguments}
        extra_args = set(params_dict.keys()) - valid_args
        if command.end_name:

            extra_args.discard(command.end_name)
        if extra_args:
            msg = f"Unexpected argument(s): {', '.join(extra_args)}"
            raise FormatError(msg)


        formatted_args = {
            arg.name: Template(arg.argument_format).render(
                value=quote(params_dict[arg.name])
                if _should_quote(params_dict[arg.name], command)
                else params_dict[arg.name]
            )
            if arg.name in params_dict
            else ""
            for arg in command.arguments
        }
        return thought, command.invoke_format.format(**formatted_args).strip()


class EditFormat(ThoughtActionParser, BaseModel):


    error_message: str = dedent("""\
    Your output was not formatted correctly. You must wrap the replacement text in backticks (```).
    Please make sure your output precisely matches the following format:
    COMMENTS
    You can write comments here about what you're going to do if you want.

    ```
    New window contents.
    Make sure you copy the entire contents of the window here, with the required indentation.
    Make the changes to the window above directly in this window.
    Remember that all of the window's contents will be replaced with the contents of this window.
    Don't include line numbers in your response.
    ```
    """)

    type: Literal["edit_format"] = "edit_format"


class Identity(AbstractParseFunction, BaseModel):


    error_message: str = """\
    It seems like something went wrong with your output. Please try again.
    """

    type: Literal["identity"] = "identity"


    def __call__(self, model_response: dict, commands: list[Command], strict=False) -> tuple[str, str]:

        return model_response["message"], model_response["message"]


class FunctionCallingParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    {%- if error_code == "missing" -%}
    Your last output did not use any tool calls!
    Please make sure your output includes exactly _ONE_ function call!
    You must invoke the function directly using the function call format.
    You cannot invoke commands with ```, you have to use the function call format.
    If you think you have already resolved the issue, please submit your changes by running the `submit` command.
    If you think you cannot solve the problem, please run `exit_forfeit` (if available) or `submit`.
    Else, please continue with a new tool call!
    {%- elif error_code == "multiple" -%}
    Your last output included multiple tool calls!
    Please make sure your output includes a thought and exactly _ONE_ function call.
    {%- elif error_code == "unexpected_arg" -%}
    Your action could not be parsed properly: {{exception_message}}.
    Make sure your function call doesn't include any extra arguments that are not in the allowed arguments, and only use the allowed commands.
    {%- else -%}
    Your action could not be parsed properly: {{exception_message}}.
    {% endif %}
    """)

    type: Literal["function_calling"] = "function_calling"


    def _parse_tool_call(self, tool_call: dict, commands: list[Command]):
        name = tool_call["function"]["name"]
        command = {c.name: c for c in commands}.get(name)
        if not command:
            msg = f"Command '{name}' not found in list of available commands."
            raise FunctionCallingFormatError(msg, "invalid_command")
        if not isinstance(tool_call["function"]["arguments"], dict):
            try:
                values = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                msg = "Tool call arguments are not valid JSON."
                raise FunctionCallingFormatError(msg, "invalid_json")
        required_args = {arg.name for arg in command.arguments if arg.required}
        missing_args = required_args - values.keys()
        if missing_args:
            msg = f"Required argument(s) missing: {', '.join(missing_args)}"
            raise FunctionCallingFormatError(msg, "missing_arg")
        valid_args = {arg.name for arg in command.arguments}
        extra_args = set(values.keys()) - valid_args
        if command.end_name:

            extra_args.discard(command.end_name)
        if extra_args:
            msg = f"Unexpected argument(s): {', '.join(extra_args)}"
            raise FunctionCallingFormatError(msg, "unexpected_arg")

        def get_quoted_arg(value: Any) -> str:
            if isinstance(value, str):
                return quote(value) if _should_quote(value, command) else value

            if value is None:
                return ""
            return value

        formatted_args = {
            arg.name: Template(arg.argument_format).render(value=get_quoted_arg(values[arg.name]))
            if arg.name in values
            else ""
            for arg in command.arguments
        }
        return command.invoke_format.format(**formatted_args).strip()

    def __call__(self, model_response: dict, commands: list[Command], strict=False):
        message = model_response["message"]
        tool_calls = model_response.get("tool_calls", None)
        if tool_calls is None or len(tool_calls) != 1:
            num_tools = len(tool_calls) if tool_calls else 0
            msg = (
                f"Expected exactly one tool call in model response - received {num_tools} "
                f"tool calls with message: {message}"
            )
            error_code = "missing" if num_tools == 0 else "multiple"
            raise FunctionCallingFormatError(msg, error_code, num_tools=num_tools)
        tool_call = tool_calls[0]
        action = self._parse_tool_call(tool_call, commands)
        return message, action


class JsonParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    Your output could not be parsed as JSON. Please make sure your output 1) is valid JSON and
    2) Includes the "thought" and "command" fields.

    """)

    type: Literal["json"] = "json"


    def __call__(self, model_response: dict, commands: list[Command], strict=False):

        try:
            data = json.loads(model_response["message"])
            if not isinstance(data, dict):
                msg = "Model output is not a JSON object."
                raise FormatError(msg)


            required_keys = ["thought", "command"]
            for key in required_keys:
                if key not in data:
                    msg = f"Key '{key}' is missing from model output."
                    raise FormatError(msg)


            data_command = data["command"]
            if not isinstance(data_command, dict):
                msg = "Value of 'command' key is not a JSON object."
                raise FormatError(msg)


            command_keys = ["name"]
            for key in command_keys:
                if key not in data_command:
                    msg = f"Key '{key}' is missing from 'command' object."
                    raise FormatError(msg)

            thought = data["thought"]
            commands_dict = {c.name: c for c in commands}
            command = commands_dict.get(data_command["name"])


            if command is None:
                if strict:
                    msg = f"Command '{data_command['name']}' not found in list of available commands."
                    raise FormatError(msg)

                return thought, " ".join([data_command["name"], *data_command.get("arguments", {}).values()])


            formatted_args = {}
            if command.arguments:
                for arg in command.arguments:
                    if arg.name in data_command.get("arguments", {}):
                        value = data_command["arguments"][arg.name]
                        if _should_quote(value, command):
                            value = quote(value)
                        formatted_args[arg.name] = Template(arg.argument_format).render(value=value)
                    elif strict and arg.required:
                        msg = f"Required argument '{arg.name}' missing for command '{command.name}'"
                        raise FormatError(msg)


            action = command.invoke_format.format(**formatted_args).strip()
            return thought, action
        except json.JSONDecodeError:
            msg = "Model output is not valid JSON."
            raise FormatError(msg)


class BashCodeBlockParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    No bash code blocks were detected in your output.
    You need to include at least one bash code block in your output.

    It must follow this format exactly to be valid:
    ```bash
    cmd arg1 arg2 ...
    ...

    Other types of code blocks (e.g. python, rust, none, etc.) won't be executed. Only bash.
    """)

    type: Literal["all_bash_code_blocks"] = "all_bash_code_blocks"

    def __call__(self, model_response: dict, commands: list[Command], strict=False):

        pattern = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(model_response["message"])
        if not matches:
            msg = "No bash code blocks were detected in your output."
            raise FormatError(msg)
        thought = pattern.sub("<extracted_code_block>", model_response["message"])
        action = "\n".join(matches)
        return thought, action


class SingleBashCodeBlockParser(AbstractParseFunction, BaseModel):


    error_message: str = dedent("""\
    We did not detect the right number of bash code blocks in your output.
    You need to include EXACTLY ONE bash code block in your output.

    It must follow this format exactly to be valid:
    ```bash
    cmd arg1 arg2 ...
    ```
    """)

    type: Literal["single_bash_code_block"] = "single_bash_code_block"

    def __call__(self, model_response: dict, commands: list[Command], strict=False):

        pattern = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(model_response["message"])
        if not matches:
            msg = "No bash code blocks were detected in your output."
            raise FormatError(msg)
        if len(matches) > 1:
            msg = (
                "We detected multiple bash code blocks in your output. "
                "You need to include EXACTLY ONE bash code block in your output."
            )
            raise FormatError(msg)
        thought = pattern.sub("<extracted_code_block>", model_response["message"])
        action = "\n".join(matches)
        return thought, action


ParseFunction = (
    ActionParser
    | ThoughtActionParser
    | ActionOnlyParser
    | XMLThoughtActionParser
    | XMLFunctionCallingParser
    | FunctionCallingParser
    | EditFormat
    | Identity
    | JsonParser
    | BashCodeBlockParser
    | SingleBashCodeBlockParser
)
