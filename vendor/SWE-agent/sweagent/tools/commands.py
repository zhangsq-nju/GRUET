from __future__ import annotations

import re
import string
from collections import Counter
from functools import cached_property

from pydantic import BaseModel, field_validator, model_validator

from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax

ARGUMENT_NAME_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]*"


def _extract_keys(format_string: str) -> set[str]:

    formatter = string.Formatter()
    keys = set()
    for _, field_name, _, _ in formatter.parse(format_string):
        if field_name is not None:
            keys.add(field_name)
    return keys


class Argument(BaseModel):
    name: str
    type: str
    items: dict[str, str] | None = None
    description: str
    required: bool
    enum: list[str] | None = None
    argument_format: str = "{{value}}"


    @field_validator("argument_format")
    def validate_argument_format(cls, value: str) -> str:
        _warn_probably_wrong_jinja_syntax(value)
        return value


class Command(BaseModel):


    name: str
    docstring: str | None
    signature: str | None = None

    end_name: str | None = None
    arguments: list[Argument] = []

    @cached_property
    def invoke_format(self) -> str:

        if self.signature:

            for arg in self.arguments:
                if not (
                    f"<{arg.name}>" in self.signature
                    or f"[<{arg.name}>]" in self.signature
                    or f"{{{arg.name}}}" in self.signature
                    or f"--{arg.name}" in self.signature
                ):
                    msg = (
                        f"Missing argument {arg.name} in signature: {self.signature}. Did you format the signature correctly? "
                        f"You must include all argument names in the signature with <{arg.name}>, [<{arg.name}>], {{{arg.name}}}, or --{arg.name} notation."
                    )
                    raise ValueError(msg)


            return re.sub(rf"\[?<({ARGUMENT_NAME_PATTERN})>\]?", r"{\1}", self.signature)
        else:

            _invoke_format = f"{self.name} "
            for arg in self.arguments:
                _invoke_format += f"{{{arg.name}}} "
            return _invoke_format

    def get_function_calling_tool(self) -> dict:

        tool = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.docstring or "",
            },
        }
        properties = {}
        required = []
        if self.arguments:
            for arg in self.arguments:
                properties[arg.name] = {"type": arg.type, "description": arg.description}

                if arg.items:
                    properties[arg.name]["items"] = arg.items

                if arg.required:
                    required.append(arg.name)


                if arg.enum:
                    properties[arg.name]["enum"] = arg.enum
        tool["function"]["parameters"] = {"type": "object", "properties": properties, "required": required}
        return tool

    @model_validator(mode="after")
    def validate_arguments(self) -> Command:

        if not self.arguments:
            return self
        found_optional = False
        for arg in self.arguments:
            if found_optional and arg.required:
                msg = f"Command '{self.name}': Required argument '{arg.name}' cannot come after optional arguments"
                raise ValueError(msg)
            if not arg.required:
                found_optional = True

        name_counts = Counter(arg.name for arg in self.arguments)
        duplicates = {name for name, count in name_counts.items() if count > 1}
        if duplicates:
            msg = f"Command '{self.name}': Duplicate argument names: {duplicates}"
            raise ValueError(msg)
        for arg in self.arguments:
            if not re.match(ARGUMENT_NAME_PATTERN, arg.name):
                msg = f"Command '{self.name}': Invalid argument name: '{arg.name}'"
                raise ValueError(msg)
        if (invoke_keys := _extract_keys(self.invoke_format)) != {arg.name for arg in self.arguments}:
            msg = f"Command '{self.name}': Argument names ({invoke_keys}) in signature / invoke_format {self.invoke_format!r} do not match argument names"
            raise ValueError(msg)
        return self


BASH_COMMAND = Command(
    name="bash",

    signature="<command>",

    docstring="runs the given command directly in bash",
    arguments=[
        Argument(
            name="command",
            type="string",
            description="The bash command to execute.",
            required=True,
        )
    ],
)
