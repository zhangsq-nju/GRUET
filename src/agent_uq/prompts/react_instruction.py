from __future__ import annotations

from typing import Any

_PREFACE_BY_ADAPTER = {
    "python": (
        "Interact with a Python execution environment using Python code to answer a question.\n"
        "Use interleaving Thought, Action, Observation steps.\n"
        "Thought can reason about the current situation, and Action can be two types:\n"
        "(1) execute, which executes Python code on the Python execution environment\n"
        "(2) submit, which indicates that the previous observation is the answer\n"
        "Every Thought MUST contain one or more `Step i:` lines, one per atomic reasoning step.\n"
        "After finishing the reasoning inside a Thought block, emit exactly one "
        "Action in one of the forms described below."
    ),
    "sql": (
        "Interact with a MySQL Database system using SQL queries to answer a question.\n"
        "Use interleaving Thought, Action, Observation steps.\n"
        "Thought can reason about the current situation, and Action can be two types:\n"
        "(1) execute, which executes SQL code on the MySQL Database system\n"
        "(2) submit, which indicates that the previous observation is the answer\n"
        "Every Thought MUST contain one or more `Step i:` lines, one per atomic reasoning step.\n"
        "After finishing the reasoning inside a Thought block, emit exactly one "
        "Action in one of the forms described below."
    ),
    "nl2bash": (
        "Interact with a Bourne Shell system using Bash commands to answer a question.\n"
        "Use interleaving Thought, Action, Observation steps.\n"
        "Thought can reason about the current situation, and Action can be two types:\n"
        "(1) execute, which executes Bash commands on the Bourne Shell system\n"
        "(2) submit, which indicates that the previous observation is the answer\n"
        "Every Thought MUST contain one or more `Step i:` lines, one per atomic reasoning step.\n"
        "After finishing the reasoning inside a Thought block, emit exactly one "
        "Action in one of the forms described below."
    ),
}


_PYTHON_MBPP_TERMINATOR = (
    "When a Thought performs an intermediate Python computation, end that "
    "Thought with `Action: execute[<python_snippet>]`. When you are confident "
    "in your final solution, end the trajectory with "
    "`Action: submit[<function_name>]` where `<function_name>` is the name of "
    "the function you defined that the MBPP grader should call. Stop "
    "generation immediately after the submit action."
)

_PYTHON_APPS_TERMINATOR = (
    "When a Thought performs an intermediate Python computation, end that "
    "Thought with `Action: execute[<python_snippet>]`. When you are confident "
    "in your final solution, end the trajectory with "
    "`Action: submit_program[<full_program>]` where `<full_program>` is the "
    "complete standalone program (it will be executed against the dataset's "
    "input/output cases). Stop generation immediately after the "
    "submit_program action."
)

_SQL_TERMINATOR = (
    "Each intermediate Thought must end with `Action: execute[<sql_query>]`. "
    "When you are confident in the final answer, end the trajectory with "
    "`Action: submit`. Stop generation immediately after the submit action."
)

_NL2BASH_TERMINATOR = (
    "Each intermediate Thought must end with `Action: execute[<bash_command>]`. "
    "When you are confident in the final answer, end the trajectory with "
    "`Action: submit`. Stop generation immediately after the submit action."
)


_TERMINATORS_BY_KEY = {

    ("python", "python_mbpp"): _PYTHON_MBPP_TERMINATOR,
    ("python", "python_apps"): _PYTHON_APPS_TERMINATOR,

    ("sql", "sql_spider"): _SQL_TERMINATOR,
    ("sql", "sql_bird"): _SQL_TERMINATOR,

    ("nl2bash", "nl2bash"): _NL2BASH_TERMINATOR,
    ("nl2bash", "nl2bash_fs1"): _NL2BASH_TERMINATOR,
}
def react_instruction(
    adapter_name: str,
    dataset_key: str,
    *,
    prompt_context: Any | None = None,
) -> str:

    if adapter_name not in {"python", "sql", "nl2bash"}:
        raise ValueError(
            f"react_instruction: unknown adapter_name {adapter_name!r}; "
            "expected one of: python, sql, nl2bash"
        )
    prefatory_text = _PREFACE_BY_ADAPTER[adapter_name]
    terminator = _TERMINATORS_BY_KEY.get((adapter_name, dataset_key))
    if terminator is None:

        for (adapt, _key), term in _TERMINATORS_BY_KEY.items():
            if adapt == adapter_name:
                terminator = term
                break
    if terminator is None:

        raise ValueError(
            f"react_instruction: no terminator registered for "
            f"adapter={adapter_name!r}, dataset_key={dataset_key!r}"
        )
    return prefatory_text + "\n" + terminator + "\nHere are some examples."
