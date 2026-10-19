from typing import Any, Literal


class FormatError(Exception):
    pass


class FunctionCallingFormatError(FormatError):


    def __init__(
        self,
        message: str,
        error_code: Literal[
            "missing", "multiple", "incorrect_args", "invalid_json", "invalid_command", "missing_arg", "unexpected_arg"
        ],
        **extra_info: Any,
    ):
        super().__init__(message + f" [error_code={error_code}]")
        self.message = message
        self.extra_info = {"error_code": error_code, **extra_info}


class ContextWindowExceededError(Exception):
    pass


class CostLimitExceededError(Exception):
    pass


class InstanceCostLimitExceededError(CostLimitExceededError):
    pass


class TotalCostLimitExceededError(CostLimitExceededError):
    pass


class InstanceCallLimitExceededError(CostLimitExceededError):
    pass


class ContentPolicyViolationError(Exception):
    pass


class ModelConfigurationError(Exception):
    pass
