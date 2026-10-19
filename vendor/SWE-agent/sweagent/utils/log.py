from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path, PurePath

from rich.logging import RichHandler
from rich.text import Text

_SET_UP_LOGGERS: set[str] = set()
_ADDITIONAL_HANDLERS: dict[str, logging.Handler] = {}
_LOG_LOCK = threading.Lock()

logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


def _interpret_level(level: int | str | None, *, default=logging.DEBUG) -> int:
    if not level:
        return default
    if isinstance(level, int):
        return level
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level(os.environ.get("SWE_AGENT_LOG_STREAM_LEVEL"))
_INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = False

_THREAD_NAME_TO_LOG_SUFFIX: dict[str, str] = {}


def register_thread_name(name: str) -> None:

    thread_name = threading.current_thread().name
    _THREAD_NAME_TO_LOG_SUFFIX[thread_name] = name


class _RichHandlerWithEmoji(RichHandler):
    def __init__(self, emoji: str, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if not emoji.endswith(" "):
            emoji += " "
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname.replace("WARNING", "WARN")
        return Text.styled((self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}")


def get_logger(name: str, *, emoji: str = "") -> logging.Logger:

    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        name = name + "-" + _THREAD_NAME_TO_LOG_SUFFIX.get(thread_name, thread_name)
    logger = logging.getLogger(name)
    if logger.hasHandlers():

        return logger
    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)

    logger.setLevel(logging.TRACE)
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    with _LOG_LOCK:
        for handler in _ADDITIONAL_HANDLERS.values():
            my_filter = getattr(handler, "my_filter", None)
            if my_filter is None:
                logger.addHandler(handler)
            elif isinstance(my_filter, str) and my_filter in name:
                logger.addHandler(handler)
            elif callable(my_filter) and my_filter(name):
                logger.addHandler(handler)
    if _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER:
        _add_logger_name_to_stream_handler(logger)
    return logger


def add_file_handler(
    path: PurePath | str,
    *,
    filter: str | Callable[[str], bool] | None = None,
    level: int | str = logging.TRACE,
    id_: str = "",
) -> str:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(_interpret_level(level))
    with _LOG_LOCK:

        for name in _SET_UP_LOGGERS:
            if filter is not None:
                if isinstance(filter, str) and filter not in name:
                    continue
                if callable(filter) and not filter(name):
                    continue
            logger = logging.getLogger(name)
            logger.addHandler(handler)
    handler.my_filter = filter
    if not id_:
        id_ = str(uuid.uuid4())
    _ADDITIONAL_HANDLERS[id_] = handler
    return id_


def remove_file_handler(id_: str) -> None:

    handler = _ADDITIONAL_HANDLERS.pop(id_)
    with _LOG_LOCK:

        for log_name in _SET_UP_LOGGERS:
            logger = logging.getLogger(log_name)
            logger.removeHandler(handler)


def _add_logger_name_to_stream_handler(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, _RichHandlerWithEmoji):
            formatter = logging.Formatter("[%(name)s] %(message)s")
            handler.setFormatter(formatter)


def add_logger_names_to_stream_handlers() -> None:

    global _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER
    _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = True
    with _LOG_LOCK:
        for logger in _SET_UP_LOGGERS:
            _add_logger_name_to_stream_handler(logging.getLogger(logger))


def set_stream_handler_levels(level: int) -> None:

    global _STREAM_LEVEL
    _STREAM_LEVEL = level
    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, _RichHandlerWithEmoji):
                    current_level = handler.level
                    if current_level < level:
                        handler.setLevel(level)
