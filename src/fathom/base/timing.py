from __future__ import annotations

import logging
import time
from functools import wraps
from inspect import iscoroutinefunction
from typing import Awaitable, Callable, Optional, ParamSpec, TypeVar, cast

ReturnType = TypeVar("ReturnType")
ParameterSpecification = ParamSpec("ParameterSpecification")


def time_it(
    *,
    operation: str,
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
) -> Callable[
    [Callable[ParameterSpecification, ReturnType]],
    Callable[ParameterSpecification, ReturnType],
]:
    """
    Measure function execution time and emit a structured timing log entry.
    """

    def __decorate(
        function: Callable[ParameterSpecification, ReturnType],
    ) -> Callable[ParameterSpecification, ReturnType]:
        resolved_logger = logging.getLogger(logger_name or function.__module__)

        if iscoroutinefunction(function):

            @wraps(function)
            async def __async_wrapper(
                *args: ParameterSpecification.args,
                **kwargs: ParameterSpecification.kwargs,
            ) -> ReturnType:
                start_time = time.perf_counter()
                try:
                    typed_function = cast(
                        "Callable[ParameterSpecification, Awaitable[ReturnType]]",
                        function,
                    )
                    return await typed_function(*args, **kwargs)
                finally:
                    duration = (time.perf_counter() - start_time) * 1000
                    resolved_logger.log(
                        level, "[TIMING] %s completed in %.2fms", operation, duration
                    )

            return cast("Callable[ParameterSpecification, ReturnType]", __async_wrapper)

        @wraps(function)
        def __sync_wrapper(
            *args: ParameterSpecification.args,
            **kwargs: ParameterSpecification.kwargs,
        ) -> ReturnType:
            """ """

            start_time = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start_time) * 1000
                resolved_logger.log(level, "[TIMING] %s completed in %.2fms", operation, duration)

        return cast("Callable[ParameterSpecification, ReturnType]", __sync_wrapper)

    return __decorate
