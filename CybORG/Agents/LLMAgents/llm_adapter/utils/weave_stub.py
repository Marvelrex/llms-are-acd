from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

# Weave is optional and only used for tracing. Provide a safe no-op fallback
# so evaluation can run when the dependency is not installed.
try:  # pragma: no cover - exercised only when weave is available
    import weave  # type: ignore
except Exception:  # pragma: no cover - weave not installed or failed to import

    def _noop_decorator(func: Callable | None = None, **_: Any):
        if func is None:
            def wrapper(fn: Callable):
                return fn
            return wrapper
        return func

    weave = SimpleNamespace(op=_noop_decorator)

