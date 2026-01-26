from __future__ import annotations

from typing import Any

# Re-export the Python DSL used throughout this repo.
try:
    from pto_as import PTO, scalar
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import `pto_as`. Ensure the repo root is on PYTHONPATH "
        "(e.g. run from the repo root, or `export PYTHONPATH=$PWD:$PYTHONPATH`)."
    ) from exc


# Optional: re-export the AST frontend helper when present (used by the performance kernels).
try:
    from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec
except ImportError:  # pragma: no cover
    KernelSpec = Any  # type: ignore[assignment]

    def compile_kernel_spec(*_: Any, **__: Any) -> Any:
        raise ImportError("`ptoas.python.ast_frontend` is not available in this environment")


__all__ = [
    "PTO",
    "scalar",
    "KernelSpec",
    "compile_kernel_spec",
]

