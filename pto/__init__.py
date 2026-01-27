from __future__ import annotations

from typing import Any

# Re-export the Python DSL used throughout this repo.
try:
    from pto_as import PTO, scalar
except ImportError:  # pragma: no cover
    # Keep the `pto` package importable even when the optional DSL backend is
    # unavailable (e.g. when only using `pto.runtime` / `ptoas` toolchain).
    class _MissingPtoAs:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError(
                "Failed to import `pto_as`. Ensure the repo root is on PYTHONPATH "
                "(e.g. run from the repo root, or `export PYTHONPATH=$PWD:$PYTHONPATH`)."
            )

    def scalar(*_: Any, **__: Any) -> Any:  # pragma: no cover
        raise ImportError(
            "Failed to import `pto_as`. Ensure the repo root is on PYTHONPATH "
            "(e.g. run from the repo root, or `export PYTHONPATH=$PWD:$PYTHONPATH`)."
        )

    PTO = _MissingPtoAs  # type: ignore[assignment]


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
