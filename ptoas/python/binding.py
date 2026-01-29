from __future__ import annotations

"""
User-facing bindings for compiling Python kernels (normal syntax) into PTO-AS text.

This module is intentionally small: it only parses Python AST and emits `.pto`
files via the AST frontend.
"""

from pathlib import Path

from . import ast_frontend
from .ast_frontend import KernelSpec
from .host_spec import HostSpec, HostTensorArg, prepend_host_spec_to_pto


def _select_kernel(source: str, kernel: str | None) -> str:
    names = ast_frontend.list_kernel_functions(source)
    if not names:
        raise ValueError("no kernel functions found")
    if kernel is not None:
        if kernel not in names:
            raise ValueError(f"kernel not found: {kernel} (available: {', '.join(names)})")
        return kernel
    if len(names) == 1:
        return names[0]
    raise ValueError(f"multiple kernels found; pass --kernel ({', '.join(names)})")


def compile_file(path: Path, *, kernel: str | None = None) -> KernelSpec:
    source = path.read_text(encoding="utf-8")
    kernel_name = _select_kernel(source, kernel)
    return ast_frontend.compile_kernel_spec_from_source(source, func_name=kernel_name)


def default_host_spec(spec: KernelSpec) -> HostSpec:
    args = list(spec.tensor_args)
    if not args:
        raise ValueError("kernel has no tensor args")

    roles = [a.role for a in args]
    if all(r is None for r in roles):
        inferred = ["in"] * len(args)
        inferred[-1] = "out"
        roles = inferred
    else:
        roles = [(r or "in") for r in roles]
        if not any(r in ("out", "inout") for r in roles):
            roles[-1] = "out"

    host_args: list[HostTensorArg] = []
    for i, a in enumerate(args):
        h, w = a.ty.shape2()
        s0, s1 = a.ty.stride2()
        layout = str(a.ty.layout)
        stride = None if (a.ty.stride is None and layout == "ND") else (int(s0), int(s1))
        host_args.append(HostTensorArg(dtype=a.ty.dtype, shape=(h, w), role=roles[i], layout=layout, stride=stride))
    return HostSpec(args=tuple(host_args), seed=0, block_dim=1, kernel_name="pto_kernel")


def write_pto(
    path: Path, *, kernel: str | None = None, out_path: Path | None = None, universal: bool = True
) -> Path:
    source = path.read_text(encoding="utf-8")
    kernel_name = _select_kernel(source, kernel)
    spec = ast_frontend.compile_kernel_spec_from_source(source, func_name=kernel_name)
    if out_path is None:
        if kernel is None and len(ast_frontend.list_kernel_functions(source)) == 1:
            out_path = path.with_suffix(".pto")
        else:
            out_path = path.with_name(f"{path.stem}.{spec.name}.pto")
    pto = spec.pto
    if universal:
        pto = prepend_host_spec_to_pto(pto=pto, spec=default_host_spec(spec))
    out_path.write_text(pto, encoding="utf-8")
    return out_path
