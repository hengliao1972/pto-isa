from __future__ import annotations

"""
`pto_as` — small, readable Python frontend for writing PTO-AS kernels.

This module is intentionally lightweight:

- The *primary* compilation path in this repo parses Python source via `ast`:
  `ptoas.python.ast_frontend` (`binding/python/ptoas/python/ast_frontend.py`) → emits PTO-AS text.
- This runtime builder exists for ergonomics and IDE friendliness: you can
  execute `build()` to produce PTO-AS text without string templates.

The textual PTO-AS emitted here is compatible with the existing toolchain:
`ptoas/tools/python_to_pto.py` / `ptoas/tools/python_kernel_flow.py`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ptoas.python.pto_asm import PTOProgram, TensorType, TileType


@dataclass(frozen=True)
class ScalarType:
    dtype: str


def scalar(dtype: str) -> ScalarType:
    return ScalarType(dtype=dtype)


@dataclass(frozen=True)
class Value:
    ref: str

    def __str__(self) -> str:
        return self.ref


Tensor = Value
Tile = Value


class PTO:
    def __init__(self, name: str) -> None:
        self.name = name
        self._p = PTOProgram()
        self._next_arg = 0
        self._next_tmp = 0

    def _fresh(self, prefix: str) -> str:
        self._next_tmp += 1
        return f"{prefix}{self._next_tmp}"

    # --- High-level structure ---

    def comment(self, text: str) -> None:
        self._p.comment(text)

    def prologue(self) -> None:
        self._p.prologue()

    def epilogue(self) -> None:
        self._p.epilogue()

    def program(self) -> PTOProgram:
        return self._p

    def emit(self) -> str:
        return self._p.emit()

    # --- Declarations ---

    def tensor(
        self,
        name: str | tuple[int, int] | None = None,
        shape: tuple[int, int] | None = None,
        *,
        dtype: str,
        stride: tuple[int, int] | None = None,
        layout: str = "ND",
        arg: int | None = None,
        role: str | None = None,
    ) -> Tensor:
        # `role` is host-metadata only; stored externally by the AST frontend. Kept here for API parity.
        _ = role
        if shape is None and isinstance(name, tuple):
            shape = name
            name = None
        if shape is None:
            raise TypeError("tensor(...) requires shape=(H, W)")

        if name is None:
            name = self._fresh("v")
        view = f"%{name}"
        ty = TensorType(dtype=dtype, shape=shape, stride=stride, layout=layout)
        if arg is None:
            arg = self._next_arg
            self._next_arg += 1
        self._p.make_tensor_view(view=view, arg_index=arg, ty=ty)
        return Tensor(ref=view)

    def _tile(
        self,
        name: str | None = None,
        *,
        loc: str,
        dtype: str,
        shape: tuple[int, int],
        blayout: str,
        slayout: str,
        valid: str | None = None,
        fractal: int | None = None,
        pad: str = "Null",
        addr: int | None = None,
        b: str | None = None,
    ) -> Tile:
        # `b` is a convenience annotation in some examples (broadcast axis); not encoded in PTO-AS tile types.
        _ = b
        rows, cols = int(shape[0]), int(shape[1])
        valid_rows = None
        valid_cols = None
        if valid is not None:
            vr, vc = valid.split("x", 1)
            valid_rows = int(vr)
            valid_cols = int(vc)
        ty = TileType(
            loc=loc,
            dtype=dtype,
            rows=rows,
            cols=cols,
            blayout=blayout,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            slayout=slayout,
            fractal=fractal,
            pad=pad,
        )
        if name is None:
            name = self._fresh("t")
        ref = f"%{name}"
        self._p.alloc_tile(ref, ty, addr=str(addr) if addr is not None else None)
        return Tile(ref=ref)

    def vec_tile(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Vec", dtype=dtype, shape=shape, blayout="RowMajor", slayout="NoneBox", **kw)

    def mat_tile(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Mat", dtype=dtype, shape=shape, blayout="ColMajor", slayout="RowMajor", **kw)

    def left_tile(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Left", dtype=dtype, shape=shape, blayout="RowMajor", slayout="RowMajor", **kw)

    def right_tile(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Right", dtype=dtype, shape=shape, blayout="RowMajor", slayout="ColMajor", **kw)

    def acc_tile(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self._tile(name, loc="Acc", dtype=dtype, shape=shape, blayout="ColMajor", slayout="RowMajor", **kw)

    # --- Short "grammar candy" aliases (supported by the AST frontend) ---

    def vec(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self.vec_tile(name, dtype=dtype, shape=shape, **kw)

    def mat(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self.mat_tile(name, dtype=dtype, shape=shape, **kw)

    def left(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self.left_tile(name, dtype=dtype, shape=shape, **kw)

    def right(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self.right_tile(name, dtype=dtype, shape=shape, **kw)

    def acc(self, name: str | None = None, *, dtype: str, shape: tuple[int, int], **kw: Any) -> Tile:
        return self.acc_tile(name, dtype=dtype, shape=shape, **kw)

    # --- Constants ---

    def const(self, name: str, value: Any, ty: ScalarType) -> Any:
        # PTO-AS uses immediates for scalars today. Keep the API for readability.
        _ = name, ty
        return value

    # --- Memory ops ---

    def tload(self, dst: Tile, src: Tensor, r: int = 0, c: int = 0) -> Tile:
        self._p.assign(dst.ref, "tload", [f"{src.ref}[{r}, {c}]"])
        return dst

    def tstore(self, dst: Tensor, src: Tile, r: int = 0, c: int = 0) -> None:
        self._p.op("tstore", [f"{dst.ref}[{r}, {c}]", src.ref])

    # Common sugar used by the AST frontend (statement form).
    def store(self, dst: Tensor, src: Tile, r: int = 0, c: int = 0) -> None:
        self.tstore(dst, src, r, c)

    # --- GM FIFO ops (prototype) ---

    def tpush(self, dst: Tensor, src: Tile, token: int = 0) -> None:
        # Statement-form op (no destination tile).
        self._p.op("tpush", [dst.ref, src.ref, self._fmt(token)])

    def push(self, dst: Tensor, src: Tile, token: int = 0) -> None:
        self.tpush(dst, src, token)

    def tpop(self, dst: Tile, src: Tensor, token: int = 0) -> Tile:
        self._p.assign(dst.ref, "tpop", [src.ref, self._fmt(token)])
        return dst

    def pop(self, dst: Tile, src: Tensor, token: int = 0) -> Tile:
        return self.tpop(dst, src, token)

    def tprint(self, src: Any) -> None:
        # Statement-form op (no destination).
        self._p.op("tprint", [self._fmt(src)])

    def print(self, src: Any) -> None:
        # Python-friendly alias.
        self.tprint(src)

    # --- Generic instruction helpers ---

    def _fmt(self, x: Any) -> str:
        if isinstance(x, Value):
            return x.ref
        if isinstance(x, bool):
            return "1" if x else "0"
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, int):
            return str(x)
        raise TypeError(f"unsupported operand: {type(x).__name__}")

    def _emit_dst_first(self, opcode: str, operands: Iterable[Any]) -> Any:
        ops = list(operands)
        if not ops:
            self._p.op(opcode, [])
            return None
        dst = ops[0]
        rest = ops[1:]
        if not isinstance(dst, Value):
            raise TypeError(f"{opcode} expects first arg as a Tile/Value (dst)")
        self._p.assign(dst.ref, opcode, [self._fmt(o) for o in rest])
        return dst

    def __getattr__(self, opcode: str) -> Callable[..., Any]:
        # Allow `pto.trowmax(dst, a, b)` without having to define every method.
        # Note: Python kernels in this repo are typically *parsed, not executed*.
        # These aliases exist to keep the surface API short and consistent.
        opcode = _OPCODE_ALIASES.get(opcode, opcode)

        def _op(*operands: Any) -> Any:
            return self._emit_dst_first(opcode, operands)

        return _op


# --- PTO op surface API -------------------------------------------------------
#
# Keep a curated list of PTO ISA entrypoints exposed to the Python frontend, so:
# - users get tab completion in IDEs
# - docs can link to a stable list of supported mnemonics
#
# The actual toolchain accepts any `pto.<mnemonic>` string, but only a subset
# lower to real kernels. This list is based on the C++ ISA entrypoints in:
#   include/pto/common/pto_instr.hpp
_PTO_KNOWN_OPS: tuple[str, ...] = (
    "tassign",
    "tadd",
    "tabs",
    "tand",
    "tor",
    "tsub",
    "tmul",
    "tmin",
    "tmax",
    "texpands",
    "tload",
    "tpush",
    "tpop",
    "tprefetch",
    "tcmps",
    "tcmp",
    "tdiv",
    "tshl",
    "tshr",
    "txor",
    "tlog",
    "tdivs",
    "tprelu",
    "tprint",
    "taddc",
    "tsubc",
    "tmatmul_mx",
    "tmatmul",
    "tmatmul_acc",
    "tmatmul_bias",
    "tneg",
    "tmrgsort",
    "textract",
    "tinsert",
    "tfillpad",
    "tfillpad_inplace",
    "tfillpad_expand",
    "tsort32",
    "tgather",
    "tscatter",
    "trem",
    "tpartadd",
    "tpartmax",
    "tpartmin",
    "mgather",
    "mscatter",
    "tcvt",
    "tmov",
    "trowsum",
    "tcolsum",
    "tcolmax",
    "tcolexpand",
    "tcolexpanddiv",
    "tcolexpandmul",
    "tcolexpandsub",
    "tcolexpandexpdif",
    "trowmax",
    "treshape",
    "trowmin",
    "tsels",
    "tsel",
    "ttrans",
    "tmins",
    "trowexpand",
    "trowexpanddiv",
    "trowexpandmul",
    "trowexpandsub",
    "trowexpandadd",
    "trowexpandmax",
    "trowexpandmin",
    "trowexpandexpdif",
    "trsqrt",
    "tsqrt",
    "texp",
    "tnot",
    "trelu",
    "tgatherb",
    "tadds",
    "tsubs",
    "tmuls",
    "trems",
    "tmaxs",
    "tands",
    "tors",
    "tshls",
    "tshrs",
    "txors",
    "tlrelu",
    "taddsc",
    "tsubsc",
    "tcolmin",
)

_OPCODE_ALIASES: dict[str, str] = {
    # Existing sugar in the AST frontend.
    "mov": "tmov",
    "load": "tload",
    "print": "tprint",
    # Requested API refinement.
    "rowmax": "trowmax",
    "matmul": "tmatmul",
    # Extra convenience (kept symmetric with existing `t*` names).
    "matmul_acc": "tmatmul_acc",
    "matmul_mx": "tmatmul_mx",
    "matmul_bias": "tmatmul_bias",
}


def _install_pto_op_methods() -> None:
    def _mk(*, py_name: str, opcode: str) -> Callable[..., Any]:
        def _op(self: PTO, *operands: Any) -> Any:
            return self._emit_dst_first(opcode, operands)

        _op.__name__ = py_name
        return _op

    for op in _PTO_KNOWN_OPS:
        if hasattr(PTO, op):
            continue
        setattr(PTO, op, _mk(py_name=op, opcode=op))

    # Prefer shorter Python method names: allow dropping the leading 't' for
    # the curated PTO ISA ops (e.g. tadd -> add, tadds -> adds, trowmax -> rowmax).
    #
    # Keep the original `t*` spellings for backward compatibility.
    for op in _PTO_KNOWN_OPS:
        if not (op.startswith("t") and len(op) > 1 and op[1].isalpha()):
            continue
        alias = op[1:]
        _OPCODE_ALIASES.setdefault(alias, op)

    for alias, real in _OPCODE_ALIASES.items():
        if hasattr(PTO, alias):
            continue
        setattr(PTO, alias, _mk(py_name=alias, opcode=real))


_install_pto_op_methods()
