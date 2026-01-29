from __future__ import annotations

"""
Python DSL stubs for writing PTO kernels with normal Python syntax.

These are intentionally NOT executable. The AST frontend parses your source code
and translates calls like `tensor(...)`, `tile(...)`, `tload(...)`, etc into PTO-AS.

Recommended usage in kernel files:

  from ptoas.python.dsl import *

  def add16():
      prologue()
      x = tensor(dtype="f16", shape=(16, 16))
      ...
      epilogue()
"""

from dataclasses import dataclass
from typing import Any, Literal, overload


class _NotExecutableError(RuntimeError):
    pass


def _noexec() -> None:
    raise _NotExecutableError("ptoas python DSL is not executable; compile it with ptoas.python.binding")


@dataclass(frozen=True)
class Tensor:
    pass


@dataclass(frozen=True)
class Tile:
    pass


def prologue() -> None:
    _noexec()


def epilogue() -> None:
    _noexec()


def get_block_idx() -> int:
    _noexec()


def get_block_num() -> int:
    _noexec()

def iadd(a: int, b: int) -> int:
    _noexec()

def isub(a: int, b: int) -> int:
    _noexec()

def imul(a: int, b: int) -> int:
    _noexec()

def idiv(a: int, b: int) -> int:
    _noexec()

def irem(a: int, b: int) -> int:
    _noexec()

def imin(a: int, b: int) -> int:
    _noexec()

def imax(a: int, b: int) -> int:
    _noexec()


@overload
def tensor(
    dtype: str,
    shape: tuple[int, int],
    stride: tuple[int, int] | None = None,
    layout: str = "ND",
    *,
    arg: int | None = None,
    arg_index: int | None = None,
    role: Literal["in", "out", "inout"] | None = None,
) -> Tensor: ...


def tensor(*args: Any, **kwargs: Any) -> Tensor:
    _noexec()


def tile(
    loc: Literal["Vec", "Mat", "Left", "Right", "Acc"],
    dtype: str,
    rows: int,
    cols: int,
    *,
    blayout: str = "RowMajor",
    valid: str | None = None,
    valid_row: int | None = None,
    valid_col: int | None = None,
    slayout: str = "NoneBox",
    fractal: int | None = None,
    pad: str = "Null",
    addr: int | None = None,
) -> Tile:
    _noexec()


def tmov(dst: Tile, src: Tile) -> None:
    _noexec()


def tadd(dst: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def tmatmul(dst: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def tmatmul_acc(dst: Tile, acc: Tile, a: Tile, b: Tile) -> None:
    _noexec()

def matmul(dst: Tile, a: Tile, b: Tile) -> None:
    _noexec()


def rowmax(dst: Tile, src: Tile, tmp: Tile) -> None:
    _noexec()


def tload(dst: Tile, src: Tensor, r: int, c: int) -> None:
    _noexec()


def tstore(dst: Tensor, r: int, c: int, src: Tile) -> None:
    _noexec()


def record_event(*, src_op: str, dst_op: str, token: int) -> None:
    _noexec()


def wait_event(*, src_op: str, dst_op: str, token: int) -> None:
    _noexec()


# Best-effort: expose the full PTO ISA surface as no-op stubs so kernel authors
# can `from ptoas.python.dsl import *` and still get import-time names.
#
# The Python AST frontend accepts `pto.<mnemonic>(...)` calls, and the set of
# legal mnemonics is determined by the underlying PTO toolchain. Keep this list
# in sync with `include/pto/common/pto_instr.hpp` (MAP_INSTR_IMPL entries).
_PTO_ISA_OPS: tuple[str, ...] = (
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


def _install_isa_stubs() -> None:
    for name in _PTO_ISA_OPS:
        if name in globals():
            continue

        def _mk(n: str):
            def _f(*args: Any, **kwargs: Any) -> Any:
                _noexec()

            _f.__name__ = n
            return _f

        globals()[name] = _mk(name)


_install_isa_stubs()
