from __future__ import annotations

"""
Tiny Python-to-PTO-AS frontend.

The frontend parses a restricted Python AST and emits the newer PTO-AS syntax:
  - `tensor(...)` declares a kernel arg bound to `%argN` via `pto.make_tensor_view`
  - `tile(...)` allocates a tile with `pto.alloc_tile`
  - `load/store/add/matmul/...` map to PTO ops (legacy `t*` spellings still supported)

Only a minimal subset of Python is supported (straight-line code, `for range`,
and simple `if` compares). This is intended for prototyping kernels quickly.

Supported kernel authoring styles:

1) Function-call DSL (legacy, still supported):

   ```
   def add16():
       prologue()
       x = tensor(dtype="f16", shape=(16, 16))
       ...
       tadd(tz, tx, ty)
       tstore(z, 0, 0, tz)
       epilogue()
   ```

2) Object DSL (recommended; matches `pto_as.PTO` examples):

   ```
   def build():
       pto = PTO("my_kernel")
       x = pto.tensor(dtype="f16", shape=(16, 16), role="in")
       y = pto.tensor(dtype="f16", shape=(16, 16), role="out")
       tx = pto.vec(dtype="f16", shape=(16, 16))
       ty = pto.vec(dtype="f16", shape=(16, 16))
       tx = pto.load(x)     # defaults to [0,0]
       ty = pto.add(tx, tx)
       ...
       pto.store(y, ty)     # defaults to [0,0]
   ```
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pto_asm import TensorType, TileType
from .dsl import _PTO_ISA_OPS as _PTO_ISA_OPS_DSL


_STRIPPED_T_OPS: set[str] = {
    op[1:]
    for op in _PTO_ISA_OPS_DSL
    if op.startswith("t") and len(op) > 1 and op[1].isalpha()
}


class FrontendError(Exception):
    pass


@dataclass(frozen=True)
class TensorArg:
    name: str
    arg_index: int
    ty: TensorType
    role: str | None = None

    def host_spec(self) -> "TensorSpec":
        from .host_codegen import TensorSpec

        return TensorSpec(dtype=self.ty.dtype, shape=self.ty.shape2())


@dataclass(frozen=True)
class KernelSpec:
    name: str
    pto: str
    tensor_args: tuple[TensorArg, ...]

    def host_tensor_specs(self) -> list["TensorSpec"]:
        return [arg.host_spec() for arg in self.tensor_args]


@dataclass
class _Sym:
    name: str  # without leading %

    @property
    def pto(self) -> str:
        return f"%{self.name}"


class _Text:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0

    def line(self, s: str) -> None:
        self._lines.append(("  " * self._indent) + s)

    def open(self, header: str) -> None:
        self.line(f"{header} {{")
        self._indent += 1

    def else_open(self) -> None:
        if self._indent <= 0:
            raise FrontendError("else without open block")
        self._indent -= 1
        self.line("} else {")
        self._indent += 1

    def close(self) -> None:
        if self._indent <= 0:
            raise FrontendError("unbalanced close()")
        self._indent -= 1
        self.line("}")

    def emit(self) -> str:
        return "\n".join(self._lines).strip() + "\n"


class _Compiler:
    def __init__(self, *, consts: dict[str, Any] | None = None) -> None:
        self._t = _Text()
        self._sym: dict[str, _Sym] = {}
        self._tmp_i = 0
        self._next_tensor_arg = 0
        # Python-name -> literal string (emits as an immediate, not an SSA value).
        self._literal: dict[str, str] = {}
        # Python-name -> compile-time value (used for shapes/strides/const folding).
        self._const_env: dict[str, Any] = {}
        # Kernel arg tracking for host codegen.
        self._tensor_args: dict[int, TensorType] = {}
        self._tensor_arg_names: dict[int, str] = {}
        self._tensor_arg_roles: dict[int, str | None] = {}
        self._explicit_kernel_name: str | None = None
        self._uses_task_id: bool = False
        # SSA name ("%x") -> metadata, used to lower tload/tstore via pto.subview.
        self._tensor_syms: dict[str, TensorType] = {}
        self._tile_syms: dict[str, TileType] = {}
        # Constant pool (int -> SSA symbol). The `.pto` format requires SSA operands (no immediates).
        self._index_consts: dict[int, _Sym] = {}
        self._i1_consts: dict[bool, _Sym] = {}
        if consts:
            self._seed_consts(consts)

    def _seed_consts(self, consts: dict[str, Any]) -> None:
        """
        Seed compile-time constants for the current compilation.

        This enables parameterized kernels without Python string templating:

          def my_kernel():
              A = tensor(dtype="f16", shape=(m, k))
              ...

          compile_kernel_spec(my_kernel, consts={"m": 4096, "k": 4096})

        Notes:
        - All values are available to `_eval_static(...)` via `_const_env`.
        - For scalar ints/bools/floats, we also seed `_literal` so they can be used
          as immediate operands in index expressions (e.g. `x = m // 16`).
        """
        for name, value in consts.items():
            if not isinstance(name, str) or not name:
                raise FrontendError("consts keys must be non-empty strings")
            self._const_env[name] = value
            if isinstance(value, bool):
                self._literal[name] = "1" if value else "0"
            elif isinstance(value, int):
                self._literal[name] = str(int(value))
            elif isinstance(value, float):
                self._literal[name] = repr(float(value))

    def _tmp(self) -> _Sym:
        self._tmp_i += 1
        name = f"t{self._tmp_i}"
        self._sym[name] = _Sym(name)
        return self._sym[name]

    def _index_const(self, value: int) -> str:
        v = int(value)
        sym = self._index_consts.get(v)
        if sym is not None:
            return sym.pto

        base = f"c{v}" if v >= 0 else f"cneg{-v}"
        name = base
        suffix = 0
        while name in self._sym:
            suffix += 1
            name = f"{base}_{suffix}"

        sym = _Sym(name)
        self._sym[name] = sym
        self._index_consts[v] = sym
        self._t.line(f"{sym.pto} = arith.constant {v} : index")
        return sym.pto

    def _i1_const(self, value: bool) -> str:
        v = bool(value)
        sym = self._i1_consts.get(v)
        if sym is not None:
            return sym.pto

        name = "true" if v else "false"
        if name in self._sym:
            # Avoid colliding with user SSA names.
            name = f"c_{name}"
        sym = _Sym(name)
        self._sym[name] = sym
        self._i1_consts[v] = sym
        lit = "true" if v else "false"
        self._t.line(f"{sym.pto} = arith.constant {lit} : i1")
        return sym.pto

    def _tensor_view_ty(self, *, dtype: str) -> str:
        # NOTE: For Ascend A2/A3 kernels, the CCE helper intrinsics expect ND/NZ
        # matrix shapes to be expressed in 5D:
        #   (1, 1, 1, rows, cols)
        # This enables ND2NZ / NZ2ND paths which assert the leading dims are 1.
        return f"!pto.tensor_view<5x{dtype}>"

    def _tile_view_ty(self, *, rows: int, cols: int, dtype: str) -> str:
        # Keep tile views consistent with the 5D GlobalTensor convention:
        #   (1, 1, 1, rows, cols)
        return f"!pto.tile_view<1x1x1x{int(rows)}x{int(cols)}x{dtype}>"

    def _require_tensor(self, sym: str) -> TensorType:
        ty = self._tensor_syms.get(sym)
        if ty is None:
            raise FrontendError(f"unknown tensor view symbol: {sym}")
        return ty

    def _require_tile(self, sym: str) -> TileType:
        ty = self._tile_syms.get(sym)
        if ty is None:
            raise FrontendError(f"unknown tile buffer symbol: {sym}")
        return ty

    def _emit_subview(self, *, tensor: str, r: str, c: str, rows: int, cols: int, dtype: str) -> str:
        tv_ty = self._tensor_view_ty(dtype=dtype)
        tile_v_ty = self._tile_view_ty(rows=rows, cols=cols, dtype=dtype)
        c0 = self._index_const(0)
        c1 = self._index_const(1)
        c_rows = self._index_const(rows)
        c_cols = self._index_const(cols)
        sub = self._tmp()
        self._t.line(
            f"{sub.pto} = pto.subview {tensor}, offsets = [{c0}, {c0}, {c0}, {r}, {c}], "
            f"sizes = [{c1}, {c1}, {c1}, {c_rows}, {c_cols}] : {tv_ty} -> {tile_v_ty}"
        )
        return sub.pto

    def _sym_for(self, py_name: str) -> _Sym:
        if py_name not in self._sym:
            self._sym[py_name] = _Sym(py_name)
        return self._sym[py_name]

    def _bind_sym(self, py_name: str, pto_name: str) -> _Sym:
        # Bind a Python variable name to a specific PTO-AS SSA name.
        # This is used by the object DSL when the first argument is an explicit name:
        #   centered = pto.vec("scores_centered", ...)
        self._sym[py_name] = _Sym(pto_name)
        return self._sym[py_name]

    def _eval_const(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as e:
            raise FrontendError(f"expected a literal, got: {ast.dump(node)}") from e

    def _eval_static(self, node: ast.AST) -> Any:
        """
        Evaluate a restricted, compile-time-only Python expression.

        This is used for things like:
          - tensor/tiling shapes: (s, d)
          - simple math for constants: 1.0 / sqrt(d)
        """

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._const_env:
                return self._const_env[node.id]
            # Fall back to literal operand strings for already-emitted literals.
            if node.id in self._literal:
                try:
                    return ast.literal_eval(self._literal[node.id])
                except Exception:
                    pass
            raise FrontendError(f"unknown compile-time name: {node.id}")
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_static(elt) for elt in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._eval_static(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            return -v
        if isinstance(node, ast.BinOp):
            lhs = self._eval_static(node.left)
            rhs = self._eval_static(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
            raise FrontendError(f"unsupported binop in const eval: {ast.dump(node.op)}")
        if isinstance(node, ast.Call):
            # Support a tiny whitelist of compile-time functions.
            fn_name: str | None = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # e.g. math.sqrt(...)
                fn_name = f"{node.func.value.id}.{node.func.attr}"
            if fn_name is None:
                raise FrontendError(f"unsupported call in const eval: {ast.dump(node)}")

            args = [self._eval_static(a) for a in node.args]
            if fn_name in ("sqrt", "math.sqrt"):
                if len(args) != 1:
                    raise FrontendError("sqrt(...) expects 1 arg")
                import math

                return math.sqrt(args[0])
            if fn_name == "scalar":
                # `scalar("f32")` is a type hint in the Python frontend.
                if len(args) != 1 or not isinstance(args[0], str):
                    raise FrontendError('scalar(...) expects one string arg like scalar("f32")')
                return args[0]
            raise FrontendError(f"unsupported compile-time call: {fn_name}")

        raise FrontendError(f"unsupported compile-time expr: {ast.dump(node)}")

    def _opnd(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id in self._const_env:
                v = self._const_env[node.id]
                if isinstance(v, bool):
                    return self._i1_const(v)
                if isinstance(v, int):
                    return self._index_const(v)
                raise FrontendError(f"unsupported const operand type: {node.id}={type(v).__name__}")
            return self._sym_for(node.id).pto
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # Support enum-like spellings in operands, e.g.:
            #   RoundMode.CAST_ROUND  ->  RoundMode::CAST_ROUND
            return f"{node.value.id}::{node.attr}"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return self._i1_const(node.value)
            if isinstance(node.value, int):
                return self._index_const(node.value)
            raise FrontendError(f"unsupported constant operand: {node.value!r}")
        raise FrontendError(f"unsupported operand node: {ast.dump(node)}")

    def _call_name(self, call: ast.Call) -> str:
        # Support both:
        #   tensor(...)
        #   pto.tensor(...)
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            # Intentionally allow any `<name>.<attr>(...)` where `<name>` is a
            # plain name (we only special-case `pto` further down).
            return call.func.attr
        raise FrontendError(f"unsupported call form: {ast.dump(call)}")

    def _opcode_alias(self, name: str) -> str:
        # Small Python-friendly aliases.
        # Keep this intentionally minimal: most PTO ops are already descriptive.
        aliased = {
            "mov": "tmov",
            "load": "tload",
            "store": "tstore",
            "print": "tprint",
            # API candy: prefer shorter names in Python kernels.
            "rowmax": "trowmax",
            "matmul": "tmatmul",
            # Op mnemonics use dotted forms.
            "matmul_acc": "tmatmul.acc",
            "tmatmul_acc": "tmatmul.acc",
        }.get(name)
        if aliased is not None:
            return aliased

        # Allow dropping the leading 't' for the PTO tile ISA ops
        # (e.g. add -> tadd, adds -> tadds, rowexpand -> trowexpand, ...).
        if name in _STRIPPED_T_OPS:
            return "t" + name

        return name

    def _emit_instr_assign(self, *, dst_name: str, call: ast.Call) -> None:
        """
        Emit an instruction where the destination is provided by Python assignment:

          dst = pto.tadd(a, b)          # emits: %dst = pto.tadd %a, %b
          dst = pto.tload(x, r, c)      # emits: %dst = pto.tload %x[r, c]

        This is the preferred "left assignment" style for readability.
        """
        fn = self._opcode_alias(self._call_name(call))
        dst = self._sym_for(dst_name).pto

        if fn == "get_task_id":
            # Object-DSL helper used by the runtime/MPMD path: task id is passed as %arg0.
            if call.args or call.keywords:
                raise FrontendError("get_task_id() takes no arguments")
            self._uses_task_id = True
            # Alias the Python name to the function argument SSA value.
            self._bind_sym(dst_name, "arg0")
            return

        if fn == "get_block_idx":
            if call.args or call.keywords:
                raise FrontendError("get_block_idx() takes no arguments")
            self._t.line(f"{dst} = pto.get_block_idx : index")
            return

        if fn == "get_block_num":
            if call.args or call.keywords:
                raise FrontendError("get_block_num() takes no arguments")
            self._t.line(f"{dst} = pto.get_block_num : index")
            return

        if fn in ("prologue", "epilogue", "comment", "program"):
            raise FrontendError(f"cannot assign the result of {fn}(...); use it as a statement")
        if fn in ("tstore", "store", "tpush"):
            raise FrontendError("tstore/store does not return a value; use it as a statement")
        if fn in ("record_event", "wait_event"):
            raise FrontendError(f"{fn}(...) does not return a value; use it as a statement")

        # Tile world ops are destination-style (DPS): the SSA name refers to a logical buffer.
        # The Python frontend allows assignment syntax for readability, but the ops do not
        # return tile buffers.
        if fn == "tload":
            if len(call.args) not in (1, 3):
                raise FrontendError("tload/load in assignment form expects: dst = tload(src, [r, c])")
            dst_ty = self._require_tile(dst)
            src = self._opnd(call.args[0])
            src_ty = self._require_tensor(src)
            if dst_ty.dtype != src_ty.dtype:
                raise FrontendError(f"tload dtype mismatch: dst={dst_ty.dtype} src={src_ty.dtype}")
            r = self._index_const(0)
            c = self._index_const(0)
            if len(call.args) == 3:
                r = self._opnd(call.args[1])
                c = self._opnd(call.args[2])
            sub = self._emit_subview(
                tensor=src,
                r=r,
                c=c,
                rows=int(dst_ty.rows),
                cols=int(dst_ty.cols),
                dtype=src_ty.dtype,
            )
            tile_v_ty = self._tile_view_ty(rows=int(dst_ty.rows), cols=int(dst_ty.cols), dtype=src_ty.dtype)
            self._t.line(f"pto.tload ins({sub} : {tile_v_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tmov":
            if len(call.args) != 1:
                raise FrontendError("tmov/mov in assignment form expects: dst = tmov(src)")
            dst_ty = self._require_tile(dst)
            src = self._opnd(call.args[0])
            src_ty = self._require_tile(src)
            self._t.line(f"pto.tmov ins({src} : {src_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tmatmul":
            if len(call.args) != 2:
                raise FrontendError("tmatmul/matmul in assignment form expects: dst = tmatmul(lhs, rhs)")
            dst_ty = self._require_tile(dst)
            lhs = self._opnd(call.args[0])
            rhs = self._opnd(call.args[1])
            lhs_ty = self._require_tile(lhs)
            rhs_ty = self._require_tile(rhs)
            self._t.line(f"pto.tmatmul ins({lhs}, {rhs} : {lhs_ty}, {rhs_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tmatmul.acc":
            if len(call.args) != 3:
                raise FrontendError("tmatmul.acc/matmul_acc in assignment form expects: dst = tmatmul.acc(acc, lhs, rhs)")
            dst_ty = self._require_tile(dst)
            acc_in = self._opnd(call.args[0])
            lhs = self._opnd(call.args[1])
            rhs = self._opnd(call.args[2])
            acc_ty = self._require_tile(acc_in)
            lhs_ty = self._require_tile(lhs)
            rhs_ty = self._require_tile(rhs)
            self._t.line(f"pto.tmatmul.acc ins({acc_in}, {lhs}, {rhs} : {acc_ty}, {lhs_ty}, {rhs_ty}) outs({dst} : {dst_ty})")
            return

        if call.keywords:
            raise FrontendError(f"{fn}(...) does not support keyword args in kernels")

        raise FrontendError(f"unsupported assignment-form op: {fn}(...)")

    def _record_tensor_arg(self, *, name: str, arg_index: int, ty: TensorType) -> None:
        existing = self._tensor_args.get(arg_index)
        if existing is None:
            self._tensor_args[arg_index] = ty
            self._tensor_arg_names[arg_index] = name
            return
        if existing != ty:
            raise FrontendError(f"tensor arg {arg_index} redeclared with a different type")

    def _declare_tensor(self, target: str, call: ast.Call) -> None:
        # Accept both forms:
        #   x = tensor(dtype="f16", shape=(16,16))
        #   x = pto.tensor("x", (16,16), dtype="f16")
        dtype: str | None = None
        shape: Any | None = None
        stride: Any | None = None
        layout: str = "ND"
        role: str | None = None
        declared_name: str | None = None

        args = list(call.args)
        # Disambiguate old vs new:
        # - old: tensor("f16", (16,16), ...)
        # - new: pto.tensor("x", (16,16), dtype="f16", ...)
        has_dtype_kw = any(kw.arg == "dtype" for kw in call.keywords if kw.arg is not None)
        if len(args) >= 2 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str) and has_dtype_kw:
            declared_name = args[0].value
            shape = self._eval_static(args[1])
        else:
            if args:
                if len(args) >= 1:
                    dtype = self._eval_const(args[0])
                if len(args) >= 2:
                    shape = self._eval_static(args[1])
                if len(args) >= 3:
                    stride = self._eval_static(args[2])
                if len(args) >= 4:
                    layout = self._eval_const(args[3])

        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "shape":
                shape = self._eval_static(kw.value)
            elif kw.arg == "stride":
                stride = self._eval_static(kw.value)
            elif kw.arg == "layout":
                layout = self._eval_const(kw.value)
            elif kw.arg == "role":
                role = str(self._eval_const(kw.value))
                if role not in ("in", "out", "inout"):
                    raise FrontendError("tensor(..., role=...) must be one of: in, out, inout")
            elif kw.arg in ("arg", "arg_index"):
                # Parsed below (controls %argN binding).
                pass
            else:
                raise FrontendError(f"unknown tensor(...) kw: {kw.arg}")

        pto_name = declared_name or target
        if dtype is None or shape is None:
            raise FrontendError("tensor(...) requires dtype and shape")

        # `tensor(...)` in the Python frontend declares a kernel tensor argument, mapped
        # to `%argN` in declaration order, and introduces a view via `pto.make_tensor_view`.
        arg_index: int | None = None
        for kw in call.keywords:
            if kw.arg in ("arg", "arg_index"):
                arg_index = int(self._eval_const(kw.value))

        if arg_index is None:
            arg_index = self._next_tensor_arg
            self._next_tensor_arg += 1

        if arg_index not in self._tensor_arg_roles:
            self._tensor_arg_roles[arg_index] = role
        else:
            existing_role = self._tensor_arg_roles[arg_index]
            if role is not None and existing_role is not None and role != existing_role:
                raise FrontendError(f"tensor arg {arg_index} redeclared with a different role")
            if existing_role is None and role is not None:
                self._tensor_arg_roles[arg_index] = role

        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise FrontendError("tensor(...) currently expects shape=(H, W)")
        h, w = int(shape[0]), int(shape[1])

        if stride is None:
            s0, s1 = w, 1
        else:
            if not isinstance(stride, (tuple, list)) or len(stride) != 2:
                raise FrontendError("tensor(..., stride=...) expects stride=(S0, S1)")
            s0, s1 = int(stride[0]), int(stride[1])

        ty = TensorType(dtype=dtype, shape=(h, w), stride=(s0, s1), layout=layout)
        self._record_tensor_arg(name=pto_name, arg_index=arg_index, ty=ty)

        sym = self._bind_sym(target, pto_name)
        self._tensor_syms[sym.pto] = ty

        c1 = self._index_const(1)
        c_h = self._index_const(h)
        c_w = self._index_const(w)
        c_s0 = self._index_const(s0)
        c_s1 = self._index_const(s1)
        tv_ty = self._tensor_view_ty(dtype=dtype)
        self._t.line(
            f"{sym.pto} = pto.make_tensor_view %arg{arg_index}, "
            f"shape = [{c1}, {c1}, {c1}, {c_h}, {c_w}] "
            f"strides = [{c1}, {c1}, {c1}, {c_s0}, {c_s1}] : {tv_ty}"
        )

    def _declare_tile(self, target: str, call: ast.Call) -> None:
        # Legacy tile(...) helper.
        loc: str | None = None
        dtype: str | None = None
        rows: int | None = None
        cols: int | None = None

        blayout: str = "RowMajor"
        valid: str | None = None
        valid_row: str | None = None
        valid_col: str | None = None
        slayout: str = "NoneBox"
        fractal: int | None = None
        pad: str = "Null"
        addr: int | None = None

        args = list(call.args)
        if args:
            if len(args) >= 1:
                loc = self._eval_const(args[0])
            if len(args) >= 2:
                dtype = self._eval_const(args[1])
            if len(args) >= 3:
                rows = int(self._eval_static(args[2]))
            if len(args) >= 4:
                cols = int(self._eval_static(args[3]))

        for kw in call.keywords:
            if kw.arg == "loc":
                loc = self._eval_const(kw.value)
            elif kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "rows":
                rows = int(self._eval_static(kw.value))
            elif kw.arg == "cols":
                cols = int(self._eval_static(kw.value))
            elif kw.arg == "blayout":
                blayout = self._eval_const(kw.value)
            elif kw.arg == "valid":
                valid = self._eval_const(kw.value)
            elif kw.arg == "valid_row":
                valid_row = self._opnd(kw.value)
            elif kw.arg == "valid_col":
                valid_col = self._opnd(kw.value)
            elif kw.arg == "slayout":
                slayout = self._eval_const(kw.value)
            elif kw.arg == "fractal":
                fractal = self._eval_const(kw.value)
            elif kw.arg == "pad":
                pad = self._eval_const(kw.value)
            elif kw.arg == "addr":
                addr = int(self._eval_const(kw.value))
            else:
                raise FrontendError(f"unknown tile(...) kw: {kw.arg}")

        if loc is None or dtype is None or rows is None or cols is None:
            raise FrontendError("tile(...) requires loc, dtype, rows, cols")

        sym = self._sym_for(target)
        if (valid_row is not None or valid_col is not None) and valid is not None:
            raise FrontendError("tile(...): cannot combine valid=... with valid_row=.../valid_col=...")

        if valid_row is not None or valid_col is not None:
            ty = TileType(
                loc=loc,
                dtype=dtype,
                rows=rows,
                cols=cols,
                blayout=blayout,
                v_row="dyn" if valid_row is not None else rows,
                v_col="dyn" if valid_col is not None else cols,
                slayout=slayout,
                fractal=fractal,
                pad=pad,
            )
            self._tile_syms[sym.pto] = ty

            opts: list[str] = []
            if addr is not None:
                opts.append(f"addr={addr}")
            if valid_row is not None:
                opts.append(f"valid_row={valid_row}")
            if valid_col is not None:
                opts.append(f"valid_col={valid_col}")
            self._t.line(f"{sym.pto} = pto.alloc_tile {' '.join(opts)} : {ty}".rstrip())
            return

        if valid is not None:
            if isinstance(valid, str) and "x" in valid:
                vr, vc = valid.split("x", 1)
                valid_rows = int(vr)
                valid_cols = int(vc)
            else:
                raise FrontendError("tile(..., valid=...) must be like '16x16'")
        else:
            valid_rows = None
            valid_cols = None

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
        self._tile_syms[sym.pto] = ty
        if addr is None:
            self._t.line(f"{sym.pto} = pto.alloc_tile : {ty}")
        else:
            self._t.line(f"{sym.pto} = pto.alloc_tile {addr} : {ty}")

    def _declare_tile_sugar(self, target: str, call: ast.Call, *, loc: str) -> None:
        # New object-DSL helpers:
        #   q_tile = pto.vec("q_tile", dtype="f32", shape=(s,d))
        declared_name: str | None = None
        dtype: str | None = None
        shape: Any | None = None

        # Per-loc defaults chosen to match the common NPU expectations for TMATMUL pipelines.
        if loc == "Mat":
            blayout = "ColMajor"
            slayout = "RowMajor"
        elif loc == "Left":
            blayout = "RowMajor"
            slayout = "RowMajor"
        elif loc == "Right":
            blayout = "RowMajor"
            slayout = "ColMajor"
        elif loc == "Acc":
            blayout = "ColMajor"
            slayout = "RowMajor"
        else:
            blayout = "RowMajor"
            slayout = "NoneBox"

        valid: str | None = None
        valid_row: str | None = None
        valid_col: str | None = None
        fractal: int | None = None
        pad: str = "Null"
        addr: int | None = None

        args = list(call.args)
        if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            declared_name = args[0].value
        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self._eval_const(kw.value)
            elif kw.arg == "shape":
                shape = self._eval_static(kw.value)
            elif kw.arg == "blayout":
                blayout = self._eval_const(kw.value)
            elif kw.arg == "valid":
                valid = self._eval_const(kw.value)
            elif kw.arg == "valid_row":
                valid_row = self._opnd(kw.value)
            elif kw.arg == "valid_col":
                valid_col = self._opnd(kw.value)
            elif kw.arg == "slayout":
                slayout = self._eval_const(kw.value)
            elif kw.arg == "fractal":
                fractal = int(self._eval_static(kw.value))
            elif kw.arg == "pad":
                pad = self._eval_const(kw.value)
            elif kw.arg == "addr":
                addr = int(self._eval_const(kw.value))
            elif kw.arg == "b":
                # Convenience annotation used by some higher-level examples (e.g. softmax broadcast axis).
                # PTO-AS tile types do not currently encode broadcast semantics, so this is ignored.
                _ = kw.value
            else:
                raise FrontendError(f"unknown {call.func} kw: {kw.arg}")

        pto_name = declared_name or target
        if dtype is None or shape is None:
            raise FrontendError("vec_tile/left_tile/right_tile/acc_tile require dtype=... and shape=(H,W)")
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise FrontendError("tile shape must be (rows, cols)")
        rows, cols = int(shape[0]), int(shape[1])

        sym = self._bind_sym(target, pto_name)
        if (valid_row is not None or valid_col is not None) and valid is not None:
            raise FrontendError(f"{call.func}(...): cannot combine valid=... with valid_row=.../valid_col=...")

        if valid_row is not None or valid_col is not None:
            ty = TileType(
                loc=loc,
                dtype=dtype,
                rows=rows,
                cols=cols,
                blayout=blayout,
                v_row="dyn" if valid_row is not None else rows,
                v_col="dyn" if valid_col is not None else cols,
                slayout=slayout,
                fractal=fractal,
                pad=pad,
            )
            self._tile_syms[sym.pto] = ty

            opts: list[str] = []
            if addr is not None:
                opts.append(f"addr={addr}")
            if valid_row is not None:
                opts.append(f"valid_row={valid_row}")
            if valid_col is not None:
                opts.append(f"valid_col={valid_col}")
            self._t.line(f"{sym.pto} = pto.alloc_tile {' '.join(opts)} : {ty}".rstrip())
            return

        if valid is not None:
            if isinstance(valid, str) and "x" in valid:
                vr, vc = valid.split("x", 1)
                valid_rows = int(vr)
                valid_cols = int(vc)
            else:
                raise FrontendError("tile(..., valid=...) must be like '16x16'")
        else:
            valid_rows = None
            valid_cols = None

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
        self._tile_syms[sym.pto] = ty
        if addr is None:
            self._t.line(f"{sym.pto} = pto.alloc_tile : {ty}")
        else:
            self._t.line(f"{sym.pto} = pto.alloc_tile {addr} : {ty}")

    def _emit_scalar_assign(self, dst: str, value: ast.AST) -> None:
        dst_sym = self._sym_for(dst)
        if isinstance(value, ast.Call):
            fn = self._call_name(value)
            if fn == "get_block_idx":
                self._t.line(f"{dst_sym.pto} = pto.get_block_idx : index")
                return
            if fn == "get_block_num":
                self._t.line(f"{dst_sym.pto} = pto.get_block_num : index")
                return
        if isinstance(value, ast.BinOp):
            lhs = self._opnd(value.left)
            rhs = self._opnd(value.right)
            if isinstance(value.op, ast.Add):
                self._t.line(f"{dst_sym.pto} = arith.addi {lhs}, {rhs} : index")
                return
            if isinstance(value.op, ast.Sub):
                self._t.line(f"{dst_sym.pto} = arith.subi {lhs}, {rhs} : index")
                return
            if isinstance(value.op, ast.Mult):
                self._t.line(f"{dst_sym.pto} = arith.muli {lhs}, {rhs} : index")
                return
            if isinstance(value.op, ast.FloorDiv):
                self._t.line(f"{dst_sym.pto} = arith.divsi {lhs}, {rhs} : index")
                return
            if isinstance(value.op, ast.Mod):
                self._t.line(f"{dst_sym.pto} = arith.remsi {lhs}, {rhs} : index")
                return
        raise FrontendError(f"unsupported scalar assignment: {dst} = {ast.dump(value)}")

    def _emit_instr_stmt(self, call: ast.Call) -> None:
        fn = self._opcode_alias(self._call_name(call))

        if fn in ("prologue", "epilogue"):
            return

        def opnds() -> list[str]:
            return [self._opnd(a) for a in call.args]

        if fn == "tassign":
            raise FrontendError(
                "tassign(...) is not supported in the new PTO-AS syntax; "
                "use tile(..., addr=0x...) or omit `addr` and run ptoas with --assign-tile-addrs"
            )

        if fn == "comment":
            if len(call.args) != 1:
                raise FrontendError('comment("...") expects one string argument')
            text = self._eval_const(call.args[0])
            if not isinstance(text, str):
                raise FrontendError('comment("...") expects one string argument')
            for line in text.splitlines():
                self._t.line(f"// {line}" if line else "//")
            return

        if fn in ("kernel", "kernel_name"):
            if len(call.args) != 1:
                raise FrontendError('kernel("name") expects one string argument')
            name0 = self._eval_const(call.args[0])
            if not isinstance(name0, str) or not name0:
                raise FrontendError('kernel("name") expects one non-empty string argument')
            self._explicit_kernel_name = name0
            return

        if fn == "program":
            # Object-DSL convention: `return pto.program()`; ignored by the compiler.
            return

        if fn == "tload":
            if len(call.args) not in (2, 4):
                raise FrontendError("tload(dst_tile, src_tensor, [r, c])")
            dst = self._opnd(call.args[0])
            dst_ty = self._require_tile(dst)
            src = self._opnd(call.args[1])
            src_ty = self._require_tensor(src)
            if dst_ty.dtype != src_ty.dtype:
                raise FrontendError(f"tload dtype mismatch: dst={dst_ty.dtype} src={src_ty.dtype}")
            r = self._index_const(0)
            c = self._index_const(0)
            if len(call.args) == 4:
                r = self._opnd(call.args[2])
                c = self._opnd(call.args[3])
            sub = self._emit_subview(
                tensor=src,
                r=r,
                c=c,
                rows=int(dst_ty.rows),
                cols=int(dst_ty.cols),
                dtype=src_ty.dtype,
            )
            tile_v_ty = self._tile_view_ty(rows=int(dst_ty.rows), cols=int(dst_ty.cols), dtype=src_ty.dtype)
            self._t.line(f"pto.tload ins({sub} : {tile_v_ty}) outs({dst} : {dst_ty})")
            return
        if fn == "tstore":
            if len(call.args) not in (2, 4):
                raise FrontendError("tstore(dst_tensor, [r, c,] src_tile)")
            dst = self._opnd(call.args[0])
            dst_ty = self._require_tensor(dst)
            r = self._index_const(0)
            c = self._index_const(0)
            src = self._opnd(call.args[-1])
            src_ty = self._require_tile(src)
            if src_ty.dtype != dst_ty.dtype:
                raise FrontendError(f"tstore dtype mismatch: src={src_ty.dtype} dst={dst_ty.dtype}")
            if len(call.args) == 4:
                r = self._opnd(call.args[1])
                c = self._opnd(call.args[2])
            sub = self._emit_subview(
                tensor=dst,
                r=r,
                c=c,
                rows=int(src_ty.rows),
                cols=int(src_ty.cols),
                dtype=dst_ty.dtype,
            )
            tile_v_ty = self._tile_view_ty(rows=int(src_ty.rows), cols=int(src_ty.cols), dtype=dst_ty.dtype)
            self._t.line(f"pto.tstore ins({src} : {src_ty}) outs({sub} : {tile_v_ty})")
            return

        if fn == "tmov":
            if len(call.args) != 2:
                raise FrontendError("tmov(dst, src)")
            dst = self._opnd(call.args[0])
            src = self._opnd(call.args[1])
            dst_ty = self._require_tile(dst)
            src_ty = self._require_tile(src)
            self._t.line(f"pto.tmov ins({src} : {src_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tmatmul":
            if len(call.args) != 3:
                raise FrontendError("tmatmul(dst, lhs, rhs)")
            dst = self._opnd(call.args[0])
            lhs = self._opnd(call.args[1])
            rhs = self._opnd(call.args[2])
            dst_ty = self._require_tile(dst)
            lhs_ty = self._require_tile(lhs)
            rhs_ty = self._require_tile(rhs)
            self._t.line(f"pto.tmatmul ins({lhs}, {rhs} : {lhs_ty}, {rhs_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tmatmul.acc":
            if len(call.args) != 4:
                raise FrontendError("tmatmul.acc(dst, acc, lhs, rhs)")
            dst = self._opnd(call.args[0])
            acc_in = self._opnd(call.args[1])
            lhs = self._opnd(call.args[2])
            rhs = self._opnd(call.args[3])
            dst_ty = self._require_tile(dst)
            acc_ty = self._require_tile(acc_in)
            lhs_ty = self._require_tile(lhs)
            rhs_ty = self._require_tile(rhs)
            self._t.line(f"pto.tmatmul.acc ins({acc_in}, {lhs}, {rhs} : {acc_ty}, {lhs_ty}, {rhs_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tadd":
            if len(call.args) != 3:
                raise FrontendError("tadd(dst, a, b)")
            dst = self._opnd(call.args[0])
            a0 = self._opnd(call.args[1])
            a1 = self._opnd(call.args[2])
            dst_ty = self._require_tile(dst)
            a0_ty = self._require_tile(a0)
            a1_ty = self._require_tile(a1)
            self._t.line(f"pto.tadd ins({a0}, {a1} : {a0_ty}, {a1_ty}) outs({dst} : {dst_ty})")
            return

        if fn == "tpush":
            if len(call.args) != 3:
                raise FrontendError("tpush(dst_tensor, src_tile, token)")
            dst = self._opnd(call.args[0])
            src = self._opnd(call.args[1])
            token = self._opnd(call.args[2])
            self._t.line(f"pto.tpush {dst}, {src}, {token}")
            return

        if fn in ("record_event", "wait_event"):
            if call.args:
                raise FrontendError(f"{fn}(...) only supports keyword args: src_op=..., dst_op=..., token=...")
            kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            if set(kwargs) != {"src_op", "dst_op", "token"}:
                raise FrontendError(f"{fn}(...) requires keyword args: src_op, dst_op, token")
            src_op = self._eval_const(kwargs["src_op"])
            dst_op = self._eval_const(kwargs["dst_op"])
            token = self._opnd(kwargs["token"])
            if not isinstance(src_op, str) or not isinstance(dst_op, str):
                raise FrontendError(f"{fn}(src_op=..., dst_op=...) must be string literals")
            # Keep this aligned with CCEmitter's `emitRecordOrWait(...)` parsing:
            # accept both `#pto.op<TLOAD>` and `#op<TLOAD>` spellings.
            self._t.line(f"pto.{fn} {{src_op=#op<{src_op}>, dst_op=#op<{dst_op}>, token={token}}}")
            return

        if fn == "tsync":
            if call.args:
                raise FrontendError('tsync(...) only supports keyword args: pipe="V"')
            if not call.keywords:
                self._t.line("pto.tsync")
                return
            kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            if set(kwargs) != {"pipe"}:
                raise FrontendError("tsync(...) only supports keyword args: pipe=...")
            pipe = self._eval_const(kwargs["pipe"])
            if not isinstance(pipe, str) or not pipe:
                raise FrontendError("tsync(pipe=...) must be a non-empty string literal")
            if pipe != "V":
                # A2/A3 single-pipe barriers only support Vector (see `include/pto/npu/a2a3/TSync.hpp`).
                raise FrontendError('tsync(pipe=...) only supports pipe="V" on A2/A3')
            self._t.line(f'pto.tsync {{pipe="{pipe}"}}')
            return

        raise FrontendError(f"unsupported statement-form op: {fn}(...)")

    def _emit_if(self, stmt: ast.If) -> None:
        # Only support simple compare -> icmp_* -> scf.if.
        if not isinstance(stmt.test, ast.Compare) or len(stmt.test.ops) != 1 or len(stmt.test.comparators) != 1:
            raise FrontendError("if condition must be a simple compare, e.g. if a < b:")

        lhs = self._opnd(stmt.test.left)
        rhs = self._opnd(stmt.test.comparators[0])
        op = stmt.test.ops[0]
        pred = None
        if isinstance(op, ast.Eq):
            pred = "eq"
        elif isinstance(op, ast.NotEq):
            pred = "ne"
        elif isinstance(op, ast.Lt):
            pred = "slt"
        elif isinstance(op, ast.LtE):
            pred = "sle"
        elif isinstance(op, ast.Gt):
            pred = "sgt"
        elif isinstance(op, ast.GtE):
            pred = "sge"
        else:
            raise FrontendError("unsupported compare op")

        cond = self._tmp()
        self._t.line(f"{cond.pto} = arith.cmpi {pred}, {lhs}, {rhs} : index")
        self._t.open(f"scf.if {cond.pto}")
        self._emit_stmts(stmt.body)
        if stmt.orelse:
            self._t.else_open()
            self._emit_stmts(stmt.orelse)
        self._t.close()

    def _emit_for(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise FrontendError("for target must be a name")
        if not isinstance(stmt.iter, ast.Call) or not isinstance(stmt.iter.func, ast.Name) or stmt.iter.func.id != "range":
            raise FrontendError("for must iterate over range(...)")

        args = stmt.iter.args
        if len(args) == 1:
            start, stop, step = ast.Constant(value=0), args[0], ast.Constant(value=1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], ast.Constant(value=1)
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise FrontendError("range expects 1..3 args")

        iv = self._sym_for(stmt.target.id)
        lb = self._opnd(start)
        ub = self._opnd(stop)
        st = self._opnd(step)
        self._t.open(f"scf.for {iv.pto} = {lb} to {ub} step {st}")
        self._emit_stmts(stmt.body)
        self._t.close()

    def _emit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Return):
            # The frontend is statement-driven; returning a "program()" object is a convention
            # in the object DSL but does not affect PTO-AS emission.
            return

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise FrontendError("only simple assignments to a name are supported")
            dst = stmt.targets[0].id

            if isinstance(stmt.value, ast.Call):
                fn = self._call_name(stmt.value)
                if fn == "PTO":
                    # `pto = PTO("name")` is the object-DSL entrypoint; record the preferred
                    # output kernel name and emit no PTO-AS for it.
                    if stmt.value.args:
                        name0 = self._eval_const(stmt.value.args[0])
                        if isinstance(name0, str):
                            self._explicit_kernel_name = name0
                    return
                if fn == "tensor":
                    self._declare_tensor(dst, stmt.value)
                    return
                if fn == "tile":
                    self._declare_tile(dst, stmt.value)
                    return
                if fn in ("vec_tile", "left_tile", "right_tile", "acc_tile", "mat_tile"):
                    loc_map = {
                        "vec_tile": "Vec",
                        "left_tile": "Left",
                        "right_tile": "Right",
                        "acc_tile": "Acc",
                        "mat_tile": "Mat",
                    }
                    self._declare_tile_sugar(dst, stmt.value, loc=loc_map[fn])
                    return
                if fn in ("vec", "mat", "left", "right", "acc"):
                    loc_map = {"vec": "Vec", "mat": "Mat", "left": "Left", "right": "Right", "acc": "Acc"}
                    self._declare_tile_sugar(dst, stmt.value, loc=loc_map[fn])
                    return
                if fn == "const":
                    # `scale = pto.const("scale", 1.0 / sqrt(d), scalar("f32"))`
                    if len(stmt.value.args) < 2:
                        raise FrontendError("const(name, value, [type]) expects at least 2 args")
                    value = self._eval_static(stmt.value.args[1])
                    if not isinstance(value, (int, float, bool)):
                        raise FrontendError("const(..., value, ...) must evaluate to a number/bool")
                    lit = "1" if value is True else "0" if value is False else repr(value)
                    if dst in self._sym:
                        raise FrontendError(f"cannot rebind existing SSA symbol as a literal: {dst}")
                    self._literal[dst] = lit
                    self._const_env[dst] = value
                    return

                # Left-assignment style instruction: `dst = pto.op(...)`.
                self._emit_instr_assign(dst_name=dst, call=stmt.value)
                return

            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, bool, float)):
                # Inline numeric literals directly (new PTO-AS removes `.const`).
                if dst in self._sym:
                    raise FrontendError(f"cannot rebind existing SSA symbol as a literal: {dst}")
                v = stmt.value.value
                if isinstance(v, bool):
                    self._literal[dst] = "1" if v else "0"
                else:
                    self._literal[dst] = repr(v) if isinstance(v, float) else str(v)
                self._const_env[dst] = v
                return
            self._emit_scalar_assign(dst, stmt.value)
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._emit_instr_stmt(stmt.value)
            return

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            # Ignore docstrings / bare string expression statements inside kernels.
            # This keeps the frontend friendlier to normal Python style.
            return

        if isinstance(stmt, ast.For):
            self._emit_for(stmt)
            return

        if isinstance(stmt, ast.If):
            self._emit_if(stmt)
            return

        raise FrontendError(f"unsupported statement: {ast.dump(stmt)}")

    def _emit_stmts(self, stmts: list[ast.stmt]) -> None:
        for s in stmts:
            self._emit_stmt(s)

    def _tensor_arg_list(self) -> list[TensorArg]:
        args: list[TensorArg] = []
        for idx in sorted(self._tensor_args):
            name = self._tensor_arg_names.get(idx, f"arg{idx}")
            args.append(TensorArg(name=name, arg_index=idx, ty=self._tensor_args[idx], role=self._tensor_arg_roles.get(idx)))
        return args

    def compile_funcdef(self, fn: ast.FunctionDef) -> KernelSpec:
        self._emit_stmts(fn.body)
        name = self._explicit_kernel_name or fn.name

        if self._uses_task_id and 0 in self._tensor_args:
            raise FrontendError("kernel uses get_task_id(), so %arg0 is reserved (tensors must start at %arg1)")

        # Function signature: keep argument ordering stable and aligned with `%argN` usage.
        fn_args: list[str] = []
        if self._uses_task_id:
            fn_args.append("%arg0: index")
        for idx in sorted(self._tensor_args):
            dt = self._tensor_args[idx].dtype
            fn_args.append(f"%arg{idx}: !pto.ptr<{dt}>")

        body = self._t.emit().rstrip("\n")
        body_indented = "\n".join(("    " + ln) if ln else "" for ln in body.splitlines())
        if body_indented:
            body_indented += "\n"

        # Default to Ascend910B1 sizing unless overridden by the caller/tooling.
        # This affects memory planning passes in llvm-project ptoas.
        pto = "\n".join(
            [
                'module attributes {"pto.device-spec" = "Ascend910B1"} {',
                f"  func.func @{name}({', '.join(fn_args)}) {{",
                body_indented.rstrip("\n"),
                "    return",
                "  }",
                "}",
                "",
            ]
        )
        return KernelSpec(name=name, pto=pto, tensor_args=tuple(self._tensor_arg_list()))


def list_kernel_functions(source: str) -> list[str]:
    module = ast.parse(textwrap.dedent(source))
    return [n.name for n in module.body if isinstance(n, ast.FunctionDef)]


def list_kernel_functions_from_file(path: Path) -> list[str]:
    return list_kernel_functions(path.read_text(encoding="utf-8"))


def compile_kernel_spec_from_source(source: str, *, func_name: str) -> KernelSpec:
    m = ast.parse(textwrap.dedent(source))
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if not fns:
        raise FrontendError(f"function not found: {func_name}")
    if len(fns) != 1:
        raise FrontendError(f"ambiguous function: {func_name}")
    return _Compiler().compile_funcdef(fns[0])


def compile_kernel_spec_from_source_with_consts(source: str, *, func_name: str, consts: dict[str, Any]) -> KernelSpec:
    """
    Like `compile_kernel_spec_from_source`, but with injected compile-time constants.

    Prefer `compile_kernel_spec(..., consts=...)` when starting from a function object.
    """
    m = ast.parse(textwrap.dedent(source))
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if not fns:
        raise FrontendError(f"function not found: {func_name}")
    if len(fns) != 1:
        raise FrontendError(f"ambiguous function: {func_name}")
    return _Compiler(consts=consts).compile_funcdef(fns[0])


def compile_kernel_spec_from_file(path: Path, *, func_name: str) -> KernelSpec:
    return compile_kernel_spec_from_source(path.read_text(encoding="utf-8"), func_name=func_name)


def compile_kernel_from_source(source: str, *, func_name: str) -> str:
    return compile_kernel_spec_from_source(source, func_name=func_name).pto


def compile_kernel(func: Callable[..., Any]) -> str:
    src = inspect.getsource(func)
    return compile_kernel_from_source(src, func_name=func.__name__)


def compile_kernel_spec(func: Callable[..., Any], *, consts: dict[str, Any] | None = None) -> KernelSpec:
    src = inspect.getsource(func)
    if consts is None:
        return compile_kernel_spec_from_source(src, func_name=func.__name__)
    return compile_kernel_spec_from_source_with_consts(src, func_name=func.__name__, consts=consts)


def make_add16_program() -> str:
    return compile_kernel_from_source(
        '''
def add16():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()
    r0 = bid * 16

    x = tensor(dtype="f16", shape=(16, 16))
    y = tensor(dtype="f16", shape=(16, 16))
    z = tensor(dtype="f16", shape=(16, 16))
    tx = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    ty = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    tz = tile(loc="Vec", dtype="f16", rows=16, cols=16)

    tload(tx, x, r0, 0)
    tload(ty, y, r0, 0)
    tadd(tz, tx, ty)
    tstore(z, r0, 0, tz)
    epilogue()
''',
        func_name="add16",
    )


def make_gemm16_program() -> str:
    return compile_kernel_from_source(
        '''
def gemm16():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)
    epilogue()
''',
        func_name="gemm16",
    )


def make_gemm16_cpu_program() -> str:
    # CPU simulator uses different matrix fractal constraints for TMATMUL.
    return compile_kernel_from_source(
        '''
def gemm16_cpu():
    prologue()
    bn = get_block_num()
    bid = get_block_idx()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)
    epilogue()
''',
        func_name="gemm16_cpu",
    )
