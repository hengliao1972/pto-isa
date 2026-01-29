from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Iterable

from compile.pto_compile_common import PTOProgram
from isa_definition.pto_isa_definition import (
    ElementType,
    ImmediateOperand,
    IndexOperand,
    MemRefOperand,
    ScalarOperand,
    TileInstruction,
    TileOperand,
    TLOAD,
    TMATMUL,
    TMATMUL_ACC,
    TMATMUL_BIAS,
    TMATMUL_MX,
    TMOV,
    TSTORE,
)


def _dtype_str(dt: ElementType) -> str:
    return str(dt.value)


def _idx_str(x: ImmediateOperand | IndexOperand) -> str:
    if isinstance(x, ImmediateOperand):
        return str(x.value)
    if isinstance(x, IndexOperand):
        return f"%{x.name}"
    raise TypeError(f"unsupported index operand: {type(x).__name__}")


@dataclass(frozen=True)
class _TileBindings:
    # Map a logical PTO tile name (without leading %) to concrete PTO-AS tile symbols.
    #
    # For GEMM we lower:
    #   a (logical) -> %a_mat, %a_left
    #   b (logical) -> %b_mat, %b_right
    #   c (logical) -> %c_acc
    mat: dict[str, str]
    left: dict[str, str]
    right: dict[str, str]
    acc: dict[str, str]

    def tile_for_load(self, name: str) -> str:
        # Prefer loading matmul operands into Mat tiles (then TMOV -> Left/Right).
        if name in self.mat:
            return self.mat[name]
        return f"%{name}"

    def tile_for_matmul_a(self, name: str) -> tuple[str, str]:
        # Returns (src_mat, dst_left)
        return (self.mat[name], self.left[name])

    def tile_for_matmul_b(self, name: str) -> tuple[str, str]:
        return (self.mat[name], self.right[name])

    def tile_for_store(self, name: str) -> str:
        if name in self.acc:
            return self.acc[name]
        return f"%{name}"

    def tile_for_matmul_dst(self, name: str) -> str:
        return self.acc[name]

    def tile_for_operand(self, name: str) -> str:
        # Prefer the most "semantic" mapping when a logical tile participates in matmul lowering.
        if name in self.acc:
            return self.acc[name]
        if name in self.left:
            return self.left[name]
        if name in self.right:
            return self.right[name]
        if name in self.mat:
            return self.mat[name]
        return f"%{name}"


def _infer_memref_shapes_from_memops(program: PTOProgram) -> dict[str, tuple[int, int]]:
    """
    Infer minimal tensor shapes for memrefs from TLOAD/TSTORE uses.

    If a memref has an explicit shape, that shape is used.
    If a memref is only accessed with dynamic (index) offsets, shape inference fails.
    """
    out: dict[str, tuple[int, int]] = {}
    # Start from explicit shapes.
    for name, mty in program.memref_declarations.items():
        if mty.shape is not None:
            out[name] = (int(mty.shape.rows), int(mty.shape.cols))

    def _ensure_min(name: str, need_rows: int | None, need_cols: int | None) -> None:
        if need_rows is None or need_cols is None:
            # Can't infer (dynamic offsets), keep unset to force explicit shape.
            return
        prev = out.get(name)
        if prev is None:
            out[name] = (int(need_rows), int(need_cols))
            return
        out[name] = (max(prev[0], int(need_rows)), max(prev[1], int(need_cols)))

    # Infer from memops.
    for ins in program.instructions:
        if isinstance(ins, TLOAD):
            mem = ins.src_mem.name
            if mem in out:
                continue
            rows = int(ins.dst.tile_type.shape.rows)
            cols = int(ins.dst.tile_type.shape.cols)
            r0 = ins.row_offset.value if isinstance(ins.row_offset, ImmediateOperand) else None
            c0 = ins.col_offset.value if isinstance(ins.col_offset, ImmediateOperand) else None
            _ensure_min(mem, (int(r0) + rows) if r0 is not None else None, (int(c0) + cols) if c0 is not None else None)
        elif isinstance(ins, TSTORE):
            mem = ins.dst_mem.name
            if mem in out:
                continue
            rows = int(ins.src.tile_type.shape.rows)
            cols = int(ins.src.tile_type.shape.cols)
            r0 = ins.row_offset.value if isinstance(ins.row_offset, ImmediateOperand) else None
            c0 = ins.col_offset.value if isinstance(ins.col_offset, ImmediateOperand) else None
            _ensure_min(mem, (int(r0) + rows) if r0 is not None else None, (int(c0) + cols) if c0 is not None else None)

    return out


def _emit_ptoas_tile_decls(*, out, program: PTOProgram, tb: _TileBindings) -> None:
    # Import locally to keep `compile/` usable without the ptoas toolchain for other paths.
    from ptoas.python.pto_asm import TileType

    matmul_tiles: set[str] = set(tb.mat) | set(tb.left) | set(tb.right) | set(tb.acc)
    for tname, tty in program.tile_declarations.items():
        if tname in matmul_tiles:
            # Matmul tiles are declared separately below.
            continue
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)
        out.alloc_tile(f"%{tname}", TileType(loc="Vec", dtype=dt, rows=rows, cols=cols, blayout="RowMajor", slayout="NoneBox"))

    # Matmul lowering: allocate Mat/Left/Right/Acc tiles with proven layouts.
    for name in tb.mat:
        tty = program.tile_declarations[name]
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)
        out.alloc_tile(tb.mat[name], TileType(loc="Mat", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))
    for name in tb.left:
        tty = program.tile_declarations[name]
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)
        out.alloc_tile(tb.left[name], TileType(loc="Left", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))
    for name in tb.right:
        tty = program.tile_declarations[name]
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)
        out.alloc_tile(tb.right[name], TileType(loc="Right", dtype=dt, rows=rows, cols=cols, blayout="RowMajor", slayout="ColMajor"))
    for name in tb.acc:
        tty = program.tile_declarations[name]
        rows = int(tty.shape.rows)
        cols = int(tty.shape.cols)
        dt = _dtype_str(tty.element_type)
        out.alloc_tile(tb.acc[name], TileType(loc="Acc", dtype=dt, rows=rows, cols=cols, blayout="ColMajor", slayout="RowMajor"))


def _operand_str(tb: _TileBindings, x: object) -> str:
    if isinstance(x, TileOperand):
        return tb.tile_for_operand(x.name)
    if isinstance(x, MemRefOperand):
        return f"%{x.name}"
    if isinstance(x, ScalarOperand):
        # PTO-AS uses `index`/`i1` scalars; the old compiler's scalar model does not map 1:1.
        # Only support scalars when they are used purely as index values and already defined.
        return f"%{x.name}"
    if isinstance(x, IndexOperand):
        return f"%{x.name}"
    if isinstance(x, ImmediateOperand):
        return str(x.value)
    raise TypeError(f"unsupported operand: {type(x).__name__}")


def export_program_to_ptoas(*, program: PTOProgram) -> str:
    """
    Export an *old* `compile.pto_compile_common.PTOProgram` into the *new* PTO-AS text format accepted by `ptoas`.

    Scope:
    - Intended for **InCore** functions only.
    - Supports straight-line Tile ISA programs with:
        - TLOAD / TSTORE
        - most tile ops with simple operands (e.g. trowmax, texp, tadd, ...)
        - TMATMUL / TMATMUL_ACC (lowered via Mat->Left/Right -> Acc)

    Notes:
    - Memref shapes are inferred from TLOAD/TSTORE when not explicitly specified on the memref declaration.
      If your kernel uses dynamic offsets (IndexOperand) without explicit memref shapes, export will fail.
    """
    from ptoas.python.pto_asm import PTOProgram as PTOASProgram
    from ptoas.python.pto_asm import TensorType

    # --- Arg order + shapes (required for make_tensor_view). ---
    memrefs = list(program.memref_declarations.items())
    if not memrefs:
        raise ValueError("program has no memref declarations")

    shapes = _infer_memref_shapes_from_memops(program)
    for name, _ in memrefs:
        if name not in shapes:
            raise ValueError(
                f"memref {name} missing shape and cannot be inferred; "
                "pass shape=(H,W) in PTOFunctionBuilder.memref(...) or use only immediate offsets"
            )

    # --- Identify matmul tiles (to pick TileType::Mat/Left/Right/Acc). ---
    matmul_a: set[str] = set()
    matmul_b: set[str] = set()
    matmul_dst: set[str] = set()
    for ins in program.instructions:
        if isinstance(ins, (TMATMUL, TMATMUL_ACC, TMATMUL_BIAS, TMATMUL_MX)):
            matmul_a.add(ins.a.name)  # type: ignore[attr-defined]
            matmul_b.add(ins.b.name)  # type: ignore[attr-defined]
            matmul_dst.add(ins.dst.name)  # type: ignore[attr-defined]
            if isinstance(ins, TMATMUL_ACC):
                matmul_dst.add(ins.acc.name)

    # Bind logical tiles to concrete PTO-AS tiles.
    mat: dict[str, str] = {}
    left: dict[str, str] = {}
    right: dict[str, str] = {}
    acc: dict[str, str] = {}
    for tname in program.tile_declarations:
        if tname in matmul_a:
            mat[tname] = f"%{tname}_mat"
            left[tname] = f"%{tname}_left"
        if tname in matmul_b:
            if tname in matmul_a:
                raise ValueError(f"tile {tname} used as both matmul A and B operand")
            mat[tname] = f"%{tname}_mat"
            right[tname] = f"%{tname}_right"
        if tname in matmul_dst:
            acc[tname] = f"%{tname}_acc"

    tb = _TileBindings(mat=mat, left=left, right=right, acc=acc)

    # --- Emit PTO-AS ---
    out = PTOASProgram()
    out.comment(f"Generated from old PTOFunctionBuilder program: {program.name}")
    out.prologue()

    # Tensor views in arg order.
    for arg_i, (name, mty) in enumerate(memrefs):
        dt = _dtype_str(mty.element_type)
        h, w = shapes[name]
        out.make_tensor_view(view=f"%{name}", arg_index=int(arg_i), ty=TensorType(dtype=dt, shape=(h, w)))

    # Tile allocations.
    _emit_ptoas_tile_decls(out=out, program=program, tb=tb)

    # Instructions.
    for ins in program.instructions:
        if isinstance(ins, TLOAD):
            dst = tb.tile_for_load(ins.dst.name)
            src = f"%{ins.src_mem.name}"
            r0 = _idx_str(ins.row_offset)
            c0 = _idx_str(ins.col_offset)
            out.assign(dst, "tload", [f"{src}[{r0}, {c0}]"])
            continue

        if isinstance(ins, TMATMUL):
            # Lower: Mat -> Left/Right via TMOV, then TMATMUL into Acc.
            a_mat, a_left = tb.tile_for_matmul_a(ins.a.name)
            b_mat, b_right = tb.tile_for_matmul_b(ins.b.name)
            out.assign(a_left, "tmov", [a_mat])
            out.assign(b_right, "tmov", [b_mat])
            out.assign(tb.tile_for_matmul_dst(ins.dst.name), "tmatmul", [a_left, b_right])
            continue

        if isinstance(ins, TMATMUL_ACC):
            a_mat, a_left = tb.tile_for_matmul_a(ins.a.name)
            b_mat, b_right = tb.tile_for_matmul_b(ins.b.name)
            out.assign(a_left, "tmov", [a_mat])
            out.assign(b_right, "tmov", [b_mat])
            dst = tb.tile_for_matmul_dst(ins.dst.name)
            acc_tile = tb.tile_for_matmul_dst(ins.acc.name)
            out.assign(dst, "tmatmul_acc", [acc_tile, a_left, b_right])
            continue

        if isinstance(ins, TMOV):
            # PTO-AS `tmov` generally infers mode from tile types. Drop explicit mode.
            out.assign(tb.tile_for_operand(ins.dst.name), "tmov", [tb.tile_for_operand(ins.src.name)])
            continue

        if isinstance(ins, TSTORE):
            src = tb.tile_for_store(ins.src.name)
            dst = f"%{ins.dst_mem.name}"
            r0 = _idx_str(ins.row_offset)
            c0 = _idx_str(ins.col_offset)
            out.op("tstore", [f"{dst}[{r0}, {c0}]", src])
            continue

        # Generic tile ops: `dst = pto.<mnemonic> <operands...>`
        if isinstance(ins, TileInstruction) and is_dataclass(ins):
            fs = list(fields(ins))
            if not fs or fs[0].name != "dst":
                raise NotImplementedError(f"unsupported tile instruction form: {type(ins).__name__}")
            mnemonic = str(ins.opcode).lower()
            dst = tb.tile_for_operand(getattr(ins, "dst").name)  # type: ignore[no-any-return]
            ops: list[str] = []
            for f in fs[1:]:
                v = getattr(ins, f.name)
                # Skip optional flags/attrs for now (only support plain operands).
                if isinstance(v, (bool, int, float, str)) or v is None:
                    raise NotImplementedError(
                        f"{type(ins).__name__}: field {f.name}={v!r} not supported in PTO-AS exporter"
                    )
                ops.append(_operand_str(tb, v))
            out.assign(dst, mnemonic, ops)
            continue

        raise NotImplementedError(f"export_program_to_ptoas: unsupported instruction: {type(ins).__name__}")

    out.epilogue()
    return out.emit()


def export_program_to_ptoas_gemm16(*, program: PTOProgram, block_dim: int = 1, kernel_name: str = "pto_kernel") -> str:
    """
    Export a small GEMM-like PTOProgram (built via the *old* PTOFunctionBuilder) into the
    *new* PTO-AS text format accepted by `ptoas`.

    Scope (fast-path):
    - Supports straight-line programs containing only: TLOAD, TMATMUL, TSTORE.
    - Lowers matmul operands via Mat->Left/Right (TMOV) implicitly, matching `ptoas/examples/gemm16_e2e.pto`.
    """
    # Keep for backward compatibility.
    _ = block_dim, kernel_name
    return export_program_to_ptoas(program=program)
