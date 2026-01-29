"""
PTO Compiler - ARM64 Code Generation

This module contains ARM64/NEON-specific code generation:
- Barrier operation code generation
- Task scheduling code for orchestration functions
- Fused loop code generation
- ARM64-specific header generation

Dependencies:
- pto_isa_definition: ARM64_TYPE_MAP, arm64_generate_header
- pto_compile_common: Common infrastructure
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import os
import sys

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from isa_definition.pto_isa_definition import ARM64_TYPE_MAP, arm64_generate_header, ElementType

from compile.pto_compile_common import (
    PTOProgram, PTOModule, MockTileInfo, MockInstruction,
    FusedLoop, FusionBarrier, FusableOp, LoopFusionOptimizer,
    TileBufferAnalyzer, convert_program_to_mock_instructions,
    apply_binary_expansion, apply_loop_replay_optimization,
)


# =============================================================================
# Orchestration Context
# =============================================================================

@dataclass
class OrchestrationContext:
    """
    Context for generating orchestration function code.
    
    Tracks state needed for task graph building code generation:
    - Task counter for unique IDs
    - Tensor to producer task mapping for dependency tracking
    - Module reference for InCore function metadata
    - Default tile dimensions for code generation
    - Current program being generated (for memref dimension lookup)
    """
    module: Optional['PTOModule'] = None
    task_counter: int = 0
    tensor_producers: Dict[str, int] = field(default_factory=dict)
    default_tile_rows: int = 32
    default_tile_cols: int = 128
    current_program: Optional['PTOProgram'] = None
    
    def alloc_task(self) -> int:
        """Allocate a new task ID."""
        task_id = self.task_counter
        self.task_counter += 1
        return task_id
    
    def set_producer(self, tensor_name: str, task_id: int):
        """Record that a tensor is produced by a task."""
        self.tensor_producers[tensor_name] = task_id
    
    def get_buffer_sizes(self, func_name: str) -> Tuple[float, float]:
        """Get buffer sizes for an InCore function."""
        if self.module:
            return self.module.get_buffer_size(func_name)
        return (0.0, 0.0)
    
    def get_default_tile_shape(self) -> Tuple[int, int]:
        """Get default tile shape (rows, cols)."""
        return (self.default_tile_rows, self.default_tile_cols)


# =============================================================================
# ARM64 Barrier Operation Code Generation
# =============================================================================

def gen_arm64_barrier_op(instr: MockInstruction, rows: int, cols: int, dtype: str, 
                         tile_info: Dict[str, MockTileInfo],
                         orch_ctx: Optional[OrchestrationContext] = None,
                         scalar_declarations: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate ARM64 code for barrier operations (non-fusable).
    
    Args:
        instr: The instruction to generate code for
        rows, cols: Tile dimensions
        dtype: Data type
        tile_info: Tile metadata
        orch_ctx: Optional orchestration context for task scheduling
    """
    lines = []
    c_type = ARM64_TYPE_MAP.get(dtype, "float")
    
    if instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        lines.append(f"// TLOAD: {dst} = load({src_mem}[{row_off}, {col_off}])")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        if row_off == "0" and col_off == "0":
            lines.append(f"        {dst}[_row][_col] = {src_mem}[_row * {cols} + _col];")
        else:
            lines.append(f"        {dst}[_row][_col] = {src_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col];")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        
        # For TSTORE, use destination memref's actual stride, not source tile's cols
        # Get destination memref stride from memref_declarations if available
        dst_stride = cols  # Default to source tile cols
        if orch_ctx and orch_ctx.current_program and dst_mem in orch_ctx.current_program.memref_declarations:
            memref_type = orch_ctx.current_program.memref_declarations[dst_mem]
            if memref_type.shape:
                dst_stride = memref_type.shape.cols  # Use memref's actual column dimension as stride
        
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        if row_off == "0" and col_off == "0":
            lines.append(f"        {dst_mem}[_row * {dst_stride} + _col] = {src}[_row][_col];")
        else:
            lines.append(f"        {dst_mem}[({row_offset_expr} + _row) * {dst_stride} + {col_offset_expr} + _col] = {src}[_row][_col];")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TROWSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWSUM: {dst} = rowsum({src})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _col = 0; _col < {src_cols}; _col++) {{")
        lines.append(f"        _sum += {src}[_row][_col];")
        lines.append(f"    }}")
        lines.append(f"    {dst}[_row][0] = _sum;}}")
    
    elif instr.opcode == "TROWMAX":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWMAX: {dst} = rowmax({src})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    {c_type} _max = {src}[_row][0];")
        lines.append(f"    for (int _col = 1; _col < {src_cols}; _col++) {{")
        lines.append(f"        if ({src}[_row][_col] > _max) _max = {src}[_row][_col];")
        lines.append(f"    }}")
        lines.append(f"    {dst}[_row][0] = _max;}}")
        
    elif instr.opcode == "TCOLSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_rows = src_info.rows if src_info else rows
        lines.append(f"// TCOLSUM: {dst} = colsum({src})")
        lines.append(f"for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _row = 0; _row < {src_rows}; _row++) {{")
        lines.append(f"        _sum += {src}[_row][_col];")
        lines.append(f"    }}")
        lines.append(f"    {dst}[0][_col] = _sum;}}")
        
    elif instr.opcode in ("TROWEXPANDSUB", "TROWEXPANDDIV", "TROWEXPANDMUL"):
        dst, src0, src1 = instr.dst, instr.operands[0], instr.operands[1]
        op_map = {"TROWEXPANDSUB": "-", "TROWEXPANDDIV": "/", "TROWEXPANDMUL": "*"}
        op = op_map.get(instr.opcode, "-")
        lines.append(f"// {instr.opcode}: {dst} = {src0} {op} broadcast({src1})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    {c_type} _broadcast_val = {src1}[_row][0];")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"        {dst}[_row][_col] = {src0}[_row][_col] {op} _broadcast_val;")
        lines.append(f"    }}}}")
        
    elif instr.opcode == "TMATMUL":
        dst, a, b = instr.dst, instr.operands[0], instr.operands[1]
        a_info = tile_info.get(a)
        k = a_info.cols if a_info else 8
        lines.append(f"// TMATMUL: {dst} = {a} @ {b}")
        lines.append(f"for (int _i = 0; _i < {rows}; _i++) {{")
        lines.append(f"    for (int _j = 0; _j < {cols}; _j++) {{")
        lines.append(f"        {c_type} _sum = 0.0f;")
        lines.append(f"        for (int _k = 0; _k < {k}; _k++) {{")
        lines.append(f"            _sum += {a}[_i][_k] * {b}[_k][_j];}}")
        lines.append(f"        {dst}[_i][_j] = _sum;}}}}")
    
    elif instr.opcode == "TTRANS":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_rows = src_info.rows if src_info else rows
        src_cols = src_info.cols if src_info else cols
        # TTRANS: dst = transpose(src)
        # For transpose, dst dimensions are swapped: dst[rows, cols] = src[cols, rows]
        lines.append(f"// TTRANS: {dst} = transpose({src})")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"        {dst}[_row][_col] = {src}[_col][_row];")
        lines.append(f"    }}}}")
    
    # =========== Control Flow Instructions ===========
    elif instr.opcode == "FOR":
        iv = instr.dst
        lb = instr.operands[0]
        ub = instr.operands[1]
        step = instr.operands[2] if len(instr.operands) > 2 else "1"
        max_range = getattr(instr, 'max_range', None)
        min_range = getattr(instr, 'min_range', None) or 1
        tile_levels = getattr(instr, 'tile_levels', None)
        
        if max_range is not None:
            bits = []
            p = max_range
            while p >= min_range:
                bits.append(p)
                p //= 2
            marker = f"// @BINARY_EXPAND: max_range={max_range}, min_range={min_range}, bits=[{','.join(str(b) for b in bits)}]"
            tile_levels_data = getattr(instr, 'tile_levels', None)
            if tile_levels_data:
                levels_str = ",".join(f"{k}:{v}" for k, v in sorted(tile_levels_data.items(), reverse=True))
                marker += f" tile_levels={{{levels_str}}}"
            lines.append(marker)
        lines.append(f"for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
        
    elif instr.opcode == "ENDFOR":
        lines.append("}")
        
    elif instr.opcode == "IF":
        cond = instr.operands[0] if instr.operands else "true"
        lines.append(f"if ({cond}) {{")
    
    elif instr.opcode == "IF_BIT":
        cond = instr.operands[0] if len(instr.operands) > 0 else "0"
        bit_value = instr.operands[1] if len(instr.operands) > 1 else "0"
        lines.append(f"if ({cond} & {bit_value}) {{")
        
    elif instr.opcode == "ELSE":
        lines.append("} else {")
        
    elif instr.opcode == "ENDIF":
        lines.append("}")
    
    # =========== Scalar Instructions ===========
    elif instr.opcode == "SLI":
        dst = instr.dst
        imm = instr.operands[0]
        # Determine scalar type from program declarations or infer from value
        scalar_type = "float"  # Default to float for FP32 scalars
        if scalar_declarations and dst in scalar_declarations:
            elem_type = scalar_declarations[dst]
            scalar_type = ARM64_TYPE_MAP.get(elem_type.value, "float")
        else:
            # Infer type from immediate value
            imm_str = str(imm)
            if '.' in imm_str or 'e' in imm_str.lower() or 'E' in imm_str:
                scalar_type = "float"
                # Add 'f' suffix for float literals if not already present
                if imm_str.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').replace('f', '').isdigit():
                    imm = f"{imm_str}f" if not imm_str.endswith('f') else imm_str
            else:
                # Try to parse as number
                try:
                    float_val = float(imm_str)
                    if float_val != int(float_val):
                        scalar_type = "float"
                        imm = f"{imm_str}f"
                    else:
                        scalar_type = "int"
                except:
                    scalar_type = "float"
        lines.append(f"{scalar_type} {dst} = {imm};")
        
    elif instr.opcode == "SCMP":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        cmp_mode = instr.operands[2] if len(instr.operands) > 2 else "eq"
        cmp_ops = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        cmp_op = cmp_ops.get(cmp_mode, ">")
        lines.append(f"int {dst} = ({src0} {cmp_op} {src1}) ? 1 : 0;")
        
    elif instr.opcode == "SADD":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        lines.append(f"int {dst} = {src0} + {src1};")
        
    elif instr.opcode == "SSUB":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        lines.append(f"int {dst} = {src0} - {src1};")
        
    elif instr.opcode == "SMUL":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        lines.append(f"int {dst} = {src0} * {src1};")
        
    elif instr.opcode == "SDIV":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        lines.append(f"int {dst} = {src0} / {src1};")
        
    elif instr.opcode == "SMOV":
        dst = instr.dst
        src = instr.operands[0]
        lines.append(f"int {dst} = {src};")
    
    # =========== Function Call Instructions ===========
    elif instr.opcode == "CALL":
        callee = instr.dst
        args = instr.operands
        
        if orch_ctx is not None:
            lines.extend(gen_task_scheduling_code(callee, args, orch_ctx, tile_info, rows, cols))
        else:
            if args:
                if isinstance(args, dict):
                    arg_names = []
                    for param, arg_val in args.items():
                        if isinstance(arg_val, tuple):
                            arg_names.append(arg_val[0])
                        else:
                            arg_names.append(str(arg_val))
                    args_str = ", ".join(arg_names)
                else:
                    args_str = ", ".join(str(arg) for arg in args)
                lines.append(f"{callee}({args_str});")
            else:
                lines.append(f"{callee}();")
    
    elif instr.opcode == "RETURN":
        if orch_ctx is not None:
            lines.append("goto __pto_orch_epilogue;")
        else:
            lines.append("return;")
        
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


# =============================================================================
# Task Scheduling Code Generation (for Orchestration Functions)
# =============================================================================

def gen_task_scheduling_code(callee: str, args: Union[List, Dict], 
                              orch_ctx: OrchestrationContext,
                              tile_info: Dict[str, MockTileInfo],
                              rows: int, cols: int) -> List[str]:
    """
    Generate task scheduling code for an InCore function call.
    
    This is called when generating code for an orchestration function that
    calls an InCore function.
    """
    lines = []
    task_id = orch_ctx.alloc_task()
    
    # Get buffer sizes for this function
    buf_without_reuse, buf_with_reuse = orch_ctx.get_buffer_sizes(callee)
    buf_bytes = int(buf_without_reuse * 1024)
    reuse_bytes = int(buf_with_reuse * 1024)
    
    # Get is_cube from callee function if available
    callee_is_cube = 0
    if orch_ctx.module:
        callee_prog = orch_ctx.module.functions.get(callee)
        if callee_prog and getattr(callee_prog, 'is_cube', False):
            callee_is_cube = 1
    
    lines.append(f"// Task {task_id}: {callee}")
    lines.append(f"int32_t t{task_id} = pto_task_alloc(rt, \"{callee}\", NULL, {buf_bytes}, {reuse_bytes}, {callee_is_cube});")
    
    # Parse arguments
    input_args = []
    output_args = []
    
    if isinstance(args, dict):
        for param, arg_value in args.items():
            is_output = "output" in param.lower() or "result" in param.lower() or "dst" in param.lower()
            
            # Extract tensor name from arg_value
            if isinstance(arg_value, tuple):
                tensor_name = arg_value[0]
            elif isinstance(arg_value, str) and "->" in arg_value:
                tensor_name = arg_value.split("->")[0].strip()
            else:
                tensor_name = str(arg_value)
            
            # Extract row/col offsets
            if isinstance(arg_value, tuple):
                row_off = str(arg_value[1]) if len(arg_value) > 1 else "0"
                col_off = str(arg_value[2]) if len(arg_value) > 2 else "0"
            else:
                row_off = "0"
                col_off = "0"
            
            # Try to get dimensions from current program's memref_declarations first
            t_rows = rows
            t_cols = cols
            if orch_ctx.current_program and tensor_name in orch_ctx.current_program.memref_declarations:
                memref_type = orch_ctx.current_program.memref_declarations[tensor_name]
                if memref_type.shape:
                    t_rows = memref_type.shape.rows
                    t_cols = memref_type.shape.cols
            else:
                # Fallback to tile_info (for tile operands)
                info = tile_info.get(tensor_name)
                if info:
                    t_rows = info.rows
                    t_cols = info.cols
            
            # Re-extract tensor_name for dimension lookup (in case it was already extracted above)
            if isinstance(arg_value, tuple):
                lookup_name = arg_value[0]
            elif isinstance(arg_value, str) and "->" in arg_value:
                lookup_name = arg_value.split("->")[0].strip()
            else:
                lookup_name = str(arg_value)
            
            # Get dimensions from current program's memref_declarations if available
            if orch_ctx.current_program and lookup_name in orch_ctx.current_program.memref_declarations:
                memref_type = orch_ctx.current_program.memref_declarations[lookup_name]
                if memref_type.shape:
                    t_rows = memref_type.shape.rows
                    t_cols = memref_type.shape.cols
            
            if is_output:
                output_args.append((tensor_name, row_off, col_off, t_rows, t_cols))
                orch_ctx.set_producer(tensor_name, task_id)
            else:
                input_args.append((tensor_name, row_off, col_off, t_rows, t_cols))
    
    # Generate input tracking
    for tensor, row_off, col_off, t_rows, t_cols in input_args:
        lines.append(f"pto_task_add_input(rt, t{task_id}, {tensor}, {row_off}, {col_off}, {t_rows}, {t_cols});")
    
    # Generate output tracking
    for tensor, row_off, col_off, t_rows, t_cols in output_args:
        lines.append(f"pto_task_add_output(rt, t{task_id}, {tensor}, {row_off}, {col_off}, {t_rows}, {t_cols});")
    
    lines.append(f"pto_task_submit(rt, t{task_id});")
    lines.append("")
    
    return lines


# =============================================================================
# ARM64 Fused Loop Code Generation
# =============================================================================

class ARM64FusedCodeGenerator:
    """Generate ARM64 code for fused loops."""
    
    def __init__(self):
        self.dtype_map = ARM64_TYPE_MAP
    
    def generate_fused_loop(self, fused_loop: FusedLoop) -> List[str]:
        """Generate code for a fused loop."""
        lines = []
        rows = fused_loop.tile_shape.rows
        cols = fused_loop.tile_shape.cols
        c_type = self.dtype_map.get(fused_loop.dtype, "float")
        
        lines.append(f"// Fused loop: {len(fused_loop.operations)} operations")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        
        for op in fused_loop.operations:
            op_code = self._generate_single_op(op, c_type)
            lines.append(f"        {op_code}")
        
        lines.append("    }}")  # Close inner and outer loops
        
        return lines
    
    def _generate_single_op(self, op: FusableOp, c_type: str) -> str:
        """Generate code for a single fusable operation."""
        dst = f"{op.dst}[_row][_col]"
        src0 = f"{op.operands[0]}[_row][_col]" if op.operands else ""
        src1 = ""
        
        # Check if this is a scalar operation (second operand is a scalar, not a tile)
        is_scalar_op = op.opcode in ("TADDS", "TSUBS", "TMULS", "TDIVS")
        
        if len(op.operands) >= 2:
            src1_val = op.operands[1]
            if is_scalar_op:
                # For scalar operations, second operand is a scalar variable or literal
                # Use it directly without indexing
                src1 = str(src1_val)
            else:
                # For tile operations, check if it's a tile or literal
                if isinstance(src1_val, str) and not src1_val.replace(".", "").replace("-", "").isdigit():
                    src1 = f"{src1_val}[_row][_col]"
                else:
                    src1 = str(src1_val)
        
        # Binary operations
        if op.opcode == "TADD": return f"{dst} = {src0} + {src1};"
        elif op.opcode == "TSUB": return f"{dst} = {src0} - {src1};"
        elif op.opcode == "TMUL": return f"{dst} = {src0} * {src1};"
        elif op.opcode == "TDIV": return f"{dst} = {src0} / {src1};"
        elif op.opcode == "TMAX": return f"{dst} = ({src0} > {src1}) ? {src0} : {src1};"
        elif op.opcode == "TMIN": return f"{dst} = ({src0} < {src1}) ? {src0} : {src1};"
        
        # Unary operations
        elif op.opcode == "TABS": return f"{dst} = fabsf({src0});"
        elif op.opcode == "TNEG": return f"{dst} = -{src0};"
        elif op.opcode == "TRECIP": return f"{dst} = 1.0f / {src0};"
        elif op.opcode == "TEXP": return f"{dst} = expf({src0});"
        elif op.opcode == "TLOG": return f"{dst} = logf({src0});"
        elif op.opcode == "TSQRT": return f"{dst} = sqrtf({src0});"
        elif op.opcode == "TRSQRT": return f"{dst} = 1.0f / sqrtf({src0});"
        elif op.opcode == "TRELU": return f"{dst} = ({src0} > 0) ? {src0} : 0;"
        elif op.opcode == "TSIGMOID": return f"{dst} = 1.0f / (1.0f + expf(-{src0}));"
        elif op.opcode == "TTANH": return f"{dst} = tanhf({src0});"
        elif op.opcode == "TGELU": return f"{dst} = 0.5f * {src0} * (1.0f + erff({src0} / 1.41421356f));"
        elif op.opcode == "TSILU": return f"{dst} = {src0} / (1.0f + expf(-{src0}));"
        elif op.opcode == "TFLOOR": return f"{dst} = floorf({src0});"
        elif op.opcode == "TCEIL": return f"{dst} = ceilf({src0});"
        elif op.opcode == "TSIN": return f"{dst} = sinf({src0});"
        elif op.opcode == "TCOS": return f"{dst} = cosf({src0});"
        elif op.opcode == "TERF": return f"{dst} = erff({src0});"
        
        # Scalar operations
        elif op.opcode == "TADDS": return f"{dst} = {src0} + {src1};"
        elif op.opcode == "TSUBS": return f"{dst} = {src0} - {src1};"
        elif op.opcode == "TMULS": return f"{dst} = {src0} * {src1};"
        elif op.opcode == "TDIVS": return f"{dst} = {src0} / {src1};"
        elif op.opcode == "TEXPANDS": return f"{dst} = {op.operands[0]};"
        
        return f"// Unknown op: {op.opcode}"


# =============================================================================
# ARM64 Code Generator
# =============================================================================

class ARM64CodeGenerator:
    """
    Generates ARM64 NEON code from PTO programs.
    """
    
    def __init__(self, enable_fusion: bool = True, analyze_buffers: bool = True,
                 module: Optional['PTOModule'] = None):
        self.enable_fusion = enable_fusion
        self.analyze_buffers = analyze_buffers
        self.module = module
    
    def generate(self, program: PTOProgram) -> str:
        """Generate ARM64 code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        is_in_core = getattr(program, 'is_in_core', True)
        in_core_str = "InCore (tile-level computation)" if is_in_core else "Orchestration (control flow only)"
        
        # Create orchestration context for non-InCore functions
        orch_ctx = None
        if not is_in_core:
            orch_ctx = OrchestrationContext(module=self.module, current_program=program)
        
        lines = [
            f"// PTO Program: {program.name}",
            f"// Function Type: {in_core_str}",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()
            report = analyzer.generate_report()
            lines.append(report)
            
            if self.module is not None:
                self.module.set_buffer_analysis(program.name, analyzer.analysis_result)
        
        # For orchestration functions, add runtime header
        if not is_in_core:
            lines.append('// Orchestration function - builds task graph using PTO runtime')
            lines.append('#include "pto_runtime.h"')
            lines.append('// Note: pto_runtime.c should be compiled separately to avoid duplicate symbols')
            lines.append('#include <string.h>  // For strcmp in main')
            lines.append('#include <time.h>    // For benchmark timing')
            lines.append('')
        
        lines.append(arm64_generate_header())
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in program.memref_declarations.items():
            c_type = ARM64_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
        
        # Find scalars initialized by SLI
        sli_initialized_scalars = set()
        for instr in mock_instructions:
            if instr.opcode == "SLI":
                sli_initialized_scalars.add(instr.dst)
        
        # Declare scalar variables as function parameters
        scalar_params = []
        for name, scalar_type in program.scalar_declarations.items():
            if scalar_type in (ElementType.U1, ElementType.INDEX):
                continue
            if name in sli_initialized_scalars:
                continue
            c_type = ARM64_TYPE_MAP.get(scalar_type.value, "int")
            scalar_params.append(f"{c_type} {name}")
        
        # For orchestration functions, add PTORuntime* as first parameter
        if not is_in_core:
            all_params = ["PTORuntime* rt"] + memref_params + scalar_params
        else:
            all_params = memref_params + scalar_params
        
        # Generate function signature
        if all_params:
            func_params = ", ".join(all_params)
            lines.append(f"void {program.name}({func_params}) {{")
        else:
            lines.append(f"void {program.name}(void) {{")
        
        # Declare tiles as local variables
        for name, info in tile_info.items():
            c_type = ARM64_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"    {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")

        if not is_in_core:
            lines.append("    // Root scope for buffer lifetime management")
            lines.append("    pto_scope_begin(rt);")
            lines.append("")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            fused_codegen = ARM64FusedCodeGenerator()
            indent_level = 1
            
            for item in fused_result:
                indent = "    " * indent_level
                
                if isinstance(item, FusedLoop):
                    fused_lines = fused_codegen.generate_fused_loop(item)
                    for fused_line in fused_lines:
                        lines.append(f"{indent}{fused_line}" if fused_line else "")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    # For TSTORE, get dimensions from source tile (operands[0]), not destination memref (dst)
                    # For other operations, get dimensions from destination tile (dst)
                    if instr.opcode == "TSTORE" and instr.operands:
                        src_tile = instr.operands[0]
                        info = tile_info.get(src_tile) if src_tile else None
                    else:
                        info = tile_info.get(instr.dst) if instr.dst else None
                    default_rows = orch_ctx.default_tile_rows if orch_ctx else 32
                    default_cols = orch_ctx.default_tile_cols if orch_ctx else 128
                    rows = info.rows if info else default_rows
                    cols = info.cols if info else default_cols
                    dtype = info.dtype if info else "f32"
                    
                    # Handle indentation changes for control flow
                    if instr.opcode in ("ENDFOR", "ENDIF"):
                        indent_level = max(1, indent_level - 1)
                        indent = "    " * indent_level
                    elif instr.opcode == "ELSE":
                        indent = "    " * max(1, indent_level - 1)
                    
                    barrier_lines = gen_arm64_barrier_op(instr, rows, cols, dtype, tile_info, orch_ctx, program.scalar_declarations)
                    for barrier_line in barrier_lines:
                        lines.append(f"{indent}{barrier_line}" if barrier_line else "")
                    
                    if instr.opcode in ("FOR", "IF", "ELSE"):
                        indent_level += 1
                    
                    lines.append("")
        
        if not is_in_core:
            lines.append("    __pto_orch_epilogue:;")
            lines.append("    pto_scope_end(rt);")
            lines.append("    return;")
        lines.append("}")
        
        code = "\n".join(lines)
        
        # Apply binary expansion if needed
        code = apply_binary_expansion(code)
        
        # Note: Loop replay optimization has been archived due to conflicts
        # with the sliding window task management scheme.
        
        # For orchestration functions, append main() for standalone execution
        if not is_in_core:
            # Extract parameters from function (exclude PTORuntime* rt)
            params = [(ptype, pname.split('[')[0]) for ptype, pname in 
                     [(p.split()[0] + ('*' if '*' in p else ''), p.split()[-1].rstrip(',')) 
                      for p in (memref_params + scalar_params)]]
            code += generate_arm64_main(program.name, params)
        
        return code


# =============================================================================
# Main Function Generator for Standalone Execution
# =============================================================================

def generate_arm64_main(orch_func_name: str, params: List[Tuple[str, str]]) -> str:
    """
    Generate main function for ARM64 standalone execution.
    
    Args:
        orch_func_name: Name of the orchestration function
        params: List of (type, name) tuples for function parameters
        
    Returns:
        C code string for main() function
    """
    # Build parameter declarations for test data
    param_decls = []
    param_args = []
    free_stmts = []
    
    # Track integer parameters that can be overridden via command line
    int_params = []
    int_param_idx = 0
    
    for ptype, pname in params:
        if '*' in ptype:
            # Pointer - allocate test data
            base_type = ptype.replace('*', '').strip()
            param_decls.append(f"    {base_type}* {pname} = ({base_type}*)calloc(1024 * 1024, sizeof({base_type}));")
            param_args.append(pname)
            free_stmts.append(f"    free({pname});")
        else:
            # Scalar - use default value or command line argument
            if 'int' in ptype:
                int_params.append((pname, int_param_idx))
                int_param_idx += 1
                param_decls.append(f"    {ptype} {pname} = 16;  // Default, override with argv[{int_param_idx}+arg_offset]")
            else:
                param_decls.append(f"    {ptype} {pname} = 1.0f;  // Default test value")
            param_args.append(pname)
    
    # Generate command line argument parsing for integer parameters
    argv_parsing = ""
    if int_params:
        argv_parsing = "\n    // Parse command line arguments for integer parameters\n"
        for pname, idx in int_params:
            argv_parsing += f"    if (argc > {idx + 1} + arg_offset) {pname} = atoi(argv[{idx + 1} + arg_offset]);\n"
    
    return f'''
// =============================================================================
// Main Function for ARM64 Standalone Execution
// =============================================================================
// Usage: {orch_func_name} [--benchmark-only] [seq_len] [tile_rows] [num_tiles] [zero]
// Flags:
//   --benchmark-only  - Only run orchestration (skip execution), output stats

int main(int argc, char** argv) {{
    // Check for --benchmark-only flag
    int benchmark_only = 0;
    int arg_offset = 0;
    
    for (int i = 1; i < argc; i++) {{
        if (strcmp(argv[i], "--benchmark-only") == 0) {{
            benchmark_only = 1;
            arg_offset = 1;
            break;
        }}
    }}
    
    printf("============================================================\\n");
    printf("  PTO ARM64 Runtime\\n");
    printf("============================================================\\n");
    
    // Initialize runtime (heap allocated - PTORuntime is too large for stack)
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {{
        fprintf(stderr, "Failed to allocate PTORuntime\\n");
        return 1;
    }}
    pto_runtime_init(rt);
    
    // Allocate test data
{chr(10).join(param_decls)}
{argv_parsing}
    
    if (benchmark_only) {{
        // Benchmark mode: only measure orchestration time
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        {orch_func_name}(rt, {', '.join(param_args)});
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        long long tasks_submitted = rt->total_tasks_scheduled;
        double tasks_per_ms = tasks_submitted / time_ms;
        
        // Output in machine-parseable format
        printf("BENCHMARK: tasks=%lld time_ms=%.3f tasks_per_ms=%.2f\\n",
               tasks_submitted, time_ms, tasks_per_ms);
    }} else {{
        // Normal execution mode
        printf("Running orchestration function: {orch_func_name}\\n");
        printf("------------------------------------------------------------\\n");
        
        {orch_func_name}(rt, {', '.join(param_args)});
        
        printf("------------------------------------------------------------\\n");
        printf("Submitted %lld tasks\\n", (long long)rt->total_tasks_scheduled);
        
        // Execute all tasks
        pto_execute_all(rt);
        
        printf("Execution complete!\\n");
    }}
    
    // Cleanup - must call shutdown before free to destroy mutexes/condvars
    fflush(stdout);
    pto_runtime_shutdown(rt);
{chr(10).join(free_stmts)}
    free(rt);
    
    return 0;
}}
'''


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'OrchestrationContext',
    'gen_arm64_barrier_op',
    'gen_task_scheduling_code',
    'ARM64FusedCodeGenerator',
    'ARM64CodeGenerator',
    'generate_arm64_main',
]
