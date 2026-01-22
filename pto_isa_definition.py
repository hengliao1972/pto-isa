"""
PTO ISA Definition - DSL for Parallel Tile Operations

This module defines the complete PTO (Parallel Tile Operations) Instruction Set Architecture
as a Python-based Domain Specific Language (DSL). The PTO ISA operates on Tiles - 2D blocks of data.

Key Concepts:
- Tile: A 2-dimensional block of data with shape (rows, cols) and element type
- TileShape: Specification of tile dimensions
- PTO Instructions: Operations on tile operands
- LOOP: Iteration constructs with iteration count derived from tile shapes
- codegen_arm64(): Each instruction can generate ARM64 NEON intrinsic code
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Any, Dict
from abc import ABC, abstractmethod


# =============================================================================
# ARM64 Code Generation Infrastructure
# =============================================================================

# ARM64 scalar type mappings
ARM64_TYPE_MAP = {
    "f32": "float",
    "f16": "__fp16",
    "f64": "double",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

# ARM64 NEON vector type mappings
ARM64_VECTOR_TYPE_MAP = {
    "f32": "float32x4_t",
    "f16": "float16x8_t",
    "f64": "float64x2_t",
    "i8": "int8x16_t",
    "i16": "int16x8_t",
    "i32": "int32x4_t",
    "i64": "int64x2_t",
    "u8": "uint8x16_t",
    "u16": "uint16x8_t",
    "u32": "uint32x4_t",
    "u64": "uint64x2_t",
}

# Elements per NEON 128-bit vector
ARM64_VECTOR_LANES = {
    "f32": 4,
    "f16": 8,
    "f64": 2,
    "i8": 16,
    "i16": 8,
    "i32": 4,
    "i64": 2,
    "u8": 16,
    "u16": 8,
    "u32": 4,
    "u64": 2,
}

# ARM64 Physical Tile Size
# Physical_Row_Size: Optimal repeat count for vector pipeline performance
ARM64_PHYSICAL_ROW_SIZE = 1          # Optimal repeat count for ARM64

# NEON intrinsic suffix mappings
ARM64_NEON_SUFFIX = {
    "f32": "f32",
    "f16": "f16",
    "f64": "f64",
    "i8": "s8",
    "i16": "s16",
    "i32": "s32",
    "i64": "s64",
    "u8": "u8",
    "u16": "u16",
    "u32": "u32",
    "u64": "u64",
}


@dataclass
class ARM64CodeGenContext:
    """
    Context for ARM64 code generation.
    
    Tracks state during code generation including indentation level,
    temporary variable counters, and declared variables.
    """
    indent_level: int = 0
    temp_counter: int = 0
    var_counter: int = 0
    declared_vars: set = None
    
    def __post_init__(self):
        if self.declared_vars is None:
            self.declared_vars = set()
    
    def get_temp(self, prefix: str = "tmp") -> str:
        """Get a unique temporary variable name."""
        name = f"{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name
    
    def get_unique_var(self, prefix: str = "_v") -> str:
        """Get a unique variable name."""
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def indent(self) -> str:
        """Get current indentation string (4 spaces per level)."""
        return "    " * self.indent_level
    
    def reset(self):
        """Reset the context to initial state."""
        self.indent_level = 0
        self.temp_counter = 0
        self.var_counter = 0
        self.declared_vars = set()


def arm64_get_neon_suffix(dtype: str) -> str:
    """Get NEON intrinsic suffix for a data type."""
    return ARM64_NEON_SUFFIX.get(dtype, "f32")


def arm64_generate_header() -> str:
    """Generate standard ARM64 NEON header includes."""
    return """// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
"""


def arm64_generate_tile_declaration(name: str, rows: int, cols: int, dtype: str = "f32") -> str:
    """Generate C declaration for a tile."""
    c_type = ARM64_TYPE_MAP.get(dtype, "float")
    return f"{c_type} {name}[{rows}][{cols}];"


# =============================================================================
# NVIDIA CUDA Code Generation Infrastructure
# =============================================================================

# CUDA scalar type mappings
CUDA_TYPE_MAP = {
    "f32": "float",
    "f16": "__half",
    "f64": "double",
    "bf16": "__nv_bfloat16",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

# CUDA vector type mappings (float4, half2, etc.)
CUDA_VECTOR_TYPE_MAP = {
    "f32": "float4",
    "f16": "half2",
    "f64": "double2",
    "i32": "int4",
    "u32": "uint4",
}

# Elements per CUDA vector type
CUDA_VECTOR_LANES = {
    "f32": 4,
    "f16": 2,
    "f64": 2,
    "i32": 4,
    "u32": 4,
}

# CUDA math intrinsics
CUDA_INTRINSICS = {
    "exp": {"f32": "__expf", "f64": "exp", "f16": "hexp"},
    "log": {"f32": "__logf", "f64": "log", "f16": "hlog"},
    "sqrt": {"f32": "__fsqrt_rn", "f64": "sqrt", "f16": "hsqrt"},
    "rsqrt": {"f32": "__frsqrt_rn", "f64": "rsqrt", "f16": "hrsqrt"},
    "sin": {"f32": "__sinf", "f64": "sin", "f16": "hsin"},
    "cos": {"f32": "__cosf", "f64": "cos", "f16": "hcos"},
    "abs": {"f32": "fabsf", "f64": "fabs", "f16": "__habs"},
    "max": {"f32": "fmaxf", "f64": "fmax", "f16": "__hmax"},
    "min": {"f32": "fminf", "f64": "fmin", "f16": "__hmin"},
}

# CUDA Physical Tile Size
# Physical_Row_Size: Optimal repeat count for vector pipeline performance
CUDA_PHYSICAL_ROW_SIZE = 1           # Optimal repeat count for CUDA


@dataclass
class CUDACodeGenContext:
    """
    Context for NVIDIA CUDA code generation.
    
    Tracks state during code generation including indentation level,
    temporary variable counters, thread block configuration, and shared memory.
    """
    indent_level: int = 0
    temp_counter: int = 0
    var_counter: int = 0
    declared_vars: set = None
    block_dim_x: int = 32   # Warp size
    block_dim_y: int = 8    # Typical tile row parallelism
    use_shared_memory: bool = True
    
    def __post_init__(self):
        if self.declared_vars is None:
            self.declared_vars = set()
    
    def get_temp(self, prefix: str = "tmp") -> str:
        """Get a unique temporary variable name."""
        name = f"{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name
    
    def get_unique_var(self, prefix: str = "_cv") -> str:
        """Get a unique variable name."""
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def indent(self) -> str:
        """Get current indentation string (4 spaces per level)."""
        return "    " * self.indent_level
    
    def reset(self):
        """Reset the context to initial state."""
        self.indent_level = 0
        self.temp_counter = 0
        self.var_counter = 0
        self.declared_vars = set()


def cuda_generate_header() -> str:
    """Generate standard CUDA header includes."""
    return """// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;
"""


def cuda_generate_tile_declaration(name: str, rows: int, cols: int, dtype: str = "f32", 
                                    shared: bool = False) -> str:
    """Generate CUDA declaration for a tile."""
    c_type = CUDA_TYPE_MAP.get(dtype, "float")
    if shared:
        return f"__shared__ {c_type} {name}[{rows}][{cols}];"
    return f"{c_type} {name}[{rows}][{cols}];"


def cuda_get_intrinsic(op: str, dtype: str) -> str:
    """Get CUDA intrinsic function for an operation and data type."""
    if op in CUDA_INTRINSICS and dtype in CUDA_INTRINSICS[op]:
        return CUDA_INTRINSICS[op][dtype]
    return f"{op}f" if dtype == "f32" else op


# =============================================================================
# Huawei Ascend A2/A3 Code Generation Infrastructure (Ascend C / AscendC)
# =============================================================================

# Ascend C scalar type mappings
ASCEND_TYPE_MAP = {
    "f32": "float",
    "f16": "half",
    "f64": "double",
    "bf16": "bfloat16_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

# Ascend vector type mappings (LocalTensor)
ASCEND_VECTOR_LANES = {
    "f32": 8,    # 256-bit vector / 32-bit
    "f16": 16,   # 256-bit vector / 16-bit
    "bf16": 16,
    "i32": 8,
    "i8": 32,
    "u8": 32,
}

# Ascend C memory spaces
ASCEND_MEMORY_SPACE = {
    "gm": "GM",        # Global Memory
    "l2": "L2",        # L2 Cache
    "l1": "L1",        # L1 Cache
    "ub": "UB",        # Unified Buffer (on-chip SRAM)
    "l0a": "L0A",      # L0 Matrix A buffer
    "l0b": "L0B",      # L0 Matrix B buffer
    "l0c": "L0C",      # L0 Matrix C buffer (accumulator)
}

# Ascend C vector operations mapping
ASCEND_VECTOR_OPS = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "max": "Max",
    "min": "Min",
    "abs": "Abs",
    "neg": "Neg",
    "exp": "Exp",
    "log": "Ln",
    "sqrt": "Sqrt",
    "rsqrt": "Rsqrt",
    "recip": "Reciprocal",
    "relu": "Relu",
}

# Ascend A2/A3 Physical Tile Size
# Physical_Row_Size: Optimal repeat count for vector pipeline performance
ASCEND_PHYSICAL_ROW_SIZE = 32         # Optimal repeat count for Ascend A2/A3 pipeline


@dataclass
class AscendCodeGenContext:
    """
    Context for Huawei Ascend A2/A3 code generation (Ascend C).
    
    Ascend A2/A3 uses Ascend C programming model with:
    - DataCopy for memory transfers
    - Vector operations for elementwise ops
    - Cube operations for matrix multiplication
    - Pipe-based synchronization
    """
    indent_level: int = 0
    temp_counter: int = 0
    var_counter: int = 0
    declared_vars: set = None
    block_dim: int = 8     # Typical AI Core count per task
    tile_size: int = 256   # Elements per vector operation
    use_double_buffer: bool = True  # Pipeline optimization
    
    def __post_init__(self):
        if self.declared_vars is None:
            self.declared_vars = set()
    
    def get_temp(self, prefix: str = "tmp") -> str:
        """Get a unique temporary variable name."""
        name = f"{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name
    
    def get_unique_var(self, prefix: str = "_av") -> str:
        """Get a unique variable name."""
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def indent(self) -> str:
        """Get current indentation string (4 spaces per level)."""
        return "    " * self.indent_level
    
    def reset(self):
        """Reset the context to initial state."""
        self.indent_level = 0
        self.temp_counter = 0
        self.var_counter = 0
        self.declared_vars = set()


def ascend_generate_header() -> str:
    """Generate standard Ascend C header includes."""
    return """// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;
"""


def ascend_generate_tile_declaration(name: str, rows: int, cols: int, dtype: str = "f32",
                                      memory_space: str = "ub") -> str:
    """Generate Ascend C declaration for a tile (LocalTensor)."""
    ascend_type = ASCEND_TYPE_MAP.get(dtype, "float")
    total_size = rows * cols
    return f"LocalTensor<{ascend_type}> {name};  // {rows}x{cols} = {total_size} elements"


def ascend_get_vector_op(op: str) -> str:
    """Get Ascend C vector operation name."""
    return ASCEND_VECTOR_OPS.get(op, op.capitalize())


# =============================================================================
# Loop IR - Intermediate Representation for Loop Structures
# =============================================================================

class LoopOpType(Enum):
    """Types of operations that can be performed inside a loop."""
    # Binary tile-tile operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MAX = "max"
    MIN = "min"
    
    # Unary operations
    ABS = "abs"
    NEG = "neg"
    RECIP = "recip"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    RELU = "relu"
    
    # Scalar operations
    ADDS = "adds"      # tile + scalar
    SUBS = "subs"      # tile - scalar
    MULS = "muls"      # tile * scalar
    DIVS = "divs"      # tile / scalar
    MAXS = "maxs"      # max(tile, scalar)
    MINS = "mins"      # min(tile, scalar)
    
    # Broadcast
    EXPANDS = "expands"  # scalar -> tile
    
    # Copy/move
    COPY = "copy"


@dataclass
class LoopBodyOp:
    """
    A single operation to be performed inside a tile loop.
    
    This represents an elementwise operation without the loop structure,
    allowing multiple operations to be fused into a single loop.
    
    Attributes:
        op_type: The type of operation (add, mul, etc.)
        dst: Destination tile name
        srcs: Source tile names (1 for unary, 2 for binary)
        scalar: Scalar value for scalar operations (ADDS, MULS, etc.)
        comment: Optional comment for this operation
    """
    op_type: LoopOpType
    dst: str
    srcs: List[str] = field(default_factory=list)
    scalar: Optional[str] = None
    comment: str = ""
    
    def __str__(self) -> str:
        if self.scalar:
            return f"{self.dst} = {self.op_type.value}({', '.join(self.srcs)}, {self.scalar})"
        else:
            return f"{self.dst} = {self.op_type.value}({', '.join(self.srcs)})"


@dataclass
class TileLoopIR:
    """
    Intermediate Representation for a tile loop structure.
    
    This represents one or more operations over a tile's elements,
    without explicitly generating C loop syntax. This allows:
    - Easy loop fusion by combining bodies of loops with same shape
    - Architecture-agnostic representation
    - Deferred code generation
    
    Attributes:
        rows: Number of rows in the tile
        cols: Number of columns in the tile
        dtype: Element data type (f32, f16, etc.)
        bodies: List of operations to perform in this loop
        vectorizable: Whether this loop can use SIMD instructions
        is_reduction: Whether this is a reduction operation (prevents fusion)
    """
    rows: int
    cols: int
    dtype: str
    bodies: List[LoopBodyOp] = field(default_factory=list)
    vectorizable: bool = True
    is_reduction: bool = False
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape as a tuple for easy comparison."""
        return (self.rows, self.cols)
    
    def can_fuse_with(self, other: "TileLoopIR") -> bool:
        """Check if this loop can be fused with another."""
        if self.is_reduction or other.is_reduction:
            return False
        return (self.rows == other.rows and 
                self.cols == other.cols and 
                self.dtype == other.dtype)
    
    def fuse(self, other: "TileLoopIR") -> "TileLoopIR":
        """
        Fuse this loop with another, combining their bodies.
        Returns a new TileLoopIR with combined operations.
        """
        if not self.can_fuse_with(other):
            raise ValueError("Cannot fuse loops with different shapes or types")
        return TileLoopIR(
            rows=self.rows,
            cols=self.cols,
            dtype=self.dtype,
            bodies=self.bodies + other.bodies,
            vectorizable=self.vectorizable and other.vectorizable,
            is_reduction=False
        )
    
    def __str__(self) -> str:
        ops = "; ".join(str(b) for b in self.bodies)
        return f"TileLoop<{self.rows}x{self.cols}x{self.dtype}>[{ops}]"


@dataclass
class NonLoopIR:
    """
    Represents an operation that doesn't use a tile loop structure.
    Used for declarations, control flow, and non-fusable operations.
    """
    op_type: str  # "decl", "matmul", "reduction", etc.
    code_lines: List[str] = field(default_factory=list)
    comment: str = ""


# Union type for codegen output
CodeGenIR = Union[TileLoopIR, NonLoopIR, List[str]]


class TileLoopCodeGen:
    """
    Code generator that converts TileLoopIR to actual C code.
    
    This class handles:
    - Loop fusion: combining multiple TileLoopIR with same shape
    - ARM64 NEON vectorization
    - Scalar fallback for non-vectorizable types
    """
    
    def __init__(self, ctx: ARM64CodeGenContext = None):
        self.ctx = ctx or ARM64CodeGenContext()
    
    def fuse_loops(self, loops: List[CodeGenIR]) -> List[CodeGenIR]:
        """
        Fuse consecutive fusable TileLoopIR objects.
        
        Args:
            loops: List of IR objects (TileLoopIR, NonLoopIR, or code lines)
            
        Returns:
            List with fusable loops combined
        """
        if not loops:
            return []
        
        result = []
        current_loop: Optional[TileLoopIR] = None
        
        for item in loops:
            if isinstance(item, TileLoopIR):
                if current_loop is None:
                    current_loop = item
                elif current_loop.can_fuse_with(item):
                    # Fuse the loops
                    current_loop = current_loop.fuse(item)
                else:
                    # Can't fuse, emit current and start new
                    result.append(current_loop)
                    current_loop = item
            else:
                # Non-loop IR or code lines - emit current loop first
                if current_loop is not None:
                    result.append(current_loop)
                    current_loop = None
                result.append(item)
        
        # Don't forget the last loop
        if current_loop is not None:
            result.append(current_loop)
        
        return result
    
    def generate(self, ir: CodeGenIR) -> List[str]:
        """Generate C code from an IR object."""
        if isinstance(ir, TileLoopIR):
            return self._generate_tile_loop(ir)
        elif isinstance(ir, NonLoopIR):
            return ir.code_lines
        elif isinstance(ir, list):
            return ir
        else:
            return [f"// Unknown IR type: {type(ir)}"]
    
    def generate_all(self, ir_list: List[CodeGenIR], fuse: bool = True) -> List[str]:
        """
        Generate C code from a list of IR objects.
        
        Args:
            ir_list: List of IR objects
            fuse: Whether to fuse compatible loops first
            
        Returns:
            List of C code lines
        """
        if fuse:
            ir_list = self.fuse_loops(ir_list)
        
        lines = []
        for ir in ir_list:
            lines.extend(self.generate(ir))
        return lines
    
    def _generate_tile_loop(self, loop: TileLoopIR) -> List[str]:
        """Generate C code for a TileLoopIR."""
        lines = []
        indent = self.ctx.indent()
        rows, cols, dtype = loop.rows, loop.cols, loop.dtype
        vec_lanes = ARM64_VECTOR_LANES.get(dtype, 4)
        suffix = arm64_get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        # Comment showing what's in this fused loop
        if len(loop.bodies) > 1:
            ops_desc = ", ".join(f"{b.dst}={b.op_type.value}" for b in loop.bodies)
            lines.append(f"{indent}// Fused loop ({len(loop.bodies)} ops): {ops_desc}")
        else:
            b = loop.bodies[0]
            lines.append(f"{indent}// {b.op_type.value.upper()}: {b}")
        
        # Outer loop over rows
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        self.ctx.indent_level += 1
        indent = self.ctx.indent()
        
        # Column loop variable
        lines.append(f"{indent}int _col;")
        
        # Vectorized inner loop
        if loop.vectorizable and cols >= vec_lanes:
            lines.append(f"{indent}for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
            self.ctx.indent_level += 1
            
            # Generate vectorized operations for each body
            vec_lines = self._generate_vectorized_body(loop.bodies, dtype, suffix, vec_type)
            lines.extend(vec_lines)
            
            self.ctx.indent_level -= 1
            indent = self.ctx.indent()
            lines.append(f"{indent}}}")
        
        # Scalar cleanup loop
        lines.append(f"{indent}for (; _col < {cols}; _col++) {{")
        self.ctx.indent_level += 1
        
        # Generate scalar operations for each body
        scalar_lines = self._generate_scalar_body(loop.bodies)
        lines.extend(scalar_lines)
        
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        # Close row loop
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    def _generate_vectorized_body(self, bodies: List[LoopBodyOp], 
                                   dtype: str, suffix: str, vec_type: str) -> List[str]:
        """Generate vectorized NEON operations for loop bodies."""
        lines = []
        indent = self.ctx.indent()
        
        # Track which vectors we've loaded
        loaded_vecs: Dict[str, str] = {}
        
        for body in bodies:
            # Load source vectors if not already loaded
            for i, src in enumerate(body.srcs):
                if src not in loaded_vecs:
                    vname = self.ctx.get_unique_var("_v")
                    lines.append(f"{indent}{vec_type} {vname} = vld1q_{suffix}(&{src}[_row][_col]);")
                    loaded_vecs[src] = vname
            
            # Generate the operation
            result_vec = self.ctx.get_unique_var("_vr")
            op_line = self._generate_vec_op(body, loaded_vecs, vec_type, suffix, result_vec)
            lines.append(f"{indent}{op_line}")
            
            # Store result
            lines.append(f"{indent}vst1q_{suffix}(&{body.dst}[_row][_col], {result_vec});")
            
            # The destination is now available as a loaded vector for subsequent ops
            loaded_vecs[body.dst] = result_vec
        
        return lines
    
    def _generate_vec_op(self, body: LoopBodyOp, loaded: Dict[str, str],
                         vec_type: str, suffix: str, result_var: str) -> str:
        """Generate a single vectorized operation."""
        op = body.op_type
        
        if op == LoopOpType.ADD:
            return f"{vec_type} {result_var} = vaddq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.SUB:
            return f"{vec_type} {result_var} = vsubq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.MUL:
            return f"{vec_type} {result_var} = vmulq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.DIV:
            return f"{vec_type} {result_var} = vdivq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.MAX:
            return f"{vec_type} {result_var} = vmaxq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.MIN:
            return f"{vec_type} {result_var} = vminq_{suffix}({loaded[body.srcs[0]]}, {loaded[body.srcs[1]]});"
        elif op == LoopOpType.ABS:
            return f"{vec_type} {result_var} = vabsq_{suffix}({loaded[body.srcs[0]]});"
        elif op == LoopOpType.NEG:
            return f"{vec_type} {result_var} = vnegq_{suffix}({loaded[body.srcs[0]]});"
        elif op == LoopOpType.SQRT:
            return f"{vec_type} {result_var} = vsqrtq_{suffix}({loaded[body.srcs[0]]});"
        elif op == LoopOpType.RSQRT:
            return f"{vec_type} {result_var} = vrsqrteq_{suffix}({loaded[body.srcs[0]]});"
        elif op == LoopOpType.RELU:
            vzero = self.ctx.get_unique_var("_vzero")
            return f"{vec_type} {vzero} = vdupq_n_{suffix}(0.0f); {vec_type} {result_var} = vmaxq_{suffix}({loaded[body.srcs[0]]}, {vzero});"
        elif op == LoopOpType.RECIP:
            vone = self.ctx.get_unique_var("_vone")
            return f"{vec_type} {vone} = vdupq_n_{suffix}(1.0f); {vec_type} {result_var} = vdivq_{suffix}({vone}, {loaded[body.srcs[0]]});"
        elif op in (LoopOpType.ADDS, LoopOpType.SUBS, LoopOpType.MULS, LoopOpType.DIVS):
            vs = self.ctx.get_unique_var("_vs")
            scalar_load = f"{vec_type} {vs} = vdupq_n_{suffix}({body.scalar});"
            if op == LoopOpType.ADDS:
                return f"{scalar_load} {vec_type} {result_var} = vaddq_{suffix}({loaded[body.srcs[0]]}, {vs});"
            elif op == LoopOpType.SUBS:
                return f"{scalar_load} {vec_type} {result_var} = vsubq_{suffix}({loaded[body.srcs[0]]}, {vs});"
            elif op == LoopOpType.MULS:
                return f"{scalar_load} {vec_type} {result_var} = vmulq_{suffix}({loaded[body.srcs[0]]}, {vs});"
            elif op == LoopOpType.DIVS:
                return f"{scalar_load} {vec_type} {result_var} = vdivq_{suffix}({loaded[body.srcs[0]]}, {vs});"
        elif op == LoopOpType.EXPANDS:
            return f"{vec_type} {result_var} = vdupq_n_{suffix}({body.scalar});"
        elif op == LoopOpType.COPY:
            return f"{vec_type} {result_var} = {loaded[body.srcs[0]]};"
        elif op in (LoopOpType.EXP, LoopOpType.LOG):
            # These don't have NEON intrinsics, fall back to scalar in vectorized section
            # (handled specially - process element by element)
            return f"// {op.value} uses scalar fallback"
        
        return f"// Unknown op: {op}"
    
    def _generate_scalar_body(self, bodies: List[LoopBodyOp]) -> List[str]:
        """Generate scalar C operations for loop bodies."""
        lines = []
        indent = self.ctx.indent()
        
        for body in bodies:
            op = body.op_type
            dst = f"{body.dst}[_row][_col]"
            
            if len(body.srcs) >= 1:
                src0 = f"{body.srcs[0]}[_row][_col]"
            if len(body.srcs) >= 2:
                src1 = f"{body.srcs[1]}[_row][_col]"
            
            if op == LoopOpType.ADD:
                lines.append(f"{indent}{dst} = {src0} + {src1};")
            elif op == LoopOpType.SUB:
                lines.append(f"{indent}{dst} = {src0} - {src1};")
            elif op == LoopOpType.MUL:
                lines.append(f"{indent}{dst} = {src0} * {src1};")
            elif op == LoopOpType.DIV:
                lines.append(f"{indent}{dst} = {src0} / {src1};")
            elif op == LoopOpType.MAX:
                lines.append(f"{indent}{dst} = fmaxf({src0}, {src1});")
            elif op == LoopOpType.MIN:
                lines.append(f"{indent}{dst} = fminf({src0}, {src1});")
            elif op == LoopOpType.ABS:
                lines.append(f"{indent}{dst} = fabsf({src0});")
            elif op == LoopOpType.NEG:
                lines.append(f"{indent}{dst} = -{src0};")
            elif op == LoopOpType.RECIP:
                lines.append(f"{indent}{dst} = 1.0f / {src0};")
            elif op == LoopOpType.EXP:
                lines.append(f"{indent}{dst} = expf({src0});")
            elif op == LoopOpType.LOG:
                lines.append(f"{indent}{dst} = logf({src0});")
            elif op == LoopOpType.SQRT:
                lines.append(f"{indent}{dst} = sqrtf({src0});")
            elif op == LoopOpType.RSQRT:
                lines.append(f"{indent}{dst} = 1.0f / sqrtf({src0});")
            elif op == LoopOpType.RELU:
                lines.append(f"{indent}{dst} = fmaxf({src0}, 0.0f);")
            elif op == LoopOpType.ADDS:
                lines.append(f"{indent}{dst} = {src0} + {body.scalar};")
            elif op == LoopOpType.SUBS:
                lines.append(f"{indent}{dst} = {src0} - {body.scalar};")
            elif op == LoopOpType.MULS:
                lines.append(f"{indent}{dst} = {src0} * {body.scalar};")
            elif op == LoopOpType.DIVS:
                lines.append(f"{indent}{dst} = {src0} / {body.scalar};")
            elif op == LoopOpType.EXPANDS:
                lines.append(f"{indent}{dst} = {body.scalar};")
            elif op == LoopOpType.COPY:
                lines.append(f"{indent}{dst} = {src0};")
            else:
                lines.append(f"{indent}// Unknown op: {op}")
        
        return lines


# =============================================================================
# Loop IR for CUDA - Code Generator
# =============================================================================

class CUDATileLoopCodeGen:
    """
    Code generator that converts TileLoopIR to CUDA kernel code.
    
    This class handles:
    - Thread block mapping to tile dimensions
    - Shared memory optimization
    - Warp-level primitives
    """
    
    def __init__(self, ctx: CUDACodeGenContext = None):
        self.ctx = ctx or CUDACodeGenContext()
    
    def generate(self, ir) -> List[str]:
        """Generate CUDA code from an IR object."""
        if isinstance(ir, TileLoopIR):
            return self._generate_tile_loop(ir)
        elif isinstance(ir, NonLoopIR):
            return ir.code_lines
        elif isinstance(ir, list):
            return ir
        else:
            return [f"// Unknown IR type: {type(ir)}"]
    
    def _generate_tile_loop(self, loop: TileLoopIR) -> List[str]:
        """Generate CUDA code for a TileLoopIR."""
        lines = []
        indent = self.ctx.indent()
        rows, cols, dtype = loop.rows, loop.cols, loop.dtype
        c_type = CUDA_TYPE_MAP.get(dtype, "float")
        
        # Comment showing what's in this fused loop
        if len(loop.bodies) > 1:
            ops_desc = ", ".join(f"{b.dst}={b.op_type.value}" for b in loop.bodies)
            lines.append(f"{indent}// CUDA fused loop ({len(loop.bodies)} ops): {ops_desc}")
        else:
            b = loop.bodies[0]
            lines.append(f"{indent}// CUDA: {b.op_type.value.upper()}: {b}")
        
        # Use thread-level parallelism
        lines.append(f"{indent}// Thread mapping: each thread handles one element")
        lines.append(f"{indent}int _row = threadIdx.y + blockIdx.y * blockDim.y;")
        lines.append(f"{indent}int _col = threadIdx.x + blockIdx.x * blockDim.x;")
        lines.append(f"{indent}if (_row < {rows} && _col < {cols}) {{")
        self.ctx.indent_level += 1
        
        # Generate operations for each body
        for body in loop.bodies:
            op_line = self._generate_cuda_op(body, dtype)
            lines.append(f"{self.ctx.indent()}{op_line}")
        
        self.ctx.indent_level -= 1
        lines.append(f"{indent}}}")
        
        return lines
    
    def _generate_cuda_op(self, body: LoopBodyOp, dtype: str) -> str:
        """Generate a single CUDA operation."""
        op = body.op_type
        dst = f"{body.dst}[_row][_col]"
        
        if len(body.srcs) >= 1:
            src0 = f"{body.srcs[0]}[_row][_col]"
        if len(body.srcs) >= 2:
            src1 = f"{body.srcs[1]}[_row][_col]"
        
        if op == LoopOpType.ADD:
            return f"{dst} = {src0} + {src1};"
        elif op == LoopOpType.SUB:
            return f"{dst} = {src0} - {src1};"
        elif op == LoopOpType.MUL:
            return f"{dst} = {src0} * {src1};"
        elif op == LoopOpType.DIV:
            return f"{dst} = {src0} / {src1};"
        elif op == LoopOpType.MAX:
            return f"{dst} = {cuda_get_intrinsic('max', dtype)}({src0}, {src1});"
        elif op == LoopOpType.MIN:
            return f"{dst} = {cuda_get_intrinsic('min', dtype)}({src0}, {src1});"
        elif op == LoopOpType.ABS:
            return f"{dst} = {cuda_get_intrinsic('abs', dtype)}({src0});"
        elif op == LoopOpType.NEG:
            return f"{dst} = -{src0};"
        elif op == LoopOpType.RECIP:
            return f"{dst} = 1.0f / {src0};"
        elif op == LoopOpType.EXP:
            return f"{dst} = {cuda_get_intrinsic('exp', dtype)}({src0});"
        elif op == LoopOpType.LOG:
            return f"{dst} = {cuda_get_intrinsic('log', dtype)}({src0});"
        elif op == LoopOpType.SQRT:
            return f"{dst} = {cuda_get_intrinsic('sqrt', dtype)}({src0});"
        elif op == LoopOpType.RSQRT:
            return f"{dst} = {cuda_get_intrinsic('rsqrt', dtype)}({src0});"
        elif op == LoopOpType.RELU:
            return f"{dst} = {cuda_get_intrinsic('max', dtype)}({src0}, 0.0f);"
        elif op == LoopOpType.ADDS:
            return f"{dst} = {src0} + {body.scalar};"
        elif op == LoopOpType.SUBS:
            return f"{dst} = {src0} - {body.scalar};"
        elif op == LoopOpType.MULS:
            return f"{dst} = {src0} * {body.scalar};"
        elif op == LoopOpType.DIVS:
            return f"{dst} = {src0} / {body.scalar};"
        elif op == LoopOpType.EXPANDS:
            return f"{dst} = {body.scalar};"
        elif op == LoopOpType.COPY:
            return f"{dst} = {src0};"
        
        return f"// Unknown op: {op}"


# =============================================================================
# Loop IR for Ascend A2/A3 - Code Generator
# =============================================================================

class AscendTileLoopCodeGen:
    """
    Code generator that converts TileLoopIR to Ascend C code.
    
    This class handles:
    - Vector operation mapping
    - Unified Buffer management
    - Data flow pipe synchronization
    """
    
    def __init__(self, ctx: AscendCodeGenContext = None):
        self.ctx = ctx or AscendCodeGenContext()
    
    def generate(self, ir) -> List[str]:
        """Generate Ascend C code from an IR object."""
        if isinstance(ir, TileLoopIR):
            return self._generate_tile_loop(ir)
        elif isinstance(ir, NonLoopIR):
            return ir.code_lines
        elif isinstance(ir, list):
            return ir
        else:
            return [f"// Unknown IR type: {type(ir)}"]
    
    def _generate_tile_loop(self, loop: TileLoopIR) -> List[str]:
        """Generate Ascend C code for a TileLoopIR using vector operations."""
        lines = []
        indent = self.ctx.indent()
        rows, cols, dtype = loop.rows, loop.cols, loop.dtype
        total_elements = rows * cols
        vec_lanes = ASCEND_VECTOR_LANES.get(dtype, 8)
        ascend_type = ASCEND_TYPE_MAP.get(dtype, "float")
        
        # Comment showing what's in this fused loop
        if len(loop.bodies) > 1:
            ops_desc = ", ".join(f"{b.dst}={b.op_type.value}" for b in loop.bodies)
            lines.append(f"{indent}// Ascend C fused vector ops ({len(loop.bodies)} ops): {ops_desc}")
        else:
            b = loop.bodies[0]
            lines.append(f"{indent}// Ascend C: {b.op_type.value.upper()}: {b}")
        
        # Generate block-wise processing
        lines.append(f"{indent}// Vector processing: {total_elements} elements, {vec_lanes} lanes")
        lines.append(f"{indent}uint32_t _dataSize = {total_elements};")
        lines.append(f"{indent}uint32_t _tileLength = {vec_lanes};")
        lines.append(f"{indent}uint32_t _loopCount = (_dataSize + _tileLength - 1) / _tileLength;")
        lines.append(f"")
        lines.append(f"{indent}for (uint32_t _i = 0; _i < _loopCount; _i++) {{")
        self.ctx.indent_level += 1
        indent = self.ctx.indent()
        
        lines.append(f"{indent}uint32_t _offset = _i * _tileLength;")
        lines.append(f"{indent}uint32_t _calcLen = (_offset + _tileLength <= _dataSize) ? _tileLength : (_dataSize - _offset);")
        
        # Generate operations for each body using Ascend C vector APIs
        for body in loop.bodies:
            op_lines = self._generate_ascend_op(body, dtype)
            for op_line in op_lines:
                lines.append(f"{indent}{op_line}")
        
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    def _generate_ascend_op(self, body: LoopBodyOp, dtype: str) -> List[str]:
        """Generate Ascend C vector operations."""
        lines = []
        op = body.op_type
        dst = body.dst
        ascend_op = ascend_get_vector_op(op.value)
        
        if len(body.srcs) >= 1:
            src0 = body.srcs[0]
        if len(body.srcs) >= 2:
            src1 = body.srcs[1]
        
        # Binary operations
        if op in (LoopOpType.ADD, LoopOpType.SUB, LoopOpType.MUL, LoopOpType.DIV,
                  LoopOpType.MAX, LoopOpType.MIN):
            lines.append(f"{ascend_op}({dst}[_offset], {src0}[_offset], {src1}[_offset], _calcLen);")
        
        # Unary operations
        elif op in (LoopOpType.ABS, LoopOpType.NEG, LoopOpType.EXP, LoopOpType.LOG,
                    LoopOpType.SQRT, LoopOpType.RSQRT, LoopOpType.RECIP, LoopOpType.RELU):
            lines.append(f"{ascend_op}({dst}[_offset], {src0}[_offset], _calcLen);")
        
        # Scalar operations
        elif op in (LoopOpType.ADDS, LoopOpType.SUBS, LoopOpType.MULS, LoopOpType.DIVS):
            scalar_op_map = {
                LoopOpType.ADDS: "Adds",
                LoopOpType.SUBS: "Subs",
                LoopOpType.MULS: "Muls",
                LoopOpType.DIVS: "Divs",
            }
            lines.append(f"{scalar_op_map[op]}({dst}[_offset], {src0}[_offset], {body.scalar}, _calcLen);")
        
        # Broadcast
        elif op == LoopOpType.EXPANDS:
            lines.append(f"Duplicate({dst}[_offset], {body.scalar}, _calcLen);")
        
        # Copy
        elif op == LoopOpType.COPY:
            lines.append(f"DataCopy({dst}[_offset], {src0}[_offset], _calcLen);")
        
        else:
            lines.append(f"// Unknown op: {op}")
        
        return lines


# =============================================================================
# Data Types
# =============================================================================

class ElementType(Enum):
    """Supported element types for tiles and scalars."""
    # Floating point types
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    BF16 = "bf16"
    
    # Integer types
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    U1 = "u1"  # Boolean/predicate
    
    # Index type
    INDEX = "index"


class MemorySpace(Enum):
    """Memory space identifiers."""
    GM = "gm"      # Global Memory
    L2 = "l2"      # L2 Cache
    L1 = "l1"      # L1 Cache
    LOCAL = "local"  # Local/On-chip memory


class CompareMode(Enum):
    """Comparison modes for TCMP and CMP instructions."""
    EQ = "EQ"    # Equal
    NE = "NE"    # Not Equal
    LT = "LT"    # Less Than
    LE = "LE"    # Less or Equal
    GT = "GT"    # Greater Than
    GE = "GE"    # Greater or Equal


class RoundMode(Enum):
    """Rounding modes for type conversion."""
    CAST_RINT = "CAST_RINT"
    ROUND_NEAREST = "ROUND_NEAREST"
    ROUND_DOWN = "ROUND_DOWN"
    ROUND_UP = "ROUND_UP"
    ROUND_ZERO = "ROUND_ZERO"


# =============================================================================
# Tile Shape and Type Definitions
# =============================================================================

@dataclass
class TileShape:
    """
    Represents the shape of a tile (2D block of data).
    
    Used to specify dimensions for tile operations and loop iteration bounds.
    """
    rows: int
    cols: int
    
    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"Tile dimensions must be positive: ({self.rows}, {self.cols})")
    
    @property
    def size(self) -> int:
        return self.rows * self.cols
    
    def __str__(self) -> str:
        return f"<{self.rows}x{self.cols}>"


@dataclass
class TileType:
    """
    Complete type specification for a tile.
    
    Combines shape with element type for full type checking.
    """
    shape: TileShape
    element_type: ElementType
    
    def __str__(self) -> str:
        return f"!pto.tile<{self.shape.rows}x{self.shape.cols}x{self.element_type.value}>"
    
    @classmethod
    def create(cls, rows: int, cols: int, dtype: ElementType) -> "TileType":
        return cls(TileShape(rows, cols), dtype)


@dataclass
class MemRefType:
    """Memory reference type for global/local memory access."""
    memory_space: MemorySpace
    element_type: ElementType
    shape: Optional[TileShape] = None
    
    def __str__(self) -> str:
        shape_str = f"{self.shape}" if self.shape else "..."
        return f"!pto.memref<{self.memory_space.value},{shape_str},{self.element_type.value}>"


@dataclass
class EventType:
    """Event type for synchronization."""
    name: str = ""
    
    def __str__(self) -> str:
        return f"!pto.event<{self.name}>" if self.name else "!pto.event<...>"


# =============================================================================
# Operand Definitions
# =============================================================================

@dataclass
class TileOperand:
    """A tile operand (register/value)."""
    name: str
    tile_type: TileType
    
    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class ScalarOperand:
    """A scalar operand."""
    name: str
    element_type: ElementType
    
    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class MemRefOperand:
    """A memory reference operand."""
    name: str
    memref_type: MemRefType
    
    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class IndexOperand:
    """An index operand for addressing."""
    name: str
    
    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class ImmediateOperand:
    """An immediate constant value."""
    value: Union[int, float]
    
    def __str__(self) -> str:
        return str(self.value)


Operand = Union[TileOperand, ScalarOperand, MemRefOperand, IndexOperand, ImmediateOperand]


# =============================================================================
# Base Instruction Classes
# =============================================================================

class PTOInstruction(ABC):
    """
    Base class for all PTO instructions.
    
    Dependency Analysis Fields (initialized by SimplifyAndColor):
        instr_id: Unique instruction index within the program
        fanin_pred: List of predecessor instruction IDs that this instruction depends on
        fanin_succ: List of successor instruction IDs (for cyclic dependencies in loops)
        color: Color assigned by graph coloring algorithm (-1 = uncolored)
    """
    
    # Dependency analysis fields - initialized lazily by SimplifyAndColor
    instr_id: int = -1
    fanin_pred: List[int] = None  # Predecessor dependencies
    fanin_succ: List[int] = None  # Successor dependencies (cyclic in loops)
    color: int = -1               # Graph coloring result
    
    def init_dependency_fields(self, instr_id: int):
        """Initialize dependency tracking fields."""
        self.instr_id = instr_id
        self.fanin_pred = []
        self.fanin_succ = []
        self.color = -1
    
    def add_pred(self, pred_id: int):
        """Add a predecessor dependency."""
        if self.fanin_pred is None:
            self.fanin_pred = []
        if pred_id not in self.fanin_pred and pred_id != self.instr_id:
            self.fanin_pred.append(pred_id)
    
    def add_succ(self, succ_id: int):
        """Add a successor dependency (for loop-carried dependencies)."""
        if self.fanin_succ is None:
            self.fanin_succ = []
        if succ_id not in self.fanin_succ and succ_id != self.instr_id:
            self.fanin_succ.append(succ_id)
    
    def get_all_neighbors(self) -> List[int]:
        """Get all neighbor instruction IDs (both pred and succ)."""
        neighbors = set()
        if self.fanin_pred:
            neighbors.update(self.fanin_pred)
        if self.fanin_succ:
            neighbors.update(self.fanin_succ)
        return list(neighbors)
    
    def get_degree(self) -> int:
        """Get the degree (number of neighbors) of this instruction."""
        return len(self.get_all_neighbors())
    
    @property
    @abstractmethod
    def opcode(self) -> str:
        """Return the instruction opcode."""
        pass
    
    @abstractmethod
    def to_pto_as(self) -> str:
        """Generate PTO assembly syntax."""
        pass
    
    def to_pto_as_with_deps(self) -> str:
        """Generate PTO assembly with dependency info."""
        base_asm = self.to_pto_as()
        deps_info = []
        if self.instr_id >= 0:
            deps_info.append(f"id={self.instr_id}")
        if self.color >= 0:
            deps_info.append(f"color={self.color}")
        if self.fanin_pred:
            deps_info.append(f"pred=[{','.join(map(str, self.fanin_pred))}]")
        if self.fanin_succ:
            deps_info.append(f"succ=[{','.join(map(str, self.fanin_succ))}]")
        if deps_info:
            return f"{base_asm}  // {' '.join(deps_info)}"
        return base_asm
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> CodeGenIR:
        """
        Generate ARM64 intermediate representation for this instruction.
        
        Returns a TileLoopIR for elementwise operations that can be fused,
        or NonLoopIR/List[str] for other operations.
        
        This is the preferred method for code generation as it enables
        loop fusion without C grammar parsing.
        
        Args:
            ctx: ARM64 code generation context for tracking state
            
        Returns:
            CodeGenIR (TileLoopIR, NonLoopIR, or List[str])
        """
        # Default: return non-loop IR with a comment
        return NonLoopIR(
            op_type="unknown",
            code_lines=[f"{ctx.indent()}// {self.opcode}: Not implemented"],
            comment=self.opcode
        )
    
    def codegen_arm64(self, ctx: ARM64CodeGenContext) -> List[str]:
        """
        Generate ARM64 NEON intrinsic code for this instruction.
        
        This method generates IR first, then converts to C code.
        For direct C code generation without fusion, use this method.
        For fusion-enabled code generation, use codegen_arm64_ir() and
        pass results through TileLoopCodeGen.
        
        Args:
            ctx: ARM64 code generation context for tracking state
            
        Returns:
            List of C code lines implementing this instruction
        """
        ir = self.codegen_arm64_ir(ctx)
        codegen = TileLoopCodeGen(ctx)
        return codegen.generate(ir)
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> CodeGenIR:
        """
        Generate CUDA intermediate representation for this instruction.
        
        Returns a TileLoopIR for elementwise operations that can be fused,
        or NonLoopIR/List[str] for other operations.
        
        Args:
            ctx: CUDA code generation context for tracking state
            
        Returns:
            CodeGenIR (TileLoopIR, NonLoopIR, or List[str])
        """
        # Default: return non-loop IR with a comment
        return NonLoopIR(
            op_type="unknown",
            code_lines=[f"{ctx.indent()}// CUDA: {self.opcode}: Not implemented"],
            comment=self.opcode
        )
    
    def codegen_cuda(self, ctx: CUDACodeGenContext) -> List[str]:
        """
        Generate NVIDIA CUDA code for this instruction.
        
        This method generates IR first, then converts to CUDA C code.
        Uses CUDA thread-level parallelism and intrinsics.
        
        Args:
            ctx: CUDA code generation context for tracking state
            
        Returns:
            List of CUDA code lines implementing this instruction
        """
        ir = self.codegen_cuda_ir(ctx)
        codegen = CUDATileLoopCodeGen(ctx)
        return codegen.generate(ir)
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> CodeGenIR:
        """
        Generate Ascend A2/A3 intermediate representation for this instruction.
        
        Returns a TileLoopIR for elementwise operations that can be fused,
        or NonLoopIR/List[str] for other operations.
        
        Args:
            ctx: Ascend code generation context for tracking state
            
        Returns:
            CodeGenIR (TileLoopIR, NonLoopIR, or List[str])
        """
        # Default: return non-loop IR with a comment
        return NonLoopIR(
            op_type="unknown",
            code_lines=[f"{ctx.indent()}// Ascend A2/A3: {self.opcode}: Not implemented"],
            comment=self.opcode
        )
    
    def codegen_ascend_910b(self, ctx: AscendCodeGenContext) -> List[str]:
        """
        Generate Huawei Ascend A2/A3 (Ascend C) code for this instruction.
        
        This method generates IR first, then converts to Ascend C code.
        Uses Ascend C vector operations and DataCopy primitives.
        
        Args:
            ctx: Ascend code generation context for tracking state
            
        Returns:
            List of Ascend C code lines implementing this instruction
        """
        ir = self.codegen_ascend_910b_ir(ctx)
        codegen = AscendTileLoopCodeGen(ctx)
        return codegen.generate(ir)


class TileInstruction(PTOInstruction):
    """Base class for tile operations."""
    pass


class ScalarInstruction(PTOInstruction):
    """Base class for scalar operations."""
    pass


class ControlFlowInstruction(PTOInstruction):
    """Base class for control flow instructions."""
    pass


# =============================================================================
# Tile Instructions - Memory Operations
# =============================================================================

@dataclass
class TLOAD(TileInstruction):
    """Load data from GlobalTensor (GM) into a Tile."""
    dst: TileOperand
    src_mem: MemRefOperand
    row_offset: Union[IndexOperand, ImmediateOperand]
    col_offset: Union[IndexOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "TLOAD"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tload {self.src_mem}[{self.row_offset}, {self.col_offset}] : ({self.src_mem.memref_type}, index, index) -> {self.dst.tile_type}"
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> NonLoopIR:
        """Generate NonLoopIR for TLOAD (memory ops are not fusable)."""
        indent = ctx.indent()
        dst_name = self.dst.name
        src_name = self.src_mem.name
        rows = self.dst.tile_type.shape.rows
        cols = self.dst.tile_type.shape.cols
        row_off = self.row_offset.value if isinstance(self.row_offset, ImmediateOperand) else self.row_offset.name
        col_off = self.col_offset.value if isinstance(self.col_offset, ImmediateOperand) else self.col_offset.name
        
        lines = [
            f"{indent}// TLOAD: {dst_name} = tload {src_name}[{row_off}, {col_off}]",
            f"{indent}for (int _row = 0; _row < {rows}; _row++) {{",
            f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{",
            f"{indent}        {dst_name}[_row][_col] = {src_name}[(_row + {row_off}) * {cols} + (_col + {col_off})];",
            f"{indent}    }}",
            f"{indent}}}"
        ]
        return NonLoopIR(op_type="load", code_lines=lines, comment="TLOAD")


@dataclass
class TSTORE(TileInstruction):
    """Store data from a Tile into GlobalTensor (GM)."""
    src: TileOperand
    dst_mem: MemRefOperand
    row_offset: Union[IndexOperand, ImmediateOperand]
    col_offset: Union[IndexOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "TSTORE"
    
    def to_pto_as(self) -> str:
        return f"tstore {self.src}, {self.dst_mem}[{self.row_offset}, {self.col_offset}]"
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> NonLoopIR:
        """Generate NonLoopIR for TSTORE (memory ops are not fusable)."""
        indent = ctx.indent()
        src_name = self.src.name
        dst_name = self.dst_mem.name
        rows = self.src.tile_type.shape.rows
        cols = self.src.tile_type.shape.cols
        row_off = self.row_offset.value if isinstance(self.row_offset, ImmediateOperand) else self.row_offset.name
        col_off = self.col_offset.value if isinstance(self.col_offset, ImmediateOperand) else self.col_offset.name
        
        lines = [
            f"{indent}// TSTORE: tstore {src_name}, {dst_name}[{row_off}, {col_off}]",
            f"{indent}for (int _row = 0; _row < {rows}; _row++) {{",
            f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{",
            f"{indent}        {dst_name}[(_row + {row_off}) * {cols} + (_col + {col_off})] = {src_name}[_row][_col];",
            f"{indent}    }}",
            f"{indent}}}"
        ]
        return NonLoopIR(op_type="store", code_lines=lines, comment="TSTORE")


@dataclass
class TSTORE_FP(TileInstruction):
    """Store accumulator tile with scaling for vector quantization."""
    src: TileOperand
    fp: TileOperand  # Scaling tile
    dst_mem: MemRefOperand
    row_offset: Union[IndexOperand, ImmediateOperand]
    col_offset: Union[IndexOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "TSTORE_FP"
    
    def to_pto_as(self) -> str:
        return f"tstore.fp {self.src}, {self.fp}, {self.dst_mem}[{self.row_offset}, {self.col_offset}]"


@dataclass
class MGATHER(TileInstruction):
    """Gather-load elements from global memory using per-element indices."""
    dst: TileOperand
    mem: MemRefOperand
    idx: TileOperand
    
    @property
    def opcode(self) -> str:
        return "MGATHER"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = mgather {self.mem}, {self.idx} : {self.mem.memref_type}, {self.idx.tile_type} -> {self.dst.tile_type}"


@dataclass
class MSCATTER(TileInstruction):
    """Scatter-store elements from a tile into global memory."""
    src: TileOperand
    mem: MemRefOperand
    idx: TileOperand
    
    @property
    def opcode(self) -> str:
        return "MSCATTER"
    
    def to_pto_as(self) -> str:
        return f"mscatter {self.src}, {self.mem}, {self.idx} : {self.mem.memref_type}, {self.src.tile_type}, {self.idx.tile_type}"


# =============================================================================
# Tile Instructions - Element Access
# =============================================================================

@dataclass
class GETVAL(TileInstruction):
    """Read a single tile element into a scalar value."""
    dst: ScalarOperand
    src: TileOperand
    offset: Union[IndexOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "GETVAL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = getval {self.src}, {self.offset} : {self.src.tile_type} -> {self.dst.element_type.value}"


@dataclass
class SETVAL(TileInstruction):
    """Write a scalar value into a single tile element."""
    dst: TileOperand
    offset: Union[IndexOperand, ImmediateOperand]
    val: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "SETVAL"
    
    def to_pto_as(self) -> str:
        return f"setval {self.dst}, {self.offset}, {self.val} : {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Elementwise Unary Operations
# =============================================================================

@dataclass
class TABS(TileInstruction):
    """Elementwise absolute value of a tile."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TABS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tabs {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        """Create architecture-agnostic TileLoopIR for TABS."""
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.ABS,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        """Generate TileLoopIR for TABS (ARM64)."""
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        """Generate TileLoopIR for TABS (CUDA)."""
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        """Generate TileLoopIR for TABS (Ascend A2/A3)."""
        return self._make_ir()


@dataclass
class TNEG(TileInstruction):
    """Elementwise negation of a tile."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TNEG"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tneg {self.src} : {self.src.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        """Create architecture-agnostic TileLoopIR."""
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.NEG,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TNOT(TileInstruction):
    """Elementwise bitwise NOT of a tile."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TNOT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tnot {self.src} : {self.src.tile_type}"


@dataclass
class TEXP(TileInstruction):
    """Elementwise exponential."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TEXP"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = texp {self.src} : {self.src.tile_type}"
    
    def _make_ir(self, vectorizable: bool = False) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.EXP,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=vectorizable
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=False)  # exp() has no NEON intrinsic
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=True)  # CUDA has __expf
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=True)  # Ascend has Exp vector op


@dataclass
class TLOG(TileInstruction):
    """Elementwise natural logarithm."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TLOG"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tlog {self.src} : {self.src.tile_type}"
    
    def _make_ir(self, vectorizable: bool = False) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.LOG,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=vectorizable
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=False)  # log() has no NEON intrinsic
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=True)  # CUDA has __logf
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir(vectorizable=True)  # Ascend has Ln vector op


@dataclass
class TSQRT(TileInstruction):
    """Elementwise square root."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSQRT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsqrt {self.src} : {self.src.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.SQRT,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TRSQRT(TileInstruction):
    """Elementwise reciprocal square root."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TRSQRT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trsqrt {self.src} : {self.src.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.RSQRT,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TRECIP(TileInstruction):
    """Elementwise reciprocal."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TRECIP"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trecip {self.src} : {self.src.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.RECIP,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TRELU(TileInstruction):
    """Elementwise ReLU."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TRELU"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trelu {self.src} : {self.src.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.RELU,
                dst=self.dst.name,
                srcs=[self.src.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TLRELU(TileInstruction):
    """Leaky ReLU with scalar slope."""
    dst: TileOperand
    src: TileOperand
    slope: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TLRELU"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tlrelu {self.src}, {self.slope} : {self.src.tile_type}, {self.slope.element_type.value}"


# =============================================================================
# Tile Instructions - Elementwise Binary Operations
# =============================================================================

@dataclass
class TADD(TileInstruction):
    """Elementwise add of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TADD"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tadd {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.ADD,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name],
                comment=f"TADD: {self.dst.name} = {self.src0.name} + {self.src1.name}"
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TSUB(TileInstruction):
    """Elementwise subtract of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSUB"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsub {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.SUB,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TMUL(TileInstruction):
    """Elementwise multiply of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMUL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmul {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.MUL,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TDIV(TileInstruction):
    """Elementwise division of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TDIV"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tdiv {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.DIV,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TREM(TileInstruction):
    """Elementwise remainder of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TREM"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trem {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TMAX(TileInstruction):
    """Elementwise maximum of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMAX"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmax {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.MAX,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TMIN(TileInstruction):
    """Elementwise minimum of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMIN"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmin {self.src0}, {self.src1} : {self.src0.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src0.tile_type.shape.rows,
            cols=self.src0.tile_type.shape.cols,
            dtype=self.src0.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.MIN,
                dst=self.dst.name,
                srcs=[self.src0.name, self.src1.name]
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


# Bitwise operations
@dataclass
class TAND(TileInstruction):
    """Elementwise bitwise AND of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TAND"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tand {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TOR(TileInstruction):
    """Elementwise bitwise OR of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TOR"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tor {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TXOR(TileInstruction):
    """Elementwise bitwise XOR of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TXOR"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = txor {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TSHL(TileInstruction):
    """Elementwise shift-left of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSHL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tshl {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TSHR(TileInstruction):
    """Elementwise shift-right of two tiles."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSHR"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tshr {self.src0}, {self.src1} : {self.src0.tile_type}"


# =============================================================================
# Tile Instructions - Scalar Operations
# =============================================================================

@dataclass
class TADDS(TileInstruction):
    """Elementwise add a scalar to a tile."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TADDS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tadds {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.ADDS,
                dst=self.dst.name,
                srcs=[self.src.name],
                scalar=self.scalar.name
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TSUBS(TileInstruction):
    """Elementwise subtract a scalar from a tile."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TSUBS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsubs {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TMULS(TileInstruction):
    """Elementwise multiply a tile by a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TMULS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmuls {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.MULS,
                dst=self.dst.name,
                srcs=[self.src.name],
                scalar=self.scalar.name
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TDIVS(TileInstruction):
    """Elementwise division with a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TDIVS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tdivs {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.src.tile_type.shape.rows,
            cols=self.src.tile_type.shape.cols,
            dtype=self.src.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.DIVS,
                dst=self.dst.name,
                srcs=[self.src.name],
                scalar=self.scalar.name
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TREMS(TileInstruction):
    """Elementwise remainder with a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TREMS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trems {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TMAXS(TileInstruction):
    """Elementwise max of a tile and a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TMAXS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmaxs {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TMINS(TileInstruction):
    """Elementwise minimum of a tile and a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TMINS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmins {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TANDS(TileInstruction):
    """Elementwise bitwise AND of a tile and a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TANDS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tands {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TORS(TileInstruction):
    """Elementwise bitwise OR of a tile and a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TORS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tors {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


@dataclass
class TXORS(TileInstruction):
    """Elementwise bitwise XOR of a tile and a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TXORS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = txors {self.src}, {self.scalar} : {self.src.tile_type}, {self.scalar.element_type.value}"


# =============================================================================
# Tile Instructions - Ternary Operations
# =============================================================================

@dataclass
class TADDC(TileInstruction):
    """Elementwise ternary add: src0 + src1 + src2."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    src2: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TADDC"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = taddc {self.src0}, {self.src1}, {self.src2} : {self.src0.tile_type}"


@dataclass
class TADDSC(TileInstruction):
    """Elementwise fused add: src0 + scalar + src1."""
    dst: TileOperand
    src0: TileOperand
    scalar: ScalarOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TADDSC"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = taddsc {self.src0}, {self.scalar}, {self.src1} : {self.src0.tile_type}, {self.scalar.element_type.value}, {self.src1.tile_type}"


@dataclass
class TSUBC(TileInstruction):
    """Elementwise ternary: src0 - src1 + src2."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    src2: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSUBC"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsubc {self.src0}, {self.src1}, {self.src2} : {self.src0.tile_type}"


@dataclass
class TSUBSC(TileInstruction):
    """Elementwise fused: src0 - scalar + src1."""
    dst: TileOperand
    src0: TileOperand
    scalar: ScalarOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSUBSC"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsubsc {self.src0}, {self.scalar}, {self.src1} : {self.src0.tile_type}, {self.scalar.element_type.value}, {self.src1.tile_type}"


# =============================================================================
# Tile Instructions - Comparison Operations
# =============================================================================

@dataclass
class TCMP(TileInstruction):
    """Compare two tiles and write a packed predicate mask."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    cmp_mode: CompareMode = CompareMode.EQ
    
    @property
    def opcode(self) -> str:
        return "TCMP"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcmp {self.src0}, {self.src1} {{cmpMode = #pto.cmp<{self.cmp_mode.value}>}} : {self.src0.tile_type} -> {self.dst.tile_type}"


@dataclass
class TCMPS(TileInstruction):
    """Compare a tile against a scalar."""
    dst: TileOperand
    src: TileOperand
    scalar: ScalarOperand
    cmp_mode: CompareMode = CompareMode.EQ
    
    @property
    def opcode(self) -> str:
        return "TCMPS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcmps {self.src}, {self.scalar} {{cmpMode = #pto.cmp<{self.cmp_mode.value}>}} : {self.src.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Selection Operations
# =============================================================================

@dataclass
class TSEL(TileInstruction):
    """Select between two tiles using a mask tile."""
    dst: TileOperand
    mask: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSEL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsel {self.mask}, {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TSELS(TileInstruction):
    """Select one of two source tiles using a scalar selectMode."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    select_mode: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TSELS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tsels {self.src0}, {self.src1}, {self.select_mode} : {self.src0.tile_type}"


# =============================================================================
# Tile Instructions - Matrix Operations
# =============================================================================

@dataclass
class TMATMUL(TileInstruction):
    """Matrix multiply (GEMM)."""
    dst: TileOperand
    a: TileOperand
    b: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMATMUL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmatmul {self.a}, {self.b} : ({self.a.tile_type}, {self.b.tile_type}) -> {self.dst.tile_type}"
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> NonLoopIR:
        """Generate NonLoopIR for TMATMUL (not fusable with elementwise ops)."""
        indent = ctx.indent()
        dst, a, b = self.dst.name, self.a.name, self.b.name
        m = self.a.tile_type.shape.rows
        k = self.a.tile_type.shape.cols
        n = self.b.tile_type.shape.cols
        
        lines = [
            f"{indent}// TMATMUL: {dst} = tmatmul {a}, {b}",
            f"{indent}// Dimensions: [{m}x{k}] @ [{k}x{n}] -> [{m}x{n}]",
            f"{indent}for (int _i = 0; _i < {m}; _i++) {{",
            f"{indent}    for (int _j = 0; _j < {n}; _j++) {{",
            f"{indent}        float _sum = 0.0f;",
            f"{indent}        for (int _k = 0; _k < {k}; _k++) {{",
            f"{indent}            _sum += {a}[_i][_k] * {b}[_k][_j];",
            f"{indent}        }}",
            f"{indent}        {dst}[_i][_j] = _sum;",
            f"{indent}    }}",
            f"{indent}}}"
        ]
        return NonLoopIR(op_type="matmul", code_lines=lines, comment="TMATMUL")


@dataclass
class TMATMUL_ACC(TileInstruction):
    """Matrix multiply with accumulator input."""
    dst: TileOperand
    acc: TileOperand
    a: TileOperand
    b: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMATMUL_ACC"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmatmul.acc {self.acc}, {self.a}, {self.b} : ({self.acc.tile_type}, {self.a.tile_type}, {self.b.tile_type}) -> {self.dst.tile_type}"


@dataclass
class TMATMUL_BIAS(TileInstruction):
    """Matrix multiply with bias add."""
    dst: TileOperand
    a: TileOperand
    b: TileOperand
    bias: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMATMUL_BIAS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmatmul.bias {self.a}, {self.b}, {self.bias} : ({self.a.tile_type}, {self.b.tile_type}, {self.bias.tile_type}) -> {self.dst.tile_type}"


@dataclass
class TMATMUL_MX(TileInstruction):
    """Matrix multiply with scaling tiles for mixed-precision."""
    dst: TileOperand
    a: TileOperand
    a_scale: TileOperand
    b: TileOperand
    b_scale: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMATMUL_MX"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmatmul.mx {self.a}, {self.a_scale}, {self.b}, {self.b_scale} : ({self.a.tile_type}, {self.a_scale.tile_type}, {self.b.tile_type}, {self.b_scale.tile_type}) -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Reduction Operations
# =============================================================================

@dataclass
class TROWSUM(TileInstruction):
    """Reduce each row by summing across columns."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWSUM"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowsum {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> NonLoopIR:
        """Generate NonLoopIR for TROWSUM (reduction, not fusable)."""
        indent = ctx.indent()
        dst, src = self.dst.name, self.src.name
        rows = self.src.tile_type.shape.rows
        cols = self.src.tile_type.shape.cols
        
        lines = [
            f"{indent}// TROWSUM: {dst} = trowsum {src}",
            f"{indent}for (int _row = 0; _row < {rows}; _row++) {{",
            f"{indent}    float _sum = 0.0f;",
            f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{",
            f"{indent}        _sum += {src}[_row][_col];",
            f"{indent}    }}",
            f"{indent}    {dst}[_row][0] = _sum;",
            f"{indent}}}"
        ]
        return NonLoopIR(op_type="reduction", code_lines=lines, comment="TROWSUM")


@dataclass
class TROWMAX(TileInstruction):
    """Reduce each row by taking maximum across columns."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWMAX"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowmax {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TROWMIN(TileInstruction):
    """Reduce each row by taking minimum across columns."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWMIN"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowmin {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TCOLSUM(TileInstruction):
    """Reduce each column by summing across rows."""
    dst: TileOperand
    src: TileOperand
    is_binary: bool = False
    
    @property
    def opcode(self) -> str:
        return "TCOLSUM"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcolsum {self.src} {{isBinary = {'true' if self.is_binary else 'false'}}} : {self.src.tile_type} -> {self.dst.tile_type}"
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> NonLoopIR:
        """Generate NonLoopIR for TCOLSUM (reduction, not fusable)."""
        indent = ctx.indent()
        dst, src = self.dst.name, self.src.name
        rows = self.src.tile_type.shape.rows
        cols = self.src.tile_type.shape.cols
        
        lines = [
            f"{indent}// TCOLSUM: {dst} = tcolsum {src}",
            f"{indent}for (int _col = 0; _col < {cols}; _col++) {{",
            f"{indent}    float _sum = 0.0f;",
            f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{",
            f"{indent}        _sum += {src}[_row][_col];",
            f"{indent}    }}",
            f"{indent}    {dst}[0][_col] = _sum;",
            f"{indent}}}"
        ]
        return NonLoopIR(op_type="reduction", code_lines=lines, comment="TCOLSUM")


@dataclass
class TCOLMAX(TileInstruction):
    """Reduce each column by taking maximum across rows."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TCOLMAX"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcolmax {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TCOLMIN(TileInstruction):
    """Reduce each column by taking minimum across rows."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TCOLMIN"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcolmin {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Broadcast/Expand Operations
# =============================================================================

@dataclass
class TEXPANDS(TileInstruction):
    """Broadcast a scalar into a destination tile."""
    dst: TileOperand
    scalar: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "TEXPANDS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = texpands {self.scalar} : {self.scalar.element_type.value}, {self.dst.tile_type}"
    
    def _make_ir(self) -> TileLoopIR:
        return TileLoopIR(
            rows=self.dst.tile_type.shape.rows,
            cols=self.dst.tile_type.shape.cols,
            dtype=self.dst.tile_type.element_type.value,
            bodies=[LoopBodyOp(
                op_type=LoopOpType.EXPANDS,
                dst=self.dst.name,
                srcs=[],
                scalar=self.scalar.name
            )],
            vectorizable=True
        )
    
    def codegen_arm64_ir(self, ctx: ARM64CodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_cuda_ir(self, ctx: CUDACodeGenContext) -> TileLoopIR:
        return self._make_ir()
    
    def codegen_ascend_910b_ir(self, ctx: AscendCodeGenContext) -> TileLoopIR:
        return self._make_ir()


@dataclass
class TROWEXPAND(TileInstruction):
    """Broadcast first element of each row across the row."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWEXPAND"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowexpand {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TCOLEXPAND(TileInstruction):
    """Broadcast first element of each column across the column."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TCOLEXPAND"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcolexpand {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TROWEXPANDMUL(TileInstruction):
    """Row-wise broadcast multiply."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWEXPANDMUL"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowexpandmul {self.src0}, {self.src1} : {self.src0.tile_type}, {self.src1.tile_type} -> {self.dst.tile_type}"


@dataclass
class TROWEXPANDDIV(TileInstruction):
    """Row-wise broadcast divide."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWEXPANDDIV"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowexpanddiv {self.src0}, {self.src1} : {self.src0.tile_type}, {self.src1.tile_type} -> {self.dst.tile_type}"


@dataclass
class TROWEXPANDSUB(TileInstruction):
    """Row-wise broadcast subtract."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TROWEXPANDSUB"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = trowexpandsub {self.src0}, {self.src1} : {self.src0.tile_type}, {self.src1.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Data Movement and Reshape
# =============================================================================

@dataclass
class TTRANS(TileInstruction):
    """Transpose a tile."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TTRANS"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = ttrans {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TRESHAPE(TileInstruction):
    """Reinterpret a tile as another shape."""
    dst: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TRESHAPE"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = treshape {self.src} : {self.src.tile_type}"


@dataclass
class TEXTRACT(TileInstruction):
    """Extract a sub-tile from a source tile."""
    dst: TileOperand
    src: TileOperand
    r0: Union[IndexOperand, ImmediateOperand]
    r1: Union[IndexOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "TEXTRACT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = textract {self.src}[{self.r0}, {self.r1}] : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TGATHER(TileInstruction):
    """Gather/select elements using an index tile."""
    dst: TileOperand
    src: TileOperand
    indices: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TGATHER"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tgather {self.src}, {self.indices} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TGATHERB(TileInstruction):
    """Gather elements using byte offsets."""
    dst: TileOperand
    src: TileOperand
    offsets: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TGATHERB"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tgatherb {self.src}, {self.offsets} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TSCATTER(TileInstruction):
    """Scatter rows using per-element row indices."""
    dst: TileOperand
    src: TileOperand
    idx: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSCATTER"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tscatter {self.src}, {self.idx} : {self.src.tile_type}, {self.idx.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Type Conversion
# =============================================================================

@dataclass
class TCVT(TileInstruction):
    """Elementwise type conversion."""
    dst: TileOperand
    src: TileOperand
    rmode: RoundMode = RoundMode.CAST_RINT
    
    @property
    def opcode(self) -> str:
        return "TCVT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tcvt {self.src} {{rmode = #pto.round_mode<{self.rmode.value}>}} : {self.src.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Move Operations
# =============================================================================

class TMovMode(Enum):
    """TMOV operation modes."""
    M2L = "m2l"   # Matrix to Left
    M2R = "m2r"   # Matrix to Right
    M2B = "m2b"   # Matrix to Bias
    M2S = "m2s"   # Matrix to Scale
    A2V = "a2v"   # Accumulator to Vector
    V2V = "v2v"   # Vector to Vector


@dataclass
class TMOV(TileInstruction):
    """Move/copy between tiles with optional conversion."""
    dst: TileOperand
    src: TileOperand
    mode: TMovMode
    
    @property
    def opcode(self) -> str:
        return "TMOV"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmov.{self.mode.value} {self.src} : {self.src.tile_type} -> {self.dst.tile_type}"


@dataclass
class TMOV_FP(TileInstruction):
    """Move with scaling tile for vector quantization."""
    dst: TileOperand
    src: TileOperand
    fp: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TMOV_FP"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmov.fp {self.src}, {self.fp} : {self.src.tile_type}, {self.fp.tile_type} -> {self.dst.tile_type}"


# =============================================================================
# Tile Instructions - Special Operations
# =============================================================================

@dataclass
class TCI(TileInstruction):
    """Generate contiguous integer sequence."""
    dst: TileOperand
    start: ScalarOperand
    descending: bool = False
    
    @property
    def opcode(self) -> str:
        return "TCI"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tci {self.start} {{descending = {'true' if self.descending else 'false'}}} : {self.dst.tile_type}"


@dataclass
class TPRELU(TileInstruction):
    """Parametric ReLU with per-element slope."""
    dst: TileOperand
    src0: TileOperand
    src1: TileOperand  # Slope tile
    
    @property
    def opcode(self) -> str:
        return "TPRELU"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tprelu {self.src0}, {self.src1} : {self.src0.tile_type}"


@dataclass
class TSORT32(TileInstruction):
    """Sort 32-element block and produce index mapping."""
    dst: TileOperand
    idx: TileOperand
    src: TileOperand
    
    @property
    def opcode(self) -> str:
        return "TSORT32"
    
    def to_pto_as(self) -> str:
        return f"{self.dst}, {self.idx} = tsort32 {self.src} : {self.src.tile_type} -> ({self.dst.tile_type}, {self.idx.tile_type})"


@dataclass
class TMRGSORT(TileInstruction):
    """Merge two sorted tiles into one sorted tile (merge sort operation)."""
    dst: TileOperand      # Merged sorted result
    src0: TileOperand     # First sorted input
    src1: TileOperand     # Second sorted input
    
    @property
    def opcode(self) -> str:
        return "TMRGSORT"
    
    def to_pto_as(self) -> str:
        return f"{self.dst} = tmrgsort {self.src0}, {self.src1} : ({self.src0.tile_type}, {self.src1.tile_type}) -> {self.dst.tile_type}"


@dataclass
class TASSIGN(TileInstruction):
    """Bind a tile to an on-chip address."""
    tile: TileOperand
    addr: IndexOperand
    
    @property
    def opcode(self) -> str:
        return "TASSIGN"
    
    def to_pto_as(self) -> str:
        return f"tassign {self.tile}, {self.addr} : {self.tile.tile_type}, index"


@dataclass
class TSYNC(TileInstruction):
    """Synchronize PTO execution."""
    e0: str  # Event name
    e1: str  # Event name
    
    @property
    def opcode(self) -> str:
        return "TSYNC"
    
    def to_pto_as(self) -> str:
        return f"tsync %{self.e0}, %{self.e1} : !pto.event<...>, !pto.event<...>"


# =============================================================================
# Scalar Instructions
# =============================================================================

@dataclass
class SADD(ScalarInstruction):
    """Add two scalar values."""
    dst: ScalarOperand
    src0: ScalarOperand
    src1: Union[ScalarOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "ADD"
    
    def to_pto_as(self) -> str:
        return f"ADD {self.dst}:{self.dst.element_type.value}, {self.src0}:{self.src0.element_type.value}, {self.src1}"


@dataclass
class SSUB(ScalarInstruction):
    """Subtract two scalar values."""
    dst: ScalarOperand
    src0: ScalarOperand
    src1: Union[ScalarOperand, ImmediateOperand]
    
    @property
    def opcode(self) -> str:
        return "SUB"
    
    def to_pto_as(self) -> str:
        return f"SUB {self.dst}:{self.dst.element_type.value}, {self.src0}:{self.src0.element_type.value}, {self.src1}"


@dataclass
class SMUL(ScalarInstruction):
    """Multiply two scalar values."""
    dst: ScalarOperand
    src0: ScalarOperand
    src1: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "MUL"
    
    def to_pto_as(self) -> str:
        return f"MUL {self.dst}:{self.dst.element_type.value}, {self.src0}:{self.src0.element_type.value}, {self.src1}:{self.src1.element_type.value}"


@dataclass
class SDIV(ScalarInstruction):
    """Divide two scalar values."""
    dst: ScalarOperand
    src0: ScalarOperand
    src1: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "DIV"
    
    def to_pto_as(self) -> str:
        return f"DIV {self.dst}:{self.dst.element_type.value}, {self.src0}:{self.src0.element_type.value}, {self.src1}:{self.src1.element_type.value}"


@dataclass
class SMOV(ScalarInstruction):
    """Move scalar value between registers."""
    dst: ScalarOperand
    src: ScalarOperand
    
    @property
    def opcode(self) -> str:
        return "MOV"
    
    def to_pto_as(self) -> str:
        return f"MOV {self.dst}:{self.dst.element_type.value}, {self.src}:{self.src.element_type.value}"


@dataclass
class SLI(ScalarInstruction):
    """Load immediate constant."""
    dst: ScalarOperand
    imm: ImmediateOperand
    
    @property
    def opcode(self) -> str:
        return "LI"
    
    def to_pto_as(self) -> str:
        return f"LI {self.dst}:{self.dst.element_type.value}, {self.imm}"


@dataclass
class SCMP(ScalarInstruction):
    """Compare two scalar values."""
    dst: ScalarOperand
    src0: ScalarOperand
    src1: ScalarOperand
    cmp_mode: CompareMode
    
    @property
    def opcode(self) -> str:
        return "CMP"
    
    def to_pto_as(self) -> str:
        return f"CMP {self.dst}:u1, {self.src0}:{self.src0.element_type.value}, {self.src1}:{self.src1.element_type.value}, {self.cmp_mode.value}"


@dataclass
class SLOAD(ScalarInstruction):
    """Load scalar from memory."""
    dst: ScalarOperand
    mem: MemRefOperand
    idx: IndexOperand
    offset: int = 0
    
    @property
    def opcode(self) -> str:
        return "LOAD"
    
    def to_pto_as(self) -> str:
        return f"LOAD {self.dst}:{self.dst.element_type.value}, {self.mem}:{self.mem.memref_type}, {self.idx}:idx, {self.offset}"


@dataclass
class SSTORE(ScalarInstruction):
    """Store scalar to memory."""
    mem: MemRefOperand
    idx: IndexOperand
    val: ScalarOperand
    offset: int = 0
    
    @property
    def opcode(self) -> str:
        return "STORE"
    
    def to_pto_as(self) -> str:
        return f"STORE {self.mem}:{self.mem.memref_type}, {self.idx}:idx, {self.offset}, {self.val}:{self.val.element_type.value}"


# =============================================================================
# Control Flow Instructions - LOOP Constructs
# =============================================================================

@dataclass
class FOR(ControlFlowInstruction):
    """
    Begin a structured for-loop.
    
    Iteration count can be derived from tile shape dimensions.
    
    When max_range is specified, the dynamic loop is converted to binary expansion:
    - The loop limit is quantized into sum of power-of-2 series
    - Each binary bit serves as a predicate to select loop body of that power-of-2
    - Example: n=45 (binary: 101101) -> 32+8+4+1 iterations
    """
    iv: IndexOperand      # Induction variable
    lb: Union[IndexOperand, ImmediateOperand]  # Lower bound
    ub: Union[IndexOperand, ImmediateOperand]  # Upper bound
    step: Union[IndexOperand, ImmediateOperand] = field(default_factory=lambda: ImmediateOperand(1))
    max_range: Optional[int] = None  # Max range for binary expansion (must be power of 2)
    min_range: Optional[int] = None  # Min range for binary expansion (smallest power of 2 block)
    tile_levels: Optional[Dict[int, int]] = None  # Mapping: block_size -> tile_rows for function variant selection
    
    @property
    def opcode(self) -> str:
        return "FOR"
    
    def to_pto_as(self) -> str:
        base = f"FOR {self.iv}:idx, {self.lb}:idx, {self.ub}:idx, {self.step}:idx"
        if self.max_range is not None:
            base = f"{base} max_range={self.max_range}"
        if self.min_range is not None:
            base = f"{base} min_range={self.min_range}"
        if self.tile_levels is not None:
            # Format: tile_levels={4096:64,2048:64,1024:32,...}
            levels_str = ",".join(f"{k}:{v}" for k, v in sorted(self.tile_levels.items(), reverse=True))
            base = f"{base} tile_levels={{{levels_str}}}"
        return base
    
    def get_binary_bits(self) -> List[int]:
        """Return list of power-of-2 values for binary expansion."""
        if self.max_range is None:
            return []
        bits = []
        n = self.max_range
        while n > 0:
            bits.append(n)
            n //= 2
        return bits  # e.g., [64, 32, 16, 8, 4, 2, 1] for max_range=64
    
    @classmethod
    def from_tile_rows(cls, iv_name: str, tile_type: TileType, step: int = 1) -> "FOR":
        """Create a FOR loop iterating over tile rows."""
        return cls(
            iv=IndexOperand(iv_name),
            lb=ImmediateOperand(0),
            ub=ImmediateOperand(tile_type.shape.rows),
            step=ImmediateOperand(step)
        )
    
    @classmethod
    def from_tile_cols(cls, iv_name: str, tile_type: TileType, step: int = 1) -> "FOR":
        """Create a FOR loop iterating over tile columns."""
        return cls(
            iv=IndexOperand(iv_name),
            lb=ImmediateOperand(0),
            ub=ImmediateOperand(tile_type.shape.cols),
            step=ImmediateOperand(step)
        )


@dataclass
class ENDFOR(ControlFlowInstruction):
    """End a structured for-loop."""
    
    @property
    def opcode(self) -> str:
        return "ENDFOR"
    
    def to_pto_as(self) -> str:
        return "ENDFOR"


@dataclass
class WHILE(ControlFlowInstruction):
    """Begin a structured while-loop."""
    outputs: List[Tuple[str, ElementType]]
    inputs: List[Tuple[str, ElementType]]
    
    @property
    def opcode(self) -> str:
        return "WHILE"
    
    def to_pto_as(self) -> str:
        out_str = ", ".join(f"%{n}:{t.value}" for n, t in self.outputs)
        in_str = ", ".join(f"%{n}:{t.value}" for n, t in self.inputs)
        return f"WHILE ({out_str}), ({in_str})"


@dataclass
class DO(ControlFlowInstruction):
    """Separate condition-region from body-region in a WHILE."""
    
    @property
    def opcode(self) -> str:
        return "DO"
    
    def to_pto_as(self) -> str:
        return "DO"


@dataclass
class ENDWHILE(ControlFlowInstruction):
    """End a structured while-loop."""
    cond: Optional[ScalarOperand] = None
    
    @property
    def opcode(self) -> str:
        return "ENDWHILE"
    
    def to_pto_as(self) -> str:
        if self.cond:
            return f"ENDWHILE {self.cond}:u1"
        return "ENDWHILE"


@dataclass
class IF(ControlFlowInstruction):
    """
    Begin a structured if-region.
    
    When bit_test is specified, the condition is (cond & bit_test) != 0.
    This is used for binary quantization of dynamic loop bounds:
    - Decompose loop count into sum of power-of-2
    - Each bit controls whether that block is executed
    - Enables fixed-count loops instead of dynamic bounds
    """
    cond: ScalarOperand
    outputs: Optional[List[Tuple[str, ElementType]]] = None
    inputs: Optional[List[Tuple[str, ElementType]]] = None
    bit_test: Optional[int] = None  # Power-of-2 value for bit testing
    
    @property
    def opcode(self) -> str:
        return "IF"
    
    def to_pto_as(self) -> str:
        if self.bit_test is not None:
            # Bit test form: IF_BIT cond, bit_value
            return f"IF_BIT {self.cond}:i32, {self.bit_test}"
        if self.outputs and self.inputs:
            out_str = ", ".join(f"%{n}:{t.value}" for n, t in self.outputs)
            in_str = ", ".join(f"%{n}:{t.value}" for n, t in self.inputs)
            return f"IF ({out_str}), {self.cond}:u1, ({in_str})"
        return f"IF {self.cond}:u1"


@dataclass
class ELSE(ControlFlowInstruction):
    """Begin else-region of an IF."""
    
    @property
    def opcode(self) -> str:
        return "ELSE"
    
    def to_pto_as(self) -> str:
        return "ELSE"


@dataclass
class ENDIF(ControlFlowInstruction):
    """End an IF construct."""
    
    @property
    def opcode(self) -> str:
        return "ENDIF"
    
    def to_pto_as(self) -> str:
        return "ENDIF"


@dataclass
class BREAK(ControlFlowInstruction):
    """Break out of nearest enclosing loop."""
    
    @property
    def opcode(self) -> str:
        return "BREAK"
    
    def to_pto_as(self) -> str:
        return "BREAK"


@dataclass
class CONTINUE(ControlFlowInstruction):
    """Continue the nearest enclosing loop."""
    
    @property
    def opcode(self) -> str:
        return "CONTINUE"
    
    def to_pto_as(self) -> str:
        return "CONTINUE"


@dataclass  
class YIELD(ControlFlowInstruction):
    """Yield values from a structured control-flow region."""
    values: List[Tuple[str, ElementType]]
    
    @property
    def opcode(self) -> str:
        return "YIELD"
    
    def to_pto_as(self) -> str:
        vals = ", ".join(f"%{n}:{t.value}" for n, t in self.values)
        return f"YIELD ({vals})"


# =============================================================================
# Function Call Instructions
# =============================================================================

@dataclass
class CALL(ControlFlowInstruction):
    """
    Call a function with arguments.
    
    Arguments are passed as a mapping from parameter names to actual arguments.
    The callee function name is used to resolve the function at link time.
    
    Arguments can be:
    - Simple: "tensor_name" (no offset)
    - With offset: ("tensor_name", "row_offset_expr", "col_offset_expr")
      - Offset expressions can be scalar variable names or integer constants
      - E.g., ("input", "tile_idx", 0) means input[tile_idx * tile_rows + 0]
    
    Examples:
        CALL @sigmoid(%input -> %x, %output -> %y)
        CALL @matmul(%input -> (%x, tile_i, 0), %output -> (%y, tile_i, 0))
    """
    callee: str  # Function name to call
    args: Dict[str, Any] = field(default_factory=dict)  # param -> arg (str or tuple)
    
    @property
    def opcode(self) -> str:
        return "CALL"
    
    def to_pto_as(self) -> str:
        if self.args:
            arg_strs = []
            for param, arg in self.args.items():
                if isinstance(arg, tuple):
                    # Format: (tensor, row_off, col_off)
                    tensor, row_off, col_off = arg
                    arg_strs.append(f"%{param} -> (%{tensor}, {row_off}, {col_off})")
                else:
                    arg_strs.append(f"%{param} -> %{arg}")
            return f"CALL @{self.callee}({', '.join(arg_strs)})"
        return f"CALL @{self.callee}()"


@dataclass
class RETURN(ControlFlowInstruction):
    """
    Return from a function.
    
    Optionally returns values (for functions with return values).
    """
    values: Optional[List[str]] = None  # Optional return values
    
    @property
    def opcode(self) -> str:
        return "RETURN"
    
    def to_pto_as(self) -> str:
        if self.values:
            vals = ", ".join(f"%{v}" for v in self.values)
            return f"RETURN ({vals})"
        return "return"


# =============================================================================
# Loop Constructs for DSL - 1 Level and 2 Level Nested Loops
# =============================================================================

@dataclass
class TileLoop:
    """
    A single-level loop that iterates based on tile dimensions.
    
    The iteration count is derived from the tile shape.
    """
    iv_name: str
    tile_shape: TileShape
    dimension: str  # "rows" or "cols"
    step: int = 1
    body: List[PTOInstruction] = field(default_factory=list)
    
    def get_iteration_count(self) -> int:
        if self.dimension == "rows":
            return self.tile_shape.rows // self.step
        elif self.dimension == "cols":
            return self.tile_shape.cols // self.step
        else:
            raise ValueError(f"Unknown dimension: {self.dimension}")
    
    def to_instructions(self) -> List[PTOInstruction]:
        """Convert loop to PTO instructions."""
        result = []
        bound = self.tile_shape.rows if self.dimension == "rows" else self.tile_shape.cols
        result.append(FOR(
            iv=IndexOperand(self.iv_name),
            lb=ImmediateOperand(0),
            ub=ImmediateOperand(bound),
            step=ImmediateOperand(self.step)
        ))
        result.extend(self.body)
        result.append(ENDFOR())
        return result


@dataclass
class NestedTileLoop:
    """
    A 2-level nested loop for iterating over 2D tile dimensions.
    
    Outer loop iterates over rows, inner loop iterates over columns.
    """
    outer_iv_name: str
    inner_iv_name: str
    tile_shape: TileShape
    outer_step: int = 1
    inner_step: int = 1
    body: List[PTOInstruction] = field(default_factory=list)
    
    def get_total_iterations(self) -> int:
        outer_iters = self.tile_shape.rows // self.outer_step
        inner_iters = self.tile_shape.cols // self.inner_step
        return outer_iters * inner_iters
    
    def to_instructions(self) -> List[PTOInstruction]:
        """Convert nested loop to PTO instructions."""
        result = []
        
        # Outer loop (rows)
        result.append(FOR(
            iv=IndexOperand(self.outer_iv_name),
            lb=ImmediateOperand(0),
            ub=ImmediateOperand(self.tile_shape.rows),
            step=ImmediateOperand(self.outer_step)
        ))
        
        # Inner loop (cols)
        result.append(FOR(
            iv=IndexOperand(self.inner_iv_name),
            lb=ImmediateOperand(0),
            ub=ImmediateOperand(self.tile_shape.cols),
            step=ImmediateOperand(self.inner_step)
        ))
        
        # Body
        result.extend(self.body)
        
        # End inner loop
        result.append(ENDFOR())
        
        # End outer loop
        result.append(ENDFOR())
        
        return result


# =============================================================================
# Instruction Registry - All PTO Instructions
# =============================================================================

TILE_INSTRUCTIONS = {
    # Memory Operations
    "TLOAD": TLOAD,
    "TSTORE": TSTORE,
    "TSTORE_FP": TSTORE_FP,
    "MGATHER": MGATHER,
    "MSCATTER": MSCATTER,
    
    # Element Access
    "GETVAL": GETVAL,
    "SETVAL": SETVAL,
    
    # Unary Operations
    "TABS": TABS,
    "TNEG": TNEG,
    "TNOT": TNOT,
    "TEXP": TEXP,
    "TLOG": TLOG,
    "TSQRT": TSQRT,
    "TRSQRT": TRSQRT,
    "TRECIP": TRECIP,
    "TRELU": TRELU,
    "TLRELU": TLRELU,
    
    # Binary Operations
    "TADD": TADD,
    "TSUB": TSUB,
    "TMUL": TMUL,
    "TDIV": TDIV,
    "TREM": TREM,
    "TMAX": TMAX,
    "TMIN": TMIN,
    "TAND": TAND,
    "TOR": TOR,
    "TXOR": TXOR,
    "TSHL": TSHL,
    "TSHR": TSHR,
    
    # Scalar Operations
    "TADDS": TADDS,
    "TSUBS": TSUBS,
    "TMULS": TMULS,
    "TDIVS": TDIVS,
    "TREMS": TREMS,
    "TMAXS": TMAXS,
    "TMINS": TMINS,
    "TANDS": TANDS,
    "TORS": TORS,
    "TXORS": TXORS,
    
    # Ternary Operations
    "TADDC": TADDC,
    "TADDSC": TADDSC,
    "TSUBC": TSUBC,
    "TSUBSC": TSUBSC,
    
    # Comparison
    "TCMP": TCMP,
    "TCMPS": TCMPS,
    
    # Selection
    "TSEL": TSEL,
    "TSELS": TSELS,
    
    # Matrix Operations
    "TMATMUL": TMATMUL,
    "TMATMUL_ACC": TMATMUL_ACC,
    "TMATMUL_BIAS": TMATMUL_BIAS,
    "TMATMUL_MX": TMATMUL_MX,
    
    # Reduction Operations
    "TROWSUM": TROWSUM,
    "TROWMAX": TROWMAX,
    "TROWMIN": TROWMIN,
    "TCOLSUM": TCOLSUM,
    "TCOLMAX": TCOLMAX,
    "TCOLMIN": TCOLMIN,
    
    # Broadcast/Expand
    "TEXPANDS": TEXPANDS,
    "TROWEXPAND": TROWEXPAND,
    "TCOLEXPAND": TCOLEXPAND,
    "TROWEXPANDMUL": TROWEXPANDMUL,
    "TROWEXPANDDIV": TROWEXPANDDIV,
    "TROWEXPANDSUB": TROWEXPANDSUB,
    
    # Data Movement
    "TTRANS": TTRANS,
    "TRESHAPE": TRESHAPE,
    "TEXTRACT": TEXTRACT,
    "TGATHER": TGATHER,
    "TGATHERB": TGATHERB,
    "TSCATTER": TSCATTER,
    
    # Type Conversion
    "TCVT": TCVT,
    
    # Move Operations
    "TMOV": TMOV,
    "TMOV_FP": TMOV_FP,
    
    # Special Operations
    "TCI": TCI,
    "TPRELU": TPRELU,
    "TSORT32": TSORT32,
    "TMRGSORT": TMRGSORT,
    "TASSIGN": TASSIGN,
    "TSYNC": TSYNC,
}

SCALAR_INSTRUCTIONS = {
    "ADD": SADD,
    "SUB": SSUB,
    "MUL": SMUL,
    "DIV": SDIV,
    "MOV": SMOV,
    "LI": SLI,
    "CMP": SCMP,
    "LOAD": SLOAD,
    "STORE": SSTORE,
}

CONTROL_FLOW_INSTRUCTIONS = {
    "FOR": FOR,
    "ENDFOR": ENDFOR,
    "WHILE": WHILE,
    "DO": DO,
    "ENDWHILE": ENDWHILE,
    "IF": IF,
    "ELSE": ELSE,
    "ENDIF": ENDIF,
    "BREAK": BREAK,
    "CONTINUE": CONTINUE,
    "YIELD": YIELD,
    "CALL": CALL,
    "RETURN": RETURN,
}

ALL_INSTRUCTIONS = {
    **TILE_INSTRUCTIONS,
    **SCALAR_INSTRUCTIONS,
    **CONTROL_FLOW_INSTRUCTIONS,
}


# =============================================================================
# Helper Functions for DSL Building
# =============================================================================

def tile(name: str, rows: int, cols: int, dtype: ElementType = ElementType.F32) -> TileOperand:
    """Create a tile operand with given dimensions."""
    return TileOperand(name, TileType.create(rows, cols, dtype))


def scalar(name: str, dtype: ElementType = ElementType.F32) -> ScalarOperand:
    """Create a scalar operand."""
    return ScalarOperand(name, dtype)


def index(name: str) -> IndexOperand:
    """Create an index operand."""
    return IndexOperand(name)


def memref(name: str, space: MemorySpace = MemorySpace.GM, 
           dtype: ElementType = ElementType.F32,
           shape: Optional[TileShape] = None) -> MemRefOperand:
    """Create a memory reference operand."""
    return MemRefOperand(name, MemRefType(space, dtype, shape))


def imm(value: Union[int, float]) -> ImmediateOperand:
    """Create an immediate operand."""
    return ImmediateOperand(value)


# =============================================================================
# Instruction Metadata for Auto-generating PTOFunctionBuilder Methods
# =============================================================================
# 
# This metadata drives automatic generation of builder methods in pto_compile.py.
# Each entry defines:
#   - builder_name: Method name exposed in PTOFunctionBuilder (e.g., "add" -> .add())
#   - pattern: Operand pattern for automatic method signature generation
#   - doc: Documentation string for the generated method
#
# Supported patterns:
#   - "binary":       (dst, src0, src1) -> Elementwise binary ops (add, sub, mul, etc.)
#   - "unary":        (dst, src) -> Elementwise unary ops (exp, log, sqrt, etc.)
#   - "scalar":       (dst, src, scalar) -> Tile-scalar ops (adds, muls, etc.)
#   - "reduce_row":   (dst, src) -> Row reduction ops (rowsum, rowmax, etc.)
#   - "reduce_col":   (dst, src) -> Column reduction ops (colsum, colmax, etc.)
#   - "broadcast_row": (dst, src0, src1) -> Row broadcast binary ops
#   - "matmul":       (dst, a, b) -> Matrix multiply
#   - "expand_scalar": (dst, scalar) -> Broadcast scalar to tile
#   - "expand_tile":  (dst, src) -> Broadcast tile elements
#
# Instructions NOT in this dict require manual builder methods (e.g., load, store, for_loop)

INSTRUCTION_METADATA: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Elementwise Binary Operations (pattern: binary)
    # =========================================================================
    "TADD": {
        "builder_name": "add",
        "pattern": "binary",
        "doc": "Elementwise add of two tiles: dst = src0 + src1"
    },
    "TSUB": {
        "builder_name": "sub",
        "pattern": "binary",
        "doc": "Elementwise subtract of two tiles: dst = src0 - src1"
    },
    "TMUL": {
        "builder_name": "mul",
        "pattern": "binary",
        "doc": "Elementwise multiply of two tiles: dst = src0 * src1"
    },
    "TDIV": {
        "builder_name": "div",
        "pattern": "binary",
        "doc": "Elementwise divide of two tiles: dst = src0 / src1"
    },
    "TREM": {
        "builder_name": "rem",
        "pattern": "binary",
        "doc": "Elementwise remainder of two tiles: dst = src0 % src1"
    },
    "TMAX": {
        "builder_name": "max",
        "pattern": "binary",
        "doc": "Elementwise maximum of two tiles: dst = max(src0, src1)"
    },
    "TMIN": {
        "builder_name": "min",
        "pattern": "binary",
        "doc": "Elementwise minimum of two tiles: dst = min(src0, src1)"
    },
    "TAND": {
        "builder_name": "bitwise_and",
        "pattern": "binary",
        "doc": "Elementwise bitwise AND of two tiles"
    },
    "TOR": {
        "builder_name": "bitwise_or",
        "pattern": "binary",
        "doc": "Elementwise bitwise OR of two tiles"
    },
    "TXOR": {
        "builder_name": "bitwise_xor",
        "pattern": "binary",
        "doc": "Elementwise bitwise XOR of two tiles"
    },
    "TSHL": {
        "builder_name": "shl",
        "pattern": "binary",
        "doc": "Elementwise left shift: dst = src0 << src1"
    },
    "TSHR": {
        "builder_name": "shr",
        "pattern": "binary",
        "doc": "Elementwise right shift: dst = src0 >> src1"
    },
    
    # =========================================================================
    # Elementwise Unary Operations (pattern: unary)
    # =========================================================================
    "TABS": {
        "builder_name": "abs",
        "pattern": "unary",
        "doc": "Elementwise absolute value: dst = |src|"
    },
    "TNEG": {
        "builder_name": "neg",
        "pattern": "unary",
        "doc": "Elementwise negation: dst = -src"
    },
    "TNOT": {
        "builder_name": "bitwise_not",
        "pattern": "unary",
        "doc": "Elementwise bitwise NOT: dst = ~src"
    },
    "TEXP": {
        "builder_name": "exp",
        "pattern": "unary",
        "doc": "Elementwise exponential: dst = exp(src)"
    },
    "TLOG": {
        "builder_name": "log",
        "pattern": "unary",
        "doc": "Elementwise natural logarithm: dst = log(src)"
    },
    "TSQRT": {
        "builder_name": "sqrt",
        "pattern": "unary",
        "doc": "Elementwise square root: dst = sqrt(src)"
    },
    "TRSQRT": {
        "builder_name": "rsqrt",
        "pattern": "unary",
        "doc": "Elementwise reciprocal square root: dst = 1/sqrt(src)"
    },
    "TRECIP": {
        "builder_name": "recip",
        "pattern": "unary",
        "doc": "Elementwise reciprocal: dst = 1/src"
    },
    "TRELU": {
        "builder_name": "relu",
        "pattern": "unary",
        "doc": "Elementwise ReLU activation: dst = max(0, src)"
    },
    
    # =========================================================================
    # Tile-Scalar Operations (pattern: scalar)
    # =========================================================================
    "TADDS": {
        "builder_name": "adds",
        "pattern": "scalar",
        "doc": "Add scalar to all elements: dst = src + scalar"
    },
    "TSUBS": {
        "builder_name": "subs",
        "pattern": "scalar",
        "doc": "Subtract scalar from all elements: dst = src - scalar"
    },
    "TMULS": {
        "builder_name": "muls",
        "pattern": "scalar",
        "doc": "Multiply all elements by scalar: dst = src * scalar"
    },
    "TDIVS": {
        "builder_name": "divs",
        "pattern": "scalar",
        "doc": "Divide all elements by scalar: dst = src / scalar"
    },
    "TREMS": {
        "builder_name": "rems",
        "pattern": "scalar",
        "doc": "Remainder of all elements by scalar: dst = src % scalar"
    },
    "TMAXS": {
        "builder_name": "maxs",
        "pattern": "scalar",
        "doc": "Maximum of elements with scalar: dst = max(src, scalar)"
    },
    "TMINS": {
        "builder_name": "mins",
        "pattern": "scalar",
        "doc": "Minimum of elements with scalar: dst = min(src, scalar)"
    },
    
    # =========================================================================
    # Row Reduction Operations (pattern: reduce_row)
    # =========================================================================
    "TROWSUM": {
        "builder_name": "rowsum",
        "pattern": "reduce_row",
        "doc": "Row-wise sum reduction: dst[i,0] = sum(src[i,:])"
    },
    "TROWMAX": {
        "builder_name": "rowmax",
        "pattern": "reduce_row",
        "doc": "Row-wise max reduction: dst[i,0] = max(src[i,:])"
    },
    "TROWMIN": {
        "builder_name": "rowmin",
        "pattern": "reduce_row",
        "doc": "Row-wise min reduction: dst[i,0] = min(src[i,:])"
    },
    
    # =========================================================================
    # Column Reduction Operations (pattern: reduce_col)
    # =========================================================================
    "TCOLSUM": {
        "builder_name": "colsum",
        "pattern": "reduce_col",
        "doc": "Column-wise sum reduction: dst[0,j] = sum(src[:,j])"
    },
    "TCOLMAX": {
        "builder_name": "colmax",
        "pattern": "reduce_col",
        "doc": "Column-wise max reduction: dst[0,j] = max(src[:,j])"
    },
    "TCOLMIN": {
        "builder_name": "colmin",
        "pattern": "reduce_col",
        "doc": "Column-wise min reduction: dst[0,j] = min(src[:,j])"
    },
    
    # =========================================================================
    # Row Broadcast Operations (pattern: broadcast_row)
    # =========================================================================
    "TROWEXPANDSUB": {
        "builder_name": "rowexpandsub",
        "pattern": "broadcast_row",
        "doc": "Row-wise broadcast subtract: dst = src0 - broadcast(src1)"
    },
    "TROWEXPANDDIV": {
        "builder_name": "rowexpanddiv",
        "pattern": "broadcast_row",
        "doc": "Row-wise broadcast divide: dst = src0 / broadcast(src1)"
    },
    "TROWEXPANDMUL": {
        "builder_name": "rowexpandmul",
        "pattern": "broadcast_row",
        "doc": "Row-wise broadcast multiply: dst = src0 * broadcast(src1)"
    },
    
    # =========================================================================
    # Matrix Operations (pattern: matmul)
    # =========================================================================
    "TMATMUL": {
        "builder_name": "matmul",
        "pattern": "matmul",
        "doc": "Matrix multiplication: dst = a @ b"
    },
    "TMATMUL_ACC": {
        "builder_name": "matmul_acc",
        "pattern": "matmul_acc",
        "doc": "Matrix multiply with accumulator: dst = acc + a @ b"
    },
    
    # =========================================================================
    # Broadcast Operations (pattern: expand_scalar, expand_tile)
    # =========================================================================
    "TEXPANDS": {
        "builder_name": "expands",
        "pattern": "expand_scalar",
        "doc": "Broadcast scalar to all elements: dst[i,j] = scalar"
    },
    "TROWEXPAND": {
        "builder_name": "rowexpand",
        "pattern": "expand_tile",
        "doc": "Broadcast first column across rows: dst[i,j] = src[i,0]"
    },
    "TCOLEXPAND": {
        "builder_name": "colexpand",
        "pattern": "expand_tile",
        "doc": "Broadcast first row across columns: dst[i,j] = src[0,j]"
    },
    
    # =========================================================================
    # Data Movement Operations
    # =========================================================================
    "TTRANS": {
        "builder_name": "transpose",
        "pattern": "unary",
        "doc": "Transpose tile: dst = src.T"
    },
    "TMOV": {
        "builder_name": "mov",
        "pattern": "unary",
        "doc": "Copy tile: dst = src"
    },
}


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Define a simple matrix multiply with nested loops
    
    # Define tile types
    tile_a_type = TileType.create(64, 64, ElementType.F16)
    tile_b_type = TileType.create(64, 64, ElementType.F16)
    tile_c_type = TileType.create(64, 64, ElementType.F32)
    
    # Create operands
    a = tile("a", 64, 64, ElementType.F16)
    b = tile("b", 64, 64, ElementType.F16)
    c = tile("c", 64, 64, ElementType.F32)
    mem_a = memref("mem_a", MemorySpace.GM, ElementType.F16)
    mem_b = memref("mem_b", MemorySpace.GM, ElementType.F16)
    mem_c = memref("mem_c", MemorySpace.GM, ElementType.F32)
    
    # Build a simple program with nested loop
    shape = TileShape(4, 4)
    
    nested_loop = NestedTileLoop(
        outer_iv_name="i",
        inner_iv_name="j",
        tile_shape=shape,
        body=[
            # Example body: load, compute, store
        ]
    )
    
    print("PTO ISA Definition Module")
    print("=" * 50)
    print(f"Total Tile Instructions: {len(TILE_INSTRUCTIONS)}")
    print(f"Total Scalar Instructions: {len(SCALAR_INSTRUCTIONS)}")
    print(f"Total Control Flow Instructions: {len(CONTROL_FLOW_INSTRUCTIONS)}")
    print(f"Total Instructions: {len(ALL_INSTRUCTIONS)}")
    print()
    
    # Print example instruction
    load_instr = TLOAD(
        dst=a,
        src_mem=mem_a,
        row_offset=imm(0),
        col_offset=imm(0)
    )
    print(f"Example TLOAD: {load_instr.to_pto_as()}")
    
    matmul_instr = TMATMUL(dst=c, a=a, b=b)
    print(f"Example TMATMUL: {matmul_instr.to_pto_as()}")
    
    # Print nested loop structure
    print()
    print("Nested Loop Example:")
    for instr in nested_loop.to_instructions():
        print(f"  {instr.to_pto_as()}")
