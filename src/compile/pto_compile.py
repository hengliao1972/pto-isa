"""
PTO ISA Compiler - Modular Entry Point

This module provides the main entry point for the PTO compiler, integrating:
- Common infrastructure (pto_compile_common.py)
- ARM64 code generation (pto_codegen_arm64.py)
- CUDA code generation (pto_codegen_cuda.py)
- Ascend code generation (pto_codegen_ascend.py)

Directory Structure:
===================
    src/
        isa_definition/     - ISA type definitions and instruction specs
        compile/            - Compiler modules (this file)
        runtime/            - C/H runtime files
    examples/
        llama/              - LLaMA example
        softmax/            - Softmax example

Usage:
======
    # Import everything from the modular compiler
    import sys
    sys.path.insert(0, 'path/to/PTO_ISA_Compiler/src')
    from compile.pto_compile import *
    
    # Build a program
    program = (PTOFunctionBuilder("my_func")
        .tile("x", 8, 8)
        .memref("input", MemorySpace.GM)
        ...
        .build())
    
    # Generate code for different platforms
    arm64_code = generate_arm64_code(program)
    cuda_code = generate_cuda_code(program)
    ascend_code = generate_ascend_code(program)
"""

import os
import sys

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# =============================================================================
# Import Common Infrastructure
# =============================================================================

from compile.pto_compile_common import (
    # Error types
    CompilerError, ParseError, TypeError, ValidationError,
    
    # Symbol table
    Symbol, SymbolTable,
    
    # Program representation
    PTOProgram, PTOModule,
    
    # Loop fusion
    OpCategory, FusionTileShape, FusableOp, FusedLoop, FusionBarrier,
    LoopFusionOptimizer, get_category, is_fusable, is_fusion_barrier,
    
    # Mock instructions
    MockTileInfo, MockInstruction, convert_program_to_mock_instructions,
    
    # Buffer analysis
    TileBufferInfo, TileBufferAnalyzer,
    
    # Type checker and optimizer
    TypeChecker, Optimizer, CodeGenerator, PTOCompiler,
    
    # Utilities
    apply_binary_expansion, apply_loop_replay_optimization, get_loop_replay_header,
)

# =============================================================================
# Import Platform-Specific Code Generators
# =============================================================================

from compile.pto_codegen_arm64 import (
    OrchestrationContext,
    gen_arm64_barrier_op,
    gen_task_scheduling_code,
    ARM64FusedCodeGenerator,
    ARM64CodeGenerator,
)

from compile.pto_codegen_cuda import (
    gen_cuda_barrier_op,
    gen_cuda_single_op,
    CUDAFusedCodeGenerator,
    CUDACodeGenerator,
)

from compile.pto_codegen_ascend import (
    gen_ascend_barrier_op,
    gen_ascend_single_op,
    AscendFusedCodeGenerator,
    AscendCodeGenerator,
)

from compile.pto_codegen_ascend_a2a3_sim import (
    AscendA2A3SimCodeGenerator,
    generate_ascend_a2a3_sim_code,
    ASCEND_A2A3_CYCLE_COSTS,
    get_cycle_cost,
    is_cube_op,
)

# =============================================================================
# Import ISA Definitions (for backward compatibility)
# =============================================================================

from isa_definition.pto_isa_definition import (
    # Types
    ElementType, MemorySpace, CompareMode, RoundMode, TMovMode,
    TileShape, TileType, MemRefType, EventType,
    
    # Operands
    TileOperand, ScalarOperand, MemRefOperand, IndexOperand, ImmediateOperand, Operand,
    
    # Base classes
    PTOInstruction, TileInstruction, ScalarInstruction, ControlFlowInstruction,
    
    # All instructions
    ALL_INSTRUCTIONS, TILE_INSTRUCTIONS, SCALAR_INSTRUCTIONS, CONTROL_FLOW_INSTRUCTIONS,
    
    # Loop constructs
    TileLoop, NestedTileLoop, FOR, ENDFOR, WHILE, DO, ENDWHILE, IF, ELSE, ENDIF,
    
    # Function call instructions
    CALL, RETURN,
    
    # Memory operations
    TLOAD, TSTORE,
    
    # Scalar instructions
    SADD, SSUB, SMUL, SDIV, SMOV, SLI, SCMP,
    
    # Helper functions
    tile, scalar, index, memref, imm,
    
    # Instruction metadata
    INSTRUCTION_METADATA,
    
    # Type maps and headers
    ARM64_TYPE_MAP, CUDA_TYPE_MAP, ASCEND_TYPE_MAP,
    arm64_generate_header, cuda_generate_header, ascend_generate_header,
)

# Import for auto-generation
import isa_definition.pto_isa_definition as _pto_isa

# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These maintain compatibility with code using the old function names
_gen_arm64_barrier_op = gen_arm64_barrier_op
_gen_task_scheduling_code = gen_task_scheduling_code
_gen_cuda_barrier_op = gen_cuda_barrier_op
_gen_cuda_single_op = gen_cuda_single_op
_gen_ascend_barrier_op = gen_ascend_barrier_op
_gen_ascend_single_op = gen_ascend_single_op

# Fused code generator (backward compat)
FusedCodeGenerator = ARM64FusedCodeGenerator

# =============================================================================
# PTOFunctionBuilder - Fluent Interface for Building PTO Programs
# =============================================================================

from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field


class PTOFunctionBuilder:
    """
    Fluent interface for building PTO programs.
    
    Example:
        program = (PTOFunctionBuilder("matmul")
            .tile("a", 64, 64, ElementType.F16)
            .tile("b", 64, 64, ElementType.F16)
            .tile("c", 64, 64, ElementType.F32)
            .memref("mem_a", MemorySpace.GM, ElementType.F16)
            .load("a", "mem_a", 0, 0)
            .matmul("c", "a", "b")
            .store("c", "mem_c", 0, 0)
            .build())
    """
    
    def __init__(self, name: str = "main", module: Optional[PTOModule] = None):
        self.program = PTOProgram(name=name)
        self.symbol_table = SymbolTable()
        self._loop_stack: List[List[PTOInstruction]] = []
        self._module = module
    
    def _get_tile(self, name: str) -> TileOperand:
        """Get a tile operand by name."""
        if name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{name}' not declared")
        return TileOperand(name, self.program.tile_declarations[name])
    
    def _get_scalar(self, name: str) -> ScalarOperand:
        """Get a scalar operand by name."""
        if name not in self.program.scalar_declarations:
            raise ValidationError(f"Scalar '{name}' not declared")
        return ScalarOperand(name, self.program.scalar_declarations[name])
    
    def _get_memref(self, name: str) -> MemRefOperand:
        """Get a memref operand by name."""
        if name not in self.program.memref_declarations:
            raise ValidationError(f"MemRef '{name}' not declared")
        return MemRefOperand(name, self.program.memref_declarations[name])
    
    def _add_instr(self, instr: PTOInstruction):
        """Add instruction to current context."""
        if self._loop_stack:
            self._loop_stack[-1].append(instr)
        else:
            self.program.add_instruction(instr)
    
    def _make_scalar_operand(self, value):
        """Create a scalar operand from a value."""
        return ScalarOperand(str(value), ElementType.F32)
    
    # Declaration methods
    def tile(self, name: str, rows: int, cols: int, 
             dtype: ElementType = ElementType.F32) -> "PTOFunctionBuilder":
        """Declare a tile variable."""
        self.program.add_tile(name, rows, cols, dtype)
        self.symbol_table.define(name, Symbol(name, "tile", TileType.create(rows, cols, dtype)))
        return self
    
    def scalar(self, name: str, dtype: ElementType = ElementType.F32) -> "PTOFunctionBuilder":
        """Declare a scalar variable."""
        self.program.add_scalar(name, dtype)
        self.symbol_table.define(name, Symbol(name, "scalar", dtype))
        return self
    
    def scalar_li(self, name: str, value: float, dtype: ElementType = ElementType.F32) -> "PTOFunctionBuilder":
        """
        Declare a scalar variable and load an immediate value.
        
        This is a convenience method combining scalar declaration and immediate assignment.
        
        Args:
            name: Name of the scalar variable
            value: Immediate value to assign
            dtype: Data type of the scalar (default: F32)
        
        Example:
            .scalar_li("scale", 0.125)  # Creates: float scale = 0.125f;
        """
        # Declare the scalar if not already declared
        if name not in self.program.scalar_declarations:
            self.program.add_scalar(name, dtype)
            self.symbol_table.define(name, Symbol(name, "scalar", dtype))
        
        # Add SLI instruction to load the immediate value
        self._add_instr(SLI(
            dst=ScalarOperand(name, dtype),
            imm=ImmediateOperand(value)
        ))
        
        return self
    
    def scalar_cmp(self, dst: str, src0: str, src1: str, cmp_mode: 'CompareMode') -> "PTOFunctionBuilder":
        """
        Compare two scalars and store the result.
        
        Args:
            dst: Destination scalar (boolean result)
            src0: First scalar operand
            src1: Second scalar operand
            cmp_mode: Comparison mode (CompareMode.GT, LT, EQ, etc.)
        
        Example:
            .scalar_cmp("result", "a", "b", CompareMode.GT)  # result = (a > b)
        """
        # Ensure dst is declared
        if dst not in self.program.scalar_declarations:
            self.program.add_scalar(dst, ElementType.U1)
            self.symbol_table.define(dst, Symbol(dst, "scalar", ElementType.U1))
        
        # Get src0 and src1 types
        src0_type = self.program.scalar_declarations.get(src0, ElementType.I32)
        src1_type = self.program.scalar_declarations.get(src1, ElementType.I32)
        
        self._add_instr(SCMP(
            dst=ScalarOperand(dst, ElementType.U1),
            src0=ScalarOperand(src0, src0_type),
            src1=ScalarOperand(src1, src1_type),
            cmp_mode=cmp_mode
        ))
        
        return self
    
    def memref(self, name: str, space: MemorySpace = MemorySpace.GM,
               dtype: ElementType = ElementType.F32,
               shape: Optional[Tuple[int, int]] = None) -> "PTOFunctionBuilder":
        """Declare a memory reference."""
        tile_shape = TileShape(*shape) if shape else None
        self.program.add_memref(name, space, dtype, tile_shape)
        self.symbol_table.define(name, Symbol(name, "memref", MemRefType(space, dtype, tile_shape)))
        return self
    
    # Function properties
    def in_core(self) -> "PTOFunctionBuilder":
        """Mark as InCore function (default)."""
        self.program.is_in_core = True
        return self
    
    def not_in_core(self) -> "PTOFunctionBuilder":
        """Mark as orchestration function."""
        self.program.is_in_core = False
        return self
    
    def cube(self, is_cube: bool = True) -> "PTOFunctionBuilder":
        """Mark as requiring cube unit (for A2A3)."""
        self.program.is_cube = is_cube
        return self
    
    def import_func(self, func_name: str) -> "PTOFunctionBuilder":
        """Import an external function."""
        self.program.add_import(func_name)
        return self
    
    # Memory operations
    def load(self, dst: str, src_mem: str, 
             row: Union[int, str] = 0, col: Union[int, str] = 0) -> "PTOFunctionBuilder":
        """Load data from memory into a tile."""
        row_op = ImmediateOperand(row) if isinstance(row, int) else IndexOperand(row)
        col_op = ImmediateOperand(col) if isinstance(col, int) else IndexOperand(col)
        self._add_instr(TLOAD(
            dst=self._get_tile(dst),
            src_mem=self._get_memref(src_mem),
            row_offset=row_op,
            col_offset=col_op
        ))
        return self
    
    def store(self, src: str, dst_mem: str, 
              row: Union[int, str] = 0, col: Union[int, str] = 0) -> "PTOFunctionBuilder":
        """Store data from a tile into memory."""
        row_op = ImmediateOperand(row) if isinstance(row, int) else IndexOperand(row)
        col_op = ImmediateOperand(col) if isinstance(col, int) else IndexOperand(col)
        self._add_instr(TSTORE(
            src=self._get_tile(src),
            dst_mem=self._get_memref(dst_mem),
            row_offset=row_op,
            col_offset=col_op
        ))
        return self
    
    # Control flow
    def for_loop(self, iv: str, start: Union[int, str], end: Union[int, str], step: int = 1,
                 max_range: Optional[int] = None, min_range: Optional[int] = None,
                 tile_levels: Optional[Dict[int, int]] = None) -> "PTOFunctionBuilder":
        """Start a for loop."""
        # Handle start (lower bound)
        if isinstance(start, int):
            lb = ImmediateOperand(start)
        else:
            lb = IndexOperand(start)
        
        # Handle end (upper bound)
        if isinstance(end, int):
            ub = ImmediateOperand(end)
        else:
            ub = IndexOperand(end)
        
        for_instr = FOR(
            iv=IndexOperand(iv),
            lb=lb,
            ub=ub,
            step=ImmediateOperand(step)
        )
        if max_range is not None:
            for_instr.max_range = max_range
        if min_range is not None:
            for_instr.min_range = min_range
        if tile_levels is not None:
            for_instr.tile_levels = tile_levels
        
        self._add_instr(for_instr)
        self._loop_stack.append([])
        return self
    
    def end_for(self) -> "PTOFunctionBuilder":
        """End a for loop."""
        if self._loop_stack:
            body = self._loop_stack.pop()
            # Add body instructions to the parent context (parent loop or program)
            # Using _add_instr ensures proper nesting for nested loops
            for instr in body:
                self._add_instr(instr)
        self._add_instr(ENDFOR())
        return self
    
    def if_stmt(self, condition: str, bit_test: Optional[int] = None) -> "PTOFunctionBuilder":
        """Start an if statement."""
        # Get scalar type if the condition is a declared scalar
        cond_type = self.program.scalar_declarations.get(condition, ElementType.U1)
        self._add_instr(IF(
            cond=ScalarOperand(condition, cond_type),
            bit_test=bit_test
        ))
        return self
    
    # Alias for if_stmt
    def if_then(self, condition: str, bit_test: Optional[int] = None) -> "PTOFunctionBuilder":
        """Start an if statement (alias for if_stmt)."""
        return self.if_stmt(condition, bit_test)
    
    def else_stmt(self) -> "PTOFunctionBuilder":
        """Add else clause."""
        self._add_instr(ELSE())
        return self
    
    # Alias for else_stmt
    def else_branch(self) -> "PTOFunctionBuilder":
        """Add else clause (alias for else_stmt)."""
        return self.else_stmt()
    
    def end_if(self) -> "PTOFunctionBuilder":
        """End if statement."""
        self._add_instr(ENDIF())
        return self
    
    # Alias for end_if
    def end_if_then(self) -> "PTOFunctionBuilder":
        """End if statement (alias for end_if)."""
        return self.end_if()
    
    # Another alias for end_if
    def endif(self) -> "PTOFunctionBuilder":
        """End if statement (alias for end_if)."""
        return self.end_if()
    
    # Function calls
    def call(self, callee: str, args: Optional[Dict[str, Any]] = None) -> "PTOFunctionBuilder":
        """Call another function."""
        self._add_instr(CALL(callee=callee, args=args or {}))
        return self
    
    def ret(self, values: Optional[List[str]] = None) -> "PTOFunctionBuilder":
        """Add a return statement."""
        self._add_instr(RETURN(values=values))
        return self
    
    def build(self) -> PTOProgram:
        """Build and return the program."""
        if self._loop_stack:
            raise ValidationError("Unclosed loop constructs")
        return self.program


# =============================================================================
# Auto-generate PTOFunctionBuilder Methods
# =============================================================================

def _auto_generate_builder_methods():
    """Dynamically add methods to PTOFunctionBuilder."""
    
    def _make_binary_method(instr_class, doc):
        def method(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
            self._add_instr(instr_class(
                dst=self._get_tile(dst),
                src0=self._get_tile(src0),
                src1=self._get_tile(src1)
            ))
            return self
        method.__doc__ = doc
        return method
    
    def _make_unary_method(instr_class, doc):
        def method(self, dst: str, src: str) -> "PTOFunctionBuilder":
            self._add_instr(instr_class(
                dst=self._get_tile(dst),
                src=self._get_tile(src)
            ))
            return self
        method.__doc__ = doc
        return method
    
    def _make_scalar_method(instr_class, doc):
        def method(self, dst: str, src: str, scalar_val: float) -> "PTOFunctionBuilder":
            self._add_instr(instr_class(
                dst=self._get_tile(dst),
                src=self._get_tile(src),
                scalar=self._make_scalar_operand(scalar_val)
            ))
            return self
        method.__doc__ = doc
        return method
    
    def _make_reduce_method(instr_class, doc):
        def method(self, dst: str, src: str) -> "PTOFunctionBuilder":
            self._add_instr(instr_class(
                dst=self._get_tile(dst),
                src=self._get_tile(src)
            ))
            return self
        method.__doc__ = doc
        return method
    
    def _make_matmul_method(instr_class, doc):
        def method(self, dst: str, a: str, b: str) -> "PTOFunctionBuilder":
            # Matmul requires cube unit - auto-set is_cube flag
            self.program.is_cube = True
            self._add_instr(instr_class(
                dst=self._get_tile(dst),
                a=self._get_tile(a),
                b=self._get_tile(b)
            ))
            return self
        method.__doc__ = doc
        return method
    
    pattern_factories = {
        "binary": _make_binary_method,
        "unary": _make_unary_method,
        "scalar": _make_scalar_method,
        "reduce_row": _make_reduce_method,
        "reduce_col": _make_reduce_method,
        "broadcast_row": _make_binary_method,
        "matmul": _make_matmul_method,
    }
    
    generated = 0
    for instr_name, meta in INSTRUCTION_METADATA.items():
        builder_name = meta["builder_name"]
        pattern = meta["pattern"]
        doc = meta.get("doc", f"Auto-generated method for {instr_name}")
        
        if hasattr(PTOFunctionBuilder, builder_name):
            continue
        
        instr_class = getattr(_pto_isa, instr_name, None)
        if instr_class is None:
            continue
        
        factory = pattern_factories.get(pattern)
        if factory is None:
            continue
        
        method = factory(instr_class, doc)
        setattr(PTOFunctionBuilder, builder_name, method)
        generated += 1
    
    return generated

_AUTO_GENERATED = _auto_generate_builder_methods()


# =============================================================================
# Multi-Backend Code Generator
# =============================================================================

import os

BACKENDS = {
    "arm64": {
        "name": "ARM64 NEON",
        "suffix": "_arm64",
        "extension": ".c",
        "header_func": arm64_generate_header,
        "type_map": ARM64_TYPE_MAP,
    },
    "cuda": {
        "name": "NVIDIA CUDA",
        "suffix": "_cuda",
        "extension": ".cu",
        "header_func": cuda_generate_header,
        "type_map": CUDA_TYPE_MAP,
    },
    "ascend_a2a3": {
        "name": "Huawei Ascend A2/A3",
        "suffix": "_ascend_a2a3",
        "extension": ".c",
        "header_func": lambda: "// Auto-generated Ascend A2/A3 code\n",
        "type_map": ARM64_TYPE_MAP,
    },
    "ascend_a5": {
        "name": "Huawei Ascend A5",
        "suffix": "_ascend_a5",
        "extension": ".cpp",
        "header_func": lambda: "// Auto-generated Ascend A5 code\n",
        "type_map": {},
    },
    "ascend_a2a3_sim": {
        "name": "Ascend A2/A3 Simulator",
        "suffix": "_ascend_a2a3_sim",
        "extension": ".c",
        "header_func": lambda: "// Ascend A2/A3 Cycle Simulator\n",
        "type_map": ARM64_TYPE_MAP,
    },
}


class PTOModuleCompiler:
    """
    Compiles a PTOModule to PTO Assembly text format.
    
    This generates human-readable PTO assembly code that can be:
    - Used for debugging and visualization
    - Parsed back to reconstruct the program
    - Used as an intermediate representation
    """
    
    def __init__(self, inline_in_core: bool = False, eliminate_redundant_mem: bool = False):
        """
        Initialize the module compiler.
        
        Args:
            inline_in_core: If True, inline InCore function calls
            eliminate_redundant_mem: If True, eliminate redundant memory operations
        """
        self.inline_in_core = inline_in_core
        self.eliminate_redundant_mem = eliminate_redundant_mem
    
    def compile_function(self, program: PTOProgram) -> str:
        """Compile a single function to PTO assembly."""
        # For InCore kernels, emit the new PTO-AS surface syntax accepted by `ptoas`.
        #
        # This is the preferred `.pto` format for the end-to-end toolchain:
        #   .pto -> ptoas -> CCE C++ -> ccec -> .o/.bin -> runtime
        #
        # Orchestration functions are still emitted in the legacy human-readable assembly format
        # (useful for debugging and round-tripping via `pto_parser.py`).
        if getattr(program, "is_in_core", True):
            from compile.pto_to_ptoas import export_program_to_ptoas

            return export_program_to_ptoas(program=program)

        return self._compile_function_legacy(program)

    def _compile_function_legacy(self, program: PTOProgram) -> str:
        """Compile a single function to the legacy PTO assembly format (for debugging / pto_parser.py)."""
        lines = []
        
        # Function header
        func_type = "InCore" if program.is_in_core else "Orchestration"
        cube_marker = " [Cube]" if program.is_cube else ""
        lines.append(f"// {func_type} Function{cube_marker}")
        lines.append(f".function {program.name}")
        
        # Tile declarations
        if program.tile_declarations:
            lines.append("  // Tile declarations")
            for name, tile_type in program.tile_declarations.items():
                lines.append(f"  .tile {name}: {tile_type.element_type.value}[{tile_type.shape.rows}, {tile_type.shape.cols}]")
        
        # Scalar declarations
        if program.scalar_declarations:
            lines.append("  // Scalar declarations")
            for name, dtype in program.scalar_declarations.items():
                lines.append(f"  .scalar {name}: {dtype.value}")
        
        # MemRef declarations
        if program.memref_declarations:
            lines.append("  // Memory references")
            for name, memref_type in program.memref_declarations.items():
                shape_str = ""
                if memref_type.shape:
                    shape_str = f"[{memref_type.shape.rows}, {memref_type.shape.cols}]"
                lines.append(f"  .memref {name}: {memref_type.memory_space.value} {memref_type.element_type.value}{shape_str}")
        
        # Instructions
        if program.instructions:
            lines.append("  // Instructions")
            indent_level = 1
            for instr in program.instructions:
                # Adjust indent for control flow
                if hasattr(instr, 'opcode'):
                    if instr.opcode in ('ENDFOR', 'ENDWHILE', 'ENDIF', 'ELSE'):
                        indent_level = max(1, indent_level - 1)
                
                indent = "  " * indent_level
                if hasattr(instr, 'to_pto_as'):
                    lines.append(f"{indent}{instr.to_pto_as()}")
                else:
                    lines.append(f"{indent}// {type(instr).__name__}")
                
                # Increase indent after control flow start
                if hasattr(instr, 'opcode'):
                    if instr.opcode in ('FOR', 'WHILE', 'DO', 'IF', 'ELSE'):
                        indent_level += 1
        
        lines.append(f".end {program.name}")
        lines.append("")
        
        return "\n".join(lines)
    
    def compile(self, module: PTOModule) -> str:
        """Compile a complete module to PTO assembly."""
        lines = []
        
        # Module header
        lines.append(f"// PTO Module: {module.name}")
        lines.append(f"// Entry: {module.entry_function}")
        lines.append(f"// Total functions: {len(module.functions)}")
        lines.append("")
        
        # Compile each function
        for func_name, program in module.functions.items():
            lines.append(self._compile_function_legacy(program))
        
        return "\n".join(lines)


class MultiBackendCodeGenerator:
    """
    Unified multi-backend code generator for PTO programs.
    """
    
    def __init__(self, enable_fusion: bool = True, analyze_buffers: bool = True,
                 module: Optional['PTOModule'] = None):
        self.enable_fusion = enable_fusion
        self.analyze_buffers = analyze_buffers
        self.module = module
    
    def generate_arm64(self, program: PTOProgram) -> str:
        """Generate ARM64 code."""
        gen = ARM64CodeGenerator(
            enable_fusion=self.enable_fusion,
            analyze_buffers=self.analyze_buffers,
            module=self.module
        )
        return gen.generate(program)
    
    def generate_cuda(self, program: PTOProgram) -> str:
        """Generate CUDA code."""
        gen = CUDACodeGenerator(
            enable_fusion=self.enable_fusion,
            analyze_buffers=self.analyze_buffers,
            module=self.module
        )
        return gen.generate(program)
    
    def generate_ascend(self, program: PTOProgram, target: str = "a2a3") -> str:
        """Generate Ascend code."""
        gen = AscendCodeGenerator(
            enable_fusion=self.enable_fusion,
            analyze_buffers=self.analyze_buffers,
            module=self.module,
            target=target
        )
        return gen.generate(program)

    def generate_ptoas(self, program: PTOProgram, *, block_dim: int = 1, kernel_name: str = "pto_kernel") -> str:
        """
        Generate new-format PTO-AS text compatible with the `ptoas` toolchain.

        Note: this is currently a small fast-path exporter intended for tiny GEMM-style programs
        (straight-line TLOAD/TMATMUL/TSTORE). It will raise NotImplementedError for unsupported ops.
        """
        from compile.pto_to_ptoas import export_program_to_ptoas_gemm16

        return export_program_to_ptoas_gemm16(program=program, block_dim=int(block_dim), kernel_name=str(kernel_name))
    
    def generate_ascend_a2a3(self, program: PTOProgram) -> str:
        """Generate Ascend A2/A3 code (same format as simulator, for real hardware)."""
        gen = AscendA2A3SimCodeGenerator(
            enable_fusion=self.enable_fusion,
            analyze_buffers=self.analyze_buffers,
            module=self.module,
            target_mode="hardware"
        )
        return gen.generate(program)
    
    def generate_ascend_a5(self, program: PTOProgram) -> str:
        """Generate Ascend A5 code (convenience method)."""
        return self.generate_ascend(program, target="a5")
    
    def generate_ascend_a2a3_sim(self, program: PTOProgram) -> str:
        """Generate Ascend A2/A3 simulation code."""
        gen = AscendA2A3SimCodeGenerator(
            enable_fusion=self.enable_fusion,
            analyze_buffers=self.analyze_buffers,
            module=self.module
        )
        return gen.generate(program)
    
    def compile_and_run_orchestration(self, program: PTOProgram, output_dir: str,
                                       extra_args: Optional[Dict[str, Any]] = None,
                                       compiler: str = "gcc",
                                       run_timeout: int = 60) -> Optional[str]:
        """
        Generate, compile, and run orchestration code to produce task graph dump.
        
        Args:
            program: The orchestration program to compile
            output_dir: Directory for output files
            extra_args: Extra runtime arguments (e.g., seq_len, num_tiles)
            compiler: Compiler to use (default: gcc)
            run_timeout: Timeout for execution in seconds
            
        Returns:
            Path to the task dump file if successful, None otherwise
        """
        import subprocess
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate orchestration code
        orch_code = self.generate_arm64(program)
        orch_file = os.path.join(output_dir, f"{program.name}_orchestration.c")
        
        with open(orch_file, 'w') as f:
            f.write(orch_code)
        
        # Compile
        exe_file = os.path.join(output_dir, f"{program.name}_orchestration")
        
        # Get runtime directory
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _src_dir = os.path.dirname(_current_dir)
        runtime_dir = os.path.join(_src_dir, "runtime")
        
        compile_flags = ["-O2", "-std=c11", "-DPTO_BINARY_EXPANSION", 
                         "-DPTO_TASK_DUMP"]
        
        compile_cmd = [
            compiler,
            *compile_flags,
            f"-I{runtime_dir}",
            "-o", exe_file,
            orch_file,
            "-lpthread"
        ]
        
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  Compilation failed: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        
        # Run and capture output
        try:
            result = subprocess.run(
                [exe_file],
                capture_output=True,
                timeout=run_timeout,
                cwd=output_dir
            )
            
            # Check for task dump file
            dump_file = os.path.join(output_dir, f"{program.name}_task_graph.txt")
            if os.path.exists(dump_file):
                return dump_file
            
            # Try alternative naming
            for f in os.listdir(output_dir):
                if 'task_graph' in f and f.endswith('.txt'):
                    return os.path.join(output_dir, f)
            
            return None
            
        except subprocess.TimeoutExpired:
            print(f"  Execution timed out after {run_timeout}s")
            return None
        except Exception as e:
            print(f"  Execution failed: {e}")
            return None
    
    def generate_all(self, program: PTOProgram, output_prefix: str,
                     output_base_dir: str = ".") -> Dict[str, str]:
        """Generate code for all backends."""
        results = {}
        
        # ARM64
        arm64_code = self.generate_arm64(program)
        arm64_dir = os.path.join(output_base_dir, "output_arm64")
        os.makedirs(arm64_dir, exist_ok=True)
        arm64_path = os.path.join(arm64_dir, f"{output_prefix}.c")
        with open(arm64_path, 'w') as f:
            f.write(arm64_code)
        results["arm64"] = arm64_path
        
        # CUDA
        cuda_code = self.generate_cuda(program)
        cuda_dir = os.path.join(output_base_dir, "output_cuda")
        os.makedirs(cuda_dir, exist_ok=True)
        cuda_path = os.path.join(cuda_dir, f"{output_prefix}.cu")
        with open(cuda_path, 'w') as f:
            f.write(cuda_code)
        results["cuda"] = cuda_path
        
        # Ascend
        ascend_code = self.generate_ascend(program)
        ascend_dir = os.path.join(output_base_dir, "output_ascend")
        os.makedirs(ascend_dir, exist_ok=True)
        ascend_path = os.path.join(ascend_dir, f"{output_prefix}.cpp")
        with open(ascend_path, 'w') as f:
            f.write(ascend_code)
        results["ascend"] = ascend_path
        
        return results


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================

def generate_all_backends(program: PTOProgram, output_prefix: str,
                          output_base_dir: str = ".",
                          enable_fusion: bool = True) -> Dict[str, str]:
    """Generate code for all backends."""
    gen = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return gen.generate_all(program, output_prefix, output_base_dir)


def generate_arm64_code(program: PTOProgram, enable_fusion: bool = True) -> str:
    """Generate ARM64 code."""
    gen = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return gen.generate_arm64(program)


def generate_cuda_code(program: PTOProgram, enable_fusion: bool = True) -> str:
    """Generate CUDA code."""
    gen = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return gen.generate_cuda(program)


def generate_ascend_code(program: PTOProgram, enable_fusion: bool = True) -> str:
    """Generate Ascend code."""
    gen = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return gen.generate_ascend(program)


# =============================================================================
# Example and CLI
# =============================================================================

def run_demo():
    """Run demo examples."""
    print("=" * 60)
    print("PTO Compiler - Modular Architecture Demo")
    print("=" * 60)
    
    program = (PTOFunctionBuilder("example")
        .tile("a", 8, 8, ElementType.F32)
        .tile("b", 8, 8, ElementType.F32)
        .tile("c", 8, 8, ElementType.F32)
        .memref("mem_a", MemorySpace.GM, ElementType.F32)
        .memref("mem_b", MemorySpace.GM, ElementType.F32)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        .load("a", "mem_a", 0, 0)
        .load("b", "mem_b", 0, 0)
        .add("c", "a", "b")
        .store("c", "mem_c", 0, 0)
        .build())
    
    print("\nGenerated ARM64 code:")
    print("-" * 40)
    print(generate_arm64_code(program))
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def main_cli():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PTO ISA Compiler")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    args = parser.parse_args()
    
    if args.version:
        print("PTO Compiler v2.0 (Modular Architecture)")
        return
    
    if args.demo:
        run_demo()
        return
    
    parser.print_help()


if __name__ == "__main__":
    main_cli()


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Error types
    'CompilerError', 'ParseError', 'TypeError', 'ValidationError',
    
    # Symbol table
    'Symbol', 'SymbolTable',
    
    # Program representation
    'PTOProgram', 'PTOModule', 'PTOFunctionBuilder',
    
    # Loop fusion
    'OpCategory', 'FusionTileShape', 'FusableOp', 'FusedLoop', 'FusionBarrier',
    'LoopFusionOptimizer', 'get_category', 'is_fusable', 'is_fusion_barrier',
    
    # Mock instructions
    'MockTileInfo', 'MockInstruction', 'convert_program_to_mock_instructions',
    
    # Buffer analysis
    'TileBufferInfo', 'TileBufferAnalyzer',
    
    # Type checker and compiler
    'TypeChecker', 'Optimizer', 'CodeGenerator', 'PTOCompiler',
    
    # Platform-specific generators
    'OrchestrationContext',
    'ARM64CodeGenerator', 'ARM64FusedCodeGenerator',
    'CUDACodeGenerator', 'CUDAFusedCodeGenerator',
    'AscendCodeGenerator', 'AscendFusedCodeGenerator',
    
    # Multi-backend
    'MultiBackendCodeGenerator', 'PTOModuleCompiler', 'BACKENDS',
    
    # Convenience functions
    'generate_all_backends', 'generate_arm64_code', 'generate_cuda_code', 'generate_ascend_code',
    
    # Utilities
    'apply_binary_expansion', 'apply_loop_replay_optimization',
    
    # ISA types
    'ElementType', 'MemorySpace', 'TileShape', 'TileType', 'MemRefType',
    
    # Demo
    'run_demo', 'main_cli',

    # PTO-AS (python frontend)
    'compile_python_kernel_to_pto', 'compile_python_file_to_pto',
]


def compile_python_kernel_to_pto(*, source: str, kernel: str, consts: Optional[Dict[str, Any]] = None) -> str:
    """
    Compile a Python kernel (AST frontend) into PTO-AS text accepted by `ptoas`.

    This is the recommended way to generate `*.pto` files in this repo.
    """
    try:
        from ptoas.python import ast_frontend  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "compile_python_kernel_to_pto: missing `ptoas.python` frontend. "
            "Ensure `ptoas/` is present in the repo root (or installed on PYTHONPATH)."
        ) from exc

    if consts is None:
        return ast_frontend.compile_kernel_spec_from_source(source, func_name=str(kernel)).pto
    return ast_frontend.compile_kernel_spec_from_source_with_consts(source, func_name=str(kernel), consts=dict(consts)).pto


def compile_python_file_to_pto(
    path: Union[str, os.PathLike[str]],
    *,
    kernel: Optional[str] = None,
    consts: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compile a Python file into PTO-AS text accepted by `ptoas`.

    If `kernel` is omitted and the file defines exactly one function, that function is used.
    """
    from pathlib import Path

    p = Path(path)
    source = p.read_text(encoding="utf-8")
    if kernel is None:
        try:
            from ptoas.python import ast_frontend  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "compile_python_file_to_pto: missing `ptoas.python` frontend. "
                "Ensure `ptoas/` is present in the repo root (or installed on PYTHONPATH)."
            ) from exc
        names = ast_frontend.list_kernel_functions(source)
        if len(names) != 1:
            raise ValueError(f"compile_python_file_to_pto: pass --kernel (available: {', '.join(names)})")
        kernel = names[0]

    return compile_python_kernel_to_pto(source=source, kernel=str(kernel), consts=consts)
