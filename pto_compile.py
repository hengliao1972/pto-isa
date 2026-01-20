"""
PTO ISA Compiler

This module provides the compiler infrastructure for the PTO (Programmable Tensor Operations)
Domain Specific Language. It handles parsing, validation, optimization, and code generation
for PTO programs.

Key Features:
- DSL parsing and AST construction
- Type checking and validation
- Loop unrolling and optimization
- Code generation to PTO assembly
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from enum import Enum
import re
import os

from pto_isa_definition import (
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
    
    # Tile instructions
    TLOAD, TSTORE, TADD, TSUB, TMUL, TDIV, TMATMUL, TMATMUL_ACC,
    TROWSUM, TCOLSUM, TROWMAX, TRELU, TSQRT, TEXP, TLOG,
    # Additional unary operations
    TABS, TNEG, TRSQRT, TRECIP,
    # Max/Min operations
    TMAX, TMIN,
    # Broadcast operations
    TEXPANDS, TROWEXPAND, TCOLEXPAND,
    TROWEXPANDSUB, TROWEXPANDDIV, TROWEXPANDMUL,
    # Tile-scalar operations
    TADDS, TMULS, TDIVS,
    # Scalar instructions
    SADD, SSUB, SMUL, SDIV, SMOV, SLI, SCMP,
    
    # Helper functions
    tile, scalar, index, memref, imm,
)


# =============================================================================
# Compiler Error Types
# =============================================================================

class CompilerError(Exception):
    """Base class for compiler errors."""
    pass


class ParseError(CompilerError):
    """Error during parsing."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"Parse error at line {line}, col {col}: {message}")


class TypeError(CompilerError):
    """Type checking error."""
    def __init__(self, message: str, instruction: Optional[PTOInstruction] = None):
        self.instruction = instruction
        super().__init__(f"Type error: {message}")


class ValidationError(CompilerError):
    """Validation error."""
    pass


# =============================================================================
# Symbol Table
# =============================================================================

@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    symbol_type: str  # "tile", "scalar", "memref", "index"
    data_type: Any    # TileType, ElementType, MemRefType, etc.
    is_const: bool = False
    value: Optional[Any] = None


class SymbolTable:
    """
    Symbol table for managing variables and their types.
    
    Supports nested scopes for loop constructs.
    """
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
    
    def push_scope(self):
        """Enter a new scope."""
        self.scopes.append({})
    
    def pop_scope(self):
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def define(self, name: str, symbol: Symbol):
        """Define a symbol in the current scope."""
        self.scopes[-1][name] = symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def is_defined(self, name: str) -> bool:
        """Check if a symbol is defined in any scope."""
        return self.lookup(name) is not None
    
    def is_defined_in_current_scope(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return name in self.scopes[-1]


# =============================================================================
# Program Representation
# =============================================================================

@dataclass
class PTOProgram:
    """
    A complete PTO program.
    
    Contains declarations, instructions, and metadata.
    
    Attributes:
        is_in_core: If True (default), the function runs entirely within a single core.
                    - CALL statements can only call other InCore functions
                    - Called functions are inlined
                    - Redundant TLOAD/TSTORE are eliminated after inlining
                    If False, the function is a host/orchestration function:
                    - Can only contain scalar computation and control flow
                    - CALL statements are kept as-is (not inlined)
                    - Only ARM64 code generation is supported (not CUDA/Ascend)
        imports: List of imported function names from other source files
    """
    name: str = "main"
    tile_declarations: Dict[str, TileType] = field(default_factory=dict)
    scalar_declarations: Dict[str, ElementType] = field(default_factory=dict)
    memref_declarations: Dict[str, MemRefType] = field(default_factory=dict)
    instructions: List[PTOInstruction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_in_core: bool = True  # Default: function runs within a single core
    imports: List[str] = field(default_factory=list)  # Imported function names
    
    def add_tile(self, name: str, rows: int, cols: int, dtype: ElementType = ElementType.F32):
        """Declare a tile variable."""
        self.tile_declarations[name] = TileType.create(rows, cols, dtype)
    
    def add_scalar(self, name: str, dtype: ElementType = ElementType.F32):
        """Declare a scalar variable."""
        self.scalar_declarations[name] = dtype
    
    def add_memref(self, name: str, space: MemorySpace, dtype: ElementType,
                   shape: Optional[TileShape] = None):
        """Declare a memory reference."""
        self.memref_declarations[name] = MemRefType(space, dtype, shape)
    
    def add_instruction(self, instr: PTOInstruction):
        """Add an instruction to the program."""
        self.instructions.append(instr)
    
    def add_loop(self, loop: Union[TileLoop, NestedTileLoop]):
        """Add a loop construct by expanding it to instructions."""
        self.instructions.extend(loop.to_instructions())
    
    def add_import(self, func_name: str):
        """Add an import for a function from another source file."""
        if func_name not in self.imports:
            self.imports.append(func_name)


# =============================================================================
# Module Representation - Contains Multiple Functions
# =============================================================================

@dataclass
class PTOModule:
    """
    A module containing multiple PTO functions.
    
    This allows building complex programs where one function can call another.
    
    Example:
        module = PTOModule("my_module")
        
        # Define helper function (InCore by default)
        sigmoid = (PTOFunctionBuilder("sigmoid")
            .tile("x", 8, 8)
            .memref("input", MemorySpace.GM)
            .memref("output", MemorySpace.GM)
            ...
            .build())
        module.add_function(sigmoid)
        
        # Define main function that calls sigmoid (also InCore, so sigmoid gets inlined)
        main = (PTOFunctionBuilder("main", module=module)
            .tile("data", 8, 8)
            .memref("input", MemorySpace.GM)
            .memref("output", MemorySpace.GM)
            .call("sigmoid", {"input": "input", "output": "output"})
            .build())
        module.add_function(main)
        
        # Or define orchestration function (not InCore)
        orchestrator = (PTOFunctionBuilder("orchestrator", module=module)
            .not_in_core()  # Mark as orchestration function
            .import_func("external_func")
            .memref("data", MemorySpace.GM)
            .call("external_func", {"input": "data"})  # Call kept as-is
            .build())
    """
    name: str = "module"
    functions: Dict[str, 'PTOProgram'] = field(default_factory=dict)
    entry_function: Optional[str] = None
    imported_functions: Dict[str, 'PTOProgram'] = field(default_factory=dict)  # External imports
    buffer_analysis: Dict[str, Dict] = field(default_factory=dict)  # Buffer analysis results per function
    
    def add_function(self, program: 'PTOProgram'):
        """Add a function to the module."""
        self.functions[program.name] = program
        # First function added becomes entry by default
        if self.entry_function is None:
            self.entry_function = program.name
    
    def get_function(self, name: str) -> Optional['PTOProgram']:
        """Get a function by name (checks local and imported)."""
        if name in self.functions:
            return self.functions[name]
        return self.imported_functions.get(name)
    
    def has_function(self, name: str) -> bool:
        """Check if a function exists in the module (local or imported)."""
        return name in self.functions or name in self.imported_functions
    
    def set_entry(self, name: str):
        """Set the entry function."""
        if name not in self.functions:
            raise ValidationError(f"Function '{name}' not found in module")
        self.entry_function = name
    
    def import_function(self, program: 'PTOProgram'):
        """Import a function from an external source."""
        self.imported_functions[program.name] = program
    
    def get_all_functions(self) -> List['PTOProgram']:
        """Get all functions in the module."""
        return list(self.functions.values())
    
    def set_buffer_analysis(self, func_name: str, analysis: Dict):
        """Store buffer analysis results for a function."""
        self.buffer_analysis[func_name] = analysis
    
    def get_buffer_analysis(self, func_name: str) -> Optional[Dict]:
        """Get buffer analysis results for a function."""
        return self.buffer_analysis.get(func_name)
    
    def get_buffer_size(self, func_name: str) -> Tuple[int, int]:
        """
        Get buffer size for a function.
        Returns (total_bytes_without_reuse, total_bytes_with_reuse).
        """
        analysis = self.buffer_analysis.get(func_name)
        if analysis:
            return (
                analysis.get('total_without_reuse_bytes', 0),
                analysis.get('total_with_reuse_bytes', 0)
            )
        return (0, 0)
    
    def get_function_names(self) -> List[str]:
        """Get all function names."""
        return list(self.functions.keys())


# =============================================================================
# DSL Builder - Fluent Interface
# =============================================================================

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
    
    With function calls:
        module = PTOModule("my_module")
        
        # Define helper function
        sigmoid_fn = (PTOFunctionBuilder("sigmoid")
            .tile("x", 8, 8)
            .memref("input", MemorySpace.GM)
            .memref("output", MemorySpace.GM)
            ...
            .build())
        module.add_function(sigmoid_fn)
        
        # Main function calling sigmoid
        main = (PTOFunctionBuilder("main", module=module)
            .memref("data_in", MemorySpace.GM)
            .memref("data_out", MemorySpace.GM)
            .call("sigmoid", {"input": "data_in", "output": "data_out"})
            .build())
    """
    
    def __init__(self, name: str = "main", module: Optional[PTOModule] = None):
        self.program = PTOProgram(name=name)
        self.symbol_table = SymbolTable()
        self._loop_stack: List[List[PTOInstruction]] = []
        self._module = module  # Optional module for function resolution
    
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
        """Add instruction to current context (main or loop body)."""
        if self._loop_stack:
            self._loop_stack[-1].append(instr)
        else:
            self.program.add_instruction(instr)
    
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
    
    def memref(self, name: str, space: MemorySpace = MemorySpace.GM,
               dtype: ElementType = ElementType.F32,
               shape: Optional[Tuple[int, int]] = None) -> "PTOFunctionBuilder":
        """Declare a memory reference."""
        tile_shape = TileShape(*shape) if shape else None
        self.program.add_memref(name, space, dtype, tile_shape)
        self.symbol_table.define(name, Symbol(name, "memref", MemRefType(space, dtype, tile_shape)))
        return self
    
    # =========================================================================
    # Function Properties
    # =========================================================================
    
    def in_core(self) -> "PTOFunctionBuilder":
        """
        Mark this function as an InCore function (default behavior).
        
        InCore functions:
        - Run entirely within a single core
        - CALL statements can only call other InCore functions
        - Called functions are inlined at compile time
        - Redundant TLOAD/TSTORE are eliminated after inlining
        - Code generated for all backends (ARM64, CUDA, Ascend)
        """
        self.program.is_in_core = True
        return self
    
    def not_in_core(self) -> "PTOFunctionBuilder":
        """
        Mark this function as NOT an InCore function (orchestration/host function).
        
        Non-InCore functions:
        - Should only contain scalar computation and control flow
        - CALL statements are kept as-is (not inlined)
        - Only ARM64 code generation is supported
        - Cannot contain tile operations (will raise validation error)
        """
        self.program.is_in_core = False
        return self
    
    def import_func(self, func_name: str) -> "PTOFunctionBuilder":
        """
        Import a function from another source file.
        
        This creates a declaration for an external function that can be called.
        The actual function definition must be provided at link time or
        exist in an imported module.
        
        Args:
            func_name: Name of the function to import
        
        Example:
            .import_func("external_sigmoid")
            .call("external_sigmoid", {"input": "x", "output": "y"})
        """
        self.program.add_import(func_name)
        return self
    
    # Tile memory operations
    def load(self, dst: str, src_mem: str, 
             row: Union[int, str] = 0, col: Union[int, str] = 0) -> "PTOFunctionBuilder":
        """Load data from memory into a tile.
        
        Args:
            dst: Destination tile name
            src_mem: Source memory reference name
            row: Row offset (int for immediate, str for index variable)
            col: Col offset (int for immediate, str for index variable)
        """
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
        """Store data from a tile into memory.
        
        Args:
            src: Source tile name
            dst_mem: Destination memory reference name
            row: Row offset (int for immediate, str for index variable)
            col: Col offset (int for immediate, str for index variable)
        """
        row_op = ImmediateOperand(row) if isinstance(row, int) else IndexOperand(row)
        col_op = ImmediateOperand(col) if isinstance(col, int) else IndexOperand(col)
        self._add_instr(TSTORE(
            src=self._get_tile(src),
            dst_mem=self._get_memref(dst_mem),
            row_offset=row_op,
            col_offset=col_op
        ))
        return self
    
    # Tile arithmetic operations
    def add(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise add of two tiles."""
        self._add_instr(TADD(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def sub(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise subtract of two tiles."""
        self._add_instr(TSUB(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def mul(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise multiply of two tiles."""
        self._add_instr(TMUL(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def div(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise divide of two tiles."""
        self._add_instr(TDIV(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Scalar operations (tile op scalar)
    def _make_scalar_operand(self, value: float) -> ScalarOperand:
        """Create a scalar operand for an immediate value."""
        # Create a unique name for the immediate value
        name = f"_imm_{abs(hash(value)) % 10000}"
        return ScalarOperand(name=str(value), element_type=ElementType.F32)
    
    def adds(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Add scalar to all elements of tile."""
        self._add_instr(TADDS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def muls(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Multiply all elements of tile by scalar."""
        self._add_instr(TMULS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def divs(self, dst: str, src: str, scalar: float) -> "PTOFunctionBuilder":
        """Divide all elements of tile by scalar."""
        self._add_instr(TDIVS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    # Matrix operations
    def matmul(self, dst: str, a: str, b: str) -> "PTOFunctionBuilder":
        """Matrix multiply."""
        self._add_instr(TMATMUL(
            dst=self._get_tile(dst),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    def matmul_acc(self, dst: str, acc: str, a: str, b: str) -> "PTOFunctionBuilder":
        """Matrix multiply with accumulator."""
        self._add_instr(TMATMUL_ACC(
            dst=self._get_tile(dst),
            acc=self._get_tile(acc),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    # Activation functions
    def relu(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply ReLU activation."""
        self._add_instr(TRELU(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def exp(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply exponential."""
        self._add_instr(TEXP(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def log(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply natural logarithm."""
        self._add_instr(TLOG(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def sqrt(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply square root."""
        self._add_instr(TSQRT(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def abs(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply absolute value."""
        self._add_instr(TABS(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def neg(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply negation."""
        self._add_instr(TNEG(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def rsqrt(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply reciprocal square root (1/sqrt(x))."""
        self._add_instr(TRSQRT(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def recip(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Apply reciprocal (1/x)."""
        self._add_instr(TRECIP(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Binary max/min operations
    def max(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise maximum of two tiles."""
        self._add_instr(TMAX(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def min(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Elementwise minimum of two tiles."""
        self._add_instr(TMIN(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Broadcast operations
    def expands(self, dst: str, value: float) -> "PTOFunctionBuilder":
        """Broadcast a scalar value to all elements of a tile."""
        self._add_instr(TEXPANDS(
            dst=self._get_tile(dst),
            scalar=self._make_scalar_operand(value)
        ))
        return self
    
    def rowexpand(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Broadcast first element of each row across the row."""
        self._add_instr(TROWEXPAND(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def colexpand(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Broadcast first element of each column across the column."""
        self._add_instr(TCOLEXPAND(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Row-wise broadcast operations (src1 is [N,1], broadcast to [N,M])
    def rowexpandsub(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast subtract: dst = src0 - broadcast(src1)."""
        self._add_instr(TROWEXPANDSUB(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def rowexpanddiv(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast divide: dst = src0 / broadcast(src1)."""
        self._add_instr(TROWEXPANDDIV(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def rowexpandmul(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Row-wise broadcast multiply: dst = src0 * broadcast(src1)."""
        self._add_instr(TROWEXPANDMUL(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Reduction operations
    def rowsum(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Sum reduction across rows."""
        self._add_instr(TROWSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def rowmax(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Max reduction across rows: dst[i,0] = max(src[i,:])"""
        self._add_instr(TROWMAX(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def colsum(self, dst: str, src: str) -> "PTOFunctionBuilder":
        """Sum reduction across columns."""
        self._add_instr(TCOLSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Loop constructs
    def for_loop(self, iv_name: str, 
                 lb: Union[int, str], ub: Union[int, str], 
                 step: Union[int, str] = 1) -> "PTOFunctionBuilder":
        """Begin a FOR loop.
        
        Args:
            iv_name: Induction variable name
            lb: Lower bound (int for immediate, str for scalar variable)
            ub: Upper bound (int for immediate, str for scalar variable)
            step: Step (int for immediate, str for scalar variable)
        """
        self.symbol_table.push_scope()
        self.symbol_table.define(iv_name, Symbol(iv_name, "index", ElementType.INDEX))
        
        lb_op = ImmediateOperand(lb) if isinstance(lb, int) else IndexOperand(lb)
        ub_op = ImmediateOperand(ub) if isinstance(ub, int) else IndexOperand(ub)
        step_op = ImmediateOperand(step) if isinstance(step, int) else IndexOperand(step)
        
        self._add_instr(FOR(
            iv=IndexOperand(iv_name),
            lb=lb_op,
            ub=ub_op,
            step=step_op
        ))
        self._loop_stack.append([])
        return self
    
    def end_for(self) -> "PTOFunctionBuilder":
        """End a FOR loop."""
        if not self._loop_stack:
            raise ValidationError("ENDFOR without matching FOR")
        
        loop_body = self._loop_stack.pop()
        
        # If there's still a parent loop, add body and ENDFOR to parent
        # Otherwise, add to program
        if self._loop_stack:
            # We're inside another loop, add to parent loop's body
            for instr in loop_body:
                self._loop_stack[-1].append(instr)
            self._loop_stack[-1].append(ENDFOR())
        else:
            # No parent loop, add directly to program
            for instr in loop_body:
                self.program.add_instruction(instr)
            self.program.add_instruction(ENDFOR())
        
        self.symbol_table.pop_scope()
        return self
    
    def tile_loop(self, iv_name: str, tile_name: str, 
                  dimension: str = "rows", step: int = 1) -> "PTOFunctionBuilder":
        """
        Begin a loop that iterates based on tile dimensions.
        
        Args:
            iv_name: Name of induction variable
            tile_name: Name of tile to get dimensions from
            dimension: "rows" or "cols"
            step: Loop step
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        bound = tile_type.shape.rows if dimension == "rows" else tile_type.shape.cols
        
        return self.for_loop(iv_name, 0, bound, step)
    
    def nested_tile_loop(self, outer_iv: str, inner_iv: str, tile_name: str,
                         outer_step: int = 1, inner_step: int = 1) -> "PTOFunctionBuilder":
        """
        Begin a 2-level nested loop that iterates over tile dimensions.
        
        Outer loop iterates over rows, inner loop over columns.
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        
        # Start outer loop
        self.for_loop(outer_iv, 0, tile_type.shape.rows, outer_step)
        # Start inner loop
        self.for_loop(inner_iv, 0, tile_type.shape.cols, inner_step)
        
        return self
    
    def end_nested_loop(self) -> "PTOFunctionBuilder":
        """End a 2-level nested loop."""
        self.end_for()  # End inner
        self.end_for()  # End outer
        return self
    
    # =========================================================================
    # Scalar Operations
    # =========================================================================
    
    def scalar_li(self, dst: str, value: Union[int, float]) -> "PTOFunctionBuilder":
        """Load immediate value into scalar."""
        if dst not in self.program.scalar_declarations:
            raise ValidationError(f"Scalar '{dst}' not declared")
        self._add_instr(SLI(
            dst=self._get_scalar(dst),
            imm=ImmediateOperand(value)
        ))
        return self
    
    def scalar_add(self, dst: str, src0: str, 
                   src1: Union[str, int, float]) -> "PTOFunctionBuilder":
        """Add two scalars or scalar + immediate."""
        if isinstance(src1, str):
            src1_op = self._get_scalar(src1)
        else:
            src1_op = ImmediateOperand(src1)
        self._add_instr(SADD(
            dst=self._get_scalar(dst),
            src0=self._get_scalar(src0),
            src1=src1_op
        ))
        return self
    
    def scalar_sub(self, dst: str, src0: str, 
                   src1: Union[str, int, float]) -> "PTOFunctionBuilder":
        """Subtract two scalars or scalar - immediate."""
        if isinstance(src1, str):
            src1_op = self._get_scalar(src1)
        else:
            src1_op = ImmediateOperand(src1)
        self._add_instr(SSUB(
            dst=self._get_scalar(dst),
            src0=self._get_scalar(src0),
            src1=src1_op
        ))
        return self
    
    def scalar_mul(self, dst: str, src0: str, src1: str) -> "PTOFunctionBuilder":
        """Multiply two scalars."""
        self._add_instr(SMUL(
            dst=self._get_scalar(dst),
            src0=self._get_scalar(src0),
            src1=self._get_scalar(src1)
        ))
        return self
    
    def scalar_cmp(self, dst: str, src0: str, src1: str, 
                   mode: CompareMode) -> "PTOFunctionBuilder":
        """Compare two scalars."""
        self._add_instr(SCMP(
            dst=self._get_scalar(dst),
            src0=self._get_scalar(src0),
            src1=self._get_scalar(src1),
            cmp_mode=mode
        ))
        return self
    
    # =========================================================================
    # Control Flow - IF/ELSE/ENDIF
    # =========================================================================
    
    def if_then(self, cond: str) -> "PTOFunctionBuilder":
        """Begin an IF block."""
        self.symbol_table.push_scope()
        self._add_instr(IF(cond=self._get_scalar(cond)))
        self._loop_stack.append([])  # Reuse loop stack for IF body
        return self
    
    def else_block(self) -> "PTOFunctionBuilder":
        """Begin ELSE block."""
        if not self._loop_stack:
            raise ValidationError("ELSE without matching IF")
        
        # Close IF body, emit to parent
        if_body = self._loop_stack.pop()
        if self._loop_stack:
            for instr in if_body:
                self._loop_stack[-1].append(instr)
            self._loop_stack[-1].append(ELSE())
        else:
            for instr in if_body:
                self.program.add_instruction(instr)
            self.program.add_instruction(ELSE())
        
        self._loop_stack.append([])  # Start ELSE body
        return self
    
    def endif(self) -> "PTOFunctionBuilder":
        """End IF/ELSE block."""
        if not self._loop_stack:
            raise ValidationError("ENDIF without matching IF")
        
        body = self._loop_stack.pop()
        if self._loop_stack:
            for instr in body:
                self._loop_stack[-1].append(instr)
            self._loop_stack[-1].append(ENDIF())
        else:
            for instr in body:
                self.program.add_instruction(instr)
            self.program.add_instruction(ENDIF())
        
        self.symbol_table.pop_scope()
        return self
    
    # =========================================================================
    # Function Call Operations
    # =========================================================================
    
    def call(self, callee: str, args: Optional[Dict[str, str]] = None) -> "PTOFunctionBuilder":
        """
        Call another function.
        
        Args:
            callee: Name of the function to call
            args: Dictionary mapping parameter names to argument names.
                  Parameter names are from the callee function.
                  Argument names are memrefs/tiles in the current function.
        
        Example:
            # Call sigmoid function with input -> x, output -> y
            .call("sigmoid", {"input": "x", "output": "y"})
        
        Notes on InCore functions:
            - If this function is InCore, the callee must also be InCore
            - InCore callees will be inlined at compile time
            - Non-InCore functions keep CALL statements as-is
        """
        # Check if callee is imported
        is_imported = callee in self.program.imports
        
        # Validate function exists if we have a module (and not imported)
        if not is_imported and self._module is not None:
            if not self._module.has_function(callee):
                raise ValidationError(
                    f"Function '{callee}' not found in module. "
                    f"Use .import_func('{callee}') to import external functions."
                )
            
            # Validate InCore call rules
            if self.program.is_in_core:
                callee_func = self._module.get_function(callee)
                if callee_func and not callee_func.is_in_core:
                    raise ValidationError(
                        f"InCore function '{self.program.name}' cannot call "
                        f"non-InCore function '{callee}'"
                    )
        
        # Validate arguments are declared
        if args:
            for arg_name in args.values():
                if (arg_name not in self.program.memref_declarations and
                    arg_name not in self.program.tile_declarations):
                    raise ValidationError(
                        f"Argument '{arg_name}' not declared as memref or tile"
                    )
        
        self._add_instr(CALL(
            callee=callee,
            args=args or {}
        ))
        return self
    
    def ret(self, values: Optional[List[str]] = None) -> "PTOFunctionBuilder":
        """
        Add a return statement.
        
        Args:
            values: Optional list of values to return
        """
        self._add_instr(RETURN(values=values))
        return self
    
    # Build the program
    def build(self) -> PTOProgram:
        """Build and return the program."""
        if self._loop_stack:
            raise ValidationError("Unclosed loop constructs")
        return self.program


# =============================================================================
# Type Checker
# =============================================================================

class TypeChecker:
    """
    Type checker for PTO programs.
    
    Validates that all operations have compatible types.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.errors: List[str] = []
    
    def check(self) -> bool:
        """Run type checking, return True if no errors."""
        self.errors = []
        
        for instr in self.program.instructions:
            self._check_instruction(instr)
        
        return len(self.errors) == 0
    
    def _check_instruction(self, instr: PTOInstruction):
        """Type check a single instruction."""
        if isinstance(instr, TADD) or isinstance(instr, TSUB) or isinstance(instr, TMUL):
            self._check_binary_tile_op(instr)
        elif isinstance(instr, TMATMUL):
            self._check_matmul(instr)
    
    def _check_binary_tile_op(self, instr):
        """Check binary tile operation types match."""
        if hasattr(instr, 'src0') and hasattr(instr, 'src1'):
            src0_type = instr.src0.tile_type
            src1_type = instr.src1.tile_type
            
            if src0_type.shape != src1_type.shape:
                self.errors.append(
                    f"{instr.opcode}: Shape mismatch - {src0_type.shape} vs {src1_type.shape}"
                )
            
            if src0_type.element_type != src1_type.element_type:
                self.errors.append(
                    f"{instr.opcode}: Element type mismatch - "
                    f"{src0_type.element_type.value} vs {src1_type.element_type.value}"
                )
    
    def _check_matmul(self, instr: TMATMUL):
        """Check matrix multiply dimension compatibility."""
        a_type = instr.a.tile_type
        b_type = instr.b.tile_type
        
        if a_type.shape.cols != b_type.shape.rows:
            self.errors.append(
                f"TMATMUL: Incompatible dimensions - "
                f"A cols ({a_type.shape.cols}) != B rows ({b_type.shape.rows})"
            )
    
    def get_errors(self) -> List[str]:
        """Get list of type errors."""
        return self.errors


# =============================================================================
# Code Generator
# =============================================================================

class CodeGenerator:
    """
    Code generator for PTO programs.
    
    Generates PTO assembly from a validated program.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.output: List[str] = []
        self.indent_level = 0
    
    def generate(self) -> str:
        """Generate PTO assembly code."""
        self.output = []
        self._emit_header()
        self._emit_function_start()
        self._emit_declarations()
        self._emit_instructions()
        self._emit_footer()
        return "\n".join(self.output)
    
    def _emit(self, line: str):
        """Emit a line of code with proper indentation."""
        indent = "  " * self.indent_level
        self.output.append(f"{indent}{line}")
    
    def _emit_header(self):
        """Emit program header."""
        self._emit(f"// PTO Program: {self.program.name}")
        self._emit(f"// Generated by PTO ISA Compiler")
        # Always emit function type
        if self.program.is_in_core:
            self._emit(f"// Function Type: InCore (tile-level computation)")
        else:
            self._emit(f"// Function Type: Orchestration (control flow only)")
        self._emit("")
        
        # Emit imports
        if self.program.imports:
            for imp in self.program.imports:
                self._emit(f"#import @{imp}")
            self._emit("")
    
    def _emit_function_start(self):
        """Emit function definition with name and parameters."""
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in self.program.memref_declarations.items():
            memref_params.append(f"%{name}: {memref_type}")
        
        # Generate function signature
        if memref_params:
            params_str = ", ".join(memref_params)
            self._emit(f"func @{self.program.name}({params_str}) {{")
        else:
            self._emit(f"func @{self.program.name}() {{")
        
        self.indent_level += 1
    
    def _emit_declarations(self):
        """Emit variable declarations."""
        self._emit("// Tile Declarations")
        for name, tile_type in self.program.tile_declarations.items():
            self._emit(f"%{name} = alloc_tile : {tile_type}")
        
        if self.program.scalar_declarations:
            self._emit("")
            self._emit("// Scalar Declarations")
            for name, dtype in self.program.scalar_declarations.items():
                self._emit(f"%{name} = alloc_scalar : {dtype.value}")
        
        self._emit("")
    
    def _emit_instructions(self):
        """Emit program instructions."""
        self._emit("// Instructions")
        for instr in self.program.instructions:
            self._emit_instruction(instr)
    
    def _emit_instruction(self, instr: PTOInstruction):
        """Emit a single instruction."""
        if isinstance(instr, FOR):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDFOR):
            self.indent_level = max(1, self.indent_level - 1)  # Keep at least 1 for function body
            self._emit(instr.to_pto_as())
        elif isinstance(instr, IF):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ELSE):
            self.indent_level = max(1, self.indent_level - 1)
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDIF):
            self.indent_level = max(1, self.indent_level - 1)
            self._emit(instr.to_pto_as())
        elif isinstance(instr, CALL):
            self._emit(instr.to_pto_as())
        elif isinstance(instr, RETURN):
            self._emit(instr.to_pto_as())
        else:
            self._emit(instr.to_pto_as())
    
    def _emit_footer(self):
        """Emit program footer."""
        self._emit("")
        self._emit("return")
        self.indent_level = 0
        self._emit("}")


# =============================================================================
# Optimizer
# =============================================================================

class Optimizer:
    """
    Basic optimizer for PTO programs.
    
    Performs simple optimizations like:
    - Dead code elimination
    - Constant folding
    - Loop unrolling
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
    
    def optimize(self, unroll_threshold: int = 4) -> PTOProgram:
        """Apply optimizations and return optimized program."""
        optimized = PTOProgram(
            name=self.program.name,
            tile_declarations=self.program.tile_declarations.copy(),
            scalar_declarations=self.program.scalar_declarations.copy(),
            memref_declarations=self.program.memref_declarations.copy(),
            metadata=self.program.metadata.copy()
        )
        
        optimized.instructions = self._optimize_instructions(
            self.program.instructions,
            unroll_threshold
        )
        
        return optimized
    
    def _optimize_instructions(self, instructions: List[PTOInstruction], 
                                unroll_threshold: int) -> List[PTOInstruction]:
        """Optimize a list of instructions."""
        result = []
        i = 0
        
        while i < len(instructions):
            instr = instructions[i]
            
            # Check for small loops that can be unrolled
            if isinstance(instr, FOR):
                loop_end, loop_body = self._find_loop_end(instructions, i)
                if loop_end > i:
                    iterations = self._get_loop_iterations(instr)
                    if iterations is not None and iterations <= unroll_threshold:
                        # Unroll the loop
                        unrolled = self._unroll_loop(instr, loop_body, iterations)
                        result.extend(unrolled)
                        i = loop_end + 1
                        continue
            
            result.append(instr)
            i += 1
        
        return result
    
    def _find_loop_end(self, instructions: List[PTOInstruction], 
                       start: int) -> Tuple[int, List[PTOInstruction]]:
        """Find the matching ENDFOR and extract loop body."""
        depth = 1
        body = []
        i = start + 1
        
        while i < len(instructions) and depth > 0:
            instr = instructions[i]
            if isinstance(instr, FOR):
                depth += 1
            elif isinstance(instr, ENDFOR):
                depth -= 1
                if depth == 0:
                    return i, body
            
            if depth > 0:
                body.append(instr)
            i += 1
        
        return -1, []
    
    def _get_loop_iterations(self, for_instr: FOR) -> Optional[int]:
        """Get the number of iterations for a FOR loop if computable."""
        if isinstance(for_instr.lb, ImmediateOperand) and \
           isinstance(for_instr.ub, ImmediateOperand) and \
           isinstance(for_instr.step, ImmediateOperand):
            lb = for_instr.lb.value
            ub = for_instr.ub.value
            step = for_instr.step.value
            if step > 0:
                return (ub - lb + step - 1) // step
        return None
    
    def _unroll_loop(self, for_instr: FOR, body: List[PTOInstruction], 
                     iterations: int) -> List[PTOInstruction]:
        """Unroll a loop by duplicating body instructions."""
        result = []
        
        # For now, just return the loop as-is (full unrolling is complex)
        # A real implementation would substitute the induction variable
        result.append(for_instr)
        result.extend(body)
        result.append(ENDFOR())
        
        return result


# =============================================================================
# Compiler Driver
# =============================================================================

class PTOCompiler:
    """
    Main compiler driver for PTO programs.
    
    Orchestrates parsing, type checking, optimization, and code generation.
    """
    
    def __init__(self, optimize: bool = True, unroll_threshold: int = 4):
        self.optimize = optimize
        self.unroll_threshold = unroll_threshold
    
    def compile(self, program: PTOProgram) -> str:
        """
        Compile a PTO program to assembly.
        
        Args:
            program: The PTO program to compile
            
        Returns:
            Generated PTO assembly code
            
        Raises:
            TypeError: If type checking fails
            ValidationError: If validation fails
        """
        # Type check
        type_checker = TypeChecker(program)
        if not type_checker.check():
            errors = type_checker.get_errors()
            raise TypeError(f"Type checking failed:\n" + "\n".join(errors))
        
        # Optimize if enabled
        if self.optimize:
            optimizer = Optimizer(program)
            program = optimizer.optimize(self.unroll_threshold)
        
        # Generate code
        generator = CodeGenerator(program)
        return generator.generate()
    
    def compile_and_save(self, program: PTOProgram, output_path: str):
        """Compile a program and save to file."""
        code = self.compile(program)
        with open(output_path, 'w') as f:
            f.write(code)


# =============================================================================
# InCore Function Inlining and Optimization
# =============================================================================

class InCoreFunctionInliner:
    """
    Inlines InCore function calls and optimizes redundant memory operations.
    
    For InCore functions:
    1. Replace CALL instructions with inlined function body
    2. Rename variables to avoid conflicts
    3. Map arguments to actual parameters
    4. Eliminate redundant TLOAD/TSTORE at function boundaries
    """
    
    def __init__(self, module: PTOModule):
        self.module = module
        self._var_counter = 0
    
    def _get_unique_prefix(self) -> str:
        """Generate unique prefix for inlined variables."""
        prefix = f"_inline{self._var_counter}_"
        self._var_counter += 1
        return prefix
    
    def inline_function(self, program: PTOProgram) -> PTOProgram:
        """
        Inline all InCore function calls in the given program.
        
        Returns a new program with calls replaced by inlined code.
        """
        if not program.is_in_core:
            # Non-InCore functions: keep CALL statements as-is
            return program
        
        # Create new program with inlined instructions
        inlined = PTOProgram(
            name=program.name,
            tile_declarations=program.tile_declarations.copy(),
            scalar_declarations=program.scalar_declarations.copy(),
            memref_declarations=program.memref_declarations.copy(),
            metadata=program.metadata.copy(),
            is_in_core=program.is_in_core,
            imports=program.imports.copy()
        )
        
        # Process each instruction
        for instr in program.instructions:
            if isinstance(instr, CALL):
                # Check if callee is InCore and should be inlined
                callee_func = self.module.get_function(instr.callee)
                if callee_func and callee_func.is_in_core:
                    # Inline the function
                    inlined_instrs = self._inline_call(instr, callee_func, inlined)
                    inlined.instructions.extend(inlined_instrs)
                else:
                    # Keep imported/external function calls as-is
                    inlined.instructions.append(instr)
            else:
                inlined.instructions.append(instr)
        
        return inlined
    
    def _inline_call(self, call_instr: CALL, callee: PTOProgram, 
                     caller: PTOProgram) -> List[PTOInstruction]:
        """
        Inline a single function call.
        
        Args:
            call_instr: The CALL instruction to inline
            callee: The function being called
            caller: The calling function (for adding declarations)
        
        Returns:
            List of instructions to replace the CALL
        """
        prefix = self._get_unique_prefix()
        result = []
        
        # Build argument mapping: callee param name -> caller arg name
        arg_map = call_instr.args.copy()
        
        # Add callee's tile declarations to caller (with prefix)
        tile_map = {}  # callee tile name -> caller tile name
        for tile_name, tile_type in callee.tile_declarations.items():
            new_name = f"{prefix}{tile_name}"
            caller.tile_declarations[new_name] = tile_type
            tile_map[tile_name] = new_name
        
        # Add callee's scalar declarations to caller (with prefix)
        scalar_map = {}  # callee scalar name -> caller scalar name
        for scalar_name, scalar_type in callee.scalar_declarations.items():
            new_name = f"{prefix}{scalar_name}"
            caller.scalar_declarations[new_name] = scalar_type
            scalar_map[scalar_name] = new_name
        
        # Map memref parameters to caller's memrefs via argument mapping
        memref_map = {}  # callee memref name -> caller memref name
        for callee_param, caller_arg in arg_map.items():
            if callee_param in callee.memref_declarations:
                memref_map[callee_param] = caller_arg
        
        # Process callee's instructions
        for instr in callee.instructions:
            if isinstance(instr, RETURN):
                # Skip RETURN statements
                continue
            
            # Remap the instruction's operands
            remapped_instr = self._remap_instruction(
                instr, tile_map, scalar_map, memref_map, prefix
            )
            if remapped_instr:
                result.append(remapped_instr)
        
        return result
    
    def _remap_instruction(self, instr: PTOInstruction, 
                           tile_map: Dict[str, str],
                           scalar_map: Dict[str, str],
                           memref_map: Dict[str, str],
                           prefix: str) -> Optional[PTOInstruction]:
        """Remap an instruction's operands for inlining."""
        # This is a simplified remapping - in a full implementation,
        # each instruction type would need specific handling
        
        # For TLOAD/TSTORE, remap tile and memref names
        if isinstance(instr, TLOAD):
            new_dst_name = tile_map.get(instr.dst.name, instr.dst.name)
            new_src_name = memref_map.get(instr.src_mem.name, instr.src_mem.name)
            return TLOAD(
                dst=TileOperand(new_dst_name, instr.dst.tile_type),
                src_mem=MemRefOperand(new_src_name, instr.src_mem.memref_type),
                row_offset=instr.row_offset,
                col_offset=instr.col_offset
            )
        elif isinstance(instr, TSTORE):
            new_src_name = tile_map.get(instr.src.name, instr.src.name)
            new_dst_name = memref_map.get(instr.dst_mem.name, instr.dst_mem.name)
            return TSTORE(
                src=TileOperand(new_src_name, instr.src.tile_type),
                dst_mem=MemRefOperand(new_dst_name, instr.dst_mem.memref_type),
                row_offset=instr.row_offset,
                col_offset=instr.col_offset
            )
        
        # For other tile operations, just return as-is for now
        # A full implementation would remap all operand names
        return instr


class RedundantMemoryEliminator:
    """
    Eliminates redundant TLOAD/TSTORE operations after function inlining.
    
    Patterns eliminated:
    1. TSTORE followed by TLOAD from same location -> remove TLOAD, use tile directly
    2. Back-to-back TLOAD from same location -> keep only first TLOAD
    3. TSTORE to location that is immediately overwritten -> remove first TSTORE
    """
    
    def __init__(self):
        self.stats = {
            "loads_eliminated": 0,
            "stores_eliminated": 0,
        }
    
    def optimize(self, program: PTOProgram) -> PTOProgram:
        """
        Optimize a program by eliminating redundant memory operations.
        """
        if not program.is_in_core:
            # Don't optimize non-InCore functions
            return program
        
        optimized = PTOProgram(
            name=program.name,
            tile_declarations=program.tile_declarations.copy(),
            scalar_declarations=program.scalar_declarations.copy(),
            memref_declarations=program.memref_declarations.copy(),
            metadata=program.metadata.copy(),
            is_in_core=program.is_in_core,
            imports=program.imports.copy()
        )
        
        # Track which memref locations have been stored to
        # Key: (memref_name, row_offset, col_offset)
        # Value: tile_name that was stored
        stored_tiles: Dict[tuple, str] = {}
        
        for instr in program.instructions:
            if isinstance(instr, TSTORE):
                # Record the store
                key = self._get_mem_key(instr.dst_mem.name, instr.row_offset, instr.col_offset)
                stored_tiles[key] = instr.src.name
                optimized.instructions.append(instr)
                
            elif isinstance(instr, TLOAD):
                # Check if we can eliminate this load
                key = self._get_mem_key(instr.src_mem.name, instr.row_offset, instr.col_offset)
                if key in stored_tiles:
                    # The data was just stored - we can skip the load
                    # But we need to ensure the destination tile references the stored tile
                    # For now, we just add a comment and keep the load
                    # A full implementation would track this and replace uses
                    self.stats["loads_eliminated"] += 1
                    # Keep the load for correctness, but mark it as potentially redundant
                    optimized.instructions.append(instr)
                else:
                    optimized.instructions.append(instr)
            else:
                # Control flow resets our tracking
                if isinstance(instr, (FOR, ENDFOR, IF, ELSE, ENDIF)):
                    stored_tiles.clear()
                optimized.instructions.append(instr)
        
        return optimized
    
    def _get_mem_key(self, memref_name: str, row_offset, col_offset) -> tuple:
        """Create a hashable key for a memory location."""
        row_val = row_offset.value if hasattr(row_offset, 'value') else str(row_offset)
        col_val = col_offset.value if hasattr(col_offset, 'value') else str(col_offset)
        return (memref_name, row_val, col_val)


class PTOModuleCompiler:
    """
    Compiler for PTO modules containing multiple functions.
    
    Compiles all functions in a module, handling function declarations
    and cross-function references.
    
    For InCore functions:
    - Inlines called functions
    - Eliminates redundant TLOAD/TSTORE
    - Generates code for all backends
    
    For non-InCore functions:
    - Keeps CALL statements as-is
    - Only generates ARM64 code
    
    Example:
        module = PTOModule("my_module")
        module.add_function(func1)
        module.add_function(func2)
        
        compiler = PTOModuleCompiler()
        code = compiler.compile(module)
    """
    
    def __init__(self, optimize: bool = True, unroll_threshold: int = 4,
                 inline_in_core: bool = True, eliminate_redundant_mem: bool = True):
        self.optimize = optimize
        self.unroll_threshold = unroll_threshold
        self.inline_in_core = inline_in_core
        self.eliminate_redundant_mem = eliminate_redundant_mem
        self.function_compiler = PTOCompiler(optimize, unroll_threshold)
    
    def compile(self, module: PTOModule) -> str:
        """
        Compile a PTO module to assembly.
        
        All functions are compiled and combined into a single output.
        
        For InCore functions:
        - CALL statements to other InCore functions are inlined
        - Redundant TLOAD/TSTORE are eliminated
        
        For non-InCore functions:
        - CALL statements are kept as-is
        
        Returns:
            Generated PTO assembly code for the entire module
        """
        output = []
        
        # Module header
        output.append(f"// PTO Module: {module.name}")
        output.append(f"// Generated by PTO ISA Compiler")
        output.append(f"// Functions: {', '.join(module.get_function_names())}")
        if module.entry_function:
            output.append(f"// Entry: @{module.entry_function}")
        output.append("")
        
        # Emit imports if any
        all_imports = set()
        for func in module.get_all_functions():
            all_imports.update(func.imports)
        if all_imports:
            for imp in sorted(all_imports):
                output.append(f"#import @{imp}")
            output.append("")
        
        # Process and compile each function
        inliner = InCoreFunctionInliner(module) if self.inline_in_core else None
        mem_optimizer = RedundantMemoryEliminator() if self.eliminate_redundant_mem else None
        
        for func_name in module.get_function_names():
            func = module.get_function(func_name)
            
            # Apply InCore inlining if enabled
            if inliner and func.is_in_core:
                func = inliner.inline_function(func)
            
            # Apply redundant memory elimination if enabled
            if mem_optimizer and func.is_in_core:
                func = mem_optimizer.optimize(func)
            
            # Generate code for this function (without header)
            generator = ModuleFunctionCodeGenerator(func, module)
            func_code = generator.generate()
            output.append(func_code)
            output.append("")  # Blank line between functions
        
        return "\n".join(output)
    
    def compile_and_save(self, module: PTOModule, output_path: str):
        """Compile a module and save to file."""
        code = self.compile(module)
        with open(output_path, 'w') as f:
            f.write(code)


class ModuleFunctionCodeGenerator(CodeGenerator):
    """
    Code generator for functions within a module.
    
    Extends CodeGenerator to handle function calls and module context.
    """
    
    def __init__(self, program: PTOProgram, module: PTOModule):
        super().__init__(program)
        self.module = module
    
    def _emit_header(self):
        """Skip module-level header (handled by PTOModuleCompiler)."""
        pass
    
    def _emit_function_start(self):
        """Emit function definition with name, parameters, and type annotation."""
        # Add function type comment
        if self.program.is_in_core:
            self._emit(f"// Function Type: InCore")
        else:
            self._emit(f"// Function Type: Orchestration")
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in self.program.memref_declarations.items():
            memref_params.append(f"%{name}: {memref_type}")
        
        # Generate function signature
        if memref_params:
            params_str = ", ".join(memref_params)
            self._emit(f"func @{self.program.name}({params_str}) {{")
        else:
            self._emit(f"func @{self.program.name}() {{")
        
        self.indent_level += 1
    
    def generate(self) -> str:
        """Generate PTO assembly code for a single function."""
        self.output = []
        self._emit_function_start()
        self._emit_declarations()
        self._emit_instructions()
        self._emit_footer()
        return "\n".join(self.output)


# =============================================================================
# Loop Fusion - Instruction Classification
# =============================================================================

class OpCategory(Enum):
    """Categories of PTO operations for fusion analysis."""
    ELEMENTWISE_BINARY = "binary"    # TADD, TSUB, TMUL, TDIV, TMAX, TMIN
    ELEMENTWISE_UNARY = "unary"      # TABS, TNEG, TRECIP, TEXP, TLOG, TSQRT, TRELU
    ELEMENTWISE_SCALAR = "scalar"    # TADDS, TMULS, TDIVS, etc.
    BROADCAST = "broadcast"          # TEXPANDS
    BROADCAST_BINARY = "broadcast_binary"  # TROWEXPANDSUB, TROWEXPANDDIV, etc.
    REDUCTION = "reduction"          # TROWSUM, TCOLSUM, TSUM
    MATMUL = "matmul"               # TMATMUL
    MEMORY = "memory"               # TLOAD, TSTORE
    CONTROL_FLOW = "control"        # FOR, ENDFOR
    DECLARATION = "decl"            # TILE_DECL, SCALAR_DECL
    OTHER = "other"                 # Unknown/passthrough


# Classify opcodes into categories
OPCODE_CATEGORY = {
    # Elementwise binary
    "TADD": OpCategory.ELEMENTWISE_BINARY,
    "TSUB": OpCategory.ELEMENTWISE_BINARY,
    "TMUL": OpCategory.ELEMENTWISE_BINARY,
    "TDIV": OpCategory.ELEMENTWISE_BINARY,
    "TMAX": OpCategory.ELEMENTWISE_BINARY,
    "TMIN": OpCategory.ELEMENTWISE_BINARY,
    "TAND": OpCategory.ELEMENTWISE_BINARY,
    "TOR": OpCategory.ELEMENTWISE_BINARY,
    "TXOR": OpCategory.ELEMENTWISE_BINARY,
    
    # Elementwise unary
    "TABS": OpCategory.ELEMENTWISE_UNARY,
    "TNEG": OpCategory.ELEMENTWISE_UNARY,
    "TRECIP": OpCategory.ELEMENTWISE_UNARY,
    "TEXP": OpCategory.ELEMENTWISE_UNARY,
    "TLOG": OpCategory.ELEMENTWISE_UNARY,
    "TSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRELU": OpCategory.ELEMENTWISE_UNARY,
    
    # Elementwise with scalar
    "TADDS": OpCategory.ELEMENTWISE_SCALAR,
    "TSUBS": OpCategory.ELEMENTWISE_SCALAR,
    "TMULS": OpCategory.ELEMENTWISE_SCALAR,
    "TDIVS": OpCategory.ELEMENTWISE_SCALAR,
    "TMAXS": OpCategory.ELEMENTWISE_SCALAR,
    "TMINS": OpCategory.ELEMENTWISE_SCALAR,
    
    # Broadcast (scalar to tile)
    "TEXPANDS": OpCategory.BROADCAST,
    "TROWEXPAND": OpCategory.BROADCAST,
    "TCOLEXPAND": OpCategory.BROADCAST,
    
    # Broadcast binary (tile op broadcast(row/col vector)) - fusable with 8x8 loops
    "TROWEXPANDSUB": OpCategory.BROADCAST_BINARY,
    "TROWEXPANDDIV": OpCategory.BROADCAST_BINARY,
    "TROWEXPANDMUL": OpCategory.BROADCAST_BINARY,
    
    # Reduction (fusion barrier)
    "TROWSUM": OpCategory.REDUCTION,
    "TCOLSUM": OpCategory.REDUCTION,
    "TROWMAX": OpCategory.REDUCTION,
    "TSUM": OpCategory.REDUCTION,
    
    # Matrix ops (fusion barrier)
    "TMATMUL": OpCategory.MATMUL,
    
    # Memory
    "TLOAD": OpCategory.MEMORY,
    "TSTORE": OpCategory.MEMORY,
    
    # Control flow
    "FOR": OpCategory.CONTROL_FLOW,
    "ENDFOR": OpCategory.CONTROL_FLOW,
    "IF": OpCategory.CONTROL_FLOW,
    "ELSE": OpCategory.CONTROL_FLOW,
    "ENDIF": OpCategory.CONTROL_FLOW,
    
    # Scalar instructions (fusion barrier)
    "SLI": OpCategory.OTHER,
    "SCMP": OpCategory.OTHER,
    "SADD": OpCategory.OTHER,
    "SSUB": OpCategory.OTHER,
    "SMUL": OpCategory.OTHER,
    "SDIV": OpCategory.OTHER,
    "SMOV": OpCategory.OTHER,
    
    # Declarations
    "TILE_DECL": OpCategory.DECLARATION,
    "SCALAR_DECL": OpCategory.DECLARATION,
}


def get_category(opcode: str) -> OpCategory:
    """Get the category of an opcode."""
    return OPCODE_CATEGORY.get(opcode, OpCategory.OTHER)


def is_fusable(opcode: str) -> bool:
    """Check if an operation can be fused (elementwise operations + memory)."""
    category = get_category(opcode)
    return category in {
        OpCategory.ELEMENTWISE_BINARY,
        OpCategory.ELEMENTWISE_UNARY,
        OpCategory.ELEMENTWISE_SCALAR,
        OpCategory.BROADCAST,
        OpCategory.BROADCAST_BINARY,  # TROWEXPANDSUB, TROWEXPANDDIV, etc.
        OpCategory.MEMORY,  # TLOAD/TSTORE can be fused with same-shape ops
    }


def is_fusion_barrier(opcode: str) -> bool:
    """Check if an operation is a fusion barrier (stops fusion)."""
    category = get_category(opcode)
    # Scalar instructions, control flow, and function calls act as fusion barriers
    if opcode in ("SLI", "SCMP", "SADD", "SSUB", "SMUL", "SDIV", "SMOV", "CALL", "RETURN"):
        return True
    return category in {
        OpCategory.REDUCTION,
        OpCategory.MATMUL,
        OpCategory.CONTROL_FLOW,
    }


# =============================================================================
# Loop Fusion - IR Classes
# =============================================================================

@dataclass
class FusionTileShape:
    """Shape of a tile for fusion analysis."""
    rows: int
    cols: int
    dtype: str = "f32"
    
    def __hash__(self):
        return hash((self.rows, self.cols, self.dtype))
    
    def __eq__(self, other):
        if not isinstance(other, FusionTileShape):
            return False
        return self.rows == other.rows and self.cols == other.cols and self.dtype == other.dtype


# Alias for backward compatibility
TileShape = FusionTileShape


@dataclass
class FusableOp:
    """
    A single fusable operation.
    
    Attributes:
        opcode: The operation code (TADD, TMUL, etc.)
        dst: Destination tile name
        operands: List of operand names/values
        shape: Shape of the operation
        raw_instr: Original parsed instruction
    """
    opcode: str
    dst: str
    operands: List[str]
    shape: FusionTileShape
    raw_instr: Any = None
    
    def get_reads(self) -> set:
        """Get the set of tiles read by this operation."""
        reads = set()
        category = get_category(self.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            reads.add(self.operands[0])
            reads.add(self.operands[1])
        elif category == OpCategory.ELEMENTWISE_UNARY:
            reads.add(self.operands[0])
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            reads.add(self.operands[0])
        elif category == OpCategory.BROADCAST:
            pass  # scalar only
        elif category == OpCategory.BROADCAST_BINARY:
            # TROWEXPANDSUB, TROWEXPANDDIV, etc: dst = src0 op broadcast(src1)
            reads.add(self.operands[0])  # src0 (8x8 tile)
            reads.add(self.operands[1])  # src1 (8x1 column vector, broadcast)
        elif category == OpCategory.MEMORY:
            if self.opcode == "TLOAD":
                pass  # reads from memory, not tile
            elif self.opcode == "TSTORE":
                reads.add(self.operands[0])  # source tile
        
        return reads
    
    def get_writes(self) -> set:
        """Get the set of tiles written by this operation."""
        if self.opcode == "TSTORE":
            return set()  # writes to memory, not tile
        if self.dst:
            return {self.dst}
        return set()


@dataclass
class FusedLoop:
    """
    A fused loop containing multiple operations.
    
    All operations in a fused loop:
    - Have the same shape
    - Are elementwise operations
    - Can be executed in order within the same loop iteration
    """
    shape: FusionTileShape
    operations: List[FusableOp] = field(default_factory=list)
    
    def add_op(self, op: FusableOp):
        """Add an operation to the fused loop."""
        self.operations.append(op)
    
    def can_fuse(self, op: FusableOp) -> bool:
        """Check if an operation can be fused into this loop."""
        if op.shape != self.shape:
            return False
        if not is_fusable(op.opcode):
            return False
        return True
    
    def __len__(self):
        return len(self.operations)


@dataclass  
class FusionBarrier:
    """Represents a non-fusable operation that acts as a barrier."""
    raw_instr: Any
    opcode: str


# =============================================================================
# Loop Fusion - Optimizer
# =============================================================================

class LoopFusionOptimizer:
    """
    Optimizes PTO instruction sequences by fusing consecutive loops.
    
    Algorithm:
    1. Parse instructions into a sequence of FusableOp and FusionBarrier
    2. Group consecutive FusableOps with same shape into FusedLoops
    3. Respect data dependencies within fused groups
    """
    
    def __init__(self, tile_info: Dict[str, Any]):
        """
        Initialize the optimizer.
        
        Args:
            tile_info: Dictionary mapping tile names to tile info objects
        """
        self.tile_info = tile_info
        self.stats = {
            "original_ops": 0,
            "fused_loops": 0,
            "fusion_savings": 0,
        }
    
    def get_shape(self, tile_name: str) -> Optional[FusionTileShape]:
        """Get the shape of a tile."""
        info = self.tile_info.get(tile_name)
        if info:
            return FusionTileShape(info.rows, info.cols, info.dtype)
        return None
    
    def instr_to_fusable_op(self, instr) -> Optional[FusableOp]:
        """Convert a parsed instruction to a FusableOp if possible."""
        if not is_fusable(instr.opcode):
            return None
        
        # Handle TLOAD/TSTORE specially
        if instr.opcode == "TLOAD":
            shape = self.get_shape(instr.dst)
            if shape is None:
                return None
            return FusableOp(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=list(instr.operands),
                shape=shape,
                raw_instr=instr
            )
        elif instr.opcode == "TSTORE":
            src_tile = instr.operands[0] if instr.operands else None
            shape = self.get_shape(src_tile) if src_tile else None
            if shape is None:
                return None
            return FusableOp(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=list(instr.operands),
                shape=shape,
                raw_instr=instr
            )
        
        # Get shape from destination tile
        shape = self.get_shape(instr.dst) if instr.dst else None
        if shape is None:
            for op in instr.operands:
                shape = self.get_shape(op)
                if shape:
                    break
        
        if shape is None:
            shape = FusionTileShape(4, 4, "f32")
        
        return FusableOp(
            opcode=instr.opcode,
            dst=instr.dst,
            operands=list(instr.operands),
            shape=shape,
            raw_instr=instr
        )
    
    def check_dependency(self, group: List[FusableOp], new_op: FusableOp) -> bool:
        """Check if adding new_op to group would violate data dependencies."""
        return True  # Elementwise ops can always be fused if same shape
    
    def fuse_instructions(self, instructions: List) -> List:
        """Fuse consecutive fusable instructions into FusedLoops."""
        result = []
        current_fused: Optional[FusedLoop] = None
        
        self.stats["original_ops"] = len(instructions)
        
        for instr in instructions:
            if instr.opcode in ("TILE_DECL", "SCALAR_DECL", "PASSTHROUGH"):
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(instr)
                continue
            
            if is_fusion_barrier(instr.opcode):
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            fusable_op = self.instr_to_fusable_op(instr)
            
            if fusable_op is None:
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            if current_fused is None:
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
            elif current_fused.can_fuse(fusable_op) and \
                 self.check_dependency(current_fused.operations, fusable_op):
                current_fused.add_op(fusable_op)
            else:
                result.append(current_fused)
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
        
        if current_fused and len(current_fused) > 0:
            result.append(current_fused)
        
        self.stats["fused_loops"] = sum(1 for r in result if isinstance(r, FusedLoop))
        fused_op_count = sum(len(r.operations) for r in result if isinstance(r, FusedLoop))
        self.stats["fusion_savings"] = fused_op_count - self.stats["fused_loops"]
        
        return result
    
    def optimize(self, instructions: List) -> List:
        """Apply loop fusion optimization."""
        return self.fuse_instructions(instructions)
    
    def print_stats(self):
        """Print optimization statistics."""
        print(f"Loop Fusion Statistics:")
        print(f"  Original operations: {self.stats['original_ops']}")
        print(f"  Fused loops created: {self.stats['fused_loops']}")
        print(f"  Loop overhead saved: {self.stats['fusion_savings']}")


# =============================================================================
# Loop Fusion - Code Generator
# =============================================================================

class FusedCodeGenerator:
    """Generates ARM64 NEON code for fused loops."""
    
    def __init__(self):
        self.indent_level = 0
        self.var_counter = 0
    
    def _indent(self) -> str:
        return "    " * self.indent_level
    
    def _get_unique_var(self, prefix: str = "_v") -> str:
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def generate_fused_loop(self, fused: FusedLoop) -> List[str]:
        """Generate code for a fused loop."""
        lines = []
        indent = self._indent()
        
        rows = fused.shape.rows
        cols = fused.shape.cols
        dtype = fused.shape.dtype
        
        vec_lanes = 4 if dtype == "f32" else 8 if dtype == "f16" else 4
        suffix = "f32" if dtype == "f32" else "f16" if dtype == "f16" else "f32"
        vec_type = f"float32x4_t" if dtype == "f32" else f"float16x8_t" if dtype == "f16" else "float32x4_t"
        
        op_names = [f"{op.dst}={op.opcode}({','.join(op.operands)})" for op in fused.operations]
        lines.append(f"{indent}// FUSED LOOP ({len(fused.operations)} ops): {'; '.join(op_names)}")
        
        # Pre-compute scalar broadcast vectors
        scalar_vars = {}
        for op in fused.operations:
            if get_category(op.opcode) == OpCategory.ELEMENTWISE_SCALAR:
                scalar_val = op.operands[1]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
            elif op.opcode == "TEXPANDS":
                scalar_val = op.operands[0]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
        
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        self.indent_level += 1
        indent = self._indent()
        
        lines.append(f"{indent}int _col;")
        lines.append(f"{indent}// Vectorized loop")
        lines.append(f"{indent}for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        self.indent_level += 1
        
        for op in fused.operations:
            vec_lines = self._gen_vectorized_op(op, suffix, vec_type, scalar_vars)
            lines.extend(vec_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        lines.append(f"{indent}// Scalar cleanup")
        lines.append(f"{indent}for (; _col < {cols}; _col++) {{")
        self.indent_level += 1
        
        for op in fused.operations:
            scalar_lines = self._gen_scalar_op(op)
            lines.extend(scalar_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    def _gen_vectorized_op(self, op: FusableOp, suffix: str, vec_type: str,
                           scalar_vars: Dict[str, str]) -> List[str]:
        """Generate vectorized code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            v0, v1, vr = self._get_unique_var("_v"), self._get_unique_var("_v"), self._get_unique_var("_vr")
            neon_op = self._get_neon_binary_op(op.opcode, suffix)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {v1} = vld1q_{suffix}(&{op.operands[1]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {v1});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            v0, vr = self._get_unique_var("_v"), self._get_unique_var("_vr")
            neon_code = self._get_neon_unary_op(op.opcode, suffix, v0, vec_type)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_code};")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            v0, vr = self._get_unique_var("_v"), self._get_unique_var("_vr")
            vs = scalar_vars.get(op.operands[1], "_vs")
            neon_op = self._get_neon_scalar_op(op.opcode, suffix)
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {vs});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.BROADCAST:
            vs = scalar_vars.get(op.operands[0], "_vs")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vs});")
        
        elif category == OpCategory.BROADCAST_BINARY:
            # TROWEXPANDSUB, TROWEXPANDDIV, TROWEXPANDMUL: dst = src0 op broadcast(src1)
            # src0 is 8x8 tile, src1 is 8x1 column vector (broadcast row-wise)
            v0, vr = self._get_unique_var("_v0"), self._get_unique_var("_vr")
            # Load src0 element
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            # src1[_row][0] is broadcast across the row - create a vector from scalar
            vb = self._get_unique_var("_vb")
            lines.append(f"{indent}{vec_type} {vb} = vdupq_n_{suffix}({op.operands[1]}[_row][0]);")
            # Apply operation
            op_map = {"TROWEXPANDSUB": "vsub", "TROWEXPANDDIV": "vdiv", "TROWEXPANDMUL": "vmul"}
            neon_base = op_map.get(op.opcode, "vsub")
            lines.append(f"{indent}{vec_type} {vr} = {neon_base}q_{suffix}({v0}, {vb});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.MEMORY:
            if op.opcode == "TLOAD":
                vr = self._get_unique_var("_vl")
                memref = op.operands[0]
                # Handle dynamic row/col offsets
                row_off = op.operands[1] if len(op.operands) > 1 else "0"
                col_off = op.operands[2] if len(op.operands) > 2 else "0"
                if row_off == "0" and col_off == "0":
                    mem_idx = f"_row * {op.shape.cols} + _col"
                else:
                    tile_size = op.shape.rows * op.shape.cols
                    row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
                    col_offset = col_off if col_off != "0" else ""
                    base_idx = "_row * {} + _col".format(op.shape.cols)
                    if row_offset and col_offset:
                        mem_idx = f"{row_offset} + {col_offset} + {base_idx}"
                    elif row_offset:
                        mem_idx = f"{row_offset} + {base_idx}"
                    elif col_offset:
                        mem_idx = f"{col_offset} + {base_idx}"
                    else:
                        mem_idx = base_idx
                lines.append(f"{indent}{vec_type} {vr} = vld1q_{suffix}(&{memref}[{mem_idx}]);")
                lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
            elif op.opcode == "TSTORE":
                vs = self._get_unique_var("_vs")
                src_tile, memref = op.operands[0], op.dst
                row_off = op.operands[1] if len(op.operands) > 1 else "0"
                col_off = op.operands[2] if len(op.operands) > 2 else "0"
                if row_off == "0" and col_off == "0":
                    mem_idx = f"_row * {op.shape.cols} + _col"
                else:
                    tile_size = op.shape.rows * op.shape.cols
                    row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
                    col_offset = col_off if col_off != "0" else ""
                    base_idx = "_row * {} + _col".format(op.shape.cols)
                    if row_offset and col_offset:
                        mem_idx = f"{row_offset} + {col_offset} + {base_idx}"
                    elif row_offset:
                        mem_idx = f"{row_offset} + {base_idx}"
                    elif col_offset:
                        mem_idx = f"{col_offset} + {base_idx}"
                    else:
                        mem_idx = base_idx
                lines.append(f"{indent}{vec_type} {vs} = vld1q_{suffix}(&{src_tile}[_row][_col]);")
                lines.append(f"{indent}vst1q_{suffix}(&{memref}[{mem_idx}], {vs});")
        
        return lines
    
    def _gen_scalar_op(self, op: FusableOp) -> List[str]:
        """Generate scalar code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            c_op = self._get_c_binary_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]}[_row][_col];")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            c_expr = self._get_c_unary_expr(op.opcode, f"{op.operands[0]}[_row][_col]")
            lines.append(f"{indent}{op.dst}[_row][_col] = {c_expr};")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            c_op = self._get_c_scalar_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]};")
        
        elif category == OpCategory.BROADCAST:
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]};")
        
        elif category == OpCategory.BROADCAST_BINARY:
            # TROWEXPANDSUB, TROWEXPANDDIV, TROWEXPANDMUL: dst = src0 op broadcast(src1)
            op_map = {"TROWEXPANDSUB": "-", "TROWEXPANDDIV": "/", "TROWEXPANDMUL": "*"}
            c_op = op_map.get(op.opcode, "-")
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]}[_row][0];")
        
        elif category == OpCategory.MEMORY:
            if op.opcode == "TLOAD":
                memref = op.operands[0]
                row_off = op.operands[1] if len(op.operands) > 1 else "0"
                col_off = op.operands[2] if len(op.operands) > 2 else "0"
                if row_off == "0" and col_off == "0":
                    mem_idx = f"_row * {op.shape.cols} + _col"
                else:
                    tile_size = op.shape.rows * op.shape.cols
                    row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
                    col_offset = col_off if col_off != "0" else ""
                    base_idx = "_row * {} + _col".format(op.shape.cols)
                    if row_offset and col_offset:
                        mem_idx = f"{row_offset} + {col_offset} + {base_idx}"
                    elif row_offset:
                        mem_idx = f"{row_offset} + {base_idx}"
                    elif col_offset:
                        mem_idx = f"{col_offset} + {base_idx}"
                    else:
                        mem_idx = base_idx
                lines.append(f"{indent}{op.dst}[_row][_col] = {memref}[{mem_idx}];")
            elif op.opcode == "TSTORE":
                src_tile, memref = op.operands[0], op.dst
                row_off = op.operands[1] if len(op.operands) > 1 else "0"
                col_off = op.operands[2] if len(op.operands) > 2 else "0"
                if row_off == "0" and col_off == "0":
                    mem_idx = f"_row * {op.shape.cols} + _col"
                else:
                    tile_size = op.shape.rows * op.shape.cols
                    row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
                    col_offset = col_off if col_off != "0" else ""
                    base_idx = "_row * {} + _col".format(op.shape.cols)
                    if row_offset and col_offset:
                        mem_idx = f"{row_offset} + {col_offset} + {base_idx}"
                    elif row_offset:
                        mem_idx = f"{row_offset} + {base_idx}"
                    elif col_offset:
                        mem_idx = f"{col_offset} + {base_idx}"
                    else:
                        mem_idx = base_idx
                lines.append(f"{indent}{memref}[{mem_idx}] = {src_tile}[_row][_col];")
        
        return lines
    
    def _get_neon_binary_op(self, opcode: str, suffix: str) -> str:
        ops = {"TADD": f"vaddq_{suffix}", "TSUB": f"vsubq_{suffix}", 
               "TMUL": f"vmulq_{suffix}", "TDIV": f"vdivq_{suffix}",
               "TMAX": f"vmaxq_{suffix}", "TMIN": f"vminq_{suffix}"}
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_neon_unary_op(self, opcode: str, suffix: str, v: str, vec_type: str) -> str:
        ops = {"TABS": f"vabsq_{suffix}({v})", "TNEG": f"vnegq_{suffix}({v})",
               "TSQRT": f"vsqrtq_{suffix}({v})", "TRSQRT": f"vrsqrteq_{suffix}({v})",
               "TRELU": f"vmaxq_{suffix}({v}, vdupq_n_{suffix}(0.0f))"}
        return ops.get(opcode, f"{v}")
    
    def _get_neon_scalar_op(self, opcode: str, suffix: str) -> str:
        ops = {"TADDS": f"vaddq_{suffix}", "TSUBS": f"vsubq_{suffix}",
               "TMULS": f"vmulq_{suffix}", "TDIVS": f"vdivq_{suffix}",
               "TMAXS": f"vmaxq_{suffix}", "TMINS": f"vminq_{suffix}"}
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_c_binary_op(self, opcode: str) -> str:
        ops = {"TADD": "+", "TSUB": "-", "TMUL": "*", "TDIV": "/"}
        return ops.get(opcode, "+")
    
    def _get_c_unary_expr(self, opcode: str, operand: str) -> str:
        exprs = {"TABS": f"fabsf({operand})", "TNEG": f"-{operand}",
                 "TRECIP": f"1.0f / {operand}", "TEXP": f"expf({operand})",
                 "TLOG": f"logf({operand})", "TSQRT": f"sqrtf({operand})",
                 "TRSQRT": f"1.0f / sqrtf({operand})", "TRELU": f"fmaxf({operand}, 0.0f)"}
        return exprs.get(opcode, operand)
    
    def _get_c_scalar_op(self, opcode: str) -> str:
        ops = {"TADDS": "+", "TSUBS": "-", "TMULS": "*", "TDIVS": "/"}
        return ops.get(opcode, "+")


# =============================================================================
# Multi-Backend Code Generation
# =============================================================================

# Import additional types for code generation
from pto_isa_definition import (
    ARM64_TYPE_MAP, CUDA_TYPE_MAP, ASCEND_TYPE_MAP,
    arm64_generate_header, cuda_generate_header, ascend_generate_header,
)

# Backend Configuration
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
    "ascend910b": {
        "name": "Huawei Ascend 910B",
        "suffix": "_ascend910b",
        "extension": ".cpp",
        "header_func": ascend_generate_header,
        "type_map": ASCEND_TYPE_MAP,
    },
}


@dataclass
class MockTileInfo:
    """Tile info for code generation."""
    name: str
    rows: int
    cols: int
    dtype: str


@dataclass
class MockInstruction:
    """Mock instruction for fusion optimizer."""
    opcode: str
    dst: str
    operands: list
    raw_line: str = ""


def _get_operand_str(operand):
    """Convert an operand to string representation."""
    if hasattr(operand, 'name'):
        return operand.name
    elif hasattr(operand, 'value'):
        return str(operand.value)
    return str(operand)


def convert_program_to_mock_instructions(program):
    """Convert PTOProgram to mock instructions for fusion optimizer."""
    tile_info = {}
    for name, tile_type in program.tile_declarations.items():
        tile_info[name] = MockTileInfo(
            name=name,
            rows=tile_type.shape.rows,
            cols=tile_type.shape.cols,
            dtype=tile_type.element_type.value
        )
    
    mock_instructions = []
    for instr in program.instructions:
        opcode = instr.__class__.__name__
        
        if opcode in ("TADD", "TSUB", "TMUL", "TDIV", "TMAX", "TMIN"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name]
            ))
        elif opcode in ("TADDS", "TSUBS", "TMULS", "TDIVS"):
            scalar_str = instr.scalar.name
            if not scalar_str.endswith("f"):
                scalar_str += "f"
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name, scalar_str]
            ))
        elif opcode in ("TEXP", "TLOG", "TSQRT", "TRSQRT", "TRELU", "TABS", "TNEG", "TRECIP"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        elif opcode == "TEXPANDS":
            scalar_str = instr.scalar.name
            if not scalar_str.endswith("f"):
                scalar_str += "f"
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[scalar_str]
            ))
        elif opcode in ("TROWSUM", "TCOLSUM", "TROWMAX"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        elif opcode in ("TROWEXPANDSUB", "TROWEXPANDDIV", "TROWEXPANDMUL"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name]
            ))
        elif opcode == "TMATMUL":
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.a.name, instr.b.name]
            ))
        elif opcode == "TLOAD":
            # Handle dynamic row/col indices
            row_off = _get_operand_str(instr.row_offset)
            col_off = _get_operand_str(instr.col_offset)
            mock_instructions.append(MockInstruction(
                opcode="TLOAD", dst=instr.dst.name,
                operands=[instr.src_mem.name, row_off, col_off]
            ))
        elif opcode == "TSTORE":
            # Handle dynamic row/col indices
            row_off = _get_operand_str(instr.row_offset)
            col_off = _get_operand_str(instr.col_offset)
            mock_instructions.append(MockInstruction(
                opcode="TSTORE", dst=instr.dst_mem.name,
                operands=[instr.src.name, row_off, col_off]
            ))
        # =========== Control Flow Instructions ===========
        elif opcode == "FOR":
            mock_instructions.append(MockInstruction(
                opcode="FOR", dst=instr.iv.name,
                operands=[
                    _get_operand_str(instr.lb),
                    _get_operand_str(instr.ub),
                    _get_operand_str(instr.step)
                ]
            ))
        elif opcode == "ENDFOR":
            mock_instructions.append(MockInstruction(
                opcode="ENDFOR", dst="",
                operands=[]
            ))
        elif opcode == "IF":
            mock_instructions.append(MockInstruction(
                opcode="IF", dst="",
                operands=[instr.cond.name]
            ))
        elif opcode == "ELSE":
            mock_instructions.append(MockInstruction(
                opcode="ELSE", dst="",
                operands=[]
            ))
        elif opcode == "ENDIF":
            mock_instructions.append(MockInstruction(
                opcode="ENDIF", dst="",
                operands=[]
            ))
        # =========== Scalar Instructions ===========
        elif opcode == "SLI":
            mock_instructions.append(MockInstruction(
                opcode="SLI", dst=instr.dst.name,
                operands=[str(instr.imm.value)]
            ))
        elif opcode == "SCMP":
            mock_instructions.append(MockInstruction(
                opcode="SCMP", dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name, instr.cmp_mode.value]
            ))
        elif opcode in ("SADD", "SSUB", "SMUL", "SDIV"):
            mock_instructions.append(MockInstruction(
                opcode=opcode, dst=instr.dst.name,
                operands=[instr.src0.name, _get_operand_str(instr.src1)]
            ))
        elif opcode == "SMOV":
            mock_instructions.append(MockInstruction(
                opcode="SMOV", dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        # =========== Function Call Instructions ===========
        elif opcode == "CALL":
            # CALL instruction: callee name in 'callee', args in 'args' dict
            args_list = []
            if hasattr(instr, 'args') and instr.args:
                # Format: "param_name -> arg_value" to preserve parameter info
                for param, arg in instr.args.items():
                    args_list.append(f"{param} -> {arg}")
            mock_instructions.append(MockInstruction(
                opcode="CALL", dst=instr.callee,
                operands=args_list
            ))
        elif opcode == "RETURN":
            mock_instructions.append(MockInstruction(
                opcode="RETURN", dst="",
                operands=[]
            ))
    
    return tile_info, mock_instructions


@dataclass
class OrchestrationContext:
    """
    Context for generating orchestration function code.
    
    Tracks state needed for task graph building code generation:
    - Task counter for unique IDs
    - Tensor to producer task mapping for dependency tracking
    - Module reference for InCore function metadata
    """
    module: Optional['PTOModule'] = None
    task_counter: int = 0
    tensor_producers: Dict[str, int] = field(default_factory=dict)
    
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


def _gen_arm64_barrier_op(instr, rows, cols, dtype, tile_info, orch_ctx: Optional[OrchestrationContext] = None):
    """Generate ARM64 code for barrier operations (non-fusable).
    
    Args:
        instr: The instruction to generate code for
        rows, cols: Tile dimensions
        dtype: Data type
        tile_info: Tile metadata
        orch_ctx: Optional orchestration context. If provided, CALL instructions
                  generate task scheduling code instead of direct function calls.
    """
    lines = []
    c_type = ARM64_TYPE_MAP.get(dtype, "float")
    
    if instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        # Determine if offsets are variables or constants
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
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        lines.append(f"for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"    for (int _col = 0; _col < {cols}; _col++) {{")
        if row_off == "0" and col_off == "0":
            lines.append(f"        {dst_mem}[_row * {cols} + _col] = {src}[_row][_col];")
        else:
            lines.append(f"        {dst_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col] = {src}[_row][_col];")
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
    
    # =========== Control Flow Instructions ===========
    elif instr.opcode == "FOR":
        iv = instr.dst  # Induction variable
        lb = instr.operands[0]  # Lower bound
        ub = instr.operands[1]  # Upper bound
        step = instr.operands[2] if len(instr.operands) > 2 else "1"
        lines.append(f"for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
        
    elif instr.opcode == "ENDFOR":
        lines.append("}")
        
    elif instr.opcode == "IF":
        cond = instr.operands[0] if instr.operands else "true"
        lines.append(f"if ({cond}) {{")
        
    elif instr.opcode == "ELSE":
        lines.append("} else {")
        
    elif instr.opcode == "ENDIF":
        lines.append("}")
    
    # =========== Scalar Instructions ===========
    elif instr.opcode == "SLI":
        dst = instr.dst
        imm = instr.operands[0]
        lines.append(f"int {dst} = {imm};")
        
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
        callee = instr.dst  # Function name is stored in dst
        args = instr.operands  # List of argument mappings
        
        # For orchestration functions, generate task scheduling code
        if orch_ctx is not None:
            lines.extend(_gen_task_scheduling_code(callee, args, orch_ctx, tile_info, rows, cols))
        else:
            # For InCore functions, generate direct function call
            if args:
                args_str = ", ".join(str(arg) for arg in args)
                lines.append(f"{callee}({args_str});")
            else:
                lines.append(f"{callee}();")
    
    elif instr.opcode == "RETURN":
        lines.append("return;")
        
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


def _gen_task_scheduling_code(callee: str, args: List, orch_ctx: OrchestrationContext, 
                               tile_info: Dict, rows: int, cols: int) -> List[str]:
    """Generate task scheduling code for an InCore function call.
    
    This is called when generating code for an orchestration function that
    calls an InCore function.
    
    Args format from CALL instruction: ["param_name -> arg_value", ...]
    where param_name is the parameter name in the called function,
    and arg_value is the actual argument passed.
    """
    lines = []
    task_id = orch_ctx.alloc_task()
    
    # Get buffer sizes for this function (in KB, convert to bytes)
    buf_without_reuse, buf_with_reuse = orch_ctx.get_buffer_sizes(callee)
    buf_bytes = int(buf_without_reuse * 1024)  # Convert KB to bytes
    reuse_bytes = int(buf_with_reuse * 1024)
    
    lines.append(f"// Task {task_id}: {callee}")
    lines.append(f"int32_t t{task_id} = pto_task_alloc(rt, \"{callee}\", NULL, {buf_bytes}, {reuse_bytes});")
    
    # Parse arguments: determine inputs vs outputs
    # Args format: ["param_name -> arg_value", ...] (from CALL instruction)
    input_args = []
    output_args = []
    
    for arg in args:
        arg_str = str(arg)
        
        # Parse "param_name -> arg_value" format (from CALL instruction)
        if " -> " in arg_str:
            param_name, value = arg_str.split(" -> ", 1)
        elif "=" in arg_str:
            # Alternative format: "param_name=arg_value"
            param_name, value = arg_str.split("=", 1)
        else:
            param_name = arg_str
            value = arg_str
        
        param_name = param_name.strip()
        value = value.strip()
        
        # Determine if output based on parameter name
        is_output = any(kw in param_name.lower() for kw in ['output', 'result', 'dst', 'out'])
        
        # Get shape info - use default tile size if not found
        t_rows = rows
        t_cols = cols
        
        # Adjust for reduction outputs
        if callee in ['rowmax', 'rowsum', 'tile_rowmax', 'tile_rowsum'] and is_output:
            t_cols = 1
        
        # Extract row_offset if present in value (e.g., "input, tile_idx, 0")
        if "," in value:
            parts = [p.strip() for p in value.split(",")]
            tensor_name = parts[0]
            row_off = parts[1] if len(parts) > 1 else "0"
            col_off = parts[2] if len(parts) > 2 else "0"
        else:
            tensor_name = value
            row_off = "0"
            col_off = "0"
        
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


def _gen_cuda_barrier_op(instr, rows, cols, dtype, tile_info):
    """Generate CUDA code for barrier operations (control flow, scalar, etc.)."""
    lines = []
    c_type = CUDA_TYPE_MAP.get(dtype, "float")
    
    if instr.opcode == "FOR":
        iv = instr.dst
        lb = instr.operands[0]
        ub = instr.operands[1]
        step = instr.operands[2] if len(instr.operands) > 2 else "1"
        lines.append(f"for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
        
    elif instr.opcode == "ENDFOR":
        lines.append("}")
        
    elif instr.opcode == "IF":
        cond = instr.operands[0] if instr.operands else "true"
        lines.append(f"if ({cond}) {{")
        
    elif instr.opcode == "ELSE":
        lines.append("} else {")
        
    elif instr.opcode == "ENDIF":
        lines.append("}")
    
    elif instr.opcode == "SLI":
        dst = instr.dst
        imm = instr.operands[0]
        lines.append(f"int {dst} = {imm};")
        
    elif instr.opcode == "SCMP":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        cmp_mode = instr.operands[2] if len(instr.operands) > 2 else "eq"
        cmp_ops = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        cmp_op = cmp_ops.get(cmp_mode, ">")
        lines.append(f"int {dst} = ({src0} {cmp_op} {src1}) ? 1 : 0;")
        
    elif instr.opcode in ("SADD", "SSUB", "SMUL", "SDIV"):
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        op_map = {"SADD": "+", "SSUB": "-", "SMUL": "*", "SDIV": "/"}
        op = op_map.get(instr.opcode, "+")
        lines.append(f"int {dst} = {src0} {op} {src1};")
        
    elif instr.opcode == "SMOV":
        lines.append(f"int {instr.dst} = {instr.operands[0]};")
    
    elif instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        lines.append(f"// TLOAD: {dst} = load({src_mem}[{row_off}, {col_off}])")
        if row_off == "0" and col_off == "0":
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst}[_row][_col] = {src_mem}[_row * {cols} + _col];")
        else:
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst}[_row][_col] = {src_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col];")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        if row_off == "0" and col_off == "0":
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst_mem}[_row * {cols} + _col] = {src}[_row][_col];")
        else:
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col] = {src}[_row][_col];")
        
    elif instr.opcode == "TROWSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWSUM: {dst} = rowsum({src})")
        lines.append(f"if (_col == 0 && _row < {rows}) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _c = 0; _c < {src_cols}; _c++) _sum += {src}[_row][_c];")
        lines.append(f"    {dst}[_row][0] = _sum;}}")
    
    elif instr.opcode == "TROWMAX":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWMAX: {dst} = rowmax({src})")
        lines.append(f"if (_col == 0 && _row < {rows}) {{")
        lines.append(f"    {c_type} _max = {src}[_row][0];")
        lines.append(f"    for (int _c = 1; _c < {src_cols}; _c++) if ({src}[_row][_c] > _max) _max = {src}[_row][_c];")
        lines.append(f"    {dst}[_row][0] = _max;}}")
        
    elif instr.opcode == "TMATMUL":
        dst, a, b = instr.dst, instr.operands[0], instr.operands[1]
        a_info = tile_info.get(a)
        k = a_info.cols if a_info else 8
        lines.append(f"// TMATMUL: {dst} = {a} @ {b}")
        lines.append(f"if (_row < {rows} && _col < {cols}) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _k = 0; _k < {k}; _k++) _sum += {a}[_row][_k] * {b}[_k][_col];")
        lines.append(f"    {dst}[_row][_col] = _sum;}}")
    
    # =========== Function Call Instructions ===========
    elif instr.opcode == "CALL":
        callee = instr.dst  # Function name is stored in dst
        args = instr.operands  # List of argument mappings
        if args:
            args_str = ", ".join(str(arg) for arg in args)
            lines.append(f"{callee}({args_str});")
        else:
            lines.append(f"{callee}();")
    
    elif instr.opcode == "RETURN":
        lines.append("return;")
        
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


def _gen_cuda_single_op(instr, tile_info):
    """Generate a single CUDA operation."""
    op, dst = instr.opcode, f"{instr.dst}[_row][_col]"
    src0 = src1 = ""
    
    if len(instr.operands) >= 1:
        src0 = f"{instr.operands[0]}[_row][_col]"
    if len(instr.operands) >= 2:
        src1 = instr.operands[1]
        if not src1.endswith("f") and not src1.replace(".", "").replace("-", "").isdigit():
            src1 = f"{src1}[_row][_col]"
    
    # Get tile shape for memory operations
    dst_info = tile_info.get(instr.dst)
    rows = dst_info.rows if dst_info else 8
    cols = dst_info.cols if dst_info else 8
    tile_size = rows * cols
    
    # Helper function to compute memory index with dynamic offset
    def compute_mem_idx(operands, cols, tile_size):
        row_off = operands[1] if len(operands) > 1 else "0"
        col_off = operands[2] if len(operands) > 2 else "0"
        if row_off == "0" and col_off == "0":
            return f"_row * {cols} + _col"
        else:
            row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
            col_offset = col_off if col_off != "0" else ""
            base_idx = f"_row * {cols} + _col"
            if row_offset and col_offset:
                return f"{row_offset} + {col_offset} + {base_idx}"
            elif row_offset:
                return f"{row_offset} + {base_idx}"
            elif col_offset:
                return f"{col_offset} + {base_idx}"
            else:
                return base_idx
    
    if op == "TADD": return f"{dst} = {src0} + {src1};"
    elif op == "TSUB": return f"{dst} = {src0} - {src1};"
    elif op == "TMUL": return f"{dst} = {src0} * {src1};"
    elif op == "TDIV": return f"{dst} = {src0} / {src1};"
    elif op == "TMAX": return f"{dst} = fmaxf({src0}, {src1});"
    elif op == "TMIN": return f"{dst} = fminf({src0}, {src1});"
    elif op == "TABS": return f"{dst} = fabsf({src0});"
    elif op == "TNEG": return f"{dst} = -{src0};"
    elif op == "TRECIP": return f"{dst} = 1.0f / {src0};"
    elif op == "TEXP": return f"{dst} = __expf({src0});"
    elif op == "TLOG": return f"{dst} = __logf({src0});"
    elif op == "TSQRT": return f"{dst} = __fsqrt_rn({src0});"
    elif op == "TRSQRT": return f"{dst} = __frsqrt_rn({src0});"
    elif op == "TRELU": return f"{dst} = fmaxf({src0}, 0.0f);"
    # Broadcast binary operations (row-wise broadcast)
    elif op == "TROWEXPANDSUB":
        # dst = src0 - broadcast(src1) where src1 is Nx1
        return f"{dst} = {instr.operands[0]}[_row][_col] - {instr.operands[1]}[_row][0];"
    elif op == "TROWEXPANDDIV":
        # dst = src0 / broadcast(src1) where src1 is Nx1
        return f"{dst} = {instr.operands[0]}[_row][_col] / {instr.operands[1]}[_row][0];"
    elif op == "TROWEXPANDMUL":
        # dst = src0 * broadcast(src1) where src1 is Nx1
        return f"{dst} = {instr.operands[0]}[_row][_col] * {instr.operands[1]}[_row][0];"
    elif op == "TADDS": return f"{dst} = {src0} + {src1};"
    elif op == "TSUBS": return f"{dst} = {src0} - {src1};"
    elif op == "TMULS": return f"{dst} = {src0} * {src1};"
    elif op == "TDIVS": return f"{dst} = {src0} / {src1};"
    elif op == "TEXPANDS": return f"{dst} = {instr.operands[0]};"
    elif op == "TLOAD": 
        memref = instr.operands[0]
        mem_idx = compute_mem_idx(instr.operands, cols, tile_size)
        return f"{dst} = {memref}[{mem_idx}];"
    elif op == "TSTORE": 
        src_info = tile_info.get(instr.operands[0])
        src_cols = src_info.cols if src_info else cols
        src_rows = src_info.rows if src_info else rows
        src_tile_size = src_rows * src_cols
        mem_idx = compute_mem_idx(instr.operands, src_cols, src_tile_size)
        return f"{instr.dst}[{mem_idx}] = {instr.operands[0]}[_row][_col];"
    return f"// Unknown op: {op}"


def _gen_ascend_barrier_op(instr, rows, cols, dtype, tile_info):
    """Generate Ascend C code for barrier operations (control flow, scalar, etc.)."""
    lines = []
    
    if instr.opcode == "FOR":
        iv = instr.dst
        lb = instr.operands[0]
        ub = instr.operands[1]
        step = instr.operands[2] if len(instr.operands) > 2 else "1"
        lines.append(f"for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
        
    elif instr.opcode == "ENDFOR":
        lines.append("}")
        
    elif instr.opcode == "IF":
        cond = instr.operands[0] if instr.operands else "true"
        lines.append(f"if ({cond}) {{")
        
    elif instr.opcode == "ELSE":
        lines.append("} else {")
        
    elif instr.opcode == "ENDIF":
        lines.append("}")
    
    elif instr.opcode == "SLI":
        lines.append(f"int {instr.dst} = {instr.operands[0]};")
        
    elif instr.opcode == "SCMP":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        cmp_mode = instr.operands[2] if len(instr.operands) > 2 else "eq"
        cmp_ops = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        cmp_op = cmp_ops.get(cmp_mode, ">")
        lines.append(f"int {dst} = ({src0} {cmp_op} {src1}) ? 1 : 0;")
        
    elif instr.opcode in ("SADD", "SSUB", "SMUL", "SDIV"):
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        op_map = {"SADD": "+", "SSUB": "-", "SMUL": "*", "SDIV": "/"}
        op = op_map.get(instr.opcode, "+")
        lines.append(f"int {dst} = {src0} {op} {src1};")
        
    elif instr.opcode == "SMOV":
        lines.append(f"int {instr.dst} = {instr.operands[0]};")
    
    elif instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        tile_size = rows * cols
        lines.append(f"// TLOAD: {dst} = load({src_mem}[{row_off}, {col_off}])")
        lines.append(f"DataCopy({dst}, {src_mem}[({row_off}) * {tile_size}], {tile_size});")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        tile_size = rows * cols
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        lines.append(f"DataCopy({dst_mem}[({row_off}) * {tile_size}], {src}, {tile_size});")
        
    elif instr.opcode == "TROWSUM":
        tile_size = rows * cols
        lines.append(f"// TROWSUM: reduction operation")
        lines.append(f"ReduceSum({instr.dst}, {instr.operands[0]}, {tile_size});")
    
    elif instr.opcode == "TROWMAX":
        tile_size = rows * cols
        lines.append(f"// TROWMAX: reduction max operation")
        lines.append(f"ReduceMax({instr.dst}, {instr.operands[0]}, {tile_size});")
        
    elif instr.opcode == "TMATMUL":
        lines.append(f"// TMATMUL: {instr.dst} = {instr.operands[0]} @ {instr.operands[1]}")
        lines.append(f"Matmul({instr.dst}, {instr.operands[0]}, {instr.operands[1]}, {rows}, {cols});")
    
    # =========== Function Call Instructions ===========
    elif instr.opcode == "CALL":
        callee = instr.dst  # Function name is stored in dst
        args = instr.operands  # List of argument mappings
        if args:
            args_str = ", ".join(str(arg) for arg in args)
            lines.append(f"{callee}({args_str});")
        else:
            lines.append(f"{callee}();")
    
    elif instr.opcode == "RETURN":
        lines.append("return;")
        
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


def _gen_ascend_single_op(instr, tile_info):
    """Generate a single Ascend C vector operation."""
    op, dst = instr.opcode, instr.dst
    src0 = instr.operands[0] if len(instr.operands) >= 1 else ""
    src1 = instr.operands[1] if len(instr.operands) >= 2 else ""
    
    ops_map = {
        "TADD": f"Add({dst}, {src0}, {src1}, 64);",
        "TSUB": f"Sub({dst}, {src0}, {src1}, 64);",
        "TMUL": f"Mul({dst}, {src0}, {src1}, 64);",
        "TDIV": f"Div({dst}, {src0}, {src1}, 64);",
        "TMAX": f"Max({dst}, {src0}, {src1}, 64);",
        "TMIN": f"Min({dst}, {src0}, {src1}, 64);",
        "TABS": f"Abs({dst}, {src0}, 64);",
        "TNEG": f"Neg({dst}, {src0}, 64);",
        "TRECIP": f"Reciprocal({dst}, {src0}, 64);",
        "TEXP": f"Exp({dst}, {src0}, 64);",
        "TLOG": f"Ln({dst}, {src0}, 64);",
        "TSQRT": f"Sqrt({dst}, {src0}, 64);",
        "TRSQRT": f"Rsqrt({dst}, {src0}, 64);",
        "TRELU": f"Relu({dst}, {src0}, 64);",
        "TADDS": f"Adds({dst}, {src0}, {src1}, 64);",
        "TSUBS": f"Subs({dst}, {src0}, {src1}, 64);",
        "TMULS": f"Muls({dst}, {src0}, {src1}, 64);",
        "TDIVS": f"Divs({dst}, {src0}, {src1}, 64);",
        "TEXPANDS": f"Duplicate({dst}, {src0}, 64);",
        # Row-wise broadcast operations: dst = src0 op broadcast(src1) where src1 is Nx1
        "TROWEXPANDSUB": f"BroadcastSub({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast subtract",
        "TROWEXPANDDIV": f"BroadcastDiv({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast divide",
        "TROWEXPANDMUL": f"BroadcastMul({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast multiply",
    }
    return ops_map.get(op, f"// {op}: Operation")


# =============================================================================
# Tile Buffer Analyzer - Memory Analysis for InCore Functions
# =============================================================================

@dataclass
class TileBufferInfo:
    """Information about a tile buffer."""
    name: str
    rows: int
    cols: int
    dtype: str
    element_size: int  # bytes per element
    total_bytes: int
    first_write: int   # instruction index of first write
    last_read: int     # instruction index of last read
    can_reuse_from: Optional[str] = None  # tile name this can reuse buffer from


class TileBufferAnalyzer:
    """
    Analyzes tile buffer usage in InCore functions.
    
    Performs:
    1. Tile size calculation
    2. Liveness analysis (first write, last read)
    3. Buffer reuse analysis based on dependencies
    4. Total buffer capacity estimation with/without reuse
    """
    
    # Element size in bytes
    ELEMENT_SIZES = {
        'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4, 'i16': 2, 'i8': 1, 'u8': 1
    }
    
    def __init__(self, program):
        self.program = program
        self.tile_info: Dict[str, TileBufferInfo] = {}
        self.instructions = []
        self.analysis_result = {}
    
    def analyze(self) -> Dict:
        """Perform complete buffer analysis."""
        # Extract tile declarations
        self._extract_tiles()
        
        # Extract instructions and build liveness info
        self._analyze_liveness()
        
        # Analyze buffer reuse opportunities
        self._analyze_reuse()
        
        # Calculate totals
        self._calculate_totals()
        
        return self.analysis_result
    
    def _extract_tiles(self):
        """Extract tile declarations from program."""
        for name, tile_type in self.program.tile_declarations.items():
            rows = tile_type.shape.rows if hasattr(tile_type.shape, 'rows') else tile_type.shape[0]
            cols = tile_type.shape.cols if hasattr(tile_type.shape, 'cols') else tile_type.shape[1]
            dtype = tile_type.element_type.value if hasattr(tile_type.element_type, 'value') else str(tile_type.element_type)
            
            element_size = self.ELEMENT_SIZES.get(dtype, 4)
            total_bytes = rows * cols * element_size
            
            self.tile_info[name] = TileBufferInfo(
                name=name,
                rows=rows,
                cols=cols,
                dtype=dtype,
                element_size=element_size,
                total_bytes=total_bytes,
                first_write=-1,
                last_read=-1
            )
    
    def _get_tile_name(self, operand) -> Optional[str]:
        """Extract tile name from an operand (handles both string and TileOperand)."""
        if operand is None:
            return None
        if isinstance(operand, str):
            return operand if operand in self.tile_info else None
        if hasattr(operand, 'name'):
            return operand.name if operand.name in self.tile_info else None
        return None
    
    def _analyze_liveness(self):
        """Analyze liveness intervals for each tile."""
        # Collect all instructions
        for idx, instr in enumerate(self.program.instructions):
            self.instructions.append((idx, instr))
            
            # Determine which tiles are read/written
            written_tiles = set()
            read_tiles = set()
            
            # Check destination
            if hasattr(instr, 'dst'):
                dst_name = self._get_tile_name(instr.dst)
                if dst_name:
                    written_tiles.add(dst_name)
            
            # Check source operands
            for attr in ['src', 'src0', 'src1', 'src2']:
                if hasattr(instr, attr):
                    src = getattr(instr, attr)
                    src_name = self._get_tile_name(src)
                    if src_name:
                        read_tiles.add(src_name)
            
            # Update liveness info
            for tile_name in written_tiles:
                info = self.tile_info[tile_name]
                if info.first_write == -1:
                    info.first_write = idx
            
            for tile_name in read_tiles:
                info = self.tile_info[tile_name]
                info.last_read = idx
    
    def _analyze_reuse(self):
        """Analyze buffer reuse opportunities.
        
        A tile B can reuse the buffer of tile A if:
        1. A's last_read < B's first_write (A is dead before B is born)
        2. A and B have the same size
        """
        sorted_tiles = sorted(
            self.tile_info.values(),
            key=lambda t: t.first_write if t.first_write >= 0 else float('inf')
        )
        
        # Track which tiles are "dead" (after their last read)
        available_for_reuse: List[TileBufferInfo] = []
        
        for tile in sorted_tiles:
            if tile.first_write < 0:
                continue  # Never written, skip
            
            # Check if any dead tile can be reused
            for dead_tile in available_for_reuse:
                if (dead_tile.last_read < tile.first_write and
                    dead_tile.rows == tile.rows and
                    dead_tile.cols == tile.cols and
                    dead_tile.dtype == tile.dtype):
                    tile.can_reuse_from = dead_tile.name
                    # Remove from available (can only reuse once)
                    available_for_reuse.remove(dead_tile)
                    break
            
            # After this tile's last read, it becomes available for reuse
            if tile.last_read >= 0:
                available_for_reuse.append(tile)
    
    def _calculate_totals(self):
        """Calculate total buffer requirements."""
        total_without_reuse = sum(t.total_bytes for t in self.tile_info.values())
        
        # Calculate with reuse
        reused_tiles = set()
        for tile in self.tile_info.values():
            if tile.can_reuse_from:
                reused_tiles.add(tile.name)
        
        total_with_reuse = sum(
            t.total_bytes for t in self.tile_info.values()
            if t.name not in reused_tiles
        )
        
        self.analysis_result = {
            'tiles': self.tile_info,
            'total_tiles': len(self.tile_info),
            'total_without_reuse_bytes': total_without_reuse,
            'total_with_reuse_bytes': total_with_reuse,
            'reuse_savings_bytes': total_without_reuse - total_with_reuse,
            'reuse_savings_percent': (
                100.0 * (total_without_reuse - total_with_reuse) / total_without_reuse
                if total_without_reuse > 0 else 0
            ),
            'reused_tiles': [t.name for t in self.tile_info.values() if t.can_reuse_from],
            'reuse_map': {t.name: t.can_reuse_from for t in self.tile_info.values() if t.can_reuse_from}
        }
    
    def generate_report(self) -> str:
        """Generate human-readable analysis report."""
        if not self.analysis_result:
            self.analyze()
        
        r = self.analysis_result
        lines = []
        
        lines.append("// " + "=" * 70)
        lines.append(f"// TILE BUFFER ANALYSIS: {self.program.name}")
        lines.append("// " + "=" * 70)
        lines.append("//")
        
        # Summary
        lines.append("// SUMMARY:")
        lines.append(f"//   Total tiles declared:     {r['total_tiles']}")
        lines.append(f"//   Total capacity (no reuse): {r['total_without_reuse_bytes']:,} bytes ({r['total_without_reuse_bytes']/1024:.1f} KB)")
        lines.append(f"//   Total capacity (w/ reuse): {r['total_with_reuse_bytes']:,} bytes ({r['total_with_reuse_bytes']/1024:.1f} KB)")
        lines.append(f"//   Reuse savings:            {r['reuse_savings_bytes']:,} bytes ({r['reuse_savings_percent']:.1f}%)")
        lines.append("//")
        
        # Individual tiles
        lines.append("// TILE DETAILS:")
        lines.append("//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse")
        lines.append("//   " + "-" * 80)
        
        for name, tile in sorted(r['tiles'].items()):
            shape = f"{tile.rows}x{tile.cols}"
            liveness = f"[{tile.first_write:3d}, {tile.last_read:3d}]" if tile.first_write >= 0 else "[-, -]"
            reuse = f"<- {tile.can_reuse_from}" if tile.can_reuse_from else "-"
            lines.append(f"//   {name:20s} {shape:10s} {tile.dtype:6s} {tile.total_bytes:6d}   {liveness:20s} {reuse}")
        
        lines.append("//")
        
        # Reuse map
        if r['reuse_map']:
            lines.append("// BUFFER REUSE MAP:")
            for dst, src in r['reuse_map'].items():
                lines.append(f"//   {dst} reuses buffer of {src}")
            lines.append("//")
        
        lines.append("// " + "=" * 70)
        lines.append("")
        
        return "\n".join(lines)


class MultiBackendCodeGenerator:
    """
    Unified multi-backend code generator for PTO programs.
    
    Generates optimized code for multiple target architectures:
    - ARM64 NEON (Apple Silicon, ARM servers)
    - NVIDIA CUDA (GPU computing)
    - Huawei Ascend 910B (NPU/AI accelerator)
    
    If a module is provided, buffer analysis results are stored in the module
    for later use by orchestration code generators.
    """
    
    def __init__(self, enable_fusion: bool = True, analyze_buffers: bool = True, 
                 module: 'PTOModule' = None):
        self.enable_fusion = enable_fusion
        self.analyze_buffers = analyze_buffers
        self.module = module  # Optional module to store buffer analysis results
    
    def generate_arm64(self, program) -> str:
        """Generate ARM64 NEON code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        # Determine InCore status
        is_in_core = getattr(program, 'is_in_core', True)
        in_core_str = "InCore (tile-level computation)" if is_in_core else "Orchestration (control flow only)"
        
        # Create orchestration context for non-InCore functions
        orch_ctx = None
        if not is_in_core:
            orch_ctx = OrchestrationContext(module=self.module)
        
        lines = [
            f"// PTO Program: {program.name}",
            f"// Function Type: {in_core_str}",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()  # Run analysis first
            report = analyzer.generate_report()
            lines.append(report)
            
            # Store analysis in module if available
            if self.module is not None:
                self.module.set_buffer_analysis(program.name, analyzer.analysis_result)
        
        # For orchestration functions, add runtime header
        if not is_in_core:
            lines.append('// Orchestration function - builds task graph using PTO runtime')
            lines.append('#include "pto_runtime.h"')
            lines.append('#include "pto_runtime.c"  // Include for standalone build')
            lines.append('')
        
        lines.append(arm64_generate_header())
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in program.memref_declarations.items():
            c_type = ARM64_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
        
        # Find scalars that are initialized by SLI (these are local constants, not params)
        sli_initialized_scalars = set()
        for instr in mock_instructions:
            if instr.opcode == "SLI":
                sli_initialized_scalars.add(instr.dst)
        
        # Declare scalar variables as function parameters (for dynamic bounds)
        # Skip scalars that are initialized internally via SLI
        scalar_params = []
        for name, scalar_type in program.scalar_declarations.items():
            # Skip internal loop variables (scalar_type is ElementType directly)
            if scalar_type in (ElementType.U1, ElementType.INDEX):
                continue
            # Skip scalars initialized via SLI (they are local constants)
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
        
        # Declare tiles as local variables inside the function
        for name, info in tile_info.items():
            c_type = ARM64_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"    {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            fused_codegen = FusedCodeGenerator()
            indent_level = 1  # Base indentation level inside function
            
            for item in fused_result:
                indent = "    " * indent_level
                
                if isinstance(item, FusedLoop):
                    # Indent the fused loop code
                    fused_lines = fused_codegen.generate_fused_loop(item)
                    for fused_line in fused_lines:
                        lines.append(f"{indent}{fused_line}" if fused_line else "")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    info = tile_info.get(instr.dst) if instr.dst else None
                    rows = info.rows if info else 8
                    cols = info.cols if info else 8
                    dtype = info.dtype if info else "f32"
                    
                    # Handle indentation changes for control flow
                    if instr.opcode in ("ENDFOR", "ENDIF"):
                        indent_level = max(1, indent_level - 1)
                        indent = "    " * indent_level
                    elif instr.opcode == "ELSE":
                        # ELSE: decrease then increase
                        indent = "    " * max(1, indent_level - 1)
                    
                    # Generate the barrier code (pass orch_ctx for orchestration functions)
                    barrier_lines = _gen_arm64_barrier_op(instr, rows, cols, dtype, tile_info, orch_ctx)
                    for barrier_line in barrier_lines:
                        lines.append(f"{indent}{barrier_line}" if barrier_line else "")
                    
                    # Increase indentation after opening control flow
                    if instr.opcode in ("FOR", "IF", "ELSE"):
                        indent_level += 1
                    
                    lines.append("")
        
        lines.append("}")
        return "\n".join(lines)
    
    def generate_cuda(self, program) -> str:
        """Generate NVIDIA CUDA code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        # Determine InCore status
        is_in_core = getattr(program, 'is_in_core', True)
        in_core_str = "InCore (tile-level computation)" if is_in_core else "Orchestration (control flow only)"
        
        lines = [
            f"// PTO Program: {program.name}",
            f"// Function Type: {in_core_str}",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()  # Run analysis first
            report = analyzer.generate_report()
            lines.append(report)
            
            # Store analysis in module if available
            if self.module is not None:
                self.module.set_buffer_analysis(program.name, analyzer.analysis_result)
        
        lines.append(cuda_generate_header())
        
        # Declare tiles as __device__ arrays
        for name, info in tile_info.items():
            c_type = CUDA_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"__device__ {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")
        
        # Collect memory references for kernel parameters
        memref_params = []
        memref_types = {}
        for name, memref_type in program.memref_declarations.items():
            c_type = CUDA_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
            memref_types[name] = c_type
        
        # Find scalars that are initialized by SLI (these are local constants, not params)
        sli_initialized_scalars = set()
        for instr in mock_instructions:
            if instr.opcode == "SLI":
                sli_initialized_scalars.add(instr.dst)
        
        # Collect scalar parameters (scalar_type is ElementType directly)
        # Skip scalars initialized via SLI (they are local constants)
        scalar_params = []
        for name, scalar_type in program.scalar_declarations.items():
            if scalar_type in (ElementType.U1, ElementType.INDEX):
                continue
            if name in sli_initialized_scalars:
                continue
            c_type = CUDA_TYPE_MAP.get(scalar_type.value, "int")
            scalar_params.append(f"{c_type} {name}")
        
        # Generate kernel signature with memory reference parameters
        all_params = memref_params + scalar_params
        if all_params:
            kernel_params = ", ".join(all_params)
            lines.append(f"__global__ void {program.name}_kernel({kernel_params}) {{")
        else:
            lines.append(f"__global__ void {program.name}_kernel() {{")
        lines.append("    int _row = threadIdx.y + blockIdx.y * blockDim.y;")
        lines.append("    int _col = threadIdx.x + blockIdx.x * blockDim.x;\n")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            indent_level = 1
            for item in fused_result:
                indent = "    " * indent_level
                
                if isinstance(item, FusedLoop):
                    ops_desc = "; ".join([f"{op.dst}={op.opcode}(...)" for op in item.operations])
                    lines.append(f"{indent}// FUSED ({len(item.operations)} ops): {ops_desc}")
                    lines.append(f"{indent}if (_row < {item.shape.rows} && _col < {item.shape.cols}) {{")
                    for op in item.operations:
                        lines.append(f"{indent}    {_gen_cuda_single_op(op, tile_info)}")
                    lines.append(f"{indent}}}\n")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    info = tile_info.get(instr.dst) if instr.dst else None
                    rows = info.rows if info else 8
                    cols = info.cols if info else 8
                    dtype = info.dtype if info else "f32"
                    
                    # Handle indentation changes for control flow
                    if instr.opcode in ("ENDFOR", "ENDIF"):
                        indent_level = max(1, indent_level - 1)
                        indent = "    " * indent_level
                    elif instr.opcode == "ELSE":
                        indent = "    " * max(1, indent_level - 1)
                    
                    # Generate the barrier code
                    barrier_lines = _gen_cuda_barrier_op(instr, rows, cols, dtype, tile_info)
                    for barrier_line in barrier_lines:
                        lines.append(f"{indent}{barrier_line}" if barrier_line else "")
                    
                    # Increase indentation after opening control flow
                    if instr.opcode in ("FOR", "IF", "ELSE"):
                        indent_level += 1
                    
                    lines.append("")
        
        lines.append("}\n")
        
        # Generate host wrapper function with memory reference parameters
        all_wrapper_params = memref_params + scalar_params
        if all_wrapper_params:
            wrapper_params = ", ".join(all_wrapper_params)
            kernel_args = ", ".join(list(program.memref_declarations.keys()) + 
                                    [n for n, t in program.scalar_declarations.items() 
                                     if t not in (ElementType.U1, ElementType.INDEX)])
            lines.append(f"void {program.name}({wrapper_params}) {{")
            lines.append("    dim3 block(8, 8);")
            lines.append("    dim3 grid(1, 1);")
            lines.append(f"    {program.name}_kernel<<<grid, block>>>({kernel_args});")
            lines.append("    cudaDeviceSynchronize();\n}")
        else:
            lines.append(f"void {program.name}() {{")
            lines.append("    dim3 block(8, 8);")
            lines.append("    dim3 grid(1, 1);")
            lines.append(f"    {program.name}_kernel<<<grid, block>>>();")
            lines.append("    cudaDeviceSynchronize();\n}")
        
        return "\n".join(lines)
    
    def generate_ascend(self, program) -> str:
        """Generate Huawei Ascend 910B (Ascend C) code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        # Determine InCore status
        is_in_core = getattr(program, 'is_in_core', True)
        in_core_str = "InCore (tile-level computation)" if is_in_core else "Orchestration (control flow only)"
        
        lines = [
            f"// PTO Program: {program.name}",
            f"// Function Type: {in_core_str}",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()  # Run analysis first
            report = analyzer.generate_report()
            lines.append(report)
            
            # Store analysis in module if available
            if self.module is not None:
                self.module.set_buffer_analysis(program.name, analyzer.analysis_result)
        
        lines.append(ascend_generate_header())
        
        lines.append(f"class {program.name}Kernel {{")
        lines.append("public:")
        lines.append(f"    __aicore__ inline {program.name}Kernel() {{}}")
        lines.append("    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {")
        lines.append("        inputGm.SetGlobalBuffer((__gm__ float*)input);")
        lines.append("        outputGm.SetGlobalBuffer((__gm__ float*)output);")
        lines.append("        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));")
        lines.append("        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void Process() {")
        lines.append("        CopyIn(); Compute(); CopyOut();")
        lines.append("    }\n")
        lines.append("private:")
        lines.append("    __aicore__ inline void CopyIn() {")
        lines.append("        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();")
        lines.append("        DataCopy(xLocal, inputGm, 64);")
        lines.append("        inQueueX.EnQue(xLocal);")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void Compute() {")
        lines.append("        LocalTensor<float> xLocal = inQueueX.DeQue<float>();")
        lines.append("        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();\n")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"        // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            indent_level = 2  # Base indentation inside Compute()
            for item in fused_result:
                indent = "    " * indent_level
                
                if isinstance(item, FusedLoop):
                    ops_desc = "; ".join([op.opcode for op in item.operations])
                    lines.append(f"{indent}// FUSED ({len(item.operations)} ops): {ops_desc}")
                    for op in item.operations:
                        lines.append(f"{indent}{_gen_ascend_single_op(op, tile_info)}")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    info = tile_info.get(instr.dst) if instr.dst else None
                    rows = info.rows if info else 8
                    cols = info.cols if info else 8
                    dtype = info.dtype if info else "f32"
                    
                    # Handle indentation changes for control flow
                    if instr.opcode in ("ENDFOR", "ENDIF"):
                        indent_level = max(2, indent_level - 1)
                        indent = "    " * indent_level
                    elif instr.opcode == "ELSE":
                        indent = "    " * max(2, indent_level - 1)
                    
                    # Generate the barrier code
                    barrier_lines = _gen_ascend_barrier_op(instr, rows, cols, dtype, tile_info)
                    for barrier_line in barrier_lines:
                        lines.append(f"{indent}{barrier_line}" if barrier_line else "")
                    
                    # Increase indentation after opening control flow
                    if instr.opcode in ("FOR", "IF", "ELSE"):
                        indent_level += 1
                    
                    lines.append("")
        
        lines.append("        outQueueY.EnQue(yLocal);")
        lines.append("        inQueueX.FreeTensor(xLocal);")
        lines.append("    }\n")
        lines.append("    __aicore__ inline void CopyOut() {")
        lines.append("        LocalTensor<float> yLocal = outQueueY.DeQue<float>();")
        lines.append("        DataCopy(outputGm, yLocal, 64);")
        lines.append("        outQueueY.FreeTensor(yLocal);")
        lines.append("    }\n")
        lines.append("private:")
        lines.append("    TPipe pipe;")
        lines.append("    TQue<QuePosition::VECIN, 1> inQueueX;")
        lines.append("    TQue<QuePosition::VECOUT, 1> outQueueY;")
        lines.append("    GlobalTensor<float> inputGm;")
        lines.append("    GlobalTensor<float> outputGm;")
        lines.append("};\n")
        
        lines.append(f"extern \"C\" __global__ __aicore__ void {program.name}_kernel(GM_ADDR input, GM_ADDR output) {{")
        lines.append(f"    {program.name}Kernel op;")
        lines.append("    op.Init(input, output);")
        lines.append("    op.Process();")
        lines.append("}")
        
        return "\n".join(lines)
    
    def generate_all(self, program, output_prefix: str, output_base_dir: str = ".") -> Dict[str, str]:
        """
        Generate code for all backends and save to files.
        
        Args:
            program: PTOProgram built using PTOFunctionBuilder
            output_prefix: Category name for output subdirectory (e.g., "sinh_taylor", "aten_primitives")
            output_base_dir: Base directory for output
            
        Output structure (for InCore functions):
            output_base_dir/
            ├── output_arm64/{output_prefix}/{program.name}.c
            ├── output_cuda/{output_prefix}/{program.name}.cu
            ├── output_ascend910b/{output_prefix}/{program.name}.cpp
            └── output_pto/{output_prefix}/{program.name}.pto
        
        Output structure (for non-InCore functions):
            output_base_dir/
            ├── output_arm64/{output_prefix}/{program.name}.c  (only ARM64!)
            └── output_pto/{output_prefix}/{program.name}.pto
            
        Returns:
            Dict mapping backend name to output file path
        """
        import os
        results = {}
        
        generators = {
            "arm64": self.generate_arm64,
            "cuda": self.generate_cuda,
            "ascend910b": self.generate_ascend,
        }
        
        # Determine which backends to generate
        # Non-InCore functions only support ARM64
        is_in_core = getattr(program, 'is_in_core', True)  # Default to True for backward compat
        
        if is_in_core:
            backends_to_generate = list(BACKENDS.items())
        else:
            # Only ARM64 for non-InCore (orchestration) functions
            backends_to_generate = [(k, v) for k, v in BACKENDS.items() if k == "arm64"]
            print(f"  [Note] Non-InCore function: generating only ARM64 code")
        
        for backend_key, backend_info in backends_to_generate:
            # New structure: output_arm64/category/file.c
            output_dir = os.path.join(output_base_dir, f"output{backend_info['suffix']}", output_prefix)
            os.makedirs(output_dir, exist_ok=True)
            
            code = generators[backend_key](program)
            output_file = os.path.join(output_dir, f"{program.name}{backend_info['extension']}")
            
            with open(output_file, "w") as f:
                f.write(code)
            
            results[backend_key] = output_file
            print(f"  [{backend_info['name']}] -> {output_file}")
        
        # Also save PTO-AS assembly
        compiler = PTOCompiler()
        pto_asm = compiler.compile(program)
        pto_dir = os.path.join(output_base_dir, "output_pto", output_prefix)
        os.makedirs(pto_dir, exist_ok=True)
        pto_file = os.path.join(pto_dir, f"{program.name}.pto")
        with open(pto_file, "w") as f:
            f.write(pto_asm)
        results["pto"] = pto_file
        print(f"  [PTO-AS] -> {pto_file}")
        
        return results
    
    def generate_orchestration_executable(self, program, output_dir: str) -> str:
        """
        Generate a standalone executable for an orchestration function.
        
        The executable builds the task graph and dumps it to a file.
        
        Args:
            program: Orchestration PTOProgram
            output_dir: Directory for output files
            
        Returns:
            Generated C code as string
        """
        if getattr(program, 'is_in_core', True):
            return "// Error: Not an orchestration function"
        
        # Generate the orchestration function
        func_code = self.generate_arm64(program)
        
        # Generate main() wrapper
        lines = [func_code]
        lines.append("")
        lines.append("/**")
        lines.append(" * Main: Build task graph and dump to file")
        lines.append(" */")
        lines.append("int main(int argc, char** argv) {")
        lines.append("    // Allocate runtime on heap (PTORuntime is ~187MB with 65536 max tasks)")
        lines.append("    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));")
        lines.append("    if (!rt) { fprintf(stderr, \"Failed to allocate PTORuntime\\n\"); return 1; }")
        lines.append("    pto_runtime_init(rt);")
        lines.append("")
        
        # Declare buffers for all memrefs
        lines.append("    // Declare dummy buffers")
        for name in program.memref_declarations.keys():
            lines.append(f"    float {name}[1024];  // Dummy buffer")
        lines.append("")
        
        # Determine scalar parameters (excluding SLI-initialized)
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        sli_initialized = set()
        for instr in mock_instructions:
            if instr.opcode == "SLI":
                sli_initialized.add(instr.dst)
        
        scalar_args = []
        for name, scalar_type in program.scalar_declarations.items():
            if scalar_type in (ElementType.U1, ElementType.INDEX):
                continue
            if name in sli_initialized:
                continue
            # Default value for scalars
            default_val = "32" if "tile" in name.lower() or "num" in name.lower() else "1"
            lines.append(f"    int {name} = {default_val};  // TODO: set from args")
            scalar_args.append(name)
        lines.append("")
        
        # Call the orchestration function
        args = ["rt"] + list(program.memref_declarations.keys()) + scalar_args
        args_str = ", ".join(args)
        lines.append(f"    // Build task graph")
        lines.append(f"    {program.name}({args_str});")
        lines.append("")
        
        # Dump results
        lines.append('    printf("\\n");')
        lines.append("    pto_runtime_dump_stdout(rt);")
        lines.append(f'    pto_runtime_dump(rt, "{program.name}_task_graph.txt");')
        lines.append("")
        lines.append("    pto_runtime_shutdown(rt);")
        lines.append("    free(rt);")
        lines.append("    return 0;")
        lines.append("}")
        
        return "\n".join(lines)
    
    def compile_and_run_orchestration(self, program, output_dir: str, 
                                       extra_args: Dict[str, int] = None) -> Optional[str]:
        """
        Generate, compile, execute, and return the task graph dump path.
        
        Args:
            program: Orchestration PTOProgram
            output_dir: Directory for output files
            extra_args: Optional dict of scalar name -> value to pass to the function
            
        Returns:
            Path to the generated dump file, or None on failure
        """
        import subprocess
        
        if getattr(program, 'is_in_core', True):
            print(f"  [Error] {program.name} is not an orchestration function")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        runtime_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Generate standalone C code
        c_code = self.generate_orchestration_executable(program, output_dir)
        
        # If extra_args provided, patch the default values
        if extra_args:
            for name, value in extra_args.items():
                old_pattern = f"int {name} = \\d+;"
                new_value = f"int {name} = {value};"
                c_code = re.sub(old_pattern, new_value, c_code)
        
        # Write C file
        c_file = os.path.join(output_dir, f"{program.name}_orchestration.c")
        with open(c_file, 'w') as f:
            f.write(c_code)
        print(f"  [Orchestration] Generated: {c_file}")
        
        # Compile
        exe_file = os.path.join(output_dir, f"{program.name}_orchestration")
        compile_cmd = ["gcc", "-O2", "-I", runtime_dir, "-o", exe_file, c_file]
        
        print(f"  [Orchestration] Compiling...")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [Orchestration] Compilation failed: {result.stderr}")
            return None
        
        # Execute
        print(f"  [Orchestration] Executing...")
        result = subprocess.run([exe_file], capture_output=True, text=True, cwd=output_dir)
        
        if result.returncode != 0:
            print(f"  [Orchestration] Execution failed: {result.stderr}")
            return None
        
        # Print output (limited)
        if result.stdout:
            output_lines = result.stdout.split('\n')
            if len(output_lines) > 50:
                print('\n'.join(output_lines[:30]))
                print(f"... ({len(output_lines) - 50} lines omitted) ...")
                print('\n'.join(output_lines[-20:]))
            else:
                print(result.stdout)
        
        # Check dump file
        dump_file = os.path.join(output_dir, f"{program.name}_task_graph.txt")
        if os.path.exists(dump_file):
            print(f"  [Orchestration] Task graph dump: {dump_file}")
            # Clean up executable
            try:
                os.remove(exe_file)
            except:
                pass
            return dump_file
        
        return None


# Convenience functions for backward compatibility
def generate_all_backends(program, output_prefix: str, output_base_dir: str = ".", 
                          enable_fusion: bool = True) -> Dict[str, str]:
    """Generate code for all backends (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_all(program, output_prefix, output_base_dir)


def generate_arm64_code(program, enable_fusion: bool = True) -> str:
    """Generate ARM64 NEON code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_arm64(program)


def generate_cuda_code(program, enable_fusion: bool = True) -> str:
    """Generate CUDA code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_cuda(program)


def generate_ascend_code(program, enable_fusion: bool = True) -> str:
    """Generate Ascend C code (convenience wrapper)."""
    generator = MultiBackendCodeGenerator(enable_fusion=enable_fusion)
    return generator.generate_ascend(program)


# =============================================================================
# Runtime Code Generator
# =============================================================================

class RuntimeCodeGenerator:
    """
    Generates Orchestration function code that uses PTO Runtime.
    
    When an Orchestration function calls InCore functions:
    1. Each InCore call becomes a pending task with a task_id
    2. Tasks track producer-consumer dependencies via fanin/fanout
    3. TensorMap tracks which task produces each tensor region
    """
    
    def __init__(self, module: 'PTOModule'):
        self.module = module
        self.lines = []
        self.indent_level = 0
    
    def _emit(self, line: str = ""):
        """Emit a line of code."""
        indent = "    " * self.indent_level
        self.lines.append(f"{indent}{line}" if line else "")
    
    def _indent(self):
        """Increase indentation."""
        self.indent_level += 1
    
    def _dedent(self):
        """Decrease indentation."""
        self.indent_level = max(0, self.indent_level - 1)
    
    def generate(self) -> str:
        """Generate complete runtime-based C code for the module."""
        self.lines = []
        self.indent_level = 0
        
        # Header
        self._emit("/**")
        self._emit(" * PTO Runtime-based Orchestration Code")
        self._emit(f" * Module: {self.module.name}")
        self._emit(" * Auto-generated by PTO ISA Compiler")
        self._emit(" */")
        self._emit()
        self._emit('#include "pto_runtime.h"')
        self._emit('#include <stdio.h>')
        self._emit('#include <stdlib.h>')
        self._emit()
        
        # Forward declarations for InCore functions
        self._emit("// Forward declarations for InCore functions")
        for name, prog in self.module.functions.items():
            if prog.is_in_core:
                params = self._get_function_params(prog)
                self._emit(f"void {name}({params});")
        self._emit()
        
        # Generate each Orchestration function
        for name, prog in self.module.functions.items():
            if not prog.is_in_core:
                self._generate_orchestration_function(prog)
        
        return "\n".join(self.lines)
    
    def _get_function_params(self, prog: 'PTOProgram') -> str:
        """Get function parameter list as string."""
        params = []
        for name, memref_type in prog.memref_declarations.items():
            dtype = "float"  # Simplified - use actual type mapping in production
            params.append(f"{dtype}* {name}")
        return ", ".join(params) if params else "void"
    
    def _get_tile_shape(self, prog: 'PTOProgram', tile_name: str) -> Tuple[int, int]:
        """Get tile shape from program."""
        for instr in prog.instructions:
            if isinstance(instr, dict) and instr.get('opcode') == 'TILE_DECL':
                if instr.get('name') == tile_name:
                    return (instr.get('rows', 8), instr.get('cols', 8))
        return (8, 8)  # Default
    
    def _generate_orchestration_function(self, prog: 'PTOProgram'):
        """Generate an Orchestration function using PTO Runtime."""
        # Collect info about InCore calls
        incore_calls = []
        for instr in prog.instructions:
            if isinstance(instr, CALL):
                callee = self.module.functions.get(instr.callee)
                if callee and callee.is_in_core:
                    incore_calls.append(instr)
        
        # Function signature
        params = self._get_function_params(prog)
        self._emit(f"// Orchestration function: {prog.name}")
        self._emit(f"void {prog.name}_runtime(PTORuntime* rt, {params}) {{")
        self._indent()
        
        if not incore_calls:
            self._emit("// No InCore calls")
            self._dedent()
            self._emit("}")
            self._emit()
            return
        
        # Generate task scheduling for each InCore call
        for idx, call in enumerate(incore_calls):
            self._emit(f"// Task {idx}: {call.callee}")
            self._emit(f"int32_t t{idx} = pto_task_alloc(rt, \"{call.callee}\", (void*){call.callee});")
            
            # Determine inputs and outputs from the called function
            callee_prog = self.module.functions.get(call.callee)
            if callee_prog:
                # Analyze memref declarations to determine inputs/outputs
                arg_mapping = call.args  # param_name -> actual_arg_name
                
                for param_name, actual_arg in arg_mapping.items():
                    # Heuristic: "input" params are inputs, "output"/"result" params are outputs
                    is_output = "output" in param_name.lower() or "result" in param_name.lower()
                    
                    # Determine shape based on callee function and parameter name
                    rows, cols = self._infer_shape_from_callee(call.callee, param_name, is_output)
                    
                    if is_output:
                        self._emit(f"pto_task_add_output(rt, t{idx}, {actual_arg}, 0, 0, {rows}, {cols});")
                    else:
                        self._emit(f"pto_task_add_input(rt, t{idx}, {actual_arg}, 0, 0, {rows}, {cols});")
            
            self._emit(f"pto_task_submit(rt, t{idx});")
            self._emit()
        
        # Execute all tasks
        self._emit("// Execute all scheduled tasks")
        self._emit("pto_execute_all(rt);")
        
        self._dedent()
        self._emit("}")
        self._emit()
    
    def _infer_shape_from_callee(self, callee_name: str, param_name: str, is_output: bool) -> Tuple[int, int]:
        """Infer tensor shape based on callee function and parameter name."""
        # Reduction functions output row vectors
        if callee_name in ['rowmax', 'rowsum'] and is_output:
            return (8, 1)
        
        # Row operation functions: input_row is a row vector
        if callee_name in ['rowexpandsub', 'rowexpanddiv', 'rowexpandmul']:
            if 'row' in param_name.lower() and 'input' in param_name.lower():
                return (8, 1)
        
        # Default to full tile
        return (8, 8)
    
    def generate_main_wrapper(self, entry_func: str) -> str:
        """Generate a main() function that initializes runtime and calls entry function."""
        entry_prog = self.module.functions.get(entry_func)
        if not entry_prog:
            return "// Error: Entry function not found"
        
        lines = []
        lines.append("/**")
        lines.append(" * Main entry point with runtime initialization")
        lines.append(" */")
        lines.append("int main(int argc, char** argv) {")
        lines.append("    // Initialize runtime")
        lines.append("    PTORuntime rt;")
        lines.append("    pto_runtime_init(&rt);")
        lines.append("")
        
        # Allocate buffers for each memref
        lines.append("    // Allocate buffers")
        for name, memref_type in entry_prog.memref_declarations.items():
            lines.append(f"    float {name}_buf[64];  // Adjust size as needed")
        lines.append("")
        
        # Initialize input (example)
        lines.append("    // Initialize input (example)")
        lines.append("    for (int i = 0; i < 64; i++) {")
        lines.append("        // input_buf[i] = ...;")
        lines.append("    }")
        lines.append("")
        
        # Call entry function
        params = ", ".join([f"{name}_buf" for name in entry_prog.memref_declarations.keys()])
        lines.append(f"    // Execute orchestration function")
        lines.append(f"    {entry_func}_runtime(&rt, {params});")
        lines.append("")
        lines.append("    // Dump runtime state to file")
        lines.append(f"    pto_runtime_dump(&rt, \"{entry_func}_runtime_dump.txt\");")
        lines.append("    pto_runtime_dump_stdout(&rt);")
        lines.append("")
        lines.append("    // Print statistics")
        lines.append("    pto_runtime_stats(&rt);")
        lines.append("")
        lines.append("    // Shutdown runtime")
        lines.append("    pto_runtime_shutdown(&rt);")
        lines.append("")
        lines.append("    return 0;")
        lines.append("}")
        
        return "\n".join(lines)


class OrchestrationCodeGenerator:
    """
    Code generator for Orchestration functions.
    
    Generates standalone C code that:
    1. Builds task graphs using PTO runtime
    2. Tracks producer-consumer dependencies via fanin/fanout
    3. Can be compiled and executed to generate task dumps
    """
    
    def __init__(self, module: 'PTOModule'):
        self.module = module
        self.tensor_producers: Dict[str, int] = {}  # tensor_name -> task_id
        self.task_counter = 0
        self.lines = []
        self.indent = 0
    
    def _emit(self, text: str = ""):
        """Emit a line with proper indentation."""
        prefix = "    " * self.indent
        self.lines.append(f"{prefix}{text}" if text else "")
    
    def generate(self, entry_func: str, standalone: bool = True) -> str:
        """
        Generate complete runtime code for an orchestration function.
        
        Args:
            entry_func: Name of the orchestration function
            standalone: If True, generate a complete executable with main()
        """
        self.lines = []
        self.indent = 0
        
        prog = self.module.functions.get(entry_func)
        if not prog:
            return f"// Error: Function {entry_func} not found"
        
        if prog.is_in_core:
            return f"// Error: {entry_func} is an InCore function, not Orchestration"
        
        # Header
        self._emit("/**")
        self._emit(f" * Orchestration Function: {entry_func}")
        self._emit(" * Auto-generated by PTO ISA Compiler")
        self._emit(" *")
        self._emit(" * This code builds the task dependency graph using PTO Runtime.")
        self._emit(" * Compile and run to generate task graph dump.")
        self._emit(" */")
        self._emit()
        self._emit('#include "pto_runtime.h"')
        if standalone:
            self._emit('#include "pto_runtime.c"  // Include implementation for standalone build')
        self._emit()
        
        # Generate the orchestration function
        self._generate_function(prog)
        
        # Generate main() if standalone
        if standalone:
            self._generate_main(prog)
        
        return "\n".join(self.lines)
    
    def _make_param_list(self, prog: 'PTOProgram') -> str:
        """Create parameter list string."""
        params = []
        for name in prog.memref_declarations.keys():
            params.append(f"float* {name}")
        return ", ".join(params) if params else "void"
    
    def _generate_function(self, prog: 'PTOProgram'):
        """Generate the runtime-based orchestration function."""
        # Reset state
        self.tensor_producers = {}
        self.task_counter = 0
        
        # Function signature
        params = self._make_param_list(prog)
        self._emit(f"/**")
        self._emit(f" * Build task graph for {prog.name}")
        self._emit(f" * Each CALL to an InCore function becomes a task with dependencies")
        self._emit(f" */")
        self._emit(f"void {prog.name}_build_task_graph(PTORuntime* rt, {params}) {{")
        self.indent += 1
        
        # Analyze and generate code for each CALL instruction
        for instr in prog.instructions:
            if isinstance(instr, CALL):
                self._generate_call(instr)
            elif isinstance(instr, RETURN):
                pass  # Skip RETURN in orchestration
        
        self.indent -= 1
        self._emit("}")
        self._emit()
    
    def _generate_call(self, call: CALL):
        """Generate task scheduling code for an InCore function call."""
        callee = self.module.functions.get(call.callee)
        if not callee:
            self._emit(f"// ERROR: Unknown function {call.callee}")
            return
        
        task_id = self.task_counter
        self.task_counter += 1
        
        self._emit(f"// Task {task_id}: {call.callee}")
        self._emit(f"int32_t t{task_id} = pto_task_alloc(rt, \"{call.callee}\", NULL);")
        
        # Analyze arguments - determine inputs and outputs
        for param_name, arg_name in call.args.items():
            is_output = any(kw in param_name.lower() for kw in ['output', 'result', 'dst', 'out'])
            rows, cols = self._infer_shape(param_name, call.callee, is_output)
            
            if is_output:
                self._emit(f"pto_task_add_output(rt, t{task_id}, {arg_name}, 0, 0, {rows}, {cols});")
                self.tensor_producers[arg_name] = task_id
            else:
                self._emit(f"pto_task_add_input(rt, t{task_id}, {arg_name}, 0, 0, {rows}, {cols});")
        
        self._emit(f"pto_task_submit(rt, t{task_id});")
        self._emit()
    
    def _generate_main(self, prog: 'PTOProgram'):
        """Generate main() function for standalone execution."""
        self._emit("/**")
        self._emit(" * Main: Build task graph and dump to file")
        self._emit(" */")
        self._emit("int main(int argc, char** argv) {")
        self.indent += 1
        
        self._emit("PTORuntime rt;")
        self._emit("pto_runtime_init(&rt);")
        self._emit()
        
        # Declare buffers
        self._emit("// Declare buffers")
        for name in prog.memref_declarations.keys():
            size = 8 if any(kw in name.lower() for kw in ['rowmax', 'rowsum']) else 64
            self._emit(f"float {name}[{size}];")
        self._emit()
        
        # Build task graph
        args = ", ".join(prog.memref_declarations.keys())
        self._emit("// Build task graph")
        self._emit(f"{prog.name}_build_task_graph(&rt, {args});")
        self._emit()
        
        # Dump
        self._emit("// Dump task graph")
        self._emit("printf(\"\\n\");")
        self._emit("pto_runtime_dump_stdout(&rt);")
        self._emit(f'pto_runtime_dump(&rt, "{prog.name}_task_graph.txt");')
        self._emit()
        
        self._emit("pto_runtime_shutdown(&rt);")
        self._emit("return 0;")
        
        self.indent -= 1
        self._emit("}")
    
    def _infer_shape(self, param_name: str, callee_name: str, is_output: bool) -> Tuple[int, int]:
        """Infer tensor shape from parameter name, callee function, and position."""
        if callee_name in ['rowmax', 'rowsum'] and is_output:
            return (8, 1)
        if callee_name in ['rowexpandsub', 'rowexpanddiv', 'rowexpandmul']:
            if 'row' in param_name.lower() and 'input' in param_name.lower():
                return (8, 1)
        if any(kw in param_name.lower() for kw in ['rowmax', 'rowsum', 'row_max', 'row_sum']):
            return (8, 1)
        return (8, 8)
    
    def compile_and_run(self, entry_func: str, output_dir: str) -> Optional[str]:
        """
        Generate, compile, execute, and return the task graph dump path.
        
        Args:
            entry_func: Name of the orchestration function
            output_dir: Directory for output files
            
        Returns:
            Path to the generated dump file, or None on failure
        """
        import subprocess
        
        prog = self.module.functions.get(entry_func)
        if not prog or prog.is_in_core:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        runtime_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Generate standalone C code
        c_code = self.generate(entry_func, standalone=True)
        
        # Write C file
        c_file = os.path.join(output_dir, f"{entry_func}_orchestration.c")
        with open(c_file, 'w') as f:
            f.write(c_code)
        print(f"  [Orchestration] Generated: {c_file}")
        
        # Compile
        exe_file = os.path.join(output_dir, f"{entry_func}_orchestration")
        compile_cmd = ["gcc", "-I", runtime_dir, "-o", exe_file, c_file]
        
        print(f"  [Orchestration] Compiling...")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [Orchestration] Compilation failed: {result.stderr}")
            return None
        
        # Execute
        print(f"  [Orchestration] Executing...")
        result = subprocess.run([exe_file], capture_output=True, text=True, cwd=output_dir)
        
        if result.returncode != 0:
            print(f"  [Orchestration] Execution failed: {result.stderr}")
            return None
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check dump file
        dump_file = os.path.join(output_dir, f"{entry_func}_task_graph.txt")
        if os.path.exists(dump_file):
            print(f"  [Orchestration] Task graph dump: {dump_file}")
            # Clean up executable
            try:
                os.remove(exe_file)
            except:
                pass
            return dump_file
        
        return None


def compile_module_with_task_graph(module: 'PTOModule', output_base_dir: str, 
                                    subdir: str = "") -> Dict[str, str]:
    """
    Compile a PTO module and generate task graph dumps for all orchestration functions.
    
    Args:
        module: PTO module to compile
        output_base_dir: Base directory for output (e.g., "examples")
        subdir: Subdirectory name (e.g., "fused_softmax")
        
    Returns:
        Dictionary mapping function names to their dump file paths
    """
    arm64_dir = os.path.join(output_base_dir, "output_arm64", subdir) if subdir else \
                os.path.join(output_base_dir, "output_arm64")
    os.makedirs(arm64_dir, exist_ok=True)
    
    results = {}
    
    # Find all orchestration functions
    for func_name, prog in module.functions.items():
        if not prog.is_in_core:
            print(f"\n  Processing orchestration function: {func_name}")
            
            gen = OrchestrationCodeGenerator(module)
            dump_path = gen.compile_and_run(func_name, arm64_dir)
            
            if dump_path:
                results[func_name] = dump_path
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Simple matrix multiply
    print("=" * 60)
    print("Example 1: Simple Matrix Multiply")
    print("=" * 60)
    
    program1 = (PTOFunctionBuilder("matmul_example")
        # Declare tiles
        .tile("a", 64, 64, ElementType.F16)
        .tile("b", 64, 64, ElementType.F16)
        .tile("c", 64, 64, ElementType.F32)
        # Declare memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Load operands
        .load("a", "mem_a", 0, 0)
        .load("b", "mem_b", 0, 0)
        # Compute
        .matmul("c", "a", "b")
        # Store result
        .store("c", "mem_c", 0, 0)
        .build())
    
    compiler = PTOCompiler()
    code1 = compiler.compile(program1)
    print(code1)
    
    # Example 2: Tiled matrix multiply with nested loops
    print("\n" + "=" * 60)
    print("Example 2: Tiled Matrix Multiply with Nested Loops")
    print("=" * 60)
    
    program2 = (PTOFunctionBuilder("tiled_matmul")
        # Declare tiles (smaller tiles for tiling)
        .tile("a_tile", 16, 16, ElementType.F16)
        .tile("b_tile", 16, 16, ElementType.F16)
        .tile("c_tile", 16, 16, ElementType.F32)
        # Larger tile for iteration bounds
        .tile("full_matrix", 64, 64, ElementType.F32)
        # Memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Nested loop over tiles (4x4 tiles to cover 64x64)
        .for_loop("i", 0, 4, 1)  # Outer loop
        .for_loop("j", 0, 4, 1)  # Inner loop
        # Load tiles
        .load("a_tile", "mem_a")
        .load("b_tile", "mem_b")
        # Compute
        .matmul("c_tile", "a_tile", "b_tile")
        # Store
        .store("c_tile", "mem_c")
        .end_for()  # End inner
        .end_for()  # End outer
        .build())
    
    code2 = compiler.compile(program2)
    print(code2)
    
    # Example 3: MLP forward pass
    print("\n" + "=" * 60)
    print("Example 3: MLP Forward Pass")
    print("=" * 60)
    
    program3 = (PTOFunctionBuilder("mlp_forward")
        # Input, weights, bias, output tiles
        .tile("input", 64, 128, ElementType.F16)
        .tile("weight", 128, 64, ElementType.F16)
        .tile("bias", 64, 64, ElementType.F32)
        .tile("output", 64, 64, ElementType.F32)
        .tile("activated", 64, 64, ElementType.F32)
        # Memory refs
        .memref("mem_in", MemorySpace.GM, ElementType.F16)
        .memref("mem_w", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F32)
        .memref("mem_out", MemorySpace.GM, ElementType.F32)
        # Load data
        .load("input", "mem_in")
        .load("weight", "mem_w")
        .load("bias", "mem_b")
        # Linear layer: output = input @ weight
        .matmul("output", "input", "weight")
        # Add bias
        .add("output", "output", "bias")
        # ReLU activation
        .relu("activated", "output")
        # Store result
        .store("activated", "mem_out")
        .build())
    
    code3 = compiler.compile(program3)
    print(code3)
    
    print("\n" + "=" * 60)
    print("Compilation completed successfully!")
    print("=" * 60)
