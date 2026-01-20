"""
PTO Dynamic Softmax with Orchestration

This module implements softmax with dynamic tiling using:
1. InCore functions: Basic tile-level operations (rowmax, rowexpandsub, exp, rowsum, rowexpanddiv)
2. Orchestration function: dynamic_softmax which schedules InCore function calls

Softmax computation:
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_i - max(x)))

The orchestration function handles:
- Loop over tiles for arbitrary input sizes
- Tail handling for non-aligned sizes
- Task dependency management via PTO runtime
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import (
    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,
    MultiBackendCodeGenerator, generate_all_backends
)
from pto_isa_definition import ElementType, MemorySpace, CompareMode

# Default configuration
DEFAULT_DTYPE = ElementType.F32
DEFAULT_ROWS = 8
DEFAULT_COLS = 8


# =============================================================================
# InCore Functions: Tile-level Operations
# =============================================================================

def create_rowmax_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Find maximum value in each row of a tile.
    
    Input: [rows, cols] tensor
    Output: [rows, 1] tensor with row-wise max values
    """
    return (PTOFunctionBuilder("rowmax")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpandsub_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Subtract row-wise values from each element.
    
    Input x: [rows, cols] tensor
    Input row_vals: [rows, 1] tensor  
    Output: [rows, cols] tensor where output[i,j] = x[i,j] - row_vals[i,0]
    """
    return (PTOFunctionBuilder("rowexpandsub")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_exp_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Element-wise exponential.
    
    Input: [rows, cols] tensor
    Output: [rows, cols] tensor where output[i,j] = exp(input[i,j])
    """
    return (PTOFunctionBuilder("elem_exp")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowsum_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Sum values in each row of a tile.
    
    Input: [rows, cols] tensor
    Output: [rows, 1] tensor with row-wise sums
    """
    return (PTOFunctionBuilder("rowsum")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpanddiv_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Divide each element by row-wise values.
    
    Input x: [rows, cols] tensor
    Input row_vals: [rows, 1] tensor
    Output: [rows, cols] tensor where output[i,j] = x[i,j] / row_vals[i,0]
    """
    return (PTOFunctionBuilder("rowexpanddiv")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Orchestration Function: Dynamic Softmax with Tiling Loop
# =============================================================================

def create_dynamic_softmax_module(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    Create a module with dynamic softmax as orchestration function.
    
    The module contains:
    - InCore functions: rowmax, rowexpandsub, elem_exp, rowsum, rowexpanddiv
    - Orchestration function: dynamic_softmax (calls InCore functions in a loop)
    
    The orchestration function handles arbitrary input sizes by:
    1. Processing full tiles in a loop
    2. Handling tail rows separately
    
    Returns:
        PTOModule with all component functions
    """
    module = PTOModule("dynamic_softmax_module")
    
    # Add InCore building block functions
    module.add_function(create_rowmax_func(rows, cols, dtype))
    module.add_function(create_rowexpandsub_func(rows, cols, dtype))
    module.add_function(create_exp_func(rows, cols, dtype))
    module.add_function(create_rowsum_func(rows, cols, dtype))
    module.add_function(create_rowexpanddiv_func(rows, cols, dtype))
    
    # Create orchestration function that calls InCore functions with dynamic tiling
    from pto_isa_definition import CompareMode
    
    dynamic_softmax = (PTOFunctionBuilder("dynamic_softmax", module=module)
        .not_in_core()  # Orchestration function
        
        # Memory references for input/output
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        # Temporary buffers for intermediate results
        .memref("temp_rowmax", MemorySpace.GM, dtype)
        .memref("temp_shifted", MemorySpace.GM, dtype)
        .memref("temp_exp", MemorySpace.GM, dtype)
        .memref("temp_rowsum", MemorySpace.GM, dtype)
        
        # Scalar loop control variables
        .scalar("total_rows", ElementType.I32)       # Total input rows
        .scalar("tile_rows", ElementType.I32)        # Rows per tile (constant)
        .scalar("num_full_tiles", ElementType.I32)   # Number of full tiles
        .scalar("tail_rows", ElementType.I32)        # Remaining rows
        .scalar("has_tail", ElementType.U1)          # Whether tail exists
        .scalar("zero", ElementType.I32)
        
        # Initialize
        .scalar_li("tile_rows", rows)
        .scalar_li("zero", 0)
        
        # ====================================================================
        # Main Loop: Process full tiles
        # ====================================================================
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            # Call InCore functions for each tile
            # Each call becomes a task in the PTO runtime
            
            # Step 1: Find row maximum
            .call("rowmax", {"input": "input", "output": "temp_rowmax"})
            
            # Step 2: Subtract row max (for numerical stability)
            .call("rowexpandsub", {
                "input_x": "input",
                "input_row": "temp_rowmax",
                "output": "temp_shifted"
            })
            
            # Step 3: Compute exponential
            .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
            
            # Step 4: Sum each row
            .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
            
            # Step 5: Normalize by row sum
            .call("rowexpanddiv", {
                "input_x": "temp_exp",
                "input_row": "temp_rowsum",
                "output": "output"
            })
        .end_for()
        
        # ====================================================================
        # Handle Tail: Process remaining rows (if any)
        # ====================================================================
        .scalar_cmp("has_tail", "tail_rows", "zero", CompareMode.GT)
        .if_then("has_tail")
            # Same sequence for tail tile
            .call("rowmax", {"input": "input", "output": "temp_rowmax"})
            .call("rowexpandsub", {
                "input_x": "input",
                "input_row": "temp_rowmax",
                "output": "temp_shifted"
            })
            .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
            .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
            .call("rowexpanddiv", {
                "input_x": "temp_exp",
                "input_row": "temp_rowsum",
                "output": "output"
            })
        .endif()
        
        .build())
    
    module.add_function(dynamic_softmax)
    module.set_entry("dynamic_softmax")
    
    return module


# =============================================================================
# Main: Generate Code and Task Graph
# =============================================================================

def main():
    """Generate code for dynamic softmax with orchestration."""
    output_base = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("PTO Dynamic Softmax - Orchestration with InCore Functions")
    print("=" * 70)
    
    # Create module
    module = create_dynamic_softmax_module()
    
    print(f"\nModule: {module.name}")
    print(f"Functions:")
    for name in module.get_function_names():
        func = module.get_function(name)
        func_type = "Orchestration" if not func.is_in_core else "InCore"
        print(f"  - {name}: {func_type}")
    print(f"Entry: {module.entry_function}")
    
    # ==========================================================================
    # Step 1: Compile module to PTO assembly
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Compile to PTO Assembly")
    print("=" * 70)
    
    compiler = PTOModuleCompiler(
        inline_in_core=False,  # Keep CALLs for orchestration
        eliminate_redundant_mem=False
    )
    pto_code = compiler.compile(module)
    
    print("\n--- Generated PTO Assembly ---")
    print(pto_code)
    
    # Save PTO assembly
    pto_dir = os.path.join(output_base, "output_pto", "fused_softmax")
    os.makedirs(pto_dir, exist_ok=True)
    pto_file = os.path.join(pto_dir, "dynamic_softmax_module.pto")
    with open(pto_file, "w") as f:
        f.write(pto_code)
    print(f"\n  [PTO] -> {pto_file}")
    
    # ==========================================================================
    # Step 2: Generate code for all functions using MultiBackendCodeGenerator
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Generate Code for All Functions")
    print("=" * 70)
    
    arm64_dir = os.path.join(output_base, "output_arm64", "fused_softmax")
    os.makedirs(arm64_dir, exist_ok=True)
    
    # Create generator with module reference (for buffer analysis tracking)
    gen = MultiBackendCodeGenerator(enable_fusion=True, analyze_buffers=True, module=module)
    
    # Generate code for all functions
    # InCore functions: generate actual computation code
    # Orchestration functions: generate task graph building code
    for func_name in module.get_function_names():
        func = module.get_function(func_name)
        arm64_code = gen.generate_arm64(func)
        func_file = os.path.join(arm64_dir, f"{func_name}.c")
        with open(func_file, "w") as f:
            f.write(arm64_code)
        func_type = "InCore" if func.is_in_core else "Orchestration"
        print(f"  [ARM64] {func_name} ({func_type}) -> {func_file}")
    
    # ==========================================================================
    # Step 3: Compile and Run Orchestration to Build Task Graph
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Compile and Run Orchestration to Build Task Graph")
    print("=" * 70)
    
    # Use the new integrated API in MultiBackendCodeGenerator
    # This automatically generates the task graph building code
    orch_func = module.get_function(module.entry_function)
    dump_file = gen.compile_and_run_orchestration(
        orch_func, 
        arm64_dir,
        extra_args={"num_full_tiles": 8, "tail_rows": 0}  # Example args
    )
    
    if dump_file:
        print(f"\n  Task graph dump: {dump_file}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Code Generation Complete!")
    print("=" * 70)
    
    print("\nGenerated files:")
    print(f"  PTO Assembly: {pto_file}")
    print(f"  ARM64 InCore functions: {arm64_dir}/{{rowmax,rowexpandsub,elem_exp,rowsum,rowexpanddiv}}.c")
    print(f"  Orchestration code: {arm64_dir}/dynamic_softmax_orchestration.c")
    print(f"  Task graph dump: {dump_file}")
    
    return module


if __name__ == "__main__":
    main()
