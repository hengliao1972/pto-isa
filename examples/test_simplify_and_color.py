#!/usr/bin/env python3
"""
Test script for PTOProgram.SimplifyAndColor() method.

This script demonstrates:
1. Building a simple PTO program with dependencies
2. Running dependency analysis and graph coloring
3. Generating visualizations
4. Dumping PTO assembly with dependency info
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import (
    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,
    MultiBackendCodeGenerator
)
from pto_isa_definition import ElementType, MemorySpace


def create_test_program():
    """Create a simple program with various dependencies for testing."""
    return (PTOFunctionBuilder("test_deps")
        .in_core()
        # Tile declarations
        .tile("a", 8, 8, ElementType.F32)
        .tile("b", 8, 8, ElementType.F32)
        .tile("c", 8, 8, ElementType.F32)
        .tile("d", 8, 8, ElementType.F32)
        .tile("e", 8, 8, ElementType.F32)
        .tile("tmp", 8, 8, ElementType.F32)
        # Memory references
        .memref("input_a", MemorySpace.GM, ElementType.F32)
        .memref("input_b", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        # Instructions - creating various dependency patterns
        # 0: Load a
        .load("a", "input_a", 0, 0)
        # 1: Load b
        .load("b", "input_b", 0, 0)
        # 2: c = a + b (RAW dependency on 0 and 1)
        .add("c", "a", "b")
        # 3: d = a * b (RAW dependency on 0 and 1)
        .mul("d", "a", "b")
        # 4: e = c + d (RAW dependency on 2 and 3)
        # This creates A->C, B->C, A->D, B->D, C->E, D->E
        # And transitive: A->E, B->E (should be removed by simplify)
        .add("e", "c", "d")
        # 5: tmp = exp(e) (RAW dependency on 4)
        .exp("tmp", "e")
        # 6: Store tmp
        .store("tmp", "output", 0, 0)
        .build())


def create_loop_program():
    """Create a program with loop-carried dependencies."""
    return (PTOFunctionBuilder("test_loop_deps")
        .in_core()
        .tile("x", 8, 8, ElementType.F32)
        .tile("y", 8, 8, ElementType.F32)
        .tile("acc", 8, 8, ElementType.F32)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        .scalar("i", ElementType.I32)
        # Initialize accumulator
        .load("acc", "input", 0, 0)
        # Loop with accumulation (creates cyclic dependency)
        .for_loop("i", 0, 4, 1)
            .load("x", "input", "i", 0)
            .mul("y", "x", "acc")
            .add("acc", "acc", "y")  # acc depends on previous iteration
        .end_for()
        # Store result
        .store("acc", "output", 0, 0)
        .build())


def create_complex_program():
    """Create a more complex program for thorough testing."""
    return (PTOFunctionBuilder("test_complex")
        .in_core()
        .tile("t0", 8, 8, ElementType.F32)
        .tile("t1", 8, 8, ElementType.F32)
        .tile("t2", 8, 8, ElementType.F32)
        .tile("t3", 8, 8, ElementType.F32)
        .tile("t4", 8, 8, ElementType.F32)
        .tile("t5", 8, 8, ElementType.F32)
        .tile("t6", 8, 8, ElementType.F32)
        .tile("t7", 8, 8, ElementType.F32)
        .tile("row_max", 8, 1, ElementType.F32)
        .tile("row_sum", 8, 1, ElementType.F32)
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        # Chain of operations (testing transitive edge removal)
        .load("t0", "input", 0, 0)           # 0
        .exp("t1", "t0")                      # 1 <- 0
        .rowmax("row_max", "t1")              # 2 <- 1
        .rowexpandsub("t2", "t1", "row_max")  # 3 <- 1, 2
        .exp("t3", "t2")                      # 4 <- 3
        .rowsum("row_sum", "t3")              # 5 <- 4
        .rowexpanddiv("t4", "t3", "row_sum")  # 6 <- 4, 5
        .mul("t5", "t4", "t0")                # 7 <- 6, 0 (long-distance dep)
        .add("t6", "t5", "t1")                # 8 <- 7, 1 (another long-distance)
        .neg("t7", "t6")                      # 9 <- 8
        .store("t7", "output", 0, 0)          # 10 <- 9
        .build())


def test_simplify_and_color():
    """Run tests on SimplifyAndColor method."""
    output_dir = os.path.join(os.path.dirname(__file__), "output_deps_test")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Testing PTOProgram.SimplifyAndColor()")
    print("=" * 60)
    
    # Test 1: Simple program
    print("\n--- Test 1: Simple dependency graph ---")
    prog1 = create_test_program()
    success1 = prog1.SimplifyAndColor(
        TOTAL_COLOR=8,
        output_dir=os.path.join(output_dir, "simple"),
        visualize=True,
        verbose=True
    )
    print(f"Coloring success: {success1}")
    
    # Dump ASM with deps
    asm1 = prog1.dump_pto_asm_with_deps(
        os.path.join(output_dir, "simple", "test_deps.pto")
    )
    print("\nPTO Assembly with dependencies:")
    print(asm1)
    
    # Test 2: Loop program
    print("\n--- Test 2: Loop-carried dependencies ---")
    prog2 = create_loop_program()
    success2 = prog2.SimplifyAndColor(
        TOTAL_COLOR=8,
        output_dir=os.path.join(output_dir, "loop"),
        visualize=True,
        verbose=True
    )
    print(f"Coloring success: {success2}")
    
    asm2 = prog2.dump_pto_asm_with_deps(
        os.path.join(output_dir, "loop", "test_loop_deps.pto")
    )
    print("\nPTO Assembly with dependencies:")
    print(asm2)
    
    # Test 3: Complex program
    print("\n--- Test 3: Complex dependency chain ---")
    prog3 = create_complex_program()
    success3 = prog3.SimplifyAndColor(
        TOTAL_COLOR=4,  # Use fewer colors to test coloring algorithm
        output_dir=os.path.join(output_dir, "complex"),
        visualize=True,
        verbose=True
    )
    print(f"Coloring success: {success3}")
    
    asm3 = prog3.dump_pto_asm_with_deps(
        os.path.join(output_dir, "complex", "test_complex.pto")
    )
    print("\nPTO Assembly with dependencies:")
    print(asm3)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Test 1 (Simple):  {'PASS' if success1 else 'FAIL'}")
    print(f"  Test 2 (Loop):    {'PASS' if success2 else 'FAIL'}")
    print(f"  Test 3 (Complex): {'PASS' if success3 else 'FAIL'}")
    print(f"\nOutput files written to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    test_simplify_and_color()
