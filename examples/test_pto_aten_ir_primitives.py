#!/usr/bin/env python3
"""
Test Suite for PTO ATen IR Primitives Implementation

Tests elementwise operations against NumPy reference on ARM64.

Usage:
    python test_pto_aten_ir_primitives.py
"""

import os
import sys
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_utils import (
    PTOTestRunner, TestResult, run_test_suite,
    random_float32, random_positive, random_small, random_nonzero,
    ref_abs, ref_neg, ref_exp, ref_log, ref_sqrt, ref_rsqrt, ref_reciprocal,
    ref_relu, ref_sigmoid, ref_tanh, ref_sinh, ref_cosh,
    ref_add, ref_sub, ref_mul, ref_div, ref_max, ref_min,
    ref_gelu, ref_silu,
    DEFAULT_RTOL, DEFAULT_ATOL, IS_ARM64
)


# =============================================================================
# Code Loading
# =============================================================================

def load_arm64_code(name: str) -> str:
    """Load generated ARM64 C code for a primitive."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, "output_arm64", "aten_primitives", f"{name}.c")
    
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Generated code not found: {code_path}")
    
    with open(code_path, 'r') as f:
        return f.read()


# =============================================================================
# Unary Operation Tests
# =============================================================================

def test_prims_abs():
    """Test prims.abs (absolute value)."""
    c_code = load_arm64_code("prims_abs")
    runner = PTOTestRunner(c_code, "prims_abs", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_abs)
    finally:
        runner.cleanup()


def test_prims_neg():
    """Test prims.neg (negation)."""
    c_code = load_arm64_code("prims_neg")
    runner = PTOTestRunner(c_code, "prims_neg", ["input"], ["output"])
    try:
        x = random_float32((8, 8), seed=42)
        return runner.run_test([x], ref_neg)
    finally:
        runner.cleanup()


def test_prims_exp():
    """Test prims.exp (exponential)."""
    c_code = load_arm64_code("prims_exp")
    runner = PTOTestRunner(c_code, "prims_exp", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)  # Avoid overflow
        return runner.run_test([x], ref_exp, rtol=1e-4, atol=1e-5)
    finally:
        runner.cleanup()


def test_prims_log():
    """Test prims.log (natural logarithm)."""
    c_code = load_arm64_code("prims_log")
    runner = PTOTestRunner(c_code, "prims_log", ["input"], ["output"])
    try:
        x = random_positive((8, 8), low=0.1, high=10.0, seed=42)
        return runner.run_test([x], ref_log)
    finally:
        runner.cleanup()


def test_prims_sqrt():
    """Test prims.sqrt (square root)."""
    c_code = load_arm64_code("prims_sqrt")
    runner = PTOTestRunner(c_code, "prims_sqrt", ["input"], ["output"])
    try:
        x = random_positive((8, 8), low=0.1, high=10.0, seed=42)
        return runner.run_test([x], ref_sqrt)
    finally:
        runner.cleanup()


def test_prims_rsqrt():
    """Test prims.rsqrt (reciprocal square root)."""
    c_code = load_arm64_code("prims_rsqrt")
    runner = PTOTestRunner(c_code, "prims_rsqrt", ["input"], ["output"])
    try:
        x = random_positive((8, 8), low=0.5, high=10.0, seed=42)
        return runner.run_test([x], ref_rsqrt, rtol=1e-3)
    finally:
        runner.cleanup()


def test_prims_reciprocal():
    """Test prims.reciprocal (1/x)."""
    c_code = load_arm64_code("prims_reciprocal")
    runner = PTOTestRunner(c_code, "prims_reciprocal", ["input"], ["output"])
    try:
        x = random_nonzero((8, 8), seed=42)
        return runner.run_test([x], ref_reciprocal, rtol=1e-4)
    finally:
        runner.cleanup()


def test_aten_relu():
    """Test aten.relu."""
    c_code = load_arm64_code("aten_relu")
    runner = PTOTestRunner(c_code, "aten_relu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_relu)
    finally:
        runner.cleanup()


def test_aten_sigmoid():
    """Test aten.sigmoid."""
    c_code = load_arm64_code("aten_sigmoid")
    runner = PTOTestRunner(c_code, "aten_sigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_sigmoid, rtol=1e-3)
    finally:
        runner.cleanup()


def test_aten_tanh():
    """Test aten.tanh."""
    c_code = load_arm64_code("aten_tanh")
    runner = PTOTestRunner(c_code, "aten_tanh", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_tanh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_aten_sinh():
    """Test aten.sinh."""
    c_code = load_arm64_code("aten_sinh")
    runner = PTOTestRunner(c_code, "aten_sinh", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)
        return runner.run_test([x], ref_sinh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_aten_cosh():
    """Test aten.cosh."""
    c_code = load_arm64_code("aten_cosh")
    runner = PTOTestRunner(c_code, "aten_cosh", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)
        return runner.run_test([x], ref_cosh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_aten_gelu():
    """Test aten.gelu."""
    c_code = load_arm64_code("aten_gelu")
    runner = PTOTestRunner(c_code, "aten_gelu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_gelu, rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_aten_silu():
    """Test aten.silu (swish)."""
    c_code = load_arm64_code("aten_silu")
    runner = PTOTestRunner(c_code, "aten_silu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_silu, rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Binary Operation Tests
# =============================================================================

def test_prims_add():
    """Test prims.add."""
    c_code = load_arm64_code("prims_add")
    runner = PTOTestRunner(c_code, "prims_add", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_add)
    finally:
        runner.cleanup()


def test_prims_sub():
    """Test prims.sub."""
    c_code = load_arm64_code("prims_sub")
    runner = PTOTestRunner(c_code, "prims_sub", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_sub)
    finally:
        runner.cleanup()


def test_prims_mul():
    """Test prims.mul."""
    c_code = load_arm64_code("prims_mul")
    runner = PTOTestRunner(c_code, "prims_mul", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_mul)
    finally:
        runner.cleanup()


def test_prims_div():
    """Test prims.div."""
    c_code = load_arm64_code("prims_div")
    runner = PTOTestRunner(c_code, "prims_div", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_nonzero((8, 8), seed=43)
        return runner.run_test([a, b], ref_div)
    finally:
        runner.cleanup()


def test_prims_maximum():
    """Test prims.maximum."""
    c_code = load_arm64_code("prims_maximum")
    runner = PTOTestRunner(c_code, "prims_maximum", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_max)
    finally:
        runner.cleanup()


def test_prims_minimum():
    """Test prims.minimum."""
    c_code = load_arm64_code("prims_minimum")
    runner = PTOTestRunner(c_code, "prims_minimum", ["input_a", "input_b"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_min)
    finally:
        runner.cleanup()


# =============================================================================
# Scalar Operation Tests  
# =============================================================================

def test_aten_add_scalar():
    """Test aten.add with scalar."""
    c_code = load_arm64_code("aten_add_scalar")
    runner = PTOTestRunner(c_code, "aten_add_scalar", ["input"], ["output"])
    try:
        x = random_float32((8, 8), seed=42)
        scalar = 2.0
        return runner.run_test([x], lambda x: x + scalar)
    finally:
        runner.cleanup()


def test_aten_mul_scalar():
    """Test aten.mul with scalar."""
    c_code = load_arm64_code("aten_mul_scalar")
    runner = PTOTestRunner(c_code, "aten_mul_scalar", ["input"], ["output"])
    try:
        x = random_float32((8, 8), seed=42)
        scalar = 2.0
        return runner.run_test([x], lambda x: x * scalar)
    finally:
        runner.cleanup()


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all ATen primitives tests."""
    print("=" * 70)
    print("PTO ATen IR Primitives Test Suite")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture.")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print()
    
    tests = [
        # Unary operations
        ("prims_abs", test_prims_abs),
        ("prims_neg", test_prims_neg),
        ("prims_exp", test_prims_exp),
        ("prims_log", test_prims_log),
        ("prims_sqrt", test_prims_sqrt),
        ("prims_rsqrt", test_prims_rsqrt),
        ("prims_reciprocal", test_prims_reciprocal),
        ("aten_relu", test_aten_relu),
        ("aten_sigmoid", test_aten_sigmoid),
        ("aten_tanh", test_aten_tanh),
        ("aten_sinh", test_aten_sinh),
        ("aten_cosh", test_aten_cosh),
        ("aten_gelu", test_aten_gelu),
        ("aten_silu", test_aten_silu),
        # Binary operations
        ("prims_add", test_prims_add),
        ("prims_sub", test_prims_sub),
        ("prims_mul", test_prims_mul),
        ("prims_div", test_prims_div),
        ("prims_maximum", test_prims_maximum),
        ("prims_minimum", test_prims_minimum),
        # Scalar operations
        ("aten_add_scalar", test_aten_add_scalar),
        ("aten_mul_scalar", test_aten_mul_scalar),
    ]
    
    return run_test_suite(tests, verbose=True)


if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if results["failed"] == 0 else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pto_aten_ir_primitives.py' first to generate the code.")
        sys.exit(1)
