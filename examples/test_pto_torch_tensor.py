#!/usr/bin/env python3
"""
Test Suite for PTO torch.Tensor Methods Implementation

Tests Tensor methods against NumPy reference on ARM64.

Usage:
    python test_pto_torch_tensor.py
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
    DEFAULT_RTOL, DEFAULT_ATOL, IS_ARM64
)


# =============================================================================
# Code Loading
# =============================================================================

def load_arm64_code(name: str) -> str:
    """Load generated ARM64 C code for a tensor method."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, "output_arm64", "torch_tensor", f"{name}.c")
    
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Generated code not found: {code_path}")
    
    with open(code_path, 'r') as f:
        return f.read()


# =============================================================================
# Arithmetic Tests
# =============================================================================

def test_tensor_add():
    """Test Tensor.add."""
    c_code = load_arm64_code("tensor_add")
    runner = PTOTestRunner(c_code, "tensor_add", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_add)
    finally:
        runner.cleanup()


def test_tensor_sub():
    """Test Tensor.sub."""
    c_code = load_arm64_code("tensor_sub")
    runner = PTOTestRunner(c_code, "tensor_sub", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_sub)
    finally:
        runner.cleanup()


def test_tensor_mul():
    """Test Tensor.mul."""
    c_code = load_arm64_code("tensor_mul")
    runner = PTOTestRunner(c_code, "tensor_mul", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_mul)
    finally:
        runner.cleanup()


def test_tensor_div():
    """Test Tensor.div."""
    c_code = load_arm64_code("tensor_div")
    runner = PTOTestRunner(c_code, "tensor_div", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_nonzero((8, 8), seed=43)
        return runner.run_test([a, b], ref_div)
    finally:
        runner.cleanup()


def test_tensor_neg():
    """Test Tensor.neg."""
    c_code = load_arm64_code("tensor_neg")
    runner = PTOTestRunner(c_code, "tensor_neg", ["input"], ["output"])
    try:
        x = random_float32((8, 8), seed=42)
        return runner.run_test([x], ref_neg)
    finally:
        runner.cleanup()


def test_tensor_abs():
    """Test Tensor.abs."""
    c_code = load_arm64_code("tensor_abs")
    runner = PTOTestRunner(c_code, "tensor_abs", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_abs)
    finally:
        runner.cleanup()


def test_tensor_sqrt():
    """Test Tensor.sqrt."""
    c_code = load_arm64_code("tensor_sqrt")
    runner = PTOTestRunner(c_code, "tensor_sqrt", ["input"], ["output"])
    try:
        x = random_positive((8, 8), seed=42)
        return runner.run_test([x], ref_sqrt)
    finally:
        runner.cleanup()


def test_tensor_rsqrt():
    """Test Tensor.rsqrt."""
    c_code = load_arm64_code("tensor_rsqrt")
    runner = PTOTestRunner(c_code, "tensor_rsqrt", ["input"], ["output"])
    try:
        x = random_positive((8, 8), low=0.5, high=10.0, seed=42)
        return runner.run_test([x], ref_rsqrt, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_reciprocal():
    """Test Tensor.reciprocal."""
    c_code = load_arm64_code("tensor_reciprocal")
    runner = PTOTestRunner(c_code, "tensor_reciprocal", ["input"], ["output"])
    try:
        x = random_nonzero((8, 8), seed=42)
        return runner.run_test([x], ref_reciprocal, rtol=1e-4)
    finally:
        runner.cleanup()


# =============================================================================
# Trigonometric Tests
# =============================================================================

def test_tensor_sin():
    """Test Tensor.sin."""
    c_code = load_arm64_code("tensor_sin")
    runner = PTOTestRunner(c_code, "tensor_sin", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.14, high=3.14, seed=42)
        return runner.run_test([x], np.sin, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_cos():
    """Test Tensor.cos."""
    c_code = load_arm64_code("tensor_cos")
    runner = PTOTestRunner(c_code, "tensor_cos", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.14, high=3.14, seed=42)
        return runner.run_test([x], np.cos, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_tan():
    """Test Tensor.tan."""
    c_code = load_arm64_code("tensor_tan")
    runner = PTOTestRunner(c_code, "tensor_tan", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-1.0, high=1.0, seed=42)
        return runner.run_test([x], np.tan, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_sinh():
    """Test Tensor.sinh."""
    c_code = load_arm64_code("tensor_sinh")
    runner = PTOTestRunner(c_code, "tensor_sinh", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)
        return runner.run_test([x], ref_sinh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_cosh():
    """Test Tensor.cosh."""
    c_code = load_arm64_code("tensor_cosh")
    runner = PTOTestRunner(c_code, "tensor_cosh", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)
        return runner.run_test([x], ref_cosh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_tanh():
    """Test Tensor.tanh."""
    c_code = load_arm64_code("tensor_tanh")
    runner = PTOTestRunner(c_code, "tensor_tanh", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_tanh, rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Exponential/Logarithmic Tests
# =============================================================================

def test_tensor_exp():
    """Test Tensor.exp."""
    c_code = load_arm64_code("tensor_exp")
    runner = PTOTestRunner(c_code, "tensor_exp", ["input"], ["output"])
    try:
        x = random_small((8, 8), low=-2.0, high=2.0, seed=42)
        return runner.run_test([x], ref_exp, rtol=1e-4)
    finally:
        runner.cleanup()


def test_tensor_log():
    """Test Tensor.log."""
    c_code = load_arm64_code("tensor_log")
    runner = PTOTestRunner(c_code, "tensor_log", ["input"], ["output"])
    try:
        x = random_positive((8, 8), seed=42)
        return runner.run_test([x], ref_log)
    finally:
        runner.cleanup()


def test_tensor_log2():
    """Test Tensor.log2."""
    c_code = load_arm64_code("tensor_log2")
    runner = PTOTestRunner(c_code, "tensor_log2", ["input"], ["output"])
    try:
        x = random_positive((8, 8), seed=42)
        return runner.run_test([x], np.log2, rtol=1e-3)
    finally:
        runner.cleanup()


def test_tensor_log10():
    """Test Tensor.log10."""
    c_code = load_arm64_code("tensor_log10")
    runner = PTOTestRunner(c_code, "tensor_log10", ["input"], ["output"])
    try:
        x = random_positive((8, 8), seed=42)
        return runner.run_test([x], np.log10, rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Activation Tests
# =============================================================================

def test_tensor_relu():
    """Test Tensor.relu."""
    c_code = load_arm64_code("tensor_relu")
    runner = PTOTestRunner(c_code, "tensor_relu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_relu)
    finally:
        runner.cleanup()


def test_tensor_sigmoid():
    """Test Tensor.sigmoid."""
    c_code = load_arm64_code("tensor_sigmoid")
    runner = PTOTestRunner(c_code, "tensor_sigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_sigmoid, rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Comparison Tests
# =============================================================================

def test_tensor_max_elementwise():
    """Test Tensor.max (elementwise)."""
    c_code = load_arm64_code("tensor_max_elementwise")
    runner = PTOTestRunner(c_code, "tensor_max_elementwise", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_max)
    finally:
        runner.cleanup()


def test_tensor_min_elementwise():
    """Test Tensor.min (elementwise)."""
    c_code = load_arm64_code("tensor_min_elementwise")
    runner = PTOTestRunner(c_code, "tensor_min_elementwise", ["input_self", "input_other"], ["output"])
    try:
        a = random_float32((8, 8), seed=42)
        b = random_float32((8, 8), seed=43)
        return runner.run_test([a, b], ref_min)
    finally:
        runner.cleanup()


def test_tensor_clamp():
    """Test Tensor.clamp."""
    c_code = load_arm64_code("tensor_clamp")
    runner = PTOTestRunner(c_code, "tensor_clamp", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], lambda x: np.clip(x, -1.0, 1.0))
    finally:
        runner.cleanup()


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all torch.Tensor tests."""
    print("=" * 70)
    print("PTO torch.Tensor Methods Test Suite")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture.")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print()
    
    tests = [
        # Arithmetic
        ("tensor_add", test_tensor_add),
        ("tensor_sub", test_tensor_sub),
        ("tensor_mul", test_tensor_mul),
        ("tensor_div", test_tensor_div),
        ("tensor_neg", test_tensor_neg),
        ("tensor_abs", test_tensor_abs),
        ("tensor_sqrt", test_tensor_sqrt),
        ("tensor_rsqrt", test_tensor_rsqrt),
        ("tensor_reciprocal", test_tensor_reciprocal),
        # Trigonometric
        ("tensor_sin", test_tensor_sin),
        ("tensor_cos", test_tensor_cos),
        ("tensor_tan", test_tensor_tan),
        ("tensor_sinh", test_tensor_sinh),
        ("tensor_cosh", test_tensor_cosh),
        ("tensor_tanh", test_tensor_tanh),
        # Exponential/Logarithmic
        ("tensor_exp", test_tensor_exp),
        ("tensor_log", test_tensor_log),
        ("tensor_log2", test_tensor_log2),
        ("tensor_log10", test_tensor_log10),
        # Activation
        ("tensor_relu", test_tensor_relu),
        ("tensor_sigmoid", test_tensor_sigmoid),
        # Comparison
        ("tensor_max_elementwise", test_tensor_max_elementwise),
        ("tensor_min_elementwise", test_tensor_min_elementwise),
        ("tensor_clamp", test_tensor_clamp),
    ]
    
    return run_test_suite(tests, verbose=True)


if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if results["failed"] == 0 else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pto_torch_tensor.py' first to generate the code.")
        sys.exit(1)
