#!/usr/bin/env python3
"""
Test Suite for PTO sinh() Taylor Expansion Implementation

Tests the sinh_taylor function against NumPy reference on ARM64.

Usage:
    python test_pto_isa_sinh.py
"""

import os
import sys
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_utils import (
    PTOTestRunner, TestResult, run_test_suite,
    random_small, ref_sinh,
    DEFAULT_RTOL, DEFAULT_ATOL, IS_ARM64
)


# =============================================================================
# Load Generated Code
# =============================================================================

def load_sinh_code() -> str:
    """Load the generated ARM64 C code for sinh."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, "output_arm64", "sinh_taylor", "sinh_taylor.c")
    
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Generated code not found: {code_path}\n"
                               "Run 'python pto_isa_sinh.py' first to generate it.")
    
    with open(code_path, 'r') as f:
        return f.read()


# =============================================================================
# Test Cases
# =============================================================================

def test_sinh_random_small():
    """Test sinh with small random values (Taylor expansion is accurate here)."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        # Small values where Taylor expansion converges well
        x = random_small((8, 8), low=-1.0, high=1.0, seed=42)
        result = runner.run_test([x], ref_sinh, rtol=1e-3, atol=1e-4)
        return result
    finally:
        runner.cleanup()


def test_sinh_zeros():
    """Test sinh(0) = 0."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        x = np.zeros((8, 8), dtype=np.float32)
        result = runner.run_test([x], ref_sinh, rtol=1e-5, atol=1e-6)
        return result
    finally:
        runner.cleanup()


def test_sinh_small_positive():
    """Test sinh with small positive values."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        x = random_small((8, 8), low=0.0, high=0.5, seed=123)
        result = runner.run_test([x], ref_sinh, rtol=1e-4, atol=1e-5)
        return result
    finally:
        runner.cleanup()


def test_sinh_small_negative():
    """Test sinh with small negative values (odd function: sinh(-x) = -sinh(x))."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        x = random_small((8, 8), low=-0.5, high=0.0, seed=456)
        result = runner.run_test([x], ref_sinh, rtol=1e-4, atol=1e-5)
        return result
    finally:
        runner.cleanup()


def test_sinh_moderate_values():
    """Test sinh with moderate values where Taylor expansion still works."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        # Moderate values - Taylor expansion with 7 terms is less accurate here
        x = random_small((8, 8), low=-2.0, high=2.0, seed=789)
        result = runner.run_test([x], ref_sinh, rtol=0.1, atol=0.01)
        return result
    finally:
        runner.cleanup()


def test_sinh_linspace():
    """Test sinh on linearly spaced values."""
    c_code = load_sinh_code()
    runner = PTOTestRunner(
        c_code, "sinh_taylor",
        input_names=["input"],
        output_names=["output"],
        rows=8, cols=8
    )
    
    try:
        x = np.linspace(-1.0, 1.0, 64).reshape(8, 8).astype(np.float32)
        result = runner.run_test([x], ref_sinh, rtol=1e-3, atol=1e-4)
        return result
    finally:
        runner.cleanup()


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all sinh tests."""
    print("=" * 70)
    print("PTO sinh() Taylor Expansion Test Suite")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture.")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print()
    
    tests = [
        ("sinh_random_small", test_sinh_random_small),
        ("sinh_zeros", test_sinh_zeros),
        ("sinh_small_positive", test_sinh_small_positive),
        ("sinh_small_negative", test_sinh_small_negative),
        ("sinh_moderate_values", test_sinh_moderate_values),
        ("sinh_linspace", test_sinh_linspace),
    ]
    
    return run_test_suite(tests, verbose=True)


if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if results["failed"] == 0 else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pto_isa_sinh.py' first to generate the code.")
        sys.exit(1)
