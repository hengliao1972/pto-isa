#!/usr/bin/env python3
"""
Test Suite for PTO torch.nn.functional Implementation

Tests F.* functions against NumPy reference on ARM64.

Usage:
    python test_pto_torch_functional.py
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
    ref_relu, ref_relu6, ref_leaky_relu, ref_elu, ref_gelu, ref_silu,
    ref_sigmoid, ref_tanh, ref_softmax, ref_mse_loss, ref_l1_loss,
    DEFAULT_RTOL, DEFAULT_ATOL, IS_ARM64
)


# =============================================================================
# Code Loading
# =============================================================================

def load_arm64_code(name: str) -> str:
    """Load generated ARM64 C code for a functional API."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, "output_arm64", "torch_functional", f"{name}.c")
    
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Generated code not found: {code_path}")
    
    with open(code_path, 'r') as f:
        return f.read()


# =============================================================================
# Activation Function Tests
# =============================================================================

def test_F_relu():
    """Test F.relu."""
    c_code = load_arm64_code("F_relu")
    runner = PTOTestRunner(c_code, "F_relu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_relu)
    finally:
        runner.cleanup()


def test_F_relu6():
    """Test F.relu6."""
    c_code = load_arm64_code("F_relu6")
    runner = PTOTestRunner(c_code, "F_relu6", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-2.0, high=10.0, seed=42)
        return runner.run_test([x], ref_relu6)
    finally:
        runner.cleanup()


def test_F_leaky_relu():
    """Test F.leaky_relu."""
    c_code = load_arm64_code("F_leaky_relu")
    runner = PTOTestRunner(c_code, "F_leaky_relu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], lambda x: ref_leaky_relu(x, 0.01), rtol=1e-3)
    finally:
        runner.cleanup()


def test_F_elu():
    """Test F.elu."""
    c_code = load_arm64_code("F_elu")
    runner = PTOTestRunner(c_code, "F_elu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: ref_elu(x, 1.0), rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_F_gelu():
    """Test F.gelu."""
    c_code = load_arm64_code("F_gelu")
    runner = PTOTestRunner(c_code, "F_gelu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_gelu, rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_F_sigmoid():
    """Test F.sigmoid."""
    c_code = load_arm64_code("F_sigmoid")
    runner = PTOTestRunner(c_code, "F_sigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_sigmoid, rtol=1e-3)
    finally:
        runner.cleanup()


def test_F_silu():
    """Test F.silu (swish)."""
    c_code = load_arm64_code("F_silu")
    runner = PTOTestRunner(c_code, "F_silu", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_silu, rtol=1e-3)
    finally:
        runner.cleanup()


def test_F_tanh():
    """Test F.tanh."""
    c_code = load_arm64_code("F_tanh")
    runner = PTOTestRunner(c_code, "F_tanh", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_tanh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_F_softplus():
    """Test F.softplus: log(1 + exp(x))."""
    c_code = load_arm64_code("F_softplus")
    runner = PTOTestRunner(c_code, "F_softplus", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: np.log(1 + np.exp(x)), rtol=1e-3)
    finally:
        runner.cleanup()


def test_F_softmax():
    """Test F.softmax."""
    c_code = load_arm64_code("F_softmax")
    runner = PTOTestRunner(c_code, "F_softmax", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: ref_softmax(x, axis=-1), rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_F_log_softmax():
    """Test F.log_softmax."""
    c_code = load_arm64_code("F_log_softmax")
    runner = PTOTestRunner(c_code, "F_log_softmax", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: np.log(ref_softmax(x, axis=-1) + 1e-10), 
                               rtol=1e-2, atol=1e-2)
    finally:
        runner.cleanup()


def test_F_logsigmoid():
    """Test F.logsigmoid: log(sigmoid(x))."""
    c_code = load_arm64_code("F_logsigmoid")
    runner = PTOTestRunner(c_code, "F_logsigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: np.log(ref_sigmoid(x)), rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Loss Function Tests
# =============================================================================

def test_F_mse_loss():
    """Test F.mse_loss."""
    c_code = load_arm64_code("F_mse_loss")
    runner = PTOTestRunner(c_code, "F_mse_loss", ["input_pred", "input_target"], ["output"])
    try:
        pred = random_float32((8, 8), seed=42)
        target = random_float32((8, 8), seed=43)
        return runner.run_test([pred, target], ref_mse_loss)
    finally:
        runner.cleanup()


def test_F_l1_loss():
    """Test F.l1_loss."""
    c_code = load_arm64_code("F_l1_loss")
    runner = PTOTestRunner(c_code, "F_l1_loss", ["input_pred", "input_target"], ["output"])
    try:
        pred = random_float32((8, 8), seed=42)
        target = random_float32((8, 8), seed=43)
        return runner.run_test([pred, target], ref_l1_loss)
    finally:
        runner.cleanup()


def test_F_smooth_l1_loss():
    """Test F.smooth_l1_loss (Huber loss with beta=1)."""
    c_code = load_arm64_code("F_smooth_l1_loss")
    runner = PTOTestRunner(c_code, "F_smooth_l1_loss", ["input_pred", "input_target"], ["output"])
    try:
        pred = random_float32((8, 8), seed=42)
        target = random_float32((8, 8), seed=43)
        def ref_smooth_l1(pred, target, beta=1.0):
            diff = np.abs(pred - target)
            return np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return runner.run_test([pred, target], ref_smooth_l1, rtol=1e-2)
    finally:
        runner.cleanup()


# =============================================================================
# Normalization Tests
# =============================================================================

def test_F_normalize():
    """Test F.normalize (L2 normalization)."""
    c_code = load_arm64_code("F_normalize")
    runner = PTOTestRunner(c_code, "F_normalize", ["input"], ["output"])
    try:
        x = random_float32((8, 8), seed=42)
        def ref_normalize(x):
            norm = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
            return x / (norm + 1e-10)
        return runner.run_test([x], ref_normalize, rtol=1e-2)
    finally:
        runner.cleanup()


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all torch.nn.functional tests."""
    print("=" * 70)
    print("PTO torch.nn.functional Test Suite")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture.")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print()
    
    tests = [
        # Activations
        ("F_relu", test_F_relu),
        ("F_relu6", test_F_relu6),
        ("F_leaky_relu", test_F_leaky_relu),
        ("F_elu", test_F_elu),
        ("F_gelu", test_F_gelu),
        ("F_sigmoid", test_F_sigmoid),
        ("F_silu", test_F_silu),
        ("F_tanh", test_F_tanh),
        ("F_softplus", test_F_softplus),
        ("F_softmax", test_F_softmax),
        ("F_log_softmax", test_F_log_softmax),
        ("F_logsigmoid", test_F_logsigmoid),
        # Loss functions
        ("F_mse_loss", test_F_mse_loss),
        ("F_l1_loss", test_F_l1_loss),
        ("F_smooth_l1_loss", test_F_smooth_l1_loss),
        # Normalization
        ("F_normalize", test_F_normalize),
    ]
    
    return run_test_suite(tests, verbose=True)


if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if results["failed"] == 0 else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pto_torch_functional.py' first to generate the code.")
        sys.exit(1)
