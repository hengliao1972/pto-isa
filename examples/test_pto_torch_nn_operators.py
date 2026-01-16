#!/usr/bin/env python3
"""
Test Suite for PTO torch.nn Operators Implementation

Tests nn.* module classes against NumPy reference on ARM64.

Usage:
    python test_pto_torch_nn_operators.py
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
    """Load generated ARM64 C code for an nn operator."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(script_dir, "output_arm64", "torch_nn", f"{name}.c")
    
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"Generated code not found: {code_path}")
    
    with open(code_path, 'r') as f:
        return f.read()


# =============================================================================
# Activation Tests
# =============================================================================

def test_nn_ReLU():
    """Test nn.ReLU."""
    c_code = load_arm64_code("nn_ReLU")
    runner = PTOTestRunner(c_code, "nn_ReLU", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_relu)
    finally:
        runner.cleanup()


def test_nn_ReLU6():
    """Test nn.ReLU6."""
    c_code = load_arm64_code("nn_ReLU6")
    runner = PTOTestRunner(c_code, "nn_ReLU6", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-2.0, high=10.0, seed=42)
        return runner.run_test([x], ref_relu6)
    finally:
        runner.cleanup()


def test_nn_LeakyReLU():
    """Test nn.LeakyReLU."""
    c_code = load_arm64_code("nn_LeakyReLU")
    runner = PTOTestRunner(c_code, "nn_LeakyReLU", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], lambda x: ref_leaky_relu(x, 0.01), rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_ELU():
    """Test nn.ELU."""
    c_code = load_arm64_code("nn_ELU")
    runner = PTOTestRunner(c_code, "nn_ELU", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: ref_elu(x, 1.0), rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Sigmoid():
    """Test nn.Sigmoid."""
    c_code = load_arm64_code("nn_Sigmoid")
    runner = PTOTestRunner(c_code, "nn_Sigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        return runner.run_test([x], ref_sigmoid, rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Tanh():
    """Test nn.Tanh."""
    c_code = load_arm64_code("nn_Tanh")
    runner = PTOTestRunner(c_code, "nn_Tanh", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_tanh, rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_GELU():
    """Test nn.GELU."""
    c_code = load_arm64_code("nn_GELU")
    runner = PTOTestRunner(c_code, "nn_GELU", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_gelu, rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_nn_SiLU():
    """Test nn.SiLU (Swish)."""
    c_code = load_arm64_code("nn_SiLU")
    runner = PTOTestRunner(c_code, "nn_SiLU", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], ref_silu, rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Mish():
    """Test nn.Mish: x * tanh(softplus(x))."""
    c_code = load_arm64_code("nn_Mish")
    runner = PTOTestRunner(c_code, "nn_Mish", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        def ref_mish(x):
            return x * np.tanh(np.log(1 + np.exp(x)))
        return runner.run_test([x], ref_mish, rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Softmax():
    """Test nn.Softmax."""
    c_code = load_arm64_code("nn_Softmax")
    runner = PTOTestRunner(c_code, "nn_Softmax", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: ref_softmax(x, axis=-1), rtol=1e-2, atol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Softplus():
    """Test nn.Softplus: log(1 + exp(x))."""
    c_code = load_arm64_code("nn_Softplus")
    runner = PTOTestRunner(c_code, "nn_Softplus", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-3.0, high=3.0, seed=42)
        return runner.run_test([x], lambda x: np.log(1 + np.exp(x)), rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Hardsigmoid():
    """Test nn.Hardsigmoid: clip((x + 3) / 6, 0, 1)."""
    c_code = load_arm64_code("nn_Hardsigmoid")
    runner = PTOTestRunner(c_code, "nn_Hardsigmoid", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        def ref_hardsigmoid(x):
            return np.clip((x + 3) / 6, 0, 1)
        return runner.run_test([x], ref_hardsigmoid, rtol=1e-3)
    finally:
        runner.cleanup()


def test_nn_Hardswish():
    """Test nn.Hardswish: x * hardsigmoid(x)."""
    c_code = load_arm64_code("nn_Hardswish")
    runner = PTOTestRunner(c_code, "nn_Hardswish", ["input"], ["output"])
    try:
        x = random_float32((8, 8), low=-5.0, high=5.0, seed=42)
        def ref_hardswish(x):
            return x * np.clip((x + 3) / 6, 0, 1)
        return runner.run_test([x], ref_hardswish, rtol=1e-3)
    finally:
        runner.cleanup()


# =============================================================================
# Loss Function Tests
# =============================================================================

def test_nn_MSELoss():
    """Test nn.MSELoss."""
    c_code = load_arm64_code("nn_MSELoss")
    runner = PTOTestRunner(c_code, "nn_MSELoss", ["input_pred", "input_target"], ["output"])
    try:
        pred = random_float32((8, 8), seed=42)
        target = random_float32((8, 8), seed=43)
        return runner.run_test([pred, target], ref_mse_loss)
    finally:
        runner.cleanup()


def test_nn_L1Loss():
    """Test nn.L1Loss."""
    c_code = load_arm64_code("nn_L1Loss")
    runner = PTOTestRunner(c_code, "nn_L1Loss", ["input_pred", "input_target"], ["output"])
    try:
        pred = random_float32((8, 8), seed=42)
        target = random_float32((8, 8), seed=43)
        return runner.run_test([pred, target], ref_l1_loss)
    finally:
        runner.cleanup()


def test_nn_SmoothL1Loss():
    """Test nn.SmoothL1Loss."""
    c_code = load_arm64_code("nn_SmoothL1Loss")
    runner = PTOTestRunner(c_code, "nn_SmoothL1Loss", ["input_pred", "input_target"], ["output"])
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
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all torch.nn tests."""
    print("=" * 70)
    print("PTO torch.nn Operators Test Suite")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture.")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print()
    
    tests = [
        # Activations
        ("nn_ReLU", test_nn_ReLU),
        ("nn_ReLU6", test_nn_ReLU6),
        ("nn_LeakyReLU", test_nn_LeakyReLU),
        ("nn_ELU", test_nn_ELU),
        ("nn_Sigmoid", test_nn_Sigmoid),
        ("nn_Tanh", test_nn_Tanh),
        ("nn_GELU", test_nn_GELU),
        ("nn_SiLU", test_nn_SiLU),
        ("nn_Mish", test_nn_Mish),
        ("nn_Softmax", test_nn_Softmax),
        ("nn_Softplus", test_nn_Softplus),
        ("nn_Hardsigmoid", test_nn_Hardsigmoid),
        ("nn_Hardswish", test_nn_Hardswish),
        # Loss functions
        ("nn_MSELoss", test_nn_MSELoss),
        ("nn_L1Loss", test_nn_L1Loss),
        ("nn_SmoothL1Loss", test_nn_SmoothL1Loss),
    ]
    
    return run_test_suite(tests, verbose=True)


if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if results["failed"] == 0 else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pto_torch_nn_operators.py' first to generate the code.")
        sys.exit(1)
