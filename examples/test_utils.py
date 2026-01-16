"""
PTO ARM64 Test Utilities

Provides utilities for testing PTO-generated ARM64 NEON code:
1. Compile C code to shared library
2. Load and call functions via ctypes
3. Compare results with NumPy reference

Requirements:
- ARM64 macOS or Linux with clang/gcc
- NumPy for reference implementations
"""

import os
import sys
import subprocess
import tempfile
import ctypes
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import platform


# =============================================================================
# Configuration
# =============================================================================

# Detect platform and set compiler
IS_ARM64 = platform.machine() in ('arm64', 'aarch64')
IS_MACOS = platform.system() == 'Darwin'

if IS_MACOS:
    CC = "clang"
    CFLAGS = ["-O2", "-fPIC", "-shared"]
    if IS_ARM64:
        CFLAGS.append("-arch")
        CFLAGS.append("arm64")
else:
    CC = "gcc"
    CFLAGS = ["-O2", "-fPIC", "-shared", "-lm"]
    if IS_ARM64:
        CFLAGS.append("-march=armv8-a+simd")

# Default tolerance for float32 comparisons
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


# =============================================================================
# Test Result
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    passed: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    error_message: str = ""
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        if self.passed:
            return f"[{status}] {self.name} (max_err={self.max_abs_error:.2e})"
        return f"[{status}] {self.name}: {self.error_message}"


# =============================================================================
# C Code Wrapper Generator
# =============================================================================

def generate_test_wrapper(c_code: str, func_name: str, 
                          inputs: List[Tuple[str, Tuple[int, ...]]],
                          outputs: List[Tuple[str, Tuple[int, ...]]]) -> str:
    """
    Generate a C wrapper that can be called via ctypes.
    
    Args:
        c_code: The generated ARM64 C code
        func_name: Name of the wrapper function to expose
        inputs: List of (name, shape) for input arrays
        outputs: List of (name, shape) for output arrays
    
    Returns:
        Complete C code with wrapper function
    """
    # Calculate sizes
    input_decls = []
    output_decls = []
    
    for name, shape in inputs:
        size = np.prod(shape)
        input_decls.append(f"float {name}[{size}];")
    
    for name, shape in outputs:
        size = np.prod(shape)
        output_decls.append(f"float {name}[{size}];")
    
    # Build wrapper function
    wrapper = f"""
// Test wrapper for {func_name}
#include <string.h>

// Input/output array declarations
{"".join(input_decls)}
{"".join(output_decls)}

{c_code}

// Exported wrapper function
void {func_name}_wrapper(
    {"".join(f"float* in_{name}, " for name, _ in inputs)}
    {"".join(f"float* out_{name}" + (", " if i < len(outputs)-1 else "") for i, (name, _) in enumerate(outputs))}
) {{
    // Copy inputs
    {"".join(f"memcpy({name}, in_{name}, {np.prod(shape)} * sizeof(float)); " for name, shape in inputs)}
    
    // Execute (the generated code operates on global arrays)
    // Note: The generated code already contains the computation
    
    // Copy outputs
    {"".join(f"memcpy(out_{name}, {name}, {np.prod(shape)} * sizeof(float)); " for name, shape in outputs)}
}}
"""
    return wrapper


def wrap_pto_code(c_code: str, func_name: str, 
                  input_names: List[str], output_names: List[str],
                  rows: int = 8, cols: int = 8) -> str:
    """
    Wrap PTO-generated C code with proper input/output handling.
    
    The generated code expects 'input' and 'output' arrays.
    We wrap it in a function with proper structure.
    """
    size = rows * cols
    
    # Build the complete C file
    wrapper = f"""// Test wrapper for {func_name}
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

"""
    
    # Add input array declarations as global arrays
    for name in input_names:
        wrapper += f"static float {name}[{size}];\n"
    
    # Add output array declarations  
    for name in output_names:
        wrapper += f"static float {name}[{size}];\n"
    
    wrapper += "\n"
    
    # Extract the tile/intermediate variable declarations from the generated code
    # and the computational code separately
    code_lines = c_code.split('\n')
    tile_declarations = []
    computational_code = []
    in_code = False
    
    for line in code_lines:
        stripped = line.strip()
        
        # Skip includes and comments at the beginning
        if stripped.startswith('#include') or stripped.startswith('// PTO Program') or stripped.startswith('// Auto-generated'):
            continue
        
        # Check if this is a tile declaration (float name[rows][cols];)
        if stripped.startswith('float ') and '[' in stripped and '];' in stripped and '=' not in stripped:
            # Convert 2D array declaration to static
            tile_declarations.append(f"static {stripped}")
            continue
        
        # Skip empty lines at the beginning
        if not in_code and stripped == '':
            continue
        
        # Everything else is computational code
        in_code = True
        computational_code.append(line)
    
    # Add tile declarations
    for decl in tile_declarations:
        wrapper += f"{decl}\n"
    
    wrapper += "\n"
    
    # Create the computation function that contains the actual computation
    input_params = ", ".join(f"float* in_{name}" for name in input_names)
    output_params = ", ".join(f"float* out_{name}" for name in output_names)
    all_params = input_params + (", " if input_params and output_params else "") + output_params
    
    wrapper += f"""// Internal computation function
static void {func_name}_compute(void) {{
"""
    
    # Add the computational code with proper indentation
    for line in computational_code:
        if line.strip():
            wrapper += f"    {line}\n"
        else:
            wrapper += "\n"
    
    wrapper += "}\n\n"
    
    # Add the exported wrapper function
    wrapper += f"""// Exported wrapper function
__attribute__((visibility("default")))
void {func_name}_run({all_params}) {{
    // Copy inputs
"""
    for name in input_names:
        wrapper += f"    memcpy({name}, in_{name}, {size} * sizeof(float));\n"
    
    wrapper += f"""
    // Execute computation
    {func_name}_compute();
    
    // Copy outputs
"""
    for name in output_names:
        wrapper += f"    memcpy(out_{name}, {name}, {size} * sizeof(float));\n"
    
    wrapper += "}\n"
    
    return wrapper


# =============================================================================
# Compiler
# =============================================================================

def compile_to_shared_lib(c_code: str, output_path: str = None) -> str:
    """
    Compile C code to a shared library.
    
    Args:
        c_code: The C source code
        output_path: Optional path for the .so/.dylib file
    
    Returns:
        Path to the compiled shared library
    """
    # Create temp directory if no output path specified
    if output_path is None:
        ext = ".dylib" if IS_MACOS else ".so"
        fd, output_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
    
    # Write source to temp file
    fd, src_path = tempfile.mkstemp(suffix=".c")
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(c_code)
        
        # Compile
        cmd = [CC] + CFLAGS + ["-o", output_path, src_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        
        return output_path
    finally:
        os.unlink(src_path)


# =============================================================================
# Function Loader
# =============================================================================

class PTOFunction:
    """Wrapper for a compiled PTO function."""
    
    def __init__(self, lib_path: str, func_name: str,
                 input_shapes: List[Tuple[int, ...]], 
                 output_shapes: List[Tuple[int, ...]]):
        """
        Load a compiled PTO function.
        
        Args:
            lib_path: Path to the shared library
            func_name: Name of the function to call
            input_shapes: List of input array shapes
            output_shapes: List of output array shapes
        """
        self.lib = ctypes.CDLL(lib_path)
        self.func_name = func_name
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        
        # Get function pointer
        self.func = getattr(self.lib, f"{func_name}_run")
        
        # Set argument types
        n_inputs = len(input_shapes)
        n_outputs = len(output_shapes)
        self.func.argtypes = [ctypes.POINTER(ctypes.c_float)] * (n_inputs + n_outputs)
        self.func.restype = None
    
    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        """
        Call the function with numpy arrays.
        
        Args:
            inputs: Input numpy arrays (float32)
        
        Returns:
            List of output numpy arrays
        """
        # Verify inputs
        assert len(inputs) == len(self.input_shapes), \
            f"Expected {len(self.input_shapes)} inputs, got {len(inputs)}"
        
        # Prepare input pointers
        input_ptrs = []
        for arr, shape in zip(inputs, self.input_shapes):
            arr = np.ascontiguousarray(arr.flatten(), dtype=np.float32)
            input_ptrs.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # Prepare output arrays
        outputs = []
        output_ptrs = []
        for shape in self.output_shapes:
            arr = np.zeros(np.prod(shape), dtype=np.float32)
            outputs.append(arr)
            output_ptrs.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # Call function
        self.func(*input_ptrs, *output_ptrs)
        
        # Reshape outputs
        return [arr.reshape(shape) for arr, shape in zip(outputs, self.output_shapes)]


# =============================================================================
# Test Runner
# =============================================================================

class PTOTestRunner:
    """
    Test runner for PTO-generated code.
    
    Compiles code, runs test cases, and compares with reference.
    """
    
    def __init__(self, c_code: str, func_name: str,
                 input_names: List[str], output_names: List[str],
                 rows: int = 8, cols: int = 8):
        """
        Initialize test runner.
        
        Args:
            c_code: Generated ARM64 C code
            func_name: Function name
            input_names: Names of input arrays
            output_names: Names of output arrays
            rows: Tile rows (default 8)
            cols: Tile columns (default 8)
        """
        self.func_name = func_name
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.input_names = input_names
        self.output_names = output_names
        
        # Wrap and compile
        wrapped_code = wrap_pto_code(c_code, func_name, input_names, output_names, rows, cols)
        
        try:
            self.lib_path = compile_to_shared_lib(wrapped_code)
            self.func = PTOFunction(
                self.lib_path, func_name,
                input_shapes=[(rows, cols)] * len(input_names),
                output_shapes=[(rows, cols)] * len(output_names)
            )
            self.compiled = True
        except Exception as e:
            print(f"Warning: Compilation failed for {func_name}: {e}")
            self.compiled = False
            self.func = None
    
    def run_test(self, inputs: List[np.ndarray], 
                 reference_fn: Callable[..., np.ndarray],
                 rtol: float = DEFAULT_RTOL,
                 atol: float = DEFAULT_ATOL) -> TestResult:
        """
        Run a single test case.
        
        Args:
            inputs: List of input arrays
            reference_fn: Function that computes expected output
            rtol: Relative tolerance
            atol: Absolute tolerance
        
        Returns:
            TestResult object
        """
        if not self.compiled:
            return TestResult(self.func_name, False, 
                            error_message="Compilation failed")
        
        try:
            # Get reference output
            expected = reference_fn(*inputs)
            if not isinstance(expected, list):
                expected = [expected]
            
            # Run PTO code
            actual = self.func(*inputs)
            
            # Compare outputs
            max_abs_err = 0.0
            max_rel_err = 0.0
            
            for exp, act in zip(expected, actual):
                exp = np.asarray(exp, dtype=np.float32)
                act = np.asarray(act, dtype=np.float32)
                
                abs_err = np.max(np.abs(exp - act))
                rel_err = np.max(np.abs(exp - act) / (np.abs(exp) + 1e-10))
                
                max_abs_err = max(max_abs_err, abs_err)
                max_rel_err = max(max_rel_err, rel_err)
            
            passed = np.allclose(expected[0], actual[0], rtol=rtol, atol=atol)
            
            return TestResult(
                self.func_name, passed,
                max_abs_error=max_abs_err,
                max_rel_error=max_rel_err,
                error_message="" if passed else f"Max error {max_abs_err:.2e} exceeds tolerance"
            )
            
        except Exception as e:
            return TestResult(self.func_name, False, error_message=str(e))
    
    def cleanup(self):
        """Remove temporary shared library."""
        if hasattr(self, 'lib_path') and os.path.exists(self.lib_path):
            try:
                os.unlink(self.lib_path)
            except:
                pass


# =============================================================================
# Test Data Generators
# =============================================================================

def random_float32(shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0,
                   seed: Optional[int] = None) -> np.ndarray:
    """Generate random float32 array."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, shape).astype(np.float32)


def random_positive(shape: Tuple[int, ...], low: float = 0.1, high: float = 2.0,
                    seed: Optional[int] = None) -> np.ndarray:
    """Generate random positive float32 array (for log, sqrt, etc.)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, shape).astype(np.float32)


def random_small(shape: Tuple[int, ...], low: float = -0.5, high: float = 0.5,
                 seed: Optional[int] = None) -> np.ndarray:
    """Generate random small values (for exp to avoid overflow)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, shape).astype(np.float32)


def random_nonzero(shape: Tuple[int, ...], low: float = 0.5, high: float = 2.0,
                   seed: Optional[int] = None) -> np.ndarray:
    """Generate random non-zero values (for division)."""
    if seed is not None:
        np.random.seed(seed)
    signs = np.random.choice([-1, 1], shape)
    values = np.random.uniform(low, high, shape)
    return (signs * values).astype(np.float32)


# =============================================================================
# Reference Implementations (NumPy-based)
# =============================================================================

def ref_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def ref_neg(x: np.ndarray) -> np.ndarray:
    return -x

def ref_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(x)

def ref_log(x: np.ndarray) -> np.ndarray:
    return np.log(x)

def ref_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(x)

def ref_rsqrt(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(x)

def ref_reciprocal(x: np.ndarray) -> np.ndarray:
    return 1.0 / x

def ref_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def ref_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ref_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def ref_sinh(x: np.ndarray) -> np.ndarray:
    return np.sinh(x)

def ref_cosh(x: np.ndarray) -> np.ndarray:
    return np.cosh(x)

def ref_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def ref_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def ref_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def ref_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / b

def ref_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

def ref_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def ref_pow(x: np.ndarray, p: float) -> np.ndarray:
    return np.power(x, p)

def ref_relu6(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0), 6)

def ref_leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, negative_slope * x)

def ref_elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def ref_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def ref_silu(x: np.ndarray) -> np.ndarray:
    return x * ref_sigmoid(x)

def ref_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def ref_mse_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (pred - target) ** 2

def ref_l1_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(pred - target)


# =============================================================================
# Test Suite Runner
# =============================================================================

def run_test_suite(tests: List[Tuple[str, Callable]], verbose: bool = True) -> Dict[str, Any]:
    """
    Run a suite of tests.
    
    Args:
        tests: List of (name, test_function) tuples
        verbose: Print results as they run
    
    Returns:
        Dictionary with test results summary
    """
    results = []
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
            
            if result.passed:
                passed += 1
            else:
                failed += 1
            
            if verbose:
                print(result)
                
        except Exception as e:
            result = TestResult(name, False, error_message=str(e))
            results.append(result)
            failed += 1
            
            if verbose:
                print(result)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Summary: {passed} passed, {failed} failed, {len(tests)} total")
        print(f"{'='*60}")
    
    return {
        "results": results,
        "passed": passed,
        "failed": failed,
        "total": len(tests)
    }


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == "__main__":
    print("PTO ARM64 Test Utilities")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Compiler: {CC}")
    print(f"Flags: {' '.join(CFLAGS)}")
    print(f"ARM64: {IS_ARM64}")
    
    if not IS_ARM64:
        print("\nWarning: Not running on ARM64. Tests may not work correctly.")
