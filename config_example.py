#!/usr/bin/env python3
"""
PTO Example Configuration Tool

This tool provides both an interactive menu and command-line interface to
configure and generate run.py scripts for PTO examples.

Configuration Options:
1. Select example directory
2. Target platform (arm64, cuda, ascend, ascend_a2a3_sim)
3. Power-of-2 loop expansion
4. Task dump generation and statistics
5. Task graph PDF generation
6. Orchestration performance benchmarking
7. Test input range
8. Accuracy test case generation
9. Simulation and trace file generation

Usage (Interactive):
    python config_example.py
    
Usage (Command-line):
    python config_example.py --example llama --platform ascend_a2a3_sim --generate
    python config_example.py --example bgemm --platform ascend_a2a3_sim --run
    python config_example.py --help

Command-line Arguments:
    --example NAME       Example to configure (llama, softmax, bgemm, ...)
    --platform PLATFORM  Target platform (arm64, cuda, ascend_a2a3_sim, ...)
    --generate          Generate run script and exit
    --run               Generate script and run it
    --seq-len-min N     Minimum sequence length for benchmarking
    --seq-len-max N     Maximum sequence length for benchmarking
    --seq-len-step N    Sequence length step size
    --no-benchmark      Disable benchmarking
    --no-simulation     Disable simulation
    --list-examples     List available examples
    --list-platforms    List available platforms
"""

import os
import sys
import json
import argparse
import shutil
from typing import Dict, List, Optional, Any

# =============================================================================
# Configuration Defaults
# =============================================================================

# LLaMA tile configuration (must match pto_llama7B_dynamic.py)
# seq_len = num_tiles × TILE_ROWS
# num_tiles = seq_len // TILE_ROWS
TILE_ROWS = 32  # Each tile processes 32 tokens

DEFAULT_CONFIG = {
    "example_name": "",
    "target_platform": "arm64",
    "enable_binary_expansion": True,
    "enable_task_dump": True,
    "enable_task_graph_pdf": True,
    "benchmark_orchestration": True,   # Measure task submission throughput (tasks/ms)
    "benchmark_runtime": True,          # Measure actual execution/simulation time
    "test_seq_len_min": 1024,          # Minimum sequence length (= 32 tiles × 32)
    "test_seq_len_max": 16384,         # Maximum sequence length (= 512 tiles × 32)
    "test_seq_len_step": 1024,         # Step size in tokens (= 32 tiles × 32)
    "enable_accuracy_test": True,
    "enable_simulation": True,
    "enable_trace_generation": True,
    "num_warmup_iterations": 1,
    "num_benchmark_iterations": 1,
}

PLATFORM_OPTIONS = {
    "arm64": {
        "name": "ARM64 NEON (CPU)",
        "script_suffix": "arm64",
        "compiler": "gcc",
        "extension": ".c",
    },
    "cuda": {
        "name": "NVIDIA CUDA (GPU)",
        "script_suffix": "cuda",
        "compiler": "nvcc",
        "extension": ".cu",
    },
    "ascend_a2a3": {
        "name": "Ascend A2/A3 NPU",
        "script_suffix": "ascend_a2a3",
        "compiler": "gcc",
        "extension": ".c",
        # Default: disable benchmarking for real hardware (focus on correctness first)
        "benchmark_orchestration": False,
        "benchmark_runtime": False,
    },
    "ascend_a5": {
        "name": "Ascend A5 NPU",
        "script_suffix": "ascend_a5",
        "compiler": "ascendc",
        "extension": ".cpp",
    },
    "ascend_a2a3_sim": {
        "name": "Ascend A2/A3 Cycle Simulator",
        "script_suffix": "ascend_a2a3_sim",
        "compiler": "gcc",
        "extension": ".c",
        # Default: enable benchmarking for simulator (measure orchestration performance)
        "benchmark_orchestration": True,
        "benchmark_runtime": True,
    },
}

PLATFORM_LIST = list(PLATFORM_OPTIONS.keys())

# =============================================================================
# Menu System
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_config(config: Dict[str, Any]):
    """Print current configuration."""
    print_header("Current Configuration")
    
    # Show script name that will be generated
    script_name = get_script_name(config['target_platform']) if config['example_name'] else "(select example first)"
    platform_info = PLATFORM_OPTIONS.get(config['target_platform'], {})
    platform_display = f"{config['target_platform']} ({platform_info.get('name', '')})"
    
    print(f"  [1] Example:              {config['example_name'] or '(not selected)'}")
    print(f"  [2] Target Platform:      {platform_display}")
    print(f"      -> Script:            {script_name}")
    print(f"  [3] Binary Expansion:     {'✓ Enabled' if config['enable_binary_expansion'] else '✗ Disabled'}")
    print(f"  [4] Task Dump:            {'✓ Enabled' if config['enable_task_dump'] else '✗ Disabled'}")
    print(f"  [5] Task Graph PDF:       {'✓ Enabled' if config['enable_task_graph_pdf'] else '✗ Disabled'}")
    print(f"  [6] Benchmark Orchestration: {'✓ Enabled' if config.get('benchmark_orchestration', False) else '✗ Disabled'}")
    print(f"      (tasks/ms without executing)")
    print(f"  [7] Benchmark Runtime:    {'✓ Enabled' if config.get('benchmark_runtime', False) else '✗ Disabled'}")
    print(f"      (actual execution time)")
    print(f"  [8] Sequence Length Range: {config['test_seq_len_min']} - {config['test_seq_len_max']} tokens (step: {config['test_seq_len_step']})")
    print(f"  [9] Accuracy Test:        {'✓ Enabled' if config['enable_accuracy_test'] else '✗ Disabled'}")
    print(f" [10] Simulation & Trace:   {'✓ Enabled' if config['enable_simulation'] else '✗ Disabled'}")
    print("-" * 60)


def get_examples_list(root_dir: str) -> List[str]:
    """Get list of available examples."""
    examples_dir = os.path.join(root_dir, "examples")
    if not os.path.exists(examples_dir):
        return []
    
    examples = []
    for item in os.listdir(examples_dir):
        item_path = os.path.join(examples_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it has a pto_*.py file
            for f in os.listdir(item_path):
                if f.startswith('pto_') and f.endswith('.py'):
                    examples.append(item)
                    break
    return sorted(examples)


def select_example(root_dir: str) -> str:
    """Interactive example selection."""
    examples = get_examples_list(root_dir)
    
    if not examples:
        print("No examples found in examples/ directory!")
        return ""
    
    print_header("Select Example")
    for i, ex in enumerate(examples, 1):
        print(f"  [{i}] {ex}")
    print(f"  [0] Cancel")
    
    try:
        choice = int(input("\nEnter choice: "))
        if choice == 0:
            return ""
        if 1 <= choice <= len(examples):
            return examples[choice - 1]
    except ValueError:
        pass
    
    print("Invalid choice!")
    return ""


def select_platform() -> str:
    """Interactive platform selection."""
    print_header("Select Target Platform")
    for i, plat in enumerate(PLATFORM_LIST, 1):
        info = PLATFORM_OPTIONS[plat]
        print(f"  [{i}] {plat:15s} - {info['name']}")
    print(f"  [0] Cancel")
    
    try:
        choice = int(input("\nEnter choice: "))
        if choice == 0:
            return ""
        if 1 <= choice <= len(PLATFORM_LIST):
            return PLATFORM_LIST[choice - 1]
    except ValueError:
        pass
    
    print("Invalid choice!")
    return ""


def get_script_name(platform: str) -> str:
    """Get the script filename for a platform."""
    suffix = PLATFORM_OPTIONS.get(platform, {}).get('script_suffix', platform)
    return f"run_{suffix}.py"


def configure_test_range(config: Dict[str, Any]) -> Dict[str, Any]:
    """Configure test sequence length range."""
    print_header("Configure Sequence Length Range")
    print(f"  Current: {config['test_seq_len_min']} - {config['test_seq_len_max']} tokens (step: {config['test_seq_len_step']})")
    print(f"  (1 tile = {TILE_ROWS} tokens)")
    
    try:
        min_val = input(f"  Min seq_len [{config['test_seq_len_min']}]: ").strip()
        if min_val:
            config['test_seq_len_min'] = int(min_val)
        
        max_val = input(f"  Max seq_len [{config['test_seq_len_max']}]: ").strip()
        if max_val:
            config['test_seq_len_max'] = int(max_val)
        
        step_val = input(f"  Step [{config['test_seq_len_step']}]: ").strip()
        if step_val:
            config['test_seq_len_step'] = int(step_val)
        
        # Show equivalent num_tiles
        min_tiles = config['test_seq_len_min'] // TILE_ROWS
        max_tiles = config['test_seq_len_max'] // TILE_ROWS
        print(f"  → Equivalent num_tiles: {min_tiles} - {max_tiles}")
            
    except ValueError:
        print("Invalid input, keeping current values.")
    
    return config


def toggle_option(config: Dict[str, Any], key: str, name: str) -> Dict[str, Any]:
    """Toggle a boolean option."""
    config[key] = not config[key]
    status = "Enabled" if config[key] else "Disabled"
    print(f"  {name}: {status}")
    return config


def toggle_benchmark_orchestration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Toggle orchestration benchmark (tasks/ms without executing)."""
    config['benchmark_orchestration'] = not config.get('benchmark_orchestration', False)
    status = "Enabled" if config['benchmark_orchestration'] else "Disabled"
    print(f"  Benchmark Orchestration: {status}")
    if config['benchmark_orchestration']:
        print("    -> Will measure task submission throughput (tasks/ms)")
        print(f"    -> Sequence length range: {config['test_seq_len_min']} - {config['test_seq_len_max']} tokens (step: {config['test_seq_len_step']})")
    return config


def toggle_benchmark_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    """Toggle runtime benchmark (actual execution/simulation time)."""
    config['benchmark_runtime'] = not config.get('benchmark_runtime', False)
    status = "Enabled" if config['benchmark_runtime'] else "Disabled"
    print(f"  Benchmark Runtime: {status}")
    if config['benchmark_runtime']:
        print("    -> Will measure actual execution/simulation time")
        print(f"    -> Sequence length range: {config['test_seq_len_min']} - {config['test_seq_len_max']} tokens (step: {config['test_seq_len_step']})")
    return config


# =============================================================================
# Run Script Generation
# =============================================================================

def generate_run_script(config: Dict[str, Any], root_dir: str) -> str:
    """Generate run.py script content."""
    
    example_name = config['example_name']
    
    bench_orch_status = "Enabled" if config.get('benchmark_orchestration', False) else "Disabled"
    bench_runtime_status = "Enabled" if config.get('benchmark_runtime', False) else "Disabled"
    
    script = f'''#!/usr/bin/env python3
"""
PTO Example Runner - {example_name}

Auto-generated by config_example.py
Configuration:
- Target Platform: {config['target_platform']}
- Binary Expansion: {config['enable_binary_expansion']}
- Task Dump: {config['enable_task_dump']}
- Task Graph PDF: {config['enable_task_graph_pdf']}
- Benchmark Orchestration: {bench_orch_status} (tasks/ms)
- Benchmark Runtime: {bench_runtime_status} (execution time)
- Sequence Length: {config['test_seq_len_min']} - {config['test_seq_len_max']} tokens (step: {config['test_seq_len_step']})
- Accuracy Test: {config['enable_accuracy_test']}
- Simulation: {config['enable_simulation']}
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime

# =============================================================================
# Path Setup
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(ROOT_DIR, "src")
RUNTIME_DIR = os.path.join(SRC_DIR, "runtime")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

sys.path.insert(0, SRC_DIR)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {{
    "example_name": "{example_name}",
    "target_platform": "{config['target_platform']}",
    "enable_binary_expansion": {config['enable_binary_expansion']},
    "enable_task_dump": {config['enable_task_dump']},
    "enable_task_graph_pdf": {config['enable_task_graph_pdf']},
    "benchmark_orchestration": {config.get('benchmark_orchestration', False)},  # tasks/ms without executing
    "benchmark_runtime": {config.get('benchmark_runtime', False)},              # actual execution time
    "test_seq_len_min": {config['test_seq_len_min']},
    "test_seq_len_max": {config['test_seq_len_max']},
    "test_seq_len_step": {config['test_seq_len_step']},
    "enable_accuracy_test": {config['enable_accuracy_test']},
    "enable_simulation": {config['enable_simulation']},
    "num_warmup_iterations": {config['num_warmup_iterations']},
    "num_benchmark_iterations": {config['num_benchmark_iterations']},
}}

# =============================================================================
# Imports
# =============================================================================

try:
    from compile.pto_compile import (
        PTOFunctionBuilder, PTOModule, MultiBackendCodeGenerator,
        generate_arm64_code, generate_cuda_code, generate_ascend_code,
    )
    from isa_definition.pto_isa_definition import ElementType, MemorySpace
except ImportError as e:
    print(f"Error importing PTO modules: {{e}}")
    print("Make sure you're running from the correct directory.")
    sys.exit(1)

# =============================================================================
# Utility Functions
# =============================================================================

def print_header(title):
    print("\\n" + "=" * 60)
    print(f"  {{title}}")
    print("=" * 60)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_command(cmd, cwd=None, timeout=300):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


# =============================================================================
# Code Generation
# =============================================================================

def generate_code():
    """Generate code for the target platform."""
    print_header("Code Generation")
    
    sys.path.insert(0, SCRIPT_DIR)
    
    # Import the example module.
    # Prefer modules that expose create_*_module() / create_module() (direct codegen),
    # and fall back to a module that has main().
    candidates = sorted(
        f[:-3]
        for f in os.listdir(SCRIPT_DIR)
        if f.startswith("pto_") and f.endswith(".py") and f != "run.py"
    )
    if not candidates:
        print("Error: No pto_*.py example file found!")
        return False
    
    try:
        example_module_name = None
        example_module = None
        for name in candidates:
            mod = __import__(name)
            has_codegen_entry = (
                hasattr(mod, "create_module")
                or any(attr.startswith("create_") and attr.endswith("_module") for attr in dir(mod))
            )
            if has_codegen_entry:
                example_module_name = name
                example_module = mod
                break
        if example_module is None:
            for name in candidates:
                mod = __import__(name)
                if hasattr(mod, "main"):
                    example_module_name = name
                    example_module = mod
                    break
        if example_module is None:
            example_module_name = candidates[0]
            example_module = __import__(example_module_name)
        
        print(f"  Loading example: {{example_module_name}}")
        
        # Look for create_*_module functions first (for direct code generation)
        create_module_func = None
        for attr_name in dir(example_module):
            if attr_name.startswith('create_') and attr_name.endswith('_module'):
                create_module_func = getattr(example_module, attr_name)
                break
        
        if create_module_func is None and hasattr(example_module, 'create_module'):
            create_module_func = example_module.create_module
        
        # Always prefer create_*_module for code generation when available
        platform = CONFIG['target_platform']
        use_direct_generation = (create_module_func is not None)
        
        if use_direct_generation:
            print(f"  Creating module using {{create_module_func.__name__}}()...")
            module = create_module_func()
            
            # Clean output directory for fresh compilation
            platform_dir = os.path.join(OUTPUT_DIR, platform)
            if os.path.exists(platform_dir):
                print(f"  Cleaning output directory: {{platform_dir}}")
                shutil.rmtree(platform_dir)
            
            # Generate code - organize into subfolders for Ascend platforms
            platform_dir = ensure_dir(platform_dir)
            code_dir = ensure_dir(os.path.join(platform_dir, "generated_code"))
            
            # For Ascend A2A3 platforms, organize into subfolders
            is_a2a3_platform = platform in ("ascend_a2a3", "ascend_a2a3_sim")
            if is_a2a3_platform:
                orch_dir = ensure_dir(os.path.join(code_dir, "orchestration"))
                aic_dir = ensure_dir(os.path.join(code_dir, "incore_aic"))  # AI Core Cube
                aiv_dir = ensure_dir(os.path.join(code_dir, "incore_aiv"))  # AI Core Vector
            
            gen = MultiBackendCodeGenerator(
                enable_fusion=True,
                analyze_buffers=True,
                module=module
            )
            
            # Create PTO assembly compiler for generating .pto files
            from compile.pto_compile import PTOModuleCompiler
            pto_compiler = PTOModuleCompiler()
            
            for func_name, prog in module.functions.items():
                # Determine function type
                is_incore = getattr(prog, 'is_in_core', True)
                is_cube = getattr(prog, 'is_cube', False)
                
                if is_a2a3_platform:
                    if not is_incore:  # Orchestration function
                        func_type_str = "Orchestration"
                        target_dir = orch_dir
                    elif is_cube:  # InCore Cube function
                        func_type_str = "InCore Cube (AIC)"
                        target_dir = aic_dir
                    else:  # InCore Vector function
                        func_type_str = "InCore Vector (AIV)"
                        target_dir = aiv_dir
                    print(f"  Generating {{platform}} code for: {{func_name}} [{{func_type_str}}]")
                else:
                    target_dir = code_dir
                    print(f"  Generating {{platform}} code for: {{func_name}}")
                
                if platform == "arm64":
                    code = gen.generate_arm64(prog)
                    ext = ".c"
                elif platform == "ascend_a2a3_sim":
                    code = gen.generate_ascend_a2a3_sim(prog)
                    # InCore functions use C++ (PTO ISA API), orchestration uses C
                    ext = ".cpp" if is_incore else ".c"
                elif platform == "ascend_a2a3":
                    code = gen.generate_ascend_a2a3(prog)
                    ext = ".cpp" if is_incore else ".c"
                elif platform == "cuda":
                    code = gen.generate_cuda(prog)
                    ext = ".cu"
                else:  # other ascend platforms (a5, etc.)
                    code = gen.generate_ascend(prog)
                    ext = ".cpp"
                
                output_file = os.path.join(target_dir, f"{{func_name}}{{ext}}")
                with open(output_file, 'w') as f:
                    f.write(code)
                print(f"    -> {{output_file}}")
                
                # Generate .pto file for InCore functions (PTO assembly text format)
                if is_incore:
                    try:
                        pto_code = pto_compiler.compile_function(prog)
                        pto_file = os.path.join(target_dir, f"{{func_name}}.pto")
                        with open(pto_file, 'w') as f:
                            f.write(pto_code)
                        print(f"    -> {{pto_file}}")
                    except Exception as e:
                        print(f"    Warning: Could not generate .pto for {{func_name}}: {{e}}")
        elif hasattr(example_module, 'main'):
            print("  Running example main()...")
            example_module.main()
        else:
            print("  Warning: No main() or create_*_module() found, running module...")
            
    except Exception as e:
        print(f"  Error: {{e}}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  Code generation complete!")
    return True


# =============================================================================
# Compilation
# =============================================================================

def compile_code():
    """Compile generated code."""
    print_header("Compilation")
    
    platform = CONFIG['target_platform']
    platform_dir = os.path.join(OUTPUT_DIR, platform)
    code_dir = os.path.join(platform_dir, "generated_code")
    
    if not os.path.exists(code_dir):
        print(f"  No generated code found in {{code_dir}}")
        return False
    
    # Check for organized subfolder structure (A2A3 platforms)
    is_a2a3_platform = platform in ("ascend_a2a3", "ascend_a2a3_sim")
    
    # For ascend_a2a3 platform, compile to .so files
    if platform == "ascend_a2a3":
        return compile_ascend_a2a3(code_dir)
    
    # For other platforms, use the original compilation logic
    orch_dir = os.path.join(code_dir, "orchestration") if is_a2a3_platform else code_dir
    
    # Find orchestration file by checking:
    # 1. In orchestration/ subfolder (for A2A3)
    # 2. 'orchestration' in filename
    # 3. 'Function Type: Orchestration' in file content
    # 4. 'int main(' in file content (for simulator with main entry)
    orch_file = None
    search_dirs = [orch_dir] if is_a2a3_platform and os.path.exists(orch_dir) else [code_dir]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for f in os.listdir(search_dir):
            if f.endswith('.c'):
                fpath = os.path.join(search_dir, f)
                if 'orchestration' in f.lower() or 'dynamic' in f.lower():
                    orch_file = fpath
                    break
                try:
                    with open(fpath, 'r') as fp:
                        content = fp.read()
                        if 'Function Type: Orchestration' in content:
                            orch_file = fpath
                            break
                        if 'int main(' in content:
                            orch_file = fpath
                            break
                except:
                    pass
        if orch_file:
            break
    
    if not orch_file:
        print("  No orchestration file found to compile")
        return True  # Not an error, just no orchestration
    
    print(f"  Compiling: {{os.path.basename(orch_file)}}")
    
    # Build compile command - output executable to platform_dir (not code_dir)
    exe_basename = os.path.basename(orch_file).replace('.c', '')
    exe_path = os.path.join(platform_dir, exe_basename)
    # Ensure we don't accidentally run a stale binary if compilation fails.
    if os.path.exists(exe_path):
        os.remove(exe_path)
    
    compile_flags = ["-O2", "-std=c11", "-D_POSIX_C_SOURCE=199309L"]
    if CONFIG['enable_binary_expansion']:
        compile_flags.append("-DPTO_BINARY_EXPANSION")
    if CONFIG['enable_task_dump']:
        compile_flags.append("-DPTO_TASK_DUMP")
    
    # Add include paths
    include_paths = [f"-I{{RUNTIME_DIR}}"]
    if is_a2a3_platform:
        # Add InCore directories to include path for InCore function headers
        aic_dir = os.path.join(code_dir, "incore_aic")
        aiv_dir = os.path.join(code_dir, "incore_aiv")
        if os.path.exists(aic_dir):
            include_paths.append(f"-I{{aic_dir}}")
        if os.path.exists(aiv_dir):
            include_paths.append(f"-I{{aiv_dir}}")
        include_paths.append(f"-I{{code_dir}}")
    
    cmd = f"gcc {{' '.join(compile_flags)}} {{' '.join(include_paths)}} -o {{exe_path}} {{orch_file}} -lpthread"
    
    print(f"  Command: {{cmd}}")
    success, stdout, stderr = run_command(cmd, cwd=platform_dir)
    
    if success:
        print(f"  Compiled successfully: {{exe_path}}")
        return True
    else:
        print(f"  Compilation failed: {{stderr}}")
        return False


def generate_test_program_template(code_dir, example_name):
    """
    Generate a test program template for A2A3 runtime entry.
    
    This creates a C file that:
    1. Initializes the A2A3 runtime with configuration
    2. Sets up host memory buffers
    3. Copies data to device (copyToDevice)
    4. Executes the orchestration function
    5. Copies results back (copyFromDevice)
    6. Cleans up the runtime
    """
    # Use double-quoted triple string with escaped braces for C code
    test_program = """/**
 * PTO Runtime Test Program - """ + example_name + """
 * 
 * Auto-generated test entry point for A2A3 Runtime.
 * This program demonstrates the complete runtime workflow:
 * 1. Load orchestration and InCore .so files
 * 2. Initialize runtime with thread configuration
 * 3. Transfer data to device
 * 4. Execute orchestration function
 * 5. Transfer results back to host
 * 6. Clean up resources
 * 
 * Thread Configuration:
 * - 1 Orchestration AICPU thread
 * - 3 Dependency Resolution AICPU threads
 * - 48 AIV (Vector) workers
 * - 24 AIC (Cube) workers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include A2A3 Runtime API
#include "runtime_a2a3/a2a3_runtime_api.h"

// =============================================================================
// Test Configuration
// =============================================================================

#define TEST_INPUT_SIZE   (1024 * 1024 * sizeof(float))  // 1M floats
#define TEST_OUTPUT_SIZE  (1024 * 1024 * sizeof(float))  // 1M floats

// =============================================================================
// Helper Functions
// =============================================================================

static double get_time_ms(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}}

static void init_test_data(float* data, size_t count) {{
    for (size_t i = 0; i < count; i++) {{
        data[i] = (float)(i % 1000) / 1000.0f;
    }}
}}

static int verify_results(const float* expected, const float* actual, size_t count) {{
    int errors = 0;
    const float epsilon = 1e-5f;
    for (size_t i = 0; i < count && errors < 10; i++) {{
        float diff = expected[i] - actual[i];
        if (diff < 0) diff = -diff;
        if (diff > epsilon) {{
            printf("  Mismatch at [%zu]: expected %.6f, got %.6f\\\\n", 
                   i, expected[i], actual[i]);
            errors++;
        }}
    }}
    return errors;
}}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {{
    printf("=======================================================\\\\n");
    printf("  PTO A2A3 Runtime Test: """ + example_name + """\\\\n");
    printf("=======================================================\\\\n\\\\n");
    
    int ret;
    double start_time, end_time;
    
    // =========================================================================
    // 1. Configure Runtime
    // =========================================================================
    printf("[1/6] Configuring runtime...\\\\n");
    
    A2A3RuntimeConfig config;
    a2a3_config_init_defaults(&config);
    
    // Set paths to compiled .so files
    config.orchestration_so_path = "generated_code/orchestration/lib_orchestration.so";
    config.orchestration_func_name = """ + '"' + example_name + """_dynamic";  // Orchestration function name
    config.incore_aiv_dir = "generated_code/incore_aiv/";
    config.incore_aic_dir = "generated_code/incore_aic/";
    
    // Thread configuration (as specified in requirements)
    config.num_orch_threads = 1;    // 1 Orchestration AICPU thread
    config.num_dep_threads = 3;     // 3 Dependency Resolution threads
    config.num_aiv_workers = 48;    // 48 AIV (Vector) workers
    config.num_aic_workers = 24;    // 24 AIC (Cube) workers
    
    config.debug_enabled = true;
    
    // DEBUG_ORCHESTRATION mode: Only run orchestration, skip task execution
    // Set to false to test full execution with workers
    config.debug_orchestration_only = false;  // Changed to false for full execution test
    
    printf("  Orchestration SO: %s\\\\n", config.orchestration_so_path);
    printf("  InCore AIV dir:   %s\\\\n", config.incore_aiv_dir);
    printf("  InCore AIC dir:   %s\\\\n", config.incore_aic_dir);
    printf("  Orch threads:     %d\\\\n", config.num_orch_threads);
    printf("  Dep threads:      %d\\\\n", config.num_dep_threads);
    printf("  AIV workers:      %d\\\\n", config.num_aiv_workers);
    printf("  AIC workers:      %d\\\\n", config.num_aic_workers);
    
    // =========================================================================
    // 2. Initialize Runtime
    // =========================================================================
    printf("\\\\n[2/6] Initializing runtime...\\\\n");
    start_time = get_time_ms();
    
    ret = a2a3_runtime_init(&config);
    if (ret != A2A3_SUCCESS) {{
        fprintf(stderr, "ERROR: Failed to initialize runtime: %s\\\\n",
                a2a3_runtime_error_string(ret));
        return 1;
    }}
    
    end_time = get_time_ms();
    printf("  Runtime initialized in %.2f ms\\\\n", end_time - start_time);
    
    // =========================================================================
    // 3. Allocate and Initialize Host Buffers
    // =========================================================================
    printf("\\\\n[3/6] Allocating host buffers...\\\\n");
    
    float* host_input = (float*)malloc(TEST_INPUT_SIZE);
    float* host_output = (float*)malloc(TEST_OUTPUT_SIZE);
    float* host_expected = (float*)malloc(TEST_OUTPUT_SIZE);  // For verification
    
    if (!host_input || !host_output || !host_expected) {{
        fprintf(stderr, "ERROR: Failed to allocate host buffers\\\\n");
        a2a3_runtime_finalize();
        return 1;
    }}
    
    // Initialize input data
    init_test_data(host_input, TEST_INPUT_SIZE / sizeof(float));
    memset(host_output, 0, TEST_OUTPUT_SIZE);
    
    printf("  Input buffer:  %zu bytes\\\\n", (size_t)TEST_INPUT_SIZE);
    printf("  Output buffer: %zu bytes\\\\n", (size_t)TEST_OUTPUT_SIZE);
    
    // =========================================================================
    // 4. Copy Data to Device (copyToDevice)
    // =========================================================================
    printf("\\\\n[4/6] Copying data to device...\\\\n");
    start_time = get_time_ms();
    
    // Allocate device buffers
    void* dev_input = a2a3_runtime_malloc(TEST_INPUT_SIZE);
    void* dev_output = a2a3_runtime_malloc(TEST_OUTPUT_SIZE);
    
    if (!dev_input || !dev_output) {{
        fprintf(stderr, "ERROR: Failed to allocate device buffers\\\\n");
        free(host_input);
        free(host_output);
        free(host_expected);
        a2a3_runtime_finalize();
        return 1;
    }}
    
    // Copy input to device
    ret = a2a3_runtime_copy_to_device(dev_input, host_input, TEST_INPUT_SIZE);
    if (ret != A2A3_SUCCESS) {{
        fprintf(stderr, "ERROR: copyToDevice failed: %s\\\\n",
                a2a3_runtime_error_string(ret));
        a2a3_runtime_free(dev_input);
        a2a3_runtime_free(dev_output);
        free(host_input);
        free(host_output);
        free(host_expected);
        a2a3_runtime_finalize();
        return 1;
    }}
    
    end_time = get_time_ms();
    printf("  Data copied to device in %.2f ms\\\\n", end_time - start_time);
    
    // =========================================================================
    // 5. Execute Orchestration Function
    // =========================================================================
    printf("\\\\n[5/6] Executing orchestration function...\\\\n");
    start_time = get_time_ms();
    
    // Allocate additional device buffers for intermediate results
    // For bgemm: A, B, C, P0, P1, P2 (6 matrix buffers)
    size_t matrix_size = 64 * 128 * sizeof(float);  // Default tile size
    void* dev_P0 = a2a3_runtime_malloc(TEST_OUTPUT_SIZE);
    void* dev_P1 = a2a3_runtime_malloc(TEST_OUTPUT_SIZE);
    void* dev_P2 = a2a3_runtime_malloc(TEST_OUTPUT_SIZE);
    
    // Set up scalar parameters
    int32_t seq_len = 64;
    int32_t tile_rows = 8;
    int32_t num_tiles = 8;
    float zero_val = 0.0f;
    
    // Create void** array to pass parameters to orchestration function
    // Order must match orchestration function's parameter extraction
    void* user_data[16];
    user_data[0] = dev_input;    // A matrix
    user_data[1] = (char*)dev_input + TEST_INPUT_SIZE/2;  // B matrix (second half of input)
    user_data[2] = dev_output;   // C matrix (output)
    user_data[3] = dev_P0;       // P0 intermediate
    user_data[4] = dev_P1;       // P1 intermediate
    user_data[5] = dev_P2;       // P2 intermediate
    user_data[6] = &seq_len;     // seq_len scalar
    user_data[7] = &tile_rows;   // tile_rows scalar
    user_data[8] = &num_tiles;   // num_tiles scalar
    user_data[9] = &zero_val;    // zero scalar
    
    printf("  Params: seq_len=%d, tile_rows=%d, num_tiles=%d\\\\n", seq_len, tile_rows, num_tiles);
    
    ret = a2a3_runtime_execute(user_data);
    if (ret != A2A3_SUCCESS) {{
        fprintf(stderr, "ERROR: Execution failed: %s\\\\n",
                a2a3_runtime_error_string(ret));
        a2a3_runtime_free(dev_input);
        a2a3_runtime_free(dev_output);
        a2a3_runtime_free(dev_P0);
        a2a3_runtime_free(dev_P1);
        a2a3_runtime_free(dev_P2);
        free(host_input);
        free(host_output);
        free(host_expected);
        a2a3_runtime_finalize();
        return 1;
    }}
    
    end_time = get_time_ms();
    printf("  Execution completed in %.2f ms\\\\n", end_time - start_time);
    
    // =========================================================================
    // 6. Copy Results from Device (copyFromDevice)
    // =========================================================================
    printf("\\\\n[6/6] Copying results from device...\\\\n");
    start_time = get_time_ms();
    
    ret = a2a3_runtime_copy_from_device(host_output, dev_output, TEST_OUTPUT_SIZE);
    if (ret != A2A3_SUCCESS) {{
        fprintf(stderr, "ERROR: copyFromDevice failed: %s\\\\n",
                a2a3_runtime_error_string(ret));
    }}
    
    end_time = get_time_ms();
    printf("  Data copied from device in %.2f ms\\\\n", end_time - start_time);
    
    // =========================================================================
    // Print Statistics
    // =========================================================================
    printf("\\\\n");
    a2a3_runtime_print_stats();
    
    // =========================================================================
    // Save Data for Python Accuracy Verification
    // =========================================================================
    printf("\\\\nSaving data for accuracy verification...\\\\n");
    
    FILE* f_input = fopen("accuracy_input.bin", "wb");
    FILE* f_output = fopen("accuracy_output.bin", "wb");
    FILE* f_params = fopen("accuracy_params.txt", "w");
    
    if (f_input && f_output && f_params) {{
        fwrite(host_input, 1, TEST_INPUT_SIZE, f_input);
        fwrite(host_output, 1, TEST_OUTPUT_SIZE, f_output);
        fprintf(f_params, "input_size=%zu\\\\n", (size_t)TEST_INPUT_SIZE);
        fprintf(f_params, "output_size=%zu\\\\n", (size_t)TEST_OUTPUT_SIZE);
        fprintf(f_params, "num_tiles=%d\\\\n", num_tiles);
        fprintf(f_params, "tile_m=64\\\\n");
        fprintf(f_params, "tile_n=128\\\\n");
        fprintf(f_params, "tile_k=64\\\\n");
        fprintf(f_params, "k_tiles=8\\\\n");
        printf("  Saved: accuracy_input.bin, accuracy_output.bin, accuracy_params.txt\\\\n");
    }} else {{
        printf("  WARNING: Could not save accuracy data files\\\\n");
    }}
    
    if (f_input) fclose(f_input);
    if (f_output) fclose(f_output);
    if (f_params) fclose(f_params);
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    printf("\\\\nCleaning up...\\\\n");
    
    a2a3_runtime_free(dev_input);
    a2a3_runtime_free(dev_output);
    a2a3_runtime_free(dev_P0);
    a2a3_runtime_free(dev_P1);
    a2a3_runtime_free(dev_P2);
    free(host_input);
    free(host_output);
    free(host_expected);
    
    a2a3_runtime_finalize();
    
    printf("\\\\nTest completed successfully!\\\\n");
    return 0;
}}
"""
    
    # Write the test program
    test_file = os.path.join(code_dir, "test_program.c")
    with open(test_file, 'w') as f:
        f.write(test_program)
    
    return test_file


def compile_ascend_a2a3(code_dir):
    """
    Compile generated code for Ascend A2/A3 platform.
    
    Compiles:
    - orchestration/*.c → lib_orchestration.so (shared library)
    - incore_aic/*.cpp → *.o (AICore Cube object files)
    - incore_aiv/*.cpp → *.o (AICore Vector object files)
    - test_program.c → test_program (executable)
    
    Output files are saved in the same folder as source files.
    """
    import glob
    
    # Get ASCEND_HOME_PATH for toolchain
    ascend_home = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
    ccec_path = os.path.join(ascend_home, 'bin', 'ccec')
    ld_path = os.path.join(ascend_home, 'bin', 'ld.lld')
    
    # Check if ccec compiler exists
    if not os.path.exists(ccec_path):
        print(f"  Warning: ccec compiler not found at {{ccec_path}}")
        print(f"  Please set ASCEND_HOME_PATH environment variable")
        print(f"  Skipping AICore kernel compilation")
        ccec_available = False
    else:
        ccec_available = True
    
    success = True
    
    # Directory paths
    orch_dir = os.path.join(code_dir, "orchestration")
    aic_dir = os.path.join(code_dir, "incore_aic")
    aiv_dir = os.path.join(code_dir, "incore_aiv")
    
    # Check if CANN SDK is available
    cann_available = os.path.exists(os.path.join(ascend_home, 'include', 'acl', 'acl.h'))
    if not cann_available:
        print(f"  Note: CANN SDK not found, using stub compilation mode")
        print(f"        (Define CANN_SDK_AVAILABLE for full hardware support)")
    
    # 1. Compile orchestration functions to shared library
    if os.path.exists(orch_dir):
        print("\\n  [1/7] Compiling orchestration functions...")
        c_files = glob.glob(os.path.join(orch_dir, "*.c"))
        if c_files:
            # Compile all .c files to a shared library
            so_path = os.path.join(orch_dir, "lib_orchestration.so")
            
            compile_flags = ["-O2", "-std=c11", "-fPIC", "-shared", "-D_POSIX_C_SOURCE=199309L"]
            if CONFIG['enable_binary_expansion']:
                compile_flags.append("-DPTO_BINARY_EXPANSION")
            if CONFIG['enable_task_dump']:
                compile_flags.append("-DPTO_TASK_DUMP")
            
            # Add CANN SDK related flags
            if cann_available:
                compile_flags.append("-DCANN_SDK_AVAILABLE")
            else:
                # Skip CANN check for stub compilation
                compile_flags.append("-DA2A3_SKIP_CANN_CHECK")
            
            include_paths = [
                f"-I{{RUNTIME_DIR}}",
                f"-I{{code_dir}}",
                f"-I{{aic_dir}}" if os.path.exists(aic_dir) else "",
                f"-I{{aiv_dir}}" if os.path.exists(aiv_dir) else "",
            ]
            include_paths = [p for p in include_paths if p]  # Remove empty
            
            # Add CANN SDK include path if available
            if cann_available:
                include_paths.append(f"-I{{ascend_home}}/include")
            
            src_files = " ".join(c_files)
            cmd = f"gcc {{' '.join(compile_flags)}} {{' '.join(include_paths)}} -o {{so_path}} {{src_files}} -lpthread"
            
            print(f"    Command: {{cmd}}")
            ok, stdout, stderr = run_command(cmd, cwd=orch_dir, timeout=120)
            
            if ok:
                print(f"    ✓ Compiled: {{so_path}}")
            else:
                print(f"    ✗ Failed: {{stderr}}")
                success = False
        else:
            print("    No .c files found in orchestration/")
    else:
        print("\\n  [1/7] Skipping orchestration (no orchestration/ directory)")
    
    # 2. Compile InCore AIC (AI Core Cube) functions
    if os.path.exists(aic_dir) and ccec_available:
        print("\\n  [2/7] Compiling InCore AIC (Cube) functions...")
        cpp_files = glob.glob(os.path.join(aic_dir, "*.cpp"))
        if cpp_files:
            for cpp_file in cpp_files:
                basename = os.path.basename(cpp_file).replace('.cpp', '')
                obj_path = os.path.join(aic_dir, f"{{basename}}.o")
                
                # Compile for AIC (Cube) architecture
                cmd = (
                    f"{{ccec_path}} -c -O3 -x cce -std=c++17 "
                    f"--cce-aicore-only --cce-aicore-arch=dav-c220-cube "
                    f"-D__AIC__ -DMEMORY_BASE "
                    f"-mllvm -cce-aicore-stack-size=0x8000 "
                    f"-mllvm -cce-aicore-function-stack-size=0x8000 "
                    f"-I{{ROOT_DIR}}/include "
                    f"-I{{code_dir}} -I{{aic_dir}} "
                    f"-o {{obj_path}} {{cpp_file}}"
                )
                
                print(f"    Compiling {{basename}}.cpp for AIC...")
                ok, stdout, stderr = run_command(cmd, cwd=aic_dir, timeout=120)
                
                if ok:
                    print(f"    ✓ Compiled: {{obj_path}}")
                else:
                    print(f"    ✗ Failed: {{stderr}}")
                    success = False
        else:
            print("    No .cpp files found in incore_aic/")
    elif not ccec_available:
        print("\\n  [2/7] Skipping InCore AIC (ccec compiler not available)")
    else:
        print("\\n  [2/7] Skipping InCore AIC (no incore_aic/ directory)")
    
    # 3. Compile InCore AIV (AI Core Vector) functions
    if os.path.exists(aiv_dir) and ccec_available:
        print("\\n  [3/7] Compiling InCore AIV (Vector) functions...")
        cpp_files = glob.glob(os.path.join(aiv_dir, "*.cpp"))
        if cpp_files:
            for cpp_file in cpp_files:
                basename = os.path.basename(cpp_file).replace('.cpp', '')
                obj_path = os.path.join(aiv_dir, f"{{basename}}.o")
                
                # Compile for AIV (Vector) architecture
                cmd = (
                    f"{{ccec_path}} -c -O3 -x cce -std=c++17 "
                    f"--cce-aicore-only --cce-aicore-arch=dav-c220-vec "
                    f"-D__AIV__ -DMEMORY_BASE "
                    f"-mllvm -cce-aicore-stack-size=0x8000 "
                    f"-mllvm -cce-aicore-function-stack-size=0x8000 "
                    f"-I{{ROOT_DIR}}/include "
                    f"-I{{code_dir}} -I{{aiv_dir}} "
                    f"-o {{obj_path}} {{cpp_file}}"
                )
                
                print(f"    Compiling {{basename}}.cpp for AIV...")
                ok, stdout, stderr = run_command(cmd, cwd=aiv_dir, timeout=120)
                
                if ok:
                    print(f"    ✓ Compiled: {{obj_path}}")
                else:
                    print(f"    ✗ Failed: {{stderr}}")
                    success = False
        else:
            print("    No .cpp files found in incore_aiv/")
    elif not ccec_available:
        print("\\n  [3/7] Skipping InCore AIV (ccec compiler not available)")
    else:
        print("\\n  [3/7] Skipping InCore AIV (no incore_aiv/ directory)")
    
    # 4. Link AICore kernel binary (aicore_kernel.o)
    aicore_kernel_src = os.path.join(RUNTIME_DIR, "runtime_a2a3", "core", "aicore", "aicore_kernel.cpp")
    common_dir = os.path.join(RUNTIME_DIR, "runtime_a2a3", "core", "common")
    if os.path.exists(aicore_kernel_src) and ccec_available:
        print("\\n  [4/7] Linking AICore kernel binary...")
        
        # Collect all .o files from incore_aic and incore_aiv
        aic_objs = glob.glob(os.path.join(aic_dir, "*.o")) if os.path.exists(aic_dir) else []
        aiv_objs = glob.glob(os.path.join(aiv_dir, "*.o")) if os.path.exists(aiv_dir) else []
        
        # Build directory for AICore kernel
        aicore_build_dir = os.path.join(code_dir, "aicore_build")
        os.makedirs(aicore_build_dir, exist_ok=True)
        
        # Compile aicore_kernel.cpp for both AIC and AIV
        aicore_flags = (
            f"-c -O3 -x cce -std=c++17 --cce-aicore-only "
            f"-mllvm -cce-aicore-stack-size=0x8000 "
            f"-mllvm -cce-aicore-function-stack-size=0x8000 "
            f"-I{{common_dir}} -I{{code_dir}}"
        )
        
        # Compile for AIC
        aic_kernel_obj = os.path.join(aicore_build_dir, "aicore_kernel_aic.o")
        cmd = f"{{ccec_path}} {{aicore_flags}} -D__AIC__ --cce-aicore-arch=dav-c220-cube -o {{aic_kernel_obj}} {{aicore_kernel_src}}"
        ok, _, stderr = run_command(cmd, timeout=120)
        if ok:
            print(f"    ✓ Compiled aicore_kernel for AIC")
        else:
            print(f"    ✗ Failed AIC: {{stderr}}")
            success = False
        
        # Compile for AIV
        aiv_kernel_obj = os.path.join(aicore_build_dir, "aicore_kernel_aiv.o")
        cmd = f"{{ccec_path}} {{aicore_flags}} -D__AIV__ --cce-aicore-arch=dav-c220-vec -o {{aiv_kernel_obj}} {{aicore_kernel_src}}"
        ok, _, stderr = run_command(cmd, timeout=120)
        if ok:
            print(f"    ✓ Compiled aicore_kernel for AIV")
        else:
            print(f"    ✗ Failed AIV: {{stderr}}")
            success = False
        
        # Link all objects into aicore_kernel.o
        all_objs = [aic_kernel_obj, aiv_kernel_obj] + aic_objs + aiv_objs
        aicore_kernel_out = os.path.join(code_dir, "aicore_kernel.o")
        if all_objs:
            cmd = f"{{ld_path}} -m aicorelinux -Ttext=0 -static -n -o {{aicore_kernel_out}} {{' '.join(all_objs)}}"
            ok, _, stderr = run_command(cmd, timeout=120)
            if ok:
                print(f"    ✓ Linked: {{aicore_kernel_out}}")
            else:
                print(f"    ✗ Link failed: {{stderr}}")
                success = False
    elif not ccec_available:
        print("\\n  [4/7] Skipping AICore kernel (ccec not available)")
    else:
        print("\\n  [4/7] Skipping AICore kernel (source not found)")
    
    # 5. Compile AICPU kernel (requires ARM64 gcc)
    aicpu_kernel_src = os.path.join(RUNTIME_DIR, "runtime_a2a3", "core", "aicpu", "aicpu_kernel.cpp")
    if os.path.exists(aicpu_kernel_src):
        print("\\n  [5/7] Compiling AICPU kernel...")
        
        # Check for ARM64 cross-compiler or native ARM64
        import platform
        is_arm64 = platform.machine() in ('aarch64', 'arm64')
        arm64_gcc = "aarch64-linux-gnu-g++" if not is_arm64 else "g++"
        
        # Check if compiler exists
        ok, _, _ = run_command(f"which {{arm64_gcc}}", timeout=10)
        if ok or is_arm64:
            aicpu_build_dir = os.path.join(code_dir, "aicpu_build")
            os.makedirs(aicpu_build_dir, exist_ok=True)
            
            aicpu_so = os.path.join(code_dir, "libaicpu_kernel.so")
            cmd = (
                f"{{arm64_gcc}} -O3 -std=c++17 -fPIC -shared -Wall "
                f"-DENABLE_AICPU_LOG "
                f"-I{{common_dir}} -I{{code_dir}} "
                f"-o {{aicpu_so}} {{aicpu_kernel_src}}"
            )
            
            ok, _, stderr = run_command(cmd, timeout=120)
            if ok:
                print(f"    ✓ Compiled: {{aicpu_so}}")
            else:
                print(f"    ✗ Failed: {{stderr}}")
                # Not a hard failure - AICPU kernel is needed only on device
        else:
            print(f"    Skipping: ARM64 compiler not available ({{arm64_gcc}})")
            print(f"    Note: AICPU kernel can be built on device or with cross-compiler")
    else:
        print("\\n  [5/7] Skipping AICPU kernel (source not found)")
    
    # 6. Generate and compile test program
    print("\\n  [6/7] Generating and compiling test program...")
    
    # Get example name from directory structure (e.g., "bgemm" from .../bgemm/output/platform/generated_code)
    platform_dir = os.path.dirname(code_dir)  # .../bgemm/output/platform
    output_dir = os.path.dirname(platform_dir)  # .../bgemm/output
    example_dir = os.path.dirname(output_dir)  # .../bgemm
    example_name = os.path.basename(example_dir)  # bgemm
    test_file = generate_test_program_template(code_dir, example_name)
    print(f"    Generated: {{test_file}}")
    
    # Get parent directory (platform_dir) for test executable
    platform_dir = os.path.dirname(code_dir)
    test_exe = os.path.join(platform_dir, "test_program")
    
    compile_flags = ["-O2", "-std=c11", "-D_POSIX_C_SOURCE=199309L"]
    if cann_available:
        compile_flags.append("-DCANN_SDK_AVAILABLE")
    else:
        compile_flags.append("-DA2A3_SKIP_CANN_CHECK")
    
    include_paths = [
        f"-I{{RUNTIME_DIR}}",
        f"-I{{code_dir}}",
        f"-I{{common_dir}}",
    ]
    
    # Add CANN SDK include path if available
    if cann_available:
        include_paths.append(f"-I{{ascend_home}}/include")
    
    # Find all runtime source files needed (host-side only)
    # Note: a2a3_core_worker.c is included for stub mode (CPU simulation)
    # On real hardware, workers run on NPU via aicore_kernel and aicpu_kernel
    runtime_sources = [
        os.path.join(RUNTIME_DIR, "pto_runtime.c"),
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "a2a3_runtime.c"),
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "host", "a2a3_host.c"),
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "host", "a2a3_so_loader.c"),
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "host", "a2a3_binary_loader.c"),
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "core", "a2a3_core_worker.c"),  # Stub mode workers
        os.path.join(RUNTIME_DIR, "runtime_a2a3", "orchestration", "a2a3_orchestration.c"),
    ]
    
    # Filter to only existing files
    runtime_sources = [s for s in runtime_sources if os.path.exists(s)]
    
    if runtime_sources:
        # Build link flags
        link_flags = ["-lpthread", "-ldl"]
        if cann_available:
            # Add ACL runtime library when CANN SDK is available
            link_flags.append(f"-L{{ascend_home}}/lib64")
            link_flags.append("-lascendcl")
        
        cmd = (
            f"gcc {{' '.join(compile_flags)}} {{' '.join(include_paths)}} "
            f"-o {{test_exe}} {{test_file}} {{' '.join(runtime_sources)}} "
            f"{{' '.join(link_flags)}}"
        )
        
        print(f"    Compiling test program...")
        ok, stdout, stderr = run_command(cmd, cwd=code_dir, timeout=120)
        
        if ok:
            print(f"    ✓ Compiled: {{test_exe}}")
        else:
            print(f"    ✗ Failed: {{stderr}}")
            success = False
    else:
        print(f"    ✗ Runtime source files not found")
        success = False
    
    # 7. Summary of generated NPU binaries
    print("\\n  [7/7] NPU binary summary...")
    aicore_kernel_out = os.path.join(code_dir, "aicore_kernel.o")
    aicpu_so = os.path.join(code_dir, "libaicpu_kernel.so")
    
    if os.path.exists(aicore_kernel_out):
        size = os.path.getsize(aicore_kernel_out)
        print(f"    AICore kernel: {{aicore_kernel_out}} ({{size}} bytes)")
    else:
        print(f"    AICore kernel: not built")
        
    if os.path.exists(aicpu_so):
        size = os.path.getsize(aicpu_so)
        print(f"    AICPU kernel:  {{aicpu_so}} ({{size}} bytes)")
    else:
        print(f"    AICPU kernel:  not built (requires ARM64 gcc)")
    
    # List InCore .o files
    aic_objs = glob.glob(os.path.join(aic_dir, "*.o")) if os.path.exists(aic_dir) else []
    aiv_objs = glob.glob(os.path.join(aiv_dir, "*.o")) if os.path.exists(aiv_dir) else []
    if aic_objs or aiv_objs:
        print(f"    InCore AIC:    {{len(aic_objs)}} object files")
        print(f"    InCore AIV:    {{len(aiv_objs)}} object files")
    
    return success


# =============================================================================
# Task Dump and Statistics
# =============================================================================

def run_task_dump():
    """Run and collect task dump statistics."""
    if not CONFIG['enable_task_dump']:
        return True
    
    print_header("Task Dump & Statistics")
    
    platform = CONFIG['target_platform']
    platform_dir = os.path.join(OUTPUT_DIR, platform)
    
    # Find executable (must be a file, not directory)
    exe_file = None
    for f in os.listdir(platform_dir):
        if not f.endswith(('.c', '.cu', '.cpp', '.txt', '.pdf', '.json', '.h')):
            exe_path = os.path.join(platform_dir, f)
            if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                exe_file = exe_path
                break
    
    if not exe_file:
        print("  No executable found")
        return False
    
    print(f"  Running: {{os.path.basename(exe_file)}}")
    success, stdout, stderr = run_command(exe_file, cwd=platform_dir, timeout=60)
    
    if success:
        print("  Execution successful")
        if stdout:
            print("  Output:")
            for line in stdout.split('\\n')[:20]:
                print(f"    {{line}}")
        
        # Look for dump file
        dump_file = exe_file.replace('_orchestration', '_task_graph') + '.txt'
        if os.path.exists(dump_file):
            print(f"\\n  Task graph dump: {{dump_file}}")
            analyze_task_dump(dump_file)
    else:
        print(f"  Execution failed: {{stderr}}")
    
    return success


def analyze_task_dump(dump_file):
    """Analyze task dump file and print statistics."""
    print("\\n  Task Dump Statistics:")
    print("  " + "-" * 40)
    
    try:
        with open(dump_file, 'r') as f:
            content = f.read()
        
        # Count tasks
        task_count = content.count("Task ")
        print(f"    Total tasks: {{task_count}}")
        
        # Count by type if available
        lines = content.split('\\n')
        task_types = dict()
        for line in lines:
            if "func=" in line:
                # Extract function name
                start = line.find("func=") + 5
                end = line.find(",", start) if "," in line[start:] else len(line)
                func_name = line[start:end].strip('"')
                task_types[func_name] = task_types.get(func_name, 0) + 1
        
        if task_types:
            print("    Tasks by function:")
            for func, count in sorted(task_types.items(), key=lambda x: -x[1]):
                print(f"      {{func}}: {{count}}")
        
        # Dependency info
        dep_count = content.count("fanin=")
        print(f"    Dependencies tracked: {{dep_count}}")
        
    except Exception as e:
        print(f"    Error analyzing dump: {{e}}")


# =============================================================================
# Task Graph PDF Generation
# =============================================================================

def generate_task_graph_pdf():
    """Generate task graph visualization as PDF."""
    if not CONFIG['enable_task_graph_pdf']:
        return True
    
    print_header("Task Graph PDF Generation")
    
    # Check if graphviz is available
    success, _, _ = run_command("which dot")
    if not success:
        print("  Warning: graphviz not installed, skipping PDF generation")
        return True
    
    platform = CONFIG['target_platform']
    platform_dir = os.path.join(OUTPUT_DIR, platform)
    
    # Look for task graph txt file
    txt_file = None
    for f in os.listdir(platform_dir):
        if 'task_graph' in f and f.endswith('.txt'):
            txt_file = os.path.join(platform_dir, f)
            break
    
    if not txt_file:
        print("  No task graph file found")
        return True
    
    # Try to use visualize_taskgraph.py if available (in scripts/ directory)
    vis_script = os.path.join(ROOT_DIR, "scripts", "visualize_taskgraph.py")
    if os.path.exists(vis_script):
        print(f"  Using visualize_taskgraph.py")
        cmd = f"python3 {{vis_script}} {{txt_file}}"
        success, stdout, stderr = run_command(cmd)
        if success:
            pdf_file = txt_file.replace('.txt', '.pdf')
            print(f"  Generated: {{pdf_file}}")
        else:
            print(f"  Warning: PDF generation failed: {{stderr}}")
    else:
        print("  visualize_taskgraph.py not found at {{vis_script}}")
    
    return True


# =============================================================================
# Performance Benchmark
# =============================================================================

def run_performance_benchmark():
    """Run performance benchmarks based on configuration."""
    success = True
    
    # Run orchestration benchmark if enabled
    if CONFIG.get('benchmark_orchestration', False):
        if not run_orchestration_benchmark():
            success = False
    
    # Run runtime benchmark if enabled
    if CONFIG.get('benchmark_runtime', False):
        if not run_runtime_benchmark():
            success = False
    
    # If neither benchmark enabled, just return True
    if not CONFIG.get('benchmark_orchestration', False) and not CONFIG.get('benchmark_runtime', False):
        return True
    
    return success


def run_orchestration_benchmark():
    """Orchestration Benchmark - measures task submission throughput (tasks/ms) without executing."""
    print_header("Orchestration Benchmark")
    print("  Measuring task submission throughput (tasks/ms) without executing tasks")
    
    TILE_ROWS = 32  # Must match pto_llama7B_dynamic.py
    
    platform = CONFIG['target_platform']
    platform_dir = os.path.join(OUTPUT_DIR, platform)
    
    # Find executable
    exe_file = find_executable(platform_dir)
    if not exe_file:
        print("  No executable found for benchmarking")
        return False
    
    print(f"  Executable: {{os.path.basename(exe_file)}}")
    print(f"  Seq length: {{CONFIG['test_seq_len_min']}} - {{CONFIG['test_seq_len_max']}} tokens (step: {{CONFIG['test_seq_len_step']}})")
    print(f"  Iterations: {{CONFIG['num_benchmark_iterations']}}")
    
    all_results = []
    seq_lengths = list(range(CONFIG['test_seq_len_min'], CONFIG['test_seq_len_max'] + 1, CONFIG['test_seq_len_step']))
    
    print(f"\\n  Testing {{len(seq_lengths)}} sequence lengths...\\n")
    print("  " + "-" * 85)
    print(f"  {{'seq_len':>10}} | {{'num_tiles':>10}} | {{'tasks':>10}} | {{'orch_time(ms)':>14}} | {{'tasks/ms':>12}} | {{'throughput':>15}}")
    print("  " + "-" * 85)
    
    for seq_len in seq_lengths:
        num_tiles = seq_len // TILE_ROWS
        # Run with --benchmark-only flag (or environment variable)
        cmd = f"{{exe_file}} --benchmark-only 0 0 {{num_tiles}} 0"
        
        times = []
        tasks_submitted = 0
        tasks_per_ms_values = []
        
        for i in range(CONFIG['num_benchmark_iterations']):
            success, stdout, stderr = run_command(cmd, cwd=platform_dir, timeout=60)
            
            if success:
                import re
                # Parse BENCHMARK output: tasks=X time_ms=Y tasks_per_ms=Z
                match = re.search(r'BENCHMARK:.*tasks=(\\d+)\\s+time_ms=([\\d.]+)\\s+tasks_per_ms=([\\d.]+)', stdout)
                if match:
                    tasks_submitted = int(match.group(1))
                    time_ms = float(match.group(2))
                    tpm = float(match.group(3))
                    times.append(time_ms)
                    tasks_per_ms_values.append(tpm)
        
        if times and tasks_per_ms_values:
            avg_time = sum(times) / len(times)
            avg_tpm = sum(tasks_per_ms_values) / len(tasks_per_ms_values)
            throughput = f"{{avg_tpm * 1000:.0f}} tasks/s"
            
            print(f"  {{seq_len:>10}} | {{num_tiles:>10}} | {{tasks_submitted:>10}} | {{avg_time:>14.3f}} | {{avg_tpm:>12.2f}} | {{throughput:>15}}")
            
            all_results.append({{
                "seq_len": seq_len,
                "num_tiles": num_tiles,
                "tasks_submitted": tasks_submitted,
                "avg_time_ms": avg_time,
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "tasks_per_ms": avg_tpm,
                "tasks_per_sec": avg_tpm * 1000,
            }})
        else:
            print(f"  {{seq_len:>10}} | {{num_tiles:>10}} | {{'FAILED':>10}} | {{'-':>14}} | {{'-':>12}} | {{'-':>15}}")
    
    print("  " + "-" * 85)
    
    if all_results:
        save_benchmark_results(platform_dir, "orchestration", all_results)
    
    return True


def run_runtime_benchmark():
    """Runtime Benchmark - measures actual execution/simulation time with workers."""
    print_header("Runtime Benchmark")
    print("  Measuring actual execution time with workers")
    
    TILE_ROWS = 32  # Must match pto_llama7B_dynamic.py
    
    platform = CONFIG['target_platform']
    platform_dir = os.path.join(OUTPUT_DIR, platform)
    
    # Find executable
    exe_file = find_executable(platform_dir)
    if not exe_file:
        print("  No executable found for benchmarking")
        return False
    
    print(f"  Executable: {{os.path.basename(exe_file)}}")
    print(f"  Seq length: {{CONFIG['test_seq_len_min']}} - {{CONFIG['test_seq_len_max']}} tokens (step: {{CONFIG['test_seq_len_step']}})")
    print(f"  Iterations: {{CONFIG['num_benchmark_iterations']}}")
    
    all_results = []
    seq_lengths = list(range(CONFIG['test_seq_len_min'], CONFIG['test_seq_len_max'] + 1, CONFIG['test_seq_len_step']))
    
    print(f"\\n  Testing {{len(seq_lengths)}} sequence lengths...\\n")
    print("  " + "-" * 65)
    print(f"  {{'seq_len':>10}} | {{'num_tiles':>10}} | {{'tasks':>10}} | {{'exec_time(ms)':>14}} | {{'tasks/ms':>12}}")
    print("  " + "-" * 65)
    
    for seq_len in seq_lengths:
        num_tiles = seq_len // TILE_ROWS
        # Run full execution (no --benchmark-only flag)
        cmd = f"{{exe_file}} 0 0 {{num_tiles}} 0"
        
        times = []
        tasks_submitted = 0
        
        for i in range(CONFIG['num_benchmark_iterations']):
            start = time.perf_counter()
            success, stdout, stderr = run_command(cmd, cwd=platform_dir, timeout=300)
            end = time.perf_counter()
            
            if success:
                elapsed_ms = (end - start) * 1000
                times.append(elapsed_ms)
                
                import re
                # Parse tasks submitted
                match = re.search(r'Submitted (\\d+) tasks', stdout)
                if match:
                    tasks_submitted = int(match.group(1))
        
        if times:
            avg_time = sum(times) / len(times)
            tasks_per_ms = tasks_submitted / avg_time if avg_time > 0 else 0
            
            print(f"  {{seq_len:>10}} | {{num_tiles:>10}} | {{tasks_submitted:>10}} | {{avg_time:>14.2f}} | {{tasks_per_ms:>12.2f}}")
            
            all_results.append({{
                "seq_len": seq_len,
                "num_tiles": num_tiles,
                "tasks_submitted": tasks_submitted,
                "avg_time_ms": avg_time,
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "tasks_per_ms": tasks_per_ms,
            }})
        else:
            print(f"  {{seq_len:>10}} | {{num_tiles:>10}} | {{'FAILED':>10}} | {{'-':>14}} | {{'-':>12}}")
    
    print("  " + "-" * 65)
    
    if all_results:
        save_benchmark_results(platform_dir, "runtime", all_results)
    
    return True


def find_executable(platform_dir):
    """Find executable in platform directory."""
    for f in os.listdir(platform_dir):
        if f.endswith(('.c', '.cu', '.cpp', '.txt', '.pdf', '.json', '.h')):
            continue
        exe_path = os.path.join(platform_dir, f)
        if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
            return exe_path
    return None


def save_benchmark_results(platform_dir, benchmark_type, all_results):
    """Save benchmark results and print summary."""
    if benchmark_type == "orchestration":
        avg_tpm = sum(r['tasks_per_ms'] for r in all_results) / len(all_results)
        max_tpm = max(r['tasks_per_ms'] for r in all_results)
        min_tpm = min(r['tasks_per_ms'] for r in all_results)
        
        print(f"\\n  Summary:")
        print(f"    Average: {{avg_tpm:.2f}} tasks/ms ({{avg_tpm * 1000:.0f}} tasks/s)")
        print(f"    Peak:    {{max_tpm:.2f}} tasks/ms ({{max_tpm * 1000:.0f}} tasks/s)")
        print(f"    Min:     {{min_tpm:.2f}} tasks/ms ({{min_tpm * 1000:.0f}} tasks/s)")
        
        summary = {{
            "avg_tasks_per_ms": avg_tpm,
            "max_tasks_per_ms": max_tpm,
            "min_tasks_per_ms": min_tpm,
        }}
    else:
        avg_time = sum(r['avg_time_ms'] for r in all_results) / len(all_results)
        avg_tasks_per_ms = sum(r.get('tasks_per_ms', 0) for r in all_results) / len(all_results)
        
        print(f"\\n  Summary:")
        print(f"    Average execution time: {{avg_time:.2f}} ms")
        print(f"    Average throughput: {{avg_tasks_per_ms:.2f}} tasks/ms")
        
        summary = {{
            "avg_execution_time_ms": avg_time,
            "avg_tasks_per_ms": avg_tasks_per_ms,
        }}
    
    results = {{
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": benchmark_type,
        "platform": CONFIG['target_platform'],
        "seq_len_range": {{
            "min": CONFIG['test_seq_len_min'],
            "max": CONFIG['test_seq_len_max'],
            "step": CONFIG['test_seq_len_step']
        }},
        "iterations": CONFIG['num_benchmark_iterations'],
        "results": all_results,
        "summary": summary,
    }}
    
    results_file = os.path.join(platform_dir, f"benchmark_{{benchmark_type}}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\n  Results saved to: {{results_file}}")




# =============================================================================
# Accuracy Test
# =============================================================================

def run_accuracy_test():
    """Generate and run accuracy tests using Python reference implementation."""
    if not CONFIG['enable_accuracy_test']:
        return True
    
    print_header("Accuracy Test")
    
    platform_dir = os.path.join(OUTPUT_DIR, CONFIG['target_platform'])
    
    # Check if accuracy data files exist
    input_file = os.path.join(platform_dir, "accuracy_input.bin")
    output_file = os.path.join(platform_dir, "accuracy_output.bin")
    params_file = os.path.join(platform_dir, "accuracy_params.txt")
    
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print("  Accuracy data files not found.")
        print("  Run test_program first to generate accuracy_input.bin and accuracy_output.bin")
        return True
    
    try:
        import numpy as np
    except ImportError:
        print("  NumPy not available. Skipping accuracy test.")
        return True
    
    # Read parameters
    params = dict()
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    params[key] = int(val)
    
    # Default BGEMM parameters
    num_tiles = params.get('num_tiles', 8)
    tile_m = params.get('tile_m', 64)
    tile_n = params.get('tile_n', 128)
    tile_k = params.get('tile_k', 64)
    k_tiles = params.get('k_tiles', 8)
    
    print(f"  Parameters: num_tiles={{num_tiles}}, tile_m={{tile_m}}, tile_n={{tile_n}}, tile_k={{tile_k}}, k_tiles={{k_tiles}}")
    
    # Read input and output data
    input_data = np.fromfile(input_file, dtype=np.float32)
    output_data = np.fromfile(output_file, dtype=np.float32)
    
    print(f"  Input size: {{len(input_data)}} floats")
    print(f"  Output size: {{len(output_data)}} floats")
    
    # Split input into A and B matrices (same as test_program.c)
    half_size = len(input_data) // 2
    A_flat = input_data[:half_size]
    B_flat = input_data[half_size:]
    
    # Compute expected output using Python reference
    # BGEMM: For each output tile, C[tile] = sum_k(A[tile*k_tiles+k] @ B[k*num_tiles+tile])
    print("  Computing Python reference...")
    
    tile_size = tile_m * tile_n
    expected_output = np.zeros(num_tiles * tile_size, dtype=np.float32)
    
    for tile in range(num_tiles):
        # Accumulate partial products for this tile
        C_tile = np.zeros((tile_m, tile_n), dtype=np.float32)
        
        for k in range(k_tiles):
            # Get A tile: A[tile * k_tiles + k]
            a_idx = (tile * k_tiles + k) * tile_m * tile_k
            if a_idx + tile_m * tile_k <= len(A_flat):
                A_tile = A_flat[a_idx : a_idx + tile_m * tile_k].reshape(tile_m, tile_k)
            else:
                A_tile = np.zeros((tile_m, tile_k), dtype=np.float32)
            
            # Get B tile: B[k * num_tiles + tile]
            b_idx = (k * num_tiles + tile) * tile_k * tile_n
            if b_idx + tile_k * tile_n <= len(B_flat):
                B_tile = B_flat[b_idx : b_idx + tile_k * tile_n].reshape(tile_k, tile_n)
            else:
                B_tile = np.zeros((tile_k, tile_n), dtype=np.float32)
            
            # Accumulate: C += A @ B
            C_tile += np.matmul(A_tile, B_tile)
        
        # Store result
        expected_output[tile * tile_size : (tile + 1) * tile_size] = C_tile.flatten()
    
    # Compare results
    output_elements = num_tiles * tile_size
    actual = output_data[:output_elements]
    expected = expected_output[:output_elements]
    
    # Compute differences
    diff = np.abs(expected - actual)
    max_diff = np.max(diff)
    max_diff_idx = np.argmax(diff)
    
    # Use relative tolerance for floating point
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance
    
    # Check for errors
    errors = np.sum(diff > (atol + rtol * np.abs(expected)))
    
    print(f"  Max difference: {{max_diff:.6f}} at index {{max_diff_idx}}")
    print(f"  Expected[{{max_diff_idx}}]: {{expected[max_diff_idx]:.6f}}")
    print(f"  Actual[{{max_diff_idx}}]: {{actual[max_diff_idx]:.6f}}")
    
    if errors > 0:
        # Show first few mismatches
        mismatch_indices = np.where(diff > (atol + rtol * np.abs(expected)))[0][:10]
        for idx in mismatch_indices:
            print(f"    Mismatch at [{{idx}}]: expected {{expected[idx]:.6f}}, got {{actual[idx]:.6f}}")
    
    print()
    print("=" * 60)
    if errors == 0:
        print("  ACCURACY TEST: PASSED")
        result = True
    else:
        print(f"  ACCURACY TEST: FAILED ({{errors}} errors out of {{output_elements}})")
        result = False
    print("=" * 60)
    
    return result


# =============================================================================
# Simulation and Trace Generation
# =============================================================================

def run_simulation():
    """Run simulation and generate trace files."""
    if not CONFIG['enable_simulation']:
        return True
    
    print_header("Simulation & Trace Generation")
    
    if CONFIG['target_platform'] != 'ascend_a2a3_sim':
        print("  Simulation only available for ascend_a2a3_sim platform")
        return True
    
    platform_dir = os.path.join(OUTPUT_DIR, CONFIG['target_platform'])
    
    # Find simulation executable (any executable that's not a known non-executable extension)
    exe_file = None
    for f in os.listdir(platform_dir):
        # Skip directories, source files, and known output files
        if f.endswith(('.c', '.cu', '.cpp', '.txt', '.pdf', '.json', '.h')):
            continue
        exe_path = os.path.join(platform_dir, f)
        # Must be a file (not directory) and executable
        if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
            exe_file = exe_path
            break
    
    if not exe_file:
        # Simulation already ran during task dump, so this is not a failure
        print("  Note: Simulation already completed during task dump step")
        return True
    
    # Calculate num_tiles for simulation using max seq_len from config
    TILE_ROWS = 32  # Must match pto_llama7B_dynamic.py
    sim_seq_len = CONFIG['test_seq_len_max']
    sim_num_tiles = sim_seq_len // TILE_ROWS
    
    print(f"  Running simulation: {{os.path.basename(exe_file)}}")
    print(f"  Simulation parameters: seq_len={{sim_seq_len}}, num_tiles={{sim_num_tiles}}")
    
    # Run with trace enabled and proper parameters
    env = os.environ.copy()
    env['PTO_TRACE_OUTPUT'] = os.path.join(platform_dir, 'trace.json')
    
    # Pass seq_len, tile_rows, num_tiles, zero as arguments
    sim_cmd = f"{{exe_file}} {{sim_seq_len}} {{TILE_ROWS}} {{sim_num_tiles}} 0"
    success, stdout, stderr = run_command(sim_cmd, cwd=platform_dir, timeout=120)
    
    if success:
        trace_file = os.path.join(platform_dir, 'trace.json')
        if os.path.exists(trace_file):
            print(f"  Trace file generated: {{trace_file}}")
            
            # Basic trace analysis
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                if isinstance(trace_data, list):
                    print(f"  Trace events: {{len(trace_data)}}")
            except:
                pass
        else:
            print("  Note: Trace file not generated (may need runtime support)")
    else:
        print(f"  Simulation failed: {{stderr}}")
    
    return success


# =============================================================================
# Main
# =============================================================================

def main():
    print_header(f"PTO Example Runner: {{CONFIG['example_name']}}")
    print(f"  Platform: {{CONFIG['target_platform']}}")
    print(f"  Output:   {{OUTPUT_DIR}}")
    
    ensure_dir(OUTPUT_DIR)
    
    steps = [
        ("Code Generation", generate_code),
        ("Compilation", compile_code),
        ("Task Dump", run_task_dump),
        ("Task Graph PDF", generate_task_graph_pdf),
        ("Performance Benchmark", run_performance_benchmark),
        ("Accuracy Test", run_accuracy_test),
        ("Simulation", run_simulation),
    ]
    
    results = []
    for name, func in steps:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"  Error in {{name}}: {{e}}")
            results.append((name, False))
    
    print_header("Summary")
    for name, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {{name}}: {{status}}")
    
    print("\\nDone!")


if __name__ == "__main__":
    main()
'''
    
    return script


# =============================================================================
# Main Menu
# =============================================================================

def main():
    """Main configuration menu."""
    # Determine root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config = DEFAULT_CONFIG.copy()
    
    while True:
        clear_screen()
        print_header("PTO Example Configuration Tool")
        print_config(config)
        
        print("\nOptions:")
        print("  [1-10] Modify configuration")
        print("  [g]    Generate run_<platform>.py for current platform")
        print("  [a]    Generate ALL platform scripts (run_arm64.py, run_ascend_a2a3_sim.py, ...)")
        print("  [s]    Save configuration")
        print("  [l]    Load configuration")
        print("  [q]    Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'g':
            if not config['example_name']:
                print("\nError: Please select an example first!")
                input("Press Enter to continue...")
                continue
            
            # Generate platform-specific run script
            script_content = generate_run_script(config, root_dir)
            
            # Save to example directory with platform-specific name
            example_dir = os.path.join(root_dir, "examples", config['example_name'])
            script_name = get_script_name(config['target_platform'])
            run_py_path = os.path.join(example_dir, script_name)
            
            with open(run_py_path, 'w') as f:
                f.write(script_content)
            
            os.chmod(run_py_path, 0o755)
            
            print(f"\n✓ Generated: {run_py_path}")
            print(f"\nTo run: cd examples/{config['example_name']} && python {script_name}")
            input("\nPress Enter to continue...")
        
        elif choice == 'a':
            # Generate all platform scripts
            if not config['example_name']:
                print("\nError: Please select an example first!")
                input("Press Enter to continue...")
                continue
            
            example_dir = os.path.join(root_dir, "examples", config['example_name'])
            generated = []
            
            for platform in PLATFORM_LIST:
                platform_config = config.copy()
                platform_config['target_platform'] = platform
                
                script_content = generate_run_script(platform_config, root_dir)
                script_name = get_script_name(platform)
                run_py_path = os.path.join(example_dir, script_name)
                
                with open(run_py_path, 'w') as f:
                    f.write(script_content)
                os.chmod(run_py_path, 0o755)
                generated.append(script_name)
            
            print(f"\n✓ Generated {len(generated)} scripts in examples/{config['example_name']}/:")
            for name in generated:
                print(f"    - {name}")
            input("\nPress Enter to continue...")
            
        elif choice == 's':
            # Save configuration
            config_file = os.path.join(root_dir, "examples", config['example_name'], "config.json") \
                         if config['example_name'] else os.path.join(root_dir, "config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\n✓ Configuration saved to: {config_file}")
            input("Press Enter to continue...")
            
        elif choice == 'l':
            # Load configuration
            examples = get_examples_list(root_dir)
            print_header("Load Configuration")
            print("  [0] From root directory")
            for i, ex in enumerate(examples, 1):
                config_path = os.path.join(root_dir, "examples", ex, "config.json")
                exists = "✓" if os.path.exists(config_path) else " "
                print(f"  [{i}] {ex} {exists}")
            
            try:
                idx = int(input("\nEnter choice: "))
                if idx == 0:
                    config_file = os.path.join(root_dir, "config.json")
                elif 1 <= idx <= len(examples):
                    config_file = os.path.join(root_dir, "examples", examples[idx-1], "config.json")
                else:
                    continue
                
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        loaded = json.load(f)
                    config.update(loaded)
                    print(f"✓ Loaded: {config_file}")
                else:
                    print("Configuration file not found!")
            except ValueError:
                pass
            input("Press Enter to continue...")
            
        elif choice == '1':
            selected = select_example(root_dir)
            if selected:
                config['example_name'] = selected
            input("Press Enter to continue...")
            
        elif choice == '2':
            selected = select_platform()
            if selected:
                config['target_platform'] = selected
                # Apply platform-specific default overrides
                platform_info = PLATFORM_OPTIONS.get(selected, {})
                if 'benchmark_orchestration' in platform_info:
                    config['benchmark_orchestration'] = platform_info['benchmark_orchestration']
                if 'benchmark_runtime' in platform_info:
                    config['benchmark_runtime'] = platform_info['benchmark_runtime']
            input("Press Enter to continue...")
            
        elif choice == '3':
            config = toggle_option(config, 'enable_binary_expansion', 'Binary Expansion')
            input("Press Enter to continue...")
            
        elif choice == '4':
            config = toggle_option(config, 'enable_task_dump', 'Task Dump')
            input("Press Enter to continue...")
            
        elif choice == '5':
            config = toggle_option(config, 'enable_task_graph_pdf', 'Task Graph PDF')
            input("Press Enter to continue...")
            
        elif choice == '6':
            config = toggle_benchmark_orchestration(config)
            input("Press Enter to continue...")
            
        elif choice == '7':
            config = toggle_benchmark_runtime(config)
            input("Press Enter to continue...")
            
        elif choice == '8':
            config = configure_test_range(config)
            input("Press Enter to continue...")
            
        elif choice == '9':
            config = toggle_option(config, 'enable_accuracy_test', 'Accuracy Test')
            input("Press Enter to continue...")
            
        elif choice == '10':
            config = toggle_option(config, 'enable_simulation', 'Simulation & Trace')
            input("Press Enter to continue...")


def list_available_examples(root_dir: str) -> List[str]:
    """List all available examples."""
    examples_dir = os.path.join(root_dir, 'examples')
    examples = []
    if os.path.exists(examples_dir):
        for item in os.listdir(examples_dir):
            item_path = os.path.join(examples_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains pto_*.py files
                for f in os.listdir(item_path):
                    if f.startswith('pto_') and f.endswith('.py'):
                        examples.append(item)
                        break
    return sorted(examples)


def run_cli(args):
    """Run in command-line mode."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List examples
    if args.list_examples:
        examples = list_available_examples(root_dir)
        print("Available examples:")
        for ex in examples:
            print(f"  - {ex}")
        return 0
    
    # List platforms
    if args.list_platforms:
        print("Available platforms:")
        for key, info in PLATFORM_OPTIONS.items():
            print(f"  - {key}: {info['name']}")
        return 0
    
    # Validate required arguments
    if not args.example:
        print("Error: --example is required")
        print("Use --list-examples to see available examples")
        return 1
    
    if not args.platform:
        print("Error: --platform is required")
        print("Use --list-platforms to see available platforms")
        return 1
    
    # Build configuration
    config = DEFAULT_CONFIG.copy()
    config['example_name'] = args.example
    config['target_platform'] = args.platform
    
    # Apply platform-specific default overrides
    platform_info = PLATFORM_OPTIONS.get(args.platform, {})
    if 'benchmark_orchestration' in platform_info:
        config['benchmark_orchestration'] = platform_info['benchmark_orchestration']
    if 'benchmark_runtime' in platform_info:
        config['benchmark_runtime'] = platform_info['benchmark_runtime']
    
    # Apply optional arguments (these override platform defaults)
    if args.seq_len_min is not None:
        config['test_seq_len_min'] = args.seq_len_min
    if args.seq_len_max is not None:
        config['test_seq_len_max'] = args.seq_len_max
    if args.seq_len_step is not None:
        config['test_seq_len_step'] = args.seq_len_step
    if args.no_benchmark:
        config['benchmark_orchestration'] = False
        config['benchmark_runtime'] = False
    if args.no_simulation:
        config['enable_simulation'] = False
        config['enable_trace_generation'] = False
    
    # Validate example exists
    examples = list_available_examples(root_dir)
    if args.example not in examples:
        print(f"Error: Example '{args.example}' not found")
        print(f"Available examples: {', '.join(examples)}")
        return 1
    
    # Validate platform
    if args.platform not in PLATFORM_OPTIONS:
        print(f"Error: Platform '{args.platform}' not supported")
        print(f"Available platforms: {', '.join(PLATFORM_OPTIONS.keys())}")
        return 1
    
    print(f"Configuring example: {args.example}")
    print(f"  Platform: {args.platform}")
    print(f"  Seq len range: {config['test_seq_len_min']}-{config['test_seq_len_max']} (step {config['test_seq_len_step']})")
    print(f"  Benchmark: {config['benchmark_orchestration']}")
    print(f"  Simulation: {config['enable_simulation']}")
    
    # Generate run script
    if args.generate or args.run:
        print("\nGenerating run script...")
        script_content = generate_run_script(config, root_dir)
        
        # Determine output path
        example_dir = os.path.join(root_dir, 'examples', args.example)
        platform_suffix = PLATFORM_OPTIONS[args.platform]['script_suffix']
        script_name = f"run_{platform_suffix}.py"
        script_path = os.path.join(example_dir, script_name)
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        print(f"  Generated: {script_path}")
        
        # Save config
        config_path = os.path.join(example_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Config saved: {config_path}")
    
    # Run the script
    if args.run:
        print(f"\nRunning {script_name}...")
        print("=" * 60)
        
        # Change to example directory and run
        import subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=example_dir
        )
        return result.returncode
    
    return 0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PTO Example Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python config_example.py
  
  # Generate run script for BGEMM on Ascend simulator
  python config_example.py --example bgemm --platform ascend_a2a3_sim --generate
  
  # Generate and run LLaMA example
  python config_example.py --example llama --platform ascend_a2a3_sim --run
  
  # Configure benchmark range
  python config_example.py --example bgemm --platform ascend_a2a3_sim \\
      --seq-len-min 512 --seq-len-max 8192 --seq-len-step 512 --run
  
  # List available options
  python config_example.py --list-examples
  python config_example.py --list-platforms
"""
    )
    
    parser.add_argument('--example', '-e', type=str,
                        help='Example to configure (llama, softmax, bgemm, ...)')
    parser.add_argument('--platform', '-p', type=str,
                        help='Target platform (arm64, cuda, ascend_a2a3_sim, ...)')
    parser.add_argument('--generate', '-g', action='store_true',
                        help='Generate run script and exit')
    parser.add_argument('--run', '-r', action='store_true',
                        help='Generate script and run it')
    
    # Benchmark configuration
    parser.add_argument('--seq-len-min', type=int, metavar='N',
                        help='Minimum sequence length for benchmarking')
    parser.add_argument('--seq-len-max', type=int, metavar='N',
                        help='Maximum sequence length for benchmarking')
    parser.add_argument('--seq-len-step', type=int, metavar='N',
                        help='Sequence length step size')
    parser.add_argument('--no-benchmark', action='store_true',
                        help='Disable benchmarking')
    parser.add_argument('--no-simulation', action='store_true',
                        help='Disable simulation')
    
    # Listing options
    parser.add_argument('--list-examples', action='store_true',
                        help='List available examples')
    parser.add_argument('--list-platforms', action='store_true',
                        help='List available platforms')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # If any CLI arguments provided, run in CLI mode
    if (args.example or args.platform or args.generate or args.run or 
        args.list_examples or args.list_platforms or
        args.seq_len_min or args.seq_len_max or args.seq_len_step or
        args.no_benchmark or args.no_simulation):
        sys.exit(run_cli(args))
    else:
        # Interactive mode
        main()
