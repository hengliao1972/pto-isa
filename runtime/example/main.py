#!/usr/bin/env python3
"""
Main Example - PTO Runtime with C++ Graph Builder

This program demonstrates how to use the refactored graph builder where
the graph initialization logic is in C++ (graphbuilder.cpp) and Python
orchestrates the runtime execution.

Flow:
1. C++ InitGraph(): Allocates graph, tensors, builds task structure, initializes data
2. Python runner.run(): Executes the graph on device
3. C++ ValidateGraph(): Validates results, frees tensors, deletes graph

Example usage:
   python main.py [device_id]
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path so we can import runtime_bindings
example_root = Path(__file__).parent
runtime_root = Path(__file__).parent.parent
runtime_dir = runtime_root / "python"
sys.path.insert(0, str(runtime_dir))

try:
    from runtime_bindings import load_runtime
    from binary_compiler import BinaryCompiler
except ImportError:
    print("Error: Cannot import runtime_bindings module")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def check_and_build_runtime():
    """
    Check if runtime libraries exist and build if necessary using BinaryCompiler.

    Returns:
        True if build successful or libraries exist, False otherwise
    """
    print("Building runtime using BinaryCompiler...")

    compiler = BinaryCompiler()

    # Compile AICore kernel
    print("\n[1/3] Compiling AICore kernel...")
    try:
        aicore_include_dirs = [
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        aicore_source_dirs = [
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        aicore_binary = compiler.compile("aicore", aicore_include_dirs, aicore_source_dirs)
    except Exception as e:
        print(f"✗ AICore compilation failed: {e}")
        return None

    # Compile AICPU kernel
    print("\n[2/3] Compiling AICPU kernel...")
    try:
        aicpu_include_dirs = [
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        aicpu_source_dirs = [
            str(runtime_root / "src" / "runtime" / "aicpu"),
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        aicpu_binary = compiler.compile("aicpu", aicpu_include_dirs, aicpu_source_dirs)
    except Exception as e:
        print(f"✗ AICPU compilation failed: {e}")
        return None

    # Compile Host runtime
    print("\n[3/3] Compiling Host runtime...")
    try:
        host_include_dirs = [
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        host_source_dirs = [
            str(runtime_root / "src" / "runtime" / "host"),
            str(runtime_root / "src" / "runtime" / "graph"),
        ]
        host_binary = compiler.compile("host", host_include_dirs, host_source_dirs)
    except Exception as e:
        print(f"✗ Host runtime compilation failed: {e}")
        return None

    print("\nBuild complete!")

    return (host_binary, aicpu_binary, aicore_binary)



def main():
    # Check and build runtime if necessary
    compile_results = check_and_build_runtime()
    if not compile_results:
        print("Error: Failed to build runtime libraries")
        return -1
    host_binary, aicpu_binary, aicore_binary = compile_results

    # Parse device ID from command line
    device_id = 9
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
            if device_id < 0 or device_id > 15:
                print(f"Error: deviceId ({device_id}) out of range [0, 15]")
                return -1
        except ValueError:
            print(f"Error: invalid deviceId argument: {sys.argv[1]}")
            return -1

    pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"

    # Load runtime library and get bindings
    print("\n=== Loading Runtime Library ===")
    DeviceRunner, Graph = load_runtime(host_binary)
    print(f"Loaded runtime ({len(host_binary)} bytes)")

    # Initialize DeviceRunner
    print("\n=== Initializing DeviceRunner ===")
    runner = DeviceRunner()
    runner.init(device_id, aicpu_binary, aicore_binary, pto_isa_root)

    # Create and initialize graph
    # C++ handles: allocate Graph, allocate tensors, build tasks, initialize data
    print("\n=== Creating and Initializing Graph ===")
    graph = Graph()
    graph.initialize()

    # Execute graph on device
    # Python now controls when the graph is executed
    print("\n=== Executing Graph on Device ===")
    runner.run(graph, num_cores=3, launch_aicpu_num=1)

    # Validate results and cleanup
    # C++ handles: copy results from device, validate, free tensors, delete graph
    print("\n=== Validating Results and Cleaning Up ===")
    graph.validate_and_cleanup()

    # Finalize runner
    print("\n=== Finalizing DeviceRunner ===")
    runner.finalize()

    return 0

if __name__ == '__main__':
    sys.exit(main())
