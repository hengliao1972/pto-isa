#!/usr/bin/env python3
"""
Graph Builder Example (Python)

This program demonstrates how to build a task dependency graph and execute it
on Ascend devices using the Python bindings for DeviceRunner and Graph.

This is a Python translation of graphbuilder.cpp.
"""

import sys
import numpy as np

# Import the PTO runtime Python module
# Make sure to build the module first and add its path to PYTHONPATH
try:
    import pto_runtime
except ImportError:
    print("Error: Cannot import pto_runtime module")
    print("Please build the Python bindings first:")
    print("  cd runtime/build && cmake .. -DBUILD_PYTHON_BINDINGS=ON && make")
    print("Then set PYTHONPATH:")
    print("  export PYTHONPATH=/path/to/runtime/build/python:$PYTHONPATH")
    sys.exit(1)


def main():
    print("=== Graph Builder Example (Python) ===\n")

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

    # Initialize DeviceRunner
    runner = pto_runtime.DeviceRunner.get()
    rc = runner.init(device_id, 3, "./aicpu/libaicpu_graph_kernel.so", "./aicore/kernel.o")
    if rc != 0:
        print("Error: DeviceRunner initialization failed")
        return rc

    # Allocate device tensors
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS  # 16384 elements
    BYTES = SIZE * 4  # sizeof(float)

    print("\n=== Allocating Device Memory ===")
    dev_a = runner.allocate_tensor(BYTES)
    dev_b = runner.allocate_tensor(BYTES)
    dev_c = runner.allocate_tensor(BYTES)
    dev_d = runner.allocate_tensor(BYTES)
    dev_e = runner.allocate_tensor(BYTES)
    dev_f = runner.allocate_tensor(BYTES)

    if not all([dev_a, dev_b, dev_c, dev_d, dev_e, dev_f]):
        print("Error: Failed to allocate device tensors")
        runner.finalize()
        return -1

    print(f"Allocated 6 tensors (128x128 each, {BYTES} bytes per tensor)")

    # Initialize input data and copy to device
    host_a = np.full((ROWS, COLS), 2.0, dtype=np.float32)
    host_b = np.full((ROWS, COLS), 3.0, dtype=np.float32)

    rc = runner.copy_to_device(dev_a, host_a)
    if rc != 0:
        print("Error: Failed to copy input a to device")
        cleanup_and_exit(runner, [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f])
        return rc

    rc = runner.copy_to_device(dev_b, host_b)
    if rc != 0:
        print("Error: Failed to copy input b to device")
        cleanup_and_exit(runner, [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f])
        return rc

    print("Initialized input tensors: a=2.0, b=3.0 (all elements)")
    print("Expected result: f = (2+3+1)*(2+3+2) = 6*7 = 42.0")

    # =========================================================================
    # BUILD GRAPH - This is the core graph building logic
    # =========================================================================
    print("\n=== Creating Task Graph for Formula ===")
    print("Formula: (a + b + 1)(a + b + 2)")
    print("Tasks:")
    print("  task0: c = a + b")
    print("  task1: d = c + 1")
    print("  task2: e = c + 2")
    print("  task3: f = d * e\n")

    test_graph = pto_runtime.Graph()

    # Task 0: c = a + b (func_id=0: kernel_add)
    t0 = test_graph.add_task([dev_a, dev_b, dev_c, SIZE], func_id=0)

    # Task 1: d = c + 1 (func_id=1: kernel_add_scalar)
    t1 = test_graph.add_task([dev_c, 1.0, dev_d, SIZE], func_id=1)

    # Task 2: e = c + 2 (func_id=1: kernel_add_scalar)
    t2 = test_graph.add_task([dev_c, 2.0, dev_e, SIZE], func_id=1)

    # Task 3: f = d * e (func_id=2: kernel_mul)
    t3 = test_graph.add_task([dev_d, dev_e, dev_f, SIZE], func_id=2)

    # Add dependencies
    test_graph.add_successor(t0, t1)  # t0 → t1
    test_graph.add_successor(t0, t2)  # t0 → t2
    test_graph.add_successor(t1, t3)  # t1 → t3
    test_graph.add_successor(t2, t3)  # t2 → t3

    print(f"Created graph with {test_graph.get_task_count()} tasks")
    test_graph.print_graph()

    # =========================================================================
    # RUN GRAPH - Execute the graph on device
    # =========================================================================
    print("\n=== Executing Graph ===")
    rc = runner.run(test_graph, 1)
    if rc != 0:
        print(f"Error: Graph execution failed: {rc}")
        cleanup_and_exit(runner, [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f])
        return rc

    # =========================================================================
    # VALIDATE RESULTS - Retrieve and verify output
    # =========================================================================
    print("\n=== Validating Results ===")
    host_result = np.zeros((ROWS, COLS), dtype=np.float32)
    rc = runner.copy_from_device(host_result, dev_f)
    if rc != 0:
        print(f"Error: Failed to copy result from device: {rc}")
        cleanup_and_exit(runner, [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f])
        return rc

    # Print sample values
    print("First 10 elements of result:")
    flat_result = host_result.flatten()
    for i in range(10):
        print(f"  f[{i}] = {flat_result[i]}")

    # Validate result
    EXPECTED = 42.0  # (2+3+1)*(2+3+2) = 6*7 = 42
    errors = np.abs(flat_result - EXPECTED) > 0.001
    error_count = np.sum(errors)

    if error_count == 0:
        print(f"\n✓ SUCCESS: All {SIZE} elements are correct (42.0)")
        print("Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42")
    else:
        print(f"\n✗ FAILED: {error_count} elements are incorrect")
        # Print first few errors
        error_indices = np.where(errors)[0]
        for idx in error_indices[:5]:
            print(f"ERROR: f[{idx}] = {flat_result[idx]}, expected {EXPECTED}")

    # Print handshake results
    runner.print_handshake_results()

    # Cleanup
    print("\n=== Cleaning Up ===")
    cleanup_and_exit(runner, [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f])
    print("Freed all device tensors")

    runner.finalize()

    if rc != 0 or error_count > 0:
        print("=== Execution Failed ===")
        return -1
    else:
        print("=== Success ===")

    return 0


def cleanup_and_exit(runner, tensors):
    """Helper to free all tensors"""
    for tensor in tensors:
        if tensor:
            runner.free_tensor(tensor)


if __name__ == '__main__':
    sys.exit(main())
