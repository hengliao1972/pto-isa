/**
 * Graph Builder Example
 * This program demonstrates how to build a task dependency graph and execute it
 * on Ascend devices using the DeviceRunner interface.
 */

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "graph/graph.h"
#include "host/devicerunner.h"

// Example usage
int main(int argc, char **argv) {
    std::cout << "=== Graph Builder Example ===" << '\n';

    // Parse device ID from command line
    int deviceId = 9;
    if (argc > 1) {
        try {
            deviceId = std::stoi(argv[1]);
            if (deviceId < 0 || deviceId > 15) {
                std::cerr << "Error: deviceId (" << deviceId << ") out of range [0, 15]" << '\n';
                return -1;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: invalid deviceId argument: " << argv[1] << '\n';
            return -1;
        }
    }

    // Initialize DeviceRunner
    DeviceRunner& runner = DeviceRunner::Get();
    int rc = runner.Init(deviceId, 3, "./aicpu/libaicpu_graph_kernel.so", "./aicore/kernel.o");
    if (rc != 0) {
        std::cerr << "Error: DeviceRunner initialization failed" << '\n';
        return rc;
    }

    // Allocate device tensors
    constexpr int ROWS = 128;
    constexpr int COLS = 128;
    constexpr int SIZE = ROWS * COLS;  // 16384 elements
    constexpr size_t BYTES = SIZE * sizeof(float);

    std::cout << "\n=== Allocating Device Memory ===" << '\n';
    void* dev_a = runner.AllocateTensor(BYTES);
    void* dev_b = runner.AllocateTensor(BYTES);
    void* dev_c = runner.AllocateTensor(BYTES);
    void* dev_d = runner.AllocateTensor(BYTES);
    void* dev_e = runner.AllocateTensor(BYTES);
    void* dev_f = runner.AllocateTensor(BYTES);

    if (!dev_a || !dev_b || !dev_c || !dev_d || !dev_e || !dev_f) {
        std::cerr << "Error: Failed to allocate device tensors" << '\n';
        runner.Finalize();
        return -1;
    }
    std::cout << "Allocated 6 tensors (128x128 each, " << BYTES << " bytes per tensor)\n";

    // Initialize input data and copy to device
    std::vector<float> host_a(SIZE, 2.0f);
    std::vector<float> host_b(SIZE, 3.0f);

    rc = runner.CopyToDevice(dev_a, host_a.data(), BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy input a to device" << '\n';
        runner.FreeTensor(dev_a); runner.FreeTensor(dev_b); runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d); runner.FreeTensor(dev_e); runner.FreeTensor(dev_f);
        runner.Finalize();
        return rc;
    }

    rc = runner.CopyToDevice(dev_b, host_b.data(), BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy input b to device" << '\n';
        runner.FreeTensor(dev_a); runner.FreeTensor(dev_b); runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d); runner.FreeTensor(dev_e); runner.FreeTensor(dev_f);
        runner.Finalize();
        return rc;
    }

    std::cout << "Initialized input tensors: a=2.0, b=3.0 (all elements)\n";
    std::cout << "Expected result: f = (2+3+1)*(2+3+2) = 6*7 = 42.0\n";

    // =========================================================================
    // BUILD GRAPH - This is the core graph building logic
    // =========================================================================
    std::cout << "\n=== Creating Task Graph for Formula ===" << '\n';
    std::cout << "Formula: (a + b + 1)(a + b + 2)\n";
    std::cout << "Tasks:\n";
    std::cout << "  task0: c = a + b\n";
    std::cout << "  task1: d = c + 1\n";
    std::cout << "  task2: e = c + 2\n";
    std::cout << "  task3: f = d * e\n\n";

    Graph testGraph;

    // Helper union to encode float scalar as uint64_t
    union {
        float f32;
        uint64_t u64;
    } scalar_converter;

    // Task 0: c = a + b (func_id=0: kernel_add)
    uint64_t args_t0[4];
    args_t0[0] = reinterpret_cast<uint64_t>(dev_a);  // src0
    args_t0[1] = reinterpret_cast<uint64_t>(dev_b);  // src1
    args_t0[2] = reinterpret_cast<uint64_t>(dev_c);  // out
    args_t0[3] = SIZE;                                // size
    int t0 = testGraph.add_task(args_t0, 4, 0);

    // Task 1: d = c + 1 (func_id=1: kernel_add_scalar)
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 1.0f;
    args_t1[1] = scalar_converter.u64;                // scalar=1.0
    args_t1[2] = reinterpret_cast<uint64_t>(dev_d);  // out
    args_t1[3] = SIZE;                                // size
    int t1 = testGraph.add_task(args_t1, 4, 1);

    // Task 2: e = c + 2 (func_id=1: kernel_add_scalar)
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 2.0f;
    args_t2[1] = scalar_converter.u64;                // scalar=2.0
    args_t2[2] = reinterpret_cast<uint64_t>(dev_e);  // out
    args_t2[3] = SIZE;                                // size
    int t2 = testGraph.add_task(args_t2, 4, 1);

    // Task 3: f = d * e (func_id=2: kernel_mul)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_d);  // src0
    args_t3[1] = reinterpret_cast<uint64_t>(dev_e);  // src1
    args_t3[2] = reinterpret_cast<uint64_t>(dev_f);  // out
    args_t3[3] = SIZE;                                // size
    int t3 = testGraph.add_task(args_t3, 4, 2);

    // Add dependencies
    testGraph.add_successor(t0, t1);  // t0 → t1
    testGraph.add_successor(t0, t2);  // t0 → t2
    testGraph.add_successor(t1, t3);  // t1 → t3
    testGraph.add_successor(t2, t3);  // t2 → t3

    std::cout << "Created graph with " << testGraph.get_task_count() << " tasks\n";
    testGraph.print_graph();

    // =========================================================================
    // RUN GRAPH - Execute the graph on device
    // =========================================================================
    std::cout << "\n=== Executing Graph ===" << '\n';
    rc = runner.Run(testGraph, 1);
    if (rc != 0) {
        std::cerr << "Error: Graph execution failed: " << rc << '\n';
        runner.FreeTensor(dev_a); runner.FreeTensor(dev_b); runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d); runner.FreeTensor(dev_e); runner.FreeTensor(dev_f);
        runner.Finalize();
        return rc;
    }

    // =========================================================================
    // VALIDATE RESULTS - Retrieve and verify output
    // =========================================================================
    std::cout << "\n=== Validating Results ===" << '\n';
    std::vector<float> host_result(SIZE);
    rc = runner.CopyFromDevice(host_result.data(), dev_f, BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy result from device: " << rc << '\n';
        runner.FreeTensor(dev_a); runner.FreeTensor(dev_b); runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d); runner.FreeTensor(dev_e); runner.FreeTensor(dev_f);
        runner.Finalize();
        return rc;
    }

    // Print sample values
    std::cout << "First 10 elements of result:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  f[" << i << "] = " << host_result[i] << '\n';
    }

    // Validate result
    constexpr float EXPECTED = 42.0f;  // (2+3+1)*(2+3+2) = 6*7 = 42
    bool all_correct = true;
    int error_count = 0;
    for (int i = 0; i < SIZE; i++) {
        if (std::abs(host_result[i] - EXPECTED) > 0.001f) {
            if (error_count < 5) {
                std::cerr << "ERROR: f[" << i << "] = " << host_result[i]
                          << ", expected " << EXPECTED << '\n';
            }
            error_count++;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "\n✓ SUCCESS: All " << SIZE << " elements are correct (42.0)\n";
        std::cout << "Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42\n";
    } else {
        std::cerr << "\n✗ FAILED: " << error_count << " elements are incorrect\n";
    }

    // Print handshake results
    runner.PrintHandshakeResults();

    // Cleanup
    std::cout << "\n=== Cleaning Up ===" << '\n';
    runner.FreeTensor(dev_a);
    runner.FreeTensor(dev_b);
    runner.FreeTensor(dev_c);
    runner.FreeTensor(dev_d);
    runner.FreeTensor(dev_e);
    runner.FreeTensor(dev_f);
    std::cout << "Freed all device tensors\n";

    runner.Finalize();

    if (rc != 0 || !all_correct) {
        std::cerr << "=== Execution Failed ===" << '\n';
        return -1;
    } else {
        std::cout << "=== Success ===" << '\n';
    }

    return 0;
}
