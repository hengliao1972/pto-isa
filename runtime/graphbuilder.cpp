/**
 * AICPU Kernel Launcher Example
 * This program demonstrates how to launch an AICPU kernel using CANN runtime
 * APIs
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <runtime/rt.h>
#include <string>
#include <vector>
#include "graph/graph.h"
#include "common/handshake.h"
#include "common/kernel_args.h"
#include <assert.h>

/**
 * @brief Kernel arguments for the AICPU kernel
 * @details This structure is used to pass arguments to the AICPU kernel.
 * It contains a pointer to the device arguments.
 * The thing important is
 *   1. the offset from KernelArgs to deviceArgs pointer
 *   2. the offset from DeviceArgs to aicpuSoBin
 *   3. the offset from DeviceArgs to aicpuSoLen
 * which are a hardcoded in the AICPU kernel("libaicpu_extend_kernels.so").
 * There are also hardcoded three function names(see graph_executor.cpp):
 *   1. StaticTileFwkBackendKernelServer
 *   2. DynTileFwkBackendKernelServerInit
 *   3. DynTileFwkBackendKernelServer
 * which will called when you launch the following kernels of
 * libtilefwk_backend_server.so:
 *   1. StaticTileFwkKernelServer
 *   2. DynTileFwkKernelServerInit
 *   3. DynTileFwkKernelServer
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
};

static int ReadFile(const std::string &path, std::vector<uint8_t> &buf) {//以二进制方式完整读取一个文件的内容到buf
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        std::cerr << "无法打开内核文件: " << path << '\n';
        return -1;
    }
    std::streamsize size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(size));
    if (!fs.read(reinterpret_cast<char *>(buf.data()), size)) {
        std::cerr << "读取内核文件失败: " << path << '\n';
        return -1;
    }
    return 0;
}

/**
 * Helper class that extends KernelArgs with initialization methods for host
 *
 * This class wraps KernelArgs and provides host-side initialization methods
 * for allocating device memory and copying data to the device. It separates
 * the concerns of device memory management (host-only) from the structure
 * layout (shared with kernels).
 *
 * The helper provides implicit conversion to KernelArgs* for seamless use
 * with runtime APIs.
 */
struct KernelArgsHelper {
    KernelArgs args;

    int InitDeviceArgs(const DeviceArgs &hostDeviceArgs) {
        // Allocate device memory for deviceArgs
        if (args.deviceArgs == nullptr) {
            void *deviceArgsDev = nullptr;
            uint64_t deviceArgsSize = sizeof(DeviceArgs);
            int allocRc = rtMalloc(&deviceArgsDev, deviceArgsSize, RT_MEMORY_HBM, 0);
            if (allocRc != 0) {
                std::cerr << "Error: rtMalloc for deviceArgs failed: " << allocRc << '\n';
                return allocRc;
            }
            args.deviceArgs = reinterpret_cast<int64_t *>(deviceArgsDev);
        }
        // Copy hostDeviceArgs to device memory via deviceArgs
        int rc =
            rtMemcpy(args.deviceArgs, sizeof(DeviceArgs), &hostDeviceArgs, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
            rtFree(args.deviceArgs);
            args.deviceArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int FinalizeDeviceArgs() {
        if (args.deviceArgs != nullptr) {
            int rc = rtFree(args.deviceArgs);
            args.deviceArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int InitGraphArgs(const Graph& hostGraph) {
        if (args.graphArgs == nullptr) {
            void *graphDev = nullptr;
            uint64_t graphSize = sizeof(Graph);
            int allocRc = rtMalloc(&graphDev, graphSize, RT_MEMORY_HBM, 0);
            if (allocRc != 0) {
                std::cerr << "Error: rtMalloc for graphArgs failed: " << allocRc << '\n';
                return allocRc;
            }
            args.graphArgs = reinterpret_cast<Graph*>(graphDev);
        }
        int rc = rtMemcpy(args.graphArgs, sizeof(Graph), &hostGraph, sizeof(Graph), RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy for graph failed: " << rc << '\n';
            rtFree(args.graphArgs);
            args.graphArgs = nullptr;
            return rc;
        }
        return 0;
    }

    int FinalizeGraphArgs() {
        if (args.graphArgs != nullptr) {
            int rc = rtFree(args.graphArgs);
            args.graphArgs = nullptr;
            return rc;
        }
        return 0;
    }

    /**
     * Implicit conversion operators for seamless use with runtime APIs
     *
     * These operators allow KernelArgsHelper to be used wherever KernelArgs*
     * is expected, enabling transparent device memory management while
     * maintaining API compatibility.
     */
    // Allow implicit conversion to KernelArgs*
    operator KernelArgs*() { return &args; }
    KernelArgs* operator&() { return &args; }
};

struct AicpuSoInfo {
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};

    int Init(const std::string &soPath) {
        std::ifstream file(soPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << soPath << '\n';
            return -1;
        }

        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);

        void *dAicpuData = nullptr;
        // Don't know why use 0 for module id, but it works.
        int rc = rtMalloc(&dAicpuData, fileSize, RT_MEMORY_HBM, 0);
        if (rc != 0) {
            std::cerr << "Error: rtMalloc failed: " << rc << '\n';
            return rc;
        }
        rc = rtMemcpy(dAicpuData, fileSize, buffer.data(), fileSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
            rtFree(dAicpuData);
            dAicpuData = nullptr;
            return rc;
        }

        aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
        aicpuSoLen = fileSize;
        return 0;
    }

    int Finalize() {
        if (aicpuSoBin != 0) {
            int rc = rtFree(reinterpret_cast<void *>(aicpuSoBin));
            aicpuSoBin = 0;
            return rc;
        }
        return 0;
    }
};

class DeviceRunner {
  public:
    static DeviceRunner &Get() {
        static DeviceRunner runner;
        return runner;
    }

    int LaunchAiCpuKernel(rtStream_t stream, KernelArgs *kArgs, const char *kernelName, int aicpuNum) {
        struct Args {
            KernelArgs kArgs;
            char kernelName[32];
            const char soName[32] = {"libaicpu_extend_kernels.so"};
            const char opName[32] = {""};
        } args;

        args.kArgs = *kArgs;
        std::strncpy(args.kernelName, kernelName, sizeof(args.kernelName) - 1);
        args.kernelName[sizeof(args.kernelName) - 1] = '\0';

        rtAicpuArgsEx_t rtArgs;
        std::memset(&rtArgs, 0, sizeof(rtArgs));
        rtArgs.args = &args;
        rtArgs.argsSize = sizeof(args);
        rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
        rtArgs.soNameAddrOffset = offsetof(struct Args, soName);

        return rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, &rtArgs,
                                             nullptr, stream, 0);
    }

    int LauncherAicoreKernel(rtStream_t stream, KernelArgs *kernelArgs) {
        const std::string binPath = "./aicore/kernel.o";
        std::vector<uint8_t> bin;
        if (ReadFile(binPath, bin) != 0) {
            return -1;
        }

        size_t binSize = bin.size();
        const void *binData = bin.data();

        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;
        binary.data = binData;
        binary.length = binSize;
        void *binHandle = nullptr;
        int rc = rtRegisterAllKernel(&binary, &binHandle);
        if (rc != RT_ERROR_NONE) {
            std::cerr << "rtRegisterAllKernel失败: " << rc << '\n';
            return rc;
        }

        struct Args {
            int64_t *hankArgs;
        };
        Args args = {kernelArgs->hankArgs};
        rtArgsEx_t rtArgs;
        std::memset(&rtArgs, 0, sizeof(rtArgs));
        rtArgs.args = &args;
        rtArgs.argsSize = sizeof(args);

        rtTaskCfgInfo_t cfg = {};
        cfg.schemMode = RT_SCHEM_MODE_BATCH;

        rc = rtKernelLaunchWithHandleV2(binHandle, 0, 1, &rtArgs, nullptr, stream, &cfg);
        if (rc != RT_ERROR_NONE) {
            std::cerr << "rtKernelLaunchWithHandleV2失败: " << rc << '\n';
            return rc;
        }

        return rc;
    }

    int Run(rtStream_t streamAicpu, rtStream_t streamAicore, KernelArgs *kernelArgs, int launchAicpuNum = 5) {
        if (kernelArgs == nullptr) {
            std::cerr << "Error: kernelArgs is null" << '\n';
            return -1;
        }

        // Launch init which save the Aicpu So to device and bind the function names
        int rc = LaunchAiCpuKernel(streamAicpu, kernelArgs, "DynTileFwkKernelServerInit", 1);
        if (rc != 0) {
            std::cerr << "Error: LaunchAiCpuKernel (init) failed: " << rc << '\n';
            return rc;
        }

        // Launch main kernel
        rc = LaunchAiCpuKernel(streamAicpu, kernelArgs, "DynTileFwkKernelServer", launchAicpuNum);
        if (rc != 0) {
            std::cerr << "Error: LaunchAiCpuKernel (main) failed: " << rc << '\n';
            return rc;
        }

        rc = LauncherAicoreKernel(streamAicore, kernelArgs);

        return 0;
    }

  private:
    DeviceRunner() {}
};

void MvHankArg(KernelArgs& kernelArgs, Handshake* args, int num_cores) {

    void *hankDev = nullptr;
    size_t total_size = sizeof(Handshake) * num_cores;
    int rc = rtMalloc(&hankDev, total_size, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        std::cerr << "Error: rtMalloc failed: " << rc << '\n';
        rtFree(hankDev);
        hankDev = nullptr;
        return;
    }

    rc = rtMemcpy(hankDev, total_size, args, total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        rtFree(hankDev);
        hankDev = nullptr;
        return;
    }

    kernelArgs.hankArgs = reinterpret_cast<int64_t *>(hankDev);
    kernelArgs.core_num = num_cores;
}


void PrintResult(KernelArgs& kernelArgs, int num_cores) {
    std::vector<Handshake> host_results(num_cores);
    size_t total_size = sizeof(Handshake) * num_cores;
    rtMemcpy(host_results.data(), total_size, kernelArgs.hankArgs, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << num_cores << " cores:" << std::endl;
    for (int i = 0; i < num_cores; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << host_results[i].aicore_done
                  << " aicpu_ready=" << host_results[i].aicpu_ready
                  << " control=" << host_results[i].control
                  << " task=" << host_results[i].task << std::endl;
    }
}


// Example usage
int main(int argc, char **argv) {
    std::cout << "=== Launching AICPU Kernel with Graph ===" << '\n';

    constexpr int NUM_CORES = 3;  // 1 AIC + 2 AIV cores for 1c2v architecture
    Handshake hankArgs[NUM_CORES] = {};

    // Initialize all cores' handshake buffers to default state
    // Protocol initial state:
    //   - aicpu_ready=0: AICPU will set to 1 when ready
    //   - aicore_done=0: AICore will set to core_id+1 when ready
    //   - control=0: Execute mode (1=quit)
    //   - task=0: No task assigned
    //   - task_status=0: Idle state
    for (int i = 0; i < NUM_CORES; i++) {
        hankArgs[i].aicpu_ready = 0;
        hankArgs[i].aicore_done = 0;
        hankArgs[i].control = 0;      // 0=execute
        hankArgs[i].task = 0;         // 0=no task initially
        hankArgs[i].task_status = 0;  // 0=idle
    }

    // Parse device id from main's argument (expected range: 0-15)
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
    int devRc = rtSetDevice(deviceId);
    if (devRc != 0) {
        std::cerr << "Error: rtSetDevice(" << deviceId << ") failed: " << devRc << '\n';
        return devRc;
    }

    rtStream_t streamAicpu = nullptr;
    rtStream_t streamAicore = nullptr;
    int rc = rtStreamCreate(&streamAicpu, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamCreate(&streamAicore, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate failed: " << rc << '\n';
        return rc;
    }

    std::string soPath = "./aicpu/libaicpu_graph_kernel.so";
    AicpuSoInfo soInfo{};
    rc = soInfo.Init(soPath);
    if (rc != 0) {
        std::cerr << "Error: AicpuSoInfo::Init failed: " << rc << '\n';
        rtStreamDestroy(streamAicpu);
        rtStreamDestroy(streamAicore);
        streamAicpu = nullptr;
        streamAicore = nullptr;
        return rc;
    }

    KernelArgsHelper kernelArgs{};
    DeviceArgs deviceArgs{};
    deviceArgs.aicpuSoBin = soInfo.aicpuSoBin;
    deviceArgs.aicpuSoLen = soInfo.aicpuSoLen;
    rc = kernelArgs.InitDeviceArgs(deviceArgs);

    MvHankArg(kernelArgs.args, hankArgs, NUM_CORES);

    if (rc != 0) {
        std::cerr << "Error: KernelArgs::InitDeviceArgs failed: " << rc << '\n';
        soInfo.Finalize();
        rtStreamDestroy(streamAicpu);
        rtStreamDestroy(streamAicore);
        streamAicpu = nullptr;
        streamAicore = nullptr;
        return rc;
    }
    // =========================================================================
    // Run formula computation in a scope to avoid goto issues
    // =========================================================================
    {
        std::cout << "\n=== Allocating Device Memory for Formula ===" << '\n';

        constexpr int ROWS = 128;
        constexpr int COLS = 128;
        constexpr int SIZE = ROWS * COLS;  // 16384 elements
        constexpr size_t BYTES = SIZE * sizeof(float);

        // Allocate device memory for 6 tensors: a, b, c, d, e, f
        void *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
        void *dev_d = nullptr, *dev_e = nullptr, *dev_f = nullptr;

        // Helper lambda for cleanup
        auto cleanup = [&]() {
            if (dev_a) rtFree(dev_a);
            if (dev_b) rtFree(dev_b);
            if (dev_c) rtFree(dev_c);
            if (dev_d) rtFree(dev_d);
            if (dev_e) rtFree(dev_e);
            if (dev_f) rtFree(dev_f);
        };

        rc = rtMalloc(&dev_a, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_a failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMalloc(&dev_b, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_b failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMalloc(&dev_c, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_c failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMalloc(&dev_d, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_d failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMalloc(&dev_e, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_e failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMalloc(&dev_f, BYTES, RT_MEMORY_HBM, 0);
        if (rc != 0) { std::cerr << "Error: rtMalloc for dev_f failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        std::cout << "Allocated 6 tensors (128x128 each, " << BYTES << " bytes per tensor)\n";

        // Initialize host arrays with constant values: a=2.0, b=3.0
        std::vector<float> host_a(SIZE, 2.0f);
        std::vector<float> host_b(SIZE, 3.0f);

        // Copy initialized data to device
        rc = rtMemcpy(dev_a, BYTES, host_a.data(), BYTES, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) { std::cerr << "Error: rtMemcpy for dev_a failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        rc = rtMemcpy(dev_b, BYTES, host_b.data(), BYTES, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) { std::cerr << "Error: rtMemcpy for dev_b failed: " << rc << '\n'; cleanup(); kernelArgs.FinalizeDeviceArgs(); soInfo.Finalize(); rtStreamDestroy(streamAicpu); rtStreamDestroy(streamAicore); return rc; }

        std::cout << "Initialized input tensors: a=2.0, b=3.0 (all elements)\n";
        std::cout << "Expected result: f = (2+3+1)*(2+3+2) = 6*7 = 42.0\n";

        // =========================================================================
        // Create task graph for formula: (a + b + 1)(a + b + 2)
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

        // Initialize graph args
        rc = kernelArgs.InitGraphArgs(testGraph);
        if (rc != 0) {
            std::cerr << "Error: KernelArgs::InitGraphArgs failed: " << rc << '\n';
            cleanup();
            kernelArgs.FinalizeDeviceArgs();
            soInfo.Finalize();
            rtStreamDestroy(streamAicpu);
            rtStreamDestroy(streamAicore);
            return rc;
        }
        std::cout << "Graph transferred to device memory\n\n";

        DeviceRunner &runner = DeviceRunner::Get();
        int launchAicpuNum = 1;
        rc = runner.Run(streamAicpu, streamAicore, &kernelArgs.args, launchAicpuNum);

        rc = rtStreamSynchronize(streamAicpu);
        if (rc != 0) {
            std::cerr << "Error: rtStreamSynchronize failed: " << rc << '\n';
            cleanup();
            return rc;
        }

        rc = rtStreamSynchronize(streamAicore);
        if (rc != 0) {
            std::cerr << "Error: rtStreamSynchronize failed: " << rc << '\n';
            cleanup();
            return rc;
        }

        // =========================================================================
        // Retrieve and validate results
        // =========================================================================
        std::cout << "\n=== Validating Results ===" << '\n';
        {
            std::vector<float> host_result(SIZE);
            rc = rtMemcpy(host_result.data(), BYTES, dev_f, BYTES, RT_MEMCPY_DEVICE_TO_HOST);
            if (rc != 0) {
                std::cerr << "Error: rtMemcpy for result failed: " << rc << '\n';
                cleanup();
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
        }

        PrintResult(kernelArgs.args, NUM_CORES);

        // Cleanup device memory
        std::cout << "\n=== Cleaning Up ===" << '\n';
        cleanup();
        std::cout << "Freed all device tensors\n";
    }

    kernelArgs.FinalizeGraphArgs();
    kernelArgs.FinalizeDeviceArgs();
    soInfo.Finalize();
    rtStreamDestroy(streamAicpu);
    rtStreamDestroy(streamAicore);
    streamAicpu = nullptr;
    streamAicore = nullptr;

    if (rc != 0) {
        std::cerr << "=== Launch Failed ===" << '\n';
    } else {
        std::cout << "=== Launch Success ===" << '\n';
    }

    return rc;
}
