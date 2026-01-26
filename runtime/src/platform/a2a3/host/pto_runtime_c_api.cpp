/**
 * PTO Runtime C API - Implementation
 *
 * Wraps C++ classes as opaque pointers, providing C interface for ctypes bindings.
 * Minimal API focused on graph initialization/validation and device runner lifecycle.
 */

#include "pto_runtime_c_api.h"
#include "devicerunner.h"
#include "graph.h"  // Included from tests/example_graph_impl via CMake include paths

extern "C" {

/* =========================================================================== */
/* Graph API Implementation */
/* =========================================================================== */
int InitGraphImpl(Graph **graph);
int ValidateGraphImpl(Graph *graph);

int InitGraph(GraphHandle graph) {
    if (graph == NULL) {
        return -1;
    }
    try {
        // graph is a void*, we need to pass its address as Graph**
        Graph** g_ptr = reinterpret_cast<Graph**>(graph);
        return InitGraphImpl(g_ptr);
    } catch (...) {
        return -1;
    }
}

int ValidateGraph(GraphHandle graph) {
    if (graph == NULL) {
        return -1;
    }
    try {
        Graph* g = static_cast<Graph*>(graph);
        return ValidateGraphImpl(g);
    } catch (...) {
        return -1;
    }
}

/* =========================================================================== */
/* DeviceRunner API Implementation */
/* =========================================================================== */

int DeviceRunner_Init(int device_id,
                      const uint8_t* aicpu_binary, size_t aicpu_size,
                      const uint8_t* aicore_binary, size_t aicore_size,
                      const char* pto_isa_root) {
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0 ||
        pto_isa_root == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        std::vector<uint8_t> aicpuVec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicoreVec(aicore_binary, aicore_binary + aicore_size);
        return runner.Init(device_id, aicpuVec, aicoreVec, std::string(pto_isa_root));
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_Run(GraphHandle graph, int num_cores, int launch_aicpu_num) {
    if (graph == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        Graph* g = static_cast<Graph*>(graph);
        return runner.Run(*g, num_cores, launch_aicpu_num);
    } catch (...) {
        return -1;
    }
}

void DeviceRunner_PrintHandshakeResults(GraphHandle graph) {
    if (graph == NULL) {
        return;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        Graph* g = static_cast<Graph*>(graph);
        runner.PrintHandshakeResults(*g);
    } catch (...) {
        // Silently ignore errors on print
    }
}

int DeviceRunner_Finalize(void) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.Finalize();
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_CompileAndLoadKernel(int func_id,
                                      const char* kernel_path,
                                      int core_type) {
    if (kernel_path == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.CompileAndLoadKernel(func_id, std::string(kernel_path), core_type);
    } catch (...) {
        return -1;
    }
}

}  /* extern "C" */
