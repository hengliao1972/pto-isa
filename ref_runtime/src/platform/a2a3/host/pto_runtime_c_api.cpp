/**
 * PTO Runtime C API - Implementation
 *
 * Wraps C++ classes as opaque pointers, providing C interface for ctypes bindings.
 * Minimal API focused on graph initialization/validation and device runner lifecycle.
 */

#include "pto_runtime_c_api.h"
#include "devicerunner.h"
#include "graph.h"  // Included from tests/example_graph_impl via CMake include paths
#include <algorithm>
#include <new>

extern "C" {

/* =========================================================================== */
/* Graph API Implementation */
/* =========================================================================== */
int InitGraphImpl(Graph **graph);
int ValidateGraphImpl(Graph *graph);

GraphHandle Graph_Create(void) {
    try {
        return static_cast<GraphHandle>(new (std::nothrow) Graph());
    } catch (...) {
        return NULL;
    }
}

int Graph_Destroy(GraphHandle graph) {
    if (graph == NULL) {
        return -1;
    }
    try {
        delete static_cast<Graph*>(graph);
        return 0;
    } catch (...) {
        return -1;
    }
}

int Graph_AddTask(GraphHandle graph,
                  const uint64_t* args,
                  int num_args,
                  int func_id,
                  int core_type) {
    if (graph == NULL) {
        return -1;
    }
    try {
        Graph* g = static_cast<Graph*>(graph);
        return g->add_task(const_cast<uint64_t*>(args), num_args, func_id, core_type);
    } catch (...) {
        return -1;
    }
}

int Graph_AddSuccessor(GraphHandle graph, int from_task, int to_task) {
    if (graph == NULL) {
        return -1;
    }
    try {
        Graph* g = static_cast<Graph*>(graph);
        g->add_successor(from_task, to_task);
        return 0;
    } catch (...) {
        return -1;
    }
}

int Graph_GetTaskCount(GraphHandle graph) {
    if (graph == NULL) {
        return -1;
    }
    try {
        Graph* g = static_cast<Graph*>(graph);
        return g->get_task_count();
    } catch (...) {
        return -1;
    }
}

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

void* DeviceRunner_AllocateTensor(size_t bytes) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.AllocateTensor(bytes);
    } catch (...) {
        return NULL;
    }
}

void DeviceRunner_FreeTensor(void* dev_ptr) {
    if (dev_ptr == NULL) {
        return;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        runner.FreeTensor(dev_ptr);
    } catch (...) {
        return;
    }
}

int DeviceRunner_CopyToDevice(void* dev_ptr, const void* host_ptr, size_t bytes) {
    if (dev_ptr == NULL || host_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.CopyToDevice(dev_ptr, host_ptr, bytes);
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_CopyFromDevice(void* host_ptr, const void* dev_ptr, size_t bytes) {
    if (dev_ptr == NULL || host_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.CopyFromDevice(host_ptr, dev_ptr, bytes);
    } catch (...) {
        return -1;
    }
}

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

int DeviceRunner_SetProfileEnabled(int enabled) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        runner.SetProfileEnabled(enabled != 0);
        return 0;
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_ProfileEnabled(void) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.ProfileEnabled() ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_HasLastProfile(void) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.HasLastProfile() ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int DeviceRunner_GetLastProfile(PtoTaskProfileRecord* out_records, int max_records) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        const std::vector<TaskProfileRecord> profile = runner.GetLastProfile();
        const int n = static_cast<int>(profile.size());
        if (out_records == NULL || max_records <= 0) {
            return n;
        }
        const int m = std::min(n, max_records);
        for (int i = 0; i < m; i++) {
            const TaskProfileRecord& src = profile[static_cast<size_t>(i)];
            PtoTaskProfileRecord& dst = out_records[i];
            dst.task_id = src.task_id;
            dst.func_id = src.func_id;
            dst.core_type = src.core_type;
            dst.exec_core_id = src.exec_core_id;
            dst.exec_core_type = src.exec_core_type;
            dst.exec_phys_core_id = src.exec_phys_core_id;
            dst.start_time = src.start_time;
            dst.end_time = src.end_time;
            for (size_t j = 0; j < PTO_PROFILE_PMU_CNT; j++) {
                dst.pmu_cnt[j] = src.pmu_cnt[j];
            }
        }
        return m;
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
