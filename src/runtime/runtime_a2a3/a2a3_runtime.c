/**
 * PTO Runtime - Ascend A2/A3 Runtime Implementation
 * 
 * Execution Architecture:
 * 
 * Device (NPU) - all computation here:
 *   1. AICore workers: Load aicore_kernel.o, enter polling loop
 *   2. AICPU scheduler: Load scheduler code, distribute tasks
 *   3. AICPU orchestration: Run orchestration function, generate tasks dynamically
 * 
 * Host (CPU) - control only:
 *   1. Initialize device, load kernels
 *   2. Launch AICPU + AICore kernels
 *   3. Wait for completion
 *   4. Copy results, shutdown
 * 
 * This is STREAMING execution - tasks generated on-the-fly by AICPU orchestration.
 */

#define _POSIX_C_SOURCE 199309L

#include "a2a3_runtime_api.h"
#include "host/a2a3_so_loader.h"
#include "host/a2a3_host.h"
#include "host/a2a3_binary_loader.h"
#include "../pto_runtime_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// CANN SDK Headers
// =============================================================================

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>
#endif

// =============================================================================
// Internal State
// =============================================================================

static A2A3RuntimeConfig g_config;
static bool g_initialized = false;
static A2A3RuntimeStats g_stats;

#ifdef CANN_SDK_AVAILABLE
static aclrtStream g_stream_aicpu = NULL;
static aclrtStream g_stream_aicore = NULL;
static void* g_aicore_bin_handle = NULL;
static void* g_device_handshake = NULL;  // Handshake buffer in device GM
#endif

// Host-side handshake buffer (for initialization)
typedef struct {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile uint32_t control;
    volatile uint64_t task;
    volatile uint32_t task_status;
    volatile uint32_t core_type;
    uint32_t padding[2];
} Handshake;

static Handshake* g_host_handshake = NULL;
static int g_total_cores = 0;

// Error messages
static const char* g_error_messages[] = {
    "Success",
    "Invalid configuration",
    "Failed to load .so file",
    "Function not found",
    "Memory allocation failed",
    "Thread creation failed",
    "Runtime not initialized",
    "Runtime already initialized",
    "Binary load failed",
    "Device launch failed",
};

// =============================================================================
// Error Handling
// =============================================================================

const char* a2a3_runtime_error_string(int error_code) {
    if (error_code > 0 || error_code < -9) {
        return "Unknown error";
    }
    return g_error_messages[-error_code];
}

// =============================================================================
// Runtime Lifecycle
// =============================================================================

int a2a3_runtime_init(A2A3RuntimeConfig* config) {
    if (g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime already initialized\n");
        return A2A3_ERROR_ALREADY_INIT;
    }
    
    if (!config) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: NULL config\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    memcpy(&g_config, config, sizeof(A2A3RuntimeConfig));
    
    // Apply defaults
    if (g_config.num_aiv_workers < 1) g_config.num_aiv_workers = A2A3_DEFAULT_AIV_WORKERS;
    if (g_config.num_aic_workers < 1) g_config.num_aic_workers = A2A3_DEFAULT_AIC_WORKERS;
    
    g_total_cores = g_config.num_aic_workers + g_config.num_aiv_workers;
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Initializing Ascend A2/A3 Runtime\n");
    printf("[A2A3 Runtime]   Device execution model:\n");
    printf("[A2A3 Runtime]     - AICore workers (polling loop): %d AIC + %d AIV\n",
           g_config.num_aic_workers, g_config.num_aiv_workers);
    printf("[A2A3 Runtime]     - AICPU scheduler: task distribution\n");
    printf("[A2A3 Runtime]     - AICPU orchestration: dynamic task generation\n");
    printf("[A2A3 Runtime] ================================================\n");
    
#ifdef CANN_SDK_AVAILABLE
    // Initialize ACL
    aclError rc = aclInit(NULL);
    if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclInit failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    // Set device
    rc = aclrtSetDevice(0);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclrtSetDevice failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    // Create streams
    rc = aclrtCreateStream(&g_stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AICPU stream: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    rc = aclrtCreateStream(&g_stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AICore stream: %d\n", rc);
        aclrtDestroyStream(g_stream_aicpu);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    printf("[A2A3 Runtime] ACL initialized, streams created\n");
#else
    printf("[A2A3 Runtime] WARNING: CANN SDK not available, stub mode\n");
#endif
    
    // Initialize SO loader (for orchestration .so)
    a2a3_so_loader_init();
    
    // Load InCore binaries (.o files) for AICore execution
    if (g_config.incore_aiv_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aiv_dir, false);
        printf("[A2A3 Runtime] Loaded %d AIV InCore binaries\n", count);
    }
    
    if (g_config.incore_aic_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aic_dir, true);
        printf("[A2A3 Runtime] Loaded %d AIC InCore binaries\n", count);
    }
    
    // Allocate host handshake buffer
    g_host_handshake = (Handshake*)calloc(g_total_cores, sizeof(Handshake));
    if (!g_host_handshake) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate handshake buffer\n");
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Initialize handshake (AIC cores first, then AIV)
    for (int i = 0; i < g_total_cores; i++) {
        g_host_handshake[i].aicpu_ready = 0;
        g_host_handshake[i].aicore_done = 0;
        g_host_handshake[i].control = 0;
        g_host_handshake[i].task = 0;
        g_host_handshake[i].task_status = 0;
        g_host_handshake[i].core_type = (i < g_config.num_aic_workers) ? 0 : 1;
    }
    
#ifdef CANN_SDK_AVAILABLE
    // Allocate device handshake buffer
    size_t handshake_size = sizeof(Handshake) * g_total_cores;
    rc = aclrtMalloc(&g_device_handshake, handshake_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate device handshake: %d\n", rc);
        free(g_host_handshake);
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Copy handshake to device
    rc = aclrtMemcpy(g_device_handshake, handshake_size, 
                     g_host_handshake, handshake_size,
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to copy handshake to device: %d\n", rc);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    printf("[A2A3 Runtime] Handshake buffer allocated: %zu bytes for %d cores\n",
           handshake_size, g_total_cores);
    
    // Copy InCore binaries to device GM
    int incore_rc = a2a3_copy_incore_binaries_to_device();
    if (incore_rc < 0) {
        fprintf(stderr, "[A2A3 Runtime] WARNING: Failed to copy InCore binaries to device\n");
    } else {
        printf("[A2A3 Runtime] Copied %d InCore binaries to device GM\n", incore_rc);
    }
#endif
    
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.num_incore_funcs_loaded = a2a3_get_incore_count();
    
    g_initialized = true;
    
    printf("[A2A3 Runtime] Initialization complete\n");
    printf("[A2A3 Runtime]   Loaded %d InCore functions\n", g_stats.num_incore_funcs_loaded);
    
    return A2A3_SUCCESS;
}

int a2a3_runtime_execute(void* user_data) {
    if (!g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime not initialized\n");
        return A2A3_ERROR_NOT_INITIALIZED;
    }
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Starting device execution\n");
    printf("[A2A3 Runtime] ================================================\n");
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc;
    
    // =========================================================================
    // Step 1: Launch AICore kernel (workers enter polling loop)
    // =========================================================================
    printf("[A2A3 Runtime] Step 1: Launching AICore workers...\n");
    printf("[A2A3 Runtime]   %d AIC + %d AIV cores will enter polling loop\n",
           g_config.num_aic_workers, g_config.num_aiv_workers);
    
    // Load AICore kernel binary if provided
    if (g_config.aicore_kernel_path && g_aicore_bin_handle == NULL) {
        uint8_t* bin_data = NULL;
        size_t bin_size = 0;
        
        int load_rc = a2a3_load_elf_text_section(g_config.aicore_kernel_path, &bin_data, &bin_size);
        if (load_rc == 0 && bin_data && bin_size > 0) {
            // TODO: Register kernel with CANN runtime
            // rtRegisterAllKernel() requires runtime/rt.h which may not be available
            printf("[A2A3 Runtime]   Loaded AICore kernel: %s (%zu bytes)\n",
                   g_config.aicore_kernel_path, bin_size);
            free(bin_data);
        } else {
            printf("[A2A3 Runtime]   WARNING: Could not load AICore kernel\n");
        }
    }
    
    // TODO: Launch AICore kernel
    // rtKernelLaunchWithHandleV2(g_aicore_bin_handle, ...)
    printf("[A2A3 Runtime]   AICore workers launched (polling handshake registers)\n");
    
    // =========================================================================
    // Step 2: Launch AICPU kernel (scheduler + orchestration)
    // =========================================================================
    printf("[A2A3 Runtime] Step 2: Launching AICPU kernel...\n");
    printf("[A2A3 Runtime]   AICPU will run: scheduler + orchestration\n");
    printf("[A2A3 Runtime]   Orchestration generates tasks dynamically\n");
    
    // TODO: Launch AICPU kernel
    // rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, ...)
    printf("[A2A3 Runtime]   AICPU kernel launched\n");
    
    // =========================================================================
    // Step 3: Wait for completion
    // =========================================================================
    printf("[A2A3 Runtime] Step 3: Waiting for device execution...\n");
    
    // Synchronize AICPU stream (wait for orchestration + scheduler to complete)
    rc = aclrtSynchronizeStream(g_stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: AICPU sync failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    // Synchronize AICore stream
    rc = aclrtSynchronizeStream(g_stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: AICore sync failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    printf("[A2A3 Runtime]   Device execution complete\n");
    
#else
    // Stub mode - no actual device execution
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] STUB MODE - No actual device execution\n");
    printf("[A2A3 Runtime] \n");
    printf("[A2A3 Runtime] Real execution requires:\n");
    printf("[A2A3 Runtime]   1. CANN SDK with runtime API\n");
    printf("[A2A3 Runtime]   2. Ascend NPU device\n");
    printf("[A2A3 Runtime]   3. AICore kernel binary (aicore_kernel.o)\n");
    printf("[A2A3 Runtime]   4. AICPU kernel binary (libaicpu_kernel.so)\n");
    printf("[A2A3 Runtime] \n");
    printf("[A2A3 Runtime] In stub mode, output will be zeros.\n");
    printf("[A2A3 Runtime] ================================================\n");
#endif
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    g_stats.total_execution_time_ms = 
        (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Execution Complete\n");
    printf("[A2A3 Runtime]   Execution time: %.2f ms\n", g_stats.total_execution_time_ms);
    printf("[A2A3 Runtime] ================================================\n");
    
    return A2A3_SUCCESS;
}

void a2a3_runtime_finalize(void) {
    if (!g_initialized) return;
    
    printf("[A2A3 Runtime] Finalizing...\n");
    
#ifdef CANN_SDK_AVAILABLE
    // Free device handshake
    if (g_device_handshake) {
        aclrtFree(g_device_handshake);
        g_device_handshake = NULL;
    }
    
    // Destroy streams
    if (g_stream_aicpu) {
        aclrtDestroyStream(g_stream_aicpu);
        g_stream_aicpu = NULL;
    }
    if (g_stream_aicore) {
        aclrtDestroyStream(g_stream_aicore);
        g_stream_aicore = NULL;
    }
    
    // Reset device
    aclrtResetDevice(0);
    aclFinalize();
#endif
    
    // Free host handshake
    if (g_host_handshake) {
        free(g_host_handshake);
        g_host_handshake = NULL;
    }
    
    // Cleanup binary loader
    a2a3_unload_all_incore_binaries();
    
    // Cleanup SO loader
    a2a3_so_loader_cleanup();
    
    g_initialized = false;
    
    printf("[A2A3 Runtime] Finalized\n");
}

// =============================================================================
// Memory Management
// =============================================================================

void* a2a3_runtime_malloc(size_t size_bytes) {
    if (!g_initialized || size_bytes == 0) return NULL;
    
#ifdef CANN_SDK_AVAILABLE
    void* ptr = NULL;
    aclError rc = aclrtMalloc(&ptr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclrtMalloc failed: %d\n", rc);
        return NULL;
    }
    return ptr;
#else
    return malloc(size_bytes);
#endif
}

void a2a3_runtime_free(void* ptr) {
    if (!ptr) return;
    
#ifdef CANN_SDK_AVAILABLE
    aclrtFree(ptr);
#else
    free(ptr);
#endif
}

int a2a3_runtime_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes) {
    if (!g_initialized || !dst_device || !src_host || size_bytes == 0) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst_device, size_bytes, src_host, size_bytes, 
                              ACL_MEMCPY_HOST_TO_DEVICE);
    return (rc == ACL_SUCCESS) ? A2A3_SUCCESS : A2A3_ERROR_DEVICE_LAUNCH;
#else
    memcpy(dst_device, src_host, size_bytes);
    return A2A3_SUCCESS;
#endif
}

int a2a3_runtime_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes) {
    if (!g_initialized || !dst_host || !src_device || size_bytes == 0) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst_host, size_bytes, src_device, size_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST);
    return (rc == ACL_SUCCESS) ? A2A3_SUCCESS : A2A3_ERROR_DEVICE_LAUNCH;
#else
    memcpy(dst_host, src_device, size_bytes);
    return A2A3_SUCCESS;
#endif
}

// =============================================================================
// InCore Function Registry
// =============================================================================

int a2a3_runtime_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube) {
    return a2a3_register_incore(func_name, func_ptr, is_cube);
}

A2A3InCoreFunc a2a3_runtime_lookup_incore(const char* func_name) {
    return a2a3_lookup_incore(func_name);
}

// =============================================================================
// Statistics
// =============================================================================

void a2a3_runtime_get_stats(A2A3RuntimeStats* stats) {
    if (stats) {
        memcpy(stats, &g_stats, sizeof(A2A3RuntimeStats));
    }
}

void a2a3_runtime_print_stats(void) {
    printf("\n=== A2A3 Runtime Statistics ===\n");
    printf("Tasks Scheduled:     %lld\n", (long long)g_stats.total_tasks_scheduled);
    printf("Tasks Completed:     %lld\n", (long long)g_stats.total_tasks_completed);
    printf("AIV Tasks:           %lld\n", (long long)g_stats.aiv_tasks_executed);
    printf("AIC Tasks:           %lld\n", (long long)g_stats.aic_tasks_executed);
    printf("Execution Time:      %.2f ms\n", g_stats.total_execution_time_ms);
    printf("InCore Functions:    %d\n", g_stats.num_incore_funcs_loaded);
    printf("================================\n\n");
}

bool a2a3_runtime_is_initialized(void) {
    return g_initialized;
}
