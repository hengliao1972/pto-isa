/**
 * PTO Runtime System - Main Header
 * 
 * This is the primary header for the PTO runtime system. It provides a unified
 * interface for all platforms while allowing platform-specific implementations.
 * 
 * Architecture:
 * =============
 * 
 * The runtime is organized into three layers:
 * 
 * 1. Common Layer (pto_runtime_common.h/c):
 *    - Task table and task management
 *    - TensorMap for dependency tracking
 *    - Input/output argument data structures
 *    - Record & Replay infrastructure
 *    - Cycle tracing
 * 
 * 2. Platform Layer (pto_runtime_<platform>.h/c):
 *    - Task completion with platform-specific dependency management
 *    - Ready queue management
 *    - Worker thread implementation
 *    - Platform-specific runtime entry points
 * 
 * 3. Unified Interface (this file):
 *    - Platform selection via macros
 *    - Backward-compatible API
 * 
 * Platform Selection:
 * ===================
 * 
 * Define one of the following before including this header to select platform:
 * 
 *   #define PTO_PLATFORM_ARM64    - ARM64 with single ready queue
 *   #define PTO_PLATFORM_A2A3     - Ascend A2/A3 with dual queue (vector/cube)
 * 
 * If no platform is defined, ARM64 is used by default.
 * 
 * Dependency Management Approaches:
 * =================================
 * 
 * ARM64 (Distributed):
 *   - Dependencies are tracked via fanin/fanout lists in task entries
 *   - Task completion increments successors' fanin_refcount until it reaches fanin_count
 *   - Ready tasks are pushed to a single unified queue
 *   - Simple and efficient for homogeneous execution
 * 
 * A2A3 (Dedicated Module):
 *   - A dedicated dependency management module routes tasks
 *   - Dual ready queues: vector (is_cube=false) and cube (is_cube=true)
 *   - Task completion goes through the dedicated module
 *   - Supports heterogeneous NPU architecture with separate execution units
 * 
 * Usage:
 * ======
 * 
 *   // For ARM64 (default)
 *   #include "pto_runtime.h"
 *   
 *   // For A2A3/Ascend
 *   #define PTO_PLATFORM_A2A3
 *   #include "pto_runtime.h"
 * 
 * Execution:
 * ==========
 * 
 *   // ARM64: Single-queue multi-threaded execution
 *   runtime_entry_arm64(orch_func, user_data, num_workers, threshold);
 *   
 *   // A2A3: Dual-queue heterogeneous execution
 *   runtime_entry_a2a3(orch_func, user_data, num_vector_workers, num_cube_workers, threshold);
 */

#ifndef PTO_RUNTIME_H
#define PTO_RUNTIME_H

// =============================================================================
// Platform Selection
// =============================================================================

// Default to ARM64 if no platform specified
#if !defined(PTO_PLATFORM_ARM64) && !defined(PTO_PLATFORM_A2A3)
#define PTO_PLATFORM_ARM64
#endif

// =============================================================================
// Include Common Infrastructure
// =============================================================================

#include "pto_runtime_common.h"

// =============================================================================
// Include Platform-Specific Headers
// =============================================================================

#ifdef PTO_PLATFORM_ARM64
#include "pto_runtime_arm64.h"
#endif

#ifdef PTO_PLATFORM_A2A3
#include "runtime_a2a3/pto_runtime_a2a3.h"
#endif

// =============================================================================
// Platform-Unified API Macros
// =============================================================================

/**
 * Platform-independent runtime entry point macro
 * 
 * For ARM64: Uses single queue
 * For A2A3:  Requires specifying vector and cube worker counts
 */
#ifdef PTO_PLATFORM_ARM64
#define pto_runtime_entry(orch_func, user_data, num_workers, threshold) \
    runtime_entry_arm64(orch_func, user_data, num_workers, threshold)
#endif

#ifdef PTO_PLATFORM_A2A3
// A2A3 requires separate vector/cube worker specification
// Use runtime_entry_a2a3() directly for full control
#define pto_runtime_entry_a2a3(orch_func, user_data, nvec, ncube, threshold) \
    runtime_entry_a2a3(orch_func, user_data, nvec, ncube, threshold)
#endif

// =============================================================================
// Backward Compatibility Layer
// =============================================================================

/**
 * These functions maintain backward compatibility with existing code.
 * When compiled for A2A3, they use the A2A3-specific implementations.
 * When compiled for ARM64, they use the ARM64 implementations.
 */

#ifdef PTO_PLATFORM_A2A3
// A2A3 platform uses dedicated dependency module
#define pto_task_submit         pto_task_submit_a2a3
#define pto_task_complete       pto_task_complete_a2a3
#define pto_task_complete_threadsafe pto_task_complete_a2a3_threadsafe
#define pto_loop_replay         pto_loop_replay_a2a3
#endif

// ARM64 implementations are in pto_runtime_arm64.c (no macro redefinition needed)

// =============================================================================
// Convenience Macros for Code Generation
// =============================================================================

/**
 * Convenience macro to schedule an InCore function call with 1 input, 1 output
 */
#define PTO_SCHEDULE_INCORE_1IN_1OUT(rt, func, in_ptr, in_row, in_col, in_rows, in_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in_ptr, in_row, in_col, in_rows, in_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

/**
 * Convenience macro to schedule an InCore function call with 2 inputs, 1 output
 */
#define PTO_SCHEDULE_INCORE_2IN_1OUT(rt, func, in1_ptr, in1_row, in1_col, in1_rows, in1_cols, \
                                      in2_ptr, in2_row, in2_col, in2_rows, in2_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in1_ptr, in1_row, in1_col, in1_rows, in1_cols); \
            pto_task_add_input(rt, _tid, in2_ptr, in2_row, in2_col, in2_rows, in2_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

// =============================================================================
// Example Usage (compile with PTO_RUNTIME_EXAMPLE to enable)
// =============================================================================

#ifdef PTO_RUNTIME_EXAMPLE

// Forward declarations for InCore functions
void rowmax(float* input, float* output);
void rowexpandsub(float* input_x, float* input_row, float* output);
void elem_exp(float* input, float* output);
void rowsum(float* input, float* output);
void rowexpanddiv(float* input_x, float* input_row, float* output);

/**
 * Example: Generated Orchestration function for fused_softmax
 */
static void fused_softmax_orchestration(PTORuntime* rt,
                                        float* input, float* output,
                                        float* temp_rowmax, float* temp_shifted,
                                        float* temp_exp, float* temp_rowsum) {
    // Task 0: rowmax(input) -> temp_rowmax
    int32_t t0 = pto_task_alloc(rt, "rowmax", (void*)rowmax, 0, 0);
    pto_task_add_input(rt, t0, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t0, temp_rowmax, 0, 0, 8, 1);
    pto_task_submit(rt, t0);
    
    // Task 1: rowexpandsub(input, temp_rowmax) -> temp_shifted
    int32_t t1 = pto_task_alloc(rt, "rowexpandsub", (void*)rowexpandsub, 0, 0);
    pto_task_add_input(rt, t1, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t1, temp_rowmax, 0, 0, 8, 1);  // Creates dependency on t0
    pto_task_add_output(rt, t1, temp_shifted, 0, 0, 8, 8);
    pto_task_submit(rt, t1);
    
    // Task 2: elem_exp(temp_shifted) -> temp_exp
    int32_t t2 = pto_task_alloc(rt, "elem_exp", (void*)elem_exp, 0, 0);
    pto_task_add_input(rt, t2, temp_shifted, 0, 0, 8, 8);  // Creates dependency on t1
    pto_task_add_output(rt, t2, temp_exp, 0, 0, 8, 8);
    pto_task_submit(rt, t2);
    
    // Task 3: rowsum(temp_exp) -> temp_rowsum
    int32_t t3 = pto_task_alloc(rt, "rowsum", (void*)rowsum, 0, 0);
    pto_task_add_input(rt, t3, temp_exp, 0, 0, 8, 8);  // Creates dependency on t2
    pto_task_add_output(rt, t3, temp_rowsum, 0, 0, 8, 1);
    pto_task_submit(rt, t3);
    
    // Task 4: rowexpanddiv(temp_exp, temp_rowsum) -> output
    int32_t t4 = pto_task_alloc(rt, "rowexpanddiv", (void*)rowexpanddiv, 0, 0);
    pto_task_add_input(rt, t4, temp_exp, 0, 0, 8, 8);     // Creates dependency on t2
    pto_task_add_input(rt, t4, temp_rowsum, 0, 0, 8, 1);  // Creates dependency on t3
    pto_task_add_output(rt, t4, output, 0, 0, 8, 8);
    pto_task_submit(rt, t4);
    
    // Execute all scheduled tasks
    pto_execute_all(rt);
}

// Example main function
static int example_main() {
    // Initialize runtime
    PTORuntime rt;
    pto_runtime_init(&rt);
    
    // Allocate buffers
    float input[64];
    float output[64];
    float temp_rowmax[8];
    float temp_shifted[64];
    float temp_exp[64];
    float temp_rowsum[8];
    
    // Initialize input
    for (int i = 0; i < 64; i++) {
        input[i] = (float)i / 64.0f;
    }
    
    // Execute orchestration function
    fused_softmax_orchestration(&rt, input, output,
                                temp_rowmax, temp_shifted,
                                temp_exp, temp_rowsum);
    
    // Print statistics
    pto_runtime_stats(&rt);
    
    // Shutdown runtime
    pto_runtime_shutdown(&rt);
    
    return 0;
}

#endif // PTO_RUNTIME_EXAMPLE

#endif // PTO_RUNTIME_H
