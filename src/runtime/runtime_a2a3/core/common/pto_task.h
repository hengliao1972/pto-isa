/**
 * PTO Runtime - Task Structures for NPU Execution
 * 
 * This header defines structures shared between:
 * - AICPU (orchestration + scheduler)
 * - AICore (workers)
 * 
 * Compiled with both gcc (AICPU) and ccec (AICore).
 */

#ifndef PTO_TASK_H
#define PTO_TASK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Memory Attributes
// =============================================================================

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// =============================================================================
// Constants
// =============================================================================

#define PTO_MAX_TASK_ARGS    16
#ifndef PTO_MAX_TASKS
#define PTO_MAX_TASKS        4096
#endif
#define PTO_MAX_WORKERS      128
#define PTO_FUNC_NAME_LEN    64

// =============================================================================
// Task Argument
// =============================================================================

typedef struct {
    uint64_t base_addr;
    uint64_t offset;
    uint64_t size;
} PTOTaskArg;

// =============================================================================
// Task Structure
// =============================================================================

/**
 * Task structure for NPU execution.
 * 
 * Key field: functionBinAddr - points to compiled kernel in GM.
 * AICore workers cast this to function pointer and invoke.
 */
typedef struct {
    int32_t task_id;
    int32_t func_id;
    char func_name[PTO_FUNC_NAME_LEN];
    
    // Function binary address for runtime dispatch
    uint64_t functionBinAddr;
    
    // Arguments
    int32_t num_args;
    PTOTaskArg args[PTO_MAX_TASK_ARGS];
    
    // Dependencies
    int32_t dep_count;
    int32_t deps_remaining;
    int32_t num_dependents;
    int32_t dependents[PTO_MAX_TASK_ARGS];
    
    // Execution state: 0=pending, 1=ready, 2=running, 3=complete
    int32_t status;
    int32_t core_type;  // 0=AIC (Cube), 1=AIV (Vector)
} PTOTask;

#define PTO_TASK_PENDING   0
#define PTO_TASK_READY     1
#define PTO_TASK_RUNNING   2
#define PTO_TASK_COMPLETE  3

// =============================================================================
// Handshake (AICore <-> AICPU Communication)
// =============================================================================

/**
 * Handshake buffer for AICore-AICPU communication.
 * 
 * Protocol:
 * 1. AICPU sets aicpu_ready = 1
 * 2. AICore polls until aicpu_ready != 0
 * 3. AICore sets aicore_done = core_id + 1
 * 4. Execution loop:
 *    - AICPU sets task pointer (task != 0)
 *    - AICore executes, sets task_status = 0
 * 5. AICPU sets control = 1 to shutdown
 */
typedef struct {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile uint32_t control;
    volatile uint64_t task;
    volatile uint32_t task_status;
    volatile uint32_t core_type;
    uint32_t padding[2];
} PTOHandshake;

// =============================================================================
// Task Graph (for batch execution mode)
// =============================================================================

/**
 * Task graph structure.
 * 
 * Note: This is used in batch mode where tasks are pre-built.
 * In streaming mode, orchestration generates tasks dynamically
 * and submits them to the scheduler one by one.
 */
typedef struct {
    int32_t num_tasks;
    int32_t tasks_completed;
    PTOTask tasks[PTO_MAX_TASKS];
} PTOTaskGraph;

// =============================================================================
// Kernel Arguments (Host -> AICPU)
// =============================================================================

typedef struct {
    int64_t* deviceArgs;
    PTOHandshake* hankArgs;
    PTOTaskGraph* graphArgs;  // Task graph (batch mode)
    int32_t core_num;
    int32_t aic_num;
    int32_t aiv_num;
} PTOKernelArgs;

#ifdef __cplusplus
}
#endif

#endif // PTO_TASK_H
