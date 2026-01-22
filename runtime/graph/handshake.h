/**
 * Handshake Structure - Shared between Host, AICPU, and AICore
 *
 * This structure facilitates communication and synchronization between
 * AICPU and AICore during task execution.
 *
 * Protocol State Machine:
 * 1. Initialization: AICPU sets aicpu_ready=1
 * 2. Acknowledgment: AICore sets aicore_done=core_id+1
 * 3. Task Dispatch: AICPU assigns task pointer and sets task_status=1
 * 4. Task Execution: AICore reads task, executes, sets task_status=0
 * 5. Task Completion: AICPU reads task_status=0, clears task=0
 * 6. Shutdown: AICPU sets control=1, AICore exits
 *
 * Each AICore instance has its own handshake buffer to enable concurrent
 * task execution across multiple cores.
 */

#ifndef RUNTIME_COMMON_HANDSHAKE_H
#define RUNTIME_COMMON_HANDSHAKE_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Field Access Patterns:
 * - aicpu_ready: Written by AICPU, read by AICore
 * - aicore_done: Written by AICore, read by AICPU
 * - task: Written by AICPU, read by AICore (0 = no task assigned)
 * - task_status: Written by both (AICPU=1 on dispatch, AICore=0 on completion)
 * - control: Written by AICPU, read by AICore (0 = continue, 1 = quit)
 */
struct Handshake {
    volatile uint32_t aicpu_ready;   // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;   // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;          // Task pointer: 0=no task, non-zero=Task* address
    volatile int32_t task_status;    // Task execution status: 0=idle, 1=busy
    volatile int32_t control;        // Control signal: 0=execute, 1=quit
} __attribute__((aligned(64)));

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_COMMON_HANDSHAKE_H
