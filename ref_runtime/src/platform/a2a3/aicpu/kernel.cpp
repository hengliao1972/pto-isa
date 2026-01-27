#include <cstdint>
#include <cstdio>
#include "device_log.h"
#include "graph.h"
#include "kernel_args.h"

// AICPU (ARM) <-> AICore (GM) synchronization:
// - AICore side uses `dcci(...)` to maintain cache coherency for GM accesses.
// - AICPU side must clean+invalidate its D-cache lines when writing/reading
//   shared GM control blocks (handshake/task mailboxes).
// This follows the same pattern used in `~/pypto/framework/src/machine/utils/dynamic/cache.h`.
#ifdef __aarch64__
static inline void flush_dcache_range(void *start, uint64_t size) {
    constexpr int64_t kCacheLine = 64;
    uint64_t xstart = (uint64_t)start & ~(static_cast<uint64_t>(kCacheLine) - 1U);
    uint64_t xend = (uint64_t)start + size;
    while (xstart < xend) {
        asm volatile("dc civac, %0" : : "r"(xstart) : "memory");
        xstart += static_cast<uint64_t>(kCacheLine);
    }
    asm volatile("dsb sy" : : : "memory");
}
#else
static inline void flush_dcache_range(void *start, uint64_t size) {
    (void)start;
    (void)size;
}
#endif

/**
 * Handshake AICore - Initialize and synchronize with AICore kernels
 *
 * This function performs the initial handshake protocol with all AICore instances:
 * 1. Set aicpu_ready flag for each core
 * 2. Wait for each core to respond with aicore_done signal
 *
 * This ensures all cores are running and ready to receive tasks before
 * graph execution begins.
 *
 * @param arg Pointer to PtoRuntimeArgs mailbox
 * @return 0 on success
 */
int HankAiCore(void *arg) {
    auto rtargs = (PtoRuntimeArgs *)arg;
    if (rtargs == nullptr || rtargs->hankArgs == nullptr) {
        DEV_ERROR("%s", "HankAiCore: invalid runtime args");
        return -1;
    }
    uint64_t core_num = static_cast<uint64_t>(rtargs->core_num);

    // Phase 1: Signal all cores that AICPU is ready
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = reinterpret_cast<Handshake*>(rtargs->hankArgs) + i;
        DEV_INFO("AICPU: hank addr = 0x%lx", (uint64_t)hank);
        hank->aicpu_ready = 1;
        flush_dcache_range(reinterpret_cast<void *>(hank), sizeof(Handshake));
    }

    // Phase 2: Wait for all cores to acknowledge (busy-wait polling)
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = reinterpret_cast<Handshake*>(rtargs->hankArgs) + i;
        // Busy-wait until AICore signals ready (aicore_done != 0)
        while (hank->aicore_done == 0) {
            flush_dcache_range(reinterpret_cast<void *>(hank), sizeof(Handshake));
            // Polling loop - no sleep to minimize latency
        };
        DEV_INFO("success hank->aicore_done = %u", (uint64_t)hank->aicore_done);
    }
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 *
 * Sets the control flag to 1 for all cores, signaling them to exit
 * their execution loops and terminate gracefully.
 *
 * @param arg Pointer to PtoRuntimeArgs mailbox
 * @return 0 on success
 */
int ShutdownAiCore(void *arg) {
    auto rtargs = (PtoRuntimeArgs *)arg;
    if (rtargs == nullptr || rtargs->hankArgs == nullptr) {
        DEV_ERROR("%s", "ShutdownAiCore: invalid runtime args");
        return -1;
    }
    uint64_t core_num = static_cast<uint64_t>(rtargs->core_num);
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = reinterpret_cast<Handshake*>(rtargs->hankArgs) + i;
        hank->control = 1;  // Set quit signal
        flush_dcache_range(reinterpret_cast<void *>(hank), sizeof(Handshake));
    }
    return 0;
}

/**
 * Execute task graph using polling-based dispatch to AICore
 *
 * This function implements a dynamic task scheduler that:
 * 1. Maintains a ready queue of tasks with all dependencies satisfied (fanin == 0)
 * 2. Polls each AICore handshake buffer to check for idle cores
 * 3. Dispatches ready tasks to idle cores
 * 4. Tracks task completion and updates successor dependencies
 *
 * The scheduler supports arbitrary DAG topologies and automatically handles
 * parallelism across multiple cores based on data dependencies.
 *
 * Algorithm:
 * - Loop while there are tasks ready to run OR tasks currently executing
 * - For each core:
 *   - If task completed (idle + task != 0): update dependencies, clear task
 *   - If core idle and tasks available: dispatch next ready task
 *
 * @param g Task graph containing all tasks and dependencies
 * @param hank Array of handshake buffers (one per core)
 * @param core_num Number of AICore instances available
 * @return Number of tasks completed
 */
int execute_graph(Graph& g, Handshake* hank, int core_num, int nr_aic) {
    if (nr_aic < 0) {
        nr_aic = 0;
    }
    if (nr_aic > core_num) {
        nr_aic = core_num;
    }
    // Get initially ready tasks from graph
    int ready_queue[GRAPH_MAX_TASKS];
    int ready_count = g.get_initial_ready_tasks(ready_queue);

    int completed = 0;
    int tasks_in_flight = 0;

    // Execute tasks using polling-based dispatch
    // Loop until all tasks are dispatched and completed
    while (ready_count > 0 || tasks_in_flight > 0) {
        // Iterate through each core
        for (int core_id = 0; core_id < core_num; core_id++) {
            DEV_INFO("  Checking core %d", core_id);
            Handshake* h = &hank[core_id];
            flush_dcache_range(reinterpret_cast<void *>(h), sizeof(Handshake));

            // Case 1: Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task
                Task* task = reinterpret_cast<Task*>(h->task);
                int task_id = task->task_id;

                DEV_INFO("  Core %d completed task %d", core_id, task_id);

                // Update fanin of successors
                for (int i = 0; i < task->fanout_count; i++) {
                    int dep_id = task->fanout[i];
                    Task* dep = g.get_task(dep_id);
                    dep->fanin--;

                    // Add to ready queue if ready
                    if (dep->fanin == 0) {
                        ready_queue[ready_count++] = dep_id;
                    }
                }

                // Clear task pointer
                h->task = 0;
                flush_dcache_range(reinterpret_cast<void *>(h), sizeof(Handshake));
                completed++;
                tasks_in_flight--;
            }

            // Case 2: Core is idle and available (idle + task is null)
            if (h->task_status == 0 && h->task == 0 && ready_count > 0) {
                const bool is_aic = (core_id < nr_aic);

                // Pick a task that matches this worker's core type.
                int pick_pos = -1;
                for (int i = 0; i < ready_count; i++) {
                    int cand_id = ready_queue[i];
                    Task* cand = g.get_task(cand_id);
                    if (cand == nullptr) {
                        continue;
                    }
                    // core_type: 0=any, 1=AIC(cube), 2=AIV(vector)
                    if (cand->core_type == 0 ||
                        (cand->core_type == 1 && is_aic) ||
                        (cand->core_type == 2 && !is_aic)) {
                        pick_pos = i;
                        break;
                    }
                }

                if (pick_pos < 0) {
                    continue;
                }

                // Pop task from ready queue (swap-remove).
                int task_id = ready_queue[pick_pos];
                ready_queue[pick_pos] = ready_queue[--ready_count];
                Task* task = g.get_task(task_id);

                DEV_INFO("  Dispatching task %d to core %d", task_id, core_id);

                // Assign task to core
                h->task = reinterpret_cast<uint64_t>(task);
                h->task_status = 1;  // Mark as busy
                flush_dcache_range(reinterpret_cast<void *>(h), sizeof(Handshake));
                tasks_in_flight++;
            }
        }
    }

    return completed;
}
extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

/**
 * AICPU kernel initialization entry point
 *
 * This function is called once during kernel initialization by the CANN runtime.
 * It initializes logging and validates kernel arguments.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to DeviceKernelArgs (TileFwk ABI)
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    InitLogSwitch();
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "Graph Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point
 *
 * This is the main entry point for the AICPU graph executor kernel.
 * It orchestrates the complete task graph execution:
 * 1. Handshake with all AICore instances
 * 2. Execute task graph using polling-based dispatch
 * 3. Shutdown all AICore instances
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to DeviceKernelArgs (TileFwk ABI). This function reads:
 *            - cfgdata (DeviceArgs) to find:
 *              - opaque: PtoRuntimeArgs mailbox pointer
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    DEV_INFO("%s", "Graph Executor: Starting AICPU kernel execution");

    auto kargs = reinterpret_cast<DeviceKernelArgs *>(arg);
    if (kargs == nullptr || kargs->cfgdata == nullptr) {
        DEV_ERROR("%s", "Invalid DeviceKernelArgs (missing cfgdata)");
        return -1;
    }
    auto devArgs = reinterpret_cast<DeviceArgs *>(kargs->cfgdata);
    if (devArgs == nullptr || devArgs->opaque == 0) {
        DEV_ERROR("%s", "Invalid DeviceArgs (missing opaque runtime mailbox)");
        return -1;
    }
    auto rtargs = reinterpret_cast<PtoRuntimeArgs *>(devArgs->opaque);
    if (rtargs == nullptr || rtargs->hankArgs == nullptr) {
        DEV_ERROR("%s", "Invalid runtime mailbox (missing hankArgs)");
        return -1;
    }

    // Step 1: Handshake with all AICore instances
    auto rc = HankAiCore(rtargs);
    if (rc != 0) {
        return rc;
    }

    // Step 2: Execute task graph if provided
    if (rtargs->graphArgs != nullptr) {
        Graph* g = rtargs->graphArgs;
        Handshake* hank = reinterpret_cast<Handshake*>(rtargs->hankArgs);
        int core_num = static_cast<int>(rtargs->core_num);
        int nr_aic = static_cast<int>(devArgs->nrAic);
        DEV_INFO("Graph has %d tasks", g->get_task_count());
        int completed = execute_graph(*g, hank, core_num, nr_aic);
        DEV_INFO("Executed %d tasks from graph", completed);
    }

    // Step 3: Shutdown all AICore instances
    rc = ShutdownAiCore(rtargs);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("%s", "Graph Executor: Kernel execution completed successfully");
    return 0;
}
