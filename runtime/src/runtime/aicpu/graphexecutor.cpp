#include <cstdint>
#include "device_log.h"
#include "graph.h"

/**
 * Execute task graph using polling-based dispatch to AICore
 *
 * This function implements a dynamic task scheduler that:
 * 1. Maintains two separate ready queues - one for AIC tasks, one for AIV tasks
 * 2. Polls each AICore handshake buffer to check for idle cores
 * 3. Dispatches ready tasks to idle cores based on core type matching
 * 4. Tracks task completion and updates successor dependencies
 *
 * The scheduler supports arbitrary DAG topologies and automatically handles
 * parallelism across multiple cores based on data dependencies and core types.
 *
 * Algorithm:
 * - Separate initially ready tasks by core type into two queues
 * - Loop while there are tasks ready to run OR tasks currently executing
 * - For each core:
 *   - If task completed (idle + task != 0): update dependencies, add to appropriate queue
 *   - If core idle: dispatch from matching queue (AIC core -> AIC queue, AIV core -> AIV queue)
 *
 * @param g Task graph containing all tasks and dependencies
 * @param hank Array of handshake buffers (one per core)
 * @param core_num Number of AICore instances available
 * @return Number of tasks completed
 */
int execute(Graph& g, Handshake* hank, int core_num, int threadId) {
    if (threadId == 0) {
        DEV_INFO("Thread %d: Executing graph", threadId);
    } else {
        return 0;
    }
    // Separate ready queues for each core type
    int ready_queue_aic[GRAPH_MAX_TASKS];
    int ready_queue_aiv[GRAPH_MAX_TASKS];
    int ready_count_aic = 0;
    int ready_count_aiv = 0;

    // Get initially ready tasks and separate by core type
    int initial_ready[GRAPH_MAX_TASKS];
    int initial_count = g.get_initial_ready_tasks(initial_ready);

    DEV_INFO("Found %d initially ready tasks", initial_count);
    for (int i = 0; i < initial_count; i++) {
        Task* task = g.get_task(initial_ready[i]);
        if (task->core_type == 0) {  // AIC
            ready_queue_aic[ready_count_aic++] = initial_ready[i];
            DEV_INFO("  Task %d -> AIC queue", initial_ready[i]);
        } else {  // AIV
            ready_queue_aiv[ready_count_aiv++] = initial_ready[i];
            DEV_INFO("  Task %d -> AIV queue", initial_ready[i]);
        }
    }

    int completed = 0;
    int tasks_in_flight = 0;

    DEV_INFO("Starting execution loop: AIC ready=%d, AIV ready=%d", ready_count_aic, ready_count_aiv);

    // Execute tasks using polling-based dispatch
    // Loop until all tasks are dispatched and completed
    while (ready_count_aic > 0 || ready_count_aiv > 0 || tasks_in_flight > 0) {
        // Iterate through each core
        for (int core_id = 0; core_id < core_num; core_id++) {
            DEV_INFO("  Checking core %d (type=%d)", core_id, hank[core_id].core_type);
            Handshake* h = &hank[core_id];

            // Case 1: Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task
                Task* task = reinterpret_cast<Task*>(h->task);
                int task_id = task->task_id;

                DEV_INFO("  Core %d completed task %d", core_id, task_id);

                // Update fanin of successors and add to appropriate ready queue
                for (int i = 0; i < task->fanout_count; i++) {
                    int dep_id = task->fanout[i];
                    Task* dep = g.get_task(dep_id);
                    dep->fanin--;

                    // Add to ready queue if ready
                    if (dep->fanin == 0) {
                        if (dep->core_type == 0) {  // AIC
                            ready_queue_aic[ready_count_aic++] = dep_id;
                            DEV_INFO("    Task %d now ready -> AIC queue", dep_id);
                        } else {  // AIV
                            ready_queue_aiv[ready_count_aiv++] = dep_id;
                            DEV_INFO("    Task %d now ready -> AIV queue", dep_id);
                        }
                    }
                }

                // Clear task pointer
                h->task = 0;
                completed++;
                tasks_in_flight--;
            }

            // Case 2: Core is idle and available (idle + task is null)
            if (h->task_status == 0 && h->task == 0) {
                // Dispatch from matching queue based on core type
                if (h->core_type == 0 && ready_count_aic > 0) {  // AIC core
                    int task_id = ready_queue_aic[--ready_count_aic];
                    Task* task = g.get_task(task_id);

                    DEV_INFO("  Dispatching AIC task %d to core %d", task_id, core_id);

                    h->task = reinterpret_cast<uint64_t>(task);
                    h->task_status = 1;  // Mark as busy
                    tasks_in_flight++;
                } else if (h->core_type == 1 && ready_count_aiv > 0) {  // AIV core
                    int task_id = ready_queue_aiv[--ready_count_aiv];
                    Task* task = g.get_task(task_id);

                    DEV_INFO("  Dispatching AIV task %d to core %d", task_id, core_id);

                    h->task = reinterpret_cast<uint64_t>(task);
                    h->task_status = 1;  // Mark as busy
                    tasks_in_flight++;
                }
            }
        }
    }

    DEV_INFO("Execution complete: %d tasks completed", completed);
    return completed;
}
