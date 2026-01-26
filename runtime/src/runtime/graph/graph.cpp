/**
 * Graph Class - Implementation
 *
 * Task dependency graph management with circular ready queue.
 * Follows patterns from pto_runtime.c for consistency.
 */

#include "graph.h"

// =============================================================================
// Constructor
// =============================================================================

Graph::Graph() {
    // Zero-initialize all arrays
    memset(tasks, 0, sizeof(tasks));
    memset(workers, 0, sizeof(workers));
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
}

// =============================================================================
// Task Management
// =============================================================================

int Graph::add_task(uint64_t* args, int num_args, int func_id, int core_type) {
    // Check bounds
    if (next_task_id >= GRAPH_MAX_TASKS) {
        fprintf(stderr, "[Graph] ERROR: Task table full (max=%d)\n", GRAPH_MAX_TASKS);
        return -1;
    }

    if (num_args > GRAPH_MAX_ARGS) {
        fprintf(stderr, "[Graph] ERROR: Too many args (%d > %d)\n",
                num_args, GRAPH_MAX_ARGS);
        return -1;
    }

    // Allocate task
    int task_id = next_task_id++;
    Task* task = &tasks[task_id];

    // Initialize task fields
    task->task_id = task_id;
    task->func_id = func_id;
    task->num_args = num_args;
    if (args && num_args > 0) {
        memcpy(task->args, args, num_args * sizeof(uint64_t));
    }
    task->functionBinAddr = 0;  // Will be set by host before copying to device
    task->core_type = core_type;  // Set core type (0=AIC, 1=AIV)
    task->fanin = 0;
    task->fanout_count = 0;
    memset(task->fanout, 0, sizeof(task->fanout));

    return task_id;
}

void Graph::add_successor(int from_task, int to_task) {
    // Validate task IDs
    if (from_task < 0 || from_task >= next_task_id) {
        fprintf(stderr, "[Graph] ERROR: Invalid from_task ID %d\n", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        fprintf(stderr, "[Graph] ERROR: Invalid to_task ID %d\n", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    // Add to_task to from_task's fanout
    if (from->fanout_count >= GRAPH_MAX_FANOUT) {
        fprintf(stderr, "[Graph] ERROR: Fanout overflow for task %d (max=%d)\n",
                from_task, GRAPH_MAX_FANOUT);
        return;
    }

    from->fanout[from->fanout_count++] = to_task;
    to->fanin++;
}

// =============================================================================
// Query Methods
// =============================================================================

Task* Graph::get_task(int task_id) {
    if (task_id < 0 || task_id >= next_task_id) {
        return nullptr;
    }
    return &tasks[task_id];
}

int Graph::get_task_count() const {
    return next_task_id;
}

int Graph::get_initial_ready_tasks(int* ready_tasks) {
    initial_ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin == 0) {
            initial_ready_tasks[initial_ready_count] = i;
            if (ready_tasks != nullptr) {
                ready_tasks[initial_ready_count] = i;
            }
            initial_ready_count++;
        }
    }
    return initial_ready_count;
}

// =============================================================================
// Utility Methods
// =============================================================================

void Graph::print_graph() const {
    printf("\n================================================================================\n");
    printf("[Graph] Task Graph Status\n");
    printf("================================================================================\n");
    printf("  Total tasks: %d\n", next_task_id);

    // Print initially ready tasks
    printf("\nInitially Ready Tasks (fanin==0):\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  ");
    int ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin == 0) {
            if (ready_count > 0) printf(", ");
            printf("%d", i);
            ready_count++;
        }
    }
    if (ready_count == 0) {
        printf("(none)");
    }
    printf("\n  Count: %d\n", ready_count);

    printf("\nTask Table:\n");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];

        printf("  Task %d: func_id=%d, fanin=%d, fanout=%d, args=%d [",
               i, t->func_id, t->fanin, t->fanout_count, t->num_args);

        // Print fanout list
        for (int j = 0; j < t->fanout_count; j++) {
            printf("%d%s", t->fanout[j],
                   j < t->fanout_count - 1 ? "," : "");
        }
        printf("]\n");
    }

    printf("================================================================================\n\n");
}
