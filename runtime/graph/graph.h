/**
 * Graph Class - Task Dependency Graph Management
 *
 * This is a simplified, standalone graph class for managing task dependencies.
 * Tasks are stored in a fixed-size array with compile-time configurable bounds.
 * Each task has:
 * - Unique ID (array index)
 * - Arguments (uint64_t array)
 * - Fanin (predecessor count)
 * - Fanout (array of successor task IDs)
 *
 * The graph maintains a ready queue for tasks with fanin == 0.
 *
 * Based on patterns from pto_runtime.h/c but simplified for educational
 * and lightweight scheduling use cases.
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>  // for fprintf, printf
#include <string.h> // for memset

// =============================================================================
// Configuration Macros
// =============================================================================

#ifndef GRAPH_MAX_TASKS
#define GRAPH_MAX_TASKS 1024
#endif

#ifndef GRAPH_MAX_ARGS
#define GRAPH_MAX_ARGS 16
#endif

#ifndef GRAPH_MAX_FANOUT
#define GRAPH_MAX_FANOUT 512
#endif

// =============================================================================
// Data Structures
// =============================================================================

// =============================================================================
// Profiling (DFX) Structures
// =============================================================================

// Per-task profiling written by the executing AICore worker.
// Keep this a single cache line so the device can `dcci(..., SINGLE_CACHE_LINE)`
// after updating it.
typedef struct __attribute__((aligned(64))) {
  uint64_t start_time;   // raw device clock ticks at task start
  uint64_t end_time;     // raw device clock ticks at task end
  uint32_t pmu_cnt[8];   // optional PMU counters (best-effort; may be zero)
  uint32_t exec_core_id; // worker id (AIC: [0,nrAic), AIV: [nrAic, nrAic+nrAiv))
  uint32_t exec_core_type; // 1=AIC (cube), 2=AIV (vector)
  uint32_t exec_phys_core_id; // physical core id as reported by CCE
  uint32_t reserved;
} TaskProfile;
static_assert(sizeof(TaskProfile) == 64, "TaskProfile must be exactly one cache line (64B)");

/**
 * Task entry in the dependency graph
 *
 * Each task has a unique ID (its index in the task array), arguments,
 * and dependency information (fanin/fanout).
 */
typedef struct {
  int task_id;                   // Unique task identifier
  int func_id;                   // Function identifier
  // Core type routing for the in-core task:
  //   0 = any (scheduler may pick any worker), 1 = AIC (cube), 2 = AIV (vector).
  // This enables mixing cube-only kernels (e.g. matmul) and vector-only kernels in one run.
  int core_type;
  uint64_t args[GRAPH_MAX_ARGS]; // Task arguments
  int num_args;                  // Number of valid arguments

  // Runtime function pointer address (NEW)
  // This is the GM address where the kernel binary resides
  // It's cast to a function pointer at runtime: (KernelFunc)functionBinAddr
  uint64_t functionBinAddr;     // Address of kernel in device GM memory

 // Dependency tracking (using PTO runtime terminology)
  int fanin;                    // Number of predecessors (dependencies)
  int fanout[GRAPH_MAX_FANOUT]; // Successor task IDs
  int fanout_count;             // Number of successors

  // DFX/perf fields (written by AICore).
  TaskProfile profile;
} Task;

// =============================================================================
// Graph Class
// =============================================================================

/**
 * Graph class for task dependency management
 *
 * Maintains a fixed-size array of tasks and uses a Queue for ready tasks.
 * Tasks are allocated monotonically and never reused within the same
 * graph instance.
 *
 * Dependencies are managed manually via add_successor().
 */
class Graph {
private:
  // Task storage
  Task tasks[GRAPH_MAX_TASKS]; // Fixed-size task array
  int next_task_id;            // Next available task ID

  // Initial ready tasks (computed once, read-only after)
  int initial_ready_tasks[GRAPH_MAX_TASKS];
  int initial_ready_count;

public:
  /**
   * Constructor - zero-initialize all arrays
   */
  Graph();

  // =========================================================================
  // Task Management
  // =========================================================================

  /**
   * Allocate a new task with the given arguments
   *
   * @param args      Array of uint64_t arguments
   * @param num_args  Number of arguments (must be <= GRAPH_MAX_ARGS)
   * @return Task ID (>= 0) on success, -1 on failure
   */
  int add_task(uint64_t *args, int num_args, int func_id);
  int add_task(uint64_t *args, int num_args, int func_id, int core_type);

  /**
   * Add a dependency edge: from_task -> to_task
   *
   * This adds to_task to from_task's fanout array and increments
   * to_task's fanin counter.
   *
   * @param from_task  Producer task ID
   * @param to_task    Consumer task ID (depends on from_task)
   */
  void add_successor(int from_task, int to_task);

  // =========================================================================
  // Query Methods
  // =========================================================================

  /**
   * Get a pointer to a task by ID
   *
   * @param task_id  Task ID to query
   * @return Pointer to task, or nullptr if invalid ID
   */
  Task *get_task(int task_id);
  const Task *get_task(int task_id) const;

  /**
   * Get the total number of tasks in the graph
   *
   * @return Total task count
   */
  int get_task_count() const;

  /**
   * Get initially ready tasks (fanin == 0) as entry point for execution
   *
   * This scans all tasks and populates the provided array with task IDs
   * that have no dependencies (fanin == 0). The runtime can use this
   * as the starting point for task scheduling.
   *
   * @param ready_tasks  Array to populate with ready task IDs (can be nullptr)
   * @return Number of initially ready tasks
   */
  int get_initial_ready_tasks(int *ready_tasks);

  // =========================================================================
  // Utility Methods
  // =========================================================================

  /**
   * Print the graph structure to stdout
   *
   * Shows task table with fanin/fanout information.
   */
  void print_graph() const;
};

#endif // GRAPH_H
