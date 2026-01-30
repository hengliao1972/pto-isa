/**
 * PTO Runtime2 - Interval Tree for Fast Overlap Queries
 * 
 * Provides O(log n + k) overlap queries where k is the number of results.
 * Used to optimize TensorMap lookups when there are many views of the same
 * raw tensor storage.
 * 
 * Key operations:
 * - Insert interval: O(log n)
 * - Query overlapping intervals: O(log n + k)
 * - Delete interval: O(log n)
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_INTERVAL_TREE_H
#define PTO_INTERVAL_TREE_H

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// Configuration
// =============================================================================

#define PTO2_INTERVAL_TREE_MAX_NODES  4096   // Maximum nodes in tree
#define PTO2_INTERVAL_TREE_NIL        (-1)   // Null node index

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Interval with associated data
 */
typedef struct {
    int64_t low;              // Interval start (inclusive)
    int64_t high;             // Interval end (inclusive)
    int32_t producer_task_id; // Associated producer task ID
    int32_t entry_index;      // Index into TensorMapEx entry pool
} PTO2Interval;

/**
 * Interval tree node (augmented BST node)
 */
typedef struct {
    PTO2Interval interval;    // Interval stored at this node
    int64_t max_high;         // Maximum high value in subtree (augmented data)
    int32_t left;             // Left child index (-1 = NIL)
    int32_t right;            // Right child index (-1 = NIL)
    int32_t parent;           // Parent index (-1 = root)
    int32_t height;           // Height for AVL balancing
    bool    in_use;           // Node is in use
} PTO2IntervalNode;

/**
 * Interval tree structure
 * 
 * Uses AVL balancing for guaranteed O(log n) operations.
 * Nodes are stored in a pool for memory efficiency.
 */
typedef struct {
    PTO2IntervalNode* nodes;  // Node pool
    int32_t capacity;         // Pool capacity
    int32_t root;             // Root node index (-1 = empty)
    int32_t size;             // Number of intervals
    int32_t free_head;        // Head of free list (-1 = none)
    
    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;
} PTO2IntervalTree;

// =============================================================================
// Initialization and Destruction
// =============================================================================

/**
 * Initialize interval tree
 * 
 * @param tree     Tree to initialize
 * @param capacity Maximum number of intervals
 * @return true on success
 */
bool pto2_interval_tree_init(PTO2IntervalTree* tree, int32_t capacity);

/**
 * Initialize with default capacity
 */
bool pto2_interval_tree_init_default(PTO2IntervalTree* tree);

/**
 * Destroy interval tree and free resources
 */
void pto2_interval_tree_destroy(PTO2IntervalTree* tree);

/**
 * Clear all intervals from tree
 */
void pto2_interval_tree_clear(PTO2IntervalTree* tree);

// =============================================================================
// Core Operations
// =============================================================================

/**
 * Insert an interval into the tree
 * 
 * @param tree             Interval tree
 * @param low              Interval start
 * @param high             Interval end
 * @param producer_task_id Associated producer task ID
 * @param entry_index      Index into TensorMapEx entry pool
 * @return true on success, false if tree is full
 */
bool pto2_interval_tree_insert(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    int32_t producer_task_id,
    int32_t entry_index
);

/**
 * Query all intervals that overlap with [low, high]
 * 
 * @param tree         Interval tree
 * @param low          Query interval start
 * @param high         Query interval end
 * @param results      Output array of producer task IDs
 * @param max_results  Maximum number of results
 * @return Number of overlapping intervals found
 */
int32_t pto2_interval_tree_query(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    int32_t* results,
    int32_t max_results
);

/**
 * Query all intervals that overlap, returning full interval info
 * 
 * @param tree         Interval tree
 * @param low          Query interval start
 * @param high         Query interval end
 * @param results      Output array of intervals
 * @param max_results  Maximum number of results
 * @return Number of overlapping intervals found
 */
int32_t pto2_interval_tree_query_full(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    PTO2Interval* results,
    int32_t max_results
);

/**
 * Remove all intervals with producer_task_id < threshold
 * 
 * Used for lazy invalidation when tasks retire.
 * 
 * @param tree            Interval tree
 * @param task_threshold  Tasks with ID < threshold are removed
 * @return Number of intervals removed
 */
int32_t pto2_interval_tree_remove_stale(
    PTO2IntervalTree* tree,
    int32_t task_threshold
);

/**
 * Update validity threshold
 */
void pto2_interval_tree_sync_validity(PTO2IntervalTree* tree, int32_t last_task_alive);

// =============================================================================
// Statistics and Debug
// =============================================================================

/**
 * Get number of intervals in tree
 */
int32_t pto2_interval_tree_size(PTO2IntervalTree* tree);

/**
 * Get tree height (for balance checking)
 */
int32_t pto2_interval_tree_height(PTO2IntervalTree* tree);

/**
 * Print tree statistics
 */
void pto2_interval_tree_print_stats(PTO2IntervalTree* tree);

/**
 * Validate tree structure (for testing)
 * Returns true if tree is valid AVL tree with correct augmentation.
 */
bool pto2_interval_tree_validate(PTO2IntervalTree* tree);

#endif // PTO_INTERVAL_TREE_H
