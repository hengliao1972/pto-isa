/**
 * PTO Runtime2 - Interval Tree Implementation
 * 
 * AVL-balanced interval tree for O(log n + k) overlap queries.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_interval_tree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// Internal Helpers
// =============================================================================

static inline int32_t max_i32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static inline int64_t max_i64(int64_t a, int64_t b) {
    return a > b ? a : b;
}

static inline int32_t node_height(PTO2IntervalTree* tree, int32_t idx) {
    if (idx == PTO2_INTERVAL_TREE_NIL) return 0;
    return tree->nodes[idx].height;
}

static inline int32_t balance_factor(PTO2IntervalTree* tree, int32_t idx) {
    if (idx == PTO2_INTERVAL_TREE_NIL) return 0;
    return node_height(tree, tree->nodes[idx].left) - 
           node_height(tree, tree->nodes[idx].right);
}

static void update_height(PTO2IntervalTree* tree, int32_t idx) {
    if (idx == PTO2_INTERVAL_TREE_NIL) return;
    tree->nodes[idx].height = 1 + max_i32(
        node_height(tree, tree->nodes[idx].left),
        node_height(tree, tree->nodes[idx].right)
    );
}

static void update_max_high(PTO2IntervalTree* tree, int32_t idx) {
    if (idx == PTO2_INTERVAL_TREE_NIL) return;
    
    PTO2IntervalNode* node = &tree->nodes[idx];
    int64_t max_val = node->interval.high;
    
    if (node->left != PTO2_INTERVAL_TREE_NIL) {
        max_val = max_i64(max_val, tree->nodes[node->left].max_high);
    }
    if (node->right != PTO2_INTERVAL_TREE_NIL) {
        max_val = max_i64(max_val, tree->nodes[node->right].max_high);
    }
    
    node->max_high = max_val;
}

// Allocate a node from pool
static int32_t alloc_node(PTO2IntervalTree* tree) {
    if (tree->free_head != PTO2_INTERVAL_TREE_NIL) {
        int32_t idx = tree->free_head;
        tree->free_head = tree->nodes[idx].right;  // Use right as next free pointer
        tree->nodes[idx].in_use = true;
        return idx;
    }
    
    // Find first unused node
    for (int32_t i = 0; i < tree->capacity; i++) {
        if (!tree->nodes[i].in_use) {
            tree->nodes[i].in_use = true;
            return i;
        }
    }
    
    return PTO2_INTERVAL_TREE_NIL;  // Pool full
}

// Free a node back to pool
static void free_node(PTO2IntervalTree* tree, int32_t idx) {
    if (idx == PTO2_INTERVAL_TREE_NIL) return;
    tree->nodes[idx].in_use = false;
    tree->nodes[idx].right = tree->free_head;
    tree->free_head = idx;
}

// =============================================================================
// AVL Rotations
// =============================================================================

static int32_t rotate_right(PTO2IntervalTree* tree, int32_t y) {
    int32_t x = tree->nodes[y].left;
    int32_t T2 = tree->nodes[x].right;
    
    // Perform rotation
    tree->nodes[x].right = y;
    tree->nodes[y].left = T2;
    
    // Update parents
    tree->nodes[x].parent = tree->nodes[y].parent;
    tree->nodes[y].parent = x;
    if (T2 != PTO2_INTERVAL_TREE_NIL) {
        tree->nodes[T2].parent = y;
    }
    
    // Update heights and max_high (order matters: y first, then x)
    update_height(tree, y);
    update_max_high(tree, y);
    update_height(tree, x);
    update_max_high(tree, x);
    
    return x;  // New root
}

static int32_t rotate_left(PTO2IntervalTree* tree, int32_t x) {
    int32_t y = tree->nodes[x].right;
    int32_t T2 = tree->nodes[y].left;
    
    // Perform rotation
    tree->nodes[y].left = x;
    tree->nodes[x].right = T2;
    
    // Update parents
    tree->nodes[y].parent = tree->nodes[x].parent;
    tree->nodes[x].parent = y;
    if (T2 != PTO2_INTERVAL_TREE_NIL) {
        tree->nodes[T2].parent = x;
    }
    
    // Update heights and max_high
    update_height(tree, x);
    update_max_high(tree, x);
    update_height(tree, y);
    update_max_high(tree, y);
    
    return y;  // New root
}

// =============================================================================
// Initialization and Destruction
// =============================================================================

bool pto2_interval_tree_init(PTO2IntervalTree* tree, int32_t capacity) {
    tree->nodes = (PTO2IntervalNode*)calloc(capacity, sizeof(PTO2IntervalNode));
    if (!tree->nodes) {
        return false;
    }
    
    tree->capacity = capacity;
    tree->root = PTO2_INTERVAL_TREE_NIL;
    tree->size = 0;
    tree->free_head = PTO2_INTERVAL_TREE_NIL;
    tree->last_task_alive = 0;
    
    // Initialize all nodes as not in use
    for (int32_t i = 0; i < capacity; i++) {
        tree->nodes[i].in_use = false;
        tree->nodes[i].left = PTO2_INTERVAL_TREE_NIL;
        tree->nodes[i].right = PTO2_INTERVAL_TREE_NIL;
        tree->nodes[i].parent = PTO2_INTERVAL_TREE_NIL;
    }
    
    return true;
}

bool pto2_interval_tree_init_default(PTO2IntervalTree* tree) {
    return pto2_interval_tree_init(tree, PTO2_INTERVAL_TREE_MAX_NODES);
}

void pto2_interval_tree_destroy(PTO2IntervalTree* tree) {
    if (tree->nodes) {
        free(tree->nodes);
        tree->nodes = NULL;
    }
    tree->capacity = 0;
    tree->root = PTO2_INTERVAL_TREE_NIL;
    tree->size = 0;
}

void pto2_interval_tree_clear(PTO2IntervalTree* tree) {
    tree->root = PTO2_INTERVAL_TREE_NIL;
    tree->size = 0;
    tree->free_head = PTO2_INTERVAL_TREE_NIL;
    
    for (int32_t i = 0; i < tree->capacity; i++) {
        tree->nodes[i].in_use = false;
        tree->nodes[i].left = PTO2_INTERVAL_TREE_NIL;
        tree->nodes[i].right = PTO2_INTERVAL_TREE_NIL;
        tree->nodes[i].parent = PTO2_INTERVAL_TREE_NIL;
    }
}

// =============================================================================
// Insert
// =============================================================================

static int32_t insert_recursive(PTO2IntervalTree* tree, int32_t node, int32_t new_node) {
    if (node == PTO2_INTERVAL_TREE_NIL) {
        return new_node;
    }
    
    PTO2IntervalNode* curr = &tree->nodes[node];
    PTO2IntervalNode* new_n = &tree->nodes[new_node];
    
    // BST insert (key = interval.low)
    if (new_n->interval.low < curr->interval.low) {
        curr->left = insert_recursive(tree, curr->left, new_node);
        tree->nodes[curr->left].parent = node;
    } else {
        curr->right = insert_recursive(tree, curr->right, new_node);
        tree->nodes[curr->right].parent = node;
    }
    
    // Update height and max_high
    update_height(tree, node);
    update_max_high(tree, node);
    
    // Balance the tree
    int32_t balance = balance_factor(tree, node);
    
    // Left Left Case
    if (balance > 1 && new_n->interval.low < tree->nodes[curr->left].interval.low) {
        return rotate_right(tree, node);
    }
    
    // Right Right Case
    if (balance < -1 && new_n->interval.low >= tree->nodes[curr->right].interval.low) {
        return rotate_left(tree, node);
    }
    
    // Left Right Case
    if (balance > 1 && new_n->interval.low >= tree->nodes[curr->left].interval.low) {
        curr->left = rotate_left(tree, curr->left);
        return rotate_right(tree, node);
    }
    
    // Right Left Case
    if (balance < -1 && new_n->interval.low < tree->nodes[curr->right].interval.low) {
        curr->right = rotate_right(tree, curr->right);
        return rotate_left(tree, node);
    }
    
    return node;
}

bool pto2_interval_tree_insert(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    int32_t producer_task_id,
    int32_t entry_index
) {
    int32_t new_node = alloc_node(tree);
    if (new_node == PTO2_INTERVAL_TREE_NIL) {
        return false;  // Pool full
    }
    
    // Initialize new node
    PTO2IntervalNode* node = &tree->nodes[new_node];
    node->interval.low = low;
    node->interval.high = high;
    node->interval.producer_task_id = producer_task_id;
    node->interval.entry_index = entry_index;
    node->max_high = high;
    node->left = PTO2_INTERVAL_TREE_NIL;
    node->right = PTO2_INTERVAL_TREE_NIL;
    node->parent = PTO2_INTERVAL_TREE_NIL;
    node->height = 1;
    
    // Insert into tree
    tree->root = insert_recursive(tree, tree->root, new_node);
    tree->size++;
    
    return true;
}

// =============================================================================
// Query
// =============================================================================

static bool intervals_overlap(int64_t low1, int64_t high1, int64_t low2, int64_t high2) {
    return (low1 <= high2) && (low2 <= high1);
}

static void query_recursive(
    PTO2IntervalTree* tree,
    int32_t node,
    int64_t low,
    int64_t high,
    int32_t* results,
    int32_t* count,
    int32_t max_results,
    int32_t last_task_alive
) {
    if (node == PTO2_INTERVAL_TREE_NIL || *count >= max_results) {
        return;
    }
    
    PTO2IntervalNode* n = &tree->nodes[node];
    
    // If max_high of this subtree is less than query.low, no overlap possible
    if (n->max_high < low) {
        return;
    }
    
    // Search left subtree
    query_recursive(tree, n->left, low, high, results, count, max_results, last_task_alive);
    
    // Check current node (skip stale entries)
    if (*count < max_results && 
        n->interval.producer_task_id >= last_task_alive &&
        intervals_overlap(n->interval.low, n->interval.high, low, high)) {
        
        // Check for duplicates (same producer)
        bool duplicate = false;
        for (int32_t i = 0; i < *count; i++) {
            if (results[i] == n->interval.producer_task_id) {
                duplicate = true;
                break;
            }
        }
        
        if (!duplicate) {
            results[(*count)++] = n->interval.producer_task_id;
        }
    }
    
    // Search right subtree only if query.low <= node.interval.low
    // (optimization: if query starts after this node, might still overlap right subtree)
    if (n->interval.low <= high) {
        query_recursive(tree, n->right, low, high, results, count, max_results, last_task_alive);
    }
}

int32_t pto2_interval_tree_query(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    int32_t* results,
    int32_t max_results
) {
    int32_t count = 0;
    query_recursive(tree, tree->root, low, high, results, &count, max_results, tree->last_task_alive);
    return count;
}

static void query_full_recursive(
    PTO2IntervalTree* tree,
    int32_t node,
    int64_t low,
    int64_t high,
    PTO2Interval* results,
    int32_t* count,
    int32_t max_results,
    int32_t last_task_alive
) {
    if (node == PTO2_INTERVAL_TREE_NIL || *count >= max_results) {
        return;
    }
    
    PTO2IntervalNode* n = &tree->nodes[node];
    
    if (n->max_high < low) {
        return;
    }
    
    query_full_recursive(tree, n->left, low, high, results, count, max_results, last_task_alive);
    
    if (*count < max_results && 
        n->interval.producer_task_id >= last_task_alive &&
        intervals_overlap(n->interval.low, n->interval.high, low, high)) {
        results[(*count)++] = n->interval;
    }
    
    if (n->interval.low <= high) {
        query_full_recursive(tree, n->right, low, high, results, count, max_results, last_task_alive);
    }
}

int32_t pto2_interval_tree_query_full(
    PTO2IntervalTree* tree,
    int64_t low,
    int64_t high,
    PTO2Interval* results,
    int32_t max_results
) {
    int32_t count = 0;
    query_full_recursive(tree, tree->root, low, high, results, &count, max_results, tree->last_task_alive);
    return count;
}

// =============================================================================
// Remove Stale
// =============================================================================

// Find minimum node in subtree
static int32_t find_min(PTO2IntervalTree* tree, int32_t node) {
    while (tree->nodes[node].left != PTO2_INTERVAL_TREE_NIL) {
        node = tree->nodes[node].left;
    }
    return node;
}

// Delete a node and rebalance
static int32_t delete_node(PTO2IntervalTree* tree, int32_t root, int32_t target) {
    if (root == PTO2_INTERVAL_TREE_NIL) {
        return root;
    }
    
    PTO2IntervalNode* curr = &tree->nodes[root];
    PTO2IntervalNode* tgt = &tree->nodes[target];
    
    // Find the node
    if (tgt->interval.low < curr->interval.low) {
        curr->left = delete_node(tree, curr->left, target);
    } else if (tgt->interval.low > curr->interval.low) {
        curr->right = delete_node(tree, curr->right, target);
    } else if (root != target) {
        // Same key but different node, could be in either subtree
        curr->right = delete_node(tree, curr->right, target);
    } else {
        // Found the node to delete
        if (curr->left == PTO2_INTERVAL_TREE_NIL || curr->right == PTO2_INTERVAL_TREE_NIL) {
            int32_t temp = (curr->left != PTO2_INTERVAL_TREE_NIL) ? curr->left : curr->right;
            
            if (temp == PTO2_INTERVAL_TREE_NIL) {
                // No children
                free_node(tree, root);
                return PTO2_INTERVAL_TREE_NIL;
            } else {
                // One child
                tree->nodes[temp].parent = curr->parent;
                free_node(tree, root);
                return temp;
            }
        } else {
            // Two children: get inorder successor
            int32_t succ = find_min(tree, curr->right);
            curr->interval = tree->nodes[succ].interval;
            curr->right = delete_node(tree, curr->right, succ);
        }
    }
    
    // Update height and max_high
    update_height(tree, root);
    update_max_high(tree, root);
    
    // Rebalance
    int32_t balance = balance_factor(tree, root);
    
    if (balance > 1 && balance_factor(tree, curr->left) >= 0) {
        return rotate_right(tree, root);
    }
    
    if (balance > 1 && balance_factor(tree, curr->left) < 0) {
        curr->left = rotate_left(tree, curr->left);
        return rotate_right(tree, root);
    }
    
    if (balance < -1 && balance_factor(tree, curr->right) <= 0) {
        return rotate_left(tree, root);
    }
    
    if (balance < -1 && balance_factor(tree, curr->right) > 0) {
        curr->right = rotate_right(tree, curr->right);
        return rotate_left(tree, root);
    }
    
    return root;
}

// Collect stale nodes
static void collect_stale(
    PTO2IntervalTree* tree,
    int32_t node,
    int32_t threshold,
    int32_t* stale_nodes,
    int32_t* count,
    int32_t max_count
) {
    if (node == PTO2_INTERVAL_TREE_NIL || *count >= max_count) {
        return;
    }
    
    PTO2IntervalNode* n = &tree->nodes[node];
    
    collect_stale(tree, n->left, threshold, stale_nodes, count, max_count);
    
    if (n->interval.producer_task_id < threshold && *count < max_count) {
        stale_nodes[(*count)++] = node;
    }
    
    collect_stale(tree, n->right, threshold, stale_nodes, count, max_count);
}

int32_t pto2_interval_tree_remove_stale(
    PTO2IntervalTree* tree,
    int32_t task_threshold
) {
    // Collect all stale nodes first
    int32_t stale_nodes[1024];
    int32_t stale_count = 0;
    
    collect_stale(tree, tree->root, task_threshold, stale_nodes, &stale_count, 1024);
    
    // Delete them one by one
    for (int32_t i = 0; i < stale_count; i++) {
        if (tree->nodes[stale_nodes[i]].in_use) {
            tree->root = delete_node(tree, tree->root, stale_nodes[i]);
            tree->size--;
        }
    }
    
    return stale_count;
}

void pto2_interval_tree_sync_validity(PTO2IntervalTree* tree, int32_t last_task_alive) {
    tree->last_task_alive = last_task_alive;
}

// =============================================================================
// Statistics and Debug
// =============================================================================

int32_t pto2_interval_tree_size(PTO2IntervalTree* tree) {
    return tree->size;
}

int32_t pto2_interval_tree_height(PTO2IntervalTree* tree) {
    return node_height(tree, tree->root);
}

void pto2_interval_tree_print_stats(PTO2IntervalTree* tree) {
    printf("=== Interval Tree Statistics ===\n");
    printf("Capacity:        %d\n", tree->capacity);
    printf("Size:            %d\n", tree->size);
    printf("Height:          %d\n", pto2_interval_tree_height(tree));
    printf("Last task alive: %d\n", tree->last_task_alive);
    printf("================================\n");
}

static bool validate_recursive(PTO2IntervalTree* tree, int32_t node, int64_t* out_max) {
    if (node == PTO2_INTERVAL_TREE_NIL) {
        *out_max = INT64_MIN;
        return true;
    }
    
    PTO2IntervalNode* n = &tree->nodes[node];
    
    // Check AVL balance
    int32_t balance = balance_factor(tree, node);
    if (balance < -1 || balance > 1) {
        printf("AVL violation at node %d: balance = %d\n", node, balance);
        return false;
    }
    
    // Check BST property
    if (n->left != PTO2_INTERVAL_TREE_NIL && 
        tree->nodes[n->left].interval.low > n->interval.low) {
        printf("BST violation: left child > parent\n");
        return false;
    }
    if (n->right != PTO2_INTERVAL_TREE_NIL && 
        tree->nodes[n->right].interval.low < n->interval.low) {
        printf("BST violation: right child < parent\n");
        return false;
    }
    
    // Check max_high augmentation
    int64_t left_max, right_max;
    if (!validate_recursive(tree, n->left, &left_max)) return false;
    if (!validate_recursive(tree, n->right, &right_max)) return false;
    
    int64_t expected_max = n->interval.high;
    if (left_max > expected_max) expected_max = left_max;
    if (right_max > expected_max) expected_max = right_max;
    
    if (n->max_high != expected_max) {
        printf("max_high violation at node %d: expected %ld, got %ld\n", 
               node, (long)expected_max, (long)n->max_high);
        return false;
    }
    
    *out_max = n->max_high;
    return true;
}

bool pto2_interval_tree_validate(PTO2IntervalTree* tree) {
    int64_t dummy;
    return validate_recursive(tree, tree->root, &dummy);
}
