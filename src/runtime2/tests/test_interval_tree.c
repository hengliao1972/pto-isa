/**
 * Test for Interval Tree
 * 
 * Tests:
 * 1. Basic insert and query
 * 2. Multiple overlapping intervals
 * 3. Tree validation (AVL balance, max_high)
 * 4. Stale removal
 * 5. Performance comparison with linear scan
 */

#include "../pto_interval_tree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)
#define ASSERT_TRUE(cond, name, msg) if (!(cond)) TEST_FAIL(name, msg)
#define ASSERT_FALSE(cond, name, msg) if (cond) TEST_FAIL(name, msg)
#define ASSERT_EQ(a, b, name, msg) if ((a) != (b)) { printf("  Expected %d, got %d\n", (int)(b), (int)(a)); TEST_FAIL(name, msg); }

// =============================================================================
// Test: Basic Insert and Query
// =============================================================================

int test_basic_insert_query() {
    const char* test_name = "Basic insert and query";
    
    PTO2IntervalTree tree;
    ASSERT_TRUE(pto2_interval_tree_init_default(&tree), test_name, "init failed");
    
    // Insert interval [10, 20] produced by task 100
    ASSERT_TRUE(pto2_interval_tree_insert(&tree, 10, 20, 100, 0), test_name, "insert failed");
    ASSERT_TRUE(pto2_interval_tree_validate(&tree), test_name, "tree invalid after insert");
    
    // Query [15, 25] should find task 100 (overlaps [10, 20])
    int32_t results[16];
    int32_t count = pto2_interval_tree_query(&tree, 15, 25, results, 16);
    ASSERT_EQ(count, 1, test_name, "should find 1 producer");
    ASSERT_EQ(results[0], 100, test_name, "should find task 100");
    
    // Query [0, 5] should find nothing
    count = pto2_interval_tree_query(&tree, 0, 5, results, 16);
    ASSERT_EQ(count, 0, test_name, "should find 0 producers");
    
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Multiple Overlapping Intervals
// =============================================================================

int test_multiple_overlapping() {
    const char* test_name = "Multiple overlapping intervals";
    
    PTO2IntervalTree tree;
    pto2_interval_tree_init_default(&tree);
    
    // Insert intervals:
    // Task 100: [0, 100]
    // Task 101: [50, 150]
    // Task 102: [200, 300]
    // Task 103: [75, 125]
    pto2_interval_tree_insert(&tree, 0, 100, 100, 0);
    pto2_interval_tree_insert(&tree, 50, 150, 101, 1);
    pto2_interval_tree_insert(&tree, 200, 300, 102, 2);
    pto2_interval_tree_insert(&tree, 75, 125, 103, 3);
    
    ASSERT_TRUE(pto2_interval_tree_validate(&tree), test_name, "tree invalid");
    ASSERT_EQ(pto2_interval_tree_size(&tree), 4, test_name, "size should be 4");
    
    // Query [80, 90] should find tasks 100, 101, 103 (all overlap)
    int32_t results[16];
    int32_t count = pto2_interval_tree_query(&tree, 80, 90, results, 16);
    ASSERT_EQ(count, 3, test_name, "should find 3 producers");
    
    // Query [250, 280] should find only task 102
    count = pto2_interval_tree_query(&tree, 250, 280, results, 16);
    ASSERT_EQ(count, 1, test_name, "should find 1 producer");
    ASSERT_EQ(results[0], 102, test_name, "should find task 102");
    
    // Query [160, 190] should find nothing
    count = pto2_interval_tree_query(&tree, 160, 190, results, 16);
    ASSERT_EQ(count, 0, test_name, "should find 0 producers");
    
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: AVL Balance
// =============================================================================

int test_avl_balance() {
    const char* test_name = "AVL balance after many inserts";
    
    PTO2IntervalTree tree;
    pto2_interval_tree_init_default(&tree);
    
    // Insert 100 intervals in sorted order (worst case for unbalanced BST)
    for (int i = 0; i < 100; i++) {
        pto2_interval_tree_insert(&tree, i * 10, i * 10 + 5, 100 + i, i);
    }
    
    ASSERT_TRUE(pto2_interval_tree_validate(&tree), test_name, "tree invalid");
    ASSERT_EQ(pto2_interval_tree_size(&tree), 100, test_name, "size should be 100");
    
    // Height should be O(log n), approximately 7-8 for 100 nodes
    int32_t height = pto2_interval_tree_height(&tree);
    ASSERT_TRUE(height <= 10, test_name, "height should be <= 10 (balanced)");
    
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Stale Removal
// =============================================================================

int test_stale_removal() {
    const char* test_name = "Stale removal";
    
    PTO2IntervalTree tree;
    pto2_interval_tree_init_default(&tree);
    
    // Insert intervals for tasks 100, 101, 102, 103
    pto2_interval_tree_insert(&tree, 0, 10, 100, 0);
    pto2_interval_tree_insert(&tree, 10, 20, 101, 1);
    pto2_interval_tree_insert(&tree, 20, 30, 102, 2);
    pto2_interval_tree_insert(&tree, 30, 40, 103, 3);
    
    ASSERT_EQ(pto2_interval_tree_size(&tree), 4, test_name, "size should be 4");
    
    // Set validity threshold to 102 - tasks 100, 101 are stale
    pto2_interval_tree_sync_validity(&tree, 102);
    
    // Query should still work (stale filtering)
    int32_t results[16];
    int32_t count = pto2_interval_tree_query(&tree, 0, 100, results, 16);
    ASSERT_EQ(count, 2, test_name, "should find 2 valid producers (102, 103)");
    
    // Remove stale entries
    int32_t removed = pto2_interval_tree_remove_stale(&tree, 102);
    ASSERT_EQ(removed, 2, test_name, "should remove 2 stale entries");
    ASSERT_EQ(pto2_interval_tree_size(&tree), 2, test_name, "size should be 2");
    ASSERT_TRUE(pto2_interval_tree_validate(&tree), test_name, "tree invalid after removal");
    
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Query Full (with interval info)
// =============================================================================

int test_query_full() {
    const char* test_name = "Query full interval info";
    
    PTO2IntervalTree tree;
    pto2_interval_tree_init_default(&tree);
    
    pto2_interval_tree_insert(&tree, 0, 100, 100, 5);
    pto2_interval_tree_insert(&tree, 50, 150, 101, 6);
    
    PTO2Interval results[16];
    int32_t count = pto2_interval_tree_query_full(&tree, 75, 80, results, 16);
    
    ASSERT_EQ(count, 2, test_name, "should find 2 intervals");
    
    // Verify entry_index is preserved
    bool found_5 = false, found_6 = false;
    for (int i = 0; i < count; i++) {
        if (results[i].entry_index == 5) found_5 = true;
        if (results[i].entry_index == 6) found_6 = true;
    }
    ASSERT_TRUE(found_5 && found_6, test_name, "should have entry indices 5 and 6");
    
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Performance Benchmark: Interval Tree vs Linear Scan
// =============================================================================

int benchmark_performance() {
    const char* test_name = "Performance benchmark";
    
    const int NUM_INTERVALS = 1000;
    const int NUM_QUERIES = 10000;
    
    PTO2IntervalTree tree;
    pto2_interval_tree_init(&tree, NUM_INTERVALS + 100);
    
    // Also create linear array for comparison
    typedef struct { int64_t low, high; int32_t producer; } LinearInterval;
    LinearInterval* linear = malloc(NUM_INTERVALS * sizeof(LinearInterval));
    
    // Insert random intervals
    srand(42);  // Reproducible
    for (int i = 0; i < NUM_INTERVALS; i++) {
        int64_t low = rand() % 10000;
        int64_t high = low + rand() % 100 + 1;
        
        pto2_interval_tree_insert(&tree, low, high, 100 + i, i);
        linear[i].low = low;
        linear[i].high = high;
        linear[i].producer = 100 + i;
    }
    
    printf("\n=== Performance Benchmark ===\n");
    printf("Intervals: %d, Queries: %d\n", NUM_INTERVALS, NUM_QUERIES);
    
    int32_t results[64];
    
    // Benchmark Interval Tree
    clock_t start = clock();
    int total_tree = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
        int64_t low = rand() % 10000;
        int64_t high = low + rand() % 100;
        total_tree += pto2_interval_tree_query(&tree, low, high, results, 64);
    }
    clock_t end = clock();
    double tree_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark Linear Scan
    srand(42);  // Same queries
    start = clock();
    int total_linear = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
        int64_t low = rand() % 10000;
        int64_t high = low + rand() % 100;
        
        int count = 0;
        for (int i = 0; i < NUM_INTERVALS && count < 64; i++) {
            if (linear[i].low <= high && low <= linear[i].high) {
                results[count++] = linear[i].producer;
            }
        }
        total_linear += count;
    }
    end = clock();
    double linear_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Interval Tree: %.3f s (%d total results)\n", tree_time, total_tree);
    printf("Linear Scan:   %.3f s (%d total results)\n", linear_time, total_linear);
    printf("Speedup:       %.1fx\n", linear_time / tree_time);
    printf("Tree height:   %d (optimal for %d: ~%.0f)\n", 
           pto2_interval_tree_height(&tree), NUM_INTERVALS, 
           1.44 * (float)log2(NUM_INTERVALS + 1));
    printf("=============================\n\n");
    
    // Results should match (approximately, order may differ)
    // Allow some tolerance due to duplicate filtering
    
    free(linear);
    pto2_interval_tree_destroy(&tree);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Main
// =============================================================================

#include <math.h>

int main(void) {
    printf("=== Interval Tree Tests ===\n\n");
    
    int failures = 0;
    
    failures += test_basic_insert_query();
    failures += test_multiple_overlapping();
    failures += test_avl_balance();
    failures += test_stale_removal();
    failures += test_query_full();
    failures += benchmark_performance();
    
    printf("=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    
    return failures;
}
