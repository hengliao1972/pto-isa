/**
 * Test for Extended TensorMap with LogicalTensor support
 * 
 * Tests:
 * 1. Insert and lookup with LogicalTensor
 * 2. Overlap detection across views of same tensor
 * 3. lookup_all finding multiple overlapping producers
 * 4. Deep copy independence
 */

#include "../pto_tensormap.h"
#include "../pto_logical_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)
#define ASSERT_TRUE(cond, name, msg) if (!(cond)) TEST_FAIL(name, msg)
#define ASSERT_FALSE(cond, name, msg) if (cond) TEST_FAIL(name, msg)
#define ASSERT_EQ(a, b, name, msg) if ((a) != (b)) { printf("  Expected %d, got %d\n", (int)(b), (int)(a)); TEST_FAIL(name, msg); }

// =============================================================================
// Test: Basic Insert and Lookup
// =============================================================================

int test_basic_insert_lookup() {
    const char* test_name = "Basic insert and lookup";
    
    PTO2TensorMapEx tm;
    if (!pto2_tensormapex_init_default(&tm)) {
        TEST_FAIL(test_name, "Failed to init tensormapex");
    }
    
    // Create a tensor and insert it
    PTO2LogicalTensor tensor;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&tensor, base, shape, 2, sizeof(float));
    
    pto2_tensormapex_insert(&tm, &tensor, 100);  // Task 100 produces this tensor
    
    // Lookup the same tensor
    int32_t producer = pto2_tensormapex_lookup(&tm, &tensor);
    ASSERT_EQ(producer, 100, test_name, "producer should be 100");
    
    // Lookup with different base (should not find)
    PTO2LogicalTensor other;
    pto2_logical_tensor_init_raw(&other, (void*)0x2000, shape, 2, sizeof(float));
    producer = pto2_tensormapex_lookup(&tm, &other);
    ASSERT_EQ(producer, -1, test_name, "should not find producer for different base");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Overlapping Views
// =============================================================================

int test_overlapping_views() {
    const char* test_name = "Overlapping views dependency";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    // Task 100 produces a 4x4 tensor
    PTO2LogicalTensor tensor100;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&tensor100, base, shape, 2, sizeof(float));
    pto2_tensormapex_insert(&tm, &tensor100, 100);
    
    // Task 200 wants to read rows 0-1 (view)
    PTO2LogicalTensor view_top;
    int64_t start_top[] = {0, 0};
    int64_t shape_top[] = {2, 4};
    pto2_logical_tensor_view(&tensor100, &view_top, start_top, shape_top, 2);
    
    int32_t producer = pto2_tensormapex_lookup(&tm, &view_top);
    ASSERT_EQ(producer, 100, test_name, "view of top rows should depend on task 100");
    
    // Task 300 wants to read rows 2-3 (view)
    PTO2LogicalTensor view_bottom;
    int64_t start_bottom[] = {2, 0};
    int64_t shape_bottom[] = {2, 4};
    pto2_logical_tensor_view(&tensor100, &view_bottom, start_bottom, shape_bottom, 2);
    
    producer = pto2_tensormapex_lookup(&tm, &view_bottom);
    ASSERT_EQ(producer, 100, test_name, "view of bottom rows should depend on task 100");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: lookup_all - Multiple Overlapping Producers
// =============================================================================

int test_lookup_all() {
    const char* test_name = "lookup_all - multiple producers";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    // Create a large tensor
    void* base = (void*)0x1000;
    
    // Task 100 writes rows 0-1
    PTO2LogicalTensor write1;
    int64_t shape1[] = {2, 8};
    pto2_logical_tensor_init_raw(&write1, base, shape1, 2, sizeof(float));
    write1.raw_total_size = 4 * 8 * sizeof(float);  // Full tensor is 4x8
    pto2_tensormapex_insert(&tm, &write1, 100);
    
    // Task 101 writes rows 2-3
    PTO2LogicalTensor write2;
    int64_t shape2[] = {2, 8};
    pto2_logical_tensor_init(&write2, base, 4*8*sizeof(float), 
                             2*8*sizeof(float), shape2, 
                             (int64_t[]){8*sizeof(float), sizeof(float)},
                             2, sizeof(float), PTO2_TENSOR_VIEW);
    pto2_tensormapex_insert(&tm, &write2, 101);
    
    // Task 200 wants to read rows 1-2 (overlaps with BOTH task 100 and 101)
    PTO2LogicalTensor read_region;
    int64_t shape_read[] = {2, 8};
    pto2_logical_tensor_init(&read_region, base, 4*8*sizeof(float),
                             1*8*sizeof(float), shape_read,
                             (int64_t[]){8*sizeof(float), sizeof(float)},
                             2, sizeof(float), PTO2_TENSOR_VIEW);
    
    int32_t producers[16];
    int32_t count = pto2_tensormapex_lookup_all(&tm, &read_region, producers, 16);
    
    ASSERT_EQ(count, 2, test_name, "should find 2 overlapping producers");
    
    // Check that both 100 and 101 are found
    bool found_100 = false, found_101 = false;
    for (int i = 0; i < count; i++) {
        if (producers[i] == 100) found_100 = true;
        if (producers[i] == 101) found_101 = true;
    }
    ASSERT_TRUE(found_100, test_name, "should find producer 100");
    ASSERT_TRUE(found_101, test_name, "should find producer 101");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Non-Overlapping Regions
// =============================================================================

int test_non_overlapping() {
    const char* test_name = "Non-overlapping regions";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    void* base = (void*)0x1000;
    
    // Task 100 writes rows 0-1
    PTO2LogicalTensor write1;
    int64_t shape1[] = {2, 4};
    pto2_logical_tensor_init_raw(&write1, base, shape1, 2, sizeof(float));
    write1.raw_total_size = 4 * 4 * sizeof(float);
    pto2_tensormapex_insert(&tm, &write1, 100);
    
    // Task 200 wants to read rows 2-3 (non-overlapping with task 100)
    PTO2LogicalTensor read_region;
    int64_t shape_read[] = {2, 4};
    pto2_logical_tensor_init(&read_region, base, 4*4*sizeof(float),
                             2*4*sizeof(float), shape_read,
                             (int64_t[]){4*sizeof(float), sizeof(float)},
                             2, sizeof(float), PTO2_TENSOR_VIEW);
    
    int32_t producers[16];
    int32_t count = pto2_tensormapex_lookup_all(&tm, &read_region, producers, 16);
    
    ASSERT_EQ(count, 0, test_name, "should find 0 overlapping producers");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Deep Copy Independence
// =============================================================================

int test_deep_copy_independence() {
    const char* test_name = "Deep copy independence";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    // Original tensor at address 0x1000
    PTO2LogicalTensor original;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    pto2_tensormapex_insert(&tm, &original, 100);
    
    // Deep copy at address 0x2000 (different raw_base)
    PTO2LogicalTensor deep_copy;
    pto2_logical_tensor_init_raw(&deep_copy, (void*)0x2000, shape, 2, sizeof(float));
    deep_copy.extraction_type = PTO2_TENSOR_DEEP_VIEW;  // Mark as deep copy
    pto2_tensormapex_insert(&tm, &deep_copy, 101);
    
    // Reading from 0x2000 should NOT depend on task 100 (different storage)
    int32_t producer = pto2_tensormapex_lookup(&tm, &deep_copy);
    ASSERT_EQ(producer, 101, test_name, "deep copy should only depend on its own producer");
    
    // Reading from 0x1000 should depend on task 100
    producer = pto2_tensormapex_lookup(&tm, &original);
    ASSERT_EQ(producer, 100, test_name, "original should depend on task 100");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Lazy Invalidation
// =============================================================================

int test_lazy_invalidation() {
    const char* test_name = "Lazy invalidation";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    // Insert entries for tasks 100, 101, 102
    void* base = (void*)0x1000;
    PTO2LogicalTensor tensor;
    int64_t shape[] = {4, 4};
    pto2_logical_tensor_init_raw(&tensor, base, shape, 2, sizeof(float));
    
    pto2_tensormapex_insert(&tm, &tensor, 100);
    pto2_tensormapex_insert(&tm, &tensor, 101);
    pto2_tensormapex_insert(&tm, &tensor, 102);
    
    // All entries valid initially
    ASSERT_EQ(pto2_tensormapex_valid_count(&tm), 3, test_name, "should have 3 valid entries");
    
    // Advance last_task_alive to 101 - task 100 becomes stale
    pto2_tensormapex_sync_validity(&tm, 101);
    ASSERT_EQ(pto2_tensormapex_valid_count(&tm), 2, test_name, "should have 2 valid entries after invalidation");
    
    // Lookup should still find 102 (most recent valid)
    int32_t producer = pto2_tensormapex_lookup(&tm, &tensor);
    ASSERT_EQ(producer, 102, test_name, "should find most recent valid producer");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Print Statistics
// =============================================================================

int test_print_stats() {
    const char* test_name = "Print statistics";
    
    PTO2TensorMapEx tm;
    pto2_tensormapex_init_default(&tm);
    
    // Insert some entries
    for (int i = 0; i < 10; i++) {
        PTO2LogicalTensor tensor;
        int64_t shape[] = {4, 4};
        pto2_logical_tensor_init_raw(&tensor, (void*)(uintptr_t)(0x1000 + i*0x100), 
                                     shape, 2, sizeof(float));
        pto2_tensormapex_insert(&tm, &tensor, 100 + i);
    }
    
    printf("--- Debug output for %s ---\n", test_name);
    pto2_tensormapex_print_stats(&tm);
    printf("--- End debug output ---\n");
    
    pto2_tensormapex_destroy(&tm);
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("=== Extended TensorMap Tests ===\n\n");
    
    int failures = 0;
    
    failures += test_basic_insert_lookup();
    failures += test_overlapping_views();
    failures += test_lookup_all();
    failures += test_non_overlapping();
    failures += test_deep_copy_independence();
    failures += test_lazy_invalidation();
    failures += test_print_stats();
    
    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    
    return failures;
}
