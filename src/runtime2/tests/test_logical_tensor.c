/**
 * Test for Logical Tensor and Overlap Detection
 * 
 * Tests:
 * 1. Bounding box computation
 * 2. View, reshape, transpose operations
 * 3. Overlap detection between tensors
 */

#include "../pto_logical_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)
#define ASSERT_TRUE(cond, name, msg) if (!(cond)) TEST_FAIL(name, msg)
#define ASSERT_FALSE(cond, name, msg) if (cond) TEST_FAIL(name, msg)
#define ASSERT_EQ(a, b, name, msg) if ((a) != (b)) TEST_FAIL(name, msg)

// =============================================================================
// Test: Bounding Box Computation
// =============================================================================

int test_bounding_box_contiguous() {
    const char* test_name = "Bounding box - contiguous tensor";
    
    // Create a 3x4 contiguous tensor (float32)
    PTO2LogicalTensor tensor;
    int64_t shape[] = {3, 4};
    void* base = (void*)0x1000;  // Dummy address
    
    pto2_logical_tensor_init_raw(&tensor, base, shape, 2, sizeof(float));
    
    // Expected: contiguous 3x4 float tensor
    // Total size: 3*4*4 = 48 bytes
    // min_offset = 0, max_offset = 47 (includes last byte)
    
    ASSERT_EQ(tensor.min_byte_offset, 0, test_name, "min_byte_offset should be 0");
    ASSERT_EQ(tensor.max_byte_offset, 47, test_name, "max_byte_offset should be 47");
    ASSERT_EQ(tensor.numel, 12, test_name, "numel should be 12");
    ASSERT_TRUE(tensor.is_contiguous, test_name, "should be contiguous");
    
    TEST_PASS(test_name);
    return 0;
}

int test_bounding_box_view() {
    const char* test_name = "Bounding box - view (slice)";
    
    // Create a 4x4 contiguous tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // Create view: [1:3, 1:3] -> 2x2 tensor starting at offset [1,1]
    PTO2LogicalTensor view;
    int64_t start[] = {1, 1};
    int64_t view_shape[] = {2, 2};
    
    bool ok = pto2_logical_tensor_view(&original, &view, start, view_shape, 2);
    ASSERT_TRUE(ok, test_name, "view creation should succeed");
    
    // Storage offset = 1*16 + 1*4 = 20 bytes
    ASSERT_EQ(view.storage_offset, 20, test_name, "storage_offset should be 20");
    
    // Bounding box: [20, 20+4+16+4-1] = [20, 43]
    // First element at byte 20, last element at byte 20 + 1*16 + 1*4 + 3 = 43
    ASSERT_EQ(view.min_byte_offset, 20, test_name, "min_byte_offset should be 20");
    ASSERT_EQ(view.max_byte_offset, 43, test_name, "max_byte_offset should be 43");
    
    // View should NOT be contiguous (strides don't match new shape)
    ASSERT_FALSE(view.is_contiguous, test_name, "view should not be contiguous");
    
    TEST_PASS(test_name);
    return 0;
}

int test_bounding_box_transpose() {
    const char* test_name = "Bounding box - transpose";
    
    // Create a 2x3 contiguous tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {2, 3};
    void* base = (void*)0x1000;
    
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // Transpose to 3x2
    PTO2LogicalTensor transposed;
    bool ok = pto2_logical_tensor_transpose(&original, &transposed, NULL);
    ASSERT_TRUE(ok, test_name, "transpose should succeed");
    
    // Shape should be reversed
    ASSERT_EQ(transposed.shape[0], 3, test_name, "transposed shape[0] should be 3");
    ASSERT_EQ(transposed.shape[1], 2, test_name, "transposed shape[1] should be 2");
    
    // Strides should be reversed
    ASSERT_EQ(transposed.strides[0], sizeof(float), test_name, "transposed stride[0] should be 4");
    ASSERT_EQ(transposed.strides[1], 3*sizeof(float), test_name, "transposed stride[1] should be 12");
    
    // Bounding box should be same as original (same memory footprint)
    ASSERT_EQ(transposed.min_byte_offset, 0, test_name, "min_byte_offset should be 0");
    ASSERT_EQ(transposed.max_byte_offset, 23, test_name, "max_byte_offset should be 23");
    
    // Transposed is NOT contiguous
    ASSERT_FALSE(transposed.is_contiguous, test_name, "transposed should not be contiguous");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Overlap Detection
// =============================================================================

int test_overlap_different_storage() {
    const char* test_name = "Overlap - different storage (no overlap)";
    
    // Two tensors with different base pointers
    PTO2LogicalTensor a, b;
    int64_t shape[] = {4, 4};
    
    pto2_logical_tensor_init_raw(&a, (void*)0x1000, shape, 2, sizeof(float));
    pto2_logical_tensor_init_raw(&b, (void*)0x2000, shape, 2, sizeof(float));
    
    bool overlap = pto2_logical_tensor_overlap_fast(&a, &b);
    ASSERT_FALSE(overlap, test_name, "different storage should not overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_overlap_same_tensor() {
    const char* test_name = "Overlap - same tensor (always overlap)";
    
    PTO2LogicalTensor tensor;
    int64_t shape[] = {4, 4};
    pto2_logical_tensor_init_raw(&tensor, (void*)0x1000, shape, 2, sizeof(float));
    
    bool overlap = pto2_logical_tensor_overlap_fast(&tensor, &tensor);
    ASSERT_TRUE(overlap, test_name, "tensor should overlap with itself");
    
    TEST_PASS(test_name);
    return 0;
}

int test_overlap_non_overlapping_views() {
    const char* test_name = "Overlap - non-overlapping views";
    
    // Create a 4x4 tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // View 1: rows 0-1 (top half)
    PTO2LogicalTensor view1;
    int64_t start1[] = {0, 0};
    int64_t shape1[] = {2, 4};
    pto2_logical_tensor_view(&original, &view1, start1, shape1, 2);
    
    // View 2: rows 2-3 (bottom half)
    PTO2LogicalTensor view2;
    int64_t start2[] = {2, 0};
    int64_t shape2[] = {2, 4};
    pto2_logical_tensor_view(&original, &view2, start2, shape2, 2);
    
    bool overlap = pto2_logical_tensor_overlap_fast(&view1, &view2);
    ASSERT_FALSE(overlap, test_name, "non-overlapping views should not overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_overlap_overlapping_views() {
    const char* test_name = "Overlap - overlapping views";
    
    // Create a 4x4 tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // View 1: rows 0-2 (first 3 rows)
    PTO2LogicalTensor view1;
    int64_t start1[] = {0, 0};
    int64_t shape1[] = {3, 4};
    pto2_logical_tensor_view(&original, &view1, start1, shape1, 2);
    
    // View 2: rows 1-3 (last 3 rows)
    PTO2LogicalTensor view2;
    int64_t start2[] = {1, 0};
    int64_t shape2[] = {3, 4};
    pto2_logical_tensor_view(&original, &view2, start2, shape2, 2);
    
    bool overlap = pto2_logical_tensor_overlap_fast(&view1, &view2);
    ASSERT_TRUE(overlap, test_name, "overlapping views should overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_overlap_partial_column_overlap() {
    const char* test_name = "Overlap - partial column overlap";
    
    // Create a 4x4 tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // View 1: [0:2, 0:2] (top-left 2x2)
    PTO2LogicalTensor view1;
    int64_t start1[] = {0, 0};
    int64_t shape1[] = {2, 2};
    pto2_logical_tensor_view(&original, &view1, start1, shape1, 2);
    
    // View 2: [1:3, 1:3] (middle 2x2)
    PTO2LogicalTensor view2;
    int64_t start2[] = {1, 1};
    int64_t shape2[] = {2, 2};
    pto2_logical_tensor_view(&original, &view2, start2, shape2, 2);
    
    bool overlap = pto2_logical_tensor_overlap_fast(&view1, &view2);
    ASSERT_TRUE(overlap, test_name, "partially overlapping views should overlap");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Reshape
// =============================================================================

int test_reshape_valid() {
    const char* test_name = "Reshape - valid reshape";
    
    // Create a 2x6 tensor
    PTO2LogicalTensor original;
    int64_t shape[] = {2, 6};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&original, base, shape, 2, sizeof(float));
    
    // Reshape to 3x4
    PTO2LogicalTensor reshaped;
    int64_t new_shape[] = {3, 4};
    bool ok = pto2_logical_tensor_reshape(&original, &reshaped, new_shape, 2);
    ASSERT_TRUE(ok, test_name, "reshape should succeed");
    
    // Check shape
    ASSERT_EQ(reshaped.shape[0], 3, test_name, "shape[0] should be 3");
    ASSERT_EQ(reshaped.shape[1], 4, test_name, "shape[1] should be 4");
    
    // Check numel unchanged
    ASSERT_EQ(reshaped.numel, 12, test_name, "numel should be 12");
    
    // Reshaped should be contiguous
    ASSERT_TRUE(reshaped.is_contiguous, test_name, "reshaped should be contiguous");
    
    // Bounding box unchanged
    ASSERT_EQ(reshaped.min_byte_offset, original.min_byte_offset, test_name, "min unchanged");
    ASSERT_EQ(reshaped.max_byte_offset, original.max_byte_offset, test_name, "max unchanged");
    
    TEST_PASS(test_name);
    return 0;
}

int test_reshape_invalid_numel() {
    const char* test_name = "Reshape - invalid numel (should fail)";
    
    PTO2LogicalTensor original;
    int64_t shape[] = {2, 6};
    pto2_logical_tensor_init_raw(&original, (void*)0x1000, shape, 2, sizeof(float));
    
    // Try to reshape to 3x3 (9 != 12)
    PTO2LogicalTensor reshaped;
    int64_t new_shape[] = {3, 3};
    bool ok = pto2_logical_tensor_reshape(&original, &reshaped, new_shape, 2);
    ASSERT_FALSE(ok, test_name, "reshape with wrong numel should fail");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: TensorMapEntryEx Overlap
// =============================================================================

int test_entry_overlap() {
    const char* test_name = "Entry overlap - tensormap entries";
    
    // Create two logical tensors that overlap
    PTO2LogicalTensor tensor1, tensor2;
    int64_t shape[] = {4, 4};
    void* base = (void*)0x1000;
    pto2_logical_tensor_init_raw(&tensor1, base, shape, 2, sizeof(float));
    
    // View of first two rows
    int64_t start[] = {0, 0};
    int64_t view_shape[] = {2, 4};
    pto2_logical_tensor_view(&tensor1, &tensor2, start, view_shape, 2);
    
    // Convert to entries
    PTO2TensorMapEntryEx entry1, entry2;
    pto2_logical_tensor_to_entry(&tensor1, &entry1, 100);
    pto2_logical_tensor_to_entry(&tensor2, &entry2, 101);
    
    bool overlap = pto2_tensormap_entry_overlap_fast(&entry1, &entry2);
    ASSERT_TRUE(overlap, test_name, "entries from same storage should overlap");
    
    // Check producer task IDs preserved
    ASSERT_EQ(entry1.producer_task_id, 100, test_name, "entry1 producer should be 100");
    ASSERT_EQ(entry2.producer_task_id, 101, test_name, "entry2 producer should be 101");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: GCD-Based Exact Overlap Detection
// =============================================================================

int test_gcd_overlap_interleaved() {
    const char* test_name = "GCD exact - interleaved (no overlap)";
    
    // A: elements at bytes [0, 8, 16, 24] (stride=8, size=4)
    // B: elements at bytes [4, 12, 20, 28] (stride=8, size=4, offset=4)
    // These interleave - no actual overlap!
    
    bool overlap = pto2_overlap_1d_exact(0, 8, 4, 4, 8, 4);
    ASSERT_FALSE(overlap, test_name, "interleaved arrays should not overlap (GCD)");
    
    // But bounding box would say they DO overlap:
    // A: [0, 24], B: [4, 28] -> bounding boxes intersect
    // This demonstrates GCD's advantage over bounding box
    
    TEST_PASS(test_name);
    return 0;
}

int test_gcd_overlap_same_stride() {
    const char* test_name = "GCD exact - same stride, offset difference";
    
    // A: elements at bytes [0, 4, 8, 12] (stride=4, size=4)
    // B: elements at bytes [8, 12, 16, 20] (stride=4, size=4, offset=8)
    // These overlap at bytes 8 and 12
    
    bool overlap = pto2_overlap_1d_exact(0, 4, 4, 8, 4, 4);
    ASSERT_TRUE(overlap, test_name, "overlapping ranges should overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_gcd_overlap_different_strides() {
    const char* test_name = "GCD exact - different strides (overlap)";
    
    // A: elements at bytes [0, 6, 12, 18] (stride=6, size=4)
    // B: elements at bytes [0, 4, 8, 12, 16, 20] (stride=4, size=6)
    // Overlap at: 0, 12 (gcd(6,4)=2, and both sequences hit 0 and 12)
    
    bool overlap = pto2_overlap_1d_exact(0, 6, 4, 0, 4, 6);
    ASSERT_TRUE(overlap, test_name, "sequences hitting same points should overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_gcd_overlap_different_strides_no_overlap() {
    const char* test_name = "GCD exact - different strides (no overlap)";
    
    // A: elements at bytes [0, 6, 12] (stride=6, size=3)
    // B: elements at bytes [2, 8, 14] (stride=6, size=3, offset=2)
    // gcd(6,6)=6, but 2 is not divisible by 6, so no overlap
    
    bool overlap = pto2_overlap_1d_exact(0, 6, 3, 2, 6, 3);
    ASSERT_FALSE(overlap, test_name, "offset difference not divisible by gcd means no overlap");
    
    TEST_PASS(test_name);
    return 0;
}

int test_gcd_tensor_overlap_exact() {
    const char* test_name = "GCD exact - tensor level";
    
    // Create two 1D tensors that interleave (no overlap)
    PTO2LogicalTensor a, b;
    void* base = (void*)0x1000;
    
    // A: 4 elements starting at offset 0, stride 8
    a.raw_base = base;
    a.raw_total_size = 32;
    a.storage_offset = 0;
    a.shape[0] = 4;
    a.strides[0] = 8;
    a.ndim = 1;
    a.elem_size = 4;
    a.numel = 4;
    a.is_contiguous = false;
    pto2_logical_tensor_update_bounding_box(&a);
    
    // B: 4 elements starting at offset 4, stride 8
    b.raw_base = base;
    b.raw_total_size = 32;
    b.storage_offset = 4;
    b.shape[0] = 4;
    b.strides[0] = 8;
    b.ndim = 1;
    b.elem_size = 4;
    b.numel = 4;
    b.is_contiguous = false;
    pto2_logical_tensor_update_bounding_box(&b);
    
    // Bounding box check: should return TRUE (conservative)
    bool fast_result = pto2_logical_tensor_overlap_fast(&a, &b);
    ASSERT_TRUE(fast_result, test_name, "bounding box should detect potential overlap");
    
    // Exact check: should return FALSE (no actual overlap)
    bool exact_result = pto2_logical_tensor_overlap_exact(&a, &b);
    ASSERT_FALSE(exact_result, test_name, "GCD exact should detect NO overlap");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Test: Print and Debug
// =============================================================================

int test_print() {
    const char* test_name = "Print - tensor info";
    
    PTO2LogicalTensor tensor;
    int64_t shape[] = {2, 3, 4};
    pto2_logical_tensor_init_raw(&tensor, (void*)0x1000, shape, 3, sizeof(float));
    
    printf("--- Debug output for %s ---\n", test_name);
    pto2_logical_tensor_print(&tensor, "test_3d_tensor");
    printf("--- End debug output ---\n");
    
    TEST_PASS(test_name);
    return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("=== Logical Tensor Tests ===\n\n");
    
    int failures = 0;
    
    // Bounding box tests
    failures += test_bounding_box_contiguous();
    failures += test_bounding_box_view();
    failures += test_bounding_box_transpose();
    
    // Overlap tests (bounding box)
    failures += test_overlap_different_storage();
    failures += test_overlap_same_tensor();
    failures += test_overlap_non_overlapping_views();
    failures += test_overlap_overlapping_views();
    failures += test_overlap_partial_column_overlap();
    
    // Reshape tests
    failures += test_reshape_valid();
    failures += test_reshape_invalid_numel();
    
    // Entry tests
    failures += test_entry_overlap();
    
    // GCD exact overlap tests
    failures += test_gcd_overlap_interleaved();
    failures += test_gcd_overlap_same_stride();
    failures += test_gcd_overlap_different_strides();
    failures += test_gcd_overlap_different_strides_no_overlap();
    failures += test_gcd_tensor_overlap_exact();
    
    // Debug tests
    failures += test_print();
    
    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    
    return failures;
}
