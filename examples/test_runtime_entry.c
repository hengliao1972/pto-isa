/**
 * Test case for runtime_entry_arm64() - Multi-threaded PTO Runtime Execution
 * 
 * This test demonstrates:
 * 1. Simple InCore functions with correct signature
 * 2. Orchestration function that builds task graph
 * 3. Multi-threaded execution using runtime_entry_arm64()
 * 
 * Compile:
 *   gcc -O2 -pthread -o test_runtime_entry test_runtime_entry.c ../pto_runtime.c -I.. -lm
 * 
 * Run:
 *   ./test_runtime_entry [num_workers] [num_tiles]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Include PTO Runtime (with implementation)
#include "../pto_runtime.h"

// =============================================================================
// Configuration
// =============================================================================

#define TILE_ROWS 8
#define TILE_COLS 8
#define DEFAULT_NUM_TILES 8
#define DEFAULT_NUM_WORKERS 4

// =============================================================================
// InCore Functions - Simple tile-level operations
// =============================================================================

/**
 * rowmax: Find maximum value in each row
 * Input:  [TILE_ROWS x TILE_COLS]
 * Output: [TILE_ROWS x 1]
 */
void rowmax_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        float max_val = input[row * TILE_COLS];
        for (int col = 1; col < TILE_COLS; col++) {
            if (input[row * TILE_COLS + col] > max_val) {
                max_val = input[row * TILE_COLS + col];
            }
        }
        output[row] = max_val;
    }
}

/**
 * rowexpandsub: Subtract row vector from each row
 * Input1: [TILE_ROWS x TILE_COLS]
 * Input2: [TILE_ROWS x 1] (row vector)
 * Output: [TILE_ROWS x TILE_COLS]
 */
void rowexpandsub_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input_x = (float*)args[0];
    float* input_row = (float*)args[1];
    float* output = (float*)args[2];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        float row_val = input_row[row];
        for (int col = 0; col < TILE_COLS; col++) {
            output[row * TILE_COLS + col] = input_x[row * TILE_COLS + col] - row_val;
        }
    }
}

/**
 * elem_exp: Element-wise exponential
 * Input:  [TILE_ROWS x TILE_COLS]
 * Output: [TILE_ROWS x TILE_COLS]
 */
void elem_exp_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    
    for (int i = 0; i < TILE_ROWS * TILE_COLS; i++) {
        output[i] = expf(input[i]);
    }
}

/**
 * rowsum: Sum values in each row
 * Input:  [TILE_ROWS x TILE_COLS]
 * Output: [TILE_ROWS x 1]
 */
void rowsum_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        float sum = 0.0f;
        for (int col = 0; col < TILE_COLS; col++) {
            sum += input[row * TILE_COLS + col];
        }
        output[row] = sum;
    }
}

/**
 * rowexpanddiv: Divide each row by row vector
 * Input1: [TILE_ROWS x TILE_COLS]
 * Input2: [TILE_ROWS x 1] (row vector)
 * Output: [TILE_ROWS x TILE_COLS]
 */
void rowexpanddiv_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input_x = (float*)args[0];
    float* input_row = (float*)args[1];
    float* output = (float*)args[2];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        float row_val = input_row[row];
        for (int col = 0; col < TILE_COLS; col++) {
            output[row * TILE_COLS + col] = input_x[row * TILE_COLS + col] / row_val;
        }
    }
}

// =============================================================================
// User Data for Orchestration
// =============================================================================

typedef struct {
    float* input;
    float* output;
    float* temp_rowmax;
    float* temp_shifted;
    float* temp_exp;
    float* temp_rowsum;
    int num_tiles;
} SoftmaxUserData;

// =============================================================================
// Orchestration Function - Builds Task Graph for Softmax
// =============================================================================

void softmax_orchestration(PTORuntime* rt, void* user_data) {
    SoftmaxUserData* data = (SoftmaxUserData*)user_data;
    
    int tile_size = TILE_ROWS * TILE_COLS;
    int row_vec_size = TILE_ROWS;
    
    printf("[Orchestration] Building softmax task graph for %d tiles...\n", data->num_tiles);
    
    for (int tile_idx = 0; tile_idx < data->num_tiles; tile_idx++) {
        int tile_offset = tile_idx * tile_size;
        int row_offset = tile_idx * row_vec_size;
        
        // Task 0: rowmax - find row maximum
        int32_t t0 = pto_task_alloc(rt, "rowmax", (void*)rowmax_incore, 0, 0);
        pto_task_add_input(rt, t0, data->input + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_add_output(rt, t0, data->temp_rowmax + row_offset, 0, 0, TILE_ROWS, 1);
        pto_task_submit(rt, t0);
        
        // Task 1: rowexpandsub - subtract row max for numerical stability
        int32_t t1 = pto_task_alloc(rt, "rowexpandsub", (void*)rowexpandsub_incore, 0, 0);
        pto_task_add_input(rt, t1, data->input + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_add_input(rt, t1, data->temp_rowmax + row_offset, 0, 0, TILE_ROWS, 1);
        pto_task_add_output(rt, t1, data->temp_shifted + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_submit(rt, t1);
        
        // Task 2: elem_exp - compute exponential
        int32_t t2 = pto_task_alloc(rt, "elem_exp", (void*)elem_exp_incore, 0, 0);
        pto_task_add_input(rt, t2, data->temp_shifted + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_add_output(rt, t2, data->temp_exp + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_submit(rt, t2);
        
        // Task 3: rowsum - sum each row
        int32_t t3 = pto_task_alloc(rt, "rowsum", (void*)rowsum_incore, 0, 0);
        pto_task_add_input(rt, t3, data->temp_exp + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_add_output(rt, t3, data->temp_rowsum + row_offset, 0, 0, TILE_ROWS, 1);
        pto_task_submit(rt, t3);
        
        // Task 4: rowexpanddiv - normalize by row sum
        int32_t t4 = pto_task_alloc(rt, "rowexpanddiv", (void*)rowexpanddiv_incore, 0, 0);
        pto_task_add_input(rt, t4, data->temp_exp + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_add_input(rt, t4, data->temp_rowsum + row_offset, 0, 0, TILE_ROWS, 1);
        pto_task_add_output(rt, t4, data->output + tile_offset, 0, 0, TILE_ROWS, TILE_COLS);
        pto_task_submit(rt, t4);
    }
    
    printf("[Orchestration] Task graph complete: %d tiles x 5 ops = %d tasks\n", 
           data->num_tiles, data->num_tiles * 5);
}

// =============================================================================
// Verification - Check softmax output
// =============================================================================

int verify_softmax(float* input, float* output, int total_rows, int cols) {
    int errors = 0;
    float tolerance = 1e-5f;
    
    for (int row = 0; row < total_rows; row++) {
        // Compute reference softmax
        float max_val = input[row * cols];
        for (int col = 1; col < cols; col++) {
            if (input[row * cols + col] > max_val) {
                max_val = input[row * cols + col];
            }
        }
        
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += expf(input[row * cols + col] - max_val);
        }
        
        for (int col = 0; col < cols; col++) {
            float expected = expf(input[row * cols + col] - max_val) / sum;
            float actual = output[row * cols + col];
            float diff = fabsf(expected - actual);
            
            if (diff > tolerance) {
                if (errors < 5) {
                    printf("[Verify] Mismatch at [%d,%d]: expected=%f, actual=%f, diff=%f\n",
                           row, col, expected, actual, diff);
                }
                errors++;
            }
        }
        
        // Check row sums to 1
        float row_sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            row_sum += output[row * cols + col];
        }
        if (fabsf(row_sum - 1.0f) > tolerance * cols) {
            if (errors < 10) {
                printf("[Verify] Row %d sum=%f (should be 1.0)\n", row, row_sum);
            }
            errors++;
        }
    }
    
    return errors;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int num_workers = DEFAULT_NUM_WORKERS;
    int num_tiles = DEFAULT_NUM_TILES;
    
    // Parse command line arguments
    if (argc > 1) {
        num_workers = atoi(argv[1]);
        if (num_workers < 1) num_workers = 1;
        if (num_workers > 64) num_workers = 64;
    }
    if (argc > 2) {
        num_tiles = atoi(argv[2]);
        if (num_tiles < 1) num_tiles = 1;
    }
    
    int total_rows = num_tiles * TILE_ROWS;
    int total_elements = total_rows * TILE_COLS;
    int row_vec_elements = total_rows;
    
    printf("================================================================================\n");
    printf("PTO Runtime Entry Test - Multi-threaded Softmax\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Workers:    %d\n", num_workers);
    printf("  Tiles:      %d\n", num_tiles);
    printf("  Tile size:  %d x %d\n", TILE_ROWS, TILE_COLS);
    printf("  Total rows: %d\n", total_rows);
    printf("  Total ops:  %d (5 ops per tile)\n", num_tiles * 5);
    printf("================================================================================\n\n");
    
    // Allocate buffers
    float* input = (float*)malloc(total_elements * sizeof(float));
    float* output = (float*)malloc(total_elements * sizeof(float));
    float* temp_rowmax = (float*)malloc(row_vec_elements * sizeof(float));
    float* temp_shifted = (float*)malloc(total_elements * sizeof(float));
    float* temp_exp = (float*)malloc(total_elements * sizeof(float));
    float* temp_rowsum = (float*)malloc(row_vec_elements * sizeof(float));
    
    if (!input || !output || !temp_rowmax || !temp_shifted || !temp_exp || !temp_rowsum) {
        fprintf(stderr, "ERROR: Failed to allocate buffers\n");
        return 1;
    }
    
    // Initialize input with random values
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]
    }
    memset(output, 0, total_elements * sizeof(float));
    
    // Prepare user data
    SoftmaxUserData user_data = {
        .input = input,
        .output = output,
        .temp_rowmax = temp_rowmax,
        .temp_shifted = temp_shifted,
        .temp_exp = temp_exp,
        .temp_rowsum = temp_rowsum,
        .num_tiles = num_tiles
    };
    
    // Measure execution time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Execute using multi-threaded runtime
    int result = runtime_entry_arm64(softmax_orchestration, &user_data, num_workers);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("\n");
    
    if (result != 0) {
        fprintf(stderr, "ERROR: runtime_entry_arm64 failed with code %d\n", result);
        return 1;
    }
    
    // Verify results
    printf("Verifying results...\n");
    int errors = verify_softmax(input, output, total_rows, TILE_COLS);
    
    printf("\n================================================================================\n");
    printf("Results\n");
    printf("================================================================================\n");
    printf("  Execution time: %.2f ms\n", elapsed_ms);
    printf("  Tasks/second:   %.0f\n", (num_tiles * 5) / (elapsed_ms / 1000.0));
    printf("  Verification:   %s\n", errors == 0 ? "PASSED" : "FAILED");
    if (errors > 0) {
        printf("  Errors:         %d\n", errors);
    }
    printf("================================================================================\n");
    
    // Print sample output
    printf("\nSample output (first tile, first 3 rows):\n");
    for (int row = 0; row < 3 && row < total_rows; row++) {
        printf("  Row %d: [", row);
        float sum = 0;
        for (int col = 0; col < TILE_COLS; col++) {
            printf("%.4f", output[row * TILE_COLS + col]);
            if (col < TILE_COLS - 1) printf(", ");
            sum += output[row * TILE_COLS + col];
        }
        printf("] sum=%.6f\n", sum);
    }
    
    // Cleanup
    free(input);
    free(output);
    free(temp_rowmax);
    free(temp_shifted);
    free(temp_exp);
    free(temp_rowsum);
    
    return errors > 0 ? 1 : 0;
}
