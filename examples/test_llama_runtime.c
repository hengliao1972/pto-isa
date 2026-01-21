/**
 * Simplified LLaMA Layer Test with runtime_entry_arm64
 * 
 * This test implements a simplified LLaMA decoder layer:
 * 1. RMSNorm
 * 2. Q/K/V Linear projections (simplified as scale)
 * 3. Attention scores (simplified)
 * 4. Softmax
 * 5. Attention output
 * 6. Residual add
 * 
 * Sequence Length: configurable (default 128)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../pto_runtime.h"

// =============================================================================
// Configuration
// =============================================================================

#define DEFAULT_SEQ_LEN     128
#define DEFAULT_NUM_WORKERS 4
#define TILE_ROWS           32
#define HIDDEN_DIM          128     // Simplified (real LLaMA: 4096)
#define HEAD_DIM            32      // Simplified (real LLaMA: 128)
#define NUM_HEADS           4       // Simplified (real LLaMA: 32)

// =============================================================================
// InCore Functions - Simplified LLaMA Operations
// =============================================================================

/**
 * RMSNorm: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
 * Shape: [TILE_ROWS x HIDDEN_DIM]
 */
void rmsnorm_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* weight = (float*)args[1];
    float* output = (float*)args[2];
    float eps = 1e-6f;
    
    for (int row = 0; row < TILE_ROWS; row++) {
        // Compute mean of squares
        float sum_sq = 0.0f;
        for (int col = 0; col < HIDDEN_DIM; col++) {
            float val = input[row * HIDDEN_DIM + col];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / HIDDEN_DIM + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize and scale
        for (int col = 0; col < HIDDEN_DIM; col++) {
            output[row * HIDDEN_DIM + col] = 
                input[row * HIDDEN_DIM + col] * inv_rms * weight[col];
        }
    }
}

/**
 * Linear projection: output = input @ weight^T (simplified)
 * Input:  [TILE_ROWS x HIDDEN_DIM]
 * Weight: [HIDDEN_DIM x HIDDEN_DIM] (simplified)
 * Output: [TILE_ROWS x HIDDEN_DIM]
 */
void linear_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* weight = (float*)args[1];
    float* output = (float*)args[2];
    
    // Simplified: element-wise multiply (real would be matmul)
    for (int row = 0; row < TILE_ROWS; row++) {
        for (int col = 0; col < HIDDEN_DIM; col++) {
            output[row * HIDDEN_DIM + col] = 
                input[row * HIDDEN_DIM + col] * weight[col];
        }
    }
}

/**
 * Attention score softmax (per-row softmax)
 * Shape: [TILE_ROWS x TILE_ROWS]
 */
void attention_softmax_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        // Find max
        float max_val = input[row * TILE_ROWS];
        for (int col = 1; col < TILE_ROWS; col++) {
            if (input[row * TILE_ROWS + col] > max_val) {
                max_val = input[row * TILE_ROWS + col];
            }
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int col = 0; col < TILE_ROWS; col++) {
            output[row * TILE_ROWS + col] = expf(input[row * TILE_ROWS + col] - max_val);
            sum += output[row * TILE_ROWS + col];
        }
        
        // Normalize
        for (int col = 0; col < TILE_ROWS; col++) {
            output[row * TILE_ROWS + col] /= sum;
        }
    }
}

/**
 * Residual add: output = input1 + input2
 * Shape: [TILE_ROWS x HIDDEN_DIM]
 */
void residual_add_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input1 = (float*)args[0];
    float* input2 = (float*)args[1];
    float* output = (float*)args[2];
    
    for (int i = 0; i < TILE_ROWS * HIDDEN_DIM; i++) {
        output[i] = input1[i] + input2[i];
    }
}

/**
 * Scale (simplified attention output projection)
 * Shape: [TILE_ROWS x HIDDEN_DIM]
 */
void scale_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    for (int i = 0; i < TILE_ROWS * HIDDEN_DIM; i++) {
        output[i] = input[i] * scale;
    }
}

// =============================================================================
// User Data
// =============================================================================

typedef struct {
    float* input;           // [seq_len x HIDDEN_DIM]
    float* output;          // [seq_len x HIDDEN_DIM]
    float* norm_weight;     // [HIDDEN_DIM]
    float* proj_weight;     // [HIDDEN_DIM] (simplified)
    float* temp_norm;       // [seq_len x HIDDEN_DIM]
    float* temp_proj;       // [seq_len x HIDDEN_DIM]
    float* temp_attn;       // [seq_len x seq_len] (attention scores)
    float* temp_softmax;    // [seq_len x seq_len]
    float* temp_out;        // [seq_len x HIDDEN_DIM]
    int seq_len;
    int num_tiles;
} LlamaUserData;

// =============================================================================
// Orchestration Function - Simplified LLaMA Layer
// =============================================================================

void llama_orchestration(PTORuntime* rt, void* user_data) {
    LlamaUserData* data = (LlamaUserData*)user_data;
    
    int tile_size = TILE_ROWS * HIDDEN_DIM;
    int attn_tile_size = TILE_ROWS * TILE_ROWS;
    
    printf("[Orchestration] Building LLaMA task graph for %d tiles...\n", data->num_tiles);
    
    // For each tile
    for (int tile_idx = 0; tile_idx < data->num_tiles; tile_idx++) {
        int offset = tile_idx * tile_size;
        int attn_offset = tile_idx * attn_tile_size;
        
        // Task 0: RMSNorm
        int32_t t0 = pto_task_alloc(rt, "rmsnorm", (void*)rmsnorm_incore, 0, 0);
        pto_task_add_input(rt, t0, data->input + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t0, data->norm_weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t0, data->temp_norm + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t0);
        
        // Task 1: Linear projection (simplified as Q projection)
        int32_t t1 = pto_task_alloc(rt, "linear_q", (void*)linear_incore, 0, 0);
        pto_task_add_input(rt, t1, data->temp_norm + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t1, data->proj_weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t1, data->temp_proj + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t1);
        
        // Task 2: Scale (simplified attention)
        int32_t t2 = pto_task_alloc(rt, "scale", (void*)scale_incore, 0, 0);
        pto_task_add_input(rt, t2, data->temp_proj + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t2, data->temp_out + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t2);
        
        // Task 3: Residual add
        int32_t t3 = pto_task_alloc(rt, "residual", (void*)residual_add_incore, 0, 0);
        pto_task_add_input(rt, t3, data->input + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t3, data->temp_out + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t3, data->output + offset, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t3);
    }
    
    printf("[Orchestration] Task graph complete: %d tiles x 4 ops = %d tasks\n", 
           data->num_tiles, data->num_tiles * 4);
}

// =============================================================================
// Reference Implementation
// =============================================================================

void llama_reference(LlamaUserData* data) {
    float eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    for (int tile_idx = 0; tile_idx < data->num_tiles; tile_idx++) {
        int offset = tile_idx * TILE_ROWS * HIDDEN_DIM;
        
        for (int row = 0; row < TILE_ROWS; row++) {
            int row_offset = offset + row * HIDDEN_DIM;
            
            // RMSNorm
            float sum_sq = 0.0f;
            for (int col = 0; col < HIDDEN_DIM; col++) {
                float val = data->input[row_offset + col];
                sum_sq += val * val;
            }
            float inv_rms = 1.0f / sqrtf(sum_sq / HIDDEN_DIM + eps);
            
            float temp_norm[HIDDEN_DIM];
            for (int col = 0; col < HIDDEN_DIM; col++) {
                temp_norm[col] = data->input[row_offset + col] * inv_rms * data->norm_weight[col];
            }
            
            // Linear + Scale
            float temp_out[HIDDEN_DIM];
            for (int col = 0; col < HIDDEN_DIM; col++) {
                temp_out[col] = temp_norm[col] * data->proj_weight[col] * scale;
            }
            
            // Residual add
            for (int col = 0; col < HIDDEN_DIM; col++) {
                data->output[row_offset + col] = data->input[row_offset + col] + temp_out[col];
            }
        }
    }
}

// =============================================================================
// Verification
// =============================================================================

int verify_llama(float* output, float* reference, int total_elements) {
    int errors = 0;
    float tolerance = 1e-4f;  // Relaxed tolerance for floating point
    
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(output[i] - reference[i]);
        float rel_diff = diff / (fabsf(reference[i]) + 1e-8f);
        
        if (rel_diff > tolerance && diff > tolerance) {
            if (errors < 5) {
                printf("[Verify] Mismatch at %d: output=%f, ref=%f, diff=%f\n",
                       i, output[i], reference[i], diff);
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
    int seq_len = DEFAULT_SEQ_LEN;
    int num_workers = DEFAULT_NUM_WORKERS;
    int threshold = 0;
    
    // Parse arguments
    if (argc > 1) seq_len = atoi(argv[1]);
    if (argc > 2) num_workers = atoi(argv[2]);
    if (argc > 3) threshold = atoi(argv[3]);
    
    if (seq_len < TILE_ROWS) seq_len = TILE_ROWS;
    
    int num_tiles = seq_len / TILE_ROWS;
    int total_elements = num_tiles * TILE_ROWS * HIDDEN_DIM;
    
    printf("================================================================================\n");
    printf("PTO Runtime - Simplified LLaMA Layer Test\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Sequence Length: %d\n", seq_len);
    printf("  Hidden Dim:      %d\n", HIDDEN_DIM);
    printf("  Tile Size:       %d x %d\n", TILE_ROWS, HIDDEN_DIM);
    printf("  Num Tiles:       %d\n", num_tiles);
    printf("  Workers:         %d\n", num_workers);
    printf("  Threshold:       %d %s\n", threshold, threshold > 0 ? "(pipelined)" : "(safe)");
    printf("  Total Elements:  %d\n", total_elements);
    printf("  Tasks:           %d (4 per tile)\n", num_tiles * 4);
    printf("================================================================================\n\n");
    
    // Allocate buffers
    float* input = (float*)malloc(total_elements * sizeof(float));
    float* output = (float*)malloc(total_elements * sizeof(float));
    float* reference = (float*)malloc(total_elements * sizeof(float));
    float* norm_weight = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* proj_weight = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* temp_norm = (float*)malloc(total_elements * sizeof(float));
    float* temp_proj = (float*)malloc(total_elements * sizeof(float));
    float* temp_attn = (float*)malloc(num_tiles * TILE_ROWS * TILE_ROWS * sizeof(float));
    float* temp_softmax = (float*)malloc(num_tiles * TILE_ROWS * TILE_ROWS * sizeof(float));
    float* temp_out = (float*)malloc(total_elements * sizeof(float));
    
    if (!input || !output || !reference || !norm_weight || !proj_weight ||
        !temp_norm || !temp_proj || !temp_attn || !temp_softmax || !temp_out) {
        fprintf(stderr, "ERROR: Failed to allocate buffers\n");
        return 1;
    }
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        norm_weight[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;  // [0.75, 1.25]
        proj_weight[i] = ((float)rand() / RAND_MAX) * 0.2f + 0.9f;   // [0.9, 1.1]
    }
    memset(output, 0, total_elements * sizeof(float));
    memset(reference, 0, total_elements * sizeof(float));
    
    // Prepare user data
    LlamaUserData user_data = {
        .input = input,
        .output = output,
        .norm_weight = norm_weight,
        .proj_weight = proj_weight,
        .temp_norm = temp_norm,
        .temp_proj = temp_proj,
        .temp_attn = temp_attn,
        .temp_softmax = temp_softmax,
        .temp_out = temp_out,
        .seq_len = seq_len,
        .num_tiles = num_tiles
    };
    
    // Compute reference
    printf("Computing reference...\n");
    LlamaUserData ref_data = user_data;
    ref_data.output = reference;
    llama_reference(&ref_data);
    
    // Execute with runtime_entry_arm64
    printf("Executing with runtime_entry_arm64...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int result = runtime_entry_arm64(llama_orchestration, &user_data, num_workers, threshold);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    if (result != 0) {
        fprintf(stderr, "ERROR: runtime_entry_arm64 failed\n");
        return 1;
    }
    
    // Verify results
    printf("\nVerifying results...\n");
    int errors = verify_llama(output, reference, total_elements);
    
    printf("\n================================================================================\n");
    printf("Results\n");
    printf("================================================================================\n");
    printf("  Execution time:  %.2f ms\n", elapsed_ms);
    printf("  Tasks/second:    %.0f\n", (num_tiles * 4) / (elapsed_ms / 1000.0));
    printf("  Verification:    %s\n", errors == 0 ? "PASSED ✓" : "FAILED ✗");
    if (errors > 0) {
        printf("  Errors:          %d / %d elements\n", errors, total_elements);
    }
    printf("================================================================================\n");
    
    // Print sample output
    printf("\nSample output (first tile, rows 0-2):\n");
    for (int row = 0; row < 3 && row < TILE_ROWS; row++) {
        printf("  Row %d: [", row);
        for (int col = 0; col < 8; col++) {
            printf("%.4f%s", output[row * HIDDEN_DIM + col], col < 7 ? ", " : "");
        }
        printf(", ...]\n");
    }
    
    printf("\nReference (first tile, rows 0-2):\n");
    for (int row = 0; row < 3 && row < TILE_ROWS; row++) {
        printf("  Row %d: [", row);
        for (int col = 0; col < 8; col++) {
            printf("%.4f%s", reference[row * HIDDEN_DIM + col], col < 7 ? ", " : "");
        }
        printf(", ...]\n");
    }
    
    // Cleanup
    free(input);
    free(output);
    free(reference);
    free(norm_weight);
    free(proj_weight);
    free(temp_norm);
    free(temp_proj);
    free(temp_attn);
    free(temp_softmax);
    free(temp_out);
    
    return errors == 0 ? 0 : 1;
}
