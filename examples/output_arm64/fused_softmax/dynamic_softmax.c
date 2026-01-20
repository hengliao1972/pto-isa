// PTO Program: dynamic_softmax
// Function Type: Orchestration (control flow only)
// Orchestration function - builds task graph using PTO runtime
#include "pto_runtime.h"
#include "pto_runtime.c"  // Include for standalone build

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void dynamic_softmax(PTORuntime* rt, float* input, float* output, float* temp_rowmax, float* temp_shifted, float* temp_exp, float* temp_rowsum, int32_t total_rows, int32_t num_full_tiles, int32_t tail_rows) {

    // Loop fusion: 0 loop overheads saved

    int tile_rows = 8;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // Task 0: rowmax
        int32_t t0 = pto_task_alloc(rt, "rowmax", NULL, 294912, 294912);
        pto_task_add_input(rt, t0, input, 0, 0, 8, 8);
        pto_task_add_output(rt, t0, temp_rowmax, 0, 0, 8, 1);
        pto_task_submit(rt, t0);


        // Task 1: rowexpandsub
        int32_t t1 = pto_task_alloc(rt, "rowexpandsub", NULL, 557056, 557056);
        pto_task_add_input(rt, t1, input, 0, 0, 8, 8);
        pto_task_add_input(rt, t1, temp_rowmax, 0, 0, 8, 8);
        pto_task_add_output(rt, t1, temp_shifted, 0, 0, 8, 8);
        pto_task_submit(rt, t1);


        // Task 2: elem_exp
        int32_t t2 = pto_task_alloc(rt, "elem_exp", NULL, 524288, 524288);
        pto_task_add_input(rt, t2, temp_shifted, 0, 0, 8, 8);
        pto_task_add_output(rt, t2, temp_exp, 0, 0, 8, 8);
        pto_task_submit(rt, t2);


        // Task 3: rowsum
        int32_t t3 = pto_task_alloc(rt, "rowsum", NULL, 294912, 294912);
        pto_task_add_input(rt, t3, temp_exp, 0, 0, 8, 8);
        pto_task_add_output(rt, t3, temp_rowsum, 0, 0, 8, 1);
        pto_task_submit(rt, t3);


        // Task 4: rowexpanddiv
        int32_t t4 = pto_task_alloc(rt, "rowexpanddiv", NULL, 557056, 557056);
        pto_task_add_input(rt, t4, temp_exp, 0, 0, 8, 8);
        pto_task_add_input(rt, t4, temp_rowsum, 0, 0, 8, 8);
        pto_task_add_output(rt, t4, output, 0, 0, 8, 8);
        pto_task_submit(rt, t4);


    }

    int has_tail = (tail_rows > zero) ? 1 : 0;

    if (has_tail) {

        // Task 5: rowmax
        int32_t t5 = pto_task_alloc(rt, "rowmax", NULL, 294912, 294912);
        pto_task_add_input(rt, t5, input, 0, 0, 8, 8);
        pto_task_add_output(rt, t5, temp_rowmax, 0, 0, 8, 1);
        pto_task_submit(rt, t5);


        // Task 6: rowexpandsub
        int32_t t6 = pto_task_alloc(rt, "rowexpandsub", NULL, 557056, 557056);
        pto_task_add_input(rt, t6, input, 0, 0, 8, 8);
        pto_task_add_input(rt, t6, temp_rowmax, 0, 0, 8, 8);
        pto_task_add_output(rt, t6, temp_shifted, 0, 0, 8, 8);
        pto_task_submit(rt, t6);


        // Task 7: elem_exp
        int32_t t7 = pto_task_alloc(rt, "elem_exp", NULL, 524288, 524288);
        pto_task_add_input(rt, t7, temp_shifted, 0, 0, 8, 8);
        pto_task_add_output(rt, t7, temp_exp, 0, 0, 8, 8);
        pto_task_submit(rt, t7);


        // Task 8: rowsum
        int32_t t8 = pto_task_alloc(rt, "rowsum", NULL, 294912, 294912);
        pto_task_add_input(rt, t8, temp_exp, 0, 0, 8, 8);
        pto_task_add_output(rt, t8, temp_rowsum, 0, 0, 8, 1);
        pto_task_submit(rt, t8);


        // Task 9: rowexpanddiv
        int32_t t9 = pto_task_alloc(rt, "rowexpanddiv", NULL, 557056, 557056);
        pto_task_add_input(rt, t9, temp_exp, 0, 0, 8, 8);
        pto_task_add_input(rt, t9, temp_rowsum, 0, 0, 8, 8);
        pto_task_add_output(rt, t9, output, 0, 0, 8, 8);
        pto_task_submit(rt, t9);


    }

}