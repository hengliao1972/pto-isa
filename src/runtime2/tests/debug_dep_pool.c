#include <stdio.h>
#include <stdlib.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_orchestrator.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2DepListPool* pool = &base->orchestrator.dep_pool;
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    // Submit tasks and track pool usage
    int max_top = 0;
    for (int batch = 0; batch < 8; batch++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 8; k++) {
                    int a_idx = batch * 64 + m * 8 + k;
                    int b_idx = batch * 64 + k * 8 + n;
                    int c_idx = batch * 64 + m * 8 + n;
                    
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(A, a_idx, 128),
                        PTO2_INPUT(B, b_idx, 128),
                        PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
        
        if (pool->top > max_top) max_top = pool->top;
    }
    
    printf("Dep pool: max_top=%d, capacity=%d (%.1f%% used)\n",
           max_top, pool->capacity, 100.0 * max_top / pool->capacity);
    
    // Check a few task fanout lists
    printf("\nSample fanout lists:\n");
    for (int tid = 0; tid < 8192; tid += 1000) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, tid);
        printf("  Task %d: fanout_head=%d, fanout_count=%d", tid, t->fanout_head, t->fanout_count);
        
        if (t->fanout_head > 0) {
            int head = t->fanout_head;
            int count = 0;
            while (head > 0 && count < 5) {
                PTO2DepListEntry* e = pto2_dep_pool_get(pool, head);
                if (!e) {
                    printf(" [NULL at %d!]", head);
                    break;
                }
                printf(" -> %d", e->task_id);
                head = e->next_offset;
                count++;
            }
        }
        printf("\n");
    }
    
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
