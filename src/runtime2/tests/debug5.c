#include <stdio.h>
#include <stdlib.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

void orch_func(PTO2Runtime* rt, void* arg) {
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    for (int b = 0; b < 4; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 500; i++) {
            int idx = b * 500 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_OUTPUT(P, idx % 1024, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
        printf("Batch %d done, last_alive=%d\n", b, rt->sm_handle->header->last_task_alive);
        fflush(stdout);
    }
    
    free(A); free(P);
}

int main() {
    printf("Creating runtime with window=2048...\n");
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 2048, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    pto2_runtime_run_threaded(rt, orch_func, NULL);
    
    printf("Done! consumed=%lld\n", (long long)rt->base.scheduler.tasks_consumed);
    pto2_runtime_destroy_threaded(rt);
    return 0;
}
