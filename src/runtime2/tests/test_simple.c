#include <stdio.h>
#include <stdlib.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

typedef struct {
    int batch, m_tiles, n_tiles, k_tiles;
    float *A, *B, *C, *P;
    int task_count;
} BgemmParams;

void bgemm_orch(PTO2Runtime* rt, void* arg) {
    BgemmParams* p = (BgemmParams*)arg;
    
    for (int b = 0; b < p->batch; b++) {
        pto2_rt_scope_begin(rt);
        for (int m = 0; m < p->m_tiles; m++) {
            for (int n = 0; n < p->n_tiles; n++) {
                pto2_rt_scope_begin(rt);
                for (int k = 0; k < p->k_tiles; k++) {
                    int idx = b*64 + m*8 + n;
                    
                    PTO2TaskParam gemm_p[3] = {
                        PTO2_INPUT(p->A, idx, 128),
                        PTO2_INPUT(p->B, idx, 128),
                        PTO2_OUTPUT(p->P, idx, 128)
                    };
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_p, 3);
                    p->task_count++;
                    
                    PTO2TaskParam add_p[3] = {
                        PTO2_INPUT(p->C, idx, 128),
                        PTO2_INPUT(p->P, idx, 128),
                        PTO2_OUTPUT(p->C, idx, 128)
                    };
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_p, 3);
                    p->task_count++;
                }
                pto2_rt_scope_end(rt);
            }
        }
        pto2_rt_scope_end(rt);
    }
}

int main(int argc, char** argv) {
    int batch = 8, m = 8, n = 8, k = 4;
    if (argc > 1) batch = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) n = atoi(argv[3]);
    if (argc > 4) k = atoi(argv[4]);
    
    int total = batch * m * n * k * 2;
    printf("BGEMM: %dx%dx%dx%d = %d tasks\n", batch, m, n, k, total);
    fflush(stdout);
    
    printf("Creating runtime...\n"); fflush(stdout);
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(4, 4, true);
    
    BgemmParams params = {batch, m, n, k, 
                          calloc(4096, sizeof(float)),
                          calloc(4096, sizeof(float)),
                          calloc(4096, sizeof(float)),
                          calloc(4096, sizeof(float)),
                          0};
    
    printf("Running...\n"); fflush(stdout);
    pto2_runtime_run_threaded(rt, bgemm_orch, &params);
    
    printf("Done! %d tasks completed\n", params.task_count);
    printf("Consumed: %lld\n", (long long)rt->base.scheduler.tasks_consumed);
    
    pto2_runtime_write_trace(rt, "trace.json");
    pto2_runtime_destroy_threaded(rt);
    
    return 0;
}
