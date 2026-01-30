#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_scheduler.h"
#include "../pto_ring_buffer.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2DepListPool* dep_pool = &rt->base.orchestrator.dep_pool;
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    // 不启动线程，只检查 fanout list
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
                        PTO2_INPUT(A, a_idx, 128), PTO2_INPUT(B, b_idx, 128), PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128), PTO2_INPUT(P, c_idx, 128), PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    printf("检查 fanout list 是否有重复...\n");
    int dup_count = 0;
    
    for (int i = 0; i < 8192; i++) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        
        // 收集 fanout list 中的所有 consumer
        int consumers[16];
        int num_consumers = 0;
        
        int32_t cur = t->fanout_head;
        while (cur > 0 && num_consumers < 16) {
            PTO2DepListEntry* e = pto2_dep_pool_get(dep_pool, cur);
            if (!e) break;
            consumers[num_consumers++] = e->task_id;
            cur = e->next_offset;
        }
        
        // 检查是否有重复
        for (int j = 0; j < num_consumers; j++) {
            for (int k = j + 1; k < num_consumers; k++) {
                if (consumers[j] == consumers[k]) {
                    if (dup_count < 10) {
                        printf("Task %d has duplicate consumer %d in fanout list!\n", i, consumers[j]);
                    }
                    dup_count++;
                }
            }
        }
    }
    
    printf("总共发现 %d 个重复 consumer\n", dup_count);
    
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
