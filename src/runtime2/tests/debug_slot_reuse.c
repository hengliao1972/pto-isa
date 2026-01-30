#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_scheduler.h"

int main() {
    printf("检查 slot 计算:\n");
    printf("task_window_size = 16384\n");
    printf("task_window_mask = %d\n", 16384 - 1);
    printf("\n");
    
    // 检查 8192 任务是否有 slot 冲突
    for (int i = 0; i < 8192; i++) {
        int slot = i & (16384 - 1);
        if (slot != i) {
            printf("Task %d has slot %d (DIFFERENT!)\n", i, slot);
        }
    }
    printf("所有 8192 任务的 slot 都等于 task_id，无冲突\n");
    
    return 0;
}
