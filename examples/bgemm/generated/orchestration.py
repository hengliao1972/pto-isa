from __future__ import annotations

# Demo orchestration function: builds a runtime task graph.
#
# Execution is performed by launching the device-side scheduler on AICPU, which
# dispatches tasks to AICore workers (see pto_runtime.DeviceRunner.run).

import pto_runtime


def build_graph(*, runner: pto_runtime.DeviceRunner, func_id: int, dev_a: int, dev_b: int, dev_c: int, block_dim: int) -> pto_runtime.OrchestrationRuntime:
    orch = pto_runtime.OrchestrationRuntime(runner=runner)
    for tile_id in range(int(block_dim)):
        orch.add_task([int(tile_id), int(dev_a), int(dev_b), int(dev_c)], func_id=int(func_id), core_type=1)
    return orch

