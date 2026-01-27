/**
 * KernelArgs Structure - Shared between Host, AICPU, and AICore
 *
 * NOTE:
 * The built-in AICPU launcher (`libaicpu_extend_kernels.so` + DynTileFwkKernelServer*)
 * expects the argument layout of `DeviceKernelArgs` and `DeviceArgs` (cfgdata),
 * matching the TileFwk / PyPTO ABI.
 *
 * This repo uses only a minimal subset of that ABI:
 * - `DeviceArgs.aicpuSoBin/aicpuSoLen` carries the in-device bytes of the backend server .so
 * - `DeviceArgs.opaque` carries a pointer to `PtoRuntimeArgs` (our own mailbox)
 * - `PtoRuntimeArgs` carries the runtime pointers for the simplified graph scheduler:
 *   handshake array + graph pointer.
 */

#ifndef RUNTIME_COMMON_KERNEL_ARGS_H
#define RUNTIME_COMMON_KERNEL_ARGS_H

#include <cstdint>

class Graph;
struct Handshake;

// Keep these definitions ABI-compatible with the TileFwk layout used by the
// built-in kernel servers on Ascend devices.

// Minimal profiling config (kept for layout compatibility).
struct ToSubMachineConfig {
    uint32_t profConfig{0};
};

struct OpMetaAddrs {
    uint64_t generalAddr{0};
    uint64_t stitchPoolAddr{0};
};

enum class ArchInfo {
    DAV_1001 = 1001,
    DAV_2201 = 2201,
    DAV_3510 = 3510,
    DAV_UNKNOWN,
};

// Device-side args blob (`cfgdata` in DeviceKernelArgs).
// This matches the layout used by TileFwk's dynamic launchers (PyPTO).
struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};
    uint64_t devQueueAddr{0};
    uint64_t sharedBuffer{0};
    uint64_t coreRegAddr{0};
    uint64_t corePmuRegAddr{0};
    uint64_t corePmuAddr{0};
    uint64_t pmuEventAddr{0};
    uint64_t taskType : 4;
    uint64_t machineConfig : 8;
    uint64_t taskId : 52;
    uint64_t taskData{0};
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
    uint64_t deviceId{0};
    uint64_t startArgsAddr{0};
    uint64_t taskQueue{0};
    uint64_t taskCtrl{0};
    uint32_t scheCpuNum{0};
    uint32_t enableCtrl : 2;
    uint32_t validGetPgMask : 2;
    uint32_t disableSync : 28;
    uint64_t generalAddr{0};
    uint64_t stitchPoolAddr{0};
    uint64_t aicpuPerfAddr{0};
    ArchInfo archInfo{ArchInfo::DAV_2201};
    ToSubMachineConfig toSubMachineConfig{};

    uint64_t GetBlockNum() const {
        if (nrAic == 0) {
            return 0;
        }
        return static_cast<uint64_t>(nrValidAic) * (static_cast<uint64_t>(nrAiv) / static_cast<uint64_t>(nrAic) + 1ULL);
    }
};

// Host->AICPU launch args (first field of the struct passed to rtAicpuKernelLaunchExWithArgs).
// The built-in AICPU launcher passes this pointer through to the backend server.
struct DeviceKernelArgs {
    int64_t *ctrlFlowCache{nullptr};
    int64_t *inputs{nullptr};
    int64_t *outputs{nullptr};
    int64_t *workspace{nullptr};
    int64_t *tilingdata{nullptr};
    int64_t *cfgdata{nullptr};
    void *costmodeldata{nullptr};
    void *aicoreModel{nullptr};
    uint64_t taskWastTime{0};
    uint8_t machineConfig{0};
    ToSubMachineConfig toSubMachineConfig{};
    OpMetaAddrs opMetaAddrs{};
};

// PTO-ISA runtime mailbox (owned by this repo; stored in DeviceArgs.opaque).
struct PtoRuntimeArgs {
    Handshake *hankArgs{nullptr};
    Graph *graphArgs{nullptr};
    int64_t core_num{0};
};

#endif  // RUNTIME_COMMON_KERNEL_ARGS_H
