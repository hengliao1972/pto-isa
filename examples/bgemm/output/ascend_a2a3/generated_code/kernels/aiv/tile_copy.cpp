// =============================================================================
// PTO ISA InCore Function: tile_copy
// Core Type: Vector
// =============================================================================

#include "pto/common/pto_tile.hpp"
#include "pto/common/pto_instr.hpp"

using namespace pto;

__aicore__ void tile_copy(__gm__ float* A, __gm__ float* C) {
    Tile<float, 64, 128> a;

    GlobalTensor<float, Shape<64, 64>, Stride<64, 1>> g_A(A);
    GlobalTensor<float, Shape<64, 64>, Stride<64, 1>> g_C(C);

    // TLOAD: a = load(A)
    TLOAD(a, g_A);
    // TSTORE: store(a) -> C
    TSTORE(g_C, a);
}

// Cycle cost estimation for simulation
extern "C" int64_t tile_copy_cycle_cost(void** args, int num_args) {
    // Estimate based on tile sizes and operations
    return get_incore_cycle_cost_sim("tile_copy", 8192);
}

// C wrapper for runtime task execution
extern "C" void tile_copy_wrapper(void** args, int num_args) {
    tile_copy((float*)args[0], (float*)args[1]);
}