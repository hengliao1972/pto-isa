// =============================================================================
// PTO ISA InCore Function: tile_add
// Core Type: Vector
// =============================================================================

#include "pto/common/pto_tile.hpp"
#include "pto/common/pto_instr.hpp"

using namespace pto;

__aicore__ void tile_add(__gm__ float* A, __gm__ float* B, __gm__ float* C) {
    Tile<float, 64, 128> a;
    Tile<float, 64, 128> b;
    Tile<float, 64, 128> c;

    GlobalTensor<float, Shape<64, 64>, Stride<64, 1>> g_A(A);
    GlobalTensor<float, Shape<64, 64>, Stride<64, 1>> g_B(B);
    GlobalTensor<float, Shape<64, 64>, Stride<64, 1>> g_C(C);

    // TLOAD: a = load(A)
    TLOAD(a, g_A);
    // TLOAD: b = load(B)
    TLOAD(b, g_B);
    // TADD: c = a op b
    TADD(c, a, b);
    // TSTORE: store(c) -> C
    TSTORE(g_C, c);
}

// Cycle cost estimation for simulation
extern "C" int64_t tile_add_cycle_cost(void** args, int num_args) {
    // Estimate based on tile sizes and operations
    return get_incore_cycle_cost_sim("tile_add", 8192);
}

// C wrapper for runtime task execution
extern "C" void tile_add_wrapper(void** args, int num_args) {
    tile_add((float*)args[0], (float*)args[1], (float*)args[2]);
}