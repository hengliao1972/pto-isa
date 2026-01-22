#!/usr/bin/env python3
"""
Test program: Matrix Multiplication with Nested Loops

This creates an InCore function with nested loops to perform tiled matrix 
multiplication for arbitrary size input matrices A and B.

Matrix multiplication: C[M,N] = A[M,K] * B[K,N]

Full tiled algorithm with M, N, K tiling:
    for i_tile in range(0, M, tile_m):        # Tile over M
        for j_tile in range(0, N, tile_n):    # Tile over N
            C_acc = 0  # Initialize accumulator
            for k_tile in range(0, K, tile_k): # Tile over K
                A_tile = load A[i_tile, k_tile]
                B_tile = load B[k_tile, j_tile]
                C_acc += A_tile @ B_tile  # Accumulate
            store C_acc to C[i_tile, j_tile]

This demonstrates:
- Nested FOR loops in InCore functions (3 levels: M, N, K)
- Loop-carried dependencies (accumulator)
- SimplifyAndColor analysis with cyclic dependencies
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import PTOFunctionBuilder, PTOModule
from pto_isa_definition import ElementType, MemorySpace


def create_full_tiled_matmul(tile_m=16, tile_n=16, tile_k=16, dtype=ElementType.F32):
    """
    Create a fully tiled matrix multiplication with 3 nested loops over M, N, K.
    
    This is the complete tiled GEMM algorithm:
    C[M,N] = A[M,K] * B[K,N]
    
    With tiling:
    - Outer loop over M dimension (i)
    - Middle loop over N dimension (j)  
    - Inner loop over K dimension (k) with accumulation
    
    Each iteration of the inner K loop:
    1. Load A tile from A[i*tile_m : (i+1)*tile_m, k*tile_k : (k+1)*tile_k]
    2. Load B tile from B[k*tile_k : (k+1)*tile_k, j*tile_n : (j+1)*tile_n]
    3. Compute partial matmul
    4. Accumulate into C tile
    
    After K loop completes, store C tile to output.
    """
    return (PTOFunctionBuilder("full_tiled_matmul_MNK")
        .in_core()
        
        # ===== Tile declarations =====
        # Working tiles for A, B blocks
        .tile("A_blk", tile_m, tile_k, dtype)       # Block from A matrix
        .tile("B_blk", tile_k, tile_n, dtype)       # Block from B matrix
        
        # Accumulator and temporary tiles
        .tile("C_acc", tile_m, tile_n, dtype)       # Accumulator for C block
        .tile("C_tmp", tile_m, tile_n, dtype)       # Temp for partial matmul
        .tile("zero", tile_m, tile_n, dtype)        # Zero tile for init
        
        # ===== Memory references =====
        .memref("A", MemorySpace.GM, dtype)         # Input A [M, K]
        .memref("B", MemorySpace.GM, dtype)         # Input B [K, N]
        .memref("C", MemorySpace.GM, dtype)         # Output C [M, N]
        .memref("zeros", MemorySpace.GM, dtype)     # Zero buffer
        
        # ===== Scalar indices and bounds =====
        .scalar("i", ElementType.I32)               # M dimension tile index
        .scalar("j", ElementType.I32)               # N dimension tile index
        .scalar("k", ElementType.I32)               # K dimension tile index
        .scalar("M_tiles", ElementType.I32)         # Number of tiles in M
        .scalar("N_tiles", ElementType.I32)         # Number of tiles in N
        .scalar("K_tiles", ElementType.I32)         # Number of tiles in K
        
        # ===== Outer loop: M dimension =====
        .for_loop("i", 0, "M_tiles", 1)
        
            # ===== Middle loop: N dimension =====
            .for_loop("j", 0, "N_tiles", 1)
            
                # Initialize C accumulator to zero
                .load("zero", "zeros", 0, 0)
                .add("C_acc", "zero", "zero")       # C_acc = 0
                
                # ===== Inner loop: K dimension (accumulation) =====
                .for_loop("k", 0, "K_tiles", 1)
                
                    # Load A block: A[i, k]
                    .load("A_blk", "A", "i", "k")
                    
                    # Load B block: B[k, j]
                    .load("B_blk", "B", "k", "j")
                    
                    # Compute partial matmul: C_tmp = A_blk @ B_blk
                    .matmul("C_tmp", "A_blk", "B_blk")
                    
                    # Accumulate: C_acc += C_tmp (LOOP-CARRIED DEPENDENCY)
                    .add("C_acc", "C_acc", "C_tmp")
                    
                .end_for()  # End K loop
                
                # Store accumulated result to C[i, j]
                .store("C_acc", "C", "i", "j")
                
            .end_for()  # End N loop
            
        .end_for()  # End M loop
        .build())


def create_gemm_with_epilogue(tile_m=16, tile_n=16, tile_k=16, dtype=ElementType.F32):
    """
    GEMM with epilogue operations: D = alpha * (A @ B) + beta * C
    
    This demonstrates a more complex pattern with:
    - 3 nested loops (M, N, K tiling)
    - Accumulation in K loop
    - Post-loop epilogue operations (scaling, bias add)
    - Multiple loop-carried dependencies
    """
    return (PTOFunctionBuilder("gemm_MNK_epilogue")
        .in_core()
        
        # Working tiles
        .tile("A_blk", tile_m, tile_k, dtype)
        .tile("B_blk", tile_k, tile_n, dtype)
        .tile("C_blk", tile_m, tile_n, dtype)       # Original C for beta*C
        .tile("acc", tile_m, tile_n, dtype)         # Accumulator
        .tile("prod", tile_m, tile_n, dtype)        # Matmul result
        .tile("scaled_acc", tile_m, tile_n, dtype)  # alpha * acc
        .tile("scaled_C", tile_m, tile_n, dtype)    # beta * C
        .tile("result", tile_m, tile_n, dtype)      # Final result
        .tile("zero", tile_m, tile_n, dtype)
        
        # Memory
        .memref("A", MemorySpace.GM, dtype)
        .memref("B", MemorySpace.GM, dtype)
        .memref("C", MemorySpace.GM, dtype)         # Input C (for beta*C)
        .memref("D", MemorySpace.GM, dtype)         # Output D
        .memref("zeros", MemorySpace.GM, dtype)
        
        # Scalars
        .scalar("i", ElementType.I32)
        .scalar("j", ElementType.I32)
        .scalar("k", ElementType.I32)
        .scalar("M_tiles", ElementType.I32)
        .scalar("N_tiles", ElementType.I32)
        .scalar("K_tiles", ElementType.I32)
        .scalar("alpha", ElementType.F32)
        .scalar("beta", ElementType.F32)
        
        # ===== Loop over M =====
        .for_loop("i", 0, "M_tiles", 1)
        
            # ===== Loop over N =====
            .for_loop("j", 0, "N_tiles", 1)
            
                # Load original C for epilogue
                .load("C_blk", "C", "i", "j")
                
                # Initialize accumulator
                .load("zero", "zeros", 0, 0)
                .add("acc", "zero", "zero")
                
                # ===== Loop over K (accumulation) =====
                .for_loop("k", 0, "K_tiles", 1)
                    .load("A_blk", "A", "i", "k")
                    .load("B_blk", "B", "k", "j")
                    .matmul("prod", "A_blk", "B_blk")
                    .add("acc", "acc", "prod")       # Loop-carried
                .end_for()
                
                # ===== Epilogue: D = alpha * acc + beta * C =====
                .muls("scaled_acc", "acc", "alpha")
                .muls("scaled_C", "C_blk", "beta")
                .add("result", "scaled_acc", "scaled_C")
                
                # Store result
                .store("result", "D", "i", "j")
                
            .end_for()
        .end_for()
        .build())


def create_batched_matmul(tile_m=16, tile_n=16, tile_k=16, dtype=ElementType.F32):
    """
    Batched matrix multiplication with 4 nested loops: Batch, M, N, K
    
    C[b, i, j] = sum_k(A[b, i, k] * B[b, k, j])
    
    This is even more complex with 4 levels of nesting.
    """
    return (PTOFunctionBuilder("batched_matmul_BMNK")
        .in_core()
        
        # Tiles
        .tile("A_blk", tile_m, tile_k, dtype)
        .tile("B_blk", tile_k, tile_n, dtype)
        .tile("C_acc", tile_m, tile_n, dtype)
        .tile("prod", tile_m, tile_n, dtype)
        .tile("zero", tile_m, tile_n, dtype)
        
        # Memory (3D tensors flattened)
        .memref("A", MemorySpace.GM, dtype)         # [B, M, K]
        .memref("B", MemorySpace.GM, dtype)         # [B, K, N]
        .memref("C", MemorySpace.GM, dtype)         # [B, M, N]
        .memref("zeros", MemorySpace.GM, dtype)
        
        # Scalars
        .scalar("b", ElementType.I32)               # Batch index
        .scalar("i", ElementType.I32)               # M index
        .scalar("j", ElementType.I32)               # N index
        .scalar("k", ElementType.I32)               # K index
        .scalar("B_size", ElementType.I32)          # Batch size
        .scalar("M_tiles", ElementType.I32)
        .scalar("N_tiles", ElementType.I32)
        .scalar("K_tiles", ElementType.I32)
        
        # ===== Loop over Batch =====
        .for_loop("b", 0, "B_size", 1)
        
            # ===== Loop over M =====
            .for_loop("i", 0, "M_tiles", 1)
            
                # ===== Loop over N =====
                .for_loop("j", 0, "N_tiles", 1)
                
                    # Initialize accumulator
                    .load("zero", "zeros", 0, 0)
                    .add("C_acc", "zero", "zero")
                    
                    # ===== Loop over K =====
                    .for_loop("k", 0, "K_tiles", 1)
                        .load("A_blk", "A", "i", "k")    # Simplified indexing
                        .load("B_blk", "B", "k", "j")
                        .matmul("prod", "A_blk", "B_blk")
                        .add("C_acc", "C_acc", "prod")   # Loop-carried
                    .end_for()
                    
                    # Store result
                    .store("C_acc", "C", "i", "j")
                    
                .end_for()
            .end_for()
        .end_for()
        .build())


def create_matmul_with_softmax(tile_m=16, tile_n=16, tile_k=16, dtype=ElementType.F32):
    """
    Attention-style: Softmax(A @ B) @ V
    
    This has two matmuls with softmax in between:
    1. S = A @ B (scores)
    2. P = softmax(S) row-wise
    3. O = P @ V (output)
    
    Complex dependencies between the operations.
    """
    return (PTOFunctionBuilder("matmul_softmax_matmul")
        .in_core()
        
        # Tiles for first matmul (scores)
        .tile("Q_blk", tile_m, tile_k, dtype)       # Query
        .tile("K_blk", tile_k, tile_n, dtype)       # Key (transposed)
        .tile("S_acc", tile_m, tile_n, dtype)       # Score accumulator
        .tile("S_prod", tile_m, tile_n, dtype)      # Partial score
        
        # Tiles for softmax
        .tile("S_max", tile_m, 1, dtype)            # Row max
        .tile("S_shifted", tile_m, tile_n, dtype)   # S - max
        .tile("S_exp", tile_m, tile_n, dtype)       # exp(S - max)
        .tile("S_sum", tile_m, 1, dtype)            # Row sum
        .tile("P", tile_m, tile_n, dtype)           # Attention weights
        
        # Tiles for second matmul (output)
        .tile("V_blk", tile_n, tile_k, dtype)       # Value
        .tile("O_acc", tile_m, tile_k, dtype)       # Output accumulator
        .tile("O_prod", tile_m, tile_k, dtype)      # Partial output
        
        .tile("zero_mn", tile_m, tile_n, dtype)
        .tile("zero_mk", tile_m, tile_k, dtype)
        
        # Memory
        .memref("Q", MemorySpace.GM, dtype)
        .memref("K", MemorySpace.GM, dtype)
        .memref("V", MemorySpace.GM, dtype)
        .memref("O", MemorySpace.GM, dtype)
        .memref("zeros", MemorySpace.GM, dtype)
        
        # Scalars
        .scalar("i", ElementType.I32)
        .scalar("j", ElementType.I32)
        .scalar("k", ElementType.I32)
        .scalar("M_tiles", ElementType.I32)
        .scalar("N_tiles", ElementType.I32)
        .scalar("K_tiles", ElementType.I32)
        
        # ===== Loop over query blocks (M) =====
        .for_loop("i", 0, "M_tiles", 1)
        
            # Initialize output accumulator
            .load("zero_mk", "zeros", 0, 0)
            .add("O_acc", "zero_mk", "zero_mk")
        
            # ===== Loop over key blocks (N) =====
            .for_loop("j", 0, "N_tiles", 1)
            
                # --- Compute S = Q @ K^T ---
                .load("zero_mn", "zeros", 0, 0)
                .add("S_acc", "zero_mn", "zero_mn")
                
                .for_loop("k", 0, "K_tiles", 1)
                    .load("Q_blk", "Q", "i", "k")
                    .load("K_blk", "K", "j", "k")    # K transposed
                    .matmul("S_prod", "Q_blk", "K_blk")
                    .add("S_acc", "S_acc", "S_prod") # Loop-carried
                .end_for()
                
                # --- Compute softmax(S) ---
                .rowmax("S_max", "S_acc")
                .rowexpandsub("S_shifted", "S_acc", "S_max")
                .exp("S_exp", "S_shifted")
                .rowsum("S_sum", "S_exp")
                .rowexpanddiv("P", "S_exp", "S_sum")
                
                # --- Accumulate O += P @ V ---
                .load("V_blk", "V", "j", 0)
                .matmul("O_prod", "P", "V_blk")
                .add("O_acc", "O_acc", "O_prod")     # Loop-carried across j
                
            .end_for()
            
            # Store output
            .store("O_acc", "O", "i", 0)
            
        .end_for()
        .build())


def main():
    """Run SimplifyAndColor on matmul programs."""
    
    output_dir = os.path.join(os.path.dirname(__file__), "output_matmul_coloring")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Matrix Multiplication with Nested Loops - SimplifyAndColor Analysis")
    print("=" * 70)
    
    # Test programs - from simple to complex
    test_programs = [
        ("full_tiled_matmul_MNK", create_full_tiled_matmul, 
         "Full tiled GEMM with 3 nested loops (M, N, K)"),
        ("gemm_MNK_epilogue", create_gemm_with_epilogue, 
         "GEMM with epilogue: D = alpha*(A@B) + beta*C"),
        ("batched_matmul_BMNK", create_batched_matmul, 
         "Batched matmul with 4 nested loops (B, M, N, K)"),
        ("matmul_softmax_matmul", create_matmul_with_softmax, 
         "Attention: Softmax(Q@K) @ V with nested loops"),
    ]
    
    results = []
    
    for name, create_func, description in test_programs:
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"Description: {description}")
        print("=" * 70)
        
        # Create program
        program = create_func()
        num_instrs = len(program.instructions)
        
        print(f"Instructions: {num_instrs}")
        
        # Create output directory for this program
        prog_output_dir = os.path.join(output_dir, name)
        os.makedirs(prog_output_dir, exist_ok=True)
        
        # Run SimplifyAndColor
        print("\nRunning SimplifyAndColor...")
        success = program.SimplifyAndColor(
            TOTAL_COLOR=8,
            output_dir=prog_output_dir,
            visualize=True,
            verbose=True
        )
        
        # Dump PTO assembly
        pto_file = os.path.join(output_dir, f"{name}_colored.pto")
        asm = program.dump_pto_asm_with_deps(pto_file)
        
        # Collect statistics
        max_degree = max((instr.get_degree() for instr in program.instructions), default=0)
        colors_used = len(set(instr.color for instr in program.instructions if instr.color >= 0))
        
        # Count loop-carried deps
        loop_carried = sum(1 for instr in program.instructions if instr.fanin_succ)
        
        results.append({
            'name': name,
            'instructions': num_instrs,
            'success': success,
            'max_degree': max_degree,
            'colors_used': colors_used,
            'loop_carried_deps': loop_carried
        })
        
        print(f"\nResults:")
        print(f"  Success: {success}")
        print(f"  Max degree: {max_degree}")
        print(f"  Colors used: {colors_used}")
        print(f"  Instructions with loop-carried deps: {loop_carried}")
        print(f"  PDF directory: {prog_output_dir}/")
        print(f"  PTO assembly: {pto_file}")
        
        print(f"\n--- PTO Assembly ({name}) ---")
        print(asm)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print("\n{:<25} {:>8} {:>8} {:>8} {:>12}".format(
        "Program", "Instrs", "MaxDeg", "Colors", "LoopCarried"))
    print("-" * 65)
    for r in results:
        status = "✓" if r['success'] else "✗"
        print("{:<25} {:>8} {:>8} {:>8} {:>12} {}".format(
            r['name'], r['instructions'], r['max_degree'], 
            r['colors_used'], r['loop_carried_deps'], status))
    
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)
    
    # Open the most complex one (attention with softmax)
    comparison_pdf = os.path.join(output_dir, "matmul_softmax_matmul", 
                                   "matmul_softmax_matmul_deps_colored.pdf")
    if os.path.exists(comparison_pdf):
        print(f"\nOpening: {comparison_pdf}")
        os.system(f"open '{comparison_pdf}'")


if __name__ == "__main__":
    main()
