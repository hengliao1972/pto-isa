#!/usr/bin/env python3
"""
Apply SimplifyAndColor algorithm to LLaMA 7B module.

This script:
1. Creates the LLaMA 7B module
2. Applies dependency analysis and graph coloring to each function
3. Generates colored PDF visualizations
4. Dumps PTO assembly code with dependency info
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_llama7B_dynamic import create_llama7b_module


def main():
    """Run SimplifyAndColor on all LLaMA functions."""
    
    # Output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_output_dir = os.path.join(script_dir, "output_pto", "llama7b_colored")
    pto_output_dir = os.path.join(script_dir, "output_pto", "llama7b_colored")
    
    os.makedirs(pdf_output_dir, exist_ok=True)
    os.makedirs(pto_output_dir, exist_ok=True)
    
    print("=" * 70)
    print("LLaMA 7B SimplifyAndColor Analysis")
    print("=" * 70)
    
    # Create the LLaMA module
    print("\nCreating LLaMA 7B module...")
    module = create_llama7b_module()
    
    print(f"\nModule: {module.name}")
    print(f"Total functions: {len(module.functions)}")
    print(f"Entry function: {module.entry_function}")
    
    # Categorize functions
    incore_funcs = [f for f in module.functions.values() if f.is_in_core]
    orch_funcs = [f for f in module.functions.values() if not f.is_in_core]
    
    print(f"\n  InCore functions: {len(incore_funcs)}")
    print(f"  Orchestration functions: {len(orch_funcs)}")
    
    # Process each function
    results = []
    
    print("\n" + "-" * 70)
    print("Processing functions...")
    print("-" * 70)
    
    for func_name, program in module.functions.items():
        func_type = "InCore" if program.is_in_core else "Orchestration"
        num_instrs = len(program.instructions)
        
        print(f"\n[{func_name}] ({func_type}, {num_instrs} instructions)")
        
        if num_instrs == 0:
            print(f"  Skipping: No instructions")
            continue
        
        # Create function-specific output directory
        func_output_dir = os.path.join(pdf_output_dir, func_name)
        os.makedirs(func_output_dir, exist_ok=True)
        
        # Apply SimplifyAndColor
        try:
            success = program.SimplifyAndColor(
                TOTAL_COLOR=8,
                output_dir=func_output_dir,
                visualize=True,
                verbose=False  # Set to True for detailed output
            )
            
            # Dump PTO assembly with dependencies
            pto_file = os.path.join(pto_output_dir, f"{func_name}_colored.pto")
            program.dump_pto_asm_with_deps(pto_file)
            
            # Collect statistics
            max_degree = max((instr.get_degree() for instr in program.instructions), default=0)
            colors_used = len(set(instr.color for instr in program.instructions if instr.color >= 0))
            
            results.append({
                'name': func_name,
                'type': func_type,
                'instructions': num_instrs,
                'success': success,
                'max_degree': max_degree,
                'colors_used': colors_used,
                'pdf_dir': func_output_dir,
                'pto_file': pto_file
            })
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {status} - max_degree={max_degree}, colors_used={colors_used}")
            print(f"  PDF: {func_output_dir}/")
            print(f"  PTO: {pto_file}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': func_name,
                'type': func_type,
                'instructions': num_instrs,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\nTotal functions processed: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if results:
        print("\n{:<40} {:>10} {:>8} {:>8} {:>8}".format(
            "Function", "Type", "Instrs", "MaxDeg", "Colors"))
        print("-" * 78)
        for r in results:
            if r.get('success'):
                print("{:<40} {:>10} {:>8} {:>8} {:>8}".format(
                    r['name'][:38], r['type'], r['instructions'], 
                    r['max_degree'], r['colors_used']))
            else:
                print("{:<40} {:>10} {:>8} {:>16}".format(
                    r['name'][:38], r['type'], r['instructions'], "FAILED"))
    
    print(f"\nOutput directory: {pdf_output_dir}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
