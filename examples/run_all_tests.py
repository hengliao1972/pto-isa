#!/usr/bin/env python3
"""
PTO ARM64 Test Runner

Runs all test suites for PTO-generated ARM64 NEON code.

Usage:
    python run_all_tests.py           # Run all tests
    python run_all_tests.py --quick   # Run quick tests only
    python run_all_tests.py --verbose # Verbose output
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_utils import IS_ARM64


# =============================================================================
# Test Suite Definitions
# =============================================================================

TEST_SUITES = [
    {
        "name": "sinh_taylor",
        "module": "test_pto_isa_sinh",
        "description": "sinh() Taylor expansion",
        "quick": True,
    },
    {
        "name": "aten_primitives",
        "module": "test_pto_aten_ir_primitives",
        "description": "ATen IR primitives (abs, exp, log, etc.)",
        "quick": True,
    },
    {
        "name": "torch_tensor",
        "module": "test_pto_torch_tensor",
        "description": "torch.Tensor methods",
        "quick": False,
    },
    {
        "name": "torch_functional",
        "module": "test_pto_torch_functional",
        "description": "torch.nn.functional APIs",
        "quick": False,
    },
    {
        "name": "torch_nn",
        "module": "test_pto_torch_nn_operators",
        "description": "torch.nn module classes",
        "quick": False,
    },
    {
        "name": "flexattention",
        "module": "test_pto_torch_flexattention",
        "description": "FlexAttention mechanisms",
        "quick": False,
    },
]


# =============================================================================
# Test Runner
# =============================================================================

def run_suite(suite: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run a single test suite."""
    module_name = suite["module"]
    
    try:
        # Import the test module
        module = __import__(module_name)
        
        # Run the tests
        start_time = time.time()
        results = module.run_all_tests()
        elapsed = time.time() - start_time
        
        results["elapsed"] = elapsed
        results["suite_name"] = suite["name"]
        return results
        
    except FileNotFoundError as e:
        print(f"  [SKIP] {suite['name']}: {e}")
        return {
            "suite_name": suite["name"],
            "passed": 0,
            "failed": 0,
            "total": 0,
            "elapsed": 0,
            "skipped": True,
            "error": str(e)
        }
    except Exception as e:
        print(f"  [ERROR] {suite['name']}: {e}")
        return {
            "suite_name": suite["name"],
            "passed": 0,
            "failed": 1,
            "total": 1,
            "elapsed": 0,
            "error": str(e)
        }


def run_all_tests(quick: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Run all test suites."""
    print("=" * 70)
    print("PTO ARM64 NEON Test Runner")
    print("=" * 70)
    print()
    
    if not IS_ARM64:
        print("WARNING: Not running on ARM64 architecture!")
        print("Tests require ARM64 NEON instructions to work correctly.")
        print("Running anyway, but expect compilation/execution failures.")
        print()
    
    # Filter suites if quick mode
    suites = TEST_SUITES
    if quick:
        suites = [s for s in TEST_SUITES if s.get("quick", False)]
        print(f"Quick mode: running {len(suites)} test suites")
    else:
        print(f"Full mode: running {len(suites)} test suites")
    print()
    
    # Run each suite
    all_results = []
    total_passed = 0
    total_failed = 0
    total_tests = 0
    total_time = 0.0
    
    for suite in suites:
        print(f"{'='*70}")
        print(f"Running: {suite['description']}")
        print(f"{'='*70}")
        print()
        
        results = run_suite(suite, verbose)
        all_results.append(results)
        
        if "skipped" not in results:
            total_passed += results.get("passed", 0)
            total_failed += results.get("failed", 0)
            total_tests += results.get("total", 0)
            total_time += results.get("elapsed", 0)
        
        print()
    
    # Print summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()
    
    for results in all_results:
        name = results["suite_name"]
        if "skipped" in results:
            print(f"  {name:25} [SKIPPED]")
        elif "error" in results and results.get("total", 0) == 0:
            print(f"  {name:25} [ERROR] {results['error']}")
        else:
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            total = results.get("total", 0)
            elapsed = results.get("elapsed", 0)
            status = "PASS" if failed == 0 else "FAIL"
            print(f"  {name:25} [{status}] {passed}/{total} tests ({elapsed:.2f}s)")
    
    print()
    print(f"Total: {total_passed} passed, {total_failed} failed, {total_tests} tests")
    print(f"Time:  {total_time:.2f} seconds")
    print("=" * 70)
    
    return {
        "suites": all_results,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_tests": total_tests,
        "total_time": total_time,
        "success": total_failed == 0
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PTO ARM64 Test Runner")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick tests only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--suite", "-s", type=str,
                       help="Run specific test suite only")
    args = parser.parse_args()
    
    if args.suite:
        # Run specific suite
        suite = next((s for s in TEST_SUITES if s["name"] == args.suite), None)
        if suite is None:
            print(f"Unknown test suite: {args.suite}")
            print(f"Available suites: {', '.join(s['name'] for s in TEST_SUITES)}")
            sys.exit(1)
        
        results = run_suite(suite, args.verbose)
        sys.exit(0 if results.get("failed", 1) == 0 else 1)
    else:
        # Run all suites
        results = run_all_tests(quick=args.quick, verbose=args.verbose)
        sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
