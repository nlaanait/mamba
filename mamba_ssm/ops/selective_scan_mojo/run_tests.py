#!/usr/bin/env python3
"""
Test runner for Mojo selective scan implementation.
Executes all tests including unit tests, integration tests, and benchmarks.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

# Add the current directory to Python path
this_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(this_dir))


def run_mojo_tests():
    """Run Mojo unit tests."""
    print("=" * 60)
    print("RUNNING MOJO UNIT TESTS")
    print("=" * 60)
    
    test_files = [
        "tests/test_selective_scan_params.mojo",
    ]
    
    results = []
    for test_file in test_files:
        test_path = this_dir / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run(
                    ['mojo', str(test_path)],
                    capture_output=True,
                    text=True,
                    cwd=this_dir
                )
                
                if result.returncode == 0:
                    print("✓ PASSED")
                    results.append((test_file, True, None))
                else:
                    print("✗ FAILED")
                    print(f"Error: {result.stderr}")
                    results.append((test_file, False, result.stderr))
                    
            except Exception as e:
                print(f"✗ ERROR: {e}")
                results.append((test_file, False, str(e)))
        else:
            print(f"⚠ SKIPPED: {test_file} not found")
            results.append((test_file, False, "File not found"))
    
    return results


def run_python_tests():
    """Run Python integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING PYTHON INTEGRATION TESTS")
    print("=" * 60)
    
    test_file = "tests/test_selective_scan_integration.py"
    test_path = this_dir / test_file
    
    if not test_path.exists():
        print(f"⚠ SKIPPED: {test_file} not found")
        return [("python_integration", False, "Test file not found")]
    
    print(f"\nRunning {test_file}...")
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            cwd=this_dir
        )
        
        if result.returncode == 0:
            print("✓ PASSED")
            return [("python_integration", True, None)]
        else:
            print("✗ FAILED")
            print(f"Error: {result.stderr}")
            return [("python_integration", False, result.stderr)]
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return [("python_integration", False, str(e))]


def run_benchmarks(device="cpu", dtype="float32", quick=False):
    """Run performance benchmarks."""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    benchmark_file = "benchmarks/benchmark_selective_scan.py"
    benchmark_path = this_dir / benchmark_file
    
    if not benchmark_path.exists():
        print(f"⚠ SKIPPED: {benchmark_file} not found")
        return [("benchmarks", False, "Benchmark file not found")]
    
    print(f"\nRunning benchmarks on {device} with {dtype}...")
    
    cmd = [
        sys.executable, str(benchmark_path),
        "--device", device,
        "--dtype", dtype,
    ]
    
    if quick:
        cmd.extend(["--warmup", "5", "--runs", "20"])
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=this_dir
        )
        end_time = time.time()
        
        if result.returncode == 0:
            print("✓ BENCHMARKS COMPLETED")
            print(f"  Duration: {end_time - start_time:.2f}s")
            return [("benchmarks", True, None)]
        else:
            print("✗ BENCHMARKS FAILED")
            print(f"Error: {result.stderr}")
            return [("benchmarks", False, result.stderr)]
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return [("benchmarks", False, str(e))]


def run_bazel_tests():
    """Run Bazel tests if available."""
    print("\n" + "=" * 60)
    print("RUNNING BAZEL TESTS")
    print("=" * 60)
    
    # Check if we're in a Bazel environment
    mamba_root = this_dir.parent.parent.parent
    bazel_workspace = mamba_root / "BUILD.bazel"
    
    if not bazel_workspace.exists():
        print("⚠ SKIPPED: Not in a Bazel workspace")
        return [("bazel_tests", False, "Not in Bazel workspace")]
    
    print("Running Bazel tests...")
    try:
        # Test the Mojo library
        result = subprocess.run([
            'bazel', 'test', 
            '//mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_params'
        ], capture_output=True, text=True, cwd=mamba_root)
        
        if result.returncode == 0:
            print("✓ BAZEL TESTS PASSED")
            return [("bazel_tests", True, None)]
        else:
            print("✗ BAZEL TESTS FAILED")
            print(f"Error: {result.stderr}")
            return [("bazel_tests", False, result.stderr)]
            
    except FileNotFoundError:
        print("⚠ SKIPPED: Bazel not found")
        return [("bazel_tests", False, "Bazel not found")]
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return [("bazel_tests", False, str(e))]


def print_summary(all_results):
    """Print a summary of all test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, success, error in all_results:
        if success:
            status = "✓ PASSED"
            passed += 1
        elif error and "not found" in error:
            status = "⚠ SKIPPED"
            skipped += 1
        else:
            status = "✗ FAILED"
            failed += 1
        
        print(f"{test_name:<25} {status}")
        if error and "not found" not in error:
            print(f"{'':25}   {error[:100]}...")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Mojo selective scan implementation")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run benchmarks on")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                       help="Data type for benchmarks")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks (fewer iterations)")
    parser.add_argument("--skip-benchmarks", action="store_true",
                       help="Skip performance benchmarks")
    parser.add_argument("--skip-bazel", action="store_true",
                       help="Skip Bazel tests")
    
    args = parser.parse_args()
    
    print("Mojo Selective Scan Test Runner")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data Type: {args.dtype}")
    print(f"Quick Mode: {args.quick}")
    
    all_results = []
    
    # Run Mojo unit tests
    mojo_results = run_mojo_tests()
    all_results.extend(mojo_results)
    
    # Run Python integration tests
    python_results = run_python_tests()
    all_results.extend(python_results)
    
    # Run Bazel tests
    if not args.skip_bazel:
        bazel_results = run_bazel_tests()
        all_results.extend(bazel_results)
    
    # Run benchmarks
    if not args.skip_benchmarks:
        benchmark_results = run_benchmarks(args.device, args.dtype, args.quick)
        all_results.extend(benchmark_results)
    
    # Print summary
    success = print_summary(all_results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 