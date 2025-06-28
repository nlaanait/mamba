#!/usr/bin/env python3
"""
Benchmark script for Mojo selective scan implementation.
Compares performance against reference and CUDA implementations.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from selective_scan_interface import selective_scan_ref, selective_scan_fn
    from selective_scan_mojo.selective_scan_python import selective_scan_mojo_forward
    MOJO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some implementations not available: {e}")
    MOJO_AVAILABLE = False


def create_benchmark_tensors(
    batch_size: int,
    dim: int,
    seq_len: int,
    d_state: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> Tuple[torch.Tensor, ...]:
    """Create tensors for benchmarking."""
    u = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    delta = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    A = torch.randn(dim, d_state, dtype=dtype, device=device)
    B = torch.randn(batch_size, d_state, seq_len, dtype=dtype, device=device)
    C = torch.randn(batch_size, d_state, seq_len, dtype=dtype, device=device)
    D = torch.randn(dim, dtype=dtype, device=device)
    z = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    delta_bias = torch.randn(dim, dtype=torch.float32, device=device)
    
    return u, delta, A, B, C, D, z, delta_bias


def benchmark_implementation(
    name: str,
    func,
    tensors: Tuple[torch.Tensor, ...],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu"
) -> Dict[str, float]:
    """Benchmark a selective scan implementation."""
    u, delta, A, B, C, D, z, delta_bias = tensors
    
    # Warmup runs
    for _ in range(num_warmup):
        try:
            _ = func(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)
        except Exception as e:
            return {"error": str(e)}
    
    # Synchronize if using GPU
    if device != "cpu":
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        try:
            _ = func(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)
        except Exception as e:
            return {"error": str(e)}
        
        if device != "cpu":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "median_time": float(np.median(times)),
        "p95_time": float(np.percentile(times, 95)),
        "p99_time": float(np.percentile(times, 99)),
    }


def benchmark_configurations():
    """Define benchmark configurations."""
    return [
        # Small configurations
        {"batch_size": 1, "dim": 4, "seq_len": 16, "d_state": 4, "name": "small"},
        {"batch_size": 2, "dim": 8, "seq_len": 32, "d_state": 8, "name": "small_medium"},
        
        # Medium configurations
        {"batch_size": 4, "dim": 16, "seq_len": 64, "d_state": 16, "name": "medium"},
        {"batch_size": 8, "dim": 32, "seq_len": 128, "d_state": 32, "name": "medium_large"},
        
        # Large configurations
        {"batch_size": 16, "dim": 64, "seq_len": 256, "d_state": 64, "name": "large"},
        {"batch_size": 32, "dim": 128, "seq_len": 512, "d_state": 128, "name": "xlarge"},
        
        # Mamba-130M like configuration
        {"batch_size": 1, "dim": 256, "seq_len": 1024, "d_state": 16, "name": "mamba_130m"},
        
        # Mamba-370M like configuration
        {"batch_size": 1, "dim": 512, "seq_len": 1024, "d_state": 16, "name": "mamba_370m"},
        
        # Mamba-790M like configuration
        {"batch_size": 1, "dim": 768, "seq_len": 1024, "d_state": 16, "name": "mamba_790m"},
    ]


def run_benchmarks(device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Run comprehensive benchmarks."""
    print(f"Running benchmarks on {device} with {dtype}")
    print("=" * 80)
    
    results = {}
    configs = benchmark_configurations()
    
    for config in configs:
        print(f"\nBenchmarking {config['name']} configuration:")
        print(f"  Batch: {config['batch_size']}, Dim: {config['dim']}, "
              f"Seq: {config['seq_len']}, State: {config['d_state']}")
        
        # Create tensors
        tensors = create_benchmark_tensors(
            config['batch_size'], config['dim'], config['seq_len'], config['d_state'],
            dtype=dtype, device=device
        )
        
        config_results = {}
        
        # Benchmark reference implementation
        print("  Running reference implementation...")
        ref_result = benchmark_implementation(
            "reference", selective_scan_ref, tensors, device=device
        )
        config_results["reference"] = ref_result
        
        # Benchmark CUDA implementation if available
        try:
            print("  Running CUDA implementation...")
            cuda_result = benchmark_implementation(
                "cuda", selective_scan_fn, tensors, device=device
            )
            config_results["cuda"] = cuda_result
        except Exception as e:
            print(f"  CUDA implementation not available: {e}")
        
        # Benchmark Mojo implementation if available
        if MOJO_AVAILABLE:
            try:
                print("  Running Mojo implementation...")
                mojo_result = benchmark_implementation(
                    "mojo", selective_scan_mojo_forward, tensors, device=device
                )
                config_results["mojo"] = mojo_result
            except Exception as e:
                print(f"  Mojo implementation failed: {e}")
        
        results[config['name']] = config_results
        
        # Print results for this configuration
        print("  Results:")
        for impl, result in config_results.items():
            if "error" in result:
                print(f"    {impl}: ERROR - {result['error']}")
            else:
                print(f"    {impl}: {result['mean_time']*1000:.3f}ms ± {result['std_time']*1000:.3f}ms")
    
    return results


def print_summary(results: Dict):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Collect all implementations
    implementations = set()
    for config_results in results.values():
        implementations.update(config_results.keys())
    
    implementations = sorted(list(implementations))
    
    # Print header
    header = f"{'Configuration':<15}"
    for impl in implementations:
        header += f"{impl:>12}"
    print(header)
    print("-" * len(header))
    
    # Print results for each configuration
    for config_name, config_results in results.items():
        row = f"{config_name:<15}"
        for impl in implementations:
            if impl in config_results:
                result = config_results[impl]
                if "error" in result:
                    row += f"{'ERROR':>12}"
                else:
                    row += f"{result['mean_time']*1000:>12.3f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    # Print speedup analysis
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS (vs Reference)")
    print("=" * 80)
    
    for config_name, config_results in results.items():
        if "reference" not in config_results or "error" in config_results["reference"]:
            continue
            
        ref_time = config_results["reference"]["mean_time"]
        print(f"\n{config_name}:")
        
        for impl in implementations:
            if impl == "reference":
                continue
            if impl in config_results and "error" not in config_results[impl]:
                impl_time = config_results[impl]["mean_time"]
                speedup = ref_time / impl_time
                print(f"  {impl}: {speedup:.2f}x faster")


def main():
    parser = argparse.ArgumentParser(description="Benchmark selective scan implementations")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], 
                       help="Device to run benchmarks on")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                       help="Data type to use")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Benchmark Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Data Type: {args.dtype}")
    print(f"  Warmup Runs: {args.warmup}")
    print(f"  Benchmark Runs: {args.runs}")
    
    # Run benchmarks
    results = run_benchmarks(device=args.device, dtype=dtype)
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{args.device}_{args.dtype}_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for config_name, config_results in results.items():
        serializable_results[config_name] = {}
        for impl, result in config_results.items():
            serializable_results[config_name][impl] = result
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main() 