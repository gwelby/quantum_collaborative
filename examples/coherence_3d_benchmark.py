#!/usr/bin/env python3
"""
Benchmark script for 3D field coherence calculation using Thread Block Clusters and CUDA Graphs.

This script compares the performance of different implementations of 3D field
coherence calculation, including standard CUDA, Thread Block Clusters, and CUDA Graphs.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

# Ensure quantum_field package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from quantum_field.backends.cuda import CUDABackend
    from quantum_field.constants import PHI, LAMBDA, SACRED_FREQUENCIES
except ImportError:
    print("Error: quantum_field package not found")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark 3D field coherence calculation optimizations"
    )
    parser.add_argument(
        "--dimensions", nargs=3, type=int, default=[64, 64, 64],
        help="Dimensions of the 3D field (depth, height, width), default: 64 64 64"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of fields to generate for benchmarking, default: 10"
    )
    parser.add_argument(
        "--frequency", type=str, default="love",
        choices=list(SACRED_FREQUENCIES.keys()),
        help="Sacred frequency to use, default: love"
    )
    parser.add_argument(
        "--no-tbc", action="store_true",
        help="Disable Thread Block Clusters even if supported"
    )
    parser.add_argument(
        "--no-graphs", action="store_true",
        help="Disable CUDA Graphs benchmarking"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save benchmark results to this file (PNG format)"
    )
    return parser.parse_args()


def generate_test_fields(
    backend: CUDABackend,
    depth: int,
    height: int,
    width: int,
    frequency: str,
    iterations: int
) -> List[np.ndarray]:
    """Generate a series of test fields with different time factors"""
    fields = []
    for i in range(iterations):
        time_factor = i * 0.1
        field = backend.generate_3d_quantum_field(
            width, height, depth, frequency, time_factor
        )
        fields.append(field)
    return fields


def benchmark_standard_coherence(
    backend: CUDABackend,
    fields: List[np.ndarray]
) -> Tuple[List[float], float]:
    """Benchmark standard coherence calculation"""
    coherence_values = []
    start_time = time.time()
    
    for field in fields:
        coherence = backend.calculate_3d_field_coherence(field)
        coherence_values.append(coherence)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return coherence_values, total_time


def benchmark_tbc_coherence(
    backend: CUDABackend,
    fields: List[np.ndarray],
    force_tbc: bool = True
) -> Tuple[List[float], float]:
    """Benchmark Thread Block Clusters coherence calculation"""
    if not getattr(backend, 'has_thread_block_cluster_support', False):
        return [], 0.0
    
    # Temporarily store original value
    original_has_tbc = backend.has_thread_block_cluster_support
    
    if force_tbc and not original_has_tbc:
        # Not supported, skip
        return [], 0.0
    
    coherence_values = []
    
    # Ensure TBC is enabled for this test
    if force_tbc:
        backend.has_thread_block_cluster_support = True
    
    try:
        start_time = time.time()
        
        for field in fields:
            try:
                coherence = backend._calculate_3d_field_coherence_tbc(field)
                coherence_values.append(coherence)
            except Exception as e:
                print(f"Error in TBC coherence calculation: {e}")
                return [], 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
    finally:
        # Restore original value
        backend.has_thread_block_cluster_support = original_has_tbc
    
    return coherence_values, total_time


def benchmark_graph_coherence(
    backend: CUDABackend,
    fields: List[np.ndarray],
    use_tbc: bool = None
) -> Tuple[List[float], float]:
    """Benchmark CUDA Graph coherence calculation"""
    if not hasattr(backend, 'calculate_3d_field_coherence_with_graph'):
        return [], 0.0
    
    try:
        start_time = time.time()
        coherence_values = backend.calculate_3d_field_coherence_with_graph(fields, use_tbc=use_tbc)
        end_time = time.time()
        total_time = end_time - start_time
        
        return coherence_values, total_time
    except Exception as e:
        print(f"Error in graph coherence calculation: {e}")
        return [], 0.0


def run_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all benchmarks and return results"""
    # Initialize CUDA backend
    backend = CUDABackend()
    backend.initialize()
    
    # Validate backend capabilities
    is_cuda_available = backend.initialized
    has_tbc_support = getattr(backend, 'has_thread_block_cluster_support', False)
    has_graph_support = hasattr(backend, 'calculate_3d_field_coherence_with_graph')
    
    if not is_cuda_available:
        print("Error: CUDA backend not properly initialized")
        sys.exit(1)
    
    # Parse dimensions
    depth, height, width = args.dimensions
    field_size = width * height * depth
    
    # Generate test fields
    print(f"Generating {args.iterations} test fields of size {width}x{height}x{depth}...")
    fields = generate_test_fields(
        backend, depth, height, width, args.frequency, args.iterations
    )
    
    results = {
        "field_size": f"{width}x{height}x{depth}",
        "iterations": args.iterations,
        "frequency": args.frequency,
        "cuda_available": is_cuda_available,
        "tbc_supported": has_tbc_support,
        "graph_supported": has_graph_support,
        "times": {},
        "coherence_values": {},
        "speedups": {}
    }
    
    # Benchmark standard coherence calculation
    print("\nBenchmarking standard coherence calculation...")
    standard_coherence, standard_time = benchmark_standard_coherence(backend, fields)
    results["times"]["standard"] = standard_time
    results["coherence_values"]["standard"] = standard_coherence
    results["speedups"]["standard"] = 1.0  # Reference
    print(f"Standard coherence: {standard_time:.6f} seconds "
          f"({args.iterations / standard_time:.2f} fields/sec)")
    
    # Benchmark Thread Block Clusters coherence calculation if supported
    if has_tbc_support and not args.no_tbc:
        print("\nBenchmarking Thread Block Clusters coherence calculation...")
        tbc_coherence, tbc_time = benchmark_tbc_coherence(backend, fields)
        results["times"]["tbc"] = tbc_time
        results["coherence_values"]["tbc"] = tbc_coherence
        
        if tbc_time > 0:
            tbc_speedup = standard_time / tbc_time
            results["speedups"]["tbc"] = tbc_speedup
            print(f"TBC coherence: {tbc_time:.6f} seconds "
                  f"({args.iterations / tbc_time:.2f} fields/sec)")
            print(f"TBC speedup: {tbc_speedup:.2f}x")
    
    # Benchmark CUDA Graph coherence calculation if supported
    if has_graph_support and not args.no_graphs:
        # Standard graph (no TBC)
        print("\nBenchmarking standard CUDA Graph coherence calculation...")
        std_graph_coherence, std_graph_time = benchmark_graph_coherence(
            backend, fields, use_tbc=False
        )
        results["times"]["std_graph"] = std_graph_time
        results["coherence_values"]["std_graph"] = std_graph_coherence
        
        if std_graph_time > 0:
            std_graph_speedup = standard_time / std_graph_time
            results["speedups"]["std_graph"] = std_graph_speedup
            print(f"Standard graph coherence: {std_graph_time:.6f} seconds "
                  f"({args.iterations / std_graph_time:.2f} fields/sec)")
            print(f"Standard graph speedup: {std_graph_speedup:.2f}x")
        
        # TBC graph if supported
        if has_tbc_support and not args.no_tbc:
            print("\nBenchmarking TBC CUDA Graph coherence calculation...")
            tbc_graph_coherence, tbc_graph_time = benchmark_graph_coherence(
                backend, fields, use_tbc=True
            )
            results["times"]["tbc_graph"] = tbc_graph_time
            results["coherence_values"]["tbc_graph"] = tbc_graph_coherence
            
            if tbc_graph_time > 0:
                tbc_graph_speedup = standard_time / tbc_graph_time
                results["speedups"]["tbc_graph"] = tbc_graph_speedup
                print(f"TBC graph coherence: {tbc_graph_time:.6f} seconds "
                      f"({args.iterations / tbc_graph_time:.2f} fields/sec)")
                print(f"TBC graph speedup: {tbc_graph_speedup:.2f}x")
    
    # Verify coherence values match
    if standard_coherence:
        ref_coherence = standard_coherence
        for method, values in results["coherence_values"].items():
            if method != "standard" and values:
                # Ensure coherence values are close
                max_diff = max(abs(a - b) for a, b in zip(ref_coherence, values))
                print(f"\nMax coherence difference ({method} vs standard): {max_diff:.6f}")
                if max_diff > 0.1:
                    print(f"Warning: Large difference in coherence values for {method}")
    
    return results


def plot_results(results: Dict[str, Any], output_file: str = None) -> None:
    """Plot benchmark results"""
    plt.figure(figsize=(12, 8))
    
    # Plot execution time per field
    methods = list(results["times"].keys())
    times = [results["times"][m] / results["iterations"] for m in methods]
    
    # Plot bar chart of execution time per field
    plt.subplot(2, 1, 1)
    bars = plt.bar(methods, times)
    
    # Add time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.6f}s',
                ha='center', va='bottom', rotation=0)
    
    plt.ylabel('Time per field (seconds)')
    plt.title(f'Coherence Calculation Performance: {results["field_size"]} field, {results["frequency"]} frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    speedups = [results["speedups"].get(m, 0) for m in methods]
    bars = plt.bar(methods, speedups)
    
    # Add speedup labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x',
                ha='center', va='bottom', rotation=0)
    
    plt.ylabel('Speedup vs standard')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Benchmark results saved to {output_file}")
    else:
        plt.show()


def main() -> None:
    """Main function"""
    args = parse_args()
    results = run_benchmarks(args)
    
    if results["times"]:
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Field size: {results['field_size']}")
        print(f"Iterations: {results['iterations']}")
        
        # Print performance comparison
        for method, time_taken in results["times"].items():
            fields_per_sec = results["iterations"] / time_taken if time_taken > 0 else 0
            speedup = results["speedups"].get(method, 0)
            print(f"{method}: {time_taken:.6f}s ({fields_per_sec:.2f} fields/sec), "
                  f"speedup: {speedup:.2f}x")
        
        # Plot results if matplotlib is available
        try:
            plot_results(results, args.output)
        except Exception as e:
            print(f"Error plotting results: {e}")


if __name__ == "__main__":
    main()