#!/usr/bin/env python3
"""
Quantum Field Acceleration Benchmarking

This module provides tools for benchmarking and comparing CPU vs GPU 
implementations of quantum field generation and analysis.
"""

import time
import numpy as np
import math
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import sacred_constants as sc
except ImportError:
    print("Warning: sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 2.1784575679375995
        
        SACRED_FREQUENCIES = {
            'love': 528,
            'unity': 432,
            'cascade': 594,
            'truth': 672,
            'vision': 720,
            'oneness': 768,
        }

# Import CUDA functionality with fallback
try:
    import quantum_cuda as qc
    CUDA_AVAILABLE = qc.CUDA_AVAILABLE
except ImportError:
    print("Warning: quantum_cuda module not found. GPU benchmarks will be disabled.")
    CUDA_AVAILABLE = False

def benchmark_implementations(sizes, frequency_name='love', iterations=3):
    """
    Benchmark different implementations of quantum field generation.
    
    Args:
        sizes: List of (width, height) tuples to benchmark
        frequency_name: Sacred frequency to use
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'sizes': sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    # Check if CUDA is available
    if not CUDA_AVAILABLE:
        print("CUDA is not available. Running CPU benchmarks only.")
    
    for width, height in sizes:
        print(f"\nBenchmarking size: {width}x{height}")
        
        # CPU implementation
        cpu_times = []
        for i in range(iterations):
            start_time = time.time()
            if CUDA_AVAILABLE:
                field = qc.generate_quantum_field_cpu(width, height, frequency_name)
            else:
                # Fallback implementation if quantum_cuda is not available
                field = generate_field_cpu_fallback(width, height, frequency_name)
            end_time = time.time()
            cpu_times.append(end_time - start_time)
            print(f"  CPU iteration {i+1}/{iterations}: {cpu_times[-1]:.4f} seconds")
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        results['cpu_times'].append(avg_cpu_time)
        print(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
        
        # GPU implementation (if available)
        if CUDA_AVAILABLE:
            gpu_times = []
            for i in range(iterations):
                start_time = time.time()
                field = qc.generate_quantum_field_cuda(width, height, frequency_name)
                end_time = time.time()
                gpu_times.append(end_time - start_time)
                print(f"  GPU iteration {i+1}/{iterations}: {gpu_times[-1]:.4f} seconds")
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            
            results['gpu_times'].append(avg_gpu_time)
            results['speedups'].append(speedup)
            
            print(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
        else:
            results['gpu_times'].append(0)
            results['speedups'].append(0)
    
    return results

def generate_field_cpu_fallback(width, height, frequency_name='love', time_factor=0):
    """CPU fallback implementation for quantum field generation"""
    # Get the frequency value
    frequency = sc.SACRED_FREQUENCIES.get(frequency_name, 528)
    
    # Scale the frequency to a more manageable number
    freq_factor = frequency / 1000.0 * sc.PHI
    
    # Initialize the field
    field = np.zeros((height, width), dtype=np.float32)
    
    # Calculate the center of the field
    center_x = width / 2
    center_y = height / 2
    
    # Generate the field values
    for y in range(height):
        for x in range(width):
            # Calculate distance from center (normalized)
            dx = (x - center_x) / (width / 2)
            dy = (y - center_y) / (height / 2)
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate the field value using phi-harmonics
            angle = math.atan2(dy, dx) * sc.PHI
            time_value = time_factor * sc.LAMBDA
            
            # Create an interference pattern
            value = (
                math.sin(distance * freq_factor + time_value) * 
                math.cos(angle * sc.PHI) * 
                math.exp(-distance / sc.PHI)
            )
            
            field[y, x] = value
    
    return field

def plot_results(results):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary with benchmark results from benchmark_implementations()
    """
    # Convert size tuples to labels
    sizes = results['sizes']
    size_labels = [f"{w}x{h}" for w, h in sizes]
    
    # Extract data
    cpu_times = results['cpu_times']
    gpu_times = results['gpu_times'] if 'gpu_times' in results else []
    speedups = results['speedups'] if 'speedups' in results else []
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot times
    x = np.arange(len(size_labels))
    width = 0.35
    
    ax1.bar(x - width/2, cpu_times, width, label='CPU')
    if gpu_times:
        ax1.bar(x + width/2, gpu_times, width, label='GPU')
    
    ax1.set_xlabel('Field Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot speedups
    if speedups:
        ax2.bar(x, speedups, width, color='green')
        ax2.set_xlabel('Field Size')
        ax2.set_ylabel('Speedup (x times)')
        ax2.set_title('GPU Speedup over CPU')
        ax2.set_xticks(x)
        ax2.set_xticklabels(size_labels)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add speedup values as text
        for i, v in enumerate(speedups):
            ax2.text(i, v + 0.5, f"{v:.2f}x", ha='center')
    else:
        ax2.text(0.5, 0.5, 'GPU results not available', ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plt.savefig('quantum_acceleration_benchmark.png')
    print("Benchmark results saved to 'quantum_acceleration_benchmark.png'")
    
    # Show plot if in interactive mode
    if hasattr(sys, 'ps1'):  # Check if running in interactive mode
        plt.show()

def benchmark_frequencies(width, height, iterations=3):
    """
    Benchmark performance across different sacred frequencies.
    
    Args:
        width: Field width
        height: Field height
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary with benchmark results for each frequency
    """
    results = {
        'frequencies': list(sc.SACRED_FREQUENCIES.keys()),
        'cpu_times': [],
        'gpu_times': [],
        'coherence_values': []
    }
    
    print(f"\nBenchmarking frequencies for field size: {width}x{height}")
    
    for freq_name in sc.SACRED_FREQUENCIES.keys():
        print(f"\nFrequency: {freq_name} ({sc.SACRED_FREQUENCIES[freq_name]} Hz)")
        
        # CPU implementation
        cpu_times = []
        for i in range(iterations):
            start_time = time.time()
            if CUDA_AVAILABLE:
                field = qc.generate_quantum_field_cpu(width, height, freq_name)
            else:
                field = generate_field_cpu_fallback(width, height, freq_name)
            end_time = time.time()
            cpu_times.append(end_time - start_time)
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        results['cpu_times'].append(avg_cpu_time)
        print(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
        
        # GPU implementation (if available)
        if CUDA_AVAILABLE:
            gpu_times = []
            coherence = 0
            
            for i in range(iterations):
                start_time = time.time()
                field = qc.generate_quantum_field_cuda(width, height, freq_name)
                end_time = time.time()
                gpu_times.append(end_time - start_time)
                
                # Calculate coherence for the last iteration
                if i == iterations - 1:
                    coherence = qc.calculate_field_coherence(field)
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            
            results['gpu_times'].append(avg_gpu_time)
            results['coherence_values'].append(coherence)
            
            print(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
            print(f"  Field coherence: {coherence:.4f}")
        else:
            results['gpu_times'].append(0)
            results['coherence_values'].append(0)
    
    return results

def plot_frequency_results(results):
    """
    Plot benchmark results across different frequencies.
    
    Args:
        results: Dictionary with benchmark results from benchmark_frequencies()
    """
    # Extract data
    frequencies = results['frequencies']
    cpu_times = results['cpu_times']
    gpu_times = results['gpu_times'] if 'gpu_times' in results else []
    coherence_values = results['coherence_values'] if 'coherence_values' in results else []
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot times
    x = np.arange(len(frequencies))
    width = 0.35
    
    ax1.bar(x - width/2, cpu_times, width, label='CPU')
    if gpu_times:
        ax1.bar(x + width/2, gpu_times, width, label='GPU')
    
    ax1.set_xlabel('Sacred Frequency')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time by Frequency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frequencies)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot coherence values if available
    if coherence_values and any(coherence_values):
        color_map = plt.cm.viridis(np.linspace(0, 1, len(coherence_values)))
        bars = ax2.bar(x, coherence_values, width, color=color_map)
        ax2.set_xlabel('Sacred Frequency')
        ax2.set_ylabel('Field Coherence')
        ax2.set_title('Quantum Field Coherence by Frequency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(frequencies)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add phi constant line
        ax2.axhline(y=sc.PHI, color='r', linestyle='--', label=f'PHI = {sc.PHI:.4f}')
        ax2.legend()
        
        # Add coherence values as text
        for i, v in enumerate(coherence_values):
            ax2.text(i, v + 0.05, f"{v:.3f}", ha='center')
    else:
        ax2.text(0.5, 0.5, 'Coherence results not available', ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plt.savefig('frequency_benchmark.png')
    print("Frequency benchmark results saved to 'frequency_benchmark.png'")
    
    # Show plot if in interactive mode
    if hasattr(sys, 'ps1'):  # Check if running in interactive mode
        plt.show()

def benchmark_thread_blocks(width, height, frequency_name='love', iterations=3):
    """
    Benchmark different thread block configurations for CUDA implementation.
    
    Args:
        width: Field width
        height: Field height
        frequency_name: Sacred frequency to use
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary with benchmark results for each block size
    """
    if not CUDA_AVAILABLE:
        print("CUDA is not available. Thread block benchmarking skipped.")
        return None
    
    # Different thread block configurations to test
    block_sizes = [(8, 8), (16, 16), (32, 8), (8, 32), (32, 32)]
    
    results = {
        'block_sizes': [f"{w}x{h}" for w, h in block_sizes],
        'times': [],
        'relative_performance': []
    }
    
    print(f"\nBenchmarking thread block configurations for field size: {width}x{height}")
    
    # Import module after CUDA_AVAILABLE check to avoid import errors
    try:
        import quantum_cuda as qc
        from cuda.core.experimental import LaunchConfig, launch
    except ImportError:
        print("Error importing CUDA modules. Thread block benchmarking skipped.")
        return None
    
    # Initialize CUDA
    if not qc.initialize_cuda():
        print("Failed to initialize CUDA. Thread block benchmarking skipped.")
        return None
    
    # Get frequency value
    frequency = sc.SACRED_FREQUENCIES.get(frequency_name, 528)
    
    # Run benchmarks for each block size
    for i, (block_x, block_y) in enumerate(block_sizes):
        times = []
        print(f"\nTesting block size: {block_x}x{block_y}")
        
        for j in range(iterations):
            try:
                # Create CuPy array for output
                import cupy as cp
                output = cp.empty((height, width), dtype=cp.float32)
                
                # Create grid dimensions based on block size
                grid_x = (width + block_x - 1) // block_x
                grid_y = (height + block_y - 1) // block_y
                
                # Set up launch configuration
                config = LaunchConfig(
                    grid=(grid_x, grid_y, 1),
                    block=(block_x, block_y, 1)
                )
                
                # Get kernel function
                kernel = qc.cuda_module.get_kernel("generate_quantum_field<float>")
                
                # Benchmark kernel execution
                start_time = time.time()
                launch(
                    qc.cuda_stream,
                    config,
                    kernel,
                    output.data.ptr,
                    width,
                    height,
                    frequency,
                    sc.PHI,
                    sc.LAMBDA,
                    0.0  # time_factor
                )
                
                # Synchronize to ensure accurate timing
                qc.cuda_stream.sync()
                end_time = time.time()
                
                times.append(end_time - start_time)
                print(f"  Iteration {j+1}/{iterations}: {times[-1]:.6f} seconds")
                
            except Exception as e:
                print(f"Error during benchmark: {e}")
                times.append(float('inf'))
        
        # Calculate average time (excluding errors)
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            results['times'].append(avg_time)
            print(f"  Average time: {avg_time:.6f} seconds")
        else:
            results['times'].append(float('inf'))
            print("  All iterations failed")
    
    # Calculate relative performance (normalized to best performer)
    valid_times = [t for t in results['times'] if t != float('inf')]
    if valid_times:
        best_time = min(valid_times)
        results['relative_performance'] = [
            best_time / t if t != float('inf') else 0 for t in results['times']
        ]
    
    return results

def plot_thread_block_results(results):
    """
    Plot thread block benchmark results.
    
    Args:
        results: Dictionary with benchmark results from benchmark_thread_blocks()
    """
    if not results:
        print("No thread block benchmark results to plot.")
        return
    
    # Extract data
    block_sizes = results['block_sizes']
    times = results['times']
    relative_performance = results['relative_performance']
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot times
    x = np.arange(len(block_sizes))
    width = 0.6
    
    # Filter out invalid times (inf)
    valid_indices = [i for i, t in enumerate(times) if t != float('inf')]
    valid_block_sizes = [block_sizes[i] for i in valid_indices]
    valid_times = [times[i] for i in valid_indices]
    valid_x = [x[i] for i in valid_indices]
    
    if valid_times:
        bars = ax1.bar(valid_x, valid_times, width, color='skyblue')
        ax1.set_xlabel('Thread Block Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Kernel Execution Time by Thread Block Size')
        ax1.set_xticks(valid_x)
        ax1.set_xticklabels(valid_block_sizes)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add time values as text
        for i, v in enumerate(valid_times):
            ax1.text(valid_x[i], v + 0.0001, f"{v:.6f}", ha='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No valid time measurements', ha='center', va='center', transform=ax1.transAxes)
    
    # Plot relative performance
    valid_perf = [relative_performance[i] for i in valid_indices]
    
    if valid_perf:
        bars = ax2.bar(valid_x, valid_perf, width, color='lightgreen')
        ax2.set_xlabel('Thread Block Size')
        ax2.set_ylabel('Relative Performance (higher is better)')
        ax2.set_title('Performance Relative to Best Configuration')
        ax2.set_xticks(valid_x)
        ax2.set_xticklabels(valid_block_sizes)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add performance values as text
        for i, v in enumerate(valid_perf):
            ax2.text(valid_x[i], v + 0.02, f"{v:.2f}x", ha='center')
    else:
        ax2.text(0.5, 0.5, 'No valid performance measurements', ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plt.savefig('thread_block_benchmark.png')
    print("Thread block benchmark results saved to 'thread_block_benchmark.png'")
    
    # Show plot if in interactive mode
    if hasattr(sys, 'ps1'):  # Check if running in interactive mode
        plt.show()

def main():
    """Main function"""
    print("\nQUANTUM FIELD ACCELERATION BENCHMARKING")
    print("======================================")
    print(f"PHI: {sc.PHI}")
    print(f"LAMBDA: {sc.LAMBDA}")
    print(f"PHI^PHI: {sc.PHI_PHI}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print("\n")
    
    # Display available benchmarks
    print("Available Benchmarks:")
    print("1. Field Size Benchmark (CPU vs GPU)")
    print("2. Sacred Frequency Benchmark")
    print("3. Thread Block Configuration Benchmark (CUDA only)")
    print("4. Run All Benchmarks")
    print("5. Exit")
    
    while True:
        # Get user choice
        choice = input("\nSelect a benchmark (1-5): ")
        
        if choice == '1':
            print("\nRunning Field Size Benchmark...")
            # Define sizes to benchmark from small to large
            sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
            results = benchmark_implementations(sizes, 'love', iterations=3)
            plot_results(results)
            
        elif choice == '2':
            print("\nRunning Sacred Frequency Benchmark...")
            results = benchmark_frequencies(512, 512, iterations=3)
            plot_frequency_results(results)
            
        elif choice == '3':
            if not CUDA_AVAILABLE:
                print("\nCUDA is not available. Thread block benchmarking skipped.")
            else:
                print("\nRunning Thread Block Configuration Benchmark...")
                results = benchmark_thread_blocks(1024, 1024, 'love', iterations=5)
                plot_thread_block_results(results)
                
        elif choice == '4':
            print("\nRunning All Benchmarks...")
            
            # Field size benchmark
            print("\n1. Field Size Benchmark:")
            sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
            results1 = benchmark_implementations(sizes, 'love', iterations=3)
            plot_results(results1)
            
            # Frequency benchmark
            print("\n2. Sacred Frequency Benchmark:")
            results2 = benchmark_frequencies(512, 512, iterations=3)
            plot_frequency_results(results2)
            
            # Thread block benchmark (CUDA only)
            if CUDA_AVAILABLE:
                print("\n3. Thread Block Configuration Benchmark:")
                results3 = benchmark_thread_blocks(1024, 1024, 'love', iterations=5)
                plot_thread_block_results(results3)
                
        elif choice == '5':
            print("\nExiting Quantum Field Acceleration Benchmarking.")
            print(f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}")
            break
            
        else:
            print("Invalid choice. Please select a number between 1 and 5.")

if __name__ == "__main__":
    main()