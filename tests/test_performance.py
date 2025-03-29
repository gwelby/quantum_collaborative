#!/usr/bin/env python3
"""
Performance regression tests for quantum field visualization

These tests ensure that performance characteristics remain stable
across versions, and verify that GPU acceleration provides the
expected speedups.
"""

import os
import sys
import pytest
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Skip all tests if quantum_field package is not available
try:
    import quantum_field
    from quantum_field.core import (
        generate_quantum_field, 
        calculate_field_coherence,
        generate_quantum_field_cpu,
        CUDA_AVAILABLE
    )
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)


# Define test fixtures
@pytest.fixture
def small_field() -> Tuple[int, int]:
    """Define a small field size for quick tests"""
    return (128, 128)


@pytest.fixture
def medium_field() -> Tuple[int, int]:
    """Define a medium field size for general tests"""
    return (512, 512)


@pytest.fixture
def large_field() -> Tuple[int, int]:
    """Define a large field size for stress tests"""
    return (1024, 1024)


@pytest.fixture
def performance_baseline_path() -> Path:
    """Path to store/load performance baseline data"""
    # Create the directory if it doesn't exist
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    return baseline_dir / "performance_baseline.json"


def load_baseline(path: Path) -> Dict[str, Any]:
    """Load performance baseline data from file"""
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_baseline(path: Path, data: Dict[str, Any]) -> None:
    """Save performance baseline data to file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


class TestFieldGenerationPerformance:
    """Test the performance of field generation"""

    def measure_generation_time(self, width: int, height: int, frequency: str = 'love',
                               iterations: int = 3) -> float:
        """Measure average field generation time"""
        times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = generate_quantum_field(width, height, frequency)
            times.append(time.time() - start_time)
        return sum(times) / len(times)

    def test_small_field_performance(self, small_field: Tuple[int, int]):
        """Verify small field generation performance"""
        width, height = small_field
        avg_time = self.measure_generation_time(width, height)
        
        # Small fields should generate quickly (adjust threshold as needed)
        # This is just a sanity check, not a strict performance requirement
        assert avg_time < 1.0, f"Small field generation took {avg_time:.3f}s (expected < 1.0s)"
        print(f"\nSmall field ({width}x{height}) generation: {avg_time:.3f}s")

    def test_medium_field_performance(self, medium_field: Tuple[int, int], 
                                     performance_baseline_path: Path):
        """Monitor medium field generation performance against baseline"""
        width, height = medium_field
        avg_time = self.measure_generation_time(width, height)
        
        # Load existing baseline data
        baseline = load_baseline(performance_baseline_path)
        medium_field_key = f"field_generation_{width}x{height}"
        
        # If no baseline exists, save the current timing as baseline
        if medium_field_key not in baseline:
            baseline[medium_field_key] = {
                "time": avg_time,
                "width": width,
                "height": height,
                "timestamp": time.time(),
                "cuda_available": CUDA_AVAILABLE
            }
            save_baseline(performance_baseline_path, baseline)
            print(f"\nEstablished new baseline for {width}x{height} field: {avg_time:.3f}s")
        else:
            # Compare with existing baseline
            baseline_time = baseline[medium_field_key]["time"]
            regression_threshold = 1.5  # Allow 50% slowdown before failing
            
            print(f"\nMedium field ({width}x{height}) generation: {avg_time:.3f}s (baseline: {baseline_time:.3f}s)")
            
            # Only enforce performance requirements if CUDA was available in both cases
            if CUDA_AVAILABLE and baseline[medium_field_key].get("cuda_available", False):
                assert avg_time < baseline_time * regression_threshold, \
                       f"Performance regression detected: {avg_time:.3f}s vs baseline {baseline_time:.3f}s"
                
                # Update baseline if performance has improved
                if avg_time < baseline_time * 0.9:  # 10% improvement
                    baseline[medium_field_key] = {
                        "time": avg_time,
                        "width": width,
                        "height": height,
                        "timestamp": time.time(),
                        "cuda_available": CUDA_AVAILABLE
                    }
                    save_baseline(performance_baseline_path, baseline)
                    print(f"Updated baseline for {width}x{height} field to {avg_time:.3f}s (improvement)")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA acceleration not available")
    def test_gpu_speedup(self, medium_field: Tuple[int, int]):
        """Verify that GPU acceleration provides speedup over CPU"""
        width, height = medium_field
        
        # Time CPU implementation
        cpu_times = []
        for _ in range(3):
            start_time = time.time()
            _ = generate_quantum_field_cpu(width, height, 'love')
            cpu_times.append(time.time() - start_time)
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        
        # Time GPU implementation
        gpu_times = []
        for _ in range(3):
            start_time = time.time()
            _ = generate_quantum_field(width, height, 'love')
            gpu_times.append(time.time() - start_time)
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time
        
        print(f"\nCPU: {avg_cpu_time:.3f}s, GPU: {avg_gpu_time:.3f}s, Speedup: {speedup:.2f}x")
        
        # GPU should be faster than CPU
        assert speedup > 1.0, "GPU implementation is not faster than CPU"
        
        # Modern GPUs should provide good speedup
        # This threshold may need adjustment based on hardware
        min_expected_speedup = 2.0
        assert speedup >= min_expected_speedup, \
               f"GPU speedup ({speedup:.2f}x) below expected minimum ({min_expected_speedup}x)"


class TestCoherenceCalculationPerformance:
    """Test the performance of coherence calculation"""

    def measure_coherence_time(self, width: int, height: int, iterations: int = 3) -> float:
        """Measure average coherence calculation time"""
        # Generate a field first
        field = generate_quantum_field(width, height)
        
        # Measure coherence calculation time
        times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = calculate_field_coherence(field)
            times.append(time.time() - start_time)
        return sum(times) / len(times)
    
    def test_coherence_performance(self, medium_field: Tuple[int, int],
                                  performance_baseline_path: Path):
        """Monitor coherence calculation performance against baseline"""
        width, height = medium_field
        avg_time = self.measure_coherence_time(width, height)
        
        # Load existing baseline data
        baseline = load_baseline(performance_baseline_path)
        coherence_key = f"coherence_calculation_{width}x{height}"
        
        # If no baseline exists, save the current timing as baseline
        if coherence_key not in baseline:
            baseline[coherence_key] = {
                "time": avg_time,
                "width": width,
                "height": height,
                "timestamp": time.time(),
                "cuda_available": CUDA_AVAILABLE
            }
            save_baseline(performance_baseline_path, baseline)
            print(f"\nEstablished new baseline for {width}x{height} coherence: {avg_time:.3f}s")
        else:
            # Compare with existing baseline
            baseline_time = baseline[coherence_key]["time"]
            regression_threshold = 1.5  # Allow 50% slowdown before failing
            
            print(f"\nCoherence calculation ({width}x{height}): {avg_time:.3f}s (baseline: {baseline_time:.3f}s)")
            
            # Only enforce performance requirements if CUDA was available in both cases
            if CUDA_AVAILABLE and baseline[coherence_key].get("cuda_available", False):
                assert avg_time < baseline_time * regression_threshold, \
                       f"Performance regression detected: {avg_time:.3f}s vs baseline {baseline_time:.3f}s"
                
                # Update baseline if performance has improved
                if avg_time < baseline_time * 0.9:  # 10% improvement
                    baseline[coherence_key] = {
                        "time": avg_time,
                        "width": width,
                        "height": height,
                        "timestamp": time.time(),
                        "cuda_available": CUDA_AVAILABLE
                    }
                    save_baseline(performance_baseline_path, baseline)
                    print(f"Updated baseline for {width}x{height} coherence to {avg_time:.3f}s (improvement)")


class TestScalabilityPerformance:
    """Test how performance scales with field size"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA acceleration not available")
    def test_scaling_with_field_size(self):
        """Test how generation time scales with field size"""
        test_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024)
        ]
        
        results = {
            "sizes": [],
            "times": [],
            "pixels": [],
            "pixels_per_second": []
        }
        
        for width, height in test_sizes:
            pixels = width * height
            results["sizes"].append(f"{width}x{height}")
            results["pixels"].append(pixels)
            
            # Measure generation time
            times = []
            for _ in range(3):
                start_time = time.time()
                _ = generate_quantum_field(width, height)
                times.append(time.time() - start_time)
            avg_time = sum(times) / len(times)
            results["times"].append(avg_time)
            
            # Calculate pixels per second
            pixels_per_second = pixels / avg_time
            results["pixels_per_second"].append(pixels_per_second)
            
            print(f"\nField size {width}x{height}: {avg_time:.3f}s ({pixels_per_second:.0f} pixels/s)")
        
        # As field size increases, pixels per second should remain relatively stable
        # or even increase (due to better GPU utilization with larger workloads)
        min_pps = min(results["pixels_per_second"])
        max_pps = max(results["pixels_per_second"])
        
        # Shouldn't drop below 50% of max efficiency
        assert min_pps >= max_pps * 0.5, "Significant efficiency drop detected with different field sizes"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA acceleration not available")
class TestCUDAMemoryUsage:
    """Test CUDA memory usage with different field sizes"""
    
    def get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes"""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes()
        except (ImportError, AttributeError):
            return 0
    
    def test_memory_scaling(self):
        """Test how memory usage scales with field size"""
        import gc
        
        # Force garbage collection to start with clean state
        gc.collect()
        
        try:
            import cupy as cp
            # Try to clear memory pool
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
            # Field sizes to test
            test_sizes = [
                (512, 512),
                (1024, 1024),
                (2048, 2048)
            ]
            
            results = []
            
            for width, height in test_sizes:
                pixels = width * height
                
                # Record memory before
                mempool.free_all_blocks()
                memory_before = self.get_gpu_memory_usage()
                
                # Generate field
                field = generate_quantum_field(width, height)
                
                # Record memory after
                memory_after = self.get_gpu_memory_usage()
                
                # Calculate memory usage for this field
                memory_used = memory_after - memory_before
                bytes_per_pixel = memory_used / pixels if pixels > 0 else 0
                
                results.append({
                    "size": f"{width}x{height}",
                    "pixels": pixels,
                    "memory_used": memory_used,
                    "bytes_per_pixel": bytes_per_pixel
                })
                
                print(f"\nField size {width}x{height}: {memory_used/1024/1024:.2f} MB " 
                      f"({bytes_per_pixel:.2f} bytes/pixel)")
                
                # Clean up
                del field
                gc.collect()
                mempool.free_all_blocks()
            
            # Memory usage should scale linearly with field size
            # Compare bytes per pixel across different field sizes
            bytes_per_pixel_values = [r["bytes_per_pixel"] for r in results if r["bytes_per_pixel"] > 0]
            
            if bytes_per_pixel_values:
                avg_bytes_per_pixel = sum(bytes_per_pixel_values) / len(bytes_per_pixel_values)
                max_deviation = max(abs(bpp - avg_bytes_per_pixel) for bpp in bytes_per_pixel_values)
                relative_deviation = max_deviation / avg_bytes_per_pixel if avg_bytes_per_pixel > 0 else 0
                
                # Memory usage per pixel should be consistent (within 30%)
                assert relative_deviation < 0.3, "Memory usage does not scale linearly with field size"
        
        except ImportError:
            pytest.skip("cupy not available for memory testing")


# Allow running benchmarks directly
if __name__ == "__main__":
    import sys
    
    print("Running performance benchmarks...")
    
    # Create test instances
    field_test = TestFieldGenerationPerformance()
    coherence_test = TestCoherenceCalculationPerformance()
    scaling_test = TestScalabilityPerformance()
    
    # Define field sizes
    small = (128, 128)
    medium = (512, 512)
    large = (1024, 1024)
    
    # Get baseline path
    baseline_path = Path(__file__).parent / "baselines"
    baseline_path.mkdir(exist_ok=True)
    performance_baseline_path = baseline_path / "performance_baseline.json"
    
    # Run performance tests
    print("\n=== Field Generation Performance ===")
    field_test.test_small_field_performance(small)
    field_test.test_medium_field_performance(medium, performance_baseline_path)
    
    if CUDA_AVAILABLE:
        field_test.test_gpu_speedup(medium)
    
    print("\n=== Coherence Calculation Performance ===")
    coherence_test.test_coherence_performance(medium, performance_baseline_path)
    
    if CUDA_AVAILABLE:
        print("\n=== Scalability Performance ===")
        scaling_test.test_scaling_with_field_size()
        
        print("\n=== CUDA Memory Usage ===")
        memory_test = TestCUDAMemoryUsage()
        memory_test.test_memory_scaling()
    
    print("\nPerformance benchmarks completed")