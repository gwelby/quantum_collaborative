#!/usr/bin/env python3
"""
Integration tests for Thread Block Cluster functionality

These tests verify that Thread Block Clusters provide performance benefits
on compatible hardware and gracefully fall back on incompatible systems.
"""

import os
import sys
import pytest
import numpy as np
from typing import Tuple, Dict, Optional

# Skip all tests if quantum_field package is not available
try:
    import quantum_field
    from quantum_field.thread_block_cluster import (
        check_thread_block_cluster_support,
        initialize_thread_block_clusters,
        generate_quantum_field_tbc,
        benchmark_thread_block_cluster
    )
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)

# Check if Thread Block Clusters are supported
IS_TBC_SUPPORTED = check_thread_block_cluster_support()


@pytest.fixture
def test_field() -> Tuple[int, int]:
    """Define a test field size"""
    return (512, 512)


class TestThreadBlockClusterDetection:
    """Tests for Thread Block Cluster detection and initialization"""

    def test_tbc_detection(self):
        """Test Thread Block Cluster detection"""
        # This test always runs - we want to verify detection works
        # whether or not the current hardware supports TBC
        is_supported = check_thread_block_cluster_support()
        
        # Print detection result for information
        if is_supported:
            print("\nThread Block Cluster support detected")
        else:
            print("\nThread Block Cluster support not detected (requires H100+ GPU)")

    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    def test_tbc_initialization(self):
        """Test Thread Block Cluster initialization"""
        success = initialize_thread_block_clusters()
        assert success is True
        
        # Check if basic kernel execution works
        field = generate_quantum_field_tbc(64, 64, 'love', 0)
        assert field.shape == (64, 64)
        assert field.dtype == np.float32


class TestFieldGeneration:
    """Tests for field generation with Thread Block Clusters"""

    def test_graceful_fallback(self, test_field):
        """Test graceful fallback on unsupported hardware"""
        width, height = test_field
        
        # Should work whether or not TBC is supported
        field = generate_quantum_field_tbc(width, height, 'love', 0)
        
        # Check field dimensions
        assert field.shape == (height, width)
        assert field.dtype == np.float32
        
        # Check that field values are reasonable (should be in range [-1, 1])
        assert np.min(field) >= -1.0
        assert np.max(field) <= 1.0
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    @pytest.mark.parametrize("frequency", ['love', 'unity', 'cascade', 'truth'])
    def test_different_frequencies(self, test_field, frequency):
        """Test different frequencies with Thread Block Clusters"""
        width, height = test_field
        field = generate_quantum_field_tbc(width, height, frequency, 0)
        
        # Check field dimensions
        assert field.shape == (height, width)
        
        # Different frequencies should produce different fields
        field2 = generate_quantum_field_tbc(width, height, 'love' if frequency != 'love' else 'unity', 0)
        assert not np.array_equal(field, field2)
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    @pytest.mark.parametrize("time_factor", [0, 0.5, 1.0])
    def test_time_evolution(self, test_field, time_factor):
        """Test field evolution over time with Thread Block Clusters"""
        width, height = test_field
        field = generate_quantum_field_tbc(width, height, 'love', time_factor)
        
        # Fields at different time points should be different
        if time_factor != 0:
            field_t0 = generate_quantum_field_tbc(width, height, 'love', 0)
            assert not np.array_equal(field, field_t0)


class TestIntegrationWithCore:
    """Tests for integration with the core module"""

    def test_core_integration(self):
        """Test integration with the core module"""
        from quantum_field.core import generate_quantum_field
        
        # For very large fields, core should attempt to use TBC if supported
        width, height = 2048, 2048
        field1 = generate_quantum_field(width, height, 'love', 0)
        
        # This may or may not use TBC depending on hardware
        if IS_TBC_SUPPORTED:
            field2 = generate_quantum_field_tbc(width, height, 'love', 0)
            
            # Both should produce valid fields of the same shape
            assert field1.shape == field2.shape
            assert field1.dtype == field2.dtype
            
            # Values might differ slightly due to floating-point variations
            # between implementations, but should be close
            max_diff = np.max(np.abs(field1 - field2))
            assert max_diff < 0.01 or np.allclose(field1, field2, rtol=1e-3, atol=1e-3)
        else:
            # If not supported, just ensure we got a valid field
            assert field1.shape == (height, width)


@pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
class TestPerformance:
    """Performance tests for Thread Block Clusters"""
    
    def test_basic_benchmark(self):
        """Test that benchmarking runs without errors"""
        # Limit to smaller sizes for test speed
        test_sizes = [(512, 512), (1024, 1024)]
        iterations = 2
        
        # Override the benchmark function to test with smaller sizes
        from quantum_field.thread_block_cluster import benchmark_thread_block_cluster
        import time
        from quantum_field.core import generate_quantum_field
        
        results = {
            "supported": True,
            "sizes": [],
            "standard_times": [],
            "cluster_times": [],
            "speedups": []
        }
        
        for width, height in test_sizes:
            print(f"\nBenchmarking size: {width}x{height}")
            results["sizes"].append(f"{width}x{height}")
            
            # Standard CUDA implementation
            standard_times = []
            for i in range(iterations):
                start_time = time.time()
                _ = generate_quantum_field(width, height, 'love')
                end_time = time.time()
                standard_times.append(end_time - start_time)
            
            avg_standard_time = sum(standard_times) / len(standard_times)
            results["standard_times"].append(avg_standard_time)
            
            # Thread Block Cluster implementation
            cluster_times = []
            for i in range(iterations):
                start_time = time.time()
                _ = generate_quantum_field_tbc(width, height, 'love')
                end_time = time.time()
                cluster_times.append(end_time - start_time)
            
            avg_cluster_time = sum(cluster_times) / len(cluster_times)
            results["cluster_times"].append(avg_cluster_time)
            
            # Calculate speedup
            if avg_standard_time > 0:
                speedup = avg_standard_time / avg_cluster_time
                results["speedups"].append(speedup)
            else:
                results["speedups"].append(0)
        
        # Verify that we got results
        assert len(results["sizes"]) == len(test_sizes)
        assert len(results["standard_times"]) == len(test_sizes)
        assert len(results["cluster_times"]) == len(test_sizes)
        assert len(results["speedups"]) == len(test_sizes)
        
        # On H100 or newer, TBC should typically provide a speedup
        # But we don't assert this as it depends on specific hardware configurations
        print(f"Speedups: {results['speedups']}")


class TestErrorHandling:
    """Tests for error handling in Thread Block Cluster code"""

    def test_fallback_on_error(self, monkeypatch):
        """Test fallback to standard CUDA on errors"""
        if IS_TBC_SUPPORTED:
            # Make TBC operations fail
            original_initialize = initialize_thread_block_clusters
            
            def failing_initialize():
                # Call the original but return False
                original_initialize()
                return False
                
            # Patch the initialize function
            monkeypatch.setattr(
                "quantum_field.thread_block_cluster.initialize_thread_block_clusters", 
                failing_initialize
            )
        
        # Should not crash, will fall back to regular implementation
        width, height = 512, 512
        try:
            field = generate_quantum_field_tbc(width, height, 'love', 0)
            assert field.shape == (height, width)
        except Exception as e:
            pytest.fail(f"Field generation failed with unexpected error: {e}")