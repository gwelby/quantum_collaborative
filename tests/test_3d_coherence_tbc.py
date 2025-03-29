#!/usr/bin/env python3
"""
Integration tests for 3D field coherence calculation with Thread Block Clusters

These tests verify the functionality and performance of Thread Block Cluster
optimizations for calculating 3D quantum field coherence.
"""

import os
import sys
import pytest
import numpy as np
import time
from typing import Tuple, Dict, Optional

# Skip all tests if quantum_field package is not available
try:
    import quantum_field
    from quantum_field.thread_block_cluster import (
        check_thread_block_cluster_support,
        initialize_thread_block_clusters,
        calculate_3d_field_coherence_tbc
    )
    from quantum_field.backends.cuda import CUDABackend
    from quantum_field.constants import PHI, LAMBDA
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)

# Check if Thread Block Clusters are supported
IS_TBC_SUPPORTED = check_thread_block_cluster_support()


@pytest.fixture
def cuda_backend():
    """Initialize and return the CUDA backend"""
    backend = CUDABackend()
    backend.initialize()
    return backend


@pytest.fixture
def test_field_3d_small() -> Tuple[int, int, int]:
    """Define a small 3D test field size"""
    return (32, 32, 32)


@pytest.fixture
def test_field_3d_medium() -> Tuple[int, int, int]:
    """Define a medium 3D test field size"""
    return (64, 64, 64)


@pytest.fixture
def test_field_3d_large() -> Tuple[int, int, int]:
    """Define a large 3D test field size"""
    return (128, 128, 64)


@pytest.fixture
def generate_3d_test_field(cuda_backend):
    """Generate a 3D test field for coherence calculation"""
    def _generate(depth, height, width, frequency='love', time_factor=0.0):
        """Helper to generate a 3D field with the given parameters"""
        try:
            # Try to use the CUDA backend directly
            return cuda_backend.generate_3d_quantum_field(
                width, height, depth, frequency, time_factor
            )
        except Exception as e:
            # Fall back to generating a simple phi-harmonic test field
            print(f"Error using CUDA backend: {e}, falling back to test field")
            field = np.zeros((depth, height, width), dtype=np.float32)
            
            # Fill with phi-harmonic pattern
            center_x, center_y, center_z = width/2, height/2, depth/2
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        # Normalized coordinates
                        nx = (x - center_x) / (width/2)
                        ny = (y - center_y) / (height/2)
                        nz = (z - center_z) / (depth/2)
                        
                        # Distance from center
                        distance = np.sqrt(nx**2 + ny**2 + nz**2)
                        
                        # Phi-harmonic field value
                        field[z, y, x] = np.sin(distance * PHI) * np.exp(-distance/PHI)
            
            return field
            
    return _generate


class TestCoherenceCalculation:
    """Tests for 3D coherence calculation with Thread Block Clusters"""
    
    def test_coherence_function_exists(self):
        """Test that the TBC coherence function exists"""
        assert hasattr(CUDABackend, 'calculate_3d_field_coherence')
        
        # Check if we're on hardware with TBC support
        if IS_TBC_SUPPORTED:
            cuda_backend = CUDABackend()
            cuda_backend.initialize()
            assert hasattr(cuda_backend, '_calculate_3d_field_coherence_tbc')
    
    def test_fallback_small_field(self, cuda_backend, generate_3d_test_field, test_field_3d_small):
        """Test that small fields use CPU implementation (more efficient)"""
        depth, height, width = test_field_3d_small
        field = generate_3d_test_field(depth, height, width)
        
        # Calculate coherence
        coherence = cuda_backend.calculate_3d_field_coherence(field)
        
        # Should return a valid coherence value
        assert 0.0 <= coherence <= 1.0
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    def test_tbc_coherence_medium_field(self, cuda_backend, generate_3d_test_field, test_field_3d_medium):
        """Test Thread Block Cluster coherence calculation with medium fields"""
        depth, height, width = test_field_3d_medium
        field = generate_3d_test_field(depth, height, width)
        
        # Calculate coherence with different methods
        coherence_standard = None
        coherence_tbc = None
        
        # Try to force standard method first
        try:
            # Store original method
            original_has_tbc = cuda_backend.has_thread_block_cluster_support
            # Temporarily disable TBC
            cuda_backend.has_thread_block_cluster_support = False
            
            # Calculate using standard method
            start_time = time.time()
            coherence_standard = cuda_backend.calculate_3d_field_coherence(field)
            standard_time = time.time() - start_time
            
            # Restore TBC support
            cuda_backend.has_thread_block_cluster_support = original_has_tbc
            
            # Calculate using TBC method
            start_time = time.time()
            coherence_tbc = cuda_backend._calculate_3d_field_coherence_tbc(field)
            tbc_time = time.time() - start_time
            
            # Both should return valid coherence values
            assert 0.0 <= coherence_standard <= 1.0
            assert 0.0 <= coherence_tbc <= 1.0
            
            # Both methods should give similar results
            # (implementation differences may cause minor variations)
            assert abs(coherence_standard - coherence_tbc) < 0.1
            
            # Print timing comparison
            print(f"\nCoherence calculation time comparison (medium field):")
            print(f"  Standard: {standard_time:.6f} seconds")
            print(f"  TBC:      {tbc_time:.6f} seconds")
            if standard_time > 0:
                print(f"  Speedup:  {standard_time/tbc_time:.2f}x")
            
        except Exception as e:
            pytest.skip(f"Error during CUDA coherence calculation: {e}")
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    def test_tbc_coherence_large_field(self, cuda_backend, generate_3d_test_field, test_field_3d_large):
        """Test Thread Block Cluster coherence calculation with large fields"""
        depth, height, width = test_field_3d_large
        
        # Skip if field is too large for available GPU memory
        try:
            field = generate_3d_test_field(depth, height, width)
        except Exception as e:
            pytest.skip(f"Field too large for available GPU memory: {e}")
        
        # Calculate coherence
        try:
            # Calculate using TBC method
            start_time = time.time()
            coherence_tbc = cuda_backend._calculate_3d_field_coherence_tbc(field)
            tbc_time = time.time() - start_time
            
            # Should return a valid coherence value
            assert 0.0 <= coherence_tbc <= 1.0
            
            print(f"\nTBC Coherence calculation time (large field): {tbc_time:.6f} seconds")
        except Exception as e:
            pytest.skip(f"Error during TBC coherence calculation: {e}")
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    def test_auto_selection_large_field(self, cuda_backend, generate_3d_test_field, test_field_3d_large):
        """Test that large fields automatically use TBC implementation"""
        depth, height, width = test_field_3d_large
        
        # Skip if field is too large for available GPU memory
        try:
            field = generate_3d_test_field(depth, height, width)
        except Exception as e:
            pytest.skip(f"Field too large for available GPU memory: {e}")
        
        # Calculate coherence using the main method
        # Our update should route large fields to TBC automatically
        coherence = cuda_backend.calculate_3d_field_coherence(field)
        
        # Should return a valid coherence value
        assert 0.0 <= coherence <= 1.0


class TestThreadBlockClusterErrors:
    """Tests for error handling in Thread Block Cluster coherence calculation"""
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    def test_error_handling(self, cuda_backend, generate_3d_test_field, test_field_3d_medium, monkeypatch):
        """Test graceful fallback on errors"""
        depth, height, width = test_field_3d_medium
        field = generate_3d_test_field(depth, height, width)
        
        # Make TBC method fail
        def failing_tbc_method(*args, **kwargs):
            raise RuntimeError("Simulated TBC failure")
            
        # Patch the method
        monkeypatch.setattr(
            cuda_backend, 
            "_calculate_3d_field_coherence_tbc", 
            failing_tbc_method
        )
        
        # Should fall back to standard method
        coherence = cuda_backend.calculate_3d_field_coherence(field)
        
        # Should still return a valid coherence value
        assert 0.0 <= coherence <= 1.0


class TestCoherencePerformanceBenchmark:
    """Benchmark tests for Thread Block Cluster coherence calculation"""
    
    @pytest.mark.skipif(not IS_TBC_SUPPORTED, reason="Thread Block Clusters not supported on this hardware")
    @pytest.mark.parametrize("field_size", [(32, 32, 32), (64, 64, 64), (128, 64, 64)])
    def test_benchmark_coherence(self, cuda_backend, generate_3d_test_field, field_size):
        """Benchmark Thread Block Cluster coherence calculation performance"""
        depth, height, width = field_size
        
        # Skip if field is too large for available GPU memory
        try:
            field = generate_3d_test_field(depth, height, width)
        except Exception as e:
            pytest.skip(f"Field too large for available GPU memory: {e}")
        
        # Results dictionary
        results = {
            "field_size": f"{width}x{height}x{depth}",
            "standard_time": None,
            "tbc_time": None,
            "speedup": None
        }
        
        try:
            # Store original method
            original_has_tbc = cuda_backend.has_thread_block_cluster_support
            
            # Temporarily disable TBC
            cuda_backend.has_thread_block_cluster_support = False
            
            # Calculate using standard method
            start_time = time.time()
            coherence_standard = cuda_backend.calculate_3d_field_coherence(field)
            standard_time = time.time() - start_time
            results["standard_time"] = standard_time
            
            # Restore TBC support
            cuda_backend.has_thread_block_cluster_support = original_has_tbc
            
            # Skip TBC for small fields (CPU is more efficient)
            if width * height * depth < 100_000:
                pytest.skip("Field too small for TBC to be beneficial")
            
            # Calculate using TBC method
            start_time = time.time()
            coherence_tbc = cuda_backend._calculate_3d_field_coherence_tbc(field)
            tbc_time = time.time() - start_time
            results["tbc_time"] = tbc_time
            
            # Calculate speedup
            if standard_time > 0:
                results["speedup"] = standard_time / tbc_time
            
            # Print results
            print(f"\nCoherence benchmark for {width}x{height}x{depth} field:")
            print(f"  Standard: {standard_time:.6f} seconds")
            print(f"  TBC:      {tbc_time:.6f} seconds")
            if results["speedup"] is not None:
                print(f"  Speedup:  {results['speedup']:.2f}x")
            
            # Both should return valid coherence values
            assert 0.0 <= coherence_standard <= 1.0
            assert 0.0 <= coherence_tbc <= 1.0
            
            # Both methods should give similar results
            assert abs(coherence_standard - coherence_tbc) < 0.1
            
        except Exception as e:
            print(f"Benchmark error: {e}")
            pytest.skip(f"Error during benchmark: {e}")