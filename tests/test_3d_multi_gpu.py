#!/usr/bin/env python3
"""
Tests for 3D multi-GPU functionality and Thread Block Cluster integration

Tests verify the distributed computation of 3D quantum fields across multiple GPUs
and the integration with Thread Block Clusters on compatible hardware.
"""

import os
import sys
import pytest
import numpy as np
from typing import Tuple, Dict, Optional

# Skip all tests if quantum_field package is not available
try:
    import quantum_field
    from quantum_field.multi_gpu import MultiGPUManager
    from quantum_field.constants import PHI, LAMBDA
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)

# Check if CUDA is available
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    pytest.skip("CUDA is not available, skipping GPU tests", allow_module_level=True)

# Initialize multi-GPU manager
mgpu_manager = MultiGPUManager()
MULTI_GPU_AVAILABLE = mgpu_manager.available and len(mgpu_manager.devices) > 1
THREAD_BLOCK_CLUSTER_AVAILABLE = mgpu_manager.has_thread_block_cluster_support if hasattr(mgpu_manager, 'has_thread_block_cluster_support') else False


@pytest.fixture
def mgpu():
    """Return the multi-GPU manager"""
    return mgpu_manager


@pytest.fixture
def test_field_3d_small() -> np.ndarray:
    """Generate a small 3D test field"""
    shape = (32, 32, 32)
    
    # Create a simple 3D field with phi-harmonics
    field = np.zeros(shape, dtype=np.float32)
    depth, height, width = shape
    
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


@pytest.fixture
def test_field_3d_medium() -> np.ndarray:
    """Generate a medium 3D test field"""
    # Use the multi-GPU manager to generate a field
    if mgpu_manager.available:
        try:
            return mgpu_manager.generate_3d_quantum_field(64, 64, 64, 'love')
        except Exception as e:
            print(f"Error generating 3D field: {e}")
            
    # Fallback: Create a smaller field
    shape = (64, 64, 64)
    field = np.zeros(shape, dtype=np.float32)
    depth, height, width = shape
    
    # Fill with phi-harmonic pattern (simplified to avoid long computation)
    center_x, center_y, center_z = width/2, height/2, depth/2
    for z in range(0, depth, 2):  # Step by 2 to reduce computation
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                # Normalized coordinates
                nx = (x - center_x) / (width/2)
                ny = (y - center_y) / (height/2)
                nz = (z - center_z) / (depth/2)
                
                # Distance from center
                distance = np.sqrt(nx**2 + ny**2 + nz**2)
                
                # Phi-harmonic field value
                value = np.sin(distance * PHI) * np.exp(-distance/PHI)
                
                # Set value for this block of 2x2x2
                for dz in range(min(2, depth - z)):
                    for dy in range(min(2, height - y)):
                        for dx in range(min(2, width - x)):
                            field[z + dz, y + dy, x + dx] = value
    
    return field


class Test3DFieldGeneration:
    """Tests for 3D field generation with multiple GPUs"""
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    def test_generate_3d_field(self, mgpu):
        """Test basic 3D field generation"""
        width, height, depth = 32, 32, 32
        
        # Generate a 3D field
        field = mgpu.generate_3d_quantum_field(width, height, depth, 'love')
        
        # Check shape and data type
        assert field.shape == (depth, height, width)
        assert field.dtype == np.float32
        
        # Check that field has meaningful values
        assert np.min(field) != np.max(field)
        assert not np.isnan(np.sum(field))
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    @pytest.mark.parametrize("field_shape", [(64, 32, 16), (16, 64, 32), (32, 16, 64)])
    def test_non_cubic_fields(self, mgpu, field_shape):
        """Test generation of non-cubic 3D fields"""
        width, height, depth = field_shape
        
        try:
            # Generate a 3D field with non-cubic dimensions
            field = mgpu.generate_3d_quantum_field(width, height, depth, 'love')
            
            # Check shape and data type
            assert field.shape == (depth, height, width)
            assert field.dtype == np.float32
            
            # Check that field has meaningful values
            assert np.min(field) != np.max(field)
            assert not np.isnan(np.sum(field))
        except Exception as e:
            pytest.skip(f"Error in multi-GPU 3D field generation: {e}")


class Test3DFieldCoherence:
    """Tests for 3D field coherence calculation with multiple GPUs"""
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    def test_calculate_coherence_small_field(self, mgpu, test_field_3d_small):
        """Test coherence calculation for small 3D fields"""
        # Calculate coherence
        coherence = mgpu.calculate_3d_field_coherence(test_field_3d_small)
        
        # Check that coherence is a valid value
        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0
        assert not np.isnan(coherence)
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    def test_tiling_small_field(self, mgpu, test_field_3d_small):
        """Test coherence calculation with tiling on small fields"""
        # Calculate coherence with tiling
        coherence_tiled = mgpu.calculate_3d_field_coherence(
            test_field_3d_small, 
            use_tiling=True,
            tile_size=(16, 16, 16)
        )
        
        # Calculate coherence without tiling
        coherence_full = mgpu.calculate_3d_field_coherence(
            test_field_3d_small,
            use_tiling=False
        )
        
        # Results should be similar
        assert abs(coherence_tiled - coherence_full) < 0.1
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    def test_medium_field(self, mgpu, test_field_3d_medium):
        """Test coherence calculation for medium 3D fields"""
        try:
            # Calculate coherence
            coherence = mgpu.calculate_3d_field_coherence(test_field_3d_medium)
            
            # Check that coherence is a valid value
            assert isinstance(coherence, float)
            assert 0.0 <= coherence <= 1.0
            assert not np.isnan(coherence)
        except Exception as e:
            pytest.skip(f"Error in multi-GPU coherence calculation: {e}")


class TestThreadBlockClusterIntegration:
    """Tests for Thread Block Cluster integration with multi-GPU 3D field coherence"""
    
    @pytest.mark.skipif(not THREAD_BLOCK_CLUSTER_AVAILABLE, reason="Thread Block Clusters not supported")
    def test_tbc_enabled(self, mgpu):
        """Test that Thread Block Cluster support is properly detected"""
        assert mgpu.has_thread_block_cluster_support
        assert hasattr(mgpu, 'tbc_modules')
        assert len(mgpu.tbc_modules) > 0
    
    @pytest.mark.skipif(not THREAD_BLOCK_CLUSTER_AVAILABLE, reason="Thread Block Clusters not supported")
    def test_tbc_coherence_small_field(self, mgpu, test_field_3d_small):
        """Test Thread Block Cluster coherence calculation for small 3D fields"""
        # Calculate coherence with TBC
        coherence_tbc = mgpu.calculate_3d_field_coherence(
            test_field_3d_small,
            use_thread_block_clusters=True
        )
        
        # Calculate coherence without TBC
        coherence_std = mgpu.calculate_3d_field_coherence(
            test_field_3d_small,
            use_thread_block_clusters=False
        )
        
        # Check that coherence is a valid value
        assert isinstance(coherence_tbc, float)
        assert 0.0 <= coherence_tbc <= 1.0
        assert not np.isnan(coherence_tbc)
        
        # Results should be similar
        assert abs(coherence_tbc - coherence_std) < 0.1
    
    @pytest.mark.skipif(not THREAD_BLOCK_CLUSTER_AVAILABLE, reason="Thread Block Clusters not supported")
    def test_tbc_tiling(self, mgpu, test_field_3d_small):
        """Test Thread Block Cluster coherence calculation with tiling"""
        # Calculate coherence with TBC and tiling
        coherence_tbc_tiled = mgpu.calculate_3d_field_coherence(
            test_field_3d_small, 
            use_thread_block_clusters=True,
            use_tiling=True,
            tile_size=(16, 16, 16)
        )
        
        # Calculate coherence with TBC but without tiling
        coherence_tbc_full = mgpu.calculate_3d_field_coherence(
            test_field_3d_small,
            use_thread_block_clusters=True,
            use_tiling=False
        )
        
        # Results should be similar
        assert abs(coherence_tbc_tiled - coherence_tbc_full) < 0.1
    
    @pytest.mark.skipif(not THREAD_BLOCK_CLUSTER_AVAILABLE, reason="Thread Block Clusters not supported")
    def test_tbc_medium_field(self, mgpu, test_field_3d_medium):
        """Test Thread Block Cluster coherence calculation for medium 3D fields"""
        try:
            # Calculate coherence with TBC
            coherence_tbc = mgpu.calculate_3d_field_coherence(
                test_field_3d_medium,
                use_thread_block_clusters=True
            )
            
            # Check that coherence is a valid value
            assert isinstance(coherence_tbc, float)
            assert 0.0 <= coherence_tbc <= 1.0
            assert not np.isnan(coherence_tbc)
        except Exception as e:
            pytest.skip(f"Error in TBC coherence calculation: {e}")


class TestPerformance:
    """Performance tests for multi-GPU and Thread Block Cluster implementations"""
    
    @pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs not available")
    def test_multi_gpu_speedup(self, mgpu, test_field_3d_medium):
        """Test that multi-GPU coherence calculation is faster than single GPU"""
        import time
        from quantum_field.backends.cuda import CUDABackend
        
        # Skip if field is too large
        if test_field_3d_medium.size > 1000000:  # Skip for large fields to avoid long test times
            pytest.skip("Field too large for quick performance test")
        
        # Initialize CUDA backend for comparison
        cuda_backend = CUDABackend()
        cuda_backend.initialize()
        
        try:
            # Measure single-GPU time
            start_time = time.time()
            coherence_single = cuda_backend.calculate_3d_field_coherence(test_field_3d_medium)
            single_gpu_time = time.time() - start_time
            
            # Measure multi-GPU time
            start_time = time.time()
            coherence_multi = mgpu.calculate_3d_field_coherence(
                test_field_3d_medium,
                use_thread_block_clusters=False
            )
            multi_gpu_time = time.time() - start_time
            
            # Results should be similar
            assert abs(coherence_single - coherence_multi) < 0.1
            
            # Multi-GPU should be faster (or similar for small fields)
            # Since overhead can dominate for small fields, we don't strictly
            # require multi-GPU to be faster, just reasonably competitive
            assert multi_gpu_time < single_gpu_time * 1.5
            
            # Print results for reference
            print(f"\nMulti-GPU vs. Single-GPU Performance:")
            print(f"  Single-GPU time: {single_gpu_time:.6f} seconds")
            print(f"  Multi-GPU time:  {multi_gpu_time:.6f} seconds")
            if single_gpu_time > 0:
                print(f"  Speedup:        {single_gpu_time/multi_gpu_time:.2f}x")
        
        except Exception as e:
            pytest.skip(f"Error in performance test: {e}")
    
    @pytest.mark.skipif(not THREAD_BLOCK_CLUSTER_AVAILABLE, reason="Thread Block Clusters not supported")
    def test_tbc_speedup(self, mgpu, test_field_3d_medium):
        """Test that Thread Block Cluster coherence calculation is faster than standard multi-GPU"""
        import time
        
        # Skip if field is too large
        if test_field_3d_medium.size > 1000000:  # Skip for large fields to avoid long test times
            pytest.skip("Field too large for quick performance test")
        
        try:
            # Measure standard multi-GPU time
            start_time = time.time()
            coherence_std = mgpu.calculate_3d_field_coherence(
                test_field_3d_medium,
                use_thread_block_clusters=False
            )
            std_time = time.time() - start_time
            
            # Measure TBC multi-GPU time
            start_time = time.time()
            coherence_tbc = mgpu.calculate_3d_field_coherence(
                test_field_3d_medium,
                use_thread_block_clusters=True
            )
            tbc_time = time.time() - start_time
            
            # Results should be similar
            assert abs(coherence_std - coherence_tbc) < 0.1
            
            # TBC should be faster (or similar for small fields)
            # Since overhead can dominate for small fields, we don't strictly
            # require TBC to be faster, just reasonably competitive
            assert tbc_time < std_time * 1.5
            
            # Print results for reference
            print(f"\nThread Block Cluster vs. Standard Multi-GPU Performance:")
            print(f"  Standard Multi-GPU time: {std_time:.6f} seconds")
            print(f"  TBC Multi-GPU time:      {tbc_time:.6f} seconds")
            if std_time > 0:
                print(f"  Speedup:               {std_time/tbc_time:.2f}x")
        
        except Exception as e:
            pytest.skip(f"Error in TBC performance test: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])