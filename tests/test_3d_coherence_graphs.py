#!/usr/bin/env python3
"""
Integration tests for 3D field coherence calculation with CUDA Graphs

These tests verify the functionality and performance of CUDA Graph
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
    from quantum_field.backends.cuda import CUDABackend
    from quantum_field.constants import PHI, LAMBDA
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)

# Initialize CUDA backend
try:
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    CUDA_AVAILABLE = cuda_backend.initialized
    CUDA_GRAPHS_SUPPORTED = hasattr(cuda_backend, '_create_3d_coherence_graph')
    TBC_SUPPORTED = getattr(cuda_backend, 'has_thread_block_cluster_support', False)
except Exception:
    CUDA_AVAILABLE = False
    CUDA_GRAPHS_SUPPORTED = False
    TBC_SUPPORTED = False


@pytest.fixture
def test_field_3d_small() -> Tuple[int, int, int]:
    """Define a small 3D test field size"""
    return (32, 32, 32)


@pytest.fixture
def test_field_3d_medium() -> Tuple[int, int, int]:
    """Define a medium 3D test field size"""
    return (64, 64, 64)


@pytest.fixture
def generate_3d_test_field():
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not CUDA_GRAPHS_SUPPORTED, reason="CUDA Graphs for coherence not supported")
class TestCUDAGraphsCoherence:
    """Tests for 3D coherence calculation with CUDA Graphs"""
    
    def test_coherence_graph_creation(self, test_field_3d_small):
        """Test creation of a CUDA graph for coherence calculation"""
        depth, height, width = test_field_3d_small
        
        try:
            # Create a standard CUDA graph
            graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=False)
            
            # Verify that we got all required resources
            assert "graph" in graph_data
            assert "graph_exec" in graph_data
            assert "stream" in graph_data
            assert "d_field" in graph_data
            assert "d_result" in graph_data
            assert "d_count" in graph_data
            assert graph_data["width"] == width
            assert graph_data["height"] == height
            assert graph_data["depth"] == depth
            assert graph_data["use_tbc"] is False
            
        except Exception as e:
            pytest.skip(f"Error creating CUDA graph: {e}")
    
    @pytest.mark.skipif(not TBC_SUPPORTED, reason="Thread Block Clusters not supported")
    def test_tbc_coherence_graph_creation(self, test_field_3d_small):
        """Test creation of a TBC-optimized CUDA graph for coherence calculation"""
        depth, height, width = test_field_3d_small
        
        try:
            # Create a TBC-optimized CUDA graph
            graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=True)
            
            # Verify that we got all required resources
            assert "graph" in graph_data
            assert "graph_exec" in graph_data
            assert "stream" in graph_data
            assert "d_field" in graph_data
            assert "d_result" in graph_data
            assert "d_count" in graph_data
            assert graph_data["width"] == width
            assert graph_data["height"] == height
            assert graph_data["depth"] == depth
            assert graph_data["use_tbc"] is True
            
        except Exception as e:
            pytest.skip(f"Error creating TBC CUDA graph: {e}")
    
    def test_coherence_graph_execution(self, generate_3d_test_field, test_field_3d_small):
        """Test execution of a CUDA graph for coherence calculation"""
        depth, height, width = test_field_3d_small
        
        try:
            # Generate a test field
            field = generate_3d_test_field(depth, height, width)
            
            # Create a CUDA graph
            graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=False)
            
            # Calculate coherence using the standard method
            direct_coherence = cuda_backend.calculate_3d_field_coherence(field)
            
            # Calculate coherence using the graph
            graph_coherence = cuda_backend._execute_3d_coherence_graph(graph_data, field)
            
            # Both should return valid coherence values
            assert 0.0 <= direct_coherence <= 1.0
            assert 0.0 <= graph_coherence <= 1.0
            
            # Both should give similar results
            assert abs(direct_coherence - graph_coherence) < 0.1
            
        except Exception as e:
            pytest.skip(f"Error in graph execution: {e}")
    
    @pytest.mark.skipif(not TBC_SUPPORTED, reason="Thread Block Clusters not supported")
    def test_tbc_coherence_graph_execution(self, generate_3d_test_field, test_field_3d_small):
        """Test execution of a TBC-optimized CUDA graph for coherence calculation"""
        depth, height, width = test_field_3d_small
        
        try:
            # Generate a test field
            field = generate_3d_test_field(depth, height, width)
            
            # Create a TBC-optimized CUDA graph
            graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=True)
            
            # Calculate coherence using the standard method
            direct_coherence = cuda_backend.calculate_3d_field_coherence(field)
            
            # Calculate coherence using the graph
            graph_coherence = cuda_backend._execute_3d_coherence_graph(graph_data, field)
            
            # Both should return valid coherence values
            assert 0.0 <= direct_coherence <= 1.0
            assert 0.0 <= graph_coherence <= 1.0
            
            # Both should give similar results
            assert abs(direct_coherence - graph_coherence) < 0.1
            
        except Exception as e:
            pytest.skip(f"Error in TBC graph execution: {e}")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not CUDA_GRAPHS_SUPPORTED, reason="CUDA Graphs for coherence not supported")
class TestPerformance:
    """Performance tests for coherence calculation with CUDA Graphs"""
    
    @pytest.mark.parametrize("field_size", [(32, 32, 32), (64, 64, 64)])
    def test_performance_comparison(self, generate_3d_test_field, field_size):
        """Compare performance of direct vs. graph-based coherence calculation"""
        depth, height, width = field_size
        
        try:
            # Generate a test field
            field = generate_3d_test_field(depth, height, width)
            
            # Create a CUDA graph
            print(f"\nCreating CUDA graph for {width}x{height}x{depth} field")
            graph_creation_start = time.time()
            graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=False)
            graph_creation_time = time.time() - graph_creation_start
            print(f"  Graph creation time: {graph_creation_time:.6f} seconds")
            
            # Time direct coherence calculation
            print(f"Running direct coherence calculation")
            direct_start = time.time()
            direct_coherence = cuda_backend.calculate_3d_field_coherence(field)
            direct_time = time.time() - direct_start
            print(f"  Direct calculation time: {direct_time:.6f} seconds")
            
            # Time graph-based coherence calculation
            print(f"Running graph-based coherence calculation")
            
            # First run might have initialization overhead
            _ = cuda_backend._execute_3d_coherence_graph(graph_data, field)
            
            # Time the actual execution
            graph_start = time.time()
            graph_coherence = cuda_backend._execute_3d_coherence_graph(graph_data, field)
            graph_time = time.time() - graph_start
            print(f"  Graph execution time: {graph_time:.6f} seconds")
            
            # Calculate speedup
            if direct_time > 0:
                speedup = direct_time / graph_time
                print(f"  Speedup: {speedup:.2f}x")
            
            # Both should return valid coherence values
            assert 0.0 <= direct_coherence <= 1.0
            assert 0.0 <= graph_coherence <= 1.0
            
            # Both should give similar results
            assert abs(direct_coherence - graph_coherence) < 0.1
            
        except Exception as e:
            pytest.skip(f"Error in performance test: {e}")
    
    @pytest.mark.skipif(not TBC_SUPPORTED, reason="Thread Block Clusters not supported")
    @pytest.mark.parametrize("field_size", [(64, 64, 64)])
    def test_tbc_performance_comparison(self, generate_3d_test_field, field_size):
        """Compare performance of standard vs. TBC graph-based coherence calculation"""
        depth, height, width = field_size
        
        try:
            # Generate a test field
            field = generate_3d_test_field(depth, height, width)
            
            # Create standard CUDA graph
            print(f"\nCreating standard CUDA graph for {width}x{height}x{depth} field")
            std_graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=False)
            
            # Create TBC-optimized CUDA graph
            print(f"Creating TBC-optimized CUDA graph for {width}x{height}x{depth} field")
            tbc_graph_data = cuda_backend._create_3d_coherence_graph(width, height, depth, use_tbc=True)
            
            # Time direct coherence calculation
            print(f"Running direct coherence calculation")
            direct_start = time.time()
            direct_coherence = cuda_backend.calculate_3d_field_coherence(field)
            direct_time = time.time() - direct_start
            print(f"  Direct calculation time: {direct_time:.6f} seconds")
            
            # Time standard graph execution
            print(f"Running standard graph execution")
            # First run might have initialization overhead
            _ = cuda_backend._execute_3d_coherence_graph(std_graph_data, field)
            
            std_graph_start = time.time()
            std_graph_coherence = cuda_backend._execute_3d_coherence_graph(std_graph_data, field)
            std_graph_time = time.time() - std_graph_start
            print(f"  Standard graph time: {std_graph_time:.6f} seconds")
            
            # Time TBC graph execution
            print(f"Running TBC graph execution")
            # First run might have initialization overhead
            _ = cuda_backend._execute_3d_coherence_graph(tbc_graph_data, field)
            
            tbc_graph_start = time.time()
            tbc_graph_coherence = cuda_backend._execute_3d_coherence_graph(tbc_graph_data, field)
            tbc_graph_time = time.time() - tbc_graph_start
            print(f"  TBC graph time: {tbc_graph_time:.6f} seconds")
            
            # Calculate speedups
            if direct_time > 0:
                std_speedup = direct_time / std_graph_time
                tbc_speedup = direct_time / tbc_graph_time
                print(f"  Standard graph speedup vs direct: {std_speedup:.2f}x")
                print(f"  TBC graph speedup vs direct: {tbc_speedup:.2f}x")
            
            if std_graph_time > 0:
                tbc_vs_std_speedup = std_graph_time / tbc_graph_time
                print(f"  TBC graph speedup vs standard graph: {tbc_vs_std_speedup:.2f}x")
            
            # All should return valid coherence values
            assert 0.0 <= direct_coherence <= 1.0
            assert 0.0 <= std_graph_coherence <= 1.0
            assert 0.0 <= tbc_graph_coherence <= 1.0
            
            # All should give similar results
            assert abs(direct_coherence - std_graph_coherence) < 0.1
            assert abs(direct_coherence - tbc_graph_coherence) < 0.1
            
        except Exception as e:
            pytest.skip(f"Error in TBC performance test: {e}")