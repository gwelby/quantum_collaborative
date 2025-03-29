"""
Tests for the 3D field generation with CUDA Graphs support.
"""

import pytest
import numpy as np
import time
from quantum_field.backends.cuda import CUDABackend
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# Skip all tests if CUDA is not available
try:
    from cuda.core.experimental import Device
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


@pytest.fixture
def cuda_backend():
    """Create and initialize a CUDA backend instance."""
    backend = CUDABackend()
    assert backend.initialize(), "CUDA backend initialization failed"
    yield backend
    backend.shutdown()


def test_create_3d_graph(cuda_backend):
    """Test creation of a 3D CUDA graph."""
    graph_name = "test_3d_graph"
    width, height, depth = 32, 32, 16
    
    # Create a 3D graph
    result = cuda_backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name="love"
    )
    
    assert result is True, "Failed to create 3D CUDA graph"
    
    # Verify the graph was created correctly
    graphs = cuda_backend.list_cuda_graphs()
    assert len(graphs) == 1, "Graph was not registered correctly"
    
    graph_info = graphs[0]
    assert graph_info["name"] == graph_name
    assert graph_info["width"] == width
    assert graph_info["height"] == height
    assert graph_info["depth"] == depth
    assert graph_info["is_3d"] is True
    
    # Clean up
    assert cuda_backend.destroy_cuda_graph(graph_name)


def test_execute_3d_graph(cuda_backend):
    """Test execution of a 3D CUDA graph."""
    graph_name = "test_exec_3d"
    width, height, depth = 32, 32, 16
    
    # Create a 3D graph
    result = cuda_backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name="love"
    )
    
    assert result is True, "Failed to create 3D CUDA graph"
    
    # Execute the graph
    field = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
    
    # Verify the result
    assert field is not None, "Graph execution returned None"
    assert field.shape == (depth, height, width), f"Unexpected shape: {field.shape}"
    assert field.dtype == np.float32, f"Unexpected dtype: {field.dtype}"
    assert not np.isnan(field).any(), "Field contains NaN values"
    assert not np.isinf(field).any(), "Field contains Inf values"
    
    # Check value range
    assert -1.0 <= np.min(field) <= 1.0, f"Minimum value out of range: {np.min(field)}"
    assert -1.0 <= np.max(field) <= 1.0, f"Maximum value out of range: {np.max(field)}"
    
    # Clean up
    assert cuda_backend.destroy_cuda_graph(graph_name)


def test_time_evolution_with_3d_graph(cuda_backend):
    """Test time evolution with a 3D CUDA graph."""
    graph_name = "test_time_evolution_3d"
    width, height, depth = 32, 32, 16
    
    # Create a 3D graph
    result = cuda_backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name="love"
    )
    
    assert result is True, "Failed to create 3D CUDA graph"
    
    # Execute the graph with different time factors
    fields = {}
    time_factors = [0.0, 0.5, 1.0, 1.5]
    
    for time_factor in time_factors:
        fields[time_factor] = cuda_backend.execute_cuda_graph(graph_name, time_factor=time_factor)
    
    # Verify the results change over time
    for i, t1 in enumerate(time_factors):
        for t2 in time_factors[i+1:]:
            # Fields at different time steps should be different
            field1 = fields[t1]
            field2 = fields[t2]
            diff = np.abs(field1 - field2).mean()
            assert diff > 0.01, f"Fields at time {t1} and {t2} are too similar (diff={diff})"
    
    # Clean up
    assert cuda_backend.destroy_cuda_graph(graph_name)


def test_3d_graph_performance(cuda_backend):
    """Test that using CUDA Graphs improves performance for 3D fields."""
    width, height, depth = 64, 64, 32
    frequency_name = "love"
    iterations = 5
    
    # Create a graph
    graph_name = "perf_test_3d"
    cuda_backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name=frequency_name
    )
    
    # Time standard method (first call may include compilation time)
    standard_field = cuda_backend.generate_3d_quantum_field(
        width=width,
        height=height, 
        depth=depth,
        frequency_name=frequency_name
    )
    
    # Time standard method multiple times
    standard_start = time.time()
    for i in range(iterations):
        cuda_backend.generate_3d_quantum_field(
            width=width,
            height=height, 
            depth=depth,
            frequency_name=frequency_name,
            time_factor=i * 0.1
        )
    standard_time = time.time() - standard_start
    
    # Time graph method (first call)
    graph_field = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
    
    # Time graph method multiple times
    graph_start = time.time()
    for i in range(iterations):
        cuda_backend.execute_cuda_graph(graph_name, time_factor=i * 0.1)
    graph_time = time.time() - graph_start
    
    # Clean up
    cuda_backend.destroy_cuda_graph(graph_name)
    
    # Verify results match
    assert standard_field.shape == graph_field.shape
    assert np.allclose(standard_field, graph_field, rtol=1e-3, atol=1e-3)
    
    # Print performance info
    print(f"\nPerformance comparison for 3D fields ({width}x{height}x{depth}):")
    print(f"Standard method: {standard_time:.4f}s for {iterations} iterations")
    print(f"Graph method: {graph_time:.4f}s for {iterations} iterations")
    print(f"Speedup: {standard_time / graph_time:.2f}x")
    
    # Graph method should be faster
    assert graph_time < standard_time, "CUDA Graphs should be faster than standard method"


def test_multi_gpu_3d_graph():
    """Test multi-GPU support for 3D CUDA Graphs, if available."""
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    
    # Skip if multi-GPU is not available
    if not cuda_backend.multi_gpu_available or len(cuda_backend.devices) <= 1:
        cuda_backend.shutdown()
        pytest.skip("Multi-GPU support not available")
    
    # Parameters for a larger field that would benefit from multi-GPU
    width, height, depth = 128, 128, 64
    total_voxels = width * height * depth
    
    # Skip if system memory might be insufficient
    if total_voxels * 4 * 3 > 2 * 1024 * 1024 * 1024:  # More than 2GB needed
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            if available_memory < total_voxels * 4 * 3:
                cuda_backend.shutdown()
                pytest.skip("Insufficient system memory for large field test")
        except ImportError:
            pass
    
    try:
        # Create a multi-GPU 3D graph (explicitly request multi-GPU)
        graph_name = "multi_gpu_3d_test"
        result = cuda_backend.create_cuda_graph(
            graph_name=graph_name,
            width=width,
            height=height,
            depth=depth,
            frequency_name="love",
            use_multi_gpu=True
        )
        
        assert result is True, "Failed to create multi-GPU 3D graph"
        
        # Verify it was created as a multi-GPU graph
        graphs = cuda_backend.list_cuda_graphs()
        graph_info = next(g for g in graphs if g["name"] == graph_name)
        assert graph_info["type"] == "3d_multi_gpu"
        assert graph_info["is_3d"] is True
        assert graph_info.get("num_gpus", 0) > 1
        
        # Execute the graph
        field = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
        
        # Verify the result
        assert field is not None
        assert field.shape == (depth, height, width)
        assert not np.isnan(field).any()
    finally:
        # Clean up
        if graph_name in locals():
            cuda_backend.destroy_cuda_graph(graph_name)
        cuda_backend.shutdown()


if __name__ == "__main__":
    # Simple manual test
    backend = CUDABackend()
    if backend.is_available() and backend.initialize():
        # Create a 3D graph
        print("Creating 3D CUDA Graph...")
        graph_name = "test_3d"
        success = backend.create_cuda_graph(
            graph_name=graph_name,
            width=64,
            height=64,
            depth=32,
            frequency_name="love"
        )
        
        if success:
            print("3D CUDA Graph created successfully")
            
            # Execute the graph
            print("Executing 3D CUDA Graph...")
            field = backend.execute_cuda_graph(graph_name, time_factor=0.0)
            print(f"3D field shape: {field.shape}")
            print(f"Value range: [{field.min():.4f}, {field.max():.4f}]")
            
            # Performance test
            print("\nPerformance test:")
            iterations = 10
            
            # Standard method
            start = time.time()
            for i in range(iterations):
                backend.generate_3d_quantum_field(64, 64, 32, time_factor=i*0.1)
            std_time = time.time() - start
            print(f"Standard method: {std_time:.4f}s for {iterations} iterations")
            
            # Graph method
            start = time.time()
            for i in range(iterations):
                backend.execute_cuda_graph(graph_name, time_factor=i*0.1)
            graph_time = time.time() - start
            print(f"Graph method: {graph_time:.4f}s for {iterations} iterations")
            print(f"Speedup: {std_time / graph_time:.2f}x")
            
            # Clean up
            backend.destroy_cuda_graph(graph_name)
        else:
            print("Failed to create 3D CUDA Graph")
        
        backend.shutdown()
    else:
        print("CUDA not available")