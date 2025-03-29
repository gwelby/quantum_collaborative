"""
Tests for 3D quantum field generation using Thread Block Clusters
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

# Skip if Thread Block Clusters are not available
pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
]


@pytest.fixture
def cuda_backend():
    """Create and initialize a CUDA backend instance."""
    backend = CUDABackend()
    assert backend.initialize(), "CUDA backend initialization failed"
    
    # Skip if TBC not available
    if not backend.thread_block_clusters_available:
        pytest.skip("Thread Block Clusters not available")
        
    yield backend
    backend.shutdown()


def test_tbc_3d_capability(cuda_backend):
    """Test that thread block clusters for 3D fields are properly detected."""
    assert cuda_backend.thread_block_clusters_available, "Thread Block Clusters should be available"
    assert cuda_backend.tbc_module is not None, "Thread Block Cluster module should be initialized"
    
    # Verify that the 3D TBC kernel was compiled successfully
    try:
        kernel = cuda_backend.tbc_module.get_kernel("generate_3d_quantum_field_tbc")
        assert kernel is not None, "3D TBC kernel should be available"
    except Exception as e:
        pytest.fail(f"Failed to get 3D TBC kernel: {e}")


def test_generate_3d_field_with_tbc(cuda_backend):
    """Test generating a 3D field using Thread Block Clusters."""
    # Parameters
    width, height, depth = 128, 128, 64
    
    try:
        # Generate field using Thread Block Clusters
        field = cuda_backend._generate_3d_field_tbc(
            width, height, depth, 
            frequency=SACRED_FREQUENCIES['love'], 
            time_factor=0.0
        )
        
        # Check basic properties
        assert field.shape == (depth, height, width), f"Expected shape {(depth, height, width)}, got {field.shape}"
        assert not np.isnan(field).any(), "Field contains NaN values"
        assert not np.isinf(field).any(), "Field contains Inf values"
        
        # Check value range
        assert -1.0 <= np.min(field) <= 1.0, f"Field values out of range: min={np.min(field)}"
        assert -1.0 <= np.max(field) <= 1.0, f"Field values out of range: max={np.max(field)}"
        
    except Exception as e:
        pytest.fail(f"Failed to generate 3D field with TBC: {e}")


def test_3d_field_dispatch(cuda_backend):
    """Test that the dispatcher correctly chooses TBC for large 3D fields."""
    # This size should be enough to trigger TBC
    width, height, depth = 128, 128, 128
    
    # First, save the original method to restore it later
    original_tbc_method = cuda_backend._generate_3d_field_tbc
    
    # Replace with a version that increments a counter
    call_count = [0]
    
    def mock_tbc_method(*args, **kwargs):
        call_count[0] += 1
        return original_tbc_method(*args, **kwargs)
    
    cuda_backend._generate_3d_field_tbc = mock_tbc_method
    
    try:
        # Generate field - should use TBC
        field = cuda_backend.generate_3d_quantum_field(
            width, height, depth,
            frequency_name='love'
        )
        
        # Check if TBC method was called
        assert call_count[0] > 0, "TBC method should have been called for large 3D field"
        
        # Basic verification of result
        assert field.shape == (depth, height, width)
        
    finally:
        # Restore original method
        cuda_backend._generate_3d_field_tbc = original_tbc_method


def test_cuda_graphs_with_tbc(cuda_backend):
    """Test CUDA graphs with Thread Block Clusters for 3D fields."""
    # Test parameters
    width, height, depth = 64, 64, 64
    
    # Create a graph using TBC explicitly
    graph_name = "tbc_3d_test"
    cuda_backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name="love",
        use_tbc=True
    )
    
    try:
        # Verify it was created as a TBC graph
        graphs = cuda_backend.list_cuda_graphs()
        graph_info = next(g for g in graphs if g["name"] == graph_name)
        assert graph_info["type"] == "3d_tbc", f"Graph type should be 3d_tbc, got {graph_info['type']}"
        assert graph_info["is_3d"] is True
        
        # Execute the graph
        field = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
        
        # Verify the result
        assert field is not None
        assert field.shape == (depth, height, width)
        assert not np.isnan(field).any()
        
        # Test time evolution
        fields = []
        for t in [0.0, 0.5, 1.0]:
            field_t = cuda_backend.execute_cuda_graph(graph_name, time_factor=t)
            fields.append(field_t)
        
        # Fields should differ with time evolution
        assert np.abs(fields[0] - fields[1]).mean() > 0.01
        assert np.abs(fields[1] - fields[2]).mean() > 0.01
        
    finally:
        # Clean up
        cuda_backend.destroy_cuda_graph(graph_name)


def test_performance_comparison():
    """Compare performance of TBC vs standard implementation for 3D fields."""
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    if not cuda_backend.initialize():
        pytest.skip("CUDA backend initialization failed")
    
    # Skip if TBC not available
    if not cuda_backend.thread_block_clusters_available:
        cuda_backend.shutdown()
        pytest.skip("Thread Block Clusters not available")
    
    try:
        # Test parameters
        width, height, depth = 128, 128, 64
        iterations = 3
        
        # Time standard implementation
        start_standard = time.time()
        for _ in range(iterations):
            field_standard = cuda_backend._generate_3d_field_single_gpu(
                width, height, depth,
                frequency=SACRED_FREQUENCIES['love'],
                time_factor=0.0
            )
        time_standard = (time.time() - start_standard) / iterations
        
        # Time TBC implementation
        start_tbc = time.time()
        for _ in range(iterations):
            field_tbc = cuda_backend._generate_3d_field_tbc(
                width, height, depth,
                frequency=SACRED_FREQUENCIES['love'],
                time_factor=0.0
            )
        time_tbc = (time.time() - start_tbc) / iterations
        
        # CUDA graph with TBC
        graph_name = "perf_test_tbc"
        cuda_backend.create_cuda_graph(
            graph_name=graph_name,
            width=width,
            height=height,
            depth=depth,
            frequency_name="love",
            use_tbc=True
        )
        
        # Warm-up call
        _ = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
        
        # Time graph execution
        start_graph = time.time()
        for _ in range(iterations):
            field_graph = cuda_backend.execute_cuda_graph(graph_name, time_factor=0.0)
        time_graph = (time.time() - start_graph) / iterations
        
        # Clean up graph
        cuda_backend.destroy_cuda_graph(graph_name)
        
        # Print performance comparison
        print(f"\n --- Performance comparison (128x128x64 field) ---")
        print(f"Standard 3D: {time_standard:.4f}s per field")
        print(f"Thread Block Clusters 3D: {time_tbc:.4f}s per field (speedup: {time_standard/time_tbc:.2f}x)")
        print(f"CUDA Graph with TBC: {time_graph:.4f}s per field (speedup: {time_standard/time_graph:.2f}x)")
        
        # Verify results are similar
        assert np.allclose(field_standard, field_tbc, atol=1e-3)
        assert np.allclose(field_standard, field_graph, atol=1e-3)
        
        # TBC should be faster than standard
        assert time_tbc < time_standard, "TBC implementation should be faster than standard"
        
        # Graph with TBC should be fastest
        assert time_graph < time_tbc, "CUDA Graph with TBC should be faster than regular TBC"
        
    finally:
        cuda_backend.shutdown()


if __name__ == "__main__":
    # Run a simple test if TBC is available
    backend = CUDABackend()
    if not backend.initialize():
        print("CUDA backend initialization failed")
        exit(1)
    
    if not backend.thread_block_clusters_available:
        print("Thread Block Clusters not available on this GPU")
        backend.shutdown()
        exit(0)
    
    print(f"Testing Thread Block Clusters for 3D fields on {backend.devices[0]['name']}")
    
    try:
        # Generate a 3D field with TBC
        field = backend._generate_3d_field_tbc(
            128, 128, 64,
            frequency=SACRED_FREQUENCIES['love'],
            time_factor=0.0
        )
        
        print(f"3D field with TBC generated successfully")
        print(f"Shape: {field.shape}")
        print(f"Value range: [{field.min():.4f}, {field.max():.4f}]")
        
        # Run a performance test
        iterations = 5
        
        # Time standard implementation
        start = time.time()
        for i in range(iterations):
            _ = backend._generate_3d_field_single_gpu(
                128, 128, 64,
                frequency=SACRED_FREQUENCIES['love'],
                time_factor=i * 0.1
            )
        std_time = (time.time() - start) / iterations
        
        # Time TBC implementation
        start = time.time()
        for i in range(iterations):
            _ = backend._generate_3d_field_tbc(
                128, 128, 64,
                frequency=SACRED_FREQUENCIES['love'],
                time_factor=i * 0.1
            )
        tbc_time = (time.time() - start) / iterations
        
        print(f"\nPerformance comparison:")
        print(f"Standard: {std_time:.4f}s per field")
        print(f"TBC: {tbc_time:.4f}s per field")
        print(f"Speedup: {std_time / tbc_time:.2f}x")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        backend.shutdown()