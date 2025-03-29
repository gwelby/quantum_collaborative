"""
Tests for CUDA Graphs optimization in the CUDA backend
"""

import unittest
import numpy as np
import time

# Try to import CUDA modules (tests will be skipped if not available)
try:
    import cupy as cp
    from cuda.core.experimental import Device
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Import quantum field modules
from quantum_field.backends import get_backend
from quantum_field.constants import SACRED_FREQUENCIES


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
class CUDAGraphsTests(unittest.TestCase):
    """Test CUDA Graphs functionality in the CUDA backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.cuda_backend = get_backend("cuda")
        if not self.cuda_backend.capabilities.get("cuda_graphs", False):
            self.skipTest("CUDA Graphs not supported on this hardware")
    
    def tearDown(self):
        """Clean up after tests"""
        # Destroy any graphs that might have been created
        graphs = self.cuda_backend.list_cuda_graphs()
        for graph in graphs:
            self.cuda_backend.destroy_cuda_graph(graph["name"])
    
    def test_create_and_execute_graph(self):
        """Test basic graph creation and execution"""
        # Create a graph for a small field
        width, height = 256, 256
        result = self.cuda_backend.create_cuda_graph("test_graph", width, height, "love")
        self.assertTrue(result, "Graph creation failed")
        
        # Execute the graph with different time factors
        for t in [0.0, 0.1, 0.2, 0.3]:
            field = self.cuda_backend.execute_cuda_graph("test_graph", t)
            self.assertIsNotNone(field, "Graph execution failed")
            self.assertEqual(field.shape, (height, width), "Field shape mismatch")
            
            # Field should contain valid values
            self.assertTrue(np.isfinite(field).all(), "Field contains invalid values")
    
    def test_graph_performance(self):
        """Test performance benefits of CUDA Graphs"""
        width, height = 512, 512
        
        # First, generate field normally 3 times to warm up
        for _ in range(3):
            self.cuda_backend.generate_quantum_field(width, height, "love", 0.0)
        
        # Measure time for normal field generation
        start_time = time.time()
        iterations = 10
        for t in np.linspace(0, 1, iterations):
            field_normal = self.cuda_backend.generate_quantum_field(width, height, "love", t)
        normal_time = time.time() - start_time
        print(f"Normal generation time for {iterations} iterations: {normal_time:.4f}s")
        
        # Create and execute graph
        self.cuda_backend.create_cuda_graph("perf_test", width, height, "love")
        
        # Warm up the graph execution
        for _ in range(3):
            self.cuda_backend.execute_cuda_graph("perf_test", 0.0)
        
        # Measure time for graph-based generation
        start_time = time.time()
        for t in np.linspace(0, 1, iterations):
            field_graph = self.cuda_backend.execute_cuda_graph("perf_test", t)
        graph_time = time.time() - start_time
        print(f"Graph-based generation time for {iterations} iterations: {graph_time:.4f}s")
        
        # Graph execution should be faster (we expect at least 20% improvement)
        self.assertLess(graph_time, normal_time * 0.8, 
                      f"Graph execution not significantly faster: {graph_time:.4f}s vs {normal_time:.4f}s")
    
    def test_graph_types(self):
        """Test different graph types (basic, TBC, multi-GPU)"""
        small_field = (256, 256)  # Should use basic graph
        large_field = (1024, 1024)  # Should use TBC or multi-GPU if available
        
        # Create a small field graph
        self.cuda_backend.create_cuda_graph("small_graph", *small_field, "love")
        
        # Create a large field graph
        self.cuda_backend.create_cuda_graph("large_graph", *large_field, "love")
        
        # List graphs and verify the types
        graphs = {g["name"]: g for g in self.cuda_backend.list_cuda_graphs()}
        
        self.assertIn("small_graph", graphs)
        self.assertIn("large_graph", graphs)
        
        # Small field should be basic type
        self.assertEqual(graphs["small_graph"]["type"], "basic")
        
        # Large field should be TBC or multi-GPU if available
        large_type = graphs["large_graph"]["type"]
        
        # Print the type for debugging
        print(f"Large field graph type: {large_type}")
        
        # Verify output from all graph types
        small_field_data = self.cuda_backend.execute_cuda_graph("small_graph", 0.1)
        self.assertEqual(small_field_data.shape, small_field[::-1])
        
        large_field_data = self.cuda_backend.execute_cuda_graph("large_graph", 0.1)
        self.assertEqual(large_field_data.shape, large_field[::-1])


if __name__ == "__main__":
    unittest.main()