"""
Tests for 3D visualization module.
"""

import unittest
import numpy as np
import os

# Try to import optional visualization packages
try:
    import plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pyvista
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Import quantum field modules
from quantum_field.visualization3d import (
    generate_3d_quantum_field,
    calculate_3d_field_coherence,
    visualize_3d_slices
)

class Test3DQuantumField(unittest.TestCase):
    """Test 3D quantum field generation and analysis."""
    
    def test_3d_field_generation_basic(self):
        """Test basic 3D field generation"""
        # Generate a small field for testing
        field = generate_3d_quantum_field(16, 16, 16, "love", 0.0)
        
        # Check dimensions
        self.assertEqual(field.shape, (16, 16, 16))
        
        # Check values are finite and in expected range
        self.assertTrue(np.isfinite(field).all())
        self.assertTrue(-1.0 <= field.min() <= field.max() <= 1.0)
    
    def test_3d_field_with_custom_frequency(self):
        """Test 3D field generation with custom frequency"""
        # Generate field with custom frequency
        field = generate_3d_quantum_field(16, 16, 16, custom_frequency=528, time_factor=0.0)
        
        # Check dimensions and values
        self.assertEqual(field.shape, (16, 16, 16))
        self.assertTrue(np.isfinite(field).all())
    
    def test_3d_field_coherence(self):
        """Test 3D field coherence calculation"""
        # Generate a field
        field = generate_3d_quantum_field(32, 32, 32, "unity", 0.0)
        
        # Calculate coherence
        coherence = calculate_3d_field_coherence(field)
        
        # Check result is a float between 0 and 1
        self.assertIsInstance(coherence, float)
        self.assertTrue(0.0 <= coherence <= 1.0)
    
    def test_time_evolution(self):
        """Test 3D field evolution over time"""
        # Generate fields at different time points
        field1 = generate_3d_quantum_field(16, 16, 16, "love", 0.0)
        field2 = generate_3d_quantum_field(16, 16, 16, "love", np.pi)
        
        # Fields should be different at different time points
        self.assertFalse(np.allclose(field1, field2))
    
    def test_different_frequencies(self):
        """Test 3D fields with different frequencies"""
        # Generate fields with different frequencies
        field1 = generate_3d_quantum_field(16, 16, 16, "love", 0.0)
        field2 = generate_3d_quantum_field(16, 16, 16, "unity", 0.0)
        
        # Fields should be different for different frequencies
        self.assertFalse(np.allclose(field1, field2))
    
    def test_slice_visualization(self):
        """Test 3D field slice visualization"""
        # Generate a field
        field = generate_3d_quantum_field(32, 32, 32, "love", 0.0)
        
        # Create slice visualization
        fig = visualize_3d_slices(field, title="Test Slices")
        
        # Check figure was created
        self.assertIsNotNone(fig)


@unittest.skipIf(not HAS_PLOTLY, "Plotly not installed")
class Test3DVisualizationPlotly(unittest.TestCase):
    """Test 3D visualization with Plotly (requires plotly package)."""
    
    def setUp(self):
        """Set up a field for testing"""
        # Create a small field for testing
        self.field = generate_3d_quantum_field(32, 32, 32, "love", 0.0)
    
    def test_volume_rendering(self):
        """Test 3D volume rendering"""
        from quantum_field.visualization3d import visualize_3d_volume
        
        # Create volume visualization
        fig = visualize_3d_volume(self.field, colormap="viridis")
        
        # Check figure was created successfully
        self.assertIsNotNone(fig)
    
    def test_isosurface_visualization(self):
        """Test 3D isosurface visualization"""
        from quantum_field.visualization3d import visualize_3d_isosurface
        
        # Create isosurface visualization
        fig = visualize_3d_isosurface(self.field, iso_values=[0.0])
        
        # Check figure was created successfully
        self.assertIsNotNone(fig)


@unittest.skipIf(not HAS_PYVISTA, "PyVista not installed")
class Test3DIsosurfaceExtraction(unittest.TestCase):
    """Test 3D isosurface extraction (requires pyvista package)."""
    
    def setUp(self):
        """Set up a field for testing"""
        # Create a small field for testing
        self.field = generate_3d_quantum_field(32, 32, 32, "love", 0.0)
    
    def test_isosurface_extraction(self):
        """Test extracting isosurfaces from 3D fields"""
        from quantum_field.visualization3d import extract_isosurface
        
        # Extract isosurface
        vertices, triangles = extract_isosurface(self.field, iso_value=0.0)
        
        # Check vertices and triangles were created
        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(triangles, np.ndarray)
        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(triangles.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)  # (x, y, z) coordinates
        self.assertEqual(triangles.shape[1], 3)  # 3 vertices per triangle


if __name__ == "__main__":
    unittest.main()