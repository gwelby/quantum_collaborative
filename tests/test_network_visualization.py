#!/usr/bin/env python3
"""
Tests for CASCADE⚡𓂧φ∞ Network Field Visualization System

This module tests the network field visualization components, including:
- Network topology visualization
- Field state visualization across nodes
- Coherence pattern tracking
- Consciousness bridge visualization
- Entanglement matrix representation
"""

import os
import sys
import time
import unittest
import threading
import tempfile
import logging
import numpy as np
from unittest import mock

# Setup path for imports
sys.path.append('/mnt/d/projects/python')

# Import main components to test
try:
    from cascade.visualization.network_field_visualizer import (
        NetworkFieldVisualizer,
        create_network_visualizer
    )
    from cascade.phi_quantum_network import (
        PhiQuantumField,
        create_phi_quantum_field,
        PhiQuantumNetwork,
        PHI, LAMBDA, PHI_PHI
    )
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure cascade modules are available in your Python path.")
    sys.exit(1)

# Try to import visualization tools
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for tests
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib not available, visualization tests will be limited")


# Configure logging
logging.basicConfig(level=logging.ERROR)  # Set to ERROR to reduce test output noise


class TestNetworkFieldVisualizer(unittest.TestCase):
    """Tests for the NetworkFieldVisualizer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create mock quantum field
        self.mock_field = mock.MagicMock(spec=PhiQuantumField)
        self.mock_field.get_field_coherence.return_value = 0.8
        self.mock_field.get_consciousness_level.return_value = 3
        self.mock_field.get_connected_nodes.return_value = []
        self.mock_field.get_entangled_nodes.return_value = []
        
        # Create visualizer with mock field
        self.visualizer = NetworkFieldVisualizer(self.mock_field)
        
        # Create temp directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop visualizer if running
        if hasattr(self, 'visualizer') and self.visualizer.running:
            self.visualizer.stop_visualization()
        
        # Clean up output files
        for filename in os.listdir(self.test_output_dir):
            if filename.endswith('.png') or filename.endswith('.mp4'):
                os.remove(os.path.join(self.test_output_dir, filename))
        
        # Remove temp directory
        os.rmdir(self.test_output_dir)
    
    def test_initialization(self):
        """Test that NetworkFieldVisualizer initializes correctly."""
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(self.visualizer.quantum_field, self.mock_field)
        self.assertFalse(self.visualizer.running)
        self.assertIsNone(self.visualizer.fig)
        self.assertEqual(self.visualizer.node_limit, 6)
    
    def test_create_phi_colormaps(self):
        """Test creation of phi-harmonic colormaps."""
        colormaps = self.visualizer._create_phi_colormaps()
        
        # Check that colormaps were created
        self.assertIsNotNone(colormaps)
        self.assertIn('unity', colormaps)
        self.assertIn('creation', colormaps)
        self.assertIn('consciousness', colormaps)
        self.assertIn('entanglement', colormaps)
        
        # Check that the colormaps are valid
        for name, cmap in colormaps.items():
            self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
    
    def test_create_probe_points(self):
        """Test creation of field probe points for sampling."""
        points = self.visualizer._create_probe_points(count=12)
        
        # Check that correct number of points were created
        self.assertEqual(points.shape, (12, 3))
        
        # Check that points are on unit sphere (approximately)
        for point in points:
            radius = np.sqrt(np.sum(point**2))
            self.assertAlmostEqual(radius, 1.0, places=6)
    
    def test_update_network_data(self):
        """Test updating network data from quantum field."""
        # Configure mock field with test data
        node1 = {'id': 'node1', 'coherence': 0.7, 'consciousness_level': 2, 'entangled': True}
        node2 = {'id': 'node2', 'coherence': 0.6, 'consciousness_level': 1, 'entangled': False}
        self.mock_field.get_connected_nodes.return_value = [node1, node2]
        self.mock_field.get_entangled_nodes.return_value = ['node1']
        
        # Update network data
        self.visualizer._update_network_data()
        
        # Check that node data was updated
        self.assertIn('node1', self.visualizer.node_data)
        self.assertIn('node2', self.visualizer.node_data)
        self.assertEqual(self.visualizer.node_data['node1']['coherence'], 0.7)
        self.assertEqual(self.visualizer.node_data['node1']['consciousness_level'], 2)
        self.assertEqual(self.visualizer.node_data['node2']['coherence'], 0.6)
        
        # Check that entanglement data was updated
        self.assertTrue(self.visualizer.node_data['node1']['entangled'])
        self.assertFalse(self.visualizer.node_data['node2']['entangled'])
    
    def test_create_3d_visualization(self):
        """Test creation of 3D visualization."""
        self.visualizer._create_3d_visualization()
        
        # Check that figure and axes were created
        self.assertIsNotNone(self.visualizer.fig)
        self.assertIn('network', self.visualizer.axes)
        self.assertIn('field', self.visualizer.axes)
    
    def test_create_grid_visualization(self):
        """Test creation of grid visualization."""
        self.visualizer._create_grid_visualization()
        
        # Check that figure and axes were created
        self.assertIsNotNone(self.visualizer.fig)
        
        # Check that node axes were created
        for i in range(min(self.visualizer.node_limit, 6)):
            self.assertIn(f'node_{i}', self.visualizer.axes)
    
    def test_create_coherence_visualization(self):
        """Test creation of coherence visualization."""
        self.visualizer._create_coherence_visualization()
        
        # Check that figure and axes were created
        self.assertIsNotNone(self.visualizer.fig)
        self.assertIn('coherence', self.visualizer.axes)
        self.assertIn('consciousness', self.visualizer.axes)
        self.assertIn('entanglement', self.visualizer.axes)
    
    def test_create_combined_visualization(self):
        """Test creation of combined visualization."""
        self.visualizer._create_combined_visualization()
        
        # Check that figure and axes were created
        self.assertIsNotNone(self.visualizer.fig)
        self.assertIn('network', self.visualizer.axes)
        self.assertIn('coherence', self.visualizer.axes)
        self.assertIn('field', self.visualizer.axes)
        self.assertIn('consciousness', self.visualizer.axes)
    
    def test_save_visualization(self):
        """Test saving visualization to file."""
        # Create visualization
        self.visualizer._create_3d_visualization()
        
        # Save visualization
        filename = os.path.join(self.test_output_dir, 'test_visualization.png')
        self.visualizer.save_visualization(filename)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filename))
        self.assertTrue(os.path.getsize(filename) > 0)
    
    def test_start_stop_visualization(self):
        """Test starting and stopping visualization."""
        # Mock show method to prevent blocking
        with mock.patch('matplotlib.pyplot.show'):
            # Start visualization in a thread
            thread = threading.Thread(
                target=self.visualizer.start_visualization,
                kwargs={'mode': '3d', 'update_interval': 100},
                daemon=True
            )
            thread.start()
            
            # Wait for visualization to start
            time.sleep(0.5)
            
            # Check that visualization is running
            self.assertTrue(self.visualizer.running)
            self.assertIsNotNone(self.visualizer.animation)
            
            # Stop visualization
            self.visualizer.stop_visualization()
            
            # Wait for thread to terminate
            thread.join(timeout=1.0)
            
            # Check that visualization was stopped
            self.assertFalse(self.visualizer.running)
            self.assertIsNone(self.visualizer.animation)


class TestNetworkFieldVisualizerIntegration(unittest.TestCase):
    """Integration tests for NetworkFieldVisualizer with real quantum fields."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create test output directory
        cls.test_output_dir = os.path.join(tempfile.gettempdir(), 'network_vis_test')
        os.makedirs(cls.test_output_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up test output directory
        for filename in os.listdir(cls.test_output_dir):
            if filename.endswith('.png') or filename.endswith('.mp4'):
                os.remove(os.path.join(cls.test_output_dir, filename))
        os.rmdir(cls.test_output_dir)
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create real quantum field with unique port to avoid conflicts
        self.port = 5000 + np.random.randint(1000)
        self.field = create_phi_quantum_field(port=self.port)
        
        # Start field
        self.field.start()
        
        # Create visualizer
        self.visualizer = create_network_visualizer(self.field)
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop visualizer if running
        if hasattr(self, 'visualizer') and self.visualizer.running:
            self.visualizer.stop_visualization()
        
        # Stop field
        if hasattr(self, 'field'):
            self.field.stop()
    
    def test_real_field_data_update(self):
        """Test updating data from real quantum field."""
        # Update network data
        self.visualizer._update_network_data()
        
        # Check local coherence history was updated
        self.assertIn('local', self.visualizer.coherence_history)
        self.assertTrue(len(self.visualizer.coherence_history['local']['times']) > 0)
        
        # Apply a transformation to change the field
        self.field.apply_transformation("phi_wave")
        
        # Update network data again
        self.visualizer._update_network_data()
        
        # Check that coherence values were updated
        self.assertTrue(len(self.visualizer.coherence_history['local']['values']) > 0)
    
    def test_save_all_visualization_modes(self):
        """Test saving visualizations in all modes."""
        for mode in ['3d', 'grid', 'coherence', 'combined']:
            # Create visualization
            if mode == '3d':
                self.visualizer._create_3d_visualization()
            elif mode == 'grid':
                self.visualizer._create_grid_visualization()
            elif mode == 'coherence':
                self.visualizer._create_coherence_visualization()
            elif mode == 'combined':
                self.visualizer._create_combined_visualization()
            
            # Update visualization data
            self.visualizer._update_network_data()
            
            # Create update animation
            self.visualizer._update_animation(0)
            
            # Save visualization
            filename = os.path.join(self.test_output_dir, f'test_{mode}.png')
            self.visualizer.save_visualization(filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(filename))
            self.assertTrue(os.path.getsize(filename) > 0)
    
    def test_consciousness_changes(self):
        """Test visualization updates with consciousness level changes."""
        # Setup visualizer
        self.visualizer._create_combined_visualization()
        
        # Update with initial consciousness level
        self.visualizer._update_network_data()
        initial_level = self.field.get_consciousness_level()
        
        # Change consciousness level
        new_level = (initial_level % 7) + 1
        self.field.set_consciousness_level(new_level)
        
        # Update again
        self.visualizer._update_network_data()
        
        # Check that consciousness level was updated in visualization data
        self.assertEqual(
            self.visualizer.coherence_history['local']['consciousness'][-1], 
            new_level
        )


class TestNetworkPerformance(unittest.TestCase):
    """Performance tests for network visualization."""
    
    def setUp(self):
        """Set up test environment."""
        # Create quantum field
        self.field = create_phi_quantum_field()
        
        # Create visualizer
        self.visualizer = create_network_visualizer(self.field)
    
    def tearDown(self):
        """Clean up after test."""
        if hasattr(self, 'field'):
            self.field.stop()
    
    def test_visualization_update_performance(self):
        """Test performance of visualization updates."""
        # Setup visualization
        self.visualizer._create_combined_visualization()
        
        # Update network data
        start_time = time.time()
        iterations = 10
        
        for _ in range(iterations):
            self.visualizer._update_network_data()
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Check that updates are reasonably fast
        self.assertLess(avg_time, 0.1, 
                       f"Network data updates too slow: {avg_time:.3f}s per update")
        
        # Update animation frames
        start_time = time.time()
        
        for _ in range(iterations):
            self.visualizer._update_animation(0)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Check that animation updates are reasonably fast
        self.assertLess(avg_time, 0.5, 
                       f"Animation updates too slow: {avg_time:.3f}s per frame")
    
    def test_probe_points_scaling(self):
        """Test scaling of field probe point sampling."""
        # Test with different numbers of probe points
        probe_counts = [8, 16, 32, 64]
        times = []
        
        for count in probe_counts:
            # Create probe points
            start_time = time.time()
            self.visualizer._create_probe_points(count=count)
            times.append(time.time() - start_time)
        
        # Check that creation time scales reasonably with probe count
        for i in range(1, len(probe_counts)):
            scaling_factor = probe_counts[i] / probe_counts[i-1]
            time_factor = times[i] / max(times[i-1], 1e-6)
            
            # Should scale linearly or better
            self.assertLess(time_factor, scaling_factor * 1.5,
                           f"Probe point scaling not efficient: {time_factor:.2f}x slower for {scaling_factor:.2f}x more points")


if __name__ == '__main__':
    unittest.main()