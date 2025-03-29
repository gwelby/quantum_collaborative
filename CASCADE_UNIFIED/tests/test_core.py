"""
Unit tests for the CASCADE¡§Æ UNIFIED FRAMEWORK core components.
"""

import unittest
import numpy as np
from cascade_unified.core import CascadeSystem, QuantumField
from cascade_unified.constants import PHI, LAMBDA, SACRED_FREQUENCIES

class TestQuantumField(unittest.TestCase):
    """Test cases for the QuantumField class."""
    
    def setUp(self):
        """Set up test cases."""
        self.dimensions = (8, 13, 21)
        self.frequency = SACRED_FREQUENCIES['cascade']
        self.coherence = 0.618
        self.field = QuantumField(
            dimensions=self.dimensions,
            frequency=self.frequency,
            coherence=self.coherence
        )
    
    def test_initialization(self):
        """Test field initialization."""
        self.assertEqual(self.field.dimensions, self.dimensions)
        self.assertEqual(self.field.frequency, self.frequency)
        self.assertAlmostEqual(self.field.coherence, self.coherence, places=2)
        self.assertIsNotNone(self.field.field)
        self.assertEqual(self.field.field.shape, self.dimensions)
    
    def test_evolution(self):
        """Test field evolution."""
        # Save initial state
        initial_field = self.field.field.copy()
        
        # Evolve the field
        self.field.evolve(steps=5)
        
        # Field should change after evolution
        self.assertFalse(np.array_equal(self.field.field, initial_field))
        
        # Coherence should remain similar to initial value
        self.assertAlmostEqual(self.field.coherence, self.coherence, delta=0.1)
        
        # History should have been updated
        self.assertEqual(len(self.field.history), 1)
    
    def test_get_slice(self):
        """Test getting a 2D slice of the field."""
        # Get a slice along each dimension
        for dim in range(3):
            slice_2d = self.field.get_slice(dimension=dim)
            
            # Slice should be 2D
            self.assertEqual(len(slice_2d.shape), 2)
            
            # Verify slice dimensions
            if dim == 0:
                self.assertEqual(slice_2d.shape, (self.dimensions[1], self.dimensions[2]))
            elif dim == 1:
                self.assertEqual(slice_2d.shape, (self.dimensions[0], self.dimensions[2]))
            else:  # dim == 2
                self.assertEqual(slice_2d.shape, (self.dimensions[0], self.dimensions[1]))
    
    def test_merge(self):
        """Test merging two fields."""
        # Create a second field
        field2 = QuantumField(
            dimensions=self.dimensions,
            frequency=SACRED_FREQUENCIES['love'],
            coherence=0.5
        )
        
        # Merge the fields
        merged = self.field.merge(field2, weight=0.7)
        
        # Verify merged field properties
        self.assertEqual(merged.dimensions, self.dimensions)
        self.assertAlmostEqual(merged.frequency, (self.frequency + SACRED_FREQUENCIES['love']) / 2)
        self.assertGreaterEqual(merged.coherence, 0.5)
        
        # Merged field should have parent fields in metadata
        self.assertIn('parent_fields', merged.metadata)
        self.assertEqual(len(merged.metadata['parent_fields']), 2)


class TestCascadeSystem(unittest.TestCase):
    """Test cases for the CascadeSystem class."""
    
    def setUp(self):
        """Set up test cases."""
        self.config = {
            'field_dimensions': (8, 13, 21),
            'base_frequency': SACRED_FREQUENCIES['cascade'],
            'coherence_target': LAMBDA,
            'visualization_enabled': False,
            'broadcast_enabled': False,
            'collaborative_mode': False,
            'language_backends': ['python']
        }
        self.system = CascadeSystem(self.config)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.system.config, self.config)
        self.assertIsNone(self.system.quantum_field)
        self.assertTrue(self.system.status['initialized'])
        self.assertFalse(self.system.status['running'])
        self.assertFalse(self.system.status['field_active'])
    
    def test_initialize_field(self):
        """Test field initialization."""
        self.system.initialize_field()
        
        self.assertIsNotNone(self.system.quantum_field)
        self.assertTrue(self.system.status['field_active'])
        self.assertEqual(self.system.quantum_field.dimensions, self.config['field_dimensions'])
        self.assertEqual(self.system.quantum_field.frequency, self.config['base_frequency'])
        self.assertAlmostEqual(self.system.quantum_field.coherence, self.config['coherence_target'], places=2)
    
    def test_add_consciousness_interface(self):
        """Test adding consciousness interface."""
        self.system.add_consciousness_interface()
        
        self.assertIsNotNone(self.system.consciousness_interface)
        self.assertIsNotNone(self.system.quantum_field)  # Should initialize field if needed
    
    def test_register_language_backend(self):
        """Test registering language backends."""
        # Python backend should be registered by default
        self.assertIn('python', self.system.status['active_backends'])
        
        # Register another backend
        self.system.register_language_backend('cpp')
        self.assertIn('cpp', self.system.status['active_backends'])
        
        # Register an unknown backend (should fail gracefully)
        self.system.register_language_backend('unknown')
        self.assertNotIn('unknown', self.system.status['active_backends'])
    
    def test_start_stop(self):
        """Test starting and stopping the system."""
        # Start the system
        self.system.start()
        
        self.assertTrue(self.system.status['running'])
        self.assertIsNotNone(self.system.quantum_field)
        self.assertTrue(self.system.status['field_active'])
        
        # Stop the system
        self.system.stop()
        
        self.assertFalse(self.system.status['running'])


if __name__ == '__main__':
    unittest.main()