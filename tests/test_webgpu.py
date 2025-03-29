"""
Tests for WebGPU backend implementation.
"""

import unittest
import numpy as np

# Try to import PyWebGPU (tests will be skipped if not available)
try:
    import pywebgpu
    HAS_WEBGPU = True
except ImportError:
    HAS_WEBGPU = False

# Import quantum field modules
from quantum_field.backends import get_backend
from quantum_field.constants import SACRED_FREQUENCIES, PHI


@unittest.skipIf(not HAS_WEBGPU, "PyWebGPU not installed")
class WebGPUBackendTests(unittest.TestCase):
    """Test WebGPU backend implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Try to get WebGPU backend
        try:
            self.backend = get_backend("webgpu")
            if not self.backend.is_available():
                self.skipTest("WebGPU backend not available")
        except ValueError:
            self.skipTest("WebGPU backend not found")
    
    def test_backend_capabilities(self):
        """Test that WebGPU backend reports correct capabilities."""
        capabilities = self.backend.get_capabilities()
        
        # Check minimum expected capabilities
        self.assertTrue(capabilities.get("web_compatible", False), 
                       "WebGPU backend should report web compatibility")
        
        # DLPack not supported yet (should be false)
        self.assertFalse(capabilities.get("dlpack_support", True),
                        "WebGPU backend should not report DLPack support yet")
    
    def test_field_generation_basic(self):
        """Test basic field generation with WebGPU backend."""
        # Generate a small field
        field = self.backend.generate_quantum_field(64, 64, "love", 0.0)
        
        # Check output shape and type
        self.assertEqual(field.shape, (64, 64))
        self.assertEqual(field.dtype, np.float32)
        
        # Check values are finite and in expected range
        self.assertTrue(np.isfinite(field).all())
        self.assertTrue(-1.0 <= field.min() <= field.max() <= 1.0)
    
    def test_field_generation_different_frequencies(self):
        """Test field generation with different frequencies."""
        fields = {}
        
        # Generate fields for each sacred frequency
        for freq_name in SACRED_FREQUENCIES:
            fields[freq_name] = self.backend.generate_quantum_field(32, 32, freq_name, 0.0)
        
        # Check all fields are generated and different
        for freq1_name, field1 in fields.items():
            for freq2_name, field2 in fields.items():
                if freq1_name != freq2_name:
                    # Fields should be different for different frequencies
                    self.assertFalse(
                        np.allclose(field1, field2),
                        f"Fields for {freq1_name} and {freq2_name} should be different"
                    )
    
    def test_field_generation_time_evolution(self):
        """Test field generation with time evolution."""
        # Generate fields at different time points
        field1 = self.backend.generate_quantum_field(32, 32, "unity", 0.0)
        field2 = self.backend.generate_quantum_field(32, 32, "unity", 1.0)
        field3 = self.backend.generate_quantum_field(32, 32, "unity", 2.0)
        
        # Check fields are different at different time points
        self.assertFalse(np.allclose(field1, field2))
        self.assertFalse(np.allclose(field2, field3))
        self.assertFalse(np.allclose(field1, field3))
    
    def test_field_generation_custom_frequency(self):
        """Test field generation with custom frequency."""
        # Generate field with custom frequency
        custom_field = self.backend.generate_quantum_field(
            32, 32, custom_frequency=PHI * 100, time_factor=0.0
        )
        
        # Should produce a valid field
        self.assertEqual(custom_field.shape, (32, 32))
        self.assertTrue(np.isfinite(custom_field).all())
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test with invalid frequency name
        with self.assertRaises(ValueError):
            self.backend.generate_quantum_field(32, 32, "invalid_frequency", 0.0)
        
        # Test with no frequency specified
        with self.assertRaises(ValueError):
            self.backend.generate_quantum_field(32, 32, None, 0.0, None)
    
    def test_coherence_calculation(self):
        """Test field coherence calculation."""
        # Generate a field
        field = self.backend.generate_quantum_field(64, 64, "love", 0.0)
        
        # Calculate coherence
        coherence = self.backend.calculate_field_coherence(field)
        
        # Coherence should be a float between 0 and 1
        self.assertIsInstance(coherence, float)
        self.assertTrue(0.0 <= coherence <= 1.0)
    
    def test_dlpack_not_supported(self):
        """Test that DLPack operations raise appropriate errors."""
        # Generate a field
        field = self.backend.generate_quantum_field(32, 32, "unity", 0.0)
        
        # to_dlpack should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.backend.to_dlpack(field)
        
        # from_dlpack should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.backend.from_dlpack(None)


if __name__ == "__main__":
    unittest.main()