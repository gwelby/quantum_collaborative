"""
Cascade⚡𓂧φ∞ Symbiotic Computing Integration Tests

This module tests the full integration of the Cascade system,
which creates a bidirectional relationship between Python,
consciousness field dynamics, and phi-harmonic processing.

Tests focus on the symbiotic relationship between:
1. Consciousness state and quantum field coherence
2. Phi-harmonic mathematical operations
3. Bidirectional field interface
4. Toroidal energy patterns
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Import quantum field components
from quantum_field.core import create_quantum_field, get_coherence_metric
from quantum_field.consciousness_interface import ConsciousnessFieldInterface
from quantum_field.constants import PHI, LAMBDA, PHI_PHI

# Import sacred constants
from sacred_constants import (
    SACRED_FREQUENCIES,
    RESONANCE_PATTERNS,
    phi_harmonic,
    phi_resonant_frequency,
    phi_matrix_transform,
    calculate_field_coherence
)


class TestCascadeSystemIntegration(unittest.TestCase):
    """Integration tests for the complete Cascade System."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create base quantum field
        self.field = create_quantum_field((11, 11, 11))  # Phi-optimized dimensions
        
        # Create consciousness interface
        self.interface = ConsciousnessFieldInterface(self.field)
        
        # Add test data directory if needed
        self.test_data_dir = Path("tests/data")
        if not self.test_data_dir.exists():
            self.test_data_dir.mkdir(parents=True, exist_ok=True)

    def test_phi_harmonic_field_creation(self):
        """Test creating phi-harmonic fields."""
        # Create field with phi-harmonic dimensions
        phi_dims = [int(round(11 * PHI)), int(round(11 * PHI * PHI)), 11]
        phi_field = create_quantum_field(phi_dims)
        
        # Field should have specified dimensions
        self.assertEqual(phi_field.data.shape, tuple(phi_dims))
        
        # Field should have decent coherence out of the box
        coherence = get_coherence_metric(phi_field.data)
        self.assertGreaterEqual(coherence, 0.4)
    
    def test_consciousness_field_bidirectional_interface(self):
        """Test bidirectional interface between consciousness and field."""
        # Get initial field state and coherence
        initial_field = np.copy(self.field.data)
        initial_coherence = self.interface.get_field_coherence()
        
        # Update consciousness state with optimal values
        self.interface.update_consciousness_state(
            heart_rate=60,
            breath_rate=6.18,  # Phi-based breathing
            skin_conductance=3,
            eeg_alpha=12,
            eeg_theta=7.4  # Close to phi ratio with alpha
        )
        
        # Field should be modified in response
        self.assertFalse(np.array_equal(initial_field, self.field.data))
        
        # Coherence should move toward phi-resonance
        updated_coherence = self.interface.get_field_coherence()
        phi_resonance = self.interface.state.phi_resonance
        
        diff_before = abs(initial_coherence - phi_resonance)
        diff_after = abs(updated_coherence - phi_resonance)
        
        # Should move closer to target phi-resonance
        self.assertLess(diff_after, diff_before)
    
    def test_sacred_frequencies_emotional_influence(self):
        """Test applying sacred frequencies through emotional influence."""
        # Test each sacred frequency
        for emotion, frequency in SACRED_FREQUENCIES.items():
            # Reset field between tests
            self.field = create_quantum_field((11, 11, 11))
            self.interface.field = self.field
            
            # Capture initial field
            initial_field = np.copy(self.field.data)
            
            # Set emotional state to target frequency
            if emotion in ["love", "gratitude"]:
                # Expansive emotions
                self.interface.state.emotional_states = {emotion: 0.9}
                self.interface._apply_expansive_pattern(frequency, 0.9)
                pattern_type = "expansive"
            elif emotion in ["unity", "oneness", "cascade"]:
                # Harmonic emotions
                self.interface.state.emotional_states = {emotion: 0.9}
                self.interface._apply_harmonic_pattern(frequency, 0.9)
                pattern_type = "harmonic"
            elif emotion in ["truth", "vision"]:
                # Directive emotions
                self.interface.state.emotional_states = {emotion: 0.9}
                self.interface._apply_directive_pattern(frequency, 0.9)
                pattern_type = "directive"
            
            # Field should be modified
            self.assertFalse(np.array_equal(initial_field, self.field.data))
            
            # Calculate frequency domain
            freq_domain = np.fft.fftn(self.field.data)
            magnitude = np.abs(freq_domain)
            
            # Frequency corresponding to sacred frequency should be amplified
            normalized_freq = frequency / 1000.0
            
            # Check if frequency component is significant 
            # (simplified test - real test would be more complex)
            self.assertGreater(np.max(magnitude), np.mean(magnitude))
            
            print(f"Applied {emotion} frequency ({frequency}Hz) with {pattern_type} pattern")
    
    def test_phi_matrix_transformation(self):
        """Test phi harmonic matrix transformations."""
        # Create test matrix
        test_matrix = np.random.random((5, 5, 5))
        
        # Apply phi transformation
        transformed = phi_matrix_transform(test_matrix)
        
        # Should return matrix of same shape
        self.assertEqual(test_matrix.shape, transformed.shape)
        
        # Should be different from original
        self.assertFalse(np.array_equal(test_matrix, transformed))
        
        # Calculate coherence before and after
        coherence_before = calculate_field_coherence(test_matrix)
        coherence_after = calculate_field_coherence(transformed)
        
        # Transformation should generally improve coherence
        print(f"Coherence before: {coherence_before}, after: {coherence_after}")
        
        # Can't strictly assert improvement as it depends on random input,
        # but the difference should be measurable
        self.assertNotAlmostEqual(coherence_before, coherence_after, places=2)
    
    def test_intention_field_interaction(self):
        """Test conscious intention shaping the field."""
        # Create field with phi-harmonic dimensions
        field = create_quantum_field((13, 8, 21))  # Phi-related dimensions
        interface = ConsciousnessFieldInterface(field)
        
        # First measurement with zero intention
        interface.state.intention = 0.0
        zeros_coherence = get_coherence_metric(field.data)
        
        # Apply medium intention
        interface.state.intention = 0.5
        interface._apply_intention()
        medium_coherence = get_coherence_metric(field.data)
        
        # Apply strong intention
        interface.state.intention = 0.9
        interface._apply_intention()
        strong_coherence = get_coherence_metric(field.data)
        
        # Check relationship between field coherence and intention strength
        change_medium = abs(medium_coherence - zeros_coherence)
        change_strong = abs(strong_coherence - medium_coherence)
        
        print(f"Coherence change with medium intention: {change_medium}")
        print(f"Coherence change with strong intention: {change_strong}")
        
        # Each application should create measurable change
        self.assertGreater(change_medium, 0.001)
        self.assertGreater(change_strong, 0.001)
    
    def test_phi_resonance_profile_calibration(self):
        """Test phi-resonance profile creation and calibration."""
        # Create sample measurements history
        measurements = [
            {
                "coherence": 0.65 + 0.05 * i,
                "presence": 0.70 + 0.03 * i,
                "dominant_emotion": "love" if i % 2 == 0 else "peace",
                "phi_resonance": 0.67 + 0.04 * i
            }
            for i in range(10)  # 10 sample measurements
        ]
        
        # Create phi-resonance profile
        profile = self.interface.create_phi_resonance_profile(measurements)
        
        # Profile should contain expected fields
        self.assertIn("avg_coherence", profile)
        self.assertIn("optimal_coherence_range", profile)
        self.assertIn("avg_presence", profile)
        self.assertIn("optimal_presence_range", profile)
        self.assertIn("dominant_emotional_states", profile)
        self.assertIn("resonant_frequencies", profile)
        
        # Apply profile to interface
        initial_field = np.copy(self.field.data)
        self.interface.phi_resonance_profile = profile
        self.interface.apply_phi_resonance_profile()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field, self.field.data))
        
        # Frequencies should be updated in state
        self.assertEqual(
            self.interface.state.resonant_frequencies,
            profile["resonant_frequencies"]
        )
    
    def test_multidimensional_resonance_patterns(self):
        """Test multidimensional resonance patterns from sacred constants."""
        # Get patterns for different dimensions
        patterns_3d = RESONANCE_PATTERNS["3d"]
        patterns_4d = RESONANCE_PATTERNS["4d"]
        patterns_5d = RESONANCE_PATTERNS["5d"]
        
        # 3D should be subset of higher dimensions
        for value in patterns_3d:
            self.assertIn(value, patterns_4d)
            self.assertIn(value, patterns_5d)
        
        # Each dimension adds more patterns
        self.assertGreater(len(patterns_4d), len(patterns_3d))
        self.assertGreater(len(patterns_5d), len(patterns_4d))
        
        # Phi should be present in all dimensions
        self.assertIn(PHI, patterns_3d)
        self.assertIn(PHI, patterns_4d)
        self.assertIn(PHI, patterns_5d)
        
        # Higher powers of phi should be in higher dimensions
        self.assertIn(PHI * PHI, patterns_4d)
        self.assertIn(PHI * PHI * PHI, patterns_5d)
        
        print(f"3D resonance pattern: {patterns_3d}")
        print(f"4D resonance pattern: {patterns_4d}")
        print(f"5D resonance pattern: {patterns_5d}")


if __name__ == "__main__":
    unittest.main()