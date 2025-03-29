"""
Test suite for the Quantum Consciousness Interface component.

This test suite validates the bidirectional interaction between
consciousness states and quantum fields, focusing on:
1. State coherence and presence calibration
2. Emotional field patterns and resonance
3. Intention-based field manipulation
4. Phi-resonance profiles and optimization
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from quantum_field.constants import PHI, LAMBDA, PHI_PHI
from quantum_field.core import QuantumField, get_coherence_metric, create_quantum_field
from quantum_field.consciousness_interface import (
    ConsciousnessState, 
    ConsciousnessFieldInterface,
    demo_consciousness_field_interface
)

from sacred_constants import (
    SACRED_FREQUENCIES,
    phi_harmonic,
    calculate_field_coherence,
    phi_resonance_spectrum
)


class TestConsciousnessState(unittest.TestCase):
    """Tests for the ConsciousnessState class."""
    
    def test_initialization(self):
        """Test default initialization of consciousness state."""
        state = ConsciousnessState()
        
        # Check default values
        self.assertAlmostEqual(state.coherence, 0.618)
        self.assertEqual(state.presence, 0.5)
        self.assertEqual(state.intention, 0.5)
        self.assertFalse(state.is_connected)
        
        # Check emotional states initialization
        self.assertIsNotNone(state.emotional_states)
        self.assertEqual(state.emotional_states["joy"], 0.5)
        self.assertEqual(state.emotional_states["peace"], 0.5)
        
        # Check resonant frequencies initialization
        self.assertIsNotNone(state.resonant_frequencies)
        self.assertEqual(state.resonant_frequencies["love"], SACRED_FREQUENCIES["love"])
    
    def test_custom_initialization(self):
        """Test custom initialization of consciousness state."""
        custom_emotions = {"joy": 0.8, "peace": 0.7}
        custom_frequencies = {"love": 432}
        
        state = ConsciousnessState(
            coherence=0.8,
            presence=0.9,
            intention=0.7,
            emotional_states=custom_emotions,
            resonant_frequencies=custom_frequencies,
            is_connected=True
        )
        
        # Check custom values
        self.assertEqual(state.coherence, 0.8)
        self.assertEqual(state.presence, 0.9)
        self.assertEqual(state.intention, 0.7)
        self.assertTrue(state.is_connected)
        
        # Check emotional states persistence
        self.assertEqual(state.emotional_states["joy"], 0.8)
        self.assertEqual(state.emotional_states["peace"], 0.7)
        
        # Check resonant frequencies persistence
        self.assertEqual(state.resonant_frequencies["love"], 432)
    
    def test_dominant_emotion(self):
        """Test dominant emotion property."""
        # Empty emotional states
        state = ConsciousnessState()
        state.emotional_states = {}
        self.assertEqual(state.dominant_emotion, ("neutral", 0.5))
        
        # Multiple emotions with clear dominant one
        state.emotional_states = {
            "joy": 0.5,
            "peace": 0.6,
            "love": 0.9,
            "gratitude": 0.3
        }
        emotion, value = state.dominant_emotion
        self.assertEqual(emotion, "love")
        self.assertEqual(value, 0.9)
    
    def test_phi_resonance(self):
        """Test phi resonance calculation."""
        state = ConsciousnessState(coherence=0.8, presence=0.6)
        
        # Expected: (0.8 * PHI + 0.6) / (PHI + 1)
        expected = (0.8 * PHI + 0.6) / (PHI + 1)
        self.assertAlmostEqual(state.phi_resonance, expected)
    
    def test_update_from_biofeedback(self):
        """Test updating state from biofeedback data."""
        state = ConsciousnessState()
        
        # Initial values
        initial_coherence = state.coherence
        initial_presence = state.presence
        
        # Update with biofeedback
        state.update_from_biofeedback(
            heart_rate=60,  # Optimal value
            breath_rate=6.18,  # Phi-based optimal
            skin_conductance=5,
            eeg_alpha=10,
            eeg_theta=6.18  # Close to phi ratio with alpha
        )
        
        # Coherence and presence should improve
        self.assertGreater(state.coherence, initial_coherence)
        self.assertGreater(state.presence, initial_presence)
        
        # Test with non-optimal values
        state = ConsciousnessState()
        state.update_from_biofeedback(
            heart_rate=90,  # Far from optimal
            breath_rate=15  # Far from optimal
        )
        
        # Coherence and presence should decrease
        self.assertLess(state.coherence, initial_coherence)


class TestConsciousnessFieldInterface(unittest.TestCase):
    """Tests for the ConsciousnessFieldInterface class."""
    
    def setUp(self):
        """Set up test fixture."""
        # Create a small test field
        self.test_field = create_quantum_field((5, 5, 5))
        
        # Create interface with field
        self.interface = ConsciousnessFieldInterface(self.test_field)
    
    def test_initialization(self):
        """Test interface initialization."""
        # Test with field
        self.assertEqual(self.interface.field, self.test_field)
        self.assertTrue(self.interface.connected)
        self.assertTrue(self.interface.state.is_connected)
        
        # Test without field
        interface_no_field = ConsciousnessFieldInterface()
        self.assertIsNone(interface_no_field.field)
        self.assertFalse(interface_no_field.connected)
        self.assertFalse(interface_no_field.state.is_connected)
    
    def test_connect_disconnect_field(self):
        """Test connecting and disconnecting from field."""
        # Start with unconnected interface
        interface = ConsciousnessFieldInterface()
        self.assertFalse(interface.connected)
        
        # Connect to field
        interface.connect_field(self.test_field)
        self.assertEqual(interface.field, self.test_field)
        self.assertTrue(interface.connected)
        self.assertTrue(interface.state.is_connected)
        
        # Disconnect
        interface.disconnect_field()
        self.assertIsNone(interface.field)
        self.assertFalse(interface.connected)
        self.assertFalse(interface.state.is_connected)
    
    def test_get_field_coherence(self):
        """Test retrieving field coherence."""
        # Should return non-zero value for connected field
        coherence = self.interface.get_field_coherence()
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
        
        # Should return 0 for disconnected field
        self.interface.disconnect_field()
        coherence = self.interface.get_field_coherence()
        self.assertEqual(coherence, 0.0)
    
    def test_update_consciousness_state(self):
        """Test updating consciousness state with biofeedback."""
        # Initial state
        initial_coherence = self.interface.state.coherence
        
        # Update with good biofeedback
        self.interface.update_consciousness_state(
            heart_rate=60,
            breath_rate=6.18
        )
        
        # Check state was updated
        self.assertNotEqual(self.interface.state.coherence, initial_coherence)
        
        # Check feedback history was recorded
        self.assertEqual(len(self.interface.feedback_history), 1)
        self.assertIn("coherence_change", self.interface.feedback_history[0])
        self.assertIn("biofeedback", self.interface.feedback_history[0])
    
    def test_apply_state_to_field(self):
        """Test applying consciousness state to field."""
        # Capture initial field data
        initial_field_data = np.copy(self.test_field.data)
        
        # Set high intention to ensure field modification
        self.interface.state.intention = 0.9
        self.interface.state.coherence = 0.9
        
        # Force apply state
        self.interface._apply_state_to_field()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))
    
    def test_emotional_patterns(self):
        """Test applying different emotional patterns to the field."""
        # Test expansive pattern
        self.interface.state.emotional_states = {"joy": 0.9}  # Set dominant emotion
        initial_field_data = np.copy(self.test_field.data)
        self.interface._apply_emotional_influence()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))
        
        # Test harmonic pattern
        self.interface.state.emotional_states = {"peace": 0.9}  # Set dominant emotion
        initial_field_data = np.copy(self.test_field.data)
        self.interface._apply_emotional_influence()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))
        
        # Test directive pattern
        self.interface.state.emotional_states = {"focus": 0.9}  # Set dominant emotion
        initial_field_data = np.copy(self.test_field.data)
        self.interface._apply_emotional_influence()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))
    
    def test_apply_intention(self):
        """Test applying intention to the field."""
        # Capture initial field
        initial_field_data = np.copy(self.test_field.data)
        
        # Set high intention
        self.interface.state.intention = 0.9
        self.interface._apply_intention()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))
    
    def test_phi_resonance_profile(self):
        """Test creating and applying phi-resonance profile."""
        # Add some sample measurements to history
        self.interface.feedback_history = [
            {
                "coherence": 0.7,
                "presence": 0.8,
                "dominant_emotion": "love",
                "phi_resonance": 0.75
            },
            {
                "coherence": 0.75,
                "presence": 0.85,
                "dominant_emotion": "peace",
                "phi_resonance": 0.8
            }
        ]
        
        # Create profile
        profile = self.interface.create_phi_resonance_profile(self.interface.feedback_history)
        
        # Check profile content
        self.assertIn("avg_coherence", profile)
        self.assertIn("optimal_coherence_range", profile)
        self.assertIn("resonant_frequencies", profile)
        
        # Apply profile
        initial_field_data = np.copy(self.test_field.data)
        self.interface.apply_phi_resonance_profile()
        
        # Field should be modified
        self.assertFalse(np.array_equal(initial_field_data, self.test_field.data))


class TestDemoFunction(unittest.TestCase):
    """Tests for the demo_consciousness_field_interface function."""
    
    @patch('quantum_field.consciousness_interface.create_quantum_field')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_demo_function(self, mock_sleep, mock_create_field):
        """Test the demonstration function runs without errors."""
        # Mock field creation
        mock_field = MagicMock()
        mock_field.data = np.random.random((21, 21, 21))
        mock_create_field.return_value = mock_field
        
        # Run demo
        interface = demo_consciousness_field_interface()
        
        # Check mock was called
        mock_create_field.assert_called_once()
        mock_sleep.assert_called()
        
        # Check interface was initialized
        self.assertIsNotNone(interface)
        self.assertEqual(interface.field, mock_field)
        self.assertTrue(len(interface.feedback_history) > 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for consciousness-field bidirectional interaction."""
    
    def test_biofeedback_to_field_coherence(self):
        """Test complete pathway from biofeedback to field coherence."""
        # Create field and interface
        field = create_quantum_field((10, 10, 10))
        interface = ConsciousnessFieldInterface(field)
        
        # Get initial field coherence
        initial_coherence = interface.get_field_coherence()
        
        # Apply optimal biofeedback
        interface.update_consciousness_state(
            heart_rate=60,
            breath_rate=6.18,
            skin_conductance=3,
            eeg_alpha=12,
            eeg_theta=7.4
        )
        
        # Check field coherence improved toward phi-resonance
        updated_coherence = interface.get_field_coherence()
        phi_resonance = interface.state.phi_resonance
        
        # Field coherence should move toward phi_resonance
        diff_before = abs(initial_coherence - phi_resonance)
        diff_after = abs(updated_coherence - phi_resonance)
        
        # Should move closer to target phi-resonance
        self.assertLess(diff_after, diff_before)
    
    def test_phi_resonance_with_sacred_constants(self):
        """Test phi-resonance integration with sacred constants."""
        # Create field and interface
        field = create_quantum_field((10, 10, 10))
        interface = ConsciousnessFieldInterface(field)
        
        # Get field data
        field_data = field.data
        
        # Calculate coherence using sacred_constants function
        sacred_coherence = calculate_field_coherence(field_data)
        
        # Calculate resonance spectrum
        spectrum = phi_resonance_spectrum(field_data)
        
        # Both measures should be between 0-1 for valid field
        self.assertGreaterEqual(sacred_coherence, 0.0)
        self.assertLessEqual(sacred_coherence, 1.0)
        self.assertGreaterEqual(spectrum["combined"], 0.0)
        self.assertLessEqual(spectrum["combined"], 1.0)
        
        # Apply emotional pattern
        interface.state.emotional_states = {"love": 0.9}
        interface._apply_emotional_influence()
        
        # Recalculate after emotional influence
        updated_sacred_coherence = calculate_field_coherence(field.data)
        updated_spectrum = phi_resonance_spectrum(field.data)
        
        # Field should change in measurable ways with both metrics
        self.assertNotEqual(sacred_coherence, updated_sacred_coherence)
        self.assertNotEqual(spectrum["combined"], updated_spectrum["combined"])


if __name__ == "__main__":
    unittest.main()