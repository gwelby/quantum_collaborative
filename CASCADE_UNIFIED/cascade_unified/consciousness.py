"""
Consciousness interface for the CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK.

This module provides the interface between quantum fields and consciousness states,
enabling bidirectional interaction between field dynamics and awareness.
"""

import numpy as np
from .constants import PHI, LAMBDA, CONSCIOUSNESS_STATES

class ConsciousnessInterface:
    """
    Interface between quantum fields and consciousness states.
    
    This class provides methods for mapping consciousness states to
    quantum fields and vice versa, enabling bidirectional interaction.
    """
    
    def __init__(self, field):
        """
        Initialize the consciousness interface.
        
        Parameters:
        -----------
        field : QuantumField
            The quantum field to interface with
        """
        self.field = field
        
        # State patterns for different consciousness states
        self.state_patterns = self._initialize_state_patterns()
        
        # Current consciousness state
        self.current_state = 'alpha'
        
        # Connection strength between field and consciousness
        self.connection_strength = LAMBDA
        
        # Field-consciousness mapping
        self.mappings = {
            'spatial': True,       # Map spatial pattern of field
            'frequency': True,     # Map frequency patterns
            'intensity': True,     # Map intensity patterns
            'coherence': True,     # Map coherence patterns
            'resonance': True      # Map resonance patterns
        }
        
        # Feedback loop parameters
        self.feedback_enabled = False
        self.feedback_strength = 0.5
        self.feedback_delay = 0.1  # seconds
        
    def _initialize_state_patterns(self):
        """Initialize patterns for different consciousness states."""
        patterns = {}
        
        # For each consciousness state, create a field pattern
        for state, properties in CONSCIOUSNESS_STATES.items():
            # Create a base pattern
            pattern = np.zeros(self.field.dimensions)
            
            # Apply state-specific patterns
            if state == 'delta':
                # Slow, large amplitude waves
                self._apply_delta_pattern(pattern)
            elif state == 'theta':
                # Meditation/dream state pattern
                self._apply_theta_pattern(pattern)
            elif state == 'alpha':
                # Relaxed awareness pattern
                self._apply_alpha_pattern(pattern)
            elif state == 'beta':
                # Active thinking pattern
                self._apply_beta_pattern(pattern)
            elif state == 'gamma':
                # Peak performance/insight pattern
                self._apply_gamma_pattern(pattern)
            elif state == 'lambda':
                # Transcendental pattern
                self._apply_lambda_pattern(pattern)
            elif state == 'epsilon':
                # Unified field pattern
                self._apply_epsilon_pattern(pattern)
                
            # Store the pattern
            patterns[state] = pattern
            
        return patterns
        
    def _apply_delta_pattern(self, pattern):
        """Apply delta state pattern (deep sleep)."""
        # Slow, large amplitude waves
        # In a real implementation, this would create a specific pattern
        # For this blueprint, we'll simulate with a simple pattern
        
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Slow wave pattern
                    wave = 0.5 + 0.4 * np.sin(0.1 * (i + j + k))
                    pattern[i, j, k] = wave
                    
    def _apply_theta_pattern(self, pattern):
        """Apply theta state pattern (meditation/dream)."""
        # Meditation/dream state pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Theta wave pattern with phi-harmonic modulation
                    theta = 0.5 + 0.3 * np.sin(0.2 * PHI * (i + j + k))
                    pattern[i, j, k] = theta
                    
    def _apply_alpha_pattern(self, pattern):
        """Apply alpha state pattern (relaxed awareness)."""
        # Relaxed awareness pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Alpha wave pattern with phi-harmonic components
                    alpha = 0.5 + 0.25 * np.sin(0.3 * PHI * i) * np.cos(0.3 * LAMBDA * j)
                    pattern[i, j, k] = alpha
                    
    def _apply_beta_pattern(self, pattern):
        """Apply beta state pattern (active thinking)."""
        # Active thinking pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Beta wave pattern with higher frequency components
                    beta = 0.5 + 0.2 * np.sin(0.5 * i) * np.sin(0.5 * j) * np.sin(0.5 * k)
                    pattern[i, j, k] = beta
                    
    def _apply_gamma_pattern(self, pattern):
        """Apply gamma state pattern (peak performance/insight)."""
        # Peak performance/insight pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Gamma wave pattern with high frequency, phi-harmonic components
                    gamma = 0.5 + 0.15 * np.sin(PHI * i) * np.sin(PHI * j) * np.sin(PHI * k)
                    pattern[i, j, k] = gamma
                    
    def _apply_lambda_pattern(self, pattern):
        """Apply lambda state pattern (transcendental)."""
        # Transcendental pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Lambda wave pattern with very high frequency, phi-harmonic components
                    # and coherent organization
                    distance = np.sqrt((i - pattern.shape[0]/2)**2 + 
                                      (j - pattern.shape[1]/2)**2 + 
                                      (k - pattern.shape[2]/2)**2)
                    phase = PHI * distance
                    lambda_wave = 0.5 + 0.1 * np.sin(2 * PHI * phase)
                    pattern[i, j, k] = lambda_wave
                    
    def _apply_epsilon_pattern(self, pattern):
        """Apply epsilon state pattern (unified field)."""
        # Unified field pattern
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                for k in range(pattern.shape[2]):
                    # Create phi-spiral pattern from center
                    dx = i - pattern.shape[0]/2
                    dy = j - pattern.shape[1]/2
                    dz = k - pattern.shape[2]/2
                    
                    # Calculate radius and angle in spherical coordinates
                    radius = np.sqrt(dx**2 + dy**2 + dz**2)
                    theta = np.arctan2(dy, dx)
                    phi = np.arccos(dz / (radius + 1e-10))
                    
                    # Phi-harmonic spiral pattern
                    spiral = 0.5 + 0.1 * np.sin(PHI * radius + theta + PHI * phi)
                    pattern[i, j, k] = spiral
    
    def set_state(self, state_name, intensity=1.0):
        """
        Set the consciousness state of the field.
        
        Parameters:
        -----------
        state_name : str
            Name of the consciousness state to set
        intensity : float
            Intensity of the state application (0.0 to 1.0)
            
        Returns:
        --------
        bool
            Whether the state was successfully set
        """
        if state_name not in self.state_patterns:
            print(f"Unknown consciousness state: {state_name}")
            return False
            
        # Get the state pattern
        pattern = self.state_patterns[state_name]
        
        # Apply the state pattern to the field
        self.field.apply_consciousness_state(pattern, intensity)
        
        # Update current state
        self.current_state = state_name
        
        print(f"Set consciousness state to {state_name} with intensity {intensity:.2f}")
        return True
    
    def blend_states(self, states_dict):
        """
        Blend multiple consciousness states.
        
        Parameters:
        -----------
        states_dict : dict
            Dictionary mapping state names to weights
            
        Returns:
        --------
        bool
            Whether the blend was successfully applied
        """
        # Validate state names
        for state_name in states_dict:
            if state_name not in self.state_patterns:
                print(f"Unknown consciousness state: {state_name}")
                return False
                
        # Normalize weights
        total_weight = sum(states_dict.values())
        normalized_weights = {s: w/total_weight for s, w in states_dict.items()}
        
        # Create blended pattern
        blended_pattern = np.zeros(self.field.dimensions)
        
        for state_name, weight in normalized_weights.items():
            pattern = self.state_patterns[state_name]
            blended_pattern += pattern * weight
            
        # Apply the blended pattern to the field
        self.field.apply_consciousness_state(blended_pattern, 1.0)
        
        # Update current state to indicate blending
        states_str = '+'.join([f"{s}:{w:.2f}" for s, w in normalized_weights.items()])
        self.current_state = f"blend({states_str})"
        
        print(f"Applied blended consciousness state: {self.current_state}")
        return True
    
    def detect_state(self):
        """
        Detect the current consciousness state from the field.
        
        Returns:
        --------
        dict
            Dictionary with detected state information
        """
        # Calculate correlation with each state pattern
        correlations = {}
        
        for state_name, pattern in self.state_patterns.items():
            # Calculate correlation coefficient
            field_flat = self.field.field.flatten()
            pattern_flat = pattern.flatten()
            
            # Center the data
            field_centered = field_flat - np.mean(field_flat)
            pattern_centered = pattern_flat - np.mean(pattern_flat)
            
            # Calculate correlation coefficient
            numerator = np.sum(field_centered * pattern_centered)
            denominator = np.sqrt(np.sum(field_centered**2) * np.sum(pattern_centered**2))
            
            if denominator == 0:
                correlation = 0
            else:
                correlation = numerator / denominator
                
            correlations[state_name] = correlation
            
        # Find the most correlated state
        best_state = max(correlations, key=correlations.get)
        best_correlation = correlations[best_state]
        
        # Create result dictionary
        result = {
            'primary_state': best_state,
            'correlation': best_correlation,
            'all_correlations': correlations,
            'coherence': self.field.coherence
        }
        
        print(f"Detected consciousness state: {best_state} (correlation: {best_correlation:.3f})")
        return result
    
    def enable_feedback(self, enabled=True, strength=None, delay=None):
        """
        Enable or disable consciousness-field feedback loop.
        
        Parameters:
        -----------
        enabled : bool
            Whether to enable the feedback loop
        strength : float, optional
            Strength of the feedback (0.0 to 1.0)
        delay : float, optional
            Delay between feedback cycles in seconds
        """
        self.feedback_enabled = enabled
        
        if strength is not None:
            self.feedback_strength = max(0.0, min(1.0, strength))
            
        if delay is not None:
            self.feedback_delay = max(0.01, delay)
            
        print(f"Consciousness-field feedback loop {'enabled' if enabled else 'disabled'}")
        if enabled:
            print(f"Feedback strength: {self.feedback_strength:.2f}")
            print(f"Feedback delay: {self.feedback_delay:.2f} seconds")
    
    def set_connection_strength(self, strength):
        """
        Set the connection strength between field and consciousness.
        
        Parameters:
        -----------
        strength : float
            Connection strength (0.0 to 1.0)
        """
        self.connection_strength = max(0.0, min(1.0, strength))
        print(f"Set field-consciousness connection strength to {self.connection_strength:.2f}")
    
    def toggle_mapping(self, mapping_type, enabled=True):
        """
        Enable or disable a specific field-consciousness mapping.
        
        Parameters:
        -----------
        mapping_type : str
            Type of mapping to toggle
        enabled : bool
            Whether to enable the mapping
            
        Returns:
        --------
        bool
            Whether the mapping was successfully toggled
        """
        if mapping_type not in self.mappings:
            print(f"Unknown mapping type: {mapping_type}")
            return False
            
        self.mappings[mapping_type] = enabled
        print(f"{mapping_type.capitalize()} mapping {'enabled' if enabled else 'disabled'}")
        return True
    
    def induce_resonance(self, duration=5.0, intensity=1.0):
        """
        Induce phi-harmonic resonance between field and consciousness.
        
        Parameters:
        -----------
        duration : float
            Duration of the resonance induction in seconds
        intensity : float
            Intensity of the resonance (0.0 to 1.0)
            
        Returns:
        --------
        dict
            Results of the resonance induction
        """
        # In a real implementation, this would synchronize field patterns
        # For this blueprint, we'll simulate the process
        
        print(f"Inducing phi-harmonic resonance for {duration} seconds (intensity: {intensity:.2f})")
        
        # Simulate resonance induction
        # In a real implementation, this would involve field manipulations over time
        
        # Gradually increase field coherence during resonance
        target_coherence = min(1.0, self.field.coherence + intensity * 0.3)
        self.field._set_coherence(target_coherence)
        
        # Results dictionary
        results = {
            'duration': duration,
            'intensity': intensity,
            'initial_coherence': self.field.coherence - intensity * 0.3,
            'final_coherence': self.field.coherence,
            'resonance_factor': intensity * PHI
        }
        
        return results
    
    def integrate_external_input(self, input_data, input_type, weight=0.5):
        """
        Integrate external input (e.g., EEG data) into the consciousness interface.
        
        Parameters:
        -----------
        input_data : ndarray
            External input data
        input_type : str
            Type of input data ('eeg', 'hrv', etc.)
        weight : float
            Weight of the integration (0.0 to 1.0)
            
        Returns:
        --------
        bool
            Whether the input was successfully integrated
        """
        print(f"Integrating external {input_type} input with weight {weight:.2f}")
        
        # Different processing for different input types
        if input_type == 'eeg':
            # Process EEG data
            # In a real implementation, this would analyze frequency bands, etc.
            pass
        elif input_type == 'hrv':
            # Process HRV data
            # In a real implementation, this would analyze heart rate variability
            pass
        else:
            # Generic processing
            pass
            
        # Update the field based on the processed input
        # In a real implementation, this would apply specific patterns
        
        return True