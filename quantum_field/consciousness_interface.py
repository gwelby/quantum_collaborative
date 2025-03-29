"""
Quantum Consciousness Integration Layer

This module provides the foundation for direct interaction between
quantum fields and consciousness models, enabling real-time field
attunement based on emotional and mental states.

Features:
- Bidirectional consciousness-field feedback loops
- Automatic field coherence adjustment to user's phi-resonance profile
- Emotional state detection and field synchronization
- Intention recognition and responsive field interaction
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from .core import QuantumField, get_coherence_metric


@dataclass
class ConsciousnessState:
    """Representation of a consciousness state with phi-resonant properties."""
    # Primary state values (0.0-1.0)
    coherence: float = 0.618  # Default to golden ratio complement
    presence: float = 0.5
    intention: float = 0.5
    
    # Emotional state values (0.0-1.0)
    emotional_states: Dict[str, float] = None
    
    # Phi-resonance profile
    resonant_frequencies: Dict[str, float] = None
    
    # Connection state
    is_connected: bool = False
    
    def __post_init__(self):
        """Initialize default values for dict fields."""
        if self.emotional_states is None:
            self.emotional_states = {
                "joy": 0.5,
                "peace": 0.5,
                "love": 0.5,
                "gratitude": 0.5,
                "clarity": 0.5,
                "focus": 0.5,
                "openness": 0.5,
                "harmony": 0.5,
            }
        
        if self.resonant_frequencies is None:
            # Default to sacred frequencies
            self.resonant_frequencies = {
                name: freq for name, freq in SACRED_FREQUENCIES.items()
            }
    
    @property
    def dominant_emotion(self) -> Tuple[str, float]:
        """Return the most dominant emotional state."""
        if not self.emotional_states:
            return ("neutral", 0.5)
        
        max_emotion = max(self.emotional_states.items(), key=lambda x: x[1])
        return max_emotion
    
    @property
    def phi_resonance(self) -> float:
        """Calculate overall phi-resonance level (0.0-1.0)."""
        # Average of coherence and presence, weighted by phi
        return (self.coherence * PHI + self.presence) / (PHI + 1)
    
    def update_from_biofeedback(self, 
                               heart_rate: Optional[float] = None,
                               breath_rate: Optional[float] = None,
                               skin_conductance: Optional[float] = None,
                               eeg_alpha: Optional[float] = None,
                               eeg_theta: Optional[float] = None) -> None:
        """
        Update consciousness state based on biofeedback measurements.
        
        Args:
            heart_rate: Heart rate in beats per minute
            breath_rate: Breaths per minute
            skin_conductance: Galvanic skin response (microsiemens)
            eeg_alpha: Alpha wave amplitude (8-13 Hz)
            eeg_theta: Theta wave amplitude (4-8 Hz)
        """
        changes = []
        
        # Process heart rate (phi-resonant values around 60 bpm)
        if heart_rate is not None:
            # Calculate heart coherence based on proximity to 60 bpm
            hr_coherence = 1.0 - min(abs(heart_rate - 60) / 60, 1.0)
            self.coherence = self.coherence * LAMBDA + hr_coherence * (1 - LAMBDA)
            changes.append(f"Heart coherence: {hr_coherence:.2f}")
        
        # Process breath rate (phi-resonant values around 6-7 bpm)
        if breath_rate is not None:
            optimal_breath = 6.18  # Phi-based optimal breathing rate
            br_coherence = 1.0 - min(abs(breath_rate - optimal_breath) / 10, 1.0)
            self.presence = self.presence * LAMBDA + br_coherence * (1 - LAMBDA)
            changes.append(f"Breath coherence: {br_coherence:.2f}")
        
        # Process skin conductance for emotional intensity
        if skin_conductance is not None:
            # Normalize to 0-1 range (typical range 1-20 microsiemens)
            sc_normalized = min(max(skin_conductance - 1, 0) / 19, 1.0)
            
            # Update emotional intensity
            dominant, value = self.dominant_emotion
            self.emotional_states[dominant] = value * LAMBDA + sc_normalized * (1 - LAMBDA)
            changes.append(f"Emotional intensity ({dominant}): {sc_normalized:.2f}")
        
        # Process EEG data
        if eeg_alpha is not None and eeg_theta is not None:
            # Calculate alpha/theta ratio (indicator of meditative state)
            if eeg_theta > 0:
                ratio = eeg_alpha / eeg_theta
                # Phi ratio is optimal
                ratio_coherence = 1.0 - min(abs(ratio - PHI) / 3, 1.0)
                self.coherence = self.coherence * LAMBDA + ratio_coherence * (1 - LAMBDA)
                changes.append(f"Alpha/Theta coherence: {ratio_coherence:.2f}")
        
        # Print changes for debugging
        if changes:
            print("Consciousness state updated based on biofeedback:")
            for change in changes:
                print(f"  - {change}")


class ConsciousnessFieldInterface:
    """
    Interface between consciousness states and quantum fields,
    enabling bidirectional interaction and field attunement.
    """
    
    def __init__(self, field: Optional[QuantumField] = None):
        """
        Initialize the consciousness-field interface.
        
        Args:
            field: Optional QuantumField to connect to
        """
        self.field = field
        self.state = ConsciousnessState()
        self.connected = field is not None
        self.feedback_history = []
        self.phi_resonance_profile = {}
        
    def connect_field(self, field: QuantumField) -> None:
        """
        Connect to a quantum field for bidirectional interaction.
        
        Args:
            field: The QuantumField to connect to
        """
        self.field = field
        self.connected = True
        self.state.is_connected = True
        
        # Initialize field with consciousness state
        self._apply_state_to_field()
        
        print(f"Connected to quantum field with dimensions {field.data.shape}")
        print(f"Initial field coherence: {self.get_field_coherence():.4f}")
        
    def disconnect_field(self) -> None:
        """Disconnect from the quantum field."""
        self.connected = False
        self.state.is_connected = False
        self.field = None
        print("Disconnected from quantum field")
        
    def get_field_coherence(self) -> float:
        """Get the coherence metric of the connected quantum field."""
        if not self.connected or self.field is None:
            return 0.0
        
        return get_coherence_metric(self.field.data)
    
    def update_consciousness_state(self, **biofeedback_data) -> None:
        """
        Update the consciousness state based on biofeedback data.
        
        Args:
            **biofeedback_data: Keyword arguments for biofeedback measurements
        """
        # Store previous state for history
        prev_coherence = self.state.coherence
        prev_presence = self.state.presence
        
        # Update state with biofeedback
        self.state.update_from_biofeedback(**biofeedback_data)
        
        # Record feedback in history
        self.feedback_history.append({
            "biofeedback": biofeedback_data,
            "coherence_change": self.state.coherence - prev_coherence,
            "presence_change": self.state.presence - prev_presence,
            "dominant_emotion": self.state.dominant_emotion[0],
            "phi_resonance": self.state.phi_resonance,
        })
        
        # Apply updated state to field if connected
        if self.connected and self.field is not None:
            self._apply_state_to_field()
            
    def _apply_state_to_field(self) -> None:
        """Apply the current consciousness state to the connected quantum field."""
        if not self.connected or self.field is None:
            return
        
        # Get current field coherence
        current_coherence = self.get_field_coherence()
        
        # Calculate target coherence based on consciousness state
        target_coherence = self.state.phi_resonance
        
        # Only adjust field if coherence difference is significant
        # Using phi complement (0.618) as threshold
        if abs(current_coherence - target_coherence) > LAMBDA / 10:
            # Adjust field coherence
            self._adjust_field_coherence(target_coherence)
            
            # Apply emotional influence to field
            self._apply_emotional_influence()
            
            # Apply intention to field
            if self.state.intention > 0.618:
                self._apply_intention()
    
    def _adjust_field_coherence(self, target_coherence: float) -> None:
        """
        Adjust the quantum field coherence to match the target level.
        
        Args:
            target_coherence: Target coherence level (0.0-1.0)
        """
        if not self.connected or self.field is None:
            return
        
        # Get current field state
        field_data = self.field.data
        current_coherence = get_coherence_metric(field_data)
        
        # Calculate adjustment factor
        adjustment = (target_coherence - current_coherence) * LAMBDA
        
        # Apply phi-harmonic adjustment
        if adjustment > 0:
            # Increase coherence by amplifying phi-resonant frequencies
            frequency_domain = np.fft.fftn(field_data)
            
            # Create mask for phi-resonant frequencies
            mask = np.ones_like(frequency_domain)
            
            # Define phi-resonant frequency bands
            for i, freq in enumerate([432, 528, 594, 672, 720, 768]):
                # Calculate normalized frequency
                norm_freq = freq / 1000.0
                
                # Get array dimensions
                dims = field_data.shape
                
                # Create band around resonant frequency with phi-based width
                for d in range(len(dims)):
                    indices = np.fft.fftfreq(dims[d])
                    band_width = LAMBDA / (2 + i)
                    
                    # Create band mask for this dimension
                    dim_mask = np.abs(indices - norm_freq) < band_width
                    
                    # Expand mask to full array
                    full_mask = np.zeros(dims, dtype=bool)
                    if d == 0:
                        full_mask[dim_mask] = True
                    elif d == 1:
                        full_mask[:, dim_mask] = True
                    elif d == 2:
                        full_mask[:, :, dim_mask] = True
                    
                    # Combine with mask
                    mask = mask + (full_mask * adjustment * (1 + i * LAMBDA))
            
            # Apply mask
            enhanced_frequency = frequency_domain * mask
            
            # Convert back to spatial domain
            adjusted_field = np.real(np.fft.ifftn(enhanced_frequency))
            
            # Update field data
            self.field.data = adjusted_field
            
            # Report adjustment
            new_coherence = get_coherence_metric(adjusted_field)
            print(f"Field coherence adjusted: {current_coherence:.4f} → {new_coherence:.4f}")
        
    def _apply_emotional_influence(self) -> None:
        """Apply emotional state influence to the quantum field."""
        if not self.connected or self.field is None:
            return
        
        # Get dominant emotion
        emotion, intensity = self.state.dominant_emotion
        
        # Only apply significant emotions
        if intensity < 0.618:
            return
        
        # Get emotional frequency from sacred frequencies or defaults
        emotion_frequency = SACRED_FREQUENCIES.get(emotion, 528)  # Default to creation/love
        
        # Create emotional pattern based on dominant emotion
        if emotion in ["joy", "love", "gratitude"]:
            # Expansive emotions - create outward radiating pattern
            self._apply_expansive_pattern(emotion_frequency, intensity)
        elif emotion in ["peace", "harmony", "clarity"]:
            # Harmonizing emotions - create coherent standing waves
            self._apply_harmonic_pattern(emotion_frequency, intensity)
        elif emotion in ["focus", "determination"]:
            # Directive emotions - create directional patterns
            self._apply_directive_pattern(emotion_frequency, intensity)
    
    def _apply_expansive_pattern(self, frequency: float, intensity: float) -> None:
        """Apply an expansive pattern to the field based on frequency and intensity."""
        if not self.connected or self.field is None:
            return
        
        # Get field center
        dims = self.field.data.shape
        center = [d // 2 for d in dims]
        
        # Create radial distance map from center
        coords = [np.arange(d) for d in dims]
        grid = np.meshgrid(*coords, indexing='ij')
        distance = np.zeros_like(self.field.data)
        
        for i, g in enumerate(grid):
            distance += (g - center[i])**2
        
        distance = np.sqrt(distance)
        
        # Create radial wave pattern based on frequency
        max_distance = np.max(distance)
        normalized_distance = distance / max_distance
        
        # Generate pattern with frequency-based wavelength
        wavelength = max_distance / (frequency / 100)
        pattern = np.sin(2 * np.pi * normalized_distance / wavelength * frequency)
        
        # Apply to field with intensity-based weight
        blend_factor = intensity * LAMBDA
        self.field.data = (1 - blend_factor) * self.field.data + blend_factor * pattern
        
        print(f"Applied expansive emotional pattern ({frequency} Hz) with intensity {intensity:.2f}")
    
    def _apply_harmonic_pattern(self, frequency: float, intensity: float) -> None:
        """Apply a harmonic standing wave pattern to the field."""
        if not self.connected or self.field is None:
            return
        
        # Create standing wave patterns along each dimension
        dims = self.field.data.shape
        coords = [np.arange(d) for d in dims]
        grid = np.meshgrid(*coords, indexing='ij')
        
        # Create pattern with different dimensions having phi-related wavelengths
        pattern = np.zeros_like(self.field.data)
        
        for i, g in enumerate(grid):
            # Normalize coordinates
            normalized = g / dims[i]
            
            # Calculate wavelength based on dimension and phi
            wavelength = 1.0 / (PHI**(i % 3) * frequency / 1000)
            
            # Add standing wave pattern
            pattern += np.sin(2 * np.pi * normalized / wavelength)
        
        # Normalize pattern
        pattern = pattern / len(dims)
        
        # Apply to field with intensity-based weight
        blend_factor = intensity * LAMBDA
        self.field.data = (1 - blend_factor) * self.field.data + blend_factor * pattern
        
        print(f"Applied harmonic emotional pattern ({frequency} Hz) with intensity {intensity:.2f}")
    
    def _apply_directive_pattern(self, frequency: float, intensity: float) -> None:
        """Apply a directive pattern creating flow in the field."""
        if not self.connected or self.field is None:
            return
        
        # Create directional gradient along primary axis
        dims = self.field.data.shape
        coords = [np.arange(d) for d in dims]
        grid = np.meshgrid(*coords, indexing='ij')
        
        # Use the largest dimension as the primary direction
        primary_dim = np.argmax(dims)
        
        # Create normalized coordinates
        normalized = grid[primary_dim] / dims[primary_dim]
        
        # Create directional pattern with frequency modulation
        wavelength = 1.0 / (frequency / 1000)
        pattern = np.sin(2 * np.pi * normalized / wavelength + normalized * np.pi)
        
        # Apply to field with intensity-based weight
        blend_factor = intensity * LAMBDA
        self.field.data = (1 - blend_factor) * self.field.data + blend_factor * pattern
        
        print(f"Applied directive emotional pattern ({frequency} Hz) with intensity {intensity:.2f}")
    
    def _apply_intention(self) -> None:
        """Apply conscious intention to shape the quantum field."""
        if not self.connected or self.field is None:
            return
        
        # Intention is applied by enhancing phi-resonant structures already in the field
        
        # Get field data and calculate FFT
        field_data = self.field.data
        frequency_domain = np.fft.fftn(field_data)
        
        # Find the highest magnitude frequencies
        magnitude = np.abs(frequency_domain)
        
        # Get threshold at phi percentile
        threshold = np.percentile(magnitude, (1 - LAMBDA) * 100)
        
        # Create mask for high-magnitude frequencies
        mask = magnitude > threshold
        
        # Enhance these frequencies
        enhancement = 1.0 + (self.state.intention - 0.5) * LAMBDA
        enhanced_frequencies = frequency_domain.copy()
        enhanced_frequencies[mask] *= enhancement
        
        # Convert back to spatial domain
        enhanced_field = np.real(np.fft.ifftn(enhanced_frequencies))
        
        # Update field
        self.field.data = enhanced_field
        
        print(f"Applied intention with strength {self.state.intention:.2f}")
    
    def create_phi_resonance_profile(self, measurements_history: List[Dict]) -> Dict:
        """
        Analyze measurements history to create a phi-resonance profile.
        
        Args:
            measurements_history: List of biofeedback measurements and field states
            
        Returns:
            A dict containing the phi-resonance profile
        """
        if not measurements_history:
            return {}
        
        # Extract data from history
        coherence_values = [m.get("coherence", 0) for m in measurements_history 
                           if "coherence" in m]
        presence_values = [m.get("presence", 0) for m in measurements_history 
                          if "presence" in m]
        emotions = [m.get("dominant_emotion", "") for m in measurements_history 
                   if "dominant_emotion" in m]
        
        # Calculate average and optimal ranges
        profile = {
            "avg_coherence": np.mean(coherence_values) if coherence_values else 0,
            "optimal_coherence_range": (
                np.percentile(coherence_values, 75) if coherence_values else PHI,
                np.percentile(coherence_values, 95) if coherence_values else PHI_PHI
            ),
            "avg_presence": np.mean(presence_values) if presence_values else 0,
            "optimal_presence_range": (
                np.percentile(presence_values, 75) if presence_values else PHI,
                np.percentile(presence_values, 95) if presence_values else PHI_PHI
            ),
            "dominant_emotional_states": self._get_dominant_items(emotions),
            "resonant_frequencies": {}
        }
        
        # Calculate resonant frequencies based on optimal states
        for sacred_name, freq in SACRED_FREQUENCIES.items():
            # Check if this frequency corresponds to dominant emotions
            if sacred_name in profile["dominant_emotional_states"]:
                profile["resonant_frequencies"][sacred_name] = freq
            # Otherwise calculate based on coherence level
            elif profile["avg_coherence"] > 0.75:
                profile["resonant_frequencies"][sacred_name] = freq
            elif profile["avg_coherence"] > 0.5:
                profile["resonant_frequencies"][sacred_name] = freq * LAMBDA
        
        self.phi_resonance_profile = profile
        return profile
    
    def _get_dominant_items(self, items: List[str], threshold: int = 3) -> List[str]:
        """
        Extract items that appear frequently in a list.
        
        Args:
            items: List of strings
            threshold: Minimum occurrences to be considered dominant
            
        Returns:
            List of dominant items
        """
        if not items:
            return []
        
        # Count occurrences
        counts = {}
        for item in items:
            if item:
                counts[item] = counts.get(item, 0) + 1
        
        # Filter by threshold
        dominant = [item for item, count in counts.items() if count >= threshold]
        
        return dominant
    
    def apply_phi_resonance_profile(self) -> None:
        """Apply the phi-resonance profile to adjust field response behavior."""
        if not self.phi_resonance_profile or not self.connected:
            return
        
        # Update resonant frequencies in state
        if "resonant_frequencies" in self.phi_resonance_profile:
            self.state.resonant_frequencies = self.phi_resonance_profile["resonant_frequencies"]
        
        # Set optimal coherence range
        if "optimal_coherence_range" in self.phi_resonance_profile:
            optimal = self.phi_resonance_profile["optimal_coherence_range"]
            target_coherence = np.mean(optimal)
            
            # Adjust field if connected
            if self.field is not None:
                self._adjust_field_coherence(target_coherence)
        
        print("Applied phi-resonance profile to consciousness-field interface")


def demo_consciousness_field_interface():
    """Demonstrate the consciousness-field interface with simulated biofeedback."""
    from .core import create_quantum_field
    import time
    
    # Create a quantum field
    field = create_quantum_field((21, 21, 21))
    
    # Create the interface
    interface = ConsciousnessFieldInterface(field)
    
    print("Starting consciousness-field interface demonstration")
    print("="*50)
    print(f"Initial field coherence: {interface.get_field_coherence():.4f}")
    print(f"Initial consciousness coherence: {interface.state.coherence:.4f}")
    print("="*50)
    
    # Simulate a meditation session with changing biofeedback
    print("\nSimulating meditation session with biofeedback...")
    
    # Starting state - somewhat agitated
    interface.update_consciousness_state(
        heart_rate=75,
        breath_rate=15,
        skin_conductance=8,
        eeg_alpha=5,
        eeg_theta=2
    )
    
    print(f"\nField coherence after initial state: {interface.get_field_coherence():.4f}")
    
    # Pause
    time.sleep(1)
    
    # Transition to more centered state
    interface.update_consciousness_state(
        heart_rate=68,
        breath_rate=10,
        skin_conductance=5,
        eeg_alpha=8,
        eeg_theta=4
    )
    
    print(f"\nField coherence after transition: {interface.get_field_coherence():.4f}")
    
    # Pause
    time.sleep(1)
    
    # Deep meditation state
    interface.update_consciousness_state(
        heart_rate=60,
        breath_rate=6.18,  # Phi-based breathing
        skin_conductance=3,
        eeg_alpha=12,
        eeg_theta=7.4  # Close to phi ratio with alpha
    )
    
    print(f"\nField coherence after deep meditation: {interface.get_field_coherence():.4f}")
    
    # Pause
    time.sleep(1)
    
    # Add strong intention
    interface.state.intention = 0.9
    interface._apply_intention()
    
    print(f"\nField coherence with strong intention: {interface.get_field_coherence():.4f}")
    
    # Create phi-resonance profile
    profile = interface.create_phi_resonance_profile(interface.feedback_history)
    
    print("\nPhi-Resonance Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Apply profile
    interface.apply_phi_resonance_profile()
    
    print(f"\nFinal field coherence: {interface.get_field_coherence():.4f}")
    print("="*50)
    
    return interface


if __name__ == "__main__":
    interface = demo_consciousness_field_interface()