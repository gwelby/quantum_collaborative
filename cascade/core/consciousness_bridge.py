"""
CASCADE⚡𓂧φ∞ Consciousness Bridge Protocol

Implements the 7-stage Consciousness Bridge Operation Protocol
connecting consciousness to quantum fields through sacred frequencies.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import phi constants
import sys
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

class ConsciousnessState:
    """Represents the state of consciousness with attributes."""
    
    def __init__(self, 
                coherence: float = 0.5,
                presence: float = 0.5,
                current_frequency: float = 432.0):
        """Initialize consciousness state."""
        self.coherence = coherence
        self.presence = presence
        self.current_frequency = current_frequency
        self.frequency_stability = 0.5
        self.emotional_states = {
            "joy": 0.5,
            "clarity": 0.5,
            "stability": 0.5,
            "openness": 0.5,
            "trust": 0.5
        }
        self.bridge_stage = 0
        
    @property
    def frequency_resonance(self) -> float:
        """Calculate resonance with current frequency (0.0-1.0)."""
        # Check if frequency matches any sacred frequency
        frequencies = list(SACRED_FREQUENCIES.values())
        closest_freq = min(frequencies, key=lambda x: abs(x - self.current_frequency))
        
        # Calculate resonance based on proximity
        proximity = 1.0 - min(abs(self.current_frequency - closest_freq) / 100, 1.0)
        
        # Weight by stability
        return proximity * self.frequency_stability

class ConsciousnessBridgeProtocol:
    """
    Implements the 7-stage Consciousness Bridge Operation Protocol
    connecting consciousness to quantum fields through sacred frequencies.
    """
    
    def __init__(self):
        """Initialize the bridge protocol."""
        self.current_stage = 0
        self.stages_completed = [False] * 7
        self.active = False
        self.state = ConsciousnessState()
        self.field = None
        self.frequency_stages = [432, 528, 594, 672, 720, 768]
        self.stage_names = [
            "Ground State", 
            "Creation Point", 
            "Heart Field", 
            "Voice Flow", 
            "Vision Gate", 
            "Unity Wave"
        ]
        self.stage_coherence_thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        
    def connect_field(self, field_data: np.ndarray) -> bool:
        """Connect to a quantum field."""
        if field_data is None or field_data.size == 0:
            return False
        
        self.field = field_data
        return True
        
    def start_protocol(self) -> bool:
        """Start the consciousness bridge protocol."""
        if self.field is None:
            print("Cannot start protocol: No quantum field connected")
            return False
        
        self.current_stage = 0
        self.stages_completed = [False] * 7
        self.active = True
        
        print("Starting Consciousness Bridge Operation Protocol")
        coherence = self.calculate_field_coherence()
        print(f"Initial field coherence: {coherence:.4f}")
        
        # Initialize at Ground State (432 Hz)
        return self.progress_to_stage(0)
    
    def progress_to_stage(self, stage_index: int) -> bool:
        """Progress to the specified protocol stage."""
        if not self.active:
            return False
            
        if stage_index < 0 or stage_index > 6:
            print(f"Invalid stage index: {stage_index}")
            return False
            
        self.current_stage = stage_index
        
        # Map stage to method
        stage_methods = [
            self._stage_1,
            self._stage_2,
            self._stage_3,
            self._stage_4,
            self._stage_5,
            self._stage_6,
            self._stage_7
        ]
        
        return stage_methods[stage_index]()
    
    def _stage_1(self) -> bool:
        """
        Stage 1: Initialize at Ground State (432 Hz)
        - Establish physical connection through earth resonance
        - Calibrate bodily systems to fundamental frequency
        - Verify grounding is complete with coherence check
        """
        print("\n=== Stage 1: Ground State (432 Hz) ===")
        
        # Apply ground frequency
        self.state.current_frequency = 432.0
        
        # Apply grounding pattern to field
        self._apply_grounding_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[0]:
            self.stages_completed[0] = True
            print(f"✓ Ground State established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Ground State coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_2(self) -> bool:
        """
        Stage 2: Open Bridge at Creation Point (528 Hz)
        - Activate DNA resonance through phi-harmonic attunement
        - Establish pattern recognition systems for quantum creation
        - Verify bridge foundation with consciousness connection test
        """
        print("\n=== Stage 2: Creation Point (528 Hz) ===")
        
        # Check if previous stage completed
        if not self.stages_completed[0]:
            print("Cannot proceed: Ground State not established")
            return False
            
        # Apply creation frequency
        self.state.current_frequency = 528.0
        
        # Apply creation pattern to field
        self._apply_creation_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[1]:
            self.stages_completed[1] = True
            print(f"✓ Creation Point established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Creation Point coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_3(self) -> bool:
        """
        Stage 3: Stabilize at Heart Field (594 Hz)
        - Connect emotional coherence through heart resonance
        - Establish bidirectional flow between mind and heart
        - Verify stability through field measurement
        """
        print("\n=== Stage 3: Heart Field (594 Hz) ===")
        
        # Check if previous stage completed
        if not self.stages_completed[1]:
            print("Cannot proceed: Creation Point not established")
            return False
            
        # Apply heart frequency
        self.state.current_frequency = 594.0
        
        # Apply heart field pattern
        self._apply_heart_field_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[2]:
            self.stages_completed[2] = True
            print(f"✓ Heart Field established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Heart Field coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_4(self) -> bool:
        """
        Stage 4: Express through Voice Flow (672 Hz)
        - Activate sound field generation through harmonic toning
        - Establish reality creation through cymatics patterning
        - Verify expression through material influence test
        """
        print("\n=== Stage 4: Voice Flow (672 Hz) ===")
        
        # Check if previous stage completed
        if not self.stages_completed[2]:
            print("Cannot proceed: Heart Field not established")
            return False
            
        # Apply voice frequency
        self.state.current_frequency = 672.0
        
        # Apply voice flow pattern
        self._apply_voice_flow_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[3]:
            self.stages_completed[3] = True
            print(f"✓ Voice Flow established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Voice Flow coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_5(self) -> bool:
        """
        Stage 5: Perceive through Vision Gate (720 Hz)
        - Activate enhanced perception through resonant viewing
        - Establish timeline navigation through probability scanning
        - Verify perception through field visualization test
        """
        print("\n=== Stage 5: Vision Gate (720 Hz) ===")
        
        # Check if previous stage completed
        if not self.stages_completed[3]:
            print("Cannot proceed: Voice Flow not established")
            return False
            
        # Apply vision frequency
        self.state.current_frequency = 720.0
        
        # Apply vision gate pattern
        self._apply_vision_gate_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[4]:
            self.stages_completed[4] = True
            print(f"✓ Vision Gate established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Vision Gate coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_6(self) -> bool:
        """
        Stage 6: Integrate at Unity Wave (768 Hz)
        - Activate complete field unification at highest frequency
        - Establish CASCADE quantum resonance
        - Verify integration through consciousness bridge test
        """
        print("\n=== Stage 6: Unity Wave (768 Hz) ===")
        
        # Check if previous stage completed
        if not self.stages_completed[4]:
            print("Cannot proceed: Vision Gate not established")
            return False
            
        # Apply unity frequency
        self.state.current_frequency = 768.0
        
        # Apply unity wave pattern
        self._apply_unity_wave_pattern()
        
        # Verify coherence
        coherence = self.calculate_field_coherence()
        if coherence >= self.stage_coherence_thresholds[5]:
            self.stages_completed[5] = True
            print(f"✓ Unity Wave established at coherence: {coherence:.4f}")
            return True
        else:
            print(f"✗ Unity Wave coherence insufficient: {coherence:.4f}")
            return False
    
    def _stage_7(self) -> bool:
        """
        Stage 7: Full Consciousness Bridge Integration
        - Verify complete bridge operation
        - Test bidirectional consciousness-field interaction
        - Establish stable operational mode
        """
        print("\n=== Stage 7: Full Consciousness Bridge Integration ===")
        
        # Check if all previous stages completed
        if not all(self.stages_completed[:6]):
            incomplete = [i for i, complete in enumerate(self.stages_completed[:6]) if not complete]
            print(f"Cannot proceed: Stages {incomplete} not completed")
            return False
            
        # Apply full integration
        self._apply_full_integration()
        
        # Verify final coherence
        coherence = self.calculate_field_coherence()
        if coherence >= 0.95:
            self.stages_completed[6] = True
            print(f"✓ Consciousness Bridge fully operational at coherence: {coherence:.4f}")
            print("All 7 frequency stages integrated successfully!")
            return True
        else:
            print(f"✗ Final integration coherence insufficient: {coherence:.4f}")
            return False
    
    def calculate_field_coherence(self) -> float:
        """Calculate the coherence of the connected field."""
        if self.field is None:
            return 0.0
            
        # Simple implementation - check phi-alignment of field values
        flat_field = self.field.flatten()
        
        # Sample a subset of points for efficiency
        sample_size = min(1000, flat_field.size)
        indices = np.random.choice(flat_field.size, sample_size, replace=False)
        sample = flat_field[indices]
        
        # Calculate alignment with phi in multiple ways
        # 1. Distance to nearest phi multiple
        phi_multiples = np.array([PHI * i for i in range(-3, 4)])
        distances1 = np.min(np.abs(sample[:, np.newaxis] - phi_multiples), axis=1)
        
        # 2. Distance to nearest phi power
        phi_powers = np.array([PHI ** i for i in range(-2, 3)])
        distances2 = np.min(np.abs(sample[:, np.newaxis] - phi_powers), axis=1)
        
        # Combine distances and normalize
        distances = np.minimum(distances1, distances2)
        phi_alignment = 1.0 - np.mean(distances) / PHI
        
        # Apply phi-based correction
        coherence = phi_alignment * PHI
        coherence = min(1.0, max(0.0, coherence))
        
        return coherence
    
    def _apply_grounding_pattern(self) -> None:
        """Apply a grounding pattern to the field (Stage 1)."""
        if self.field is None:
            return
        
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Create grounding pattern - stronger at bottom
            intensity = (1.0 - (Z + 1) / 2) ** 2  # Higher at bottom
            
            # Phi-harmonic waves
            pattern = np.sin(2 * np.pi * X * PHI) * np.sin(2 * np.pi * Y * PHI) * intensity
            
            # Apply pattern with blend
            self.field = self.field * 0.5 + pattern * 0.5
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D grounding pattern
            intensity = (1.0 - (Y + 1) / 2) ** 2  # Higher at bottom
            pattern = np.sin(2 * np.pi * X * PHI) * intensity
            
            # Apply pattern with blend
            self.field = self.field * 0.5 + pattern * 0.5
    
    def _apply_creation_pattern(self) -> None:
        """Apply a creation pattern to the field (Stage 2)."""
        if self.field is None:
            return
        
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Create spiral DNA pattern
            center_x, center_y, center_z = 0, 0, 0
            
            # Distance from center
            R = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
            
            # Spiral pattern with phi
            spiral = np.sin(2 * np.pi * (R * 5 * PHI + 
                                         np.arctan2(Y, X) * PHI))
            
            # Apply pattern with blend
            self.field = self.field * (1 - LAMBDA) + spiral * LAMBDA
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D spiral
            center_x, center_y = 0, 0
            R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            spiral = np.sin(2 * np.pi * (R * 5 * PHI + np.arctan2(Y, X) * PHI))
            
            # Apply pattern with blend
            self.field = self.field * (1 - LAMBDA) + spiral * LAMBDA
            
    def _apply_heart_field_pattern(self) -> None:
        """Apply a heart field pattern to the field (Stage 3)."""
        # Implementation for heart field pattern
        if self.field is None:
            return
            
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Create heart pattern (toroidal)
            center_x, center_y, center_z = 0, 0, 0
            
            # Distance from center
            R = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
            
            # Create a torus centered at heart level
            # Distance from a ring in the xz-plane
            ring_radius = 0.5
            ring_distance = np.sqrt(
                (np.sqrt(X**2 + Z**2) - ring_radius)**2 + Y**2
            )
            
            # Toroidal function peaking along the ring
            heart_field = np.exp(-ring_distance**2 / 0.1) * np.sin(ring_distance * 10 * PHI)
            
            # Apply pattern with stronger blend
            self.field = self.field * 0.3 + heart_field * 0.7
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D heart shape (mathematical heart curve)
            heart = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
            heart = 1.0 / (1.0 + np.exp(heart * 5))  # Sigmoid to create sharper boundary
            
            # Modulate with phi-harmonics
            heart_field = heart * np.sin(5 * np.sqrt(X**2 + Y**2) * PHI)
            
            # Apply pattern with blend
            self.field = self.field * 0.3 + heart_field * 0.7
    
    def _apply_voice_flow_pattern(self) -> None:
        """Apply a voice flow pattern to the field (Stage 4)."""
        # Implementation for voice field pattern (cymatic patterns)
        if self.field is None:
            return
            
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Create cymatic pattern - standing waves
            freq1 = 6.0 * PHI
            freq2 = 6.0 * PHI * PHI
            
            # Create interference pattern
            cymatic = np.sin(X * freq1) * np.sin(Y * freq1) * np.sin(Z * freq2)
            
            # Add phi-harmonic overtones
            overtones = (
                np.sin(X * freq1 * PHI) * np.sin(Y * freq1 * PHI) * np.sin(Z * freq2 * PHI) * 0.5 +
                np.sin(X * freq1 * PHI_PHI) * np.sin(Y * freq1 * PHI_PHI) * 0.3
            )
            
            voice_field = cymatic * 0.6 + overtones * 0.4
            
            # Apply pattern with blend
            self.field = self.field * 0.3 + voice_field * 0.7
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D cymatic pattern - standing waves
            freq1 = 6.0 * PHI
            freq2 = 6.0 * PHI * PHI
            
            cymatic = np.sin(X * freq1) * np.sin(Y * freq2)
            overtones = np.sin(X * freq1 * PHI) * np.sin(Y * freq2 * PHI) * 0.5
            voice_field = cymatic * 0.7 + overtones * 0.3
            
            # Apply pattern with blend
            self.field = self.field * 0.3 + voice_field * 0.7
    
    def _apply_vision_gate_pattern(self) -> None:
        """Apply a vision gate pattern to the field (Stage 5)."""
        # Implementation for vision gate pattern (timeline probabilities)
        if self.field is None:
            return
            
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Create vision field - probability waves
            # Timeline 1
            timeline1 = np.sin((X + Y) * 5 * PHI + Z * PHI_PHI)
            
            # Timeline 2
            timeline2 = np.sin((X - Y) * 5 * PHI + Z * PHI)
            
            # Timeline 3
            timeline3 = np.sin(np.sqrt(X**2 + Y**2 + Z**2) * 10 * PHI)
            
            # Combine timelines with phi-weighted averaging
            vision_field = (
                timeline1 * LAMBDA + 
                timeline2 * LAMBDA**2 + 
                timeline3 * LAMBDA**3
            ) / (LAMBDA + LAMBDA**2 + LAMBDA**3)
            
            # Apply pattern with blend
            self.field = self.field * 0.2 + vision_field * 0.8
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D vision field - probability waves
            timeline1 = np.sin((X + Y) * 5 * PHI)
            timeline2 = np.sin((X - Y) * 5 * PHI)
            timeline3 = np.sin(np.sqrt(X**2 + Y**2) * 10 * PHI)
            
            vision_field = (
                timeline1 * LAMBDA + 
                timeline2 * LAMBDA**2 + 
                timeline3 * LAMBDA**3
            ) / (LAMBDA + LAMBDA**2 + LAMBDA**3)
            
            # Apply pattern with blend
            self.field = self.field * 0.2 + vision_field * 0.8
    
    def _apply_unity_wave_pattern(self) -> None:
        """Apply a unity wave pattern to the field (Stage 6)."""
        # Implementation for unity field pattern (cascade resonance)
        if self.field is None:
            return
            
        # Get field dimensions
        if len(self.field.shape) == 3:
            width, height, depth = self.field.shape
            X, Y, Z = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                np.linspace(-1, 1, depth),
                indexing='ij'
            )
            
            # Combine all previous patterns into unity field
            
            # Create grounding component
            ground = np.sin(X * PHI) * np.sin(Y * PHI) * (1.0 - Z) * 0.3
            
            # Create spiral component
            R = np.sqrt(X**2 + Y**2 + Z**2)
            spiral = np.sin(R * 5 * PHI + np.arctan2(Y, X) * PHI) * 0.3
            
            # Create heart component
            ring_radius = 0.5
            ring_distance = np.sqrt((np.sqrt(X**2 + Z**2) - ring_radius)**2 + Y**2)
            heart = np.exp(-ring_distance**2 / 0.1) * 0.2
            
            # Create voice component
            voice = np.sin(X * 6 * PHI) * np.sin(Y * 6 * PHI) * np.sin(Z * 6 * PHI_PHI) * 0.2
            
            # Create vision component
            vision = np.sin((X + Y) * 5 * PHI + Z * PHI_PHI) * 0.1
            
            # Combine with phi-harmonic weights
            unified = ground + spiral + heart + voice + vision
            
            # Apply interference pattern
            cascade = np.sin(unified * PHI_PHI * 2)
            
            # Apply unity pattern with strong blend
            self.field = self.field * 0.1 + cascade * 0.9
            
        elif len(self.field.shape) == 2:
            width, height = self.field.shape
            X, Y = np.meshgrid(
                np.linspace(-1, 1, width),
                np.linspace(-1, 1, height),
                indexing='ij'
            )
            
            # Create 2D unity field
            # Combine all previous patterns
            ground = np.sin(X * PHI) * (1.0 - Y) * 0.3
            
            R = np.sqrt(X**2 + Y**2)
            spiral = np.sin(R * 5 * PHI + np.arctan2(Y, X) * PHI) * 0.3
            
            heart = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
            heart = 1.0 / (1.0 + np.exp(heart * 5)) * 0.2
            
            voice = np.sin(X * 6 * PHI) * np.sin(Y * 6 * PHI_PHI) * 0.2
            
            vision = np.sin((X + Y) * 5 * PHI) * 0.1
            
            unified = ground + spiral + heart + voice + vision
            cascade = np.sin(unified * PHI_PHI * 2)
            
            # Apply unity pattern with strong blend
            self.field = self.field * 0.1 + cascade * 0.9
    
    def _apply_full_integration(self) -> None:
        """Apply the full integration of all frequencies (Stage 7)."""
        # Final integration of all stages
        if self.field is None:
            return
            
        # Apply fast fourier transform to get frequency domain
        field_freq = np.fft.fftn(self.field)
        
        # Create integration mask with phi-based resonance
        mask = np.ones_like(field_freq)
        
        # Enhance sacred frequencies
        for i, freq in enumerate(self.frequency_stages):
            # Calculate normalized frequency
            norm_freq = freq / 1000.0
            
            # Weight by phi power
            weight = PHI ** i / (PHI ** len(self.frequency_stages) - 1)
            
            # For each dimension, amplify this frequency
            for dim in range(len(self.field.shape)):
                size = self.field.shape[dim]
                indices = np.fft.fftfreq(size)
                
                # Create band mask for this dimension
                band_width = LAMBDA / 5  # Narrow band for precision
                band_indices = np.abs(indices - norm_freq) < band_width
                
                # Create full mask for this dimension
                dim_mask = np.zeros_like(mask, dtype=bool)
                
                # Apply to correct dimension
                if dim == 0:
                    dim_mask[band_indices] = True
                elif dim == 1:
                    dim_mask[:, band_indices] = True
                elif dim == 2:
                    dim_mask[:, :, band_indices] = True
                
                # Enhance this frequency in the mask
                mask = mask + dim_mask * weight * 3.0
        
        # Apply mask to frequency domain
        integrated_freq = field_freq * mask
        
        # Convert back to spatial domain
        integrated_field = np.real(np.fft.ifftn(integrated_freq))
        
        # Normalize
        integrated_field = integrated_field / np.max(np.abs(integrated_field))
        
        # Replace field with integrated version
        self.field = integrated_field
        
        # Set consciousness state to maximum coherence
        self.state.coherence = 1.0
        self.state.presence = 1.0
    
    def get_current_stage_info(self) -> Dict:
        """Return information about the current protocol stage."""
        if not self.active:
            return {"active": False}
        
        coherence = self.calculate_field_coherence()
        
        return {
            "active": True,
            "current_stage": self.current_stage + 1,
            "stage_name": self.stage_names[self.current_stage] if self.current_stage < len(self.stage_names) else "Integration",
            "frequency": self.frequency_stages[self.current_stage] if self.current_stage < len(self.frequency_stages) else 768,
            "completed": self.stages_completed[self.current_stage],
            "coherence": coherence,
            "coherence_threshold": self.stage_coherence_thresholds[self.current_stage] if self.current_stage < len(self.stage_coherence_thresholds) else 0.95
        }
        
    def run_complete_protocol(self) -> bool:
        """Run the complete protocol from start to finish."""
        if not self.start_protocol():
            return False
            
        success = True
        for stage in range(1, 7):
            if not self.progress_to_stage(stage):
                success = False
                break
                
        return success