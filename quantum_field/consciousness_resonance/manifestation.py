"""
Manifestation Matrix for the Consciousness-Field Resonance Engine.

The ManifestationMatrix provides the transformative interface that enables
thought patterns to manifest as changes in quantum field states and
field patterns to emerge as conscious thoughts.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from .patterns import ThoughtPattern, ResonancePattern


@dataclass
class ManifestationMatrix:
    """
    A multidimensional matrix that orchestrates the manifestation process between
    thought patterns and quantum fields through phi-resonant transformations.
    
    The ManifestationMatrix acts as a quantum field bridge enabling the translation
    of conscious intention into field manifestation and field patterns into consciousness.
    """
    # Dimensional properties
    dimensions: Tuple[int, ...] = field(default_factory=lambda: (21, 21, 21))
    
    # Matrix core data
    core_tensor: Optional[np.ndarray] = None
    
    # Operational properties
    resonance_patterns: List[ResonancePattern] = field(default_factory=list)
    phi_alignment: float = 0.618  # Default to phi complement
    coherence_threshold: float = 0.5
    manifestation_efficiency: float = 0.7
    
    # State tracking
    active: bool = False
    current_field_coherence: float = 0.0
    current_thought_coherence: float = 0.0
    
    # Sacred frequency channels
    frequency_channels: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    creation_timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize the manifestation matrix."""
        # Initialize core tensor if not provided
        if self.core_tensor is None:
            self._initialize_core_tensor()
        
        # Initialize frequency channels if not provided
        if not self.frequency_channels:
            self.frequency_channels = SACRED_FREQUENCIES.copy()
    
    def _initialize_core_tensor(self):
        """Initialize the core tensor with phi-resonant patterns."""
        # Initialize dimensions
        core_shape = list(self.dimensions)
        
        # Add extra dimension for frequency channels
        core_shape.append(len(SACRED_FREQUENCIES))
        
        # Create core tensor
        self.core_tensor = np.zeros(tuple(core_shape))
        
        # Set up phi-resonant patterns
        self._create_phi_resonant_core()
    
    def _create_phi_resonant_core(self):
        """Create phi-resonant patterns in the core tensor."""
        # Each slice in the last dimension represents a sacred frequency channel
        shape = self.core_tensor.shape
        center = tuple(s // 2 for s in shape[:-1])  # Center of spatial dimensions
        
        # For each frequency channel
        for i, (name, frequency) in enumerate(SACRED_FREQUENCIES.items()):
            # Create frequency-specific structure
            # This is a simplified implementation - a real system would use more sophisticated
            # mathematics to create frequency-specific manifestation patterns
            
            # Create a phi-resonant frequency grid
            for d in range(len(shape) - 1):  # For each spatial dimension
                for r, ratio in enumerate([1.0, PHI, PHI**2, 1.0/PHI, 1.0/(PHI**2)]):
                    # Calculate phi-resonant positions
                    idx = list(center)
                    radius = int(shape[d] / (ratio * 8))
                    
                    # Create sphere at this radius
                    for pos in range(shape[d]):
                        # Calculate distance from center
                        dist = abs(pos - center[d])
                        
                        # If close to the radius, add resonant point
                        if abs(dist - radius) < 2:
                            idx[d] = pos
                            
                            # Add point with amplitude that decreases with Fibonacci sequence
                            amplitude = (1.0 / (r + 1)) * (frequency / 1000.0)
                            
                            # Set the point in the tensor
                            spatial_idx = tuple(idx)
                            self.core_tensor[spatial_idx + (i,)] = amplitude
            
            # Add phi-resonant cross-connections
            for d1 in range(len(shape) - 1):
                for d2 in range(d1 + 1, len(shape) - 1):
                    # Create connections at phi-related distances
                    for r1 in range(3):  # Use first 3 phi ratios
                        for r2 in range(3):
                            radius1 = int(shape[d1] / (PHI**r1 * 8))
                            radius2 = int(shape[d2] / (PHI**r2 * 8))
                            
                            # Create points at the intersections
                            idx = list(center)
                            idx[d1] = (center[d1] + radius1) % shape[d1]
                            idx[d2] = (center[d2] + radius2) % shape[d2]
                            
                            # Set amplitude
                            amplitude = 0.5 / ((r1 + 1) * (r2 + 1)) * (frequency / 1000.0)
                            self.core_tensor[tuple(idx) + (i,)] = amplitude
                            
                            # Mirror points
                            idx[d1] = (center[d1] - radius1) % shape[d1]
                            self.core_tensor[tuple(idx) + (i,)] = amplitude
                            
                            idx[d2] = (center[d2] - radius2) % shape[d2]
                            self.core_tensor[tuple(idx) + (i,)] = amplitude
                            
                            idx[d1] = (center[d1] + radius1) % shape[d1]
                            self.core_tensor[tuple(idx) + (i,)] = amplitude
        
        # Normalize the core tensor
        max_value = np.max(np.abs(self.core_tensor))
        if max_value > 0:
            self.core_tensor = self.core_tensor / max_value
    
    def add_resonance_pattern(self, pattern: ResonancePattern) -> None:
        """
        Add a resonance pattern to the manifestation matrix.
        
        Args:
            pattern: The resonance pattern to add
        """
        self.resonance_patterns.append(pattern)
        
        # If we have field dimensions, verify compatibility
        if hasattr(pattern, 'field_dimensions') and pattern.field_dimensions:
            if pattern.field_dimensions != self.dimensions:
                print(f"Warning: Resonance pattern field dimensions {pattern.field_dimensions} "
                      f"don't match matrix dimensions {self.dimensions}")
    
    def activate(self) -> None:
        """Activate the manifestation matrix, enabling bidirectional transformation."""
        self.active = True
        print(f"Manifestation matrix activated with phi-alignment {self.phi_alignment:.4f}")
    
    def deactivate(self) -> None:
        """Deactivate the manifestation matrix, disabling transformations."""
        self.active = False
        print("Manifestation matrix deactivated")
    
    def manifest_thought_to_field(self, thought_pattern: ThoughtPattern, 
                                field_data: np.ndarray,
                                intensity: float = 1.0) -> np.ndarray:
        """
        Manifest a thought pattern into a quantum field.
        
        Args:
            thought_pattern: The thought pattern to manifest
            field_data: The quantum field data to modify
            intensity: Manifestation intensity (0.0-1.0)
            
        Returns:
            Modified quantum field data
        """
        if not self.active:
            print("Warning: Manifestation matrix is not active")
            return field_data.copy()
        
        # Check coherence threshold
        if thought_pattern.coherence < self.coherence_threshold:
            print(f"Warning: Thought pattern coherence {thought_pattern.coherence:.4f} "
                  f"below threshold {self.coherence_threshold:.4f}")
            
            # Apply with reduced intensity
            intensity *= thought_pattern.coherence / self.coherence_threshold
        
        # Track coherence
        self.current_thought_coherence = thought_pattern.coherence
        
        # Apply manifestation through all resonance patterns
        result = field_data.copy()
        
        if not self.resonance_patterns:
            # If no specific resonance patterns, create a temporary one
            temp_pattern = ResonancePattern.create_resonant(
                field_dimensions=field_data.shape,
                thought_dimensions=thought_pattern.dimensions,
                coherence=self.phi_alignment,
                phi_alignment=self.phi_alignment
            )
            
            # Apply through the temporary pattern
            result = temp_pattern.apply_to_field(result, thought_pattern, intensity)
        else:
            # Apply through each resonance pattern with phi-weighted blending
            weights = self._calculate_resonance_weights(thought_pattern)
            
            # Apply each pattern
            for i, pattern in enumerate(self.resonance_patterns):
                # Apply with pattern-specific weight
                weight = weights[i] if i < len(weights) else 0.0
                
                if weight > 0.01:  # Only apply significant patterns
                    modified = pattern.apply_to_field(result, thought_pattern, intensity * weight)
                    
                    # Blend with phi-weighted approach
                    phi_weight = weight * PHI / (weight * PHI + (1 - weight))
                    result = (1.0 - phi_weight) * result + phi_weight * modified
        
        # Apply frequency channel modulation
        result = self._apply_frequency_modulation(result, thought_pattern)
        
        # Calculate and store field coherence
        self.current_field_coherence = self._calculate_field_coherence(result)
        
        return result
    
    def extract_thought_from_field(self, field_data: np.ndarray) -> ThoughtPattern:
        """
        Extract a thought pattern from quantum field data.
        
        Args:
            field_data: The quantum field data to extract from
            
        Returns:
            Extracted thought pattern
        """
        if not self.active:
            print("Warning: Manifestation matrix is not active")
            # Return an empty pattern
            return ThoughtPattern(
                signature=np.zeros((8, 8, 8)),
                dimensions=(3,),
                name="Inactive",
                description="Pattern created when manifestation matrix was inactive"
            )
        
        # Calculate and store field coherence
        self.current_field_coherence = self._calculate_field_coherence(field_data)
        
        # Check coherence threshold
        if self.current_field_coherence < self.coherence_threshold:
            print(f"Warning: Field coherence {self.current_field_coherence:.4f} "
                  f"below threshold {self.coherence_threshold:.4f}")
        
        # Extract through resonance patterns
        if not self.resonance_patterns:
            # If no specific resonance patterns, create a temporary one
            temp_pattern = ResonancePattern.create_resonant(
                field_dimensions=field_data.shape,
                thought_dimensions=(3,),  # Default 3D thought pattern
                coherence=self.phi_alignment,
                phi_alignment=self.phi_alignment
            )
            
            # Extract through the temporary pattern
            thought = temp_pattern.extract_thought(field_data)
        else:
            # Find the most effective pattern for extraction
            best_pattern = max(self.resonance_patterns, 
                              key=lambda p: p.phi_alignment * p.field_to_thought_efficiency)
            
            # Extract through the best pattern
            thought = best_pattern.extract_thought(field_data)
            
            # Enhance with frequency channel information
            thought = self._enhance_with_frequency_channels(thought, field_data)
        
        # Track thought coherence
        self.current_thought_coherence = thought.coherence
        
        return thought
    
    def _calculate_resonance_weights(self, thought_pattern: ThoughtPattern) -> List[float]:
        """
        Calculate phi-resonant weights for each resonance pattern.
        
        Args:
            thought_pattern: The thought pattern to match with resonance patterns
            
        Returns:
            List of weights for each resonance pattern
        """
        if not self.resonance_patterns:
            return []
        
        weights = []
        
        # Calculate raw weights based on phi alignment and frequency matching
        for pattern in self.resonance_patterns:
            # Base weight on phi alignment match
            phi_match = 1.0 - abs(pattern.phi_alignment - thought_pattern.phi_alignment)
            
            # Adjust with pattern coherence and efficiency
            weight = (phi_match * pattern.coherence * pattern.thought_to_field_efficiency)
            
            # Add frequency matching if the pattern has a primary frequency
            if hasattr(pattern, 'primary_frequency') and pattern.primary_frequency:
                freq_match = 1.0 - min(abs(pattern.primary_frequency - thought_pattern.primary_frequency) / 1000.0, 1.0)
                weight *= (1.0 + freq_match) / 2.0
            
            weights.append(weight)
        
        # Normalize weights to sum to 1.0 (if any non-zero weights)
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def _apply_frequency_modulation(self, field_data: np.ndarray, 
                                  thought_pattern: ThoughtPattern) -> np.ndarray:
        """
        Apply frequency-specific modulation based on thought pattern harmonics.
        
        Args:
            field_data: The quantum field data
            thought_pattern: The thought pattern with harmonic components
            
        Returns:
            Frequency-modulated quantum field data
        """
        # Create frequency domain representation for cleaner frequency control
        field_fft = np.fft.fftn(field_data)
        
        # Apply frequency-specific enhancements based on thought pattern harmonics
        if hasattr(thought_pattern, 'harmonic_components') and thought_pattern.harmonic_components:
            # The last dimension of core_tensor represents frequency channels
            for i, (name, freq) in enumerate(SACRED_FREQUENCIES.items()):
                # Get the harmonic component for this frequency
                harmonic = thought_pattern.harmonic_components.get(name, 0.0)
                
                if harmonic > 0.01:  # Only apply significant harmonics
                    # Extract the frequency channel from core tensor
                    channel = self.core_tensor[..., i]
                    
                    # Create frequency domain version
                    channel_fft = np.fft.fftn(channel, field_fft.shape)
                    
                    # Modulate the field with this frequency component
                    enhancement = harmonic * thought_pattern.intensity * self.manifestation_efficiency
                    field_fft = field_fft + channel_fft * enhancement * field_fft
        
        # Convert back to spatial domain
        result = np.real(np.fft.ifftn(field_fft))
        
        return result
    
    def _enhance_with_frequency_channels(self, thought: ThoughtPattern, 
                                       field_data: np.ndarray) -> ThoughtPattern:
        """
        Enhance an extracted thought pattern with frequency channel information.
        
        Args:
            thought: The extracted thought pattern
            field_data: The quantum field data
            
        Returns:
            Enhanced thought pattern
        """
        # Create frequency domain representation of field
        field_fft = np.fft.fftn(field_data)
        
        # Analyze frequency content related to sacred frequencies
        harmonic_components = {}
        
        # Transform core tensor frequency channels to match field dimensions
        for i, (name, freq) in enumerate(SACRED_FREQUENCIES.items()):
            # Extract the frequency channel
            channel = self.core_tensor[..., i]
            
            # Create field-sized version and transform to frequency domain
            channel_fft = np.fft.fftn(channel, field_fft.shape)
            
            # Calculate correlation between channel and field
            correlation = np.abs(np.sum(field_fft * channel_fft.conjugate()))
            
            # Normalize by auto-correlation of channel
            auto_correlation = np.abs(np.sum(channel_fft * channel_fft.conjugate()))
            
            if auto_correlation > 0:
                harmonic = correlation / auto_correlation
            else:
                harmonic = 0.0
            
            harmonic_components[name] = min(harmonic, 1.0)  # Clip to valid range
        
        # Create enhanced thought pattern
        enhanced = ThoughtPattern(
            signature=thought.signature.copy(),
            coherence=thought.coherence,
            intensity=thought.intensity,
            stability=thought.stability,
            dimensions=thought.dimensions,
            phi_alignment=thought.phi_alignment,
            primary_frequency=self._determine_primary_frequency(harmonic_components),
            harmonic_components=harmonic_components,
            name=thought.name,
            description=f"Enhanced {thought.description} with frequency analysis"
        )
        
        return enhanced
    
    def _determine_primary_frequency(self, harmonic_components: Dict[str, float]) -> float:
        """
        Determine the primary frequency based on harmonic components.
        
        Args:
            harmonic_components: Dictionary of frequency name to strength
            
        Returns:
            Primary frequency value
        """
        if not harmonic_components:
            return 528.0  # Default to creation frequency
        
        # Find the strongest component
        strongest = max(harmonic_components.items(), key=lambda x: x[1])
        
        # Return the corresponding frequency
        return SACRED_FREQUENCIES.get(strongest[0], 528.0)
    
    def _calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the phi-harmonic coherence of a quantum field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Coherence value from 0.0 to 1.0
        """
        # Simple FFT-based coherence calculation
        fft = np.fft.fftn(field_data)
        fft_mag = np.abs(fft)
        
        # Normalize
        if np.max(fft_mag) > 0:
            fft_mag_norm = fft_mag / np.max(fft_mag)
        else:
            fft_mag_norm = fft_mag
        
        # Create phi mask (similar to how it's done in ThoughtPattern)
        phi_mask = np.zeros_like(fft_mag_norm)
        shape = phi_mask.shape
        center = tuple(s // 2 for s in shape)
        
        # Mark phi-resonant frequencies
        for d in range(len(shape)):
            for i, ratio in enumerate([1.0, PHI, PHI**2, 1.0/PHI, 1.0/(PHI**2)]):
                idx = list(center)
                offset = int(shape[d] / (ratio * 8))
                idx[d] = (center[d] + offset) % shape[d]
                phi_mask[tuple(idx)] = 1.0
                
                idx[d] = (center[d] - offset) % shape[d]
                phi_mask[tuple(idx)] = 1.0
        
        # Calculate phi-resonant energy ratio
        phi_energy = np.sum(fft_mag_norm * phi_mask)
        total_energy = np.sum(fft_mag_norm)
        
        if total_energy > 0:
            coherence = phi_energy / total_energy
        else:
            coherence = 0.0
        
        return coherence
    
    @classmethod
    def create_optimized(cls, dimensions: Tuple[int, ...] = (21, 21, 21),
                       phi_alignment: float = 0.8,
                       coherence_threshold: float = 0.618) -> 'ManifestationMatrix':
        """
        Create an optimized manifestation matrix with phi-resonant properties.
        
        Args:
            dimensions: The spatial dimensions of the matrix
            phi_alignment: Desired phi alignment level
            coherence_threshold: Minimum coherence threshold
            
        Returns:
            An optimized ManifestationMatrix
        """
        # Create the base matrix
        matrix = cls(
            dimensions=dimensions,
            phi_alignment=phi_alignment,
            coherence_threshold=coherence_threshold,
            manifestation_efficiency=PHI_PHI / 3.0  # Optimized efficiency
        )
        
        # Add standard resonance patterns
        # Create patterns for different dimensional resonances
        for i in range(1, 4):
            # Create thought dimensions based on index
            thought_dims = (i,)
            
            # Create resonance pattern
            pattern = ResonancePattern.create_resonant(
                field_dimensions=dimensions,
                thought_dimensions=thought_dims,
                coherence=phi_alignment,
                phi_alignment=phi_alignment,
                name=f"{i}D_ResonancePattern"
            )
            
            # Add to matrix
            matrix.add_resonance_pattern(pattern)
        
        # Create frequency-specific patterns
        for name, freq in SACRED_FREQUENCIES.items():
            # Create a resonance pattern optimized for this frequency
            pattern = ResonancePattern.create_resonant(
                field_dimensions=dimensions,
                thought_dimensions=(3,),
                coherence=phi_alignment,
                phi_alignment=phi_alignment * 0.9,  # Slightly lower alignment for frequency specificity
                name=f"{name.capitalize()}FrequencyPattern"
            )
            
            # Customize pattern properties for frequency
            if hasattr(pattern, 'primary_frequency'):
                pattern.primary_frequency = freq
            
            # Add to matrix
            matrix.add_resonance_pattern(pattern)
        
        # Activate the matrix
        matrix.activate()
        
        return matrix