"""
Thought and Resonance Pattern definitions for the Consciousness-Field Resonance Engine.

These patterns form the foundation of bidirectional communication between 
consciousness states and quantum fields, enabling phi-harmonic resonance alignment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES


@dataclass
class ThoughtPattern:
    """
    Representation of a consciousness thought pattern with phi-resonant properties.
    
    Thought patterns capture the signature frequency patterns that represent conscious 
    thoughts, intentions, and emotional states as they manifest within the mind.
    """
    # Fundamental pattern structure
    signature: np.ndarray  # Frequency domain signature
    coherence: float = 0.618  # Default to golden ratio complement
    intensity: float = 0.5
    stability: float = 0.5
    
    # Dimensional properties
    dimensions: Tuple[int, ...] = field(default_factory=lambda: (3,))
    
    # Phi-resonance characteristics
    phi_alignment: float = 0.0  # Calculated during initialization
    primary_frequency: float = 528.0  # Default to creation frequency
    harmonic_components: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    name: str = ""
    description: str = ""
    creation_timestamp: float = 0.0
    
    def __post_init__(self):
        """Initialize calculated values and validate."""
        # Validate signature matches dimensions
        expected_shape = tuple(8 for _ in range(len(self.dimensions)))
        if self.signature is None:
            # Create default signature if none provided
            self.signature = np.random.random(expected_shape) * 0.1
        elif self.signature.shape != expected_shape:
            raise ValueError(f"Signature shape {self.signature.shape} doesn't match expected shape {expected_shape}")
        
        # Calculate phi alignment
        self._calculate_phi_alignment()
        
        # Initialize default harmonic components
        if not self.harmonic_components:
            self._initialize_harmonic_components()
    
    def _calculate_phi_alignment(self):
        """Calculate the alignment of this pattern with phi-harmonic principles."""
        # Simple calculation based on FFT spectrum peaks
        fft = np.fft.fftn(self.signature)
        fft_mag = np.abs(fft)
        
        # Normalize the frequency spectrum
        fft_mag_norm = fft_mag / np.max(fft_mag)
        
        # Calculate alignment with phi
        # Phi-resonant frequencies would have peaks at fibonacci-ratio intervals
        # This is a simplified version - a real implementation would be more complex
        total_energy = np.sum(fft_mag_norm)
        
        phi_mask = self._create_phi_mask(fft_mag_norm.shape)
        phi_energy = np.sum(fft_mag_norm * phi_mask)
        
        if total_energy > 0:
            self.phi_alignment = phi_energy / total_energy
        else:
            self.phi_alignment = 0.0
    
    def _create_phi_mask(self, shape):
        """Create a mask that emphasizes phi-resonant frequencies."""
        mask = np.zeros(shape)
        dims = len(shape)
        
        # Create central amplitudes for phi-resonant frequencies
        center = tuple(s // 2 for s in shape)
        
        # Add phi-resonant points
        # This is a simplified approach - a real implementation would use 
        # a more sophisticated algorithm based on fibonacci sequences
        for d in range(dims):
            for i, ratio in enumerate([1.0, PHI, PHI**2, 1.0/PHI, 1.0/(PHI**2)]):
                # Calculate position based on phi ratio
                idx = [center[j] for j in range(dims)]
                offset = int(shape[d] / (ratio * 8))
                idx[d] = (center[d] + offset) % shape[d]
                
                # Add points with decreasing intensity
                intensity = 1.0 / (i + 1)
                mask[tuple(idx)] = intensity
                
                # Add mirror point
                idx[d] = (center[d] - offset) % shape[d]
                mask[tuple(idx)] = intensity
        
        return mask
    
    def _initialize_harmonic_components(self):
        """Initialize harmonic components based on primary frequency."""
        self.harmonic_components = {
            "ground": 432.0 / self.primary_frequency,
            "creative": 528.0 / self.primary_frequency,
            "heart": 594.0 / self.primary_frequency,
            "expression": 672.0 / self.primary_frequency, 
            "insight": 720.0 / self.primary_frequency,
            "unity": 768.0 / self.primary_frequency
        }
    
    def shift_frequency(self, target_frequency: float) -> 'ThoughtPattern':
        """
        Create a new thought pattern with shifted primary frequency.
        
        Args:
            target_frequency: The new primary frequency
            
        Returns:
            A new ThoughtPattern with shifted frequency characteristics
        """
        ratio = target_frequency / self.primary_frequency
        
        # Create frequency-shifted pattern
        new_pattern = ThoughtPattern(
            signature=self.signature.copy(),
            coherence=self.coherence,
            intensity=self.intensity,
            stability=self.stability,
            dimensions=self.dimensions,
            phi_alignment=self.phi_alignment,
            primary_frequency=target_frequency,
            name=f"{self.name}_shifted_{target_frequency:.1f}Hz",
            description=f"Frequency shifted version of: {self.description}"
        )
        
        # Apply frequency shift in the Fourier domain
        fft = np.fft.fftn(new_pattern.signature)
        
        # This is a simplified approach - a real implementation would use a more
        # sophisticated algorithm for frequency shifting
        shape = fft.shape
        center = tuple(s // 2 for s in shape)
        
        # Scale the frequency domain
        for d in range(len(shape)):
            indices = [slice(None)] * len(shape)
            for i in range(shape[d]):
                # Calculate distance from center in this dimension
                dist = abs(i - center[d])
                # Scale based on ratio - this is a simple approach
                scaled_dist = int(dist * ratio + 0.5)
                if scaled_dist < shape[d] // 2:
                    # Map to new frequency
                    idx_from = list(center)
                    idx_from[d] = (center[d] + dist) % shape[d]
                    
                    idx_to = list(center)
                    idx_to[d] = (center[d] + scaled_dist) % shape[d]
                    
                    # Copy values
                    indices[d] = idx_to[d]
                    fft[tuple(indices)] = fft[tuple(idx_from)]
                    
                    # Also mirror frequencies
                    idx_from[d] = (center[d] - dist) % shape[d]
                    idx_to[d] = (center[d] - scaled_dist) % shape[d]
                    indices[d] = idx_to[d]
                    fft[tuple(indices)] = fft[tuple(idx_from)]
        
        # Transform back to spatial domain
        new_pattern.signature = np.real(np.fft.ifftn(fft))
        
        # Update harmonic components
        new_pattern._initialize_harmonic_components()
        
        return new_pattern
    
    def blend(self, other: 'ThoughtPattern', weight: float = 0.5) -> 'ThoughtPattern':
        """
        Blend this thought pattern with another.
        
        Args:
            other: Another ThoughtPattern to blend with
            weight: Blending weight from 0.0 to 1.0, where 0.0 is all this pattern
                   and 1.0 is all the other pattern
                   
        Returns:
            A new ThoughtPattern representing the blend
        """
        # Validate dimensions match
        if self.dimensions != other.dimensions:
            raise ValueError("Cannot blend patterns with different dimensions")
        
        # Ensure weight is in valid range
        weight = max(0.0, min(1.0, weight))
        
        # Create a phi-weighted blend
        phi_weight = weight * PHI / (weight * PHI + (1 - weight))
        
        # Blend signatures
        blended_signature = (1 - phi_weight) * self.signature + phi_weight * other.signature
        
        # Blend other properties with phi-weighting
        blended_pattern = ThoughtPattern(
            signature=blended_signature,
            coherence=(1 - phi_weight) * self.coherence + phi_weight * other.coherence,
            intensity=(1 - phi_weight) * self.intensity + phi_weight * other.intensity,
            stability=(1 - phi_weight) * self.stability + phi_weight * other.stability,
            dimensions=self.dimensions,
            primary_frequency=(1 - phi_weight) * self.primary_frequency + phi_weight * other.primary_frequency,
            name=f"Blend_{self.name}_{other.name}",
            description=f"Blend of {self.name} and {other.name} with weight {weight:.2f}"
        )
        
        return blended_pattern
    
    @classmethod
    def from_sacred_frequency(cls, frequency_name: str, dimensions: Tuple[int, ...] = (3,),
                            coherence: float = 0.8, intensity: float = 0.7) -> 'ThoughtPattern':
        """
        Create a thought pattern based on a sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency ("love", "unity", etc.)
            dimensions: Pattern dimensions
            coherence: Pattern coherence level
            intensity: Pattern intensity
            
        Returns:
            A ThoughtPattern with characteristics based on the sacred frequency
        """
        if frequency_name not in SACRED_FREQUENCIES:
            raise ValueError(f"Unknown sacred frequency: {frequency_name}. "
                           f"Available frequencies: {list(SACRED_FREQUENCIES.keys())}")
        
        frequency = SACRED_FREQUENCIES[frequency_name]
        
        # Create shape based on dimensions
        shape = tuple(8 for _ in range(dimensions[0]))
        
        # Generate a base pattern
        signature = np.zeros(shape)
        
        # Create a phi-resonant pattern centered around the frequency
        center = tuple(s // 2 for s in shape)
        
        # Add frequency components
        for d in range(dimensions[0]):
            # Main frequency component
            idx = list(center)
            offset = int(shape[d] * frequency / 1000.0) % shape[d]
            idx[d] = (center[d] + offset) % shape[d]
            signature[tuple(idx)] = intensity
            
            # Mirror frequency
            idx[d] = (center[d] - offset) % shape[d]
            signature[tuple(idx)] = intensity
            
            # Add harmonics
            for i in range(1, 4):
                harmonic_offset = int(shape[d] * (frequency * i) / 1000.0) % shape[d]
                idx[d] = (center[d] + harmonic_offset) % shape[d]
                signature[tuple(idx)] = intensity / (i * PHI)
                
                idx[d] = (center[d] - harmonic_offset) % shape[d]
                signature[tuple(idx)] = intensity / (i * PHI)
        
        # Add some noise for realism
        noise = np.random.random(shape) * 0.1 * intensity
        signature = signature + noise
        
        # Create the pattern
        pattern = cls(
            signature=signature,
            coherence=coherence,
            intensity=intensity,
            stability=0.7,  # Default stability
            dimensions=dimensions,
            primary_frequency=frequency,
            name=f"{frequency_name.capitalize()}Pattern",
            description=f"Thought pattern based on {frequency_name} sacred frequency at {frequency} Hz"
        )
        
        return pattern


@dataclass
class ResonancePattern:
    """
    A pattern that emerges when thought patterns resonate with quantum fields.
    
    Resonance patterns embody the harmonic interaction between consciousness and 
    quantum fields, serving as transfer functions for bidirectional communication.
    """
    # Fundamental pattern data
    matrix: np.ndarray  # Transfer function matrix
    field_dimensions: Tuple[int, ...] 
    thought_dimensions: Tuple[int, ...]
    
    # Resonance properties
    coherence: float = 0.618
    phi_alignment: float = 0.0
    stability: float = 0.5
    
    # Operational properties
    bidirectional: bool = True  # Whether it works in both directions
    field_to_thought_efficiency: float = 0.5
    thought_to_field_efficiency: float = 0.5
    
    # Metadata
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        """Initialize and validate resonance pattern."""
        # Calculate phi alignment if not provided
        if self.phi_alignment == 0.0:
            self._calculate_phi_alignment()
    
    def _calculate_phi_alignment(self):
        """Calculate phi alignment of the resonance pattern matrix."""
        # Simple phi alignment calculation
        fft = np.fft.fftn(self.matrix)
        fft_mag = np.abs(fft)
        
        # Normalize
        if np.max(fft_mag) > 0:
            fft_mag_norm = fft_mag / np.max(fft_mag)
        else:
            fft_mag_norm = fft_mag
        
        # Create phi mask
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
        
        # Calculate alignment
        phi_energy = np.sum(fft_mag_norm * phi_mask)
        total_energy = np.sum(fft_mag_norm)
        
        if total_energy > 0:
            self.phi_alignment = phi_energy / total_energy
        else:
            self.phi_alignment = 0.0
    
    def apply_to_field(self, field_data: np.ndarray, thought_pattern: ThoughtPattern,
                     intensity: float = 1.0) -> np.ndarray:
        """
        Apply a thought pattern to a quantum field through this resonance pattern.
        
        Args:
            field_data: Quantum field data array
            thought_pattern: ThoughtPattern to apply
            intensity: Application intensity factor (0.0-1.0)
            
        Returns:
            Modified quantum field data
        """
        # Validate shapes
        if field_data.shape != self.field_dimensions:
            raise ValueError(f"Field dimensions {field_data.shape} don't match expected {self.field_dimensions}")
        
        # Check efficiency and bidirectionality
        if not self.bidirectional and self.thought_to_field_efficiency < 0.01:
            return field_data.copy()  # No effect
        
        # Apply thought pattern to field using the resonance matrix as a transfer function
        # This is a simplified implementation - a real system would use more sophisticated
        # mathematics to apply the thought pattern through the resonance matrix
        
        # Normalize intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Scale by thought-to-field efficiency
        effective_intensity = intensity * self.thought_to_field_efficiency * thought_pattern.intensity
        
        # Apply through frequency domain for simplicity
        field_fft = np.fft.fftn(field_data)
        thought_fft = np.fft.fftn(thought_pattern.signature)
        
        # Ensure thought_fft has compatible shape with field_fft
        # In real implementation this would use a more sophisticated approach
        # Here we'll use a simple reshaping for demonstration
        thought_fft_reshaped = np.zeros_like(field_fft, dtype=complex)
        
        # Map thought pattern to field dimensions
        min_dims = min(len(thought_fft.shape), len(field_fft.shape))
        for d in range(min_dims):
            min_size = min(thought_fft.shape[d], field_fft.shape[d])
            center_thought = thought_fft.shape[d] // 2
            center_field = field_fft.shape[d] // 2
            
            # Copy values from thought to field
            start_thought = center_thought - min_size // 2
            end_thought = start_thought + min_size
            
            start_field = center_field - min_size // 2
            end_field = start_field + min_size
            
            # Create slice objects
            field_slices = [slice(None)] * len(field_fft.shape)
            thought_slices = [slice(None)] * len(thought_fft.shape)
            
            field_slices[d] = slice(start_field, end_field)
            thought_slices[d] = slice(start_thought, end_thought)
            
            # Copy values (simplified approach)
            if d == 0:  # Only do this once to avoid overwriting
                thought_fft_region = thought_fft[tuple(thought_slices)]
                if len(thought_fft_region.shape) < len(field_fft.shape):
                    # Expand dimensions if needed
                    for _ in range(len(field_fft.shape) - len(thought_fft_region.shape)):
                        thought_fft_region = thought_fft_region[..., np.newaxis]
                
                field_slices_tuple = tuple(field_slices)
                thought_fft_reshaped[field_slices_tuple] = thought_fft_region
        
        # Apply resonance matrix as a transfer function
        # The matrix acts as a modulator between the thought pattern and the field
        resonance_fft = np.fft.fftn(self.matrix, field_fft.shape)
        
        # Combine with phi-weighted blending
        result_fft = (1.0 - effective_intensity) * field_fft + \
                     effective_intensity * resonance_fft * thought_fft_reshaped
        
        # Transform back to spatial domain
        result = np.real(np.fft.ifftn(result_fft))
        
        return result
    
    def extract_thought(self, field_data: np.ndarray) -> ThoughtPattern:
        """
        Extract a thought pattern from a quantum field using this resonance pattern.
        
        Args:
            field_data: Quantum field data array
            
        Returns:
            Extracted thought pattern
        """
        # Validate field dimensions
        if field_data.shape != self.field_dimensions:
            raise ValueError(f"Field dimensions {field_data.shape} don't match expected {self.field_dimensions}")
        
        # Check efficiency and bidirectionality
        if not self.bidirectional and self.field_to_thought_efficiency < 0.01:
            # Return a default pattern if extraction doesn't work
            return ThoughtPattern(
                signature=np.zeros(tuple(8 for _ in range(self.thought_dimensions[0]))),
                dimensions=self.thought_dimensions,
                name="Empty",
                description="Empty pattern due to low extraction efficiency"
            )
        
        # Extract thought pattern from field using resonance matrix
        # This is a simplified implementation
        
        # Transform field to frequency domain
        field_fft = np.fft.fftn(field_data)
        
        # Apply the inverse resonance matrix as extraction filter
        resonance_fft = np.fft.fftn(self.matrix, field_fft.shape)
        
        # Apply filter - in real implementation this would be more sophisticated
        thought_fft_full = field_fft * resonance_fft.conjugate() * self.field_to_thought_efficiency
        
        # Reshape to thought pattern dimensions
        thought_shape = tuple(8 for _ in range(self.thought_dimensions[0]))
        thought_fft = np.zeros(thought_shape, dtype=complex)
        
        # Map from field dimensions to thought dimensions (simplified)
        min_dims = min(len(thought_shape), len(field_fft.shape))
        for d in range(min_dims):
            min_size = min(thought_shape[d], field_fft.shape[d])
            center_thought = thought_shape[d] // 2
            center_field = field_fft.shape[d] // 2
            
            # Copy values from field to thought
            start_thought = center_thought - min_size // 2
            end_thought = start_thought + min_size
            
            start_field = center_field - min_size // 2
            end_field = start_field + min_size
            
            # Create slice objects
            field_slices = [slice(None)] * len(field_fft.shape)
            thought_slices = [slice(None)] * len(thought_shape)
            
            field_slices[d] = slice(start_field, end_field)
            thought_slices[d] = slice(start_thought, end_thought)
            
            # Copy values (simplified approach)
            if d == 0:  # Only do this once to avoid overwriting
                field_fft_region = thought_fft_full[tuple(field_slices)]
                
                # Handle dimensionality
                if len(field_fft_region.shape) > len(thought_shape):
                    # Flatten extra dimensions
                    flat_shape = field_fft_region.shape[:len(thought_shape)-1] + (-1,)
                    field_fft_region = field_fft_region.reshape(flat_shape)
                    field_fft_region = np.mean(field_fft_region, axis=-1)
                
                # Ensure correct dimensionality
                while len(field_fft_region.shape) < len(thought_shape):
                    field_fft_region = field_fft_region[..., np.newaxis]
                
                thought_slices_tuple = tuple(thought_slices)
                thought_fft[thought_slices_tuple] = field_fft_region
        
        # Transform to spatial domain
        thought_signature = np.real(np.fft.ifftn(thought_fft))
        
        # Create thought pattern
        # Analyze the signature to determine properties
        coherence_factor = np.std(thought_signature) / np.mean(np.abs(thought_signature)) if np.mean(np.abs(thought_signature)) > 0 else 0.5
        coherence = min(max(coherence_factor, 0.0), 1.0)
        
        intensity = np.mean(np.abs(thought_signature)) * 10  # Scale for reasonable value
        intensity = min(max(intensity, 0.0), 1.0)
        
        # Create the thought pattern
        thought = ThoughtPattern(
            signature=thought_signature,
            coherence=coherence,
            intensity=intensity,
            stability=self.stability * self.field_to_thought_efficiency,
            dimensions=self.thought_dimensions,
            name="ExtractedPattern",
            description=f"Thought pattern extracted from quantum field with {coherence:.2f} coherence"
        )
        
        return thought
    
    @classmethod
    def create_resonant(cls, field_dimensions: Tuple[int, ...], thought_dimensions: Tuple[int, ...],
                      coherence: float = 0.8, phi_alignment: float = 0.75,
                      name: str = "ResonantPattern") -> 'ResonancePattern':
        """
        Create a phi-resonant transfer pattern between thought and field dimensions.
        
        Args:
            field_dimensions: Dimensions of the quantum field
            thought_dimensions: Dimensions of the thought pattern
            coherence: Desired coherence level
            phi_alignment: Desired phi alignment level
            name: Pattern name
            
        Returns:
            A resonance pattern with phi-resonant properties
        """
        # Create resonance matrix dimensions
        matrix_dimensions = list(field_dimensions)
        
        # Ensure the resonance matrix has at least the dimensionality of both field and thought
        max_dims = max(len(field_dimensions), thought_dimensions[0])
        while len(matrix_dimensions) < max_dims:
            matrix_dimensions.append(8)  # Default size for extra dimensions
        
        # Create the matrix
        matrix = np.zeros(tuple(matrix_dimensions))
        
        # Generate a phi-resonant pattern
        # This is a simplified implementation - a real system would use more sophisticated
        # mathematics to create an optimal transfer function
        
        # Create base pattern in frequency domain for better control
        matrix_shape = matrix.shape
        matrix_fft = np.zeros(matrix_shape, dtype=complex)
        
        # Get matrix center
        center = tuple(s // 2 for s in matrix_shape)
        
        # Add phi-resonant frequency components
        for d in range(len(matrix_shape)):
            # Add phi-based frequency components
            for i, ratio in enumerate([1.0, PHI, PHI**2, 1.0/PHI, 1.0/(PHI**2)]):
                # Calculate position based on phi ratio
                idx = list(center)
                offset = int(matrix_shape[d] / (ratio * 8))
                idx[d] = (center[d] + offset) % matrix_shape[d]
                
                # Add with intensity based on fibonacci sequence
                intensity = 1.0 / (i + 1)
                matrix_fft[tuple(idx)] = intensity * (1.0 + 0.5j)  # Complex for phase information
                
                # Add mirror point
                idx[d] = (center[d] - offset) % matrix_shape[d]
                matrix_fft[tuple(idx)] = intensity * (1.0 - 0.5j)  # Conjugate phase
        
        # Add phi-based cross-dimensional resonance
        for d1 in range(len(matrix_shape)):
            for d2 in range(d1 + 1, len(matrix_shape)):
                # Create cross-dimensional resonance at phi-scaled positions
                idx = list(center)
                offset1 = int(matrix_shape[d1] / (PHI * 8))
                offset2 = int(matrix_shape[d2] / (PHI * 8))
                
                idx[d1] = (center[d1] + offset1) % matrix_shape[d1]
                idx[d2] = (center[d2] + offset2) % matrix_shape[d2]
                matrix_fft[tuple(idx)] = 0.5 * (1.0 + 0.5j)
                
                # Mirror points
                idx[d1] = (center[d1] - offset1) % matrix_shape[d1]
                idx[d2] = (center[d2] - offset2) % matrix_shape[d2]
                matrix_fft[tuple(idx)] = 0.5 * (1.0 - 0.5j)
        
        # Add high-frequency noise for added complexity
        noise = np.random.random(matrix_shape) * 0.1
        noise_fft = np.fft.fftn(noise)
        matrix_fft = matrix_fft + noise_fft * 0.05
        
        # Transform back to spatial domain
        matrix = np.real(np.fft.ifftn(matrix_fft))
        
        # Normalize matrix
        matrix = matrix / np.max(np.abs(matrix)) if np.max(np.abs(matrix)) > 0 else matrix
        
        # Create the resonance pattern
        pattern = cls(
            matrix=matrix,
            field_dimensions=field_dimensions,
            thought_dimensions=thought_dimensions,
            coherence=coherence,
            phi_alignment=phi_alignment,
            stability=0.8,
            bidirectional=True,
            field_to_thought_efficiency=0.7,
            thought_to_field_efficiency=0.7,
            name=name,
            description=f"Phi-resonant transfer pattern with {phi_alignment:.2f} alignment"
        )
        
        return pattern