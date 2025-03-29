"""
Multi-dimensional Data Structures

Core data structures for representing and manipulating multi-dimensional 
quantum fields that extend beyond conventional 3D space.
"""

import numpy as np
from typing import Tuple, List, Union, Dict, Optional, Callable

# Import sacred constants
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sacred_constants import (
    PHI, PHI_SQUARED, PHI_CUBED, PHI_PHI, 
    LAMBDA, PHI_DIMENSIONS, RESONANCE_PATTERNS
)


class DimensionalData:
    """
    Core data structure for multi-dimensional data with phi-scaled dimensions.
    
    This class represents data in up to 7 dimensions (3 spatial + 4 perceptual),
    with dimensions scaled according to phi-harmonic principles.
    """
    
    def __init__(
        self, 
        dimensions: Tuple[int, ...],
        ndim: int = 3,
        phi_scaled: bool = True,
        dtype=np.float32
    ):
        """
        Initialize a new DimensionalData object.
        
        Args:
            dimensions: Base dimensions for the data structure
            ndim: Number of dimensions (3-7 supported)
            phi_scaled: Whether to apply phi-scaling to higher dimensions
            dtype: Data type for the underlying numpy array
        """
        self.base_dimensions = dimensions
        self.ndim = max(3, min(7, ndim))  # Clamp between 3 and 7
        self.phi_scaled = phi_scaled
        self.dtype = dtype
        
        # Calculate actual dimensions with phi-scaling if requested
        if phi_scaled and ndim > 3:
            # Start with the base dimensions for the first 3
            actual_dims = list(dimensions)
            
            # Apply phi-scaling to higher dimensions
            for i in range(3, ndim):
                # Determine scale factor based on dimension index
                if i == 3:  # 4th dimension (time)
                    scale = LAMBDA  # Inverse phi for temporal dimension
                elif i == 4:  # 5th dimension (consciousness)
                    scale = PHI
                elif i == 5:  # 6th dimension (intention)
                    scale = PHI_SQUARED
                else:  # 7th dimension (unified field)
                    scale = PHI_CUBED
                
                # Calculate size for this dimension (base size * scale)
                size = max(2, int(dimensions[0] * scale))
                actual_dims.append(size)
                
            self.dimensions = tuple(actual_dims)
        else:
            # Use same size for all dimensions if not phi-scaled
            self.dimensions = dimensions + (dimensions[0],) * (ndim - 3)
        
        # Initialize data array
        self.data = np.zeros(self.dimensions, dtype=dtype)
        
        # Track coherence and resonance information
        self.coherence = 0.0
        self.resonance = {}
    
    def transform_dimension(
        self, 
        dim_index: int, 
        transform_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """
        Apply a transformation along a specific dimension.
        
        Args:
            dim_index: Index of the dimension to transform
            transform_func: Function to apply along the dimension
        """
        if dim_index < 0 or dim_index >= self.ndim:
            raise ValueError(f"Dimension index {dim_index} out of bounds (0-{self.ndim-1})")
        
        # Create slices for all axes
        slices = [slice(None)] * self.ndim
        
        # Apply transformation along the specified dimension
        for i in range(self.dimensions[dim_index]):
            slices[dim_index] = i
            self.data[tuple(slices)] = transform_func(self.data[tuple(slices)])
    
    def project_to_3d(self) -> np.ndarray:
        """
        Project higher-dimensional data to 3D for visualization.
        
        Returns:
            3D numpy array with the projection
        """
        if self.ndim == 3:
            return self.data.copy()
        
        # For higher dimensions, perform phi-weighted averaging
        result = np.zeros(self.dimensions[:3], dtype=self.dtype)
        
        # Define dimension weights based on phi relationships
        dim_weights = {
            3: LAMBDA,         # Time dimension (inverse phi)
            4: 1.0,            # Consciousness dimension (neutral weight)
            5: PHI,            # Intention dimension (phi)
            6: PHI_SQUARED,    # Unified field dimension (phi²)
        }
        
        # Start with the 3D data
        total_weight = 1.0
        weighted_sum = result.copy()
        
        # Add contributions from higher dimensions
        for dim in range(3, self.ndim):
            dim_size = self.dimensions[dim]
            weight = dim_weights.get(dim, 1.0)
            total_weight += weight
            
            # Extract cross-sections from each higher dimension and add them
            for i in range(dim_size):
                # Create a slice tuple to extract this cross-section
                slices = [slice(None), slice(None), slice(None)] + [0] * (dim - 3)
                slices[dim] = i
                
                # Apply phi-resonant weighting based on position in dimension
                pos_weight = (PHI ** ((i / dim_size) - 0.5)) * weight
                
                # Add to weighted sum
                cross_section = self.data[tuple(slices)]
                weighted_sum += cross_section * pos_weight
                total_weight += pos_weight - weight  # Adjust for position weight
        
        # Normalize the result
        if total_weight > 0:
            result = weighted_sum / total_weight
        
        return result
    
    def calculate_coherence(self) -> float:
        """
        Calculate the phi-harmonic coherence of the data.
        
        Returns:
            Coherence value between 0 and 1
        """
        from sacred_constants import calculate_field_coherence
        
        # Calculate coherence for 3D projection
        projection = self.project_to_3d()
        coherence_3d = calculate_field_coherence(projection)
        
        # If only 3D, return that value
        if self.ndim == 3:
            self.coherence = coherence_3d
            return self.coherence
        
        # For higher dimensions, calculate cross-dimensional coherence
        cross_dim_coherence = []
        
        # Calculate coherence between dimensions
        for dim in range(3, self.ndim):
            # Sample points from this dimension
            dim_size = self.dimensions[dim]
            
            # Create sample indices
            phi_indices = [int(dim_size * i / PHI) % dim_size for i in range(1, 4)]
            
            # Calculate coherence between these samples
            samples = []
            for idx in phi_indices:
                slices = [slice(None), slice(None), slice(None)] + [0] * (self.ndim - 3)
                slices[dim] = idx
                samples.append(self.data[tuple(slices)])
            
            # Calculate coherence between these samples
            if len(samples) > 1:
                sample_coherence = np.mean([
                    calculate_field_coherence(np.abs(a - b))
                    for i, a in enumerate(samples)
                    for b in samples[i+1:]
                ])
                cross_dim_coherence.append(sample_coherence)
        
        # Combine 3D coherence with cross-dimensional coherence
        if cross_dim_coherence:
            # Weight by phi-squared to emphasize 3D coherence
            self.coherence = (coherence_3d * PHI_SQUARED + np.mean(cross_dim_coherence)) / (PHI_SQUARED + 1)
        else:
            self.coherence = coherence_3d
        
        return self.coherence
    
    def analyze_resonance(self, dimensions: int = None) -> Dict[str, float]:
        """
        Analyze phi-resonance patterns in the data.
        
        Args:
            dimensions: Number of dimensions to analyze (default: self.ndim)
            
        Returns:
            Dictionary with resonance metrics
        """
        from sacred_constants import phi_resonance_spectrum
        
        # Use self.ndim if dimensions not specified
        if dimensions is None:
            dimensions = self.ndim
        
        # Limit to available dimensions
        dimensions = min(dimensions, self.ndim)
        
        # For 3D, analyze the data directly
        if dimensions == 3:
            self.resonance = phi_resonance_spectrum(self.data, dimensions=3)
            return self.resonance
        
        # For higher dimensions, analyze the projection and each dimension
        projection = self.project_to_3d()
        self.resonance = phi_resonance_spectrum(projection, dimensions=3)
        
        # Analyze cross-sections from higher dimensions
        for dim in range(3, dimensions):
            dim_size = self.dimensions[dim]
            
            # Create sample indices at phi-scaled positions
            idx = min(dim_size - 1, int(dim_size / PHI))
            
            # Extract a cross-section at this index
            slices = [slice(None), slice(None), slice(None)] + [0] * (self.ndim - 3)
            slices[dim] = idx
            cross_section = self.data[tuple(slices)]
            
            # Analyze this cross-section
            dim_resonance = phi_resonance_spectrum(cross_section, dimensions=3)
            
            # Add to overall resonance with dimension name
            for key, value in dim_resonance.items():
                self.resonance[f"dim{dim}_{key}"] = value
        
        return self.resonance
    
    def apply_phi_mask(
        self, 
        scale: float = 1.0, 
        center: Optional[Tuple[float, ...]] = None
    ) -> None:
        """
        Apply a phi-harmonic mask to enhance coherence.
        
        Args:
            scale: Scale factor for the mask intensity
            center: Center point for the mask (default: center of data)
        """
        # Determine mask center
        if center is None:
            center = [d / 2 for d in self.dimensions]
        else:
            # Ensure center matches dimensions
            center = list(center)
            while len(center) < self.ndim:
                center.append(self.dimensions[len(center)] / 2)
            center = center[:self.ndim]
        
        # Create coordinate meshgrid
        coords = [np.arange(d) for d in self.dimensions]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Calculate phi-weighted distance from center
        distance_sq = np.zeros_like(self.data)
        for i, grid in enumerate(grids):
            # Apply phi-weighting based on dimension
            if i < 3:  # Spatial dimensions (equal weight)
                weight = 1.0
            elif i == 3:  # Time dimension
                weight = LAMBDA
            elif i == 4:  # Consciousness dimension
                weight = 1.0 / PHI_SQUARED
            elif i == 5:  # Intention dimension
                weight = 1.0 / PHI
            else:  # Unified field dimension
                weight = 1.0 / PHI_CUBED
            
            # Add weighted squared distance
            distance_sq += weight * ((grid - center[i]) / self.dimensions[i] * 2) ** 2
        
        # Calculate mask based on phi-harmonic function
        distance = np.sqrt(distance_sq)
        mask = np.exp(-distance * PHI * scale)
        
        # Apply the mask
        self.data *= mask


class HyperField(DimensionalData):
    """
    Enhanced multi-dimensional field with advanced phi-harmonic operations.
    
    This class extends DimensionalData with methods specifically designed
    for quantum field operations across multiple dimensions of perception.
    """
    
    def __init__(
        self,
        dimensions: Tuple[int, ...],
        ndim: int = 5,  # Default to 5D (3 spatial + time + consciousness)
        phi_scaled: bool = True,
        initialize: bool = False,
        dtype=np.float32
    ):
        """
        Initialize a new HyperField.
        
        Args:
            dimensions: Base dimensions for spatial components
            ndim: Number of dimensions (3-7 supported)
            phi_scaled: Whether to apply phi-scaling to higher dimensions
            initialize: Whether to initialize with coherent patterns
            dtype: Data type for the underlying numpy array
        """
        # Initialize parent class
        super().__init__(dimensions, ndim, phi_scaled, dtype)
        
        # Set up frequency bands for perceptual mapping
        self.frequency_bands = {
            "delta": (0.5, 4),    # Slow waves - deep state
            "theta": (4, 8),      # Light meditation/sleep
            "alpha": (8, 13),     # Relaxed awareness
            "beta": (13, 30),     # Active thinking
            "gamma": (30, 100),   # Higher mental activity
            "unity": (432, 433),  # Unity frequency band
            "love": (528, 529),   # Love/healing band
            "cascade": (594, 595),# Cascade frequency
            "truth": (672, 673),  # Truth expression
            "vision": (720, 721), # Vision clarity
            "oneness": (768, 769),# Unity consciousness
        }
        
        # Initialize with coherent patterns if requested
        if initialize:
            self.initialize_coherent_field()
    
    def initialize_coherent_field(self, 
                                 base_frequency: str = "unity",
                                 coherence_target: float = 0.618) -> None:
        """
        Initialize the field with phi-coherent patterns.
        
        Args:
            base_frequency: Base frequency to initialize with (default: "unity")
            coherence_target: Target coherence level (default: phi complement)
        """
        # Get the base frequency value
        if base_frequency in self.frequency_bands:
            freq_min, freq_max = self.frequency_bands[base_frequency]
            freq_value = (freq_min + freq_max) / 2
        else:
            # Default to unity frequency
            freq_value = 432.0
        
        # Start with 3D initialization (spatial dimensions)
        x, y, z = np.meshgrid(
            np.linspace(0, 1, self.dimensions[0]),
            np.linspace(0, 1, self.dimensions[1]),
            np.linspace(0, 1, self.dimensions[2]),
            indexing='ij'
        )
        
        # Calculate radial distance from center
        center = [0.5, 0.5, 0.5]
        r = np.sqrt(
            ((x - center[0]) * PHI) ** 2 + 
            ((y - center[1]) * 1.0) ** 2 + 
            ((z - center[2]) * LAMBDA) ** 2
        )
        
        # Create base 3D pattern with frequency oscillations
        pattern = (
            0.5 + 0.5 * np.sin(r * freq_value * PHI_PHI) * 
            np.exp(-r * 5)  # Exponential decay
        )
        
        # Fill the 3D component of the field
        slices_3d = tuple([slice(None)] * 3 + [0] * (self.ndim - 3))
        self.data[slices_3d] = pattern
        
        # For higher dimensions, create phi-coherent patterns
        for dim in range(3, self.ndim):
            dim_size = self.dimensions[dim]
            
            # Get dimension-specific phi factor
            if dim == 3:  # Time dimension
                phi_factor = LAMBDA
            elif dim == 4:  # Consciousness dimension
                phi_factor = PHI
            elif dim == 5:  # Intention dimension
                phi_factor = PHI_SQUARED
            else:  # Unified field dimension
                phi_factor = PHI_CUBED
            
            # Fill each slice with a phi-transformed version of the 3D pattern
            for i in range(dim_size):
                # Calculate transformation factor based on position
                pos_factor = phi_factor * np.sin(np.pi * i / dim_size)
                
                # Create the slice tuple for this position
                slices = [slice(None), slice(None), slice(None)] + [0] * (self.ndim - 3)
                slices[dim] = i
                
                # Apply transformation to the pattern
                if i == 0:
                    self.data[tuple(slices)] = pattern
                else:
                    transformed = pattern * (1.0 - pos_factor) + \
                                 np.roll(pattern, int(i * PHI), axis=0) * pos_factor
                    self.data[tuple(slices)] = transformed
        
        # Apply smooth transitions between dimensions
        self.smooth_dimensions()
        
        # Apply final coherence adjustment
        current_coherence = self.calculate_coherence()
        if current_coherence < coherence_target:
            # Apply phi mask to enhance coherence
            self.apply_phi_mask(scale=1.0 - (current_coherence / coherence_target))
            self.calculate_coherence()  # Recalculate coherence after mask
    
    def smooth_dimensions(self) -> None:
        """Apply smoothing between dimensions for continuity."""
        if self.ndim <= 3:
            return  # No higher dimensions to smooth
        
        # For each higher dimension
        for dim in range(3, self.ndim):
            dim_size = self.dimensions[dim]
            
            # Skip if dimension has only one element
            if dim_size <= 1:
                continue
            
            # Apply smoothing along this dimension
            smoothed = self.data.copy()
            
            for i in range(dim_size):
                # Create slices for this position
                slices = [slice(None)] * self.ndim
                slices[dim] = i
                
                # Calculate weighted average with adjacent positions
                if i == 0:
                    # First position - average with next
                    next_slices = slices.copy()
                    next_slices[dim] = 1
                    smoothed[tuple(slices)] = (
                        self.data[tuple(slices)] * PHI + 
                        self.data[tuple(next_slices)] * LAMBDA
                    ) / (PHI + LAMBDA)
                
                elif i == dim_size - 1:
                    # Last position - average with previous
                    prev_slices = slices.copy()
                    prev_slices[dim] = i - 1
                    smoothed[tuple(slices)] = (
                        self.data[tuple(slices)] * PHI + 
                        self.data[tuple(prev_slices)] * LAMBDA
                    ) / (PHI + LAMBDA)
                
                else:
                    # Middle position - average with both neighbors
                    prev_slices = slices.copy()
                    prev_slices[dim] = i - 1
                    next_slices = slices.copy()
                    next_slices[dim] = i + 1
                    
                    smoothed[tuple(slices)] = (
                        self.data[tuple(slices)] * PHI + 
                        self.data[tuple(prev_slices)] * LAMBDA / 2 +
                        self.data[tuple(next_slices)] * LAMBDA / 2
                    ) / (PHI + LAMBDA)
            
            # Update the data
            self.data = smoothed
    
    def extract_frequency_band(self, band_name: str) -> np.ndarray:
        """
        Extract a specific frequency band from the hyperfield.
        
        Args:
            band_name: Name of the frequency band to extract
            
        Returns:
            3D array with the extracted frequency band
        """
        if band_name not in self.frequency_bands:
            raise ValueError(f"Unknown frequency band: {band_name}")
        
        # Get the frequency band limits
        freq_min, freq_max = self.frequency_bands[band_name]
        
        # Project to 3D for processing
        projection = self.project_to_3d()
        
        # Compute FFT
        fft_data = np.fft.fftn(projection)
        fft_mag = np.abs(fft_data)
        
        # Create a mask for the frequency band
        freq_resolution = 1.0  # Arbitrary resolution scaling
        freq_coords = [np.fft.fftfreq(d) * d * freq_resolution for d in projection.shape]
        freq_grids = np.meshgrid(*freq_coords, indexing='ij')
        
        # Calculate magnitude of frequency
        freq_mag = np.sqrt(sum(grid**2 for grid in freq_grids))
        
        # Create a band-pass filter
        band_filter = np.logical_and(freq_mag >= freq_min, freq_mag <= freq_max)
        
        # Apply filter
        filtered_fft = fft_data.copy()
        filtered_fft[~band_filter] = 0
        
        # Inverse FFT to get spatial representation
        return np.real(np.fft.ifftn(filtered_fft))
    
    def shift_dimension(self, from_dim: int, to_dim: int, 
                       intensity: float = 0.5) -> None:
        """
        Shift information between dimensions.
        
        Args:
            from_dim: Source dimension index
            to_dim: Target dimension index
            intensity: Intensity of the shift (0-1)
        """
        if from_dim < 0 or from_dim >= self.ndim or to_dim < 0 or to_dim >= self.ndim:
            raise ValueError(f"Dimension indices must be between 0 and {self.ndim-1}")
        
        if from_dim == to_dim:
            return  # No shift needed
        
        # Calculate phi-weighted intensity
        phi_intensity = intensity * (PHI / (PHI + 1))
        
        # Extract data from source dimension
        source_slices = [slice(None)] * self.ndim
        source_samples = []
        
        # Sample source dimension at phi-harmonic intervals
        dim_size = self.dimensions[from_dim]
        sample_points = [int(dim_size * i / PHI) % dim_size for i in range(1, 4)]
        
        for idx in sample_points:
            source_slices[from_dim] = idx
            source_samples.append(self.data[tuple(source_slices)].copy())
        
        # Average the samples with phi-weighting
        source_data = sum(
            sample * (PHI ** (-(i+1))) 
            for i, sample in enumerate(source_samples)
        ) / sum(PHI ** (-(i+1)) for i in range(len(source_samples)))
        
        # Apply to target dimension
        target_slices = [slice(None)] * self.ndim
        dim_size = self.dimensions[to_dim]
        
        # Apply to target with phi-harmonic distribution
        for i in range(dim_size):
            # Calculate phi-weighted intensity based on position
            pos_weight = np.sin(np.pi * i / dim_size * PHI) * phi_intensity
            
            # Skip if weight is negligible
            if pos_weight < 0.01:
                continue
            
            # Create slice for this position
            target_slices[to_dim] = i
            
            # Blend source data into this position
            self.data[tuple(target_slices)] = (
                self.data[tuple(target_slices)] * (1.0 - pos_weight) +
                source_data * pos_weight
            )
    
    def get_dimensional_profile(self) -> Dict[str, float]:
        """
        Get a profile of information distribution across dimensions.
        
        Returns:
            Dictionary with metrics for each dimension
        """
        profile = {}
        
        # For each dimension, calculate variance and energy
        for dim in range(self.ndim):
            # Calculate variance along this dimension
            variance = np.var(self.data, axis=dim)
            
            # Calculate energy (sum of squared values)
            slices = [slice(None)] * self.ndim
            slices[dim] = slice(1, None)  # Exclude first element for gradient
            gradient = np.diff(self.data, axis=dim)
            energy = np.sum(gradient ** 2)
            
            # Calculate phi-resonance
            # Check if dimension size is sufficient for FFT
            if self.dimensions[dim] >= 4:
                # Compute FFT along this dimension
                fft_data = np.fft.fft(self.data, axis=dim)
                fft_mag = np.abs(fft_data)
                
                # Check for phi-harmonic frequencies
                freqs = np.fft.fftfreq(self.dimensions[dim])
                phi_idx = int(self.dimensions[dim] / PHI) % self.dimensions[dim]
                phi2_idx = int(self.dimensions[dim] / PHI_SQUARED) % self.dimensions[dim]
                
                phi_energy = np.mean(fft_mag.take(phi_idx, axis=dim))
                phi2_energy = np.mean(fft_mag.take(phi2_idx, axis=dim))
                
                # Store resonance metrics
                profile[f"dim{dim}_phi_resonance"] = phi_energy
                profile[f"dim{dim}_phi2_resonance"] = phi2_energy
            
            # Store basic metrics
            profile[f"dim{dim}_variance"] = float(np.mean(variance))
            profile[f"dim{dim}_energy"] = float(energy)
            
            # Calculate information density estimate
            if dim < 3:
                # Spatial dimension - use standard entropy
                flat_dim = self.data.swapaxes(0, dim).reshape(self.dimensions[dim], -1)
                hist, _ = np.histogram(flat_dim, bins=20, density=True)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                profile[f"dim{dim}_entropy"] = float(entropy)
            else:
                # Higher dimension - use phi-weighted entropy
                flat_dim = self.data.swapaxes(0, dim).reshape(self.dimensions[dim], -1)
                hist, _ = np.histogram(flat_dim, bins=20, density=True)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                
                # Weight by phi factor based on dimension
                if dim == 3:  # Time
                    phi_factor = LAMBDA
                elif dim == 4:  # Consciousness
                    phi_factor = 1.0
                elif dim == 5:  # Intention
                    phi_factor = PHI
                else:  # Unified field
                    phi_factor = PHI_SQUARED
                
                profile[f"dim{dim}_phi_entropy"] = float(entropy * phi_factor)
        
        return profile