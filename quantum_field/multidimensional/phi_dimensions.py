"""
Phi Dimensional Scaling Module

Tools for scaling and manipulating dimensions according to phi-harmonic principles,
enabling dimensional expansion beyond conventional 3D space.
"""

import numpy as np
from typing import Tuple, List, Dict, Union, Optional

# Import sacred constants
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sacred_constants import (
    PHI, PHI_SQUARED, PHI_CUBED, PHI_PHI, 
    LAMBDA, PHI_DIMENSIONS, RESONANCE_PATTERNS
)


class PhiDimensionalScaling:
    """
    Manages phi-harmonic scaling between dimensions, enabling expansion
    and contraction along phi-scaled relationships.
    """
    
    def __init__(self, base_dimension: int = 21):
        """
        Initialize phi-dimensional scaling with a base dimension.
        
        Args:
            base_dimension: Base dimension size (default: 21)
        """
        self.base_dimension = base_dimension
        
        # Initialize scaling dimensions
        self.dimensions = self._calculate_phi_dimensions(base_dimension)
        
        # Phi-harmonic frequency ratios for dimensional scaling
        self.frequency_ratios = {
            3: 1.0,                # 3D (spatial) - base frequency
            4: LAMBDA,             # 4D (time) - compressed by inverse phi
            5: 1.0,                # 5D (consciousness) - same as base
            6: PHI,                # 6D (intention) - expanded by phi
            7: PHI_SQUARED,        # 7D (unified field) - expanded by phi²
        }
        
        # Dimensional attributes with phi-harmonic characteristics
        self.dimensional_attributes = {
            3: {
                "name": "spatial",
                "phi_factor": 1.0,
                "resonance_pattern": [1.0, PHI, LAMBDA],
                "coherence_weight": 1.0
            },
            4: {
                "name": "temporal",
                "phi_factor": LAMBDA,
                "resonance_pattern": [LAMBDA, 1.0, PHI],
                "coherence_weight": LAMBDA
            },
            5: {
                "name": "conscious",
                "phi_factor": PHI,
                "resonance_pattern": [PHI, 1.0, PHI_SQUARED],
                "coherence_weight": PHI
            },
            6: {
                "name": "intention",
                "phi_factor": PHI_SQUARED,
                "resonance_pattern": [PHI_SQUARED, PHI, 1.0],
                "coherence_weight": PHI_SQUARED
            },
            7: {
                "name": "unified",
                "phi_factor": PHI_CUBED,
                "resonance_pattern": [PHI_CUBED, PHI_SQUARED, PHI],
                "coherence_weight": PHI_CUBED
            }
        }
    
    def _calculate_phi_dimensions(self, base: int) -> Dict[int, int]:
        """
        Calculate dimension sizes using phi-scaling from a base dimension.
        
        Args:
            base: Base dimension size
            
        Returns:
            Dictionary mapping dimension number to its size
        """
        dimensions = {
            3: base,  # 3D - base dimension
            4: max(2, int(base * LAMBDA)),  # 4D - phi inverse scaling (compressed)
            5: base,  # 5D - same as base
            6: max(2, int(base * PHI)),  # 6D - phi scaling (expanded)
            7: max(2, int(base * PHI_SQUARED)),  # 7D - phi² scaling (expanded more)
        }
        return dimensions
    
    @staticmethod
    def dimension_name(dim: int) -> str:
        """
        Get the name of a specific dimension.
        
        Args:
            dim: Dimension number (3-7)
            
        Returns:
            String name of the dimension
        """
        names = {
            3: "spatial",
            4: "temporal",
            5: "consciousness",
            6: "intention",
            7: "unified_field"
        }
        return names.get(dim, f"dimension_{dim}")
    
    def get_dimension_size(self, dim: int) -> int:
        """
        Get the size of a specific dimension with phi-scaling applied.
        
        Args:
            dim: Dimension number
            
        Returns:
            The size of the dimension
        """
        if dim in self.dimensions:
            return self.dimensions[dim]
        
        # For unknown dimensions, apply phi-scaling based on pattern
        if dim < 3:
            return self.base_dimension  # Default for spatial dimensions
        
        # For higher dimensions, apply phi^n scaling
        phi_power = dim - 5  # 0 for dim 5, 1 for dim 6, etc.
        return max(2, int(self.base_dimension * (PHI ** phi_power)))
    
    def get_scaling_matrix(self, from_dim: int, to_dim: int) -> np.ndarray:
        """
        Generate a scaling matrix for transforming between dimensions.
        
        Args:
            from_dim: Source dimension
            to_dim: Target dimension
            
        Returns:
            Transformation matrix for scaling
        """
        if from_dim not in self.dimensional_attributes or to_dim not in self.dimensional_attributes:
            raise ValueError(f"Dimensions must be between 3 and 7")
        
        # Get phi factors for each dimension
        from_factor = self.dimensional_attributes[from_dim]["phi_factor"]
        to_factor = self.dimensional_attributes[to_dim]["phi_factor"]
        
        # Calculate transformation ratio
        if from_factor == 0 or to_factor == 0:
            ratio = 1.0
        else:
            ratio = to_factor / from_factor
        
        # Create scaling matrix
        from_size = self.dimensions[from_dim]
        to_size = self.dimensions[to_dim]
        
        # Initialize empty matrix
        matrix = np.zeros((to_size, from_size))
        
        # Fill the matrix based on phi-harmonic scaling
        for i in range(to_size):
            normalized_pos = i / to_size
            
            # Calculate source positions based on phi-weighted scaling
            if ratio >= 1.0:
                # Expanding - spread out source values
                num_sources = max(1, int(ratio))
                for j in range(num_sources):
                    src_pos = (normalized_pos + j / num_sources) % 1.0
                    src_idx = min(from_size - 1, int(src_pos * from_size))
                    weight = (PHI ** (-j)) / sum(PHI ** (-k) for k in range(num_sources))
                    matrix[i, src_idx] += weight
            else:
                # Compressing - average source values
                src_start = normalized_pos - (1 / ratio) / 2
                src_end = normalized_pos + (1 / ratio) / 2
                
                # Handle edge cases
                src_start = max(0, src_start)
                src_end = min(1, src_end)
                
                # Calculate source indices
                start_idx = max(0, min(from_size - 1, int(src_start * from_size)))
                end_idx = max(0, min(from_size - 1, int(src_end * from_size) + 1))
                
                # Handle degenerate case
                if start_idx == end_idx:
                    matrix[i, start_idx] = 1.0
                else:
                    # Distribute weight over source indices with phi-weighting
                    num_indices = end_idx - start_idx
                    weights = [PHI ** (-(j+1)) for j in range(num_indices)]
                    weight_sum = sum(weights)
                    
                    for j in range(num_indices):
                        matrix[i, start_idx + j] = weights[j] / weight_sum
        
        # Normalize each row to ensure preservation of total energy
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        normalized_matrix = matrix / row_sums
        
        return normalized_matrix
    
    def transform_data(self, data: np.ndarray, 
                      from_dim: int, to_dim: int, 
                      dim_axis: int = 0) -> np.ndarray:
        """
        Transform data from one dimensional scaling to another.
        
        Args:
            data: Input data array
            from_dim: Source dimension
            to_dim: Target dimension
            dim_axis: Axis representing the dimension to transform
            
        Returns:
            Transformed data
        """
        # Get the transformation matrix
        transform = self.get_scaling_matrix(from_dim, to_dim)
        
        # Determine shapes
        from_size = self.dimensions[from_dim]
        to_size = self.dimensions[to_dim]
        
        # Validate input shape
        if data.shape[dim_axis] != from_size:
            raise ValueError(f"Data size along dimension axis ({data.shape[dim_axis]}) "
                           f"doesn't match expected size for dimension {from_dim} ({from_size})")
        
        # Prepare output array
        new_shape = list(data.shape)
        new_shape[dim_axis] = to_size
        result = np.zeros(new_shape, dtype=data.dtype)
        
        # Create slices for the transformation
        src_slices = [slice(None)] * len(data.shape)
        dst_slices = [slice(None)] * len(data.shape)
        
        # Apply transformation
        for i in range(to_size):
            dst_slices[dim_axis] = i
            
            # Weighted sum of source elements
            weighted_sum = None
            
            for j in range(from_size):
                if transform[i, j] == 0:
                    continue
                    
                src_slices[dim_axis] = j
                src_data = data[tuple(src_slices)]
                
                if weighted_sum is None:
                    weighted_sum = src_data * transform[i, j]
                else:
                    weighted_sum += src_data * transform[i, j]
            
            # Store the result
            if weighted_sum is not None:
                result[tuple(dst_slices)] = weighted_sum
        
        return result
    
    def create_phi_dimension_grid(self, 
                                 dim: int, 
                                 frequency: float = 1.0) -> np.ndarray:
        """
        Create a grid for a specific dimension with phi-harmonic patterning.
        
        Args:
            dim: Dimension to create grid for
            frequency: Base frequency for the pattern
            
        Returns:
            Array with phi-harmonic grid pattern
        """
        if dim not in self.dimensions:
            raise ValueError(f"Invalid dimension: {dim}")
        
        size = self.dimensions[dim]
        
        # Get phi_factor for this dimension
        phi_factor = self.dimensional_attributes[dim]["phi_factor"]
        
        # Create coordinate array
        coords = np.linspace(0, 1, size)
        
        # Apply phi-harmonic patterning
        phi_pattern = 0.5 + 0.5 * np.sin(coords * frequency * np.pi * 2 * phi_factor)
        
        # Add phi-harmonic overtones
        for harmonic in range(2, 5):
            # Calculate phi-scaled harmonic frequency
            harmonic_freq = frequency * harmonic * PHI
            amplitude = 1.0 / (harmonic * PHI)
            
            # Add the harmonic
            phi_pattern += amplitude * np.sin(coords * harmonic_freq * np.pi * 2 * phi_factor)
        
        # Normalize to 0-1 range
        min_val = np.min(phi_pattern)
        max_val = np.max(phi_pattern)
        if max_val > min_val:
            phi_pattern = (phi_pattern - min_val) / (max_val - min_val)
        
        return phi_pattern
    
    def get_dimension_profile(self) -> Dict[str, Dict[str, Union[str, float, List[float]]]]:
        """
        Get a complete profile of all dimensions with their phi-harmonic characteristics.
        
        Returns:
            Dictionary with dimensional profiles
        """
        profile = {}
        
        for dim in sorted(self.dimensions.keys()):
            dim_profile = {
                "name": self.dimension_name(dim),
                "size": self.dimensions[dim],
                "phi_factor": self.dimensional_attributes[dim]["phi_factor"],
                "resonance_pattern": self.dimensional_attributes[dim]["resonance_pattern"],
                "coherence_weight": self.dimensional_attributes[dim]["coherence_weight"],
                "frequency_ratio": self.frequency_ratios.get(dim, 1.0),
            }
            
            profile[f"dimension_{dim}"] = dim_profile
        
        return profile
    
    def create_dimension_frequency_map(self, base_frequency: float = 432.0) -> Dict[int, float]:
        """
        Create a mapping of dimensions to their phi-resonant frequencies.
        
        Args:
            base_frequency: Base frequency (default: 432Hz - unity)
        
        Returns:
            Dictionary mapping dimension numbers to frequencies
        """
        frequency_map = {}
        
        for dim in self.dimensions:
            # Calculate phi-resonant frequency for this dimension
            if dim == 3:
                # 3D space - base frequency (432Hz)
                frequency_map[dim] = base_frequency
            elif dim == 4:
                # 4D time - base * phi (528Hz - love frequency)
                frequency_map[dim] = base_frequency * PHI
            elif dim == 5:
                # 5D consciousness - base * phi² (594Hz - cascade frequency)
                frequency_map[dim] = base_frequency * PHI_SQUARED
            elif dim == 6:
                # 6D intention - base * phi³ (672Hz - truth frequency)
                frequency_map[dim] = base_frequency * PHI_CUBED
            elif dim == 7:
                # 7D unified field - base * phi⁴ (768Hz - oneness frequency)
                frequency_map[dim] = base_frequency * (PHI ** 4)
        
        return frequency_map