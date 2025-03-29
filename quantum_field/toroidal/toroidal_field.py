"""
Toroidal Field Implementation

This module implements toroidal field structures that maintain phi-harmonic energy flow
through balanced input/output cycles and continuous energy circulation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import math

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.constants import FIELD_3D_HARMONICS, GEOMETRIC_CONSTANTS

class ToroidalField:
    """
    A quantum field with toroidal energy flow structures implementing perfect phi-harmonic
    balance through continuous circulation patterns.
    
    Attributes:
        data: The 3D numpy array containing the field data
        major_radius: The major radius of the torus (distance from center to torus center)
        minor_radius: The minor radius of the torus (radius of the tube)
        dimensions: The dimensions of the field data array
        flow_rate: The current flow rate of energy through the torus
        coherence: The current coherence level of the field
        energy_level: The current total energy contained in the field
        phi_alignment: The degree of alignment with phi-harmonic principles (0-1)
        balance_factor: The balance between input and output energy flows (ideally 1.0)
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int] = (32, 32, 32),
                 major_radius_factor: float = 0.35,
                 minor_radius_factor: float = 0.15,
                 frequency_name: str = 'unity',
                 initialize: bool = True):
        """
        Initialize a toroidal quantum field.
        
        Args:
            dimensions: 3D dimensions of the field (height, width, depth)
            major_radius_factor: Factor of field size for major radius (0.0-0.5)
            minor_radius_factor: Factor of field size for minor radius (0.0-0.3)
            frequency_name: The sacred frequency to use for initial field resonance
            initialize: Whether to initialize the field data immediately
        """
        self.dimensions = dimensions
        self.data = np.zeros(dimensions, dtype=np.float32)
        
        # Set torus parameters
        size = min(dimensions)
        self.major_radius = size * major_radius_factor
        self.minor_radius = size * minor_radius_factor
        
        # Get the sacred frequency
        self.frequency = SACRED_FREQUENCIES.get(frequency_name, SACRED_FREQUENCIES['unity'])
        
        # Energy flow metrics
        self.flow_rate = 0.0
        self.coherence = 0.0
        self.energy_level = 0.0
        self.phi_alignment = 0.0
        self.balance_factor = 1.0
        
        # Initialize the field
        if initialize:
            self.initialize_toroidal_field()

    def initialize_toroidal_field(self) -> None:
        """
        Initialize the toroidal field with a phi-harmonic energy distribution.
        """
        # Create coordinate grids
        x = np.linspace(-1, 1, self.dimensions[0])
        y = np.linspace(-1, 1, self.dimensions[1])
        z = np.linspace(-1, 1, self.dimensions[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from torus center ring (centered at origin)
        # The torus central axis is along the z-axis
        
        # Distance from the z-axis
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        
        # Distance from the torus ring
        dist_from_torus_ring = np.sqrt((dist_from_center_axis - self.major_radius)**2 + Z**2)
        
        # Generate field values based on distance from the torus ring
        # Values are highest at the torus ring and fall off with distance
        field_values = np.exp(-dist_from_torus_ring**2 / (self.minor_radius**2 * 2))
        
        # Add phi-harmonic variations around the torus
        # Calculate angle around the torus (in xy-plane)
        theta = np.arctan2(Y, X)
        
        # Add phi-harmonic wave patterns around the torus
        pattern_factor = self.frequency / 100.0 * PHI
        phi_harmonic = np.sin(theta * pattern_factor) * np.sin(dist_from_torus_ring * PHI * 5)
        
        # Combine the base field with the phi-harmonic pattern
        self.data = field_values * (1 + phi_harmonic * LAMBDA)
        
        # Add flow directionality
        flow_dir = np.sin(theta * PHI) * LAMBDA * np.exp(-dist_from_torus_ring / self.minor_radius)
        self.data += flow_dir
        
        # Initialize flow metrics
        self.calculate_flow_metrics()
    
    def calculate_flow_metrics(self) -> Dict[str, float]:
        """
        Calculate the current flow metrics of the toroidal field.
        
        Returns:
            Dictionary of flow metrics
        """
        # Calculate total energy in field
        self.energy_level = np.sum(np.abs(self.data))
        
        # Calculate flow coherence - measured by smoothness of gradient
        gradients = np.gradient(self.data)
        gradient_magnitude = np.sqrt(sum(np.square(g) for g in gradients))
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) * PHI)
        
        # Calculate phi alignment by sampling points
        samples = 1000
        sampled_values = self.data.flatten()[
            np.random.choice(self.data.size, samples, replace=False)
        ]
        
        phi_aligned_count = 0
        for value in sampled_values:
            # Check if value is close to a multiple of PHI
            nearest_multiple = round(value / PHI)
            deviation = abs(value - (nearest_multiple * PHI))
            if deviation < (PHI * 0.1):  # Within 10% of PHI
                phi_aligned_count += 1
        
        self.phi_alignment = phi_aligned_count / samples
        
        # Calculate flow rate - measured by directional coherence
        if self.data.size > 1:
            # Estimate flow using curl of field
            curl_x = np.gradient(self.data, axis=1, edge_order=2)
            curl_y = np.gradient(self.data, axis=0, edge_order=2) 
            curl_z = np.gradient(self.data, axis=2, edge_order=2)
            
            curl_magnitude = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
            self.flow_rate = np.mean(curl_magnitude) * PHI
        
        # Calculate overall coherence
        self.coherence = (smoothness + self.phi_alignment) / 2
        
        # Balance factor - ideally 1.0 for perfect balance
        # Calculate by comparing energy in different regions of the torus
        x_center = self.dimensions[0] // 2
        y_center = self.dimensions[1] // 2
        
        left_half = self.data[:x_center, :, :]
        right_half = self.data[x_center:, :, :]
        
        top_half = self.data[:, :y_center, :]
        bottom_half = self.data[:, y_center:, :]
        
        left_energy = np.sum(np.abs(left_half))
        right_energy = np.sum(np.abs(right_half))
        
        top_energy = np.sum(np.abs(top_half))
        bottom_energy = np.sum(np.abs(bottom_half))
        
        # Calculate balance ratios (ideal is 1.0)
        x_balance = min(left_energy, right_energy) / max(left_energy, right_energy)
        y_balance = min(top_energy, bottom_energy) / max(top_energy, bottom_energy)
        
        self.balance_factor = (x_balance + y_balance) / 2
        
        # Return all metrics
        return {
            "energy_level": self.energy_level,
            "flow_rate": self.flow_rate,
            "coherence": self.coherence,
            "phi_alignment": self.phi_alignment,
            "balance_factor": self.balance_factor
        }
    
    def apply_flow(self, time_factor: float = 0.1) -> None:
        """
        Apply toroidal flow evolution to the field for one time step.
        
        Args:
            time_factor: Scaling factor for time evolution speed
        """
        # Create coordinate grids
        x = np.linspace(-1, 1, self.dimensions[0])
        y = np.linspace(-1, 1, self.dimensions[1])
        z = np.linspace(-1, 1, self.dimensions[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from the z-axis
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        
        # Distance from the torus ring
        dist_from_torus_ring = np.sqrt((dist_from_center_axis - self.major_radius)**2 + Z**2)
        
        # Calculate angle around the torus (in xy-plane)
        theta = np.arctan2(Y, X)
        
        # Calculate phi-based rotation rate
        # Faster near the torus ring, slower away from it
        rotation_rate = PHI * np.exp(-dist_from_torus_ring / self.minor_radius)
        
        # Rotation angle depends on distance from torus ring
        rotation_angle = rotation_rate * time_factor
        
        # Apply rotation by shifting the field in theta direction
        # First, convert to cylindrical coordinates
        r = dist_from_center_axis
        z_cyl = Z
        
        # Shift theta based on rotation angle
        new_theta = theta + rotation_angle
        
        # Convert back to Cartesian
        new_X = r * np.cos(new_theta)
        new_Y = r * np.sin(new_theta)
        
        # Interpolate field values at new positions
        # We need to map from normalized coordinates back to array indices
        x_idx = (new_X + 1) * (self.dimensions[0] - 1) / 2
        y_idx = (new_Y + 1) * (self.dimensions[1] - 1) / 2
        z_idx = (Z + 1) * (self.dimensions[2] - 1) / 2
        
        # Clip indices to valid range
        x_idx = np.clip(x_idx, 0, self.dimensions[0] - 1)
        y_idx = np.clip(y_idx, 0, self.dimensions[1] - 1)
        z_idx = np.clip(z_idx, 0, self.dimensions[2] - 1)
        
        # Trilinear interpolation - simple version using nearest neighbors
        x_idx_floor = np.floor(x_idx).astype(int)
        y_idx_floor = np.floor(y_idx).astype(int)
        z_idx_floor = np.floor(z_idx).astype(int)
        
        # Ensure indices are within bounds
        x_idx_floor = np.clip(x_idx_floor, 0, self.dimensions[0] - 1)
        y_idx_floor = np.clip(y_idx_floor, 0, self.dimensions[1] - 1)
        z_idx_floor = np.clip(z_idx_floor, 0, self.dimensions[2] - 1)
        
        # Get rotated field
        rotated_field = np.zeros_like(self.data)
        
        # Apply field rotation - simple method
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    src_i = x_idx_floor[i, j, k]
                    src_j = y_idx_floor[i, j, k]
                    src_k = z_idx_floor[i, j, k]
                    
                    rotated_field[i, j, k] = self.data[src_i, src_j, src_k]
        
        # Apply some phi-harmonic diffusion for smoothness
        diffusion_factor = LAMBDA * 0.1
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)
        
        from scipy.ndimage import convolve
        smoothed_field = convolve(rotated_field, kernel, mode='constant', cval=0.0)
        
        # Blend between rotated and smoothed field
        self.data = (1 - diffusion_factor) * rotated_field + diffusion_factor * smoothed_field
        
        # Recalculate flow metrics
        self.calculate_flow_metrics()
    
    def optimize_coherence(self, target_coherence: float = 0.95) -> float:
        """
        Optimize the field coherence to reach a target level.
        
        Args:
            target_coherence: Target coherence level (0.0-1.0)
            
        Returns:
            The new coherence level
        """
        current_coherence = self.coherence
        
        # Only perform optimization if needed
        if abs(current_coherence - target_coherence) < 0.01:
            return current_coherence
        
        # Determine if we need to increase or decrease coherence
        increase_coherence = target_coherence > current_coherence
        
        # Apply appropriate field transformations to adjust coherence
        if increase_coherence:
            # Enhance phi-resonant structures
            self._enhance_phi_structures()
        else:
            # Introduce controlled variability
            self._add_controlled_variability()
        
        # Calculate new coherence
        self.calculate_flow_metrics()
        return self.coherence
    
    def _enhance_phi_structures(self) -> None:
        """Enhance phi-resonant structures to increase coherence."""
        # Calculate FFT
        fft_data = np.fft.fftn(self.data)
        
        # Get spectrum magnitude
        magnitude = np.abs(fft_data)
        
        # Find phi-resonant frequencies
        freq_space = np.fft.fftfreq(self.dimensions[0])
        phi_freqs = []
        
        # Find frequencies that are multiples of 1/phi
        for i, freq in enumerate(freq_space):
            nearest_multiple = round(freq * PHI)
            deviation = abs(freq * PHI - nearest_multiple)
            
            if deviation < 0.1:  # Within 10% of a multiple of 1/phi
                phi_freqs.append(i)
        
        # Create mask to enhance phi-resonant frequencies
        mask = np.ones_like(fft_data, dtype=float)
        
        # Enhance frequencies along each dimension
        for i in phi_freqs:
            mask[i, :, :] *= (1 + LAMBDA)
            mask[:, i, :] *= (1 + LAMBDA)
            mask[:, :, i] *= (1 + LAMBDA)
        
        # Apply mask
        fft_data_enhanced = fft_data * mask
        
        # Convert back to spatial domain
        enhanced_field = np.real(np.fft.ifftn(fft_data_enhanced))
        
        # Normalize to preserve energy
        if np.max(np.abs(enhanced_field)) > 0:
            enhanced_field = enhanced_field * (np.max(np.abs(self.data)) / np.max(np.abs(enhanced_field)))
        
        # Update field
        self.data = enhanced_field
    
    def _add_controlled_variability(self) -> None:
        """Add controlled variability to decrease coherence."""
        # Generate noise with phi-harmonic structure
        noise = np.random.randn(*self.dimensions).astype(np.float32)
        
        # Apply phi-based smoothing
        from scipy.ndimage import gaussian_filter
        noise_smooth = gaussian_filter(noise, sigma=LAMBDA)
        
        # Scale noise to a fraction of the field magnitude
        max_field = np.max(np.abs(self.data))
        noise_scale = max_field * LAMBDA * 0.1
        
        # Add noise to field
        self.data = self.data + noise_smooth * noise_scale
    
    def sync_to_frequency(self, frequency_name: str) -> None:
        """
        Synchronize the toroidal field to a specific sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency to synchronize with
        """
        # Get the sacred frequency
        if frequency_name in SACRED_FREQUENCIES:
            new_frequency = SACRED_FREQUENCIES[frequency_name]
        else:
            new_frequency = SACRED_FREQUENCIES['unity']  # Default to 432 Hz
        
        # Store the current frequency
        old_frequency = self.frequency
        self.frequency = new_frequency
        
        # Calculate coordinate grids
        x = np.linspace(-1, 1, self.dimensions[0])
        y = np.linspace(-1, 1, self.dimensions[1])
        z = np.linspace(-1, 1, self.dimensions[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from the z-axis
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        
        # Distance from the torus ring
        dist_from_torus_ring = np.sqrt((dist_from_center_axis - self.major_radius)**2 + Z**2)
        
        # Calculate angle around the torus (in xy-plane)
        theta = np.arctan2(Y, X)
        
        # Adjust field frequency patterns
        # Only modify within the torus region
        torus_mask = dist_from_torus_ring <= (self.minor_radius * 1.5)
        
        # Calculate new frequency pattern
        pattern_factor = new_frequency / old_frequency
        freq_adjustment = np.sin(theta * pattern_factor * PHI) * np.sin(dist_from_torus_ring * PHI * 5 * pattern_factor)
        
        # Apply frequency adjustment only within torus region
        adjustment_strength = LAMBDA * np.exp(-dist_from_torus_ring / self.minor_radius)
        self.data[torus_mask] = self.data[torus_mask] * (1 - adjustment_strength[torus_mask]) + \
                               self.data[torus_mask] * freq_adjustment[torus_mask] * adjustment_strength[torus_mask]
        
        # Update flow metrics
        self.calculate_flow_metrics()
        
        print(f"Field synchronized to {frequency_name} frequency ({new_frequency} Hz)")
        print(f"New coherence: {self.coherence:.4f}, Flow rate: {self.flow_rate:.4f}")
    
    def create_input_output_cycle(self, io_ratio: float = 1.0) -> None:
        """
        Create a balanced input/output cycle in the toroidal field.
        
        Args:
            io_ratio: Ratio of input to output (1.0 for perfect balance)
        """
        # Calculate coordinate grids
        x = np.linspace(-1, 1, self.dimensions[0])
        y = np.linspace(-1, 1, self.dimensions[1])
        z = np.linspace(-1, 1, self.dimensions[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from the z-axis
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        
        # Distance from the torus ring
        dist_from_torus_ring = np.sqrt((dist_from_center_axis - self.major_radius)**2 + Z**2)
        
        # Calculate angle around the torus (in xy-plane)
        theta = np.arctan2(Y, X)
        
        # Define input and output regions (opposing sides of the torus)
        input_region = (theta >= 0) & (theta <= np.pi)
        output_region = (theta < 0) | (theta > np.pi)
        
        # Only apply within the torus
        torus_region = dist_from_torus_ring <= (self.minor_radius * 1.5)
        
        input_mask = input_region & torus_region
        output_mask = output_region & torus_region
        
        # Calculate input/output strengths
        input_factor = 1.0 + (LAMBDA * (io_ratio - 1.0))
        output_factor = 1.0 - (LAMBDA * (io_ratio - 1.0))
        
        # Ensure output factor doesn't go below minimum threshold
        output_factor = max(output_factor, 0.5)
        
        # Apply factors
        self.data[input_mask] *= input_factor
        self.data[output_mask] *= output_factor
        
        # Add flow circulation to balance the field
        circulation = np.sin(theta * PHI) * np.exp(-dist_from_torus_ring / self.minor_radius) * LAMBDA
        
        # Apply stronger circulation to compensate for imbalance
        imbalance = abs(io_ratio - 1.0)
        circulation *= (1.0 + imbalance)
        
        self.data[torus_region] += circulation[torus_region]
        
        # Update flow metrics
        self.calculate_flow_metrics()
        
        print(f"Created input/output cycle with ratio {io_ratio:.2f}")
        print(f"Balance factor: {self.balance_factor:.4f}")

    def animate(self, frames: int = 10, time_factor: float = 0.1) -> List[np.ndarray]:
        """
        Generate animation frames of the toroidal field evolution.
        
        Args:
            frames: Number of frames to generate
            time_factor: Time factor for each frame
            
        Returns:
            List of field data arrays for each frame
        """
        animation_frames = []
        
        # Store original field data
        original_data = self.data.copy()
        
        for i in range(frames):
            # Apply flow evolution
            self.apply_flow(time_factor)
            
            # Store frame
            animation_frames.append(self.data.copy())
        
        # Restore original field
        self.data = original_data
        
        return animation_frames

    def to_slices(self, axis: int = 2, center_slice: bool = True) -> List[np.ndarray]:
        """
        Convert the 3D toroidal field to 2D slices for visualization.
        
        Args:
            axis: The axis to slice along (0, 1, or 2)
            center_slice: Whether to take only the central slice
            
        Returns:
            List of 2D arrays representing slices of the field
        """
        if center_slice:
            # Return only the central slice
            if axis == 0:
                return [self.data[self.dimensions[0]//2, :, :]]
            elif axis == 1:
                return [self.data[:, self.dimensions[1]//2, :]]
            else:  # axis == 2
                return [self.data[:, :, self.dimensions[2]//2]]
        else:
            # Return all slices
            slices = []
            dim = self.dimensions[axis]
            
            for i in range(dim):
                if axis == 0:
                    slices.append(self.data[i, :, :])
                elif axis == 1:
                    slices.append(self.data[:, i, :])
                else:  # axis == 2
                    slices.append(self.data[:, :, i])
            
            return slices
    
    def visualize_slice(self, slice_index: int = None, axis: int = 2) -> np.ndarray:
        """
        Create a visualization of a specific slice of the toroidal field.
        
        Args:
            slice_index: Index of the slice to visualize (None for center slice)
            axis: The axis to slice along (0, 1, or 2)
            
        Returns:
            2D array with visualization data
        """
        # Get dimensions for the selected axis
        dim = self.dimensions[axis]
        
        # Use center slice if index not provided
        if slice_index is None:
            slice_index = dim // 2
        
        # Get the requested slice
        if axis == 0:
            field_slice = self.data[slice_index, :, :]
        elif axis == 1:
            field_slice = self.data[:, slice_index, :]
        else:  # axis == 2
            field_slice = self.data[:, :, slice_index]
        
        return field_slice

def demo_toroidal_field():
    """
    Demonstrate the toroidal field functionality.
    
    Returns:
        The created toroidal field
    """
    # Create a toroidal field
    print("Creating toroidal field...")
    field = ToroidalField(dimensions=(32, 32, 32), frequency_name='unity')
    
    # Print initial metrics
    metrics = field.calculate_flow_metrics()
    print("\nInitial field metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Demonstrate flow evolution
    print("\nApplying flow evolution...")
    field.apply_flow(time_factor=0.2)
    
    metrics = field.calculate_flow_metrics()
    print("\nMetrics after flow:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Optimize coherence
    print("\nOptimizing field coherence...")
    new_coherence = field.optimize_coherence(target_coherence=0.95)
    print(f"Coherence after optimization: {new_coherence:.4f}")
    
    # Synchronize to different frequency
    print("\nSynchronizing to 'love' frequency...")
    field.sync_to_frequency('love')
    
    # Create balanced I/O cycle
    print("\nCreating balanced input/output cycle...")
    field.create_input_output_cycle(io_ratio=1.0)
    
    return field

if __name__ == "__main__":
    field = demo_toroidal_field()