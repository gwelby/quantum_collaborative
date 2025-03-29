"""
Counter-Rotating Field Systems

This module implements counter-rotating field systems for dimensional stability,
enabling balanced energy dynamics and coherent field synchronization.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import math

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from .toroidal_field import ToroidalField

class CounterRotatingField:
    """
    Implements counter-rotating field systems for dimensional stability.
    
    Attributes:
        toroidal_field: The primary toroidal field
        clockwise_field: Field component rotating clockwise
        counterclockwise_field: Field component rotating counterclockwise
        rotation_ratio: Ratio of rotation speeds (phi-based)
        stability_factor: Dimensional stability factor (0.0-1.0)
        field_separation: Degree of separation between the counter-rotating fields
        harmonization: Degree of harmonic alignment between the fields
    """
    
    def __init__(self, 
                 toroidal_field: Optional[ToroidalField] = None,
                 rotation_ratio: float = PHI,
                 initial_separation: float = 0.5):
        """
        Initialize a counter-rotating field system.
        
        Args:
            toroidal_field: The main toroidal field to work with
            rotation_ratio: Ratio of rotation speeds (phi-based)
            initial_separation: Initial degree of field separation (0.0-1.0)
        """
        self.toroidal_field = toroidal_field
        self.rotation_ratio = rotation_ratio
        
        # Initialize with default fields
        if toroidal_field is not None:
            # Split field into clockwise and counterclockwise components
            self.clockwise_field = np.zeros_like(toroidal_field.data)
            self.counterclockwise_field = np.zeros_like(toroidal_field.data)
            self.split_field(separation=initial_separation)
        else:
            self.clockwise_field = None
            self.counterclockwise_field = None
        
        # Initialize metrics
        self.stability_factor = 0.0
        self.field_separation = initial_separation
        self.harmonization = 0.0
        
        # Counter for rotation steps
        self.rotation_steps = 0
    
    def connect_field(self, toroidal_field: ToroidalField) -> None:
        """
        Connect to a toroidal field and initialize the counter-rotating components.
        
        Args:
            toroidal_field: The toroidal field to connect
        """
        self.toroidal_field = toroidal_field
        
        # Initialize component fields
        self.clockwise_field = np.zeros_like(toroidal_field.data)
        self.counterclockwise_field = np.zeros_like(toroidal_field.data)
        
        # Split the field
        self.split_field(separation=self.field_separation)
    
    def split_field(self, separation: float = 0.5) -> None:
        """
        Split the main field into clockwise and counterclockwise components.
        
        Args:
            separation: Degree of separation between components (0.0-1.0)
        """
        if self.toroidal_field is None:
            return
        
        # Get field data
        field_data = self.toroidal_field.data
        
        # Create coordinate grids
        dims = field_data.shape
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate angle around the torus
        theta = np.arctan2(Y, X)
        
        # Distance from center axis
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        
        # Distance from torus ring
        dist_from_torus_ring = np.sqrt(
            (dist_from_center_axis - self.toroidal_field.major_radius)**2 + Z**2
        )
        
        # Create masks for clockwise and counterclockwise regions
        # Use balanced separation to ensure symmetry
        mask_cw = np.logical_or(
            theta < 0,  # Lower half of torus
            dist_from_torus_ring > (self.toroidal_field.minor_radius * separation * 2)  # Outside core
        )
        
        mask_ccw = np.logical_or(
            theta >= 0,  # Upper half of torus
            dist_from_torus_ring > (self.toroidal_field.minor_radius * separation * 2)  # Outside core
        )
        
        # Create harmonic transition zone
        transition_width = self.toroidal_field.minor_radius * 0.2
        transition_mask = np.abs(theta) < (np.pi / 8)  # Narrow band around x-axis
        
        # Initialize component fields
        self.clockwise_field = np.zeros_like(field_data)
        self.counterclockwise_field = np.zeros_like(field_data)
        
        # Apply direct masks
        self.clockwise_field[mask_cw & ~transition_mask] = field_data[mask_cw & ~transition_mask]
        self.counterclockwise_field[mask_ccw & ~transition_mask] = field_data[mask_ccw & ~transition_mask]
        
        # Apply harmonic transition in overlap zone
        blend_factor = 0.5  # Equal contribution in transition zone
        self.clockwise_field[transition_mask] = field_data[transition_mask] * blend_factor
        self.counterclockwise_field[transition_mask] = field_data[transition_mask] * blend_factor
        
        # Add phase differences to each component
        phase_diff = np.pi * LAMBDA
        
        # Create phi-harmonic phase patterns
        cw_phase = np.cos(theta * PHI + phase_diff) * LAMBDA
        ccw_phase = np.cos(theta * PHI - phase_diff) * LAMBDA
        
        # Apply phases to transition regions
        self.clockwise_field[transition_mask] *= (1 + cw_phase[transition_mask])
        self.counterclockwise_field[transition_mask] *= (1 + ccw_phase[transition_mask])
        
        # Store separation level
        self.field_separation = separation
        
        # Calculate stability metrics
        self.calculate_stability_metrics()
    
    def rotate_components(self, time_factor: float = 0.1) -> None:
        """
        Rotate the field components in opposite directions.
        
        Args:
            time_factor: Time scaling factor for rotation
        """
        if self.toroidal_field is None or self.clockwise_field is None or self.counterclockwise_field is None:
            return
        
        # Get dimensions
        dims = self.toroidal_field.data.shape
        
        # Create coordinate grids
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate polar coordinates
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Rotation angles
        cw_rotation = self.rotation_ratio * time_factor
        ccw_rotation = -time_factor  # Opposite direction
        
        # Rotate clockwise field
        theta_cw = theta - cw_rotation
        X_cw = r * np.cos(theta_cw)
        Y_cw = r * np.sin(theta_cw)
        
        # Rotate counterclockwise field
        theta_ccw = theta - ccw_rotation
        X_ccw = r * np.cos(theta_ccw)
        Y_ccw = r * np.sin(theta_ccw)
        
        # Create interpolation indices
        x_cw_idx = ((X_cw + 1) / 2 * (dims[0] - 1)).astype(int)
        y_cw_idx = ((Y_cw + 1) / 2 * (dims[1] - 1)).astype(int)
        
        x_ccw_idx = ((X_ccw + 1) / 2 * (dims[0] - 1)).astype(int)
        y_ccw_idx = ((Y_ccw + 1) / 2 * (dims[1] - 1)).astype(int)
        
        # Clip indices to valid range
        x_cw_idx = np.clip(x_cw_idx, 0, dims[0] - 1)
        y_cw_idx = np.clip(y_cw_idx, 0, dims[1] - 1)
        
        x_ccw_idx = np.clip(x_ccw_idx, 0, dims[0] - 1)
        y_ccw_idx = np.clip(y_ccw_idx, 0, dims[1] - 1)
        
        # Create rotated fields
        rotated_cw = np.zeros_like(self.clockwise_field)
        rotated_ccw = np.zeros_like(self.counterclockwise_field)
        
        # Apply rotation through interpolation
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    # Calculate source indices
                    src_i_cw = x_cw_idx[i, j, k]
                    src_j_cw = y_cw_idx[i, j, k]
                    
                    src_i_ccw = x_ccw_idx[i, j, k]
                    src_j_ccw = y_ccw_idx[i, j, k]
                    
                    # Apply rotations
                    rotated_cw[i, j, k] = self.clockwise_field[src_i_cw, src_j_cw, k]
                    rotated_ccw[i, j, k] = self.counterclockwise_field[src_i_ccw, src_j_ccw, k]
        
        # Update component fields
        self.clockwise_field = rotated_cw
        self.counterclockwise_field = rotated_ccw
        
        # Recombine fields
        self.recombine_fields()
        
        # Update rotation counter
        self.rotation_steps += 1
        
        # Periodically adjust harmonization
        if self.rotation_steps % 10 == 0:
            self.harmonize_fields()
    
    def recombine_fields(self) -> None:
        """Recombine the counter-rotating fields into the main field."""
        if self.toroidal_field is None or self.clockwise_field is None or self.counterclockwise_field is None:
            return
        
        # Simple recombination - average the fields
        combined_field = (self.clockwise_field + self.counterclockwise_field) / 2
        
        # Apply interference pattern in the center
        dims = self.toroidal_field.data.shape
        
        # Create coordinate grids
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from torus center
        dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Distance from torus ring
        dist_from_center_axis = np.sqrt(X**2 + Y**2)
        dist_from_torus_ring = np.sqrt(
            (dist_from_center_axis - self.toroidal_field.major_radius)**2 + Z**2
        )
        
        # Create interference pattern in the core
        core_mask = dist_from_torus_ring < (self.toroidal_field.minor_radius * 0.5)
        
        if np.any(core_mask):
            # Calculate interference pattern
            interference = (self.clockwise_field * self.counterclockwise_field) / \
                          (np.max(np.abs(self.clockwise_field)) * np.max(np.abs(self.counterclockwise_field)))
            
            # Apply interference in core
            combined_field[core_mask] += interference[core_mask] * LAMBDA
        
        # Update the main field
        self.toroidal_field.data = combined_field
        
        # Update field metrics
        self.toroidal_field.calculate_flow_metrics()
        self.calculate_stability_metrics()
    
    def calculate_stability_metrics(self) -> Dict[str, float]:
        """
        Calculate stability metrics for the counter-rotating system.
        
        Returns:
            Dictionary of stability metrics
        """
        if self.toroidal_field is None or self.clockwise_field is None or self.counterclockwise_field is None:
            return {"stability_factor": 0.0, "harmonization": 0.0}
        
        # Calculate balance between the two fields
        cw_energy = np.sum(np.abs(self.clockwise_field))
        ccw_energy = np.sum(np.abs(self.counterclockwise_field))
        
        if cw_energy > 0 and ccw_energy > 0:
            energy_balance = min(cw_energy, ccw_energy) / max(cw_energy, ccw_energy)
        else:
            energy_balance = 0.0
        
        # Calculate phase coherence between fields
        phase_coherence = self._calculate_phase_coherence()
        
        # Calculate overall stability factor
        self.stability_factor = (energy_balance + self.toroidal_field.coherence) / 2
        
        # Calculate harmonization level
        self.harmonization = phase_coherence
        
        return {
            "stability_factor": self.stability_factor,
            "energy_balance": energy_balance,
            "phase_coherence": phase_coherence,
            "harmonization": self.harmonization
        }
    
    def _calculate_phase_coherence(self) -> float:
        """
        Calculate phase coherence between the counter-rotating fields.
        
        Returns:
            Phase coherence value (0.0-1.0)
        """
        # Skip if fields not initialized
        if self.clockwise_field is None or self.counterclockwise_field is None:
            return 0.0
        
        # Calculate FFT of both fields
        fft_cw = np.fft.fftn(self.clockwise_field)
        fft_ccw = np.fft.fftn(self.counterclockwise_field)
        
        # Extract phase information
        phase_cw = np.angle(fft_cw)
        phase_ccw = np.angle(fft_ccw)
        
        # Calculate phase difference - should be opposite for counter-rotating
        phase_diff = np.abs(np.abs(phase_cw - phase_ccw) - np.pi)
        
        # Perfect counter-rotation would have phase difference of pi
        coherence = 1.0 - np.mean(phase_diff) / np.pi
        
        return coherence
    
    def harmonize_fields(self, harmonization_factor: float = 0.2) -> float:
        """
        Adjust phase relationships to improve harmonization between fields.
        
        Args:
            harmonization_factor: Strength of harmonization adjustment (0.0-1.0)
            
        Returns:
            New harmonization level
        """
        if self.toroidal_field is None or self.clockwise_field is None or self.counterclockwise_field is None:
            return 0.0
        
        # Calculate current phase coherence
        current_harmonization = self.harmonization
        
        # Only harmonize if needed
        if current_harmonization > 0.9:
            return current_harmonization
        
        # Calculate FFT of both fields
        fft_cw = np.fft.fftn(self.clockwise_field)
        fft_ccw = np.fft.fftn(self.counterclockwise_field)
        
        # Extract magnitude and phase
        mag_cw = np.abs(fft_cw)
        phase_cw = np.angle(fft_cw)
        
        mag_ccw = np.abs(fft_ccw)
        phase_ccw = np.angle(fft_ccw)
        
        # Target phase difference - perfect counter-rotation
        target_diff = np.pi  # 180 degrees
        
        # Calculate current phase difference
        current_diff = np.abs(phase_cw - phase_ccw)
        
        # Adjust phase of counterclockwise field to move toward target
        adjustment = (target_diff - current_diff) * harmonization_factor
        new_phase_ccw = phase_ccw + adjustment
        
        # Reconstruct FFT with new phase
        new_fft_ccw = mag_ccw * np.exp(1j * new_phase_ccw)
        
        # Convert back to spatial domain
        new_ccw_field = np.real(np.fft.ifftn(new_fft_ccw))
        
        # Update counterclockwise field
        self.counterclockwise_field = new_ccw_field
        
        # Recombine fields
        self.recombine_fields()
        
        # Calculate new harmonization
        self.calculate_stability_metrics()
        
        return self.harmonization
    
    def perform_stability_cycle(self, steps: int = 20, time_factor: float = 0.1) -> float:
        """
        Perform a full stability cycle with counter-rotation.
        
        Args:
            steps: Number of rotation steps in the cycle
            time_factor: Time scaling factor for each step
            
        Returns:
            Final stability factor
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Initial stability
        initial_stability = self.stability_factor
        
        # Perform rotation steps
        for step in range(steps):
            self.rotate_components(time_factor)
            
            # Periodically adjust harmonization
            if step % 5 == 0:
                self.harmonize_fields()
        
        # Final harmonization
        self.harmonize_fields(harmonization_factor=0.5)
        
        # Calculate final stability
        self.calculate_stability_metrics()
        
        return self.stability_factor
    
    def create_dimensional_stability(self, target_stability: float = 0.9) -> float:
        """
        Create dimensional stability through counter-rotation and harmonization.
        
        Args:
            target_stability: Target stability factor (0.0-1.0)
            
        Returns:
            Final stability factor
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Current stability
        current_stability = self.stability_factor
        
        # Only proceed if needed
        if current_stability >= target_stability:
            return current_stability
        
        # Multiple cycles may be needed
        cycles = 0
        max_cycles = 5
        
        while current_stability < target_stability and cycles < max_cycles:
            # Perform stability cycle
            self.perform_stability_cycle()
            
            # Check new stability
            current_stability = self.stability_factor
            cycles += 1
        
        return current_stability


def demo_counter_rotation():
    """
    Demonstrate the functionality of counter-rotating fields.
    
    Returns:
        The created counter-rotating field system
    """
    from .toroidal_field import ToroidalField
    
    # Create a toroidal field
    print("Creating toroidal field...")
    field = ToroidalField(dimensions=(32, 32, 32), frequency_name='unity')
    
    # Print initial metrics
    metrics = field.calculate_flow_metrics()
    print("\nInitial field metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Create counter-rotating field system
    print("\nCreating counter-rotating field system...")
    cr_field = CounterRotatingField(toroidal_field=field, initial_separation=0.5)
    
    # Print initial stability
    stability_metrics = cr_field.calculate_stability_metrics()
    print("\nInitial stability metrics:")
    for name, value in stability_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Perform rotation steps
    print("\nPerforming counter-rotation steps...")
    cr_field.rotate_components(time_factor=0.2)
    
    # Recalculate metrics
    stability_metrics = cr_field.calculate_stability_metrics()
    print("\nStability after rotation:")
    for name, value in stability_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Harmonize fields
    print("\nHarmonizing counter-rotating fields...")
    new_harmonization = cr_field.harmonize_fields(harmonization_factor=0.5)
    print(f"New harmonization level: {new_harmonization:.4f}")
    
    # Perform complete stability cycle
    print("\nPerforming complete stability cycle...")
    final_stability = cr_field.perform_stability_cycle()
    print(f"Final stability factor: {final_stability:.4f}")
    
    # Create dimensional stability
    print("\nCreating dimensional stability...")
    optimal_stability = cr_field.create_dimensional_stability(target_stability=0.9)
    print(f"Optimal stability factor: {optimal_stability:.4f}")
    
    return cr_field

if __name__ == "__main__":
    cr_field = demo_counter_rotation()