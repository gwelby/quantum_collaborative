"""
Compression and Expansion Cycles

This module implements field coherence amplification through toroidal 
compression/expansion cycles, enabling energy conservation and field sustainability.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import math

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.constants import GEOMETRIC_CONSTANTS
from .toroidal_field import ToroidalField

class CompressionCycle:
    """
    Implements field compression within a toroidal structure to amplify coherence.
    
    Attributes:
        toroidal_field: The toroidal field this cycle operates on
        compression_ratio: Current compression ratio
        min_radius: Minimum radius allowed for compression
        max_radius: Maximum radius allowed for expansion
        cycle_phase: Current phase in the compression cycle (0.0-1.0)
        energy_conservation: Energy conservation factor during compression
        coherence_gain: Coherence gain from compression
    """
    
    def __init__(self, 
                 toroidal_field: Optional[ToroidalField] = None,
                 min_compression: float = 0.5,
                 max_expansion: float = 1.5,
                 energy_conservation: float = 0.95):
        """
        Initialize a compression cycle.
        
        Args:
            toroidal_field: The toroidal field to connect
            min_compression: Minimum compression ratio (< 1.0)
            max_expansion: Maximum expansion ratio (> 1.0)
            energy_conservation: Energy conservation factor (0.0-1.0)
        """
        self.toroidal_field = toroidal_field
        
        # Ensure ratios are valid
        self.min_compression = max(0.1, min(min_compression, 0.9))
        self.max_expansion = max(1.1, min(max_expansion, 2.0))
        
        # Current compression ratio (1.0 = no compression)
        self.compression_ratio = 1.0
        
        # Store original radii if field is connected
        if toroidal_field is not None:
            self.original_major_radius = toroidal_field.major_radius
            self.original_minor_radius = toroidal_field.minor_radius
            self.min_radius = self.original_minor_radius * self.min_compression
            self.max_radius = self.original_minor_radius * self.max_expansion
        else:
            self.original_major_radius = 0.0
            self.original_minor_radius = 0.0
            self.min_radius = 0.0
            self.max_radius = 0.0
        
        # Cycle properties
        self.cycle_phase = 0.0  # 0.0 to 1.0
        self.energy_conservation = energy_conservation
        self.coherence_gain = 0.0
    
    def connect_field(self, toroidal_field: ToroidalField) -> None:
        """
        Connect to a toroidal field.
        
        Args:
            toroidal_field: The toroidal field to connect
        """
        self.toroidal_field = toroidal_field
        self.original_major_radius = toroidal_field.major_radius
        self.original_minor_radius = toroidal_field.minor_radius
        self.min_radius = self.original_minor_radius * self.min_compression
        self.max_radius = self.original_minor_radius * self.max_expansion
    
    def set_compression(self, ratio: float) -> None:
        """
        Set a specific compression ratio.
        
        Args:
            ratio: Compression ratio (min_compression to max_expansion)
        """
        if self.toroidal_field is None:
            return
        
        # Validate ratio
        ratio = max(self.min_compression, min(ratio, self.max_expansion))
        
        # Store current ratio
        self.compression_ratio = ratio
        
        # Calculate new radii
        new_minor_radius = self.original_minor_radius * ratio
        
        # Calculate new major radius to conserve total volume
        if ratio > 0:
            # Approximate volume conservation for torus
            # V = 2π² * R * r²
            volume_ratio = 1.0 / (ratio * ratio)
            
            # Apply energy conservation factor
            volume_ratio = volume_ratio * self.energy_conservation + (1 - self.energy_conservation)
            
            new_major_radius = self.original_major_radius * volume_ratio
            
            # Ensure major radius is always larger than minor
            new_major_radius = max(new_major_radius, new_minor_radius * 1.5)
        else:
            new_major_radius = self.original_major_radius
        
        # Apply new radii to field
        self.toroidal_field.major_radius = new_major_radius
        self.toroidal_field.minor_radius = new_minor_radius
        
        # Reinitialize field with new geometry
        self.toroidal_field.initialize_toroidal_field()
    
    def compress(self, amount: float = 0.1) -> float:
        """
        Compress the field by a specified amount.
        
        Args:
            amount: Amount to compress (0.0-1.0)
            
        Returns:
            New compression ratio
        """
        if self.toroidal_field is None:
            return 1.0
        
        # Calculate new ratio
        new_ratio = max(self.min_compression, self.compression_ratio - amount)
        
        # Apply compression
        self.set_compression(new_ratio)
        
        # Calculate coherence gain
        old_coherence = self.toroidal_field.coherence
        self.toroidal_field.calculate_flow_metrics()
        new_coherence = self.toroidal_field.coherence
        
        self.coherence_gain = new_coherence - old_coherence
        
        return new_ratio
    
    def expand(self, amount: float = 0.1) -> float:
        """
        Expand the field by a specified amount.
        
        Args:
            amount: Amount to expand (0.0-1.0)
            
        Returns:
            New compression ratio
        """
        if self.toroidal_field is None:
            return 1.0
        
        # Calculate new ratio
        new_ratio = min(self.max_expansion, self.compression_ratio + amount)
        
        # Apply expansion
        self.set_compression(new_ratio)
        
        # Calculate coherence change
        old_coherence = self.toroidal_field.coherence
        self.toroidal_field.calculate_flow_metrics()
        new_coherence = self.toroidal_field.coherence
        
        self.coherence_gain = new_coherence - old_coherence
        
        return new_ratio
    
    def reset(self) -> None:
        """Reset to original dimensions."""
        if self.toroidal_field is None:
            return
        
        self.set_compression(1.0)
        self.cycle_phase = 0.0
    
    def perform_cycle_step(self, step_size: float = 0.05) -> float:
        """
        Perform one step in a compression-expansion cycle.
        
        Args:
            step_size: Size of the step in the cycle (0.0-1.0)
            
        Returns:
            Current coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Update cycle phase
        self.cycle_phase = (self.cycle_phase + step_size) % 1.0
        
        # Calculate target compression based on phi-harmonic oscillation
        # Full cycle: 1.0 -> min_compression -> 1.0 -> max_expansion -> 1.0
        
        if self.cycle_phase < 0.25:
            # Compress phase
            cycle_progress = self.cycle_phase / 0.25
            target_ratio = 1.0 - (1.0 - self.min_compression) * cycle_progress
            
        elif self.cycle_phase < 0.5:
            # Decompress phase
            cycle_progress = (self.cycle_phase - 0.25) / 0.25
            target_ratio = self.min_compression + (1.0 - self.min_compression) * cycle_progress
            
        elif self.cycle_phase < 0.75:
            # Expand phase
            cycle_progress = (self.cycle_phase - 0.5) / 0.25
            target_ratio = 1.0 + (self.max_expansion - 1.0) * cycle_progress
            
        else:
            # Contract phase
            cycle_progress = (self.cycle_phase - 0.75) / 0.25
            target_ratio = self.max_expansion - (self.max_expansion - 1.0) * cycle_progress
        
        # Set compression to target
        self.set_compression(target_ratio)
        
        # Calculate coherence gain
        old_coherence = self.toroidal_field.coherence
        self.toroidal_field.calculate_flow_metrics()
        new_coherence = self.toroidal_field.coherence
        
        self.coherence_gain = new_coherence - old_coherence
        
        return new_coherence
    
    def perform_complete_cycle(self, steps: int = 20) -> float:
        """
        Perform a complete compression-expansion cycle.
        
        Args:
            steps: Number of steps in the cycle
            
        Returns:
            Final coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Store initial coherence
        initial_coherence = self.toroidal_field.coherence
        
        # Perform cycle
        for _ in range(steps):
            step_size = 1.0 / steps
            self.perform_cycle_step(step_size)
        
        # Reset phase
        self.cycle_phase = 0.0
        
        # Ensure we end at ratio 1.0
        self.set_compression(1.0)
        
        # Calculate final coherence
        self.toroidal_field.calculate_flow_metrics()
        final_coherence = self.toroidal_field.coherence
        
        # Calculate cycle efficiency
        coherence_gain = final_coherence - initial_coherence
        self.coherence_gain = coherence_gain
        
        return final_coherence
    
    def optimize_field(self, target_coherence: float = 0.9, max_cycles: int = 5) -> float:
        """
        Optimize field coherence by performing multiple cycles.
        
        Args:
            target_coherence: Target coherence level
            max_cycles: Maximum number of cycles to perform
            
        Returns:
            Final coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        current_coherence = self.toroidal_field.coherence
        
        # Only optimize if needed
        if current_coherence >= target_coherence:
            return current_coherence
        
        # Perform cycles until target is reached or max_cycles
        for i in range(max_cycles):
            self.perform_complete_cycle()
            current_coherence = self.toroidal_field.coherence
            
            if current_coherence >= target_coherence:
                break
        
        return current_coherence


class ExpansionCycle:
    """
    Implements phi-harmonic expansion and contraction of a toroidal field
    to create resonant energy patterns.
    
    Attributes:
        toroidal_field: The toroidal field this cycle operates on
        expansion_phases: List of expansion phases in the cycle
        phi_ratio_sequence: Sequence of phi-related expansion ratios
        current_phase: Current phase in the expansion cycle
        energy_amplification: Energy amplification factor during cycle
        coherence_improvement: Coherence improvement from cycle
    """
    
    def __init__(self, 
                 toroidal_field: Optional[ToroidalField] = None,
                 phi_based_expansion: bool = True):
        """
        Initialize an expansion cycle.
        
        Args:
            toroidal_field: The toroidal field to connect
            phi_based_expansion: Whether to use phi-based expansion ratios
        """
        self.toroidal_field = toroidal_field
        
        # Define cycle phases
        if phi_based_expansion:
            # Phi-based expansion ratios
            self.phi_ratio_sequence = [
                1.0,               # Base state
                LAMBDA,            # Contraction to phi complement
                1.0,               # Return to base
                PHI,               # Expansion to phi
                1.0,               # Return to base
                PHI_PHI / PHI,     # Expansion to second level
                1.0                # Return to base
            ]
        else:
            # Simple expansion ratios
            self.phi_ratio_sequence = [
                1.0,   # Base state
                0.8,   # Slight contraction
                1.0,   # Return to base
                1.2,   # Slight expansion
                1.0,   # Return to base
                1.4,   # Larger expansion
                1.0    # Return to base
            ]
        
        # Current state
        self.expansion_phases = len(self.phi_ratio_sequence)
        self.current_phase = 0
        self.energy_amplification = 1.0
        self.coherence_improvement = 0.0
        
        # Store original field dimensions if connected
        if toroidal_field is not None:
            self.original_dims = toroidal_field.dimensions
        else:
            self.original_dims = (0, 0, 0)
    
    def connect_field(self, toroidal_field: ToroidalField) -> None:
        """
        Connect to a toroidal field.
        
        Args:
            toroidal_field: The toroidal field to connect
        """
        self.toroidal_field = toroidal_field
        self.original_dims = toroidal_field.dimensions
    
    def apply_expansion_phase(self, phase_index: Optional[int] = None) -> None:
        """
        Apply a specific expansion phase to the field.
        
        Args:
            phase_index: Index of the phase to apply (None for next phase)
        """
        if self.toroidal_field is None:
            return
        
        # Use next phase if not specified
        if phase_index is None:
            phase_index = self.current_phase
            # Advance to next phase
            self.current_phase = (self.current_phase + 1) % self.expansion_phases
        else:
            # Validate phase index
            phase_index = phase_index % self.expansion_phases
            self.current_phase = phase_index
        
        # Get expansion ratio for this phase
        expansion_ratio = self.phi_ratio_sequence[phase_index]
        
        # Apply field transformation
        self._expand_field(expansion_ratio)
    
    def _expand_field(self, ratio: float) -> None:
        """
        Expand or contract the field by transforming it.
        
        Args:
            ratio: Expansion ratio to apply
        """
        if self.toroidal_field is None:
            return
        
        # Get current field data
        field_data = self.toroidal_field.data
        
        # Get dimensions
        dims = field_data.shape
        
        # Create coordinate grids for original space
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create scaled coordinates
        X_scaled = X / ratio
        Y_scaled = Y / ratio
        Z_scaled = Z / ratio
        
        # Create new field data
        new_field = np.zeros_like(field_data)
        
        # Set value at each point by sampling original field at scaled coordinates
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    # Get scaled coordinates
                    x_scaled = X_scaled[i, j, k]
                    y_scaled = Y_scaled[i, j, k]
                    z_scaled = Z_scaled[i, j, k]
                    
                    # Skip if outside original boundaries
                    if abs(x_scaled) > 1 or abs(y_scaled) > 1 or abs(z_scaled) > 1:
                        continue
                    
                    # Map to indices in original field
                    x_idx = int((x_scaled + 1) * 0.5 * (dims[0] - 1))
                    y_idx = int((y_scaled + 1) * 0.5 * (dims[1] - 1))
                    z_idx = int((z_scaled + 1) * 0.5 * (dims[2] - 1))
                    
                    # Get value from original field
                    new_field[i, j, k] = field_data[x_idx, y_idx, z_idx]
        
        # Apply energy scaling - conserve or amplify based on ratio
        if ratio < 1.0:
            # Contraction - concentrate energy
            energy_factor = 1.0 / ratio
        else:
            # Expansion - distribute energy
            energy_factor = 1.0
        
        # Scale field values
        new_field *= energy_factor
        
        # Update field data
        self.toroidal_field.data = new_field
        
        # Update metrics
        old_coherence = self.toroidal_field.coherence
        self.toroidal_field.calculate_flow_metrics()
        
        # Calculate improvement
        self.coherence_improvement = self.toroidal_field.coherence - old_coherence
        self.energy_amplification = energy_factor
    
    def perform_complete_cycle(self) -> float:
        """
        Perform a complete expansion cycle through all phases.
        
        Returns:
            Final coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Store initial coherence
        initial_coherence = self.toroidal_field.coherence
        
        # Reset to first phase
        self.current_phase = 0
        
        # Perform each phase in sequence
        for phase_index in range(self.expansion_phases):
            self.apply_expansion_phase(phase_index)
        
        # Ensure we end with ratio 1.0
        self._expand_field(1.0)
        
        # Calculate improvement
        self.toroidal_field.calculate_flow_metrics()
        final_coherence = self.toroidal_field.coherence
        
        self.coherence_improvement = final_coherence - initial_coherence
        
        return final_coherence
    
    def optimize_with_phi_cycles(self, target_coherence: float = 0.9, max_cycles: int = 3) -> float:
        """
        Optimize field coherence using phi-harmonic cycles.
        
        Args:
            target_coherence: Target coherence level
            max_cycles: Maximum number of cycles to perform
            
        Returns:
            Final coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        # Store initial coherence
        current_coherence = self.toroidal_field.coherence
        
        # Only optimize if needed
        if current_coherence >= target_coherence:
            return current_coherence
        
        # Perform cycles until target reached or max_cycles
        for cycle in range(max_cycles):
            # Perform complete cycle
            self.perform_complete_cycle()
            
            # Check current coherence
            current_coherence = self.toroidal_field.coherence
            
            if current_coherence >= target_coherence:
                break
        
        return current_coherence


def demo_compression_cycle():
    """
    Demonstrate the functionality of compression and expansion cycles.
    
    Returns:
        The created compression cycle
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
    
    # Create compression cycle
    print("\nCreating compression cycle...")
    cycle = CompressionCycle(toroidal_field=field)
    
    # Compress field
    print("\nCompressing field...")
    compression_ratio = cycle.compress(0.2)
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"Field coherence: {field.coherence:.4f}")
    print(f"Coherence gain: {cycle.coherence_gain:.4f}")
    
    # Expand back to original size
    print("\nExpanding field back to original size...")
    cycle.reset()
    print(f"Field coherence: {field.coherence:.4f}")
    
    # Perform complete cycle
    print("\nPerforming complete compression-expansion cycle...")
    final_coherence = cycle.perform_complete_cycle()
    print(f"Final coherence: {final_coherence:.4f}")
    print(f"Coherence gain from cycle: {cycle.coherence_gain:.4f}")
    
    # Create expansion cycle
    print("\nCreating phi-based expansion cycle...")
    exp_cycle = ExpansionCycle(toroidal_field=field)
    
    # Perform expansion cycle
    print("\nPerforming phi-harmonic expansion cycle...")
    phi_coherence = exp_cycle.perform_complete_cycle()
    print(f"Final coherence: {phi_coherence:.4f}")
    print(f"Coherence improvement: {exp_cycle.coherence_improvement:.4f}")
    
    return cycle, exp_cycle

if __name__ == "__main__":
    compression_cycle, expansion_cycle = demo_compression_cycle()