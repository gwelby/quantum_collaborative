"""
CASCADE⚡𓂧φ∞ Toroidal Field Engine

Implements quantum field systems based on toroidal energy flow principles,
enabling balanced input/output cycles and self-sustaining energy patterns.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

# Import phi constants
import sys
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

class ToroidalFieldEngine:
    """
    Toroidal Field dynamics engine implementing balanced energy flow and
    self-sustaining patterns based on phi-harmonic principles.
    """
    
    def __init__(self, 
                 major_radius: float = PHI, 
                 minor_radius: float = LAMBDA,
                 base_frequency: float = SACRED_FREQUENCIES['unity']):
        """
        Initialize the toroidal field engine.
        
        Args:
            major_radius: Torus major radius (default: PHI)
            minor_radius: Torus minor radius (default: LAMBDA)
            base_frequency: Base frequency in Hz (default: 432)
        """
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.base_frequency = base_frequency
        self.coherence_metrics = {
            'overall': 0.0,
            'phi_alignment': 0.0,
            'flow_balance': 0.0,
            'circulation': 0.0
        }
    
    def generate_field(self, 
                     width: int, 
                     height: int, 
                     depth: int, 
                     time_factor: float = 0.0) -> np.ndarray:
        """
        Generate a 3D quantum field using Toroidal Field Dynamics.
        
        Args:
            width: Width of the field
            height: Height of the field
            depth: Depth of the field
            time_factor: Time factor for animation
            
        Returns:
            3D NumPy array with toroidal field values
        """
        # Create coordinate grids
        x = np.linspace(-1.0, 1.0, width)
        y = np.linspace(-1.0, 1.0, height)
        z = np.linspace(-1.0, 1.0, depth)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert to toroidal coordinates
        # Distance from the circle in the xy-plane with radius R
        distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - self.major_radius)**2 + Z**2)
        
        # Normalized distance (0 at torus surface, 1 at maximum distance)
        torus_distance = distance_from_ring / self.minor_radius
        
        # Azimuthal angle around the z-axis
        theta = np.arctan2(Y, X)  # 0 to 2π
        
        # Poloidal angle around the torus ring
        poloidal_angle = np.arctan2(Z, np.sqrt(X**2 + Y**2) - self.major_radius)  # 0 to 2π
        
        # Create toroidal flow pattern
        poloidal_flow = poloidal_angle * PHI
        toroidal_flow = theta * LAMBDA
        time_component = time_factor * PHI * LAMBDA
        
        # Create self-sustaining energy pattern with balanced in/out cycles
        # Using phi-harmonic resonance in the toroidal structure
        freq_factor = self.base_frequency / 1000.0
        
        # Combine flows with phi-weighted balance
        inflow = np.sin(poloidal_flow + time_component) * PHI
        circulation = np.cos(toroidal_flow + time_component) * LAMBDA
        
        # Create interference pattern that self-sustains
        field = (inflow * circulation) * np.exp(-torus_distance * LAMBDA)
        
        # Add phi-harmonic resonance inside the torus body
        mask = torus_distance < 1.0
        resonance = np.sin(torus_distance * PHI * PHI + time_component) * (1.0 - torus_distance)
        field[mask] += resonance[mask] * 0.2
        
        # Normalize field
        field = field / np.max(np.abs(field))
        
        return field
    
    def calculate_coherence(self, field_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate the coherence metrics of a toroidal quantum field.
        
        Args:
            field_data: 3D NumPy array containing the field
            
        Returns:
            Dictionary with coherence metrics
        """
        # Calculate field gradients
        grad_x, grad_y, grad_z = np.gradient(field_data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Calculate vorticity (curl)
        curl_x = np.gradient(grad_z, axis=1) - np.gradient(grad_y, axis=2)
        curl_y = np.gradient(grad_x, axis=2) - np.gradient(grad_z, axis=0)
        curl_z = np.gradient(grad_y, axis=0) - np.gradient(grad_x, axis=1)
        curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        
        # Calculate divergence
        div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1) + np.gradient(grad_z, axis=2)
        
        # Calculate phi alignment
        phi_values = np.linspace(0, 5, 6)
        phi_powers = [PHI ** p for p in phi_values]
        
        phi_distances = np.zeros_like(field_data)
        for p in phi_powers:
            phi_distances = np.minimum(phi_distances, np.abs(field_data - p))
        
        phi_alignment = 1.0 - np.mean(phi_distances) / PHI
        
        # Calculate flow balance (input/output cycle)
        flow_balance = 1.0 - np.mean(np.abs(div)) / (np.mean(grad_mag) + 1e-10)
        
        # Calculate circulation metric
        circulation = 1.0 - np.std(curl_mag) / (np.mean(curl_mag) + 1e-10)
        
        # Calculate overall coherence with phi-weighted components
        overall = (phi_alignment * 0.5 * PHI + 
                  flow_balance * 0.3 * PHI +
                  circulation * 0.2 * LAMBDA)
        
        # Ensure values are in [0, 1] range
        overall = max(0.0, min(1.0, overall))
        phi_alignment = max(0.0, min(1.0, phi_alignment))
        flow_balance = max(0.0, min(1.0, flow_balance))
        circulation = max(0.0, min(1.0, circulation))
        
        # Store and return coherence metrics
        self.coherence_metrics = {
            'overall': overall,
            'phi_alignment': phi_alignment,
            'flow_balance': flow_balance,
            'circulation': circulation
        }
        
        return self.coherence_metrics