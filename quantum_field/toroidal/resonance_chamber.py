"""
Resonance Chamber Implementation

This module provides phi-harmonic resonance chambers within toroidal structures,
allowing for perfect field amplification and frequency entrainment.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import math

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.constants import FIELD_3D_HARMONICS, GEOMETRIC_CONSTANTS
from .toroidal_field import ToroidalField

class ResonanceEqualizer:
    """
    A system for equalizing and balancing resonant frequencies within a toroidal field.
    
    Attributes:
        frequency_bands: Dictionary of frequency bands to equalize
        band_strengths: Current strength of each frequency band
        optimal_ratios: The optimal phi-harmonic ratios between frequency bands
        equalization_curve: The current equalization curve
        balance_threshold: Threshold for considering the field balanced
    """
    
    def __init__(self, frequency_bands: Optional[Dict[str, float]] = None):
        """
        Initialize a resonance equalizer.
        
        Args:
            frequency_bands: Dictionary of frequency bands to equalize
                            (defaults to sacred frequencies)
        """
        # Use sacred frequencies as default
        if frequency_bands is None:
            self.frequency_bands = SACRED_FREQUENCIES
        else:
            self.frequency_bands = frequency_bands
            
        # Initialize band strengths at equal levels
        self.band_strengths = {name: 1.0 for name in self.frequency_bands}
        
        # Calculate optimal phi-harmonic ratios between bands
        self.optimal_ratios = {}
        sorted_bands = sorted(self.frequency_bands.items(), key=lambda x: x[1])
        
        for i, (name1, freq1) in enumerate(sorted_bands):
            self.optimal_ratios[name1] = {}
            for name2, freq2 in sorted_bands:
                if name1 != name2:
                    # Calculate phi-harmonic ratio
                    if freq2 > freq1:
                        harmonic_ratio = freq2 / freq1
                        nearest_phi_power = round(math.log(harmonic_ratio, PHI))
                        self.optimal_ratios[name1][name2] = PHI ** nearest_phi_power
                    else:
                        harmonic_ratio = freq1 / freq2
                        nearest_phi_power = round(math.log(harmonic_ratio, PHI))
                        self.optimal_ratios[name1][name2] = 1 / (PHI ** nearest_phi_power)
        
        # Initialize equalization curve - flat response
        self.equalization_curve = np.ones(100)
        
        # Balance threshold
        self.balance_threshold = 0.9
    
    def analyze_field_frequencies(self, field: ToroidalField) -> Dict[str, float]:
        """
        Analyze the strength of different frequency bands in a toroidal field.
        
        Args:
            field: The toroidal field to analyze
            
        Returns:
            Dictionary mapping frequency bands to their strengths
        """
        # Calculate FFT of field data
        fft_data = np.fft.fftn(field.data)
        fft_magnitude = np.abs(fft_data)
        
        # Normalize magnitude
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
        
        # Calculate frequency resolution
        freq_resolution = 1.0 / max(field.dimensions)
        
        # Analyze each frequency band
        band_strengths = {}
        
        for name, center_freq in self.frequency_bands.items():
            # Convert frequency to normalized units
            norm_freq = center_freq / 1000.0
            
            # Define band width proportional to phi
            band_width = LAMBDA * norm_freq
            
            # Calculate frequency range
            low_freq = norm_freq - band_width/2
            high_freq = norm_freq + band_width/2
            
            # Convert to FFT indices
            low_idx = max(1, int(low_freq / freq_resolution))
            high_idx = min(int(high_freq / freq_resolution), field.dimensions[0]//2)
            
            # Sum magnitude in this band
            band_sum = 0.0
            count = 0
            
            # Check all three dimensions
            for dim in range(3):
                # Create slice selectors for each dimension
                if dim == 0:
                    freq_slice = fft_magnitude[low_idx:high_idx, :, :]
                elif dim == 1:
                    freq_slice = fft_magnitude[:, low_idx:high_idx, :]
                else:  # dim == 2
                    freq_slice = fft_magnitude[:, :, low_idx:high_idx]
                
                band_sum += np.sum(freq_slice)
                count += freq_slice.size
            
            # Calculate average magnitude in band
            if count > 0:
                band_strengths[name] = band_sum / count
            else:
                band_strengths[name] = 0.0
        
        # Update current band strengths
        self.band_strengths = band_strengths
        
        return band_strengths
    
    def calculate_balance_metrics(self) -> Dict[str, float]:
        """
        Calculate balance metrics for current frequency distribution.
        
        Returns:
            Dictionary of balance metrics
        """
        metrics = {}
        
        # Overall band variance (lower is better)
        band_values = list(self.band_strengths.values())
        metrics["variance"] = np.var(band_values)
        
        # Phi-harmonic ratio compliance
        ratio_errors = []
        
        for name1, ratios in self.optimal_ratios.items():
            for name2, target_ratio in ratios.items():
                if self.band_strengths[name2] > 0:
                    actual_ratio = self.band_strengths[name1] / self.band_strengths[name2]
                    ratio_error = abs(actual_ratio - target_ratio) / target_ratio
                    ratio_errors.append(ratio_error)
        
        if ratio_errors:
            metrics["ratio_error"] = np.mean(ratio_errors)
        else:
            metrics["ratio_error"] = 0.0
        
        # Overall balance score (1.0 is perfect)
        metrics["balance_score"] = 1.0 - min(metrics["variance"] + metrics["ratio_error"], 1.0)
        
        return metrics
    
    def is_balanced(self) -> bool:
        """
        Check if the current frequency distribution is balanced.
        
        Returns:
            True if balanced, False otherwise
        """
        metrics = self.calculate_balance_metrics()
        return metrics["balance_score"] >= self.balance_threshold
    
    def calculate_equalization_curve(self) -> np.ndarray:
        """
        Calculate an equalization curve to balance the frequencies.
        
        Returns:
            Numpy array with equalization multipliers for different frequencies
        """
        # Start with flat response
        eq_curve = np.ones(100)
        
        # Calculate target strengths based on optimal ratios
        target_strengths = {name: 0.0 for name in self.band_strengths}
        
        # Use the most balanced frequency as reference
        metrics = self.calculate_balance_metrics()
        balance_score = metrics["balance_score"]
        
        if balance_score < self.balance_threshold:
            # Find the reference frequency (highest strength)
            ref_name = max(self.band_strengths.items(), key=lambda x: x[1])[0]
            ref_strength = self.band_strengths[ref_name]
            
            # Set targets for other frequencies
            for name in self.band_strengths:
                if name != ref_name and ref_name in self.optimal_ratios[name]:
                    target_ratio = self.optimal_ratios[name][ref_name]
                    target_strengths[name] = ref_strength * target_ratio
                else:
                    target_strengths[name] = ref_strength
            
            # Create equalization curve
            for name, freq in self.frequency_bands.items():
                # Calculate normalized frequency
                norm_freq = int((freq / 1000.0) * 100)
                norm_freq = min(max(norm_freq, 0), 99)
                
                # Calculate adjustment factor
                if self.band_strengths[name] > 0:
                    adjustment = target_strengths[name] / self.band_strengths[name]
                    
                    # Limit adjustment range
                    adjustment = min(max(adjustment, 0.5), 2.0)
                else:
                    adjustment = 1.0
                
                # Apply adjustment to EQ curve with phi-based smoothing
                eq_width = int(LAMBDA * 10)
                
                for i in range(max(0, norm_freq - eq_width), min(100, norm_freq + eq_width + 1)):
                    # Apply bell curve for smooth transition
                    distance = abs(i - norm_freq)
                    weight = math.exp(-(distance**2) / (2 * (eq_width/3)**2))
                    
                    # Blend with existing curve
                    eq_curve[i] = eq_curve[i] * (1 - weight) + adjustment * weight
        
        self.equalization_curve = eq_curve
        return eq_curve
    
    def apply_equalization(self, field: ToroidalField) -> None:
        """
        Apply equalization to balance frequencies in the field.
        
        Args:
            field: The toroidal field to equalize
        """
        # Analyze current frequencies
        self.analyze_field_frequencies(field)
        
        # Calculate equalization curve
        eq_curve = self.calculate_equalization_curve()
        
        # Apply equalization if needed
        if not self.is_balanced():
            # Calculate FFT of field data
            fft_data = np.fft.fftn(field.data)
            
            # Calculate frequency resolution
            freq_resolution = 1.0 / max(field.dimensions)
            
            # Apply equalization to FFT data
            for i in range(100):
                # Calculate actual frequency
                norm_freq = i / 100.0
                
                # Convert to FFT indices
                freq_idx = int(norm_freq / freq_resolution)
                freq_idx = min(freq_idx, field.dimensions[0]//2)
                
                # Skip DC component
                if freq_idx == 0:
                    continue
                
                # Get equalization factor
                eq_factor = eq_curve[i]
                
                # Apply to all three dimensions
                for dim in range(3):
                    if dim == 0:
                        # Use broadcasting to equalize all frequencies at this index
                        fft_data[freq_idx, :, :] *= eq_factor
                        if freq_idx > 0:  # Apply to negative frequencies too
                            fft_data[-freq_idx, :, :] *= eq_factor
                    elif dim == 1:
                        fft_data[:, freq_idx, :] *= eq_factor
                        if freq_idx > 0:
                            fft_data[:, -freq_idx, :] *= eq_factor
                    else:  # dim == 2
                        fft_data[:, :, freq_idx] *= eq_factor
                        if freq_idx > 0:
                            fft_data[:, :, -freq_idx] *= eq_factor
            
            # Convert back to spatial domain
            equalized_field = np.real(np.fft.ifftn(fft_data))
            
            # Update field data
            field.data = equalized_field
            
            # Recalculate field metrics
            field.calculate_flow_metrics()
            
            # Analyze new frequency distribution
            self.analyze_field_frequencies(field)
            new_metrics = self.calculate_balance_metrics()
            
            print("Applied frequency equalization:")
            print(f"  Balance score: {new_metrics['balance_score']:.4f}")
            print(f"  Field coherence: {field.coherence:.4f}")


class ResonanceChamber:
    """
    A system that creates phi-harmonic resonance chambers within toroidal structures.
    
    Attributes:
        toroidal_field: The toroidal field this chamber is connected to
        chamber_shape: Shape of the resonance chamber ('phi', 'cube', 'octahedron', 'sphere')
        chamber_size: Size of the chamber relative to the field
        active_frequency: Currently active frequency
        harmonic_pattern: Current harmonic pattern in the chamber
        equalizer: Frequency equalizer for the chamber
        energy_level: Current energy level in the chamber
        coherence: Coherence level of the chamber
    """
    
    def __init__(self, 
                 toroidal_field: Optional[ToroidalField] = None, 
                 chamber_shape: str = 'phi',
                 chamber_size: float = 0.3):
        """
        Initialize a resonance chamber.
        
        Args:
            toroidal_field: The toroidal field to connect to the chamber
            chamber_shape: Shape of the resonance chamber
            chamber_size: Size of the chamber relative to the field
        """
        self.toroidal_field = toroidal_field
        self.chamber_shape = chamber_shape
        self.chamber_size = chamber_size
        
        # Initialize with unity frequency
        self.active_frequency = SACRED_FREQUENCIES['unity']
        
        # Initialize harmonic pattern
        if toroidal_field is not None:
            self.harmonic_pattern = np.zeros_like(toroidal_field.data)
            self.create_chamber()
        else:
            self.harmonic_pattern = None
        
        # Create frequency equalizer
        self.equalizer = ResonanceEqualizer()
        
        # Initialize metrics
        self.energy_level = 0.0
        self.coherence = 0.0
    
    def connect_field(self, toroidal_field: ToroidalField) -> None:
        """
        Connect the chamber to a toroidal field.
        
        Args:
            toroidal_field: The toroidal field to connect
        """
        self.toroidal_field = toroidal_field
        self.harmonic_pattern = np.zeros_like(toroidal_field.data)
        self.create_chamber()
    
    def create_chamber(self) -> None:
        """Create the resonance chamber within the toroidal field."""
        if self.toroidal_field is None:
            return
        
        # Get field dimensions
        dims = self.toroidal_field.dimensions
        
        # Calculate chamber center (same as field center)
        center = [d // 2 for d in dims]
        
        # Calculate chamber size in voxels
        chamber_radius = int(min(dims) * self.chamber_size / 2)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create chamber mask based on shape
        if self.chamber_shape == 'phi':
            # Golden rectangle-based chamber
            aspect_ratio = PHI
            width = chamber_radius
            height = chamber_radius
            depth = chamber_radius / aspect_ratio
            
            chamber_mask = (
                (np.abs(X) <= width / (dims[0]/2)) & 
                (np.abs(Y) <= height / (dims[1]/2)) & 
                (np.abs(Z) <= depth / (dims[2]/2))
            )
            
        elif self.chamber_shape == 'sphere':
            # Spherical chamber
            distance = np.sqrt(X**2 + Y**2 + Z**2)
            chamber_mask = distance <= (chamber_radius / (dims[0]/2))
            
        elif self.chamber_shape == 'octahedron':
            # Octahedral chamber
            distance = np.abs(X) + np.abs(Y) + np.abs(Z)
            chamber_mask = distance <= (chamber_radius * 1.5 / (dims[0]/2))
            
        else:  # 'cube' or default
            # Cubic chamber
            chamber_mask = (
                (np.abs(X) <= chamber_radius / (dims[0]/2)) & 
                (np.abs(Y) <= chamber_radius / (dims[1]/2)) & 
                (np.abs(Z) <= chamber_radius / (dims[2]/2))
            )
        
        # Create harmonic pattern within chamber
        pattern = np.zeros_like(self.toroidal_field.data)
        
        # Frequency-based standing waves
        freq_factor = self.active_frequency / 100.0
        
        x_waves = np.sin(X * np.pi * freq_factor * PHI)
        y_waves = np.sin(Y * np.pi * freq_factor * PHI_PHI)
        z_waves = np.sin(Z * np.pi * freq_factor * LAMBDA)
        
        # Create phi-harmonic interference pattern
        pattern = (x_waves + y_waves + z_waves) / 3
        
        # Apply pattern only within chamber
        self.harmonic_pattern = np.zeros_like(pattern)
        self.harmonic_pattern[chamber_mask] = pattern[chamber_mask]
        
        # Calculate chamber metrics
        self.energy_level = np.sum(np.abs(self.harmonic_pattern))
        
        # Calculate coherence based on phi-harmonic alignments
        coherence_sample = self.harmonic_pattern[chamber_mask]
        if coherence_sample.size > 0:
            # Calculate gradient
            grad = np.gradient(coherence_sample)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(grad)) * PHI)
            
            # Calculate frequency purity
            fft = np.fft.fftn(coherence_sample)
            magnitude = np.abs(fft)
            
            # Calculate energy concentration
            total_energy = np.sum(magnitude)
            threshold = np.percentile(magnitude, 90)  # Top 10%
            high_energy = np.sum(magnitude[magnitude > threshold])
            
            energy_concentration = high_energy / total_energy if total_energy > 0 else 0
            
            # Combine metrics
            self.coherence = (smoothness + energy_concentration) / 2
        else:
            self.coherence = 0.0
    
    def tune_to_frequency(self, frequency_name: str) -> None:
        """
        Tune the resonance chamber to a specific sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency
        """
        if frequency_name in SACRED_FREQUENCIES:
            self.active_frequency = SACRED_FREQUENCIES[frequency_name]
        else:
            self.active_frequency = SACRED_FREQUENCIES['unity']  # Default
        
        # Recreate chamber with new frequency
        self.create_chamber()
        
        print(f"Resonance chamber tuned to {frequency_name} frequency ({self.active_frequency} Hz)")
        print(f"Chamber coherence: {self.coherence:.4f}")
    
    def apply_resonance(self, strength: float = 1.0) -> None:
        """
        Apply resonance effect from chamber to the connected toroidal field.
        
        Args:
            strength: Strength of the resonance effect (0.0-1.0)
        """
        if self.toroidal_field is None or self.harmonic_pattern is None:
            return
        
        # Scale strength by phi to ensure harmonic influence
        phi_strength = strength * LAMBDA
        
        # Apply harmonic pattern to field
        field_data = self.toroidal_field.data
        resonant_field = field_data * (1 - phi_strength) + self.harmonic_pattern * phi_strength
        
        # Update field
        self.toroidal_field.data = resonant_field
        
        # Recalculate field metrics
        self.toroidal_field.calculate_flow_metrics()
        
        # Equalize frequencies
        self.equalizer.apply_equalization(self.toroidal_field)
        
        print(f"Applied resonance with strength {strength:.2f}")
        print(f"Field coherence: {self.toroidal_field.coherence:.4f}")
    
    def amplify_coherence(self, target_coherence: float = 0.9) -> float:
        """
        Amplify the coherence of the connected field through resonance.
        
        Args:
            target_coherence: Target coherence level
            
        Returns:
            New coherence level
        """
        if self.toroidal_field is None:
            return 0.0
        
        current_coherence = self.toroidal_field.coherence
        
        # Only amplify if needed
        if current_coherence >= target_coherence:
            return current_coherence
        
        # Calculate required amplification
        coherence_gap = target_coherence - current_coherence
        amplification_strength = min(coherence_gap * 2, 1.0)
        
        # Apply resonance
        self.apply_resonance(strength=amplification_strength)
        
        # Optimize field coherence
        self.toroidal_field.optimize_coherence(target_coherence)
        
        return self.toroidal_field.coherence
    
    def create_phi_harmonic_standing_waves(self) -> None:
        """Create phi-harmonic standing waves in the resonance chamber."""
        if self.toroidal_field is None or self.harmonic_pattern is None:
            return
        
        # Get field dimensions
        dims = self.toroidal_field.dimensions
        
        # Create coordinate grids
        x = np.linspace(-1, 1, dims[0])
        y = np.linspace(-1, 1, dims[1])
        z = np.linspace(-1, 1, dims[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate chamber size in voxels
        chamber_radius = int(min(dims) * self.chamber_size / 2)
        
        # Calculate chamber mask based on current shape
        if self.chamber_shape == 'phi':
            # Golden rectangle-based chamber
            aspect_ratio = PHI
            width = chamber_radius
            height = chamber_radius
            depth = chamber_radius / aspect_ratio
            
            chamber_mask = (
                (np.abs(X) <= width / (dims[0]/2)) & 
                (np.abs(Y) <= height / (dims[1]/2)) & 
                (np.abs(Z) <= depth / (dims[2]/2))
            )
        elif self.chamber_shape == 'sphere':
            # Spherical chamber
            distance = np.sqrt(X**2 + Y**2 + Z**2)
            chamber_mask = distance <= (chamber_radius / (dims[0]/2))
        elif self.chamber_shape == 'octahedron':
            # Octahedral chamber
            distance = np.abs(X) + np.abs(Y) + np.abs(Z)
            chamber_mask = distance <= (chamber_radius * 1.5 / (dims[0]/2))
        else:  # 'cube' or default
            # Cubic chamber
            chamber_mask = (
                (np.abs(X) <= chamber_radius / (dims[0]/2)) & 
                (np.abs(Y) <= chamber_radius / (dims[1]/2)) & 
                (np.abs(Z) <= chamber_radius / (dims[2]/2))
            )
        
        # Create phi-harmonic standing waves
        pattern = np.zeros_like(self.toroidal_field.data)
        
        # Use phi-related frequencies in each dimension
        freq_x = self.active_frequency / 100.0
        freq_y = freq_x * PHI
        freq_z = freq_x * PHI_PHI
        
        # Generate complex standing wave patterns
        x_waves = np.sin(X * np.pi * freq_x) * np.cos(Y * np.pi * LAMBDA)
        y_waves = np.sin(Y * np.pi * freq_y) * np.cos(Z * np.pi * LAMBDA)
        z_waves = np.sin(Z * np.pi * freq_z) * np.cos(X * np.pi * LAMBDA)
        
        # Create phi-harmonic interference pattern
        pattern = (x_waves + y_waves + z_waves) / 3
        
        # Apply pattern only within chamber
        self.harmonic_pattern = np.zeros_like(pattern)
        self.harmonic_pattern[chamber_mask] = pattern[chamber_mask]
        
        # Recalculate chamber metrics
        self.energy_level = np.sum(np.abs(self.harmonic_pattern))
        
        # Apply to field
        self.apply_resonance(strength=0.618)  # Golden ratio complement
        
        print("Created phi-harmonic standing waves in resonance chamber")
        print(f"Chamber energy: {self.energy_level:.4f}, Coherence: {self.coherence:.4f}")

def demo_resonance_chamber():
    """
    Demonstrate the functionality of a resonance chamber.
    
    Returns:
        The created resonance chamber
    """
    from .toroidal_field import ToroidalField
    
    # Create a toroidal field
    print("Creating toroidal field...")
    field = ToroidalField(dimensions=(32, 32, 32), frequency_name='unity')
    
    # Create resonance chamber
    print("\nCreating resonance chamber...")
    chamber = ResonanceChamber(toroidal_field=field, chamber_shape='phi')
    
    # Print initial metrics
    print(f"Initial field coherence: {field.coherence:.4f}")
    print(f"Chamber coherence: {chamber.coherence:.4f}")
    
    # Tune to a different frequency
    print("\nTuning chamber to 'love' frequency...")
    chamber.tune_to_frequency('love')
    
    # Apply resonance
    print("\nApplying resonance effect...")
    chamber.apply_resonance(strength=0.5)
    
    # Create phi-harmonic standing waves
    print("\nCreating phi-harmonic standing waves...")
    chamber.create_phi_harmonic_standing_waves()
    
    # Amplify coherence
    print("\nAmplifying field coherence...")
    new_coherence = chamber.amplify_coherence(0.9)
    print(f"Final field coherence: {new_coherence:.4f}")
    
    return chamber

if __name__ == "__main__":
    chamber = demo_resonance_chamber()