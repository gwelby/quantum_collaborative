"""
Cymatic Field Module

Implementation of sound-based quantum field patterns that can directly influence
physical matter through cymatics principles and phi-harmonic resonance.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from ..toroidal import ToroidalField
import sys
sys.path.append('/mnt/d/projects/Python')
from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES, 
    calculate_phi_resonance, get_frequency_by_name
)

class CymaticField:
    """
    A field that uses sound frequencies to generate cymatic patterns capable of
    influencing physical matter through phi-harmonic resonance.
    
    The field creates standing wave patterns with precise frequency relationships
    based on phi-harmonic ratios, enabling direct manifestation of quantum patterns
    into physical reality.
    """
    
    def __init__(
        self, 
        base_frequency: float = SACRED_FREQUENCIES['unity'],
        dimensions: Tuple[int, int, int] = (32, 32, 32),
        resolution: float = 0.1,
        toroidal_base: Optional[ToroidalField] = None
    ):
        """
        Initialize a new cymatic field.
        
        Args:
            base_frequency: The fundamental frequency in Hz (default: 432Hz)
            dimensions: 3D grid dimensions for the field
            resolution: Spatial resolution of the grid
            toroidal_base: Optional toroidal field to use as base for the cymatic patterns
        """
        self.base_frequency = base_frequency
        self.dimensions = dimensions
        self.resolution = resolution
        self.toroidal_base = toroidal_base
        
        # Field properties
        self.amplitude = 1.0
        self.phase = 0.0
        self.coherence = 1.0
        self.pattern_complexity = 1.0
        
        # Initialize the frequency spectrum based on phi-harmonic relationships
        self.frequencies = self._initialize_frequencies()
        
        # Initialize the 3D grid for the cymatic field
        self.grid = np.zeros(dimensions, dtype=np.complex128)
        
        # Initialize field with base pattern
        self._initialize_field()
        
        # Material influence parameters
        self.material_coupling = {
            'water': 0.95,    # Water is highly responsive to cymatics
            'crystal': 0.85,  # Crystalline structures respond to specific frequencies
            'metal': 0.75,    # Metals respond to resonant frequencies
            'sand': 0.90,     # Sand creates clear cymatic patterns
            'plasma': 0.80    # Plasma can be shaped by electromagnetic frequencies
        }
        
        # Pattern memory stores previously generated stable patterns
        self.pattern_memory = {}
        
    def _initialize_frequencies(self) -> Dict[str, float]:
        """
        Initialize the frequency spectrum based on phi-harmonic relationships.
        
        Returns:
            Dictionary mapping frequency names to their values in Hz
        """
        frequencies = {
            'unity': self.base_frequency,
            'phi': self.base_frequency * PHI,
            'phi_squared': self.base_frequency * PHI * PHI,
            'phi_inverse': self.base_frequency * LAMBDA,
            'octave': self.base_frequency * 2.0,
            'octave_phi': self.base_frequency * 2.0 * PHI,
            'perfect_fifth': self.base_frequency * 1.5,
            'perfect_fourth': self.base_frequency * 4.0/3.0
        }
        
        # Add sacred frequencies normalized to our base frequency
        scale_factor = self.base_frequency / SACRED_FREQUENCIES['unity']
        for name, freq in SACRED_FREQUENCIES.items():
            frequencies[f'sacred_{name}'] = freq * scale_factor
            
        return frequencies
    
    def _initialize_field(self):
        """Initialize the cymatic field with a basic standing wave pattern"""
        if self.toroidal_base:
            # Use toroidal field as base if provided
            self._initialize_from_toroidal()
        else:
            # Create a basic 3D standing wave pattern
            self._initialize_basic_pattern()
    
    def _initialize_from_toroidal(self):
        """Initialize the cymatic field from a toroidal field base"""
        if not self.toroidal_base:
            return
            
        # Get the toroidal field data
        torus_data = self.toroidal_base.get_field_data()
        
        # Resample to match our dimensions if needed
        if torus_data.shape != self.dimensions:
            # Simple resampling for demonstration - in practice, would use proper interpolation
            x_scale = torus_data.shape[0] / self.dimensions[0]
            y_scale = torus_data.shape[1] / self.dimensions[1]
            z_scale = torus_data.shape[2] / self.dimensions[2]
            
            for i in range(self.dimensions[0]):
                for j in range(self.dimensions[1]):
                    for k in range(self.dimensions[2]):
                        ti = min(int(i * x_scale), torus_data.shape[0]-1)
                        tj = min(int(j * y_scale), torus_data.shape[1]-1)
                        tk = min(int(k * z_scale), torus_data.shape[2]-1)
                        
                        # Use the toroidal field value as amplitude
                        self.grid[i, j, k] = torus_data[ti, tj, tk]
        else:
            # Same dimensions, can use directly
            self.grid = torus_data.copy()
        
        # Apply frequency modulation to convert field values to wave patterns
        self._apply_frequency_modulation()
    
    def _initialize_basic_pattern(self):
        """Create a basic 3D standing wave pattern"""
        nx, ny, nz = self.dimensions
        
        # Create coordinate grids
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from center
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Create spherical standing wave pattern with phi-harmonic wavelength
        wavelength = self.resolution * PHI
        k = 2 * np.pi / wavelength  # Wave number
        
        # Standing wave with phi-harmonic modulation
        self.grid = np.sin(k * R) * np.exp(-R**2 / (PHI * 2))
        
        # Add phi-based harmonics to create complexity
        self.grid += 0.5 * np.sin(k * PHI * R) * np.exp(-R**2 / (PHI * 3))
        self.grid += 0.25 * np.sin(k * PHI * PHI * R) * np.exp(-R**2 / (PHI * 4))
        
        # Normalize
        self.grid = self.grid / np.max(np.abs(self.grid))
        
        # Convert to complex representation for phase information
        phase = np.angle(X + 1j*Y) * LAMBDA  # Phi-modulated phase
        self.grid = self.grid * np.exp(1j * phase)
    
    def _apply_frequency_modulation(self):
        """Apply frequency modulation to the field to create wavelike patterns"""
        nx, ny, nz = self.dimensions
        
        # Create coordinate grids
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert field values to amplitudes
        amplitudes = np.abs(self.grid)
        
        # Create phase based on toroidal field's inherent structure
        phase = np.angle(X + 1j*Y + 1j*Z * LAMBDA)
        
        # Apply frequency-dependent phase shift
        k = 2 * np.pi * self.base_frequency / 343.0  # Wave number (speed of sound ~343 m/s)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Standing wave pattern with phi-harmonic adjustments
        standing_wave = np.sin(k * r * PHI) * np.exp(1j * phase)
        
        # Combine amplitude from toroidal field with standing wave pattern
        self.grid = amplitudes * standing_wave
        
        # Normalize
        max_amp = np.max(np.abs(self.grid))
        if max_amp > 0:
            self.grid = self.grid / max_amp
    
    def set_frequency(self, frequency: float):
        """
        Set the base frequency for the cymatic field.
        
        Args:
            frequency: The fundamental frequency in Hz
        """
        self.base_frequency = frequency
        self.frequencies = self._initialize_frequencies()
        self._update_field()
    
    def set_frequency_by_name(self, name: str):
        """
        Set the base frequency using a sacred frequency name.
        
        Args:
            name: Name of the frequency (e.g., 'unity', 'love', 'truth')
        """
        frequency = get_frequency_by_name(name)
        if frequency:
            self.set_frequency(frequency)
    
    def _update_field(self):
        """Update the field patterns based on current parameters"""
        # Save current field state for blending
        old_field = self.grid.copy()
        
        # Reinitialize with new parameters
        if self.toroidal_base:
            self._initialize_from_toroidal()
        else:
            self._initialize_basic_pattern()
        
        # Blend with old field for smooth transition
        blend_factor = 0.3  # 30% new, 70% old
        self.grid = blend_factor * self.grid + (1 - blend_factor) * old_field
        
        # Apply amplitude modulation
        self.grid = self.grid * self.amplitude
    
    def apply_phi_harmonic_modulation(self, intensity: float = 1.0):
        """
        Apply phi-harmonic modulation to the field patterns.
        
        Args:
            intensity: The intensity of the phi modulation (0.0 to 1.0)
        """
        # Create phi-harmonic modulation pattern
        nx, ny, nz = self.dimensions
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Phi-harmonic distance function
        R = np.sqrt(X**2 + (Y**2 * PHI) + (Z**2 * PHI_PHI)) * intensity
        
        # Create modulation pattern with phi-harmonic wavelengths
        wavelength = self.resolution * PHI
        k = 2 * np.pi / wavelength
        
        # Phi-harmonic modulation
        mod_pattern = (
            np.sin(k * R) + 
            LAMBDA * np.sin(k * PHI * R) + 
            LAMBDA**2 * np.sin(k * PHI_PHI * R)
        ) / (1 + LAMBDA + LAMBDA**2)
        
        # Apply modulation to amplitude
        amplitudes = np.abs(self.grid)
        phases = np.angle(self.grid)
        
        # Modulate amplitudes
        modulated_amp = amplitudes * (1.0 + mod_pattern * intensity * LAMBDA)
        
        # Recreate complex field
        self.grid = modulated_amp * np.exp(1j * phases)
        
        # Update coherence based on phi-harmony of the modulation
        self.coherence = self._calculate_coherence()
    
    def calculate_material_influence(self, material: str) -> float:
        """
        Calculate the influence strength of the cymatic field on a specific material.
        
        Args:
            material: The type of material ('water', 'crystal', 'metal', etc.)
            
        Returns:
            A value between 0 and 1 representing influence strength
        """
        # Get the coupling factor for this material
        coupling = self.material_coupling.get(material, 0.5)
        
        # Calculate coherence of the field
        coherence = self._calculate_coherence()
        
        # Calculate resonance between field frequency and material's natural resonance
        material_resonances = {
            'water': 432.0,       # Water responds well to unity frequency
            'crystal': 528.0,     # Crystal responds to creation frequency
            'metal': 672.0,       # Metal responds to truth frequency
            'sand': 528.0,        # Sand responds to creation patterns
            'plasma': 768.0,      # Plasma responds to oneness frequency
        }
        
        material_resonance = material_resonances.get(material, self.base_frequency)
        frequency_match = calculate_phi_resonance(self.base_frequency, material_resonance)
        
        # Calculate total influence (phi-weighted average)
        influence = (
            coherence * PHI + 
            coupling * 1.0 + 
            frequency_match * LAMBDA
        ) / (PHI + 1.0 + LAMBDA)
        
        return min(max(influence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_coherence(self) -> float:
        """
        Calculate the internal coherence of the cymatic field.
        
        Returns:
            A value between 0 and 1 representing field coherence
        """
        # Extract amplitude pattern
        amplitudes = np.abs(self.grid)
        
        # Calculate field gradient
        gradients = np.gradient(amplitudes)
        gradient_magnitude = np.sqrt(sum(np.square(g) for g in gradients))
        
        # Calculate field average and standard deviation
        field_avg = np.mean(amplitudes)
        field_std = np.std(amplitudes)
        
        # Smoothness - lower gradient means smoother field
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) * PHI)
        
        # Uniformity - lower standard deviation means more uniform field
        uniformity = 1.0 / (1.0 + field_std / (field_avg + 1e-10) * PHI_INVERSE)
        
        # Pattern stability - phase coherence across the field
        phases = np.angle(self.grid)
        phase_std = np.std(phases)
        phase_coherence = 1.0 / (1.0 + phase_std * LAMBDA)
        
        # Calculate phi-weighted coherence metric
        coherence = (
            smoothness * PHI_INVERSE +
            uniformity * 1.0 +
            phase_coherence * PHI
        ) / (PHI_INVERSE + 1.0 + PHI)
        
        return min(max(coherence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def store_pattern(self, name: str):
        """
        Store the current pattern in pattern memory.
        
        Args:
            name: Name to identify the stored pattern
        """
        self.pattern_memory[name] = {
            'grid': self.grid.copy(),
            'frequency': self.base_frequency,
            'coherence': self._calculate_coherence(),
            'complexity': self.pattern_complexity
        }
    
    def recall_pattern(self, name: str) -> bool:
        """
        Recall a stored pattern from memory.
        
        Args:
            name: Name of the pattern to recall
            
        Returns:
            True if pattern was found and loaded, False otherwise
        """
        if name in self.pattern_memory:
            pattern = self.pattern_memory[name]
            self.grid = pattern['grid'].copy()
            self.base_frequency = pattern['frequency']
            self.coherence = pattern['coherence']
            self.pattern_complexity = pattern['complexity']
            self.frequencies = self._initialize_frequencies()
            return True
        return False
    
    def blend_patterns(self, pattern1: str, pattern2: str, blend_factor: float = 0.5) -> bool:
        """
        Blend two stored patterns.
        
        Args:
            pattern1: Name of the first pattern
            pattern2: Name of the second pattern
            blend_factor: Blending factor (0.0 = only pattern1, 1.0 = only pattern2)
            
        Returns:
            True if both patterns were found and blended, False otherwise
        """
        if pattern1 in self.pattern_memory and pattern2 in self.pattern_memory:
            p1 = self.pattern_memory[pattern1]['grid']
            p2 = self.pattern_memory[pattern2]['grid']
            
            # Blend patterns
            self.grid = (1 - blend_factor) * p1 + blend_factor * p2
            
            # Blend other properties
            self.base_frequency = (
                (1 - blend_factor) * self.pattern_memory[pattern1]['frequency'] +
                blend_factor * self.pattern_memory[pattern2]['frequency']
            )
            
            self.coherence = self._calculate_coherence()
            self.pattern_complexity = max(
                self.pattern_memory[pattern1]['complexity'],
                self.pattern_memory[pattern2]['complexity']
            )
            
            self.frequencies = self._initialize_frequencies()
            return True
        return False
    
    def get_pattern_at_frequency(self, frequency: float) -> np.ndarray:
        """
        Get a 2D slice of the pattern at a specific frequency.
        
        Args:
            frequency: The frequency to generate the pattern for
            
        Returns:
            2D numpy array representing the cymatic pattern at that frequency
        """
        # Calculate wavelength for the given frequency
        wavelength = 343.0 / frequency  # Speed of sound / frequency
        
        # Extract middle slice of the 3D field
        mid_z = self.dimensions[2] // 2
        slice_2d = np.abs(self.grid[:, :, mid_z])
        
        # Scale pattern based on frequency relationship
        freq_ratio = frequency / self.base_frequency
        
        # Apply frequency-dependent transformation
        if freq_ratio != 1.0:
            # Create coordinate grid for transformation
            nx, ny = slice_2d.shape
            x = np.linspace(-1, 1, nx)
            y = np.linspace(-1, 1, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Calculate radial distance
            R = np.sqrt(X**2 + Y**2)
            
            # Create frequency-modulated pattern
            k = 2 * np.pi / (wavelength * self.resolution)
            pattern = np.sin(k * R * freq_ratio) * np.exp(-R**2 / (2 * freq_ratio))
            
            # Combine with original slice
            slice_2d = (slice_2d + pattern) / 2.0
        
        # Normalize
        if np.max(slice_2d) > 0:
            slice_2d = slice_2d / np.max(slice_2d)
            
        return slice_2d
    
    def visualize_pattern(self, frequency: Optional[float] = None) -> np.ndarray:
        """
        Generate a 2D visualization of the cymatic pattern.
        
        Args:
            frequency: Optional frequency override (uses base_frequency if None)
            
        Returns:
            2D numpy array representing the cymatic pattern
        """
        if frequency is None:
            frequency = self.base_frequency
            
        return self.get_pattern_at_frequency(frequency)
    
    def align_with_consciousness(self, consciousness_state: int, intensity: float = 1.0):
        """
        Align the cymatic field with a specific consciousness state.
        
        Args:
            consciousness_state: The consciousness state to align with
            intensity: Intensity of the alignment effect (0.0 to 1.0)
        """
        # Map consciousness states to frequencies
        consciousness_frequencies = {
            0: SACRED_FREQUENCIES['unity'],      # BE state
            1: SACRED_FREQUENCIES['love'],       # DO state
            2: SACRED_FREQUENCIES['truth'],      # WITNESS state
            3: SACRED_FREQUENCIES['love'],       # CREATE state
            4: SACRED_FREQUENCIES['cascade'],    # INTEGRATE state
            5: SACRED_FREQUENCIES['vision'],     # TRANSCEND state
        }
        
        # Get frequency for the specified consciousness state
        target_frequency = consciousness_frequencies.get(
            consciousness_state, 
            self.base_frequency
        )
        
        # Blend current frequency with target frequency
        new_frequency = (
            (1 - intensity) * self.base_frequency +
            intensity * target_frequency
        )
        
        # Update field
        self.set_frequency(new_frequency)
        
        # Apply consciousness-specific pattern modulation
        nx, ny, nz = self.dimensions
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Different modulation patterns for different consciousness states
        if consciousness_state == 0:  # BE - Unity/ground state
            # Spherical, balanced pattern
            R = np.sqrt(X**2 + Y**2 + Z**2)
            mod_pattern = np.exp(-R**2 / PHI) * np.sin(R * np.pi * 2)
            
        elif consciousness_state == 1:  # DO - Action state
            # Dynamic, flowing pattern
            mod_pattern = np.sin(X * PHI * np.pi) * np.cos(Y * PHI * np.pi) * np.sin(Z * PHI * np.pi)
            
        elif consciousness_state == 2:  # WITNESS - Observation state
            # Clear, structured pattern
            R = np.sqrt(X**2 + Y**2 + Z**2)
            mod_pattern = np.sin(R * PHI * np.pi * 2) * np.cos(Z * PHI * np.pi)
            
        elif consciousness_state == 3:  # CREATE - Manifestation state
            # Creative, expansive pattern
            R = np.sqrt(X**2 + Y**2 + Z**2)
            theta = np.arctan2(Y, X)
            mod_pattern = np.sin(R * PHI * np.pi) * np.cos(theta * 5) * np.sin(Z * PHI * np.pi)
            
        elif consciousness_state == 4:  # INTEGRATE - Harmony state
            # Balanced, integrative pattern
            mod_pattern = (
                np.sin(X * PHI * np.pi) * 
                np.sin(Y * PHI * np.pi) * 
                np.sin(Z * PHI * np.pi)
            )
            
        elif consciousness_state == 5:  # TRANSCEND - Vision state
            # Complex, multidimensional pattern
            R = np.sqrt(X**2 + Y**2 + Z**2)
            theta = np.arctan2(Y, X)
            phi = np.arccos(Z / (R + 1e-10))
            mod_pattern = (
                np.sin(R * PHI_PHI * np.pi) * 
                np.cos(theta * PHI) * 
                np.sin(phi * PHI_PHI)
            )
            
        else:  # Default
            mod_pattern = np.ones_like(X)
        
        # Apply modulation to field with intensity factor
        amplitudes = np.abs(self.grid)
        phases = np.angle(self.grid)
        
        # Modulate amplitudes
        modulated_amp = amplitudes * (1.0 + mod_pattern * intensity * LAMBDA)
        
        # Recreate complex field
        self.grid = modulated_amp * np.exp(1j * phases)
        
        # Update coherence based on new pattern
        self.coherence = self._calculate_coherence()
    
    def extract_pattern_metrics(self) -> Dict[str, float]:
        """
        Extract key metrics about the current cymatic pattern.
        
        Returns:
            Dictionary of pattern metrics
        """
        # Calculate field coherence
        coherence = self._calculate_coherence()
        
        # Calculate central intensity
        nx, ny, nz = self.dimensions
        center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
        central_region = self.grid[
            center_x-3:center_x+4,
            center_y-3:center_y+4,
            center_z-3:center_z+4
        ]
        central_intensity = np.mean(np.abs(central_region))
        
        # Calculate pattern complexity using FFT energy distribution
        fft = np.fft.fftn(np.abs(self.grid))
        fft_magnitude = np.abs(fft)
        
        # Normalize and sort FFT magnitudes
        norm_fft = fft_magnitude / np.sum(fft_magnitude)
        sorted_values = np.sort(norm_fft.flatten())[::-1]  # Sort in descending order
        
        # Calculate energy distribution - how quickly energy drops off
        # Lower values mean more complex patterns (energy spread across frequencies)
        top_energy = np.sum(sorted_values[:100]) / np.sum(sorted_values)
        complexity = 1.0 - top_energy
        
        # Calculate symmetry metrics
        x_symmetry = self._calculate_symmetry(0)
        y_symmetry = self._calculate_symmetry(1)
        z_symmetry = self._calculate_symmetry(2)
        
        # Calculate frequency purity (how closely it matches pure frequency)
        phase_variance = np.var(np.angle(self.grid))
        frequency_purity = 1.0 / (1.0 + phase_variance)
        
        # Calculate phi-alignment of the pattern
        phi_alignment = self._calculate_phi_alignment()
        
        # Collect all metrics
        metrics = {
            'coherence': coherence,
            'central_intensity': central_intensity,
            'complexity': complexity,
            'x_symmetry': x_symmetry,
            'y_symmetry': y_symmetry,
            'z_symmetry': z_symmetry,
            'overall_symmetry': (x_symmetry + y_symmetry + z_symmetry) / 3,
            'frequency_purity': frequency_purity,
            'phi_alignment': phi_alignment,
            'materialization_potential': (coherence + phi_alignment + frequency_purity) / 3
        }
        
        return metrics
    
    def _calculate_symmetry(self, axis: int) -> float:
        """
        Calculate symmetry of the field along a specific axis.
        
        Args:
            axis: Axis to calculate symmetry along (0=x, 1=y, 2=z)
            
        Returns:
            Symmetry value between 0 and 1
        """
        # Get pattern magnitudes
        pattern = np.abs(self.grid)
        
        # Calculate half-point along the axis
        half_point = pattern.shape[axis] // 2
        
        # Split the pattern along the axis
        slices = [slice(None), slice(None), slice(None)]
        
        # First half
        slices[axis] = slice(0, half_point)
        first_half = pattern[tuple(slices)]
        
        # Second half (flipped)
        slices[axis] = slice(None, half_point, -1)
        second_half = pattern[tuple(slices)]
        
        # Resize smaller half if needed
        if first_half.shape != second_half.shape:
            # Use the smaller shape
            min_shape = [min(s1, s2) for s1, s2 in zip(first_half.shape, second_half.shape)]
            slices = tuple(slice(0, s) for s in min_shape)
            first_half = first_half[slices]
            second_half = second_half[slices]
        
        # Calculate difference between halves
        diff = np.abs(first_half - second_half)
        symmetry = 1.0 - np.mean(diff) / (np.mean(first_half) + 1e-10)
        
        return max(min(symmetry, 1.0), 0.0)
    
    def _calculate_phi_alignment(self) -> float:
        """
        Calculate how well the pattern aligns with phi-harmonic proportions.
        
        Returns:
            Phi-alignment value between 0 and 1
        """
        # Extract amplitude pattern
        amplitudes = np.abs(self.grid)
        
        # Calculate FFT to find frequency components
        fft = np.fft.fftn(amplitudes)
        fft_magnitude = np.abs(fft)
        
        # Normalize FFT magnitude
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
        
        # Create frequency grid
        nx, ny, nz = fft_magnitude.shape
        fx = np.fft.fftfreq(nx) * nx
        fy = np.fft.fftfreq(ny) * ny
        fz = np.fft.fftfreq(nz) * nz
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        
        # Calculate radial frequency
        F_radial = np.sqrt(FX**2 + FY**2 + FZ**2)
        
        # Phi-harmonic frequencies we want to check for
        phi_harmonics = [1, PHI, PHI_PHI, PHI*PHI, PHI*PHI*PHI]
        
        # Calculate alignment with each phi-harmonic frequency
        phi_resonances = []
        for phi_harmonic in phi_harmonics:
            # Find how much energy is at this harmonic
            # (with some tolerance around the exact frequency)
            target_f = phi_harmonic * 10  # Scale factor for better resolution
            tolerance = 0.2 * target_f
            
            # Calculate mask for frequencies near this harmonic
            freq_mask = np.abs(F_radial - target_f) < tolerance
            
            # Get energy at this harmonic
            harmonic_energy = np.sum(fft_magnitude[freq_mask])
            total_energy = np.sum(fft_magnitude)
            
            if total_energy > 0:
                phi_resonances.append(harmonic_energy / total_energy)
            else:
                phi_resonances.append(0.0)
        
        # Calculate overall phi alignment as weighted sum
        weights = [PHI**(-i) for i in range(len(phi_harmonics))]
        total_weight = sum(weights)
        
        if total_weight > 0:
            phi_alignment = sum(r * w for r, w in zip(phi_resonances, weights)) / total_weight
        else:
            phi_alignment = 0.0
            
        return min(max(phi_alignment, 0.0), 1.0)