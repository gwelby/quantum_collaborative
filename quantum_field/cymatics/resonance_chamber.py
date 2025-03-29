"""
Resonance Chamber Module

Implementation of material-specific resonance chambers for pattern amplification
through phi-harmonic frequency relationships and standing wave patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
sys.path.append('/mnt/d/projects/Python')
from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES, 
    calculate_phi_resonance, phi_harmonic
)

class MaterialResonator:
    """
    Base class for material-specific resonance chambers that amplify
    cymatic patterns through phi-harmonic relationships.
    """
    
    def __init__(
        self,
        name: str,
        resonant_frequency: float,
        q_factor: float = 10.0,
        dimensions: Tuple[float, float, float] = (1.0, 1.0, 0.1)
    ):
        """
        Initialize a new MaterialResonator.
        
        Args:
            name: Name of the resonator
            resonant_frequency: Primary resonant frequency in Hz
            q_factor: Quality factor (sharpness of resonance)
            dimensions: Physical dimensions (width, length, depth) in meters
        """
        self.name = name
        self.resonant_frequency = resonant_frequency
        self.q_factor = q_factor
        self.dimensions = dimensions
        
        # Calculate phi-harmonic resonant frequencies
        self.harmonic_frequencies = self._calculate_harmonics()
        
        # Material-specific properties
        self.density = 1000.0  # kg/m³, default is water
        self.speed_of_sound = 343.0  # m/s, default is air
        
        # Resonance properties
        self.wavelength = self.speed_of_sound / self.resonant_frequency
        self.resonant_modes = self._calculate_resonant_modes()
        
        # State
        self.current_amplitude = 0.0
        self.current_frequency = resonant_frequency
        self.current_pattern = None
        self.energy_level = 0.0
        self.coherence = 1.0
        
        # Field properties
        self.field_grid = None
        self.grid_resolution = (50, 50, 10)  # Default grid resolution
    
    def _calculate_harmonics(self) -> Dict[str, float]:
        """
        Calculate phi-harmonic frequencies for this material.
        
        Returns:
            Dictionary of harmonic frequencies
        """
        harmonics = {
            'fundamental': self.resonant_frequency,
            'phi': self.resonant_frequency * PHI,
            'phi_squared': self.resonant_frequency * PHI * PHI,
            'phi_inverse': self.resonant_frequency * LAMBDA,
            'phi_phi': self.resonant_frequency * PHI_PHI,
            'octave': self.resonant_frequency * 2.0,
            'octave_phi': self.resonant_frequency * 2.0 * PHI,
            'fifth': self.resonant_frequency * 1.5,
            'fourth': self.resonant_frequency * 4.0/3.0
        }
        
        return harmonics
    
    def _calculate_resonant_modes(self) -> List[Tuple[int, int, int]]:
        """
        Calculate resonant modes based on chamber dimensions.
        
        Returns:
            List of (nx, ny, nz) mode indices for resonant frequencies
        """
        # Calculate fundamental wavelengths for each dimension
        width, length, depth = self.dimensions
        
        # Maximum mode numbers to consider
        max_nx, max_ny, max_nz = 5, 5, 3
        
        # List to store resonant modes
        modes = []
        
        # Calculate modes
        for nx in range(1, max_nx + 1):
            for ny in range(1, max_ny + 1):
                for nz in range(1, max_nz + 1):
                    # Calculate mode frequency
                    freq = self.speed_of_sound * 0.5 * np.sqrt(
                        (nx/width)**2 + (ny/length)**2 + (nz/depth)**2
                    )
                    
                    # Check if this mode is near a phi-harmonic of the fundamental
                    for _, harmonic_freq in self.harmonic_frequencies.items():
                        # Calculate frequency ratio
                        ratio = freq / harmonic_freq
                        
                        # Check if close to an integer ratio
                        if abs(ratio - round(ratio)) < 0.1:
                            modes.append((nx, ny, nz))
                            break
        
        return modes
    
    def initialize_field_grid(self, resolution: Tuple[int, int, int] = None):
        """
        Initialize the resonance field grid.
        
        Args:
            resolution: Optional resolution for the grid (defaults to self.grid_resolution)
        """
        if resolution is not None:
            self.grid_resolution = resolution
            
        # Create empty grid
        self.field_grid = np.zeros(self.grid_resolution, dtype=np.complex128)
        
        # Create coordinate grids
        nx, ny, nz = self.grid_resolution
        x = np.linspace(0, self.dimensions[0], nx)
        y = np.linspace(0, self.dimensions[1], ny)
        z = np.linspace(0, self.dimensions[2], nz)
        
        # Set up basic standing wave pattern
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate wave numbers for fundamental mode
        kx = np.pi / self.dimensions[0]
        ky = np.pi / self.dimensions[1]
        kz = np.pi / self.dimensions[2]
        
        # Initialize with fundamental mode
        self.field_grid = np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
        
        # Add phi-harmonic modulation
        phi_mod = (
            np.sin(kx * PHI * X) * 
            np.sin(ky * PHI * Y) * 
            np.sin(kz * PHI * Z)
        ) * LAMBDA
        
        # Combine
        self.field_grid = self.field_grid + phi_mod
        
        # Normalize
        if np.max(np.abs(self.field_grid)) > 0:
            self.field_grid = self.field_grid / np.max(np.abs(self.field_grid))
    
    def apply_frequency(self, frequency: float, amplitude: float = 1.0) -> float:
        """
        Apply a frequency to the resonance chamber and calculate response.
        
        Args:
            frequency: Frequency to apply in Hz
            amplitude: Amplitude of the input signal (0 to 1)
            
        Returns:
            Response amplitude (0 to 1)
        """
        # Update current state
        self.current_frequency = frequency
        self.current_amplitude = amplitude
        
        # Calculate response using resonance curve
        response = self._calculate_resonance_response(frequency, amplitude)
        
        # Update energy level based on response
        self.energy_level = response * amplitude
        
        # Update field pattern
        self._update_field_pattern(frequency, response)
        
        return response
    
    def _calculate_resonance_response(self, frequency: float, amplitude: float) -> float:
        """
        Calculate resonance response using a Q-factor based resonance curve.
        
        Args:
            frequency: Input frequency in Hz
            amplitude: Input amplitude (0 to 1)
            
        Returns:
            Response amplitude (0 to 1)
        """
        # Check response for each harmonic
        max_response = 0.0
        
        for _, harmonic_freq in self.harmonic_frequencies.items():
            # Calculate normalized frequency difference
            f_ratio = frequency / harmonic_freq
            f_diff = abs(f_ratio - 1.0)
            
            # Calculate response using Q-factor resonance formula
            response = 1.0 / np.sqrt(1.0 + (self.q_factor * f_diff)**2)
            
            # Track maximum response
            max_response = max(max_response, response)
            
        # Apply material-specific amplification
        response = self._apply_material_amplification(max_response, frequency)
        
        # Scale by input amplitude
        response = response * amplitude
        
        return min(max(response, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _apply_material_amplification(self, base_response: float, frequency: float) -> float:
        """
        Apply material-specific amplification to the resonance response.
        Base implementation - to be overridden by subclasses.
        
        Args:
            base_response: Base response amplitude (0 to 1)
            frequency: Input frequency in Hz
            
        Returns:
            Amplified response (0 to 1)
        """
        # Base implementation just returns the input response
        return base_response
    
    def _update_field_pattern(self, frequency: float, response: float):
        """
        Update the resonance field pattern based on input frequency and response.
        
        Args:
            frequency: Input frequency in Hz
            response: Response amplitude (0 to 1)
        """
        if self.field_grid is None:
            self.initialize_field_grid()
            
        # Create coordinate grids
        nx, ny, nz = self.grid_resolution
        x = np.linspace(0, self.dimensions[0], nx)
        y = np.linspace(0, self.dimensions[1], ny)
        z = np.linspace(0, self.dimensions[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate wavelength and wavenumber
        wavelength = self.speed_of_sound / frequency
        k = 2 * np.pi / wavelength
        
        # Find closest resonant mode
        closest_mode = self._find_closest_mode(frequency)
        nx_mode, ny_mode, nz_mode = closest_mode
        
        # Calculate resonant pattern for this mode
        kx = nx_mode * np.pi / self.dimensions[0]
        ky = ny_mode * np.pi / self.dimensions[1]
        kz = nz_mode * np.pi / self.dimensions[2]
        
        # Create resonant pattern
        resonant_pattern = (
            np.sin(kx * X) * 
            np.sin(ky * Y) * 
            np.sin(kz * Z)
        )
        
        # Add phi-harmonic modulation for complexity
        phi_mod = (
            np.sin(kx * PHI * X) * 
            np.sin(ky * PHI * Y) * 
            np.sin(kz * PHI * Z)
        ) * LAMBDA
        
        # Combine patterns
        new_pattern = resonant_pattern + phi_mod
        
        # Normalize
        if np.max(np.abs(new_pattern)) > 0:
            new_pattern = new_pattern / np.max(np.abs(new_pattern))
        
        # Blend with existing pattern for smooth transition
        blend_factor = 0.3  # 30% new, 70% old
        self.field_grid = (
            blend_factor * new_pattern + 
            (1 - blend_factor) * self.field_grid
        )
        
        # Scale by response amplitude
        self.field_grid = self.field_grid * response
        
        # Calculate coherence of the pattern
        self.coherence = self._calculate_pattern_coherence()
        
        # Store 2D pattern slice for reference
        self.current_pattern = self.get_2d_pattern_slice()
    
    def _find_closest_mode(self, frequency: float) -> Tuple[int, int, int]:
        """
        Find the resonant mode closest to the specified frequency.
        
        Args:
            frequency: Frequency to match in Hz
            
        Returns:
            Tuple of (nx, ny, nz) mode numbers
        """
        if not self.resonant_modes:
            # Default to fundamental mode if no modes calculated
            return (1, 1, 1)
            
        # Calculate all mode frequencies
        best_mode = self.resonant_modes[0]
        min_diff = float('inf')
        
        for mode in self.resonant_modes:
            nx, ny, nz = mode
            
            # Calculate mode frequency
            mode_freq = self.speed_of_sound * 0.5 * np.sqrt(
                (nx/self.dimensions[0])**2 + 
                (ny/self.dimensions[1])**2 + 
                (nz/self.dimensions[2])**2
            )
            
            # Calculate frequency difference
            diff = abs(mode_freq - frequency)
            
            # Update if closer
            if diff < min_diff:
                min_diff = diff
                best_mode = mode
                
        return best_mode
    
    def get_2d_pattern_slice(self, z_level: float = 0.5) -> np.ndarray:
        """
        Get a 2D slice of the current resonance pattern.
        
        Args:
            z_level: Relative height level for the slice (0 to 1)
            
        Returns:
            2D numpy array with the pattern slice
        """
        if self.field_grid is None:
            self.initialize_field_grid()
            
        # Convert relative z_level to index
        z_index = int(z_level * (self.field_grid.shape[2] - 1))
        z_index = max(0, min(z_index, self.field_grid.shape[2] - 1))
        
        # Extract slice
        slice_2d = np.abs(self.field_grid[:, :, z_index])
        
        # Normalize if needed
        if np.max(slice_2d) > 0:
            slice_2d = slice_2d / np.max(slice_2d)
            
        return slice_2d
    
    def _calculate_pattern_coherence(self) -> float:
        """
        Calculate the coherence of the current resonance pattern.
        
        Returns:
            Coherence value between 0 and 1
        """
        if self.field_grid is None:
            return 0.0
            
        # Extract amplitude pattern
        amplitudes = np.abs(self.field_grid)
        
        # Calculate field gradient
        gradients = np.gradient(amplitudes)
        gradient_magnitude = np.sqrt(sum(np.square(g) for g in gradients))
        
        # Calculate smoothness metric
        avg_gradient = np.mean(gradient_magnitude)
        smoothness = 1.0 / (1.0 + avg_gradient * PHI)
        
        # Calculate uniformity
        field_avg = np.mean(amplitudes)
        field_std = np.std(amplitudes)
        
        if field_avg > 0:
            uniformity = 1.0 / (1.0 + field_std / field_avg * PHI_INVERSE)
        else:
            uniformity = 0.0
            
        # Calculate structure (presence of clear nodal lines/points)
        zero_points = np.sum(amplitudes < 0.1) / amplitudes.size
        structure = 4.0 * zero_points * (1.0 - zero_points)  # Peaks at 0.5 (balanced nodes)
        
        # Combine metrics with phi-weighted averaging
        coherence = (
            smoothness * LAMBDA +
            uniformity * 1.0 +
            structure * PHI
        ) / (LAMBDA + 1.0 + PHI)
        
        return min(max(coherence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def calculate_optimal_frequencies(self) -> List[float]:
        """
        Calculate the optimal frequencies for this resonator.
        
        Returns:
            List of optimal frequencies in Hz
        """
        # Start with the harmonic frequencies
        optimal_freqs = list(self.harmonic_frequencies.values())
        
        # Add resonant mode frequencies
        for mode in self.resonant_modes:
            nx, ny, nz = mode
            
            # Calculate mode frequency
            mode_freq = self.speed_of_sound * 0.5 * np.sqrt(
                (nx/self.dimensions[0])**2 + 
                (ny/self.dimensions[1])**2 + 
                (nz/self.dimensions[2])**2
            )
            
            optimal_freqs.append(mode_freq)
            
        # Sort and remove duplicates
        optimal_freqs = sorted(list(set([round(f, 2) for f in optimal_freqs])))
        
        return optimal_freqs
    
    def get_resonance_curve(
        self, 
        freq_range: Tuple[float, float],
        num_points: int = 100
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate the resonance response curve over a frequency range.
        
        Args:
            freq_range: (min_freq, max_freq) in Hz
            num_points: Number of frequency points to calculate
            
        Returns:
            Tuple of (frequencies, responses) lists
        """
        # Generate frequency points
        frequencies = np.linspace(freq_range[0], freq_range[1], num_points)
        
        # Calculate response at each frequency
        responses = [self._calculate_resonance_response(f, 1.0) for f in frequencies]
        
        return frequencies.tolist(), responses
    
    def apply_frequency_sweep(
        self, 
        start_freq: float,
        end_freq: float,
        duration: float = 5.0,
        amplitude: float = 1.0,
        steps: int = 50
    ) -> Dict[str, List[float]]:
        """
        Apply a frequency sweep and record the resonance response.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration: Duration of the sweep in seconds
            amplitude: Amplitude of the input signal
            steps: Number of frequency steps
            
        Returns:
            Dictionary with frequencies, responses, and coherence values
        """
        # Generate logarithmic frequency steps
        frequencies = np.logspace(
            np.log10(start_freq),
            np.log10(end_freq),
            steps
        )
        
        # Calculate time points
        times = np.linspace(0, duration, steps)
        
        # Record responses
        responses = []
        coherence_values = []
        
        for freq in frequencies:
            # Apply frequency
            response = self.apply_frequency(freq, amplitude)
            
            # Record results
            responses.append(response)
            coherence_values.append(self.coherence)
            
        return {
            'frequencies': frequencies.tolist(),
            'times': times.tolist(),
            'responses': responses,
            'coherence': coherence_values
        }
    
    def find_peak_resonances(
        self, 
        min_freq: float = 20.0,
        max_freq: float = 2000.0,
        threshold: float = 0.7
    ) -> List[Dict[str, float]]:
        """
        Find frequency ranges with peak resonance responses.
        
        Args:
            min_freq: Minimum frequency to search in Hz
            max_freq: Maximum frequency to search in Hz
            threshold: Threshold for considering a peak (0 to 1)
            
        Returns:
            List of dictionaries with peak information
        """
        # Get resonance curve
        freqs, responses = self.get_resonance_curve((min_freq, max_freq), 200)
        
        # Find peaks
        peaks = []
        in_peak = False
        peak_start = 0
        
        for i, (freq, resp) in enumerate(zip(freqs, responses)):
            if resp >= threshold and not in_peak:
                # Start of peak
                in_peak = True
                peak_start = i
                
            elif resp < threshold and in_peak:
                # End of peak
                peak_center = peak_start + np.argmax(responses[peak_start:i])
                
                # Record peak info
                peaks.append({
                    'frequency': freqs[peak_center],
                    'response': responses[peak_center],
                    'start_freq': freqs[peak_start],
                    'end_freq': freqs[i-1],
                    'q_factor': self._estimate_q_factor(
                        freqs[peak_start:i], 
                        responses[peak_start:i]
                    )
                })
                
                in_peak = False
                
        # Check if we ended in a peak
        if in_peak:
            peak_center = peak_start + np.argmax(responses[peak_start:])
            
            peaks.append({
                'frequency': freqs[peak_center],
                'response': responses[peak_center],
                'start_freq': freqs[peak_start],
                'end_freq': freqs[-1],
                'q_factor': self._estimate_q_factor(
                    freqs[peak_start:], 
                    responses[peak_start:]
                )
            })
            
        return peaks
    
    def _estimate_q_factor(self, frequencies: List[float], responses: List[float]) -> float:
        """
        Estimate Q-factor from a resonance peak.
        
        Args:
            frequencies: List of frequencies around peak
            responses: Corresponding response values
            
        Returns:
            Estimated Q-factor
        """
        if len(frequencies) < 3:
            return self.q_factor
            
        # Find peak frequency and response
        peak_idx = np.argmax(responses)
        peak_freq = frequencies[peak_idx]
        peak_resp = responses[peak_idx]
        
        # Find half-power frequencies (response = peak/sqrt(2))
        half_power = peak_resp / np.sqrt(2)
        
        # Find frequencies below and above peak where response crosses half-power
        f_below = None
        for i in range(peak_idx, 0, -1):
            if responses[i] <= half_power:
                f_below = frequencies[i]
                break
                
        f_above = None
        for i in range(peak_idx, len(frequencies)):
            if responses[i] <= half_power:
                f_above = frequencies[i]
                break
                
        # Calculate Q-factor
        if f_below is not None and f_above is not None:
            bandwidth = f_above - f_below
            if bandwidth > 0:
                q_factor = peak_freq / bandwidth
                return q_factor
                
        # Return default if calculation failed
        return self.q_factor

class WaterResonator(MaterialResonator):
    """
    A resonance chamber specifically designed for water-based cymatic patterns.
    """
    
    def __init__(
        self,
        name: str = "Water Resonator",
        resonant_frequency: float = SACRED_FREQUENCIES['unity'],  # 432 Hz
        dimensions: Tuple[float, float, float] = (0.15, 0.15, 0.02),
        water_depth: float = 0.01
    ):
        """
        Initialize a water-based resonance chamber.
        
        Args:
            name: Name of the resonator
            resonant_frequency: Primary resonant frequency in Hz
            dimensions: Chamber dimensions (width, length, height) in meters
            water_depth: Depth of water in the chamber in meters
        """
        # Calculate Q-factor based on water depth
        q_factor = 8.0 * water_depth / 0.01  # Higher Q for deeper water
        
        # Initialize base resonator
        super().__init__(name, resonant_frequency, q_factor, dimensions)
        
        # Water-specific properties
        self.water_depth = water_depth
        self.density = 1000.0  # kg/m³
        self.speed_of_sound = 1480.0  # m/s in water
        self.surface_tension = 0.072  # N/m
        
        # Water responds well to specific frequencies
        self.optimal_frequencies = [
            SACRED_FREQUENCIES['unity'],  # 432 Hz - unity/ground
            SACRED_FREQUENCIES['love'],   # 528 Hz - healing/DNA
            421.0,  # Water structuring frequency
            396.0,  # Liberation frequency
            417.0   # Transformation frequency
        ]
        
        # Update wavelength and resonant modes based on water properties
        self.wavelength = self.speed_of_sound / self.resonant_frequency
        self.resonant_modes = self._calculate_resonant_modes()
    
    def _apply_material_amplification(self, base_response: float, frequency: float) -> float:
        """
        Apply water-specific amplification to the resonance response.
        
        Args:
            base_response: Base response amplitude (0 to 1)
            frequency: Input frequency in Hz
            
        Returns:
            Amplified response (0 to 1)
        """
        # Check if frequency is close to water's optimal frequencies
        optimal_factor = 0.0
        for opt_freq in self.optimal_frequencies:
            # Calculate frequency ratio
            freq_ratio = frequency / opt_freq
            
            # Calculate resonance using phi-based algorithm
            resonance = calculate_phi_resonance(frequency, opt_freq)
            
            # Update factor if better match found
            optimal_factor = max(optimal_factor, resonance)
            
        # Calculate capillary wave factor based on frequency
        # Frequencies that create stable capillary waves enhance pattern formation
        wavelength = self.speed_of_sound / frequency
        capillary_wavelength = 2 * np.pi * np.sqrt(self.surface_tension / 
                                                 (self.density * (2*np.pi*frequency)**2))
        
        capillary_factor = 1.0 / (1.0 + abs(wavelength/capillary_wavelength - 1.0) * 5.0)
        
        # Combine factors with phi-weighted averaging
        amplified_response = (
            base_response * 1.0 + 
            optimal_factor * PHI + 
            capillary_factor * LAMBDA
        ) / (1.0 + PHI + LAMBDA)
        
        return min(max(amplified_response, 0.0), 1.0)  # Clamp to [0, 1]
    
    def calculate_meniscus_effect(self, amplitude: float) -> float:
        """
        Calculate the edge meniscus effect for water in the chamber.
        
        Args:
            amplitude: Current amplitude of oscillation
            
        Returns:
            Meniscus factor (0 to 1) representing edge pattern enhancement
        """
        # Calculate capillary length
        gravity = 9.81  # m/s²
        capillary_length = np.sqrt(self.surface_tension / (self.density * gravity))
        
        # Calculate meniscus width relative to chamber width
        relative_width = capillary_length / min(self.dimensions[0], self.dimensions[1])
        
        # Calculate meniscus factor - higher for smaller chambers
        meniscus_factor = LAMBDA / (relative_width + LAMBDA)
        
        # Scale by amplitude - meniscus effects are stronger at higher amplitudes
        meniscus_factor = meniscus_factor * (0.2 + 0.8 * amplitude)
        
        return meniscus_factor
    
    def update_water_depth(self, depth: float):
        """
        Update the water depth and recalculate resonance properties.
        
        Args:
            depth: New water depth in meters
        """
        self.water_depth = depth
        
        # Update Q-factor based on new depth
        self.q_factor = 8.0 * depth / 0.01
        
        # Recalculate resonant modes
        self.resonant_modes = self._calculate_resonant_modes()

class CrystalResonator(MaterialResonator):
    """
    A resonance chamber designed for crystalline structures and materials.
    """
    
    def __init__(
        self,
        name: str = "Crystal Resonator",
        resonant_frequency: float = SACRED_FREQUENCIES['love'],  # 528 Hz
        dimensions: Tuple[float, float, float] = (0.10, 0.10, 0.10),
        crystal_type: str = "quartz"
    ):
        """
        Initialize a crystal resonance chamber.
        
        Args:
            name: Name of the resonator
            resonant_frequency: Primary resonant frequency in Hz
            dimensions: Crystal dimensions (width, length, height) in meters
            crystal_type: Type of crystal ("quartz", "selenite", "amethyst", etc.)
        """
        # Crystals have sharp resonances
        q_factor = 25.0
        
        # Initialize base resonator
        super().__init__(name, resonant_frequency, q_factor, dimensions)
        
        # Crystal-specific properties
        self.crystal_type = crystal_type
        
        # Set crystal properties based on type
        self._set_crystal_properties()
        
        # Crystal formation quality (0 to 1)
        self.crystal_quality = 0.9
        
        # Geometric factors
        self.facet_count = 6  # Hexagonal by default
        self.lattice_alignment = 0.95  # How well aligned the crystal lattice is
        
        # Update wavelength and resonant modes based on crystal properties
        self.wavelength = self.speed_of_sound / self.resonant_frequency
        self.resonant_modes = self._calculate_resonant_modes()
    
    def _set_crystal_properties(self):
        """Set physical properties based on crystal type."""
        # Properties: density (kg/m³), speed of sound (m/s), optimal frequencies
        crystal_properties = {
            'quartz': {
                'density': 2650.0,
                'speed_of_sound': 5760.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['love'],    # 528 Hz - healing/DNA
                    SACRED_FREQUENCIES['vision'],  # 720 Hz - clarity/connection
                    786.0,  # Higher octave of 393 Hz (crystal activation)
                    963.0   # Source connection
                ]
            },
            'selenite': {
                'density': 2320.0,
                'speed_of_sound': 4800.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['unity'],    # 432 Hz - unity/ground
                    SACRED_FREQUENCIES['oneness'],  # 768 Hz - cosmic connection
                    852.0,  # High intuition frequency
                    963.0   # Source connection
                ]
            },
            'amethyst': {
                'density': 2650.0,
                'speed_of_sound': 5200.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['vision'],   # 720 Hz - clarity/connection
                    SACRED_FREQUENCIES['truth'],    # 672 Hz - truth/expression
                    936.0,  # Third eye activation
                    741.0   # Awakening intuition
                ]
            }
        }
        
        # Use quartz as default if crystal type not found
        props = crystal_properties.get(self.crystal_type, crystal_properties['quartz'])
        
        # Set properties
        self.density = props['density']
        self.speed_of_sound = props['speed_of_sound']
        self.optimal_frequencies = props['optimal_frequencies']
    
    def _apply_material_amplification(self, base_response: float, frequency: float) -> float:
        """
        Apply crystal-specific amplification to the resonance response.
        
        Args:
            base_response: Base response amplitude (0 to 1)
            frequency: Input frequency in Hz
            
        Returns:
            Amplified response (0 to 1)
        """
        # Check if frequency is close to crystal's optimal frequencies
        optimal_factor = 0.0
        for opt_freq in self.optimal_frequencies:
            # Calculate phi-based resonance
            resonance = calculate_phi_resonance(frequency, opt_freq)
            
            # Update factor if better match found
            optimal_factor = max(optimal_factor, resonance)
            
        # Calculate lattice resonance factor
        # Crystals have specific lattice frequencies that enhance response
        wavelength = self.speed_of_sound / frequency
        
        # Calculate if wavelength matches crystal dimensions through phi-relationship
        dimension_match = 0.0
        for dim in self.dimensions:
            # Check various harmonic relationships
            for n in range(1, 6):
                for phi_power in [1.0, PHI, PHI*PHI]:
                    # Calculate match score for this harmonic
                    expected_wavelength = 2 * dim / (n * phi_power)
                    match = 1.0 / (1.0 + abs(wavelength/expected_wavelength - 1.0) * 5.0)
                    
                    # Update best match
                    dimension_match = max(dimension_match, match)
        
        # Calculate geometric resonance based on facet count
        geometric_factor = 1.0
        if self.facet_count > 0:
            # Frequencies that match geometric pattern enhance response
            for n in range(1, 4):
                facet_freq = self.resonant_frequency * self.facet_count / (2 * n)
                facet_match = 1.0 / (1.0 + abs(frequency/facet_freq - 1.0) * 10.0)
                geometric_factor = max(geometric_factor, facet_match)
        
        # Apply crystalline quality factor
        quality_factor = 0.5 + 0.5 * self.crystal_quality
        lattice_factor = 0.5 + 0.5 * self.lattice_alignment
        
        # Combine factors with phi-weighted averaging
        amplified_response = (
            base_response * 1.0 + 
            optimal_factor * PHI + 
            dimension_match * PHI_INVERSE + 
            geometric_factor * LAMBDA
        ) / (1.0 + PHI + PHI_INVERSE + LAMBDA)
        
        # Apply quality factors
        amplified_response = amplified_response * quality_factor * lattice_factor
        
        return min(max(amplified_response, 0.0), 1.0)  # Clamp to [0, 1]
    
    def set_crystal_geometry(self, facet_count: int, lattice_alignment: float):
        """
        Set the crystal's geometric properties.
        
        Args:
            facet_count: Number of crystal facets
            lattice_alignment: Quality of lattice alignment (0 to 1)
        """
        self.facet_count = max(1, facet_count)
        self.lattice_alignment = max(0.0, min(1.0, lattice_alignment))
    
    def calculate_resonant_nodes(self) -> np.ndarray:
        """
        Calculate the resonant nodes within the crystal volume.
        
        Returns:
            3D numpy array showing node intensity
        """
        # Initialize grid if needed
        if self.field_grid is None:
            self.initialize_field_grid()
            
        # Calculate node map (where amplitude is close to zero)
        amplitudes = np.abs(self.field_grid)
        node_map = 1.0 - amplitudes
        
        # Apply threshold
        node_threshold = 0.7
        node_map = (node_map > node_threshold).astype(float)
        
        # Apply lattice alignment factor
        node_map = node_map * self.lattice_alignment
        
        return node_map

class MetalResonator(MaterialResonator):
    """
    A resonance chamber designed for metal-based cymatic patterns.
    """
    
    def __init__(
        self,
        name: str = "Metal Resonator",
        resonant_frequency: float = SACRED_FREQUENCIES['truth'],  # 672 Hz
        dimensions: Tuple[float, float, float] = (0.20, 0.20, 0.002),
        metal_type: str = "steel"
    ):
        """
        Initialize a metal-based resonance chamber.
        
        Args:
            name: Name of the resonator
            resonant_frequency: Primary resonant frequency in Hz
            dimensions: Metal plate dimensions (width, length, thickness) in meters
            metal_type: Type of metal ("steel", "copper", "aluminum", etc.)
        """
        # Metals have very sharp resonances
        q_factor = 40.0
        
        # Initialize base resonator
        super().__init__(name, resonant_frequency, q_factor, dimensions)
        
        # Metal-specific properties
        self.metal_type = metal_type
        
        # Set metal properties based on type
        self._set_metal_properties()
        
        # Plate properties
        self.tension = 1.0  # Tension factor
        self.mounting_points = 4  # How many points the plate is mounted at
        self.mounting_quality = 0.9  # How well mounted (0 to 1)
        
        # Update wavelength and resonant modes based on metal properties
        self.wavelength = self.speed_of_sound / self.resonant_frequency
        self.resonant_modes = self._calculate_resonant_modes()
    
    def _set_metal_properties(self):
        """Set physical properties based on metal type."""
        # Properties: density (kg/m³), speed of sound (m/s), optimal frequencies, Young's modulus (Pa)
        metal_properties = {
            'steel': {
                'density': 7850.0,
                'speed_of_sound': 5130.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['truth'],    # 672 Hz - truth expression
                    SACRED_FREQUENCIES['vision'],   # 720 Hz - vision/clarity
                    789.0,  # Steel plate harmonic
                    396.0   # Liberation frequency
                ],
                'youngs_modulus': 200e9
            },
            'copper': {
                'density': 8960.0,
                'speed_of_sound': 3810.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['cascade'],  # 594 Hz - heart resonance
                    SACRED_FREQUENCIES['love'],     # 528 Hz - healing/DNA
                    417.0,  # Transformation
                    639.0   # Heart/thought connection
                ],
                'youngs_modulus': 130e9
            },
            'aluminum': {
                'density': 2700.0,
                'speed_of_sound': 6320.0,
                'optimal_frequencies': [
                    SACRED_FREQUENCIES['truth'],    # 672 Hz - truth expression
                    SACRED_FREQUENCIES['oneness'],  # 768 Hz - unity consciousness
                    852.0,  # High intuition
                    741.0   # Consciousness awakening
                ],
                'youngs_modulus': 69e9
            }
        }
        
        # Use steel as default if metal type not found
        props = metal_properties.get(self.metal_type, metal_properties['steel'])
        
        # Set properties
        self.density = props['density']
        self.speed_of_sound = props['speed_of_sound']
        self.optimal_frequencies = props['optimal_frequencies']
        self.youngs_modulus = props['youngs_modulus']
    
    def _calculate_resonant_modes(self) -> List[Tuple[int, int, int]]:
        """
        Calculate resonant modes for a metal plate.
        
        Returns:
            List of (nx, ny, nz) mode indices for resonant frequencies
        """
        # For a rectangular plate, the mode frequencies follow a different formula
        width, length, thickness = self.dimensions
        
        # Maximum mode numbers to consider
        max_nx, max_ny = 10, 10
        
        # Calculate plate stiffness
        poisson_ratio = 0.3  # Approximate for most metals
        stiffness = self.youngs_modulus * thickness**3 / (12 * (1 - poisson_ratio**2))
        
        # List to store resonant modes
        modes = []
        
        # Calculate modes
        for nx in range(1, max_nx + 1):
            for ny in range(1, max_ny + 1):
                # Calculate mode frequency for a rectangular plate
                freq = (np.pi/2) * np.sqrt(stiffness / (self.density * thickness)) * \
                       np.sqrt((nx/width)**2 + (ny/length)**2)
                
                # Add mode
                modes.append((nx, ny, 1))
                
                # Check if this mode is near a phi-harmonic of the fundamental
                for _, harmonic_freq in self.harmonic_frequencies.items():
                    # Calculate frequency ratio
                    ratio = freq / harmonic_freq
                    
                    # Check if close to a phi-harmonic integer ratio
                    for i in range(1, 5):
                        if abs(ratio - i*PHI) < 0.1 or abs(ratio - i*LAMBDA) < 0.1:
                            # This is a phi-resonant mode, give it priority
                            modes.append((nx, ny, 2))  # Add duplicate to increase weight
                            break
        
        return modes
    
    def _apply_material_amplification(self, base_response: float, frequency: float) -> float:
        """
        Apply metal-specific amplification to the resonance response.
        
        Args:
            base_response: Base response amplitude (0 to 1)
            frequency: Input frequency in Hz
            
        Returns:
            Amplified response (0 to 1)
        """
        # Check if frequency is close to metal's optimal frequencies
        optimal_factor = 0.0
        for opt_freq in self.optimal_frequencies:
            # Calculate phi-based resonance
            resonance = calculate_phi_resonance(frequency, opt_freq)
            
            # Update factor if better match found
            optimal_factor = max(optimal_factor, resonance)
            
        # Calculate plate resonance factor
        # The chladni patterns depend strongly on plate mode matching
        plate_match = 0.0
        for mode in self.resonant_modes:
            nx, ny, _ = mode
            
            # Calculate mode frequency
            width, length, thickness = self.dimensions
            poisson_ratio = 0.3  # Approximate for most metals
            stiffness = self.youngs_modulus * thickness**3 / (12 * (1 - poisson_ratio**2))
            
            mode_freq = (np.pi/2) * np.sqrt(stiffness / (self.density * thickness)) * \
                       np.sqrt((nx/width)**2 + (ny/length)**2)
            
            # Calculate match with this mode
            match = 1.0 / (1.0 + abs(frequency/mode_freq - 1.0) * self.q_factor)
            
            # Update best match
            plate_match = max(plate_match, match)
            
        # Account for plate tension and mounting
        tension_factor = 0.5 + 0.5 * self.tension
        
        # Mounting factor depends on both number of points and quality
        if self.mounting_points == 0:
            mounting_factor = 0.2  # Free plate has poor pattern definition
        else:
            # Optimal mounting points depend on the pattern
            optimal_points = min(max(4, nx * ny), 8)
            point_match = 1.0 / (1.0 + abs(self.mounting_points - optimal_points) * 0.2)
            mounting_factor = point_match * self.mounting_quality
            
        # Combine factors with phi-weighted averaging
        amplified_response = (
            base_response * 1.0 + 
            optimal_factor * LAMBDA + 
            plate_match * PHI +
            tension_factor * LAMBDA * PHI +
            mounting_factor * 1.0
        ) / (1.0 + LAMBDA + PHI + LAMBDA*PHI + 1.0)
        
        return min(max(amplified_response, 0.0), 1.0)  # Clamp to [0, 1]
    
    def set_plate_properties(self, tension: float, mounting_points: int, mounting_quality: float):
        """
        Set the properties of the metal plate.
        
        Args:
            tension: Plate tension factor (0 to 1)
            mounting_points: Number of mounting points (0 for free plate)
            mounting_quality: Quality of mounting (0 to 1)
        """
        self.tension = max(0.0, min(1.0, tension))
        self.mounting_points = max(0, mounting_points)
        self.mounting_quality = max(0.0, min(1.0, mounting_quality))
        
        # Recalculate resonant modes as mounting affects resonance
        self.resonant_modes = self._calculate_resonant_modes()
    
    def calculate_chladni_pattern(self, frequency: float) -> np.ndarray:
        """
        Calculate the Chladni pattern for a specific frequency.
        
        Args:
            frequency: Input frequency in Hz
            
        Returns:
            2D numpy array showing the Chladni pattern
        """
        # Get dimensions
        width, length, _ = self.dimensions
        resolution = (100, 100)
        
        # Create coordinate grid
        x = np.linspace(0, width, resolution[0])
        y = np.linspace(0, length, resolution[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Find closest resonant modes
        modes = []
        mode_weights = []
        
        for mode in self.resonant_modes:
            nx, ny, importance = mode
            
            # Calculate mode frequency
            poisson_ratio = 0.3
            thickness = self.dimensions[2]
            stiffness = self.youngs_modulus * thickness**3 / (12 * (1 - poisson_ratio**2))
            
            mode_freq = (np.pi/2) * np.sqrt(stiffness / (self.density * thickness)) * \
                       np.sqrt((nx/width)**2 + (ny/length)**2)
            
            # Calculate match with this mode
            match = 1.0 / (1.0 + abs(frequency/mode_freq - 1.0) * self.q_factor)
            
            # If good match, add to active modes
            if match > 0.1:
                modes.append((nx, ny))
                
                # Weight by match quality and mode importance
                mode_weights.append(match * importance)
        
        # Normalize weights
        if mode_weights:
            weight_sum = sum(mode_weights)
            if weight_sum > 0:
                mode_weights = [w / weight_sum for w in mode_weights]
        
        # If no matching modes, use simple circular pattern
        if not modes:
            R = np.sqrt((X - width/2)**2 + (Y - length/2)**2)
            k = 2 * np.pi * frequency / self.speed_of_sound
            pattern = np.sin(k * R)
            return pattern
            
        # Calculate Chladni pattern as superposition of modes
        pattern = np.zeros_like(X)
        
        for (nx, ny), weight in zip(modes, mode_weights):
            # Calculate mode shape
            mode_shape = np.sin(nx * np.pi * X / width) * np.sin(ny * np.pi * Y / length)
            
            # Add to pattern with weight
            pattern += mode_shape * weight
            
        # For Chladni patterns, the nodes (zero crossings) form the patterns
        # Convert to absolute value and invert
        chladni = 1.0 - np.abs(pattern) / np.max(np.abs(pattern) + 1e-10)
        
        # Apply threshold to highlight nodal lines
        chladni = (chladni > 0.8).astype(float)
        
        return chladni