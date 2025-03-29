"""
Frequency Modulator Module

Implements phi-harmonic frequency modulation for precise cymatic pattern control,
enabling the transformation of consciousness into material form through sound.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
sys.path.append('/mnt/d/projects/Python')
from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES, 
    calculate_phi_resonance, phi_harmonic
)

class FrequencyModulator:
    """
    A system for modulating frequencies with phi-harmonic precision to create
    specific cymatic patterns in various materials.
    
    This class enables the precise control of frequency patterns, creating
    standing waves and interference patterns that can directly influence
    physical matter through resonance principles.
    """
    
    def __init__(
        self, 
        base_frequency: float = SACRED_FREQUENCIES['unity'],
        modulation_depth: float = 0.3,
        phi_harmonic_count: int = 5
    ):
        """
        Initialize a new FrequencyModulator.
        
        Args:
            base_frequency: The fundamental frequency in Hz (default: 432Hz)
            modulation_depth: Depth of frequency modulation (0.0 to 1.0)
            phi_harmonic_count: Number of phi harmonics to include
        """
        self.base_frequency = base_frequency
        self.modulation_depth = modulation_depth
        self.phi_harmonic_count = phi_harmonic_count
        
        # Current active frequency
        self.current_frequency = base_frequency
        
        # Modulation properties
        self.modulation_rate = 0.0  # Hz
        self.modulation_waveform = 'sine'  # sine, triangle, square
        self.phi_weight = 1.0  # Weight of phi component in modulation
        
        # Initialize frequency spectrum
        self.harmonics = self._initialize_harmonics()
        
        # Material-specific resonance properties
        self.material_resonances = {
            'water': {
                'primary_freq': 432.0,
                'sensitivity': 0.95,
                'resonance_q': 12.0,  # Q factor for resonance sharpness
                'harmonic_weights': [1.0, 0.7, 0.5, 0.3, 0.2]
            },
            'crystal': {
                'primary_freq': 528.0,
                'sensitivity': 0.85,
                'resonance_q': 25.0,
                'harmonic_weights': [0.8, 1.0, 0.9, 0.7, 0.5]
            },
            'metal': {
                'primary_freq': 672.0,
                'sensitivity': 0.75,
                'resonance_q': 40.0,
                'harmonic_weights': [0.6, 0.8, 1.0, 0.9, 0.7]
            }
        }
        
        # Time-based properties
        self.time = 0.0  # Internal time counter
        self.time_step = 0.01  # Time increment per update
        
        # Frequency history for pattern stability
        self.frequency_history = []
        self.max_history_length = 100
    
    def _initialize_harmonics(self) -> Dict[str, float]:
        """
        Initialize harmonic frequencies based on phi relationships.
        
        Returns:
            Dictionary of harmonic frequencies
        """
        harmonics = {
            'base': self.base_frequency,
            'phi': self.base_frequency * PHI,
            'phi_squared': self.base_frequency * PHI * PHI,
            'phi_inverse': self.base_frequency * LAMBDA,
            'phi_cubed': self.base_frequency * PHI * PHI * PHI,
            'octave': self.base_frequency * 2.0,
            'octave_phi': self.base_frequency * 2.0 * PHI,
            'fifth': self.base_frequency * 1.5,
            'fourth': self.base_frequency * 4.0/3.0
        }
        
        # Add numbered phi harmonics
        for i in range(1, self.phi_harmonic_count + 1):
            harmonics[f'phi_{i}'] = self.base_frequency * (PHI ** i)
            harmonics[f'phi_neg_{i}'] = self.base_frequency * (PHI ** -i)
        
        return harmonics
    
    def set_base_frequency(self, frequency: float):
        """
        Set the base frequency and recalculate harmonics.
        
        Args:
            frequency: New base frequency in Hz
        """
        self.base_frequency = frequency
        self.current_frequency = frequency
        self.harmonics = self._initialize_harmonics()
        
        # Add to frequency history
        self.frequency_history.append(frequency)
        if len(self.frequency_history) > self.max_history_length:
            self.frequency_history = self.frequency_history[-self.max_history_length:]
    
    def set_modulation_parameters(
        self, 
        rate: float, 
        depth: float, 
        waveform: str = 'sine',
        phi_weight: float = 1.0
    ):
        """
        Set frequency modulation parameters.
        
        Args:
            rate: Modulation rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            waveform: Type of modulation waveform ('sine', 'triangle', 'square')
            phi_weight: Weight of phi-based modulation (0.0 to 1.0)
        """
        self.modulation_rate = rate
        self.modulation_depth = max(0.0, min(1.0, depth))
        self.modulation_waveform = waveform.lower()
        self.phi_weight = max(0.0, min(1.0, phi_weight))
    
    def update(self, dt: float = None):
        """
        Update the frequency modulator state based on time increment.
        
        Args:
            dt: Time increment in seconds (uses internal time_step if None)
        """
        if dt is None:
            dt = self.time_step
            
        # Update internal time
        self.time += dt
        
        # Apply modulation if active
        if self.modulation_rate > 0:
            # Calculate modulation phase
            phase = self.time * self.modulation_rate * 2 * np.pi
            
            # Generate modulation signal based on waveform type
            if self.modulation_waveform == 'sine':
                mod_signal = np.sin(phase)
            elif self.modulation_waveform == 'triangle':
                # Triangle wave
                mod_signal = 2 * np.abs(2 * (phase/(2*np.pi) - np.floor(phase/(2*np.pi) + 0.5))) - 1
            elif self.modulation_waveform == 'square':
                # Square wave
                mod_signal = np.sign(np.sin(phase))
            else:  # Default to sine
                mod_signal = np.sin(phase)
            
            # Apply phi-based modulation component
            if self.phi_weight > 0:
                # Create phi-weighted composite modulation
                phi_phase = phase * PHI
                phi_mod = np.sin(phi_phase)
                
                # Mix with standard modulation
                mod_signal = (
                    (1 - self.phi_weight) * mod_signal + 
                    self.phi_weight * phi_mod
                )
            
            # Apply modulation to frequency
            self.current_frequency = self.base_frequency * (
                1.0 + self.modulation_depth * mod_signal
            )
            
            # Add to frequency history
            self.frequency_history.append(self.current_frequency)
            if len(self.frequency_history) > self.max_history_length:
                self.frequency_history = self.frequency_history[-self.max_history_length:]
    
    def get_current_frequency(self) -> float:
        """
        Get the current modulated frequency.
        
        Returns:
            Current frequency in Hz
        """
        return self.current_frequency
    
    def get_harmonic_frequencies(self) -> Dict[str, float]:
        """
        Get all harmonic frequencies based on current frequency.
        
        Returns:
            Dictionary of harmonic frequencies
        """
        # Recalculate harmonics based on current frequency
        harmonics = {}
        for name, base_harmonic in self.harmonics.items():
            ratio = base_harmonic / self.base_frequency
            harmonics[name] = self.current_frequency * ratio
            
        return harmonics
    
    def get_phi_harmonic_stack(self, count: int = 5) -> List[float]:
        """
        Get a stack of phi-harmonic frequencies based on current frequency.
        
        Args:
            count: Number of harmonics to include
            
        Returns:
            List of frequencies in the phi-harmonic stack
        """
        stack = [self.current_frequency]
        
        # Add positive phi powers
        for i in range(1, count + 1):
            stack.append(self.current_frequency * (PHI ** i))
            
        # Add negative phi powers
        for i in range(1, count + 1):
            stack.append(self.current_frequency * (PHI ** -i))
            
        return sorted(stack)
    
    def generate_cymatic_pattern(
        self, 
        size: Tuple[int, int] = (100, 100),
        harmonics: Optional[List[float]] = None,
        harmonic_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate a 2D cymatic pattern based on current frequency.
        
        Args:
            size: Size of the 2D array to generate (width, height)
            harmonics: Specific harmonic frequencies to use (uses phi harmonics if None)
            harmonic_weights: Weights for each harmonic (equal weights if None)
            
        Returns:
            2D numpy array representing the cymatic pattern
        """
        # Default to phi harmonics if none provided
        if harmonics is None:
            harmonics = [
                self.current_frequency,
                self.current_frequency * PHI,
                self.current_frequency * PHI * PHI,
                self.current_frequency * PHI_PHI,
                self.current_frequency * 2.0
            ]
        
        # Default to equal weights if none provided
        if harmonic_weights is None:
            # Phi-decreasing weights
            harmonic_weights = [PHI ** -i for i in range(len(harmonics))]
            # Normalize weights
            weight_sum = sum(harmonic_weights)
            harmonic_weights = [w / weight_sum for w in harmonic_weights]
        
        # Ensure we have weights for all harmonics
        if len(harmonic_weights) != len(harmonics):
            # Extend or truncate weights list to match harmonics
            if len(harmonic_weights) < len(harmonics):
                # Extend with zeros
                harmonic_weights.extend([0.0] * (len(harmonics) - len(harmonic_weights)))
            else:
                # Truncate
                harmonic_weights = harmonic_weights[:len(harmonics)]
        
        # Create coordinate grid
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate radial distance from center
        R = np.sqrt(X**2 + Y**2)
        
        # Generate pattern as superposition of standing waves
        pattern = np.zeros(size)
        
        for freq, weight in zip(harmonics, harmonic_weights):
            # Calculate wavelength (assuming speed of sound in air)
            wavelength = 343.0 / freq  # speed of sound / frequency
            
            # Scale wavelength to grid size
            scaled_wavelength = wavelength / 10.0  # arbitrary scaling factor
            
            # Wave number
            k = 2 * np.pi / scaled_wavelength
            
            # Standing wave pattern
            wave = np.sin(k * R) * weight
            
            # Add to composite pattern
            pattern += wave
        
        # Normalize pattern
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
        
        return pattern
    
    def generate_material_pattern(
        self, 
        material: str,
        size: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Generate a cymatic pattern optimized for a specific material.
        
        Args:
            material: Material type ('water', 'crystal', 'metal')
            size: Size of the 2D pattern array
            
        Returns:
            2D numpy array containing the cymatic pattern
        """
        if material not in self.material_resonances:
            # Default to water if material not recognized
            material = 'water'
            
        # Get material properties
        mat_props = self.material_resonances[material]
        
        # Calculate optimal frequency as blend of current and material's preferred frequency
        optimal_freq = (
            self.current_frequency * 0.3 + 
            mat_props['primary_freq'] * 0.7
        )
        
        # Generate harmonics based on material's preferred resonance
        harmonics = [optimal_freq]
        for i in range(1, len(mat_props['harmonic_weights'])):
            harmonics.append(optimal_freq * (PHI ** i))
            
        # Create pattern
        pattern = self.generate_cymatic_pattern(
            size=size,
            harmonics=harmonics,
            harmonic_weights=mat_props['harmonic_weights']
        )
        
        # Apply material-specific resonance sharpening
        # Higher Q factor means sharper resonance peaks
        if mat_props['resonance_q'] > 0:
            # Simple resonance sharpening via non-linear transformation
            q_factor = mat_props['resonance_q'] / 10.0  # Normalize Q factor
            sharpened = np.sign(pattern) * (np.abs(pattern) ** (1.0 / (1.0 + q_factor)))
            
            # Blend with original pattern
            pattern = pattern * 0.3 + sharpened * 0.7
            
            # Renormalize
            if np.max(np.abs(pattern)) > 0:
                pattern = pattern / np.max(np.abs(pattern))
        
        return pattern
    
    def get_frequency_stability(self) -> float:
        """
        Calculate stability of the frequency over time.
        
        Returns:
            Stability metric between 0 and 1
        """
        if len(self.frequency_history) < 2:
            return 1.0  # Perfectly stable if not enough history
            
        # Calculate standard deviation of frequency history
        std_dev = np.std(self.frequency_history)
        
        # Calculate stability as inverse of normalized standard deviation
        mean_freq = np.mean(self.frequency_history)
        
        if mean_freq > 0:
            relative_std = std_dev / mean_freq
            stability = 1.0 / (1.0 + relative_std * 10.0)  # Scale factor to get 0-1 range
        else:
            stability = 0.0
            
        return min(max(stability, 0.0), 1.0)  # Clamp to [0, 1]
    
    def create_frequency_progression(
        self, 
        duration: float, 
        target_frequency: float, 
        step_count: int = 8
    ) -> List[float]:
        """
        Create a phi-harmonic frequency progression between current and target frequency.
        
        Args:
            duration: Duration of the progression in seconds
            target_frequency: Target frequency to reach
            step_count: Number of steps in the progression
            
        Returns:
            List of frequencies in the progression
        """
        # Convert to phi-based logarithmic scale
        start_log_phi = np.log(self.current_frequency) / np.log(PHI)
        target_log_phi = np.log(target_frequency) / np.log(PHI)
        
        # Calculate step size in phi-logarithmic space
        step_size = (target_log_phi - start_log_phi) / (step_count - 1)
        
        # Create progression
        progression = []
        for i in range(step_count):
            # Calculate phi-logarithmic value
            log_value = start_log_phi + step_size * i
            
            # Convert back to frequency
            freq = PHI ** log_value
            progression.append(freq)
            
        # Calculate time points
        time_points = np.linspace(0, duration, step_count)
        
        # Combine into frequency-time pairs
        return list(zip(progression, time_points))
    
    def apply_frequency_progression(
        self, 
        progression: List[Tuple[float, float]]
    ):
        """
        Apply a frequency progression over time.
        
        Args:
            progression: List of (frequency, time) pairs
        """
        # Extract frequencies and times
        frequencies = [p[0] for p in progression]
        times = [p[1] for p in progression]
        
        # Set up modulation
        self.modulation_waveform = 'progression'
        self._progression_data = {
            'frequencies': frequencies,
            'times': times,
            'start_time': self.time,
            'current_index': 0
        }
    
    def calculate_resonance_for_material(self, material: str) -> float:
        """
        Calculate how well the current frequency resonates with a given material.
        
        Args:
            material: Target material type
            
        Returns:
            Resonance value between 0 and 1
        """
        if material not in self.material_resonances:
            return 0.5  # Default resonance for unknown materials
            
        # Get material's primary resonant frequency
        mat_freq = self.material_resonances[material]['primary_freq']
        
        # Calculate phi-based resonance between current frequency and material frequency
        resonance = calculate_phi_resonance(self.current_frequency, mat_freq)
        
        # Scale by material sensitivity
        sensitivity = self.material_resonances[material]['sensitivity']
        resonance = resonance * sensitivity
        
        return min(max(resonance, 0.0), 1.0)  # Clamp to [0, 1]
    
    def get_optimal_frequencies_for_pattern(
        self, 
        target_pattern: np.ndarray,
        frequency_range: Tuple[float, float] = (20, 2000),
        step_count: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Find optimal frequencies to recreate a target cymatic pattern.
        
        Args:
            target_pattern: 2D numpy array representing the target pattern
            frequency_range: Range of frequencies to test
            step_count: Number of frequency steps to test
            
        Returns:
            List of (frequency, similarity) pairs, sorted by similarity
        """
        # Normalize target pattern
        if np.max(np.abs(target_pattern)) > 0:
            target_norm = target_pattern / np.max(np.abs(target_pattern))
        else:
            target_norm = target_pattern
            
        # Create frequencies to test (phi-logarithmic spacing)
        start_log = np.log(frequency_range[0]) / np.log(PHI)
        end_log = np.log(frequency_range[1]) / np.log(PHI)
        
        log_steps = np.linspace(start_log, end_log, step_count)
        test_frequencies = [PHI ** log_val for log_val in log_steps]
        
        # Test each frequency
        results = []
        for freq in test_frequencies:
            # Generate pattern at this frequency
            test_pattern = self.generate_cymatic_pattern(
                size=target_pattern.shape,
                harmonics=[freq, freq*PHI, freq*PHI*PHI]
            )
            
            # Calculate similarity with target pattern
            similarity = self._calculate_pattern_similarity(test_pattern, target_norm)
            
            # Store result
            results.append((freq, similarity))
            
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _calculate_pattern_similarity(
        self, 
        pattern1: np.ndarray, 
        pattern2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity value between 0 and 1
        """
        # Ensure patterns are the same size
        if pattern1.shape != pattern2.shape:
            # Resize to smaller shape
            min_shape = (
                min(pattern1.shape[0], pattern2.shape[0]),
                min(pattern1.shape[1], pattern2.shape[1])
            )
            p1 = pattern1[:min_shape[0], :min_shape[1]]
            p2 = pattern2[:min_shape[0], :min_shape[1]]
        else:
            p1 = pattern1
            p2 = pattern2
            
        # Normalize patterns
        if np.max(np.abs(p1)) > 0:
            p1 = p1 / np.max(np.abs(p1))
        if np.max(np.abs(p2)) > 0:
            p2 = p2 / np.max(np.abs(p2))
            
        # Calculate difference
        diff = np.abs(p1 - p2)
        mean_diff = np.mean(diff)
        
        # Calculate similarity
        similarity = 1.0 - mean_diff
        
        return max(min(similarity, 1.0), 0.0)  # Clamp to [0, 1]