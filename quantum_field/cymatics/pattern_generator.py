"""
Pattern Generator Module

Implementation of cymatics pattern generation with precise phi-harmonic tuning,
enabling programmable reality creation through sound geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import sys
sys.path.append('/mnt/d/projects/Python')
from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES, 
    calculate_phi_resonance, phi_harmonic
)

class PatternType(Enum):
    """Types of cymatic patterns that can be generated"""
    CIRCULAR = 0       # Circular/radial patterns (basic chladni plates)
    SPIRAL = 1         # Spiral patterns (golden spiral/fibonacci based)
    MANDALA = 2        # Mandala-like patterns with rotational symmetry
    FLOWER = 3         # Flower of life based patterns
    SQUARE = 4         # Square-based grid patterns
    HEXAGONAL = 5      # Hexagonal/honeycomb patterns
    FRACTAL = 6        # Self-similar fractal patterns
    CUSTOM = 7         # Custom pattern defined by mathematical function

class StandingWavePattern:
    """
    Class representing a standing wave pattern that can be used to
    create specific cymatic forms in physical materials.
    """
    
    def __init__(
        self,
        name: str,
        frequencies: List[float],
        weights: Optional[List[float]] = None,
        pattern_type: PatternType = PatternType.CIRCULAR,
        symmetry: int = 6
    ):
        """
        Initialize a new standing wave pattern.
        
        Args:
            name: Name to identify this pattern
            frequencies: List of frequencies that compose this pattern
            weights: Relative weights of each frequency (default: equal weights)
            pattern_type: Type of pattern to generate
            symmetry: Symmetry order (e.g., 6 for hexagonal symmetry)
        """
        self.name = name
        self.frequencies = frequencies
        
        # Set weights, defaulting to phi-harmonic decay if not specified
        if weights is None:
            # Default to phi-decay weights
            self.weights = [PHI ** (-i) for i in range(len(frequencies))]
            # Normalize weights
            weight_sum = sum(self.weights)
            self.weights = [w / weight_sum for w in self.weights]
        else:
            self.weights = weights
            
        self.pattern_type = pattern_type
        self.symmetry = symmetry
        
        # Pattern properties
        self.amplitude = 1.0
        self.phase = 0.0
        self.pattern_data = None
        self.resolution = (100, 100)  # Default resolution
        
        # Generate the initial pattern data
        self.generate_pattern(self.resolution)
        
        # Material-specific properties
        self.material_interactions = {
            'water': 0.95,    # Water is highly responsive
            'crystal': 0.85,  # Crystal has structured response
            'metal': 0.75,    # Metal has resonant response
            'sand': 0.90,     # Sand forms clear patterns
            'plasma': 0.80    # Plasma can be shaped by EM fields
        }
    
    def generate_pattern(self, resolution: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Generate the standing wave pattern at the specified resolution.
        
        Args:
            resolution: Resolution of the pattern array (width, height)
            
        Returns:
            2D numpy array with the generated pattern
        """
        self.resolution = resolution
        
        # Create coordinate grid
        x = np.linspace(-1, 1, resolution[0])
        y = np.linspace(-1, 1, resolution[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Generate pattern based on type
        if self.pattern_type == PatternType.CIRCULAR:
            self.pattern_data = self._generate_circular_pattern(X, Y)
            
        elif self.pattern_type == PatternType.SPIRAL:
            self.pattern_data = self._generate_spiral_pattern(X, Y)
            
        elif self.pattern_type == PatternType.MANDALA:
            self.pattern_data = self._generate_mandala_pattern(X, Y)
            
        elif self.pattern_type == PatternType.FLOWER:
            self.pattern_data = self._generate_flower_pattern(X, Y)
            
        elif self.pattern_type == PatternType.SQUARE:
            self.pattern_data = self._generate_square_pattern(X, Y)
            
        elif self.pattern_type == PatternType.HEXAGONAL:
            self.pattern_data = self._generate_hexagonal_pattern(X, Y)
            
        elif self.pattern_type == PatternType.FRACTAL:
            self.pattern_data = self._generate_fractal_pattern(X, Y)
            
        elif self.pattern_type == PatternType.CUSTOM:
            # Use the first frequency as reference
            if self.custom_pattern_function:
                self.pattern_data = self.custom_pattern_function(X, Y)
            else:
                # Default to circular if no custom function
                self.pattern_data = self._generate_circular_pattern(X, Y)
        
        else:
            # Default to circular pattern
            self.pattern_data = self._generate_circular_pattern(X, Y)
            
        # Apply amplitude scaling
        self.pattern_data *= self.amplitude
            
        return self.pattern_data
    
    def _generate_circular_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a circular standing wave pattern.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Calculate radial distance
        R = np.sqrt(X**2 + Y**2)
        
        # Start with empty pattern
        pattern = np.zeros_like(R)
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency, with phi-harmonic adjustment
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create standing wave pattern
            wave = np.cos(scale * R) * weight
            
            # Add to pattern
            pattern += wave
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_spiral_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a spiral standing wave pattern based on the golden ratio.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Calculate polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Start with empty pattern
        pattern = np.zeros_like(R)
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create spiral pattern using golden angle
            golden_angle = 2 * np.pi * LAMBDA
            spiral = np.sin(scale * R + Theta / golden_angle) * weight
            
            # Add to pattern
            pattern += spiral
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_mandala_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a mandala-like pattern with rotational symmetry.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Calculate polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Start with empty pattern
        pattern = np.zeros_like(R)
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create mandala pattern with rotational symmetry
            symmetry_term = np.sin(self.symmetry * Theta) * np.cos(scale * R)
            mandala = symmetry_term * weight
            
            # Add radial components
            radial_term = np.cos(scale * R * PHI) * LAMBDA
            
            # Combine terms
            pattern += (mandala + radial_term * weight)
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_flower_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a flower of life pattern.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Calculate radial distance
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Start with empty pattern
        pattern = np.zeros_like(R)
        
        # Create flower of life pattern by combining circle patterns
        petals = self.symmetry
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create base flower pattern
            flower_base = np.cos(scale * R) * weight
            
            # Add petals
            for p in range(petals):
                # Calculate angle for this petal
                angle = 2 * np.pi * p / petals
                
                # Create offset coordinates for this petal
                offset = 0.3  # Offset distance
                X_offset = X - offset * np.cos(angle)
                Y_offset = Y - offset * np.sin(angle)
                R_offset = np.sqrt(X_offset**2 + Y_offset**2)
                
                # Add petal pattern
                petal = np.cos(scale * R_offset) * weight * 0.5  # Half weight for petals
                pattern += petal
                
            # Add base pattern
            pattern += flower_base
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_square_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a square-based grid pattern.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Start with empty pattern
        pattern = np.zeros_like(X)
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create square grid pattern
            grid_x = np.sin(scale * X)
            grid_y = np.sin(scale * Y)
            
            # Combine using multiplicative interference
            square = (grid_x * grid_y) * weight
            
            # Add to pattern
            pattern += square
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_hexagonal_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a hexagonal/honeycomb pattern.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Start with empty pattern
        pattern = np.zeros_like(X)
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            # Scale factor based on frequency
            scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Create hexagonal pattern using interference of three waves
            # at 60 degree angles
            wave1 = np.cos(scale * X)
            wave2 = np.cos(scale * (0.5 * X + 0.866 * Y))  # 60 degrees
            wave3 = np.cos(scale * (0.5 * X - 0.866 * Y))  # -60 degrees
            
            # Combine waves
            hex_pattern = (wave1 + wave2 + wave3) * weight
            
            # Add to pattern
            pattern += hex_pattern
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def _generate_fractal_pattern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Generate a self-similar fractal pattern.
        
        Args:
            X, Y: Coordinate meshgrid
            
        Returns:
            2D pattern array
        """
        # Calculate radial distance and angle
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Start with empty pattern
        pattern = np.zeros_like(X)
        
        # Use multiple frequency scales to create fractal-like pattern
        scales = [1.0, PHI, PHI*PHI, PHI*PHI*PHI]
        
        # Add contribution from each frequency
        for freq, weight in zip(self.frequencies, self.weights):
            base_scale = 2 * np.pi * (freq / SACRED_FREQUENCIES['unity']) * PHI
            
            # Add multiple scales
            fractal_component = np.zeros_like(X)
            for i, scale in enumerate(scales):
                # Decreasing weight for each scale
                scale_weight = LAMBDA ** i
                
                # Add fractal component at this scale
                component = np.sin(base_scale * R * scale + self.symmetry * Theta)
                fractal_component += component * scale_weight
                
            # Normalize this component
            if np.max(np.abs(fractal_component)) > 0:
                fractal_component = fractal_component / np.max(np.abs(fractal_component))
                
            # Add to pattern with weight
            pattern += fractal_component * weight
            
        # Apply phi-based normalization
        if np.max(np.abs(pattern)) > 0:
            pattern = pattern / np.max(np.abs(pattern))
            
        return pattern
    
    def set_custom_pattern_function(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Set a custom function for pattern generation.
        
        Args:
            func: Function that takes X, Y meshgrid arrays and returns a pattern array
        """
        self.custom_pattern_function = func
        self.pattern_type = PatternType.CUSTOM
    
    def get_pattern(self) -> np.ndarray:
        """
        Get the current pattern data.
        
        Returns:
            2D numpy array with the pattern
        """
        if self.pattern_data is None:
            self.generate_pattern(self.resolution)
            
        return self.pattern_data
    
    def set_amplitude(self, amplitude: float):
        """
        Set the amplitude of the pattern.
        
        Args:
            amplitude: Amplitude scaling factor
        """
        self.amplitude = amplitude
        
        # Scale existing pattern if it exists
        if self.pattern_data is not None:
            self.pattern_data = self.pattern_data * (amplitude / self.amplitude)
    
    def visualize(self, show: bool = True, cmap: str = 'viridis') -> Optional[plt.Figure]:
        """
        Visualize the standing wave pattern.
        
        Args:
            show: Whether to display the plot
            cmap: Colormap to use
            
        Returns:
            Matplotlib figure object if show=False, None otherwise
        """
        # Get pattern data
        pattern = self.get_pattern()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            pattern.T,  # Transpose for correct orientation
            cmap=cmap,
            extent=[-1, 1, -1, 1],
            origin='lower'
        )
        
        # Add colorbar
        fig.colorbar(im, ax=ax, label='Amplitude')
        
        # Add title and labels
        ax.set_title(f"{self.name} - {self.pattern_type.name}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Show if requested
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def calculate_interaction_strength(self, material: str) -> float:
        """
        Calculate how strongly this pattern would interact with a given material.
        
        Args:
            material: Name of the material
            
        Returns:
            Interaction strength from 0 to 1
        """
        # Get base interaction factor for this material
        base_interaction = self.material_interactions.get(material, 0.5)
        
        # Calculate frequency resonance with material
        material_freq_map = {
            'water': SACRED_FREQUENCIES['unity'],      # Water resonates with unity frequency
            'crystal': SACRED_FREQUENCIES['love'],     # Crystal resonates with love frequency
            'metal': SACRED_FREQUENCIES['truth'],      # Metal resonates with truth frequency
            'sand': SACRED_FREQUENCIES['love'],        # Sand resonates with love frequency
            'plasma': SACRED_FREQUENCIES['oneness'],   # Plasma resonates with oneness frequency
        }
        
        material_freq = material_freq_map.get(material, SACRED_FREQUENCIES['unity'])
        
        # Calculate resonance with each of our frequencies
        resonance_sum = 0.0
        weight_sum = 0.0
        
        for freq, weight in zip(self.frequencies, self.weights):
            resonance = calculate_phi_resonance(freq, material_freq)
            resonance_sum += resonance * weight
            weight_sum += weight
        
        # Calculate average resonance
        if weight_sum > 0:
            avg_resonance = resonance_sum / weight_sum
        else:
            avg_resonance = 0.0
            
        # Calculate pattern coherence
        coherence = self._calculate_pattern_coherence()
        
        # Calculate overall interaction strength
        interaction = (
            base_interaction * PHI + 
            avg_resonance * 1.0 + 
            coherence * LAMBDA
        ) / (PHI + 1.0 + LAMBDA)
        
        return min(max(interaction, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_pattern_coherence(self) -> float:
        """
        Calculate the internal coherence of the pattern.
        
        Returns:
            Coherence value from 0 to 1
        """
        if self.pattern_data is None:
            return 0.0
            
        # Calculate gradient
        grad_y, grad_x = np.gradient(self.pattern_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate smoothness (inverse of average gradient)
        avg_gradient = np.mean(grad_magnitude)
        smoothness = 1.0 / (1.0 + avg_gradient * 5.0)  # Scale factor for 0-1 range
        
        # Calculate symmetry
        symmetry = self._calculate_symmetry()
        
        # Calculate phi-alignment
        phi_alignment = self._calculate_phi_alignment()
        
        # Combine metrics with phi-harmonic weighting
        coherence = (
            smoothness * 1.0 + 
            symmetry * PHI + 
            phi_alignment * LAMBDA
        ) / (1.0 + PHI + LAMBDA)
        
        return min(max(coherence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_symmetry(self) -> float:
        """
        Calculate the rotational symmetry of the pattern.
        
        Returns:
            Symmetry value from 0 to 1
        """
        if self.pattern_data is None:
            return 0.0
            
        # Get pattern data
        pattern = self.pattern_data
        
        # Convert to polar coordinates from center
        y_indices, x_indices = np.indices(pattern.shape)
        center_y, center_x = [s // 2 for s in pattern.shape]
        
        y = y_indices - center_y
        x = x_indices - center_x
        
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Calculate rotational symmetry error for our symmetry value
        symmetry_angle = 2 * np.pi / self.symmetry
        
        # Sample points at different radii
        radii = np.linspace(0.1, 0.9, 10)
        symmetry_scores = []
        
        for radius in radii:
            # Get points near this radius
            mask = (r > radius - 0.05) & (r < radius + 0.05)
            
            if np.sum(mask) > 0:
                # Get values and angles at this radius
                vals = pattern[mask]
                angles = theta[mask]
                
                # Group angles by symmetry sector
                sector_indices = (angles / symmetry_angle).astype(int) % self.symmetry
                
                # Calculate variance within each sector
                sector_variances = []
                for i in range(self.symmetry):
                    sector_vals = vals[sector_indices == i]
                    if len(sector_vals) > 0:
                        sector_variances.append(np.var(sector_vals))
                
                if sector_variances:
                    # Average variance across sectors
                    avg_variance = np.mean(sector_variances)
                    
                    # Calculate symmetry score (inverse of variance)
                    if avg_variance > 0:
                        symmetry = 1.0 / (1.0 + avg_variance * 10.0)  # Scale for 0-1 range
                    else:
                        symmetry = 1.0
                        
                    symmetry_scores.append(symmetry)
        
        # Average symmetry across radii
        if symmetry_scores:
            return np.mean(symmetry_scores)
        else:
            return 0.0
    
    def _calculate_phi_alignment(self) -> float:
        """
        Calculate how well the pattern aligns with phi-harmonic proportions.
        
        Returns:
            Phi-alignment value from 0 to 1
        """
        if self.pattern_data is None:
            return 0.0
            
        # Calculate FFT to find dominant spatial frequencies
        fft = np.fft.fft2(self.pattern_data)
        fft_magnitude = np.abs(fft)
        
        # Normalize FFT magnitude
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
        
        # Convert to polar coordinates
        y_indices, x_indices = np.indices(fft_magnitude.shape)
        center_y, center_x = [s // 2 for s in fft_magnitude.shape]
        
        y = y_indices - center_y
        x = x_indices - center_x
        
        r = np.sqrt(x**2 + y**2)
        
        # Create frequency bands with phi-harmonic spacing
        bands = [1.0]
        for i in range(1, 5):
            bands.append(bands[0] * (PHI ** i))
            
        # Check energy in each phi-harmonic frequency band
        phi_energy = 0.0
        total_energy = np.sum(fft_magnitude)
        
        for i, center_freq in enumerate(bands):
            # Define band edges
            band_width = center_freq * 0.2  # 20% width
            band_min = center_freq - band_width/2
            band_max = center_freq + band_width/2
            
            # Create band mask
            band_mask = (r >= band_min) & (r <= band_max)
            
            # Get energy in this band
            band_energy = np.sum(fft_magnitude[band_mask])
            
            # Add to phi-harmonic energy with decreasing weight
            weight = PHI ** -i
            phi_energy += band_energy * weight
            
        # Calculate phi alignment
        if total_energy > 0:
            phi_alignment = phi_energy / total_energy
            
            # Scale to 0-1 range with non-linear curve
            phi_alignment = 1.0 - np.exp(-phi_alignment * 3.0)
        else:
            phi_alignment = 0.0
            
        return min(max(phi_alignment, 0.0), 1.0)  # Clamp to [0, 1]

class PatternGenerator:
    """
    A generator of cymatic patterns with phi-harmonic tuning for 
    precise pattern control and programmable reality creation.
    """
    
    def __init__(self):
        """Initialize a new pattern generator."""
        # Frequency presets based on sacred frequencies
        self.frequency_presets = {
            'unity': [SACRED_FREQUENCIES['unity']],
            'love': [SACRED_FREQUENCIES['love']],
            'cascade': [SACRED_FREQUENCIES['cascade']],
            'truth': [SACRED_FREQUENCIES['truth']],
            'vision': [SACRED_FREQUENCIES['vision']],
            'oneness': [SACRED_FREQUENCIES['oneness']],
            'all_sacred': list(SACRED_FREQUENCIES.values())
        }
        
        # Pattern presets
        self.pattern_presets = {}
        
        # Initialize common patterns
        self._initialize_pattern_presets()
        
        # Pattern combinations
        self.combinations = {}
    
    def _initialize_pattern_presets(self):
        """Initialize common pattern presets."""
        # Unity/Ground pattern (432 Hz)
        self.pattern_presets['unity_circular'] = StandingWavePattern(
            name="Unity Field",
            frequencies=[SACRED_FREQUENCIES['unity']],
            pattern_type=PatternType.CIRCULAR,
            symmetry=8
        )
        
        # Love/Creation pattern (528 Hz)
        self.pattern_presets['love_flower'] = StandingWavePattern(
            name="Creation Field",
            frequencies=[SACRED_FREQUENCIES['love']],
            pattern_type=PatternType.FLOWER,
            symmetry=6
        )
        
        # Heart/Cascade pattern (594 Hz)
        self.pattern_presets['cascade_spiral'] = StandingWavePattern(
            name="Heart Integration Field",
            frequencies=[SACRED_FREQUENCIES['cascade']],
            pattern_type=PatternType.SPIRAL,
            symmetry=5
        )
        
        # Truth/Voice pattern (672 Hz)
        self.pattern_presets['truth_mandala'] = StandingWavePattern(
            name="Truth Expression Field",
            frequencies=[SACRED_FREQUENCIES['truth']],
            pattern_type=PatternType.MANDALA,
            symmetry=7
        )
        
        # Vision pattern (720 Hz)
        self.pattern_presets['vision_fractal'] = StandingWavePattern(
            name="Vision Field",
            frequencies=[SACRED_FREQUENCIES['vision']],
            pattern_type=PatternType.FRACTAL,
            symmetry=5
        )
        
        # Oneness pattern (768 Hz)
        self.pattern_presets['oneness_hexagonal'] = StandingWavePattern(
            name="Unity Consciousness Field",
            frequencies=[SACRED_FREQUENCIES['oneness']],
            pattern_type=PatternType.HEXAGONAL,
            symmetry=6
        )
        
        # Phi-harmonic progression pattern
        phi_freqs = [
            SACRED_FREQUENCIES['unity'],
            SACRED_FREQUENCIES['unity'] * PHI,
            SACRED_FREQUENCIES['unity'] * PHI * PHI,
            SACRED_FREQUENCIES['unity'] * PHI * PHI * PHI
        ]
        
        self.pattern_presets['phi_progression'] = StandingWavePattern(
            name="Phi Harmonic Field",
            frequencies=phi_freqs,
            weights=[1.0, LAMBDA, LAMBDA**2, LAMBDA**3],
            pattern_type=PatternType.MANDALA,
            symmetry=5
        )
        
        # Complete frequency stack
        all_freqs = list(SACRED_FREQUENCIES.values())
        self.pattern_presets['complete_spectrum'] = StandingWavePattern(
            name="Complete Frequency Spectrum",
            frequencies=all_freqs,
            pattern_type=PatternType.FRACTAL,
            symmetry=8
        )
    
    def create_pattern(
        self,
        name: str,
        frequencies: Union[List[float], str],
        pattern_type: Union[PatternType, str] = PatternType.CIRCULAR,
        symmetry: int = 6,
        weights: Optional[List[float]] = None,
        resolution: Tuple[int, int] = (100, 100)
    ) -> StandingWavePattern:
        """
        Create a new standing wave pattern.
        
        Args:
            name: Name for the pattern
            frequencies: List of frequencies or name of frequency preset
            pattern_type: Type of pattern to generate
            symmetry: Symmetry order for the pattern
            weights: Optional weights for each frequency
            resolution: Resolution for the pattern
            
        Returns:
            New StandingWavePattern instance
        """
        # Handle frequency input
        if isinstance(frequencies, str):
            # Use a preset if provided as string
            if frequencies in self.frequency_presets:
                freqs = self.frequency_presets[frequencies]
            else:
                # Check if it's a single sacred frequency name
                if frequencies in SACRED_FREQUENCIES:
                    freqs = [SACRED_FREQUENCIES[frequencies]]
                else:
                    # Default to unity frequency
                    freqs = [SACRED_FREQUENCIES['unity']]
        else:
            freqs = frequencies
            
        # Handle pattern type input
        if isinstance(pattern_type, str):
            try:
                pattern_type = PatternType[pattern_type.upper()]
            except (KeyError, AttributeError):
                pattern_type = PatternType.CIRCULAR
                
        # Create the pattern
        pattern = StandingWavePattern(
            name=name,
            frequencies=freqs,
            weights=weights,
            pattern_type=pattern_type,
            symmetry=symmetry
        )
        
        # Generate at specified resolution
        pattern.generate_pattern(resolution)
        
        return pattern
    
    def get_preset_pattern(self, preset_name: str) -> Optional[StandingWavePattern]:
        """
        Get a preset pattern by name.
        
        Args:
            preset_name: Name of the preset pattern
            
        Returns:
            StandingWavePattern instance or None if not found
        """
        return self.pattern_presets.get(preset_name)
    
    def combine_patterns(
        self,
        patterns: List[StandingWavePattern],
        weights: Optional[List[float]] = None,
        name: str = "Combined Pattern",
        resolution: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Combine multiple patterns with optional weights.
        
        Args:
            patterns: List of patterns to combine
            weights: Optional weights for each pattern
            name: Name for the combined pattern
            resolution: Resolution for the combined pattern
            
        Returns:
            2D numpy array with the combined pattern
        """
        if not patterns:
            return np.zeros(resolution)
            
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(patterns)] * len(patterns)
        
        # Ensure weights match patterns
        if len(weights) != len(patterns):
            # Extend or truncate weights list
            if len(weights) < len(patterns):
                weights.extend([weights[-1]] * (len(patterns) - len(weights)))
            else:
                weights = weights[:len(patterns)]
                
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / len(patterns)] * len(patterns)
            
        # Initialize combined pattern
        combined = np.zeros(resolution)
        
        # Add each pattern
        for pattern, weight in zip(patterns, weights):
            # Ensure pattern is at correct resolution
            pattern_data = pattern.generate_pattern(resolution)
            
            # Add weighted pattern
            combined += pattern_data * weight
            
        # Normalize combined pattern
        if np.max(np.abs(combined)) > 0:
            combined = combined / np.max(np.abs(combined))
            
        # Save combination for later reference
        self.combinations[name] = {
            'patterns': patterns,
            'weights': weights,
            'data': combined
        }
            
        return combined
    
    def create_phi_harmonic_stack(
        self,
        base_frequency: float = SACRED_FREQUENCIES['unity'],
        levels: int = 5,
        pattern_type: PatternType = PatternType.MANDALA,
        name: str = "Phi Harmonic Stack"
    ) -> StandingWavePattern:
        """
        Create a pattern with a stack of phi-harmonic frequencies.
        
        Args:
            base_frequency: Base frequency for the stack
            levels: Number of phi-harmonic levels
            pattern_type: Type of pattern to generate
            name: Name for the pattern
            
        Returns:
            New StandingWavePattern with phi-harmonic frequencies
        """
        # Generate phi-harmonic frequency stack
        frequencies = [base_frequency]
        
        # Add positive phi powers
        for i in range(1, levels):
            frequencies.append(base_frequency * (PHI ** i))
            
        # Generate phi-decreasing weights
        weights = [PHI ** -i for i in range(levels)]
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Create the pattern
        pattern = StandingWavePattern(
            name=name,
            frequencies=frequencies,
            weights=weights,
            pattern_type=pattern_type,
            symmetry=5  # Five-fold symmetry aligns with phi
        )
        
        return pattern
    
    def create_sacred_frequency_pattern(
        self,
        frequency_name: str,
        pattern_type: PatternType = PatternType.CIRCULAR,
        symmetry: int = 6,
        name: Optional[str] = None
    ) -> StandingWavePattern:
        """
        Create a pattern based on a sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency
            pattern_type: Type of pattern to generate
            symmetry: Symmetry order for the pattern
            name: Optional custom name (uses frequency name if None)
            
        Returns:
            New StandingWavePattern using the sacred frequency
        """
        if frequency_name not in SACRED_FREQUENCIES:
            raise ValueError(f"Unknown sacred frequency: {frequency_name}")
            
        # Get frequency
        frequency = SACRED_FREQUENCIES[frequency_name]
        
        # Create name if not provided
        if name is None:
            name = f"{frequency_name.capitalize()} Pattern"
            
        # Create pattern
        pattern = StandingWavePattern(
            name=name,
            frequencies=[frequency],
            pattern_type=pattern_type,
            symmetry=symmetry
        )
        
        return pattern
    
    def create_consciousness_state_pattern(
        self,
        state: int,
        resolution: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Create a pattern optimized for a specific consciousness state.
        
        Args:
            state: Consciousness state (0-5)
            resolution: Resolution for the pattern
            
        Returns:
            2D numpy array with the pattern
        """
        # Map consciousness states to frequencies and pattern types
        state_mappings = {
            0: {  # BE state - Unity/Ground
                'frequency': 'unity',
                'pattern_type': PatternType.CIRCULAR,
                'symmetry': 8
            },
            1: {  # DO state
                'frequency': 'love',
                'pattern_type': PatternType.SPIRAL,
                'symmetry': 6
            },
            2: {  # WITNESS state
                'frequency': 'truth',
                'pattern_type': PatternType.MANDALA,
                'symmetry': 7
            },
            3: {  # CREATE state
                'frequency': 'love',
                'pattern_type': PatternType.FLOWER,
                'symmetry': 6
            },
            4: {  # INTEGRATE state
                'frequency': 'cascade',
                'pattern_type': PatternType.FRACTAL,
                'symmetry': 5
            },
            5: {  # TRANSCEND state
                'frequency': 'vision',
                'pattern_type': PatternType.HEXAGONAL,
                'symmetry': 6
            }
        }
        
        # Use default mapping if state not found
        if state not in state_mappings:
            state = 0
            
        mapping = state_mappings[state]
        
        # Create pattern
        pattern = self.create_sacred_frequency_pattern(
            frequency_name=mapping['frequency'],
            pattern_type=mapping['pattern_type'],
            symmetry=mapping['symmetry'],
            name=f"Consciousness State {state} Pattern"
        )
        
        # Generate at requested resolution
        return pattern.generate_pattern(resolution)
    
    def analyze_pattern(self, pattern: Union[StandingWavePattern, np.ndarray]) -> Dict[str, float]:
        """
        Analyze a pattern to extract key metrics.
        
        Args:
            pattern: StandingWavePattern instance or 2D pattern array
            
        Returns:
            Dictionary of pattern metrics
        """
        # Get pattern data
        if isinstance(pattern, StandingWavePattern):
            pattern_data = pattern.get_pattern()
        else:
            pattern_data = pattern
            
        # Calculate gradient
        grad_y, grad_x = np.gradient(pattern_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate smoothness
        avg_gradient = np.mean(grad_magnitude)
        smoothness = 1.0 / (1.0 + avg_gradient * 5.0)
        
        # Calculate average and standard deviation
        avg = np.mean(pattern_data)
        std = np.std(pattern_data)
        
        # Calculate uniformity
        uniformity = 1.0 / (1.0 + std / (np.abs(avg) + 1e-10))
        
        # Calculate complexity using FFT energy distribution
        fft = np.fft.fft2(pattern_data)
        fft_magnitude = np.abs(fft)
        
        # Normalize FFT magnitude
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.sum(fft_magnitude)
            
        # Sort FFT magnitudes
        sorted_vals = np.sort(fft_magnitude.flatten())[::-1]
        
        # Calculate energy concentration in top frequencies
        top_energy = np.sum(sorted_vals[:100]) / np.sum(sorted_vals)
        complexity = 1.0 - top_energy
        
        # Calculate phi-harmonic alignment
        phi_alignment = self._calculate_phi_alignment(pattern_data)
        
        # Calculate centrality - how focused the pattern is at the center
        center_y, center_x = [s // 2 for s in pattern_data.shape]
        center_region = pattern_data[
            center_y-5:center_y+6,
            center_x-5:center_x+6
        ]
        center_energy = np.mean(np.abs(center_region))
        
        # Calculate overall prominence - average over whole pattern
        overall_energy = np.mean(np.abs(pattern_data))
        
        # Calculate centrality ratio
        if overall_energy > 0:
            centrality = center_energy / overall_energy
        else:
            centrality = 0.0
            
        # Calculate materialization potential - how likely to form in physical matter
        materialization = (
            smoothness * LAMBDA +
            phi_alignment * PHI +
            complexity * 0.5 +
            centrality * 1.0
        ) / (LAMBDA + PHI + 0.5 + 1.0)
        
        # Collect metrics
        metrics = {
            'smoothness': smoothness,
            'uniformity': uniformity,
            'complexity': complexity,
            'phi_alignment': phi_alignment,
            'centrality': centrality,
            'materialization_potential': materialization,
        }
        
        return metrics
    
    def _calculate_phi_alignment(self, pattern_data: np.ndarray) -> float:
        """
        Calculate how well a pattern aligns with phi-harmonic proportions.
        
        Args:
            pattern_data: 2D pattern array
            
        Returns:
            Phi-alignment value from 0 to 1
        """
        # Calculate FFT to find frequency components
        fft = np.fft.fft2(pattern_data)
        fft_magnitude = np.abs(fft)
        
        # Normalize FFT magnitude
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
            
        # Create coordinate grid
        y_indices, x_indices = np.indices(fft_magnitude.shape)
        center_y, center_x = [s // 2 for s in fft_magnitude.shape]
        
        y = y_indices - center_y
        x = x_indices - center_x
        
        r = np.sqrt(x**2 + y**2)
        
        # Define phi-harmonic frequency bands
        phi_bands = [1.0]
        for i in range(1, 5):
            phi_bands.append(phi_bands[0] * (PHI ** i))
            
        # Calculate energy in phi-harmonic bands
        phi_energy = 0.0
        total_energy = np.sum(fft_magnitude)
        
        for i, band_center in enumerate(phi_bands):
            # Define band width
            band_width = band_center * 0.2
            
            # Create band mask
            band_mask = (r >= band_center - band_width/2) & (r <= band_center + band_width/2)
            
            # Calculate energy in this band
            band_energy = np.sum(fft_magnitude[band_mask])
            
            # Add to phi energy with decreasing weight
            weight = PHI ** -i
            phi_energy += band_energy * weight
            
        # Calculate phi alignment
        if total_energy > 0:
            phi_alignment = phi_energy / total_energy
            
            # Scale to 0-1 range
            phi_alignment = 1.0 - np.exp(-phi_alignment * 3.0)
        else:
            phi_alignment = 0.0
            
        return min(max(phi_alignment, 0.0), 1.0)
    
    def find_optimal_material(self, pattern: StandingWavePattern) -> str:
        """
        Find the material that would best resonate with a given pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Name of the optimal material
        """
        # Materials to check
        materials = ['water', 'crystal', 'metal', 'sand', 'plasma']
        
        # Calculate interaction strength for each material
        strengths = {}
        for material in materials:
            strength = pattern.calculate_interaction_strength(material)
            strengths[material] = strength
            
        # Find material with highest interaction strength
        optimal_material = max(strengths, key=strengths.get)
        
        return optimal_material