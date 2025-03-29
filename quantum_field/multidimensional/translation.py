"""
Sensory Translation System

Framework for translating phi-harmonic field patterns across different sensory modalities,
enabling synchronized audio-visual-tactile perception of quantum fields.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable
import colorsys

# Import sacred constants
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sacred_constants import (
    PHI, PHI_SQUARED, PHI_CUBED, PHI_PHI, 
    LAMBDA, SACRED_FREQUENCIES
)


class ModalityMap:
    """
    Maps quantum field patterns to specific sensory modalities with phi-harmonic principles.
    
    Each modality (visual, auditory, tactile, etc.) has specific phi-harmonic
    transformations that preserve the essential field patterns while translating
    them to perception-appropriate formats.
    """
    
    def __init__(self, dimensions=(21, 21, 21)):
        """
        Initialize the modality map.
        
        Args:
            dimensions: Base dimensions for the field
        """
        self.dimensions = dimensions
        
        # Define modality properties with phi-harmonic relationships
        self.modalities = {
            "visual": {
                "channels": 3,  # RGB
                "frequency_range": (380, 750),  # Visible light (nm)
                "resolution": self._phi_scale_resolution(dimensions, 1.0),
                "phi_factor": 1.0,
                "transformations": self._generate_visual_transformations()
            },
            "auditory": {
                "channels": 2,  # Stereo
                "frequency_range": (20, 20000),  # Audible range (Hz)
                "resolution": self._phi_scale_resolution(dimensions, LAMBDA),
                "phi_factor": LAMBDA,  # Compressed by inverse phi
                "transformations": self._generate_auditory_transformations()
            },
            "tactile": {
                "channels": 1,  # Intensity
                "frequency_range": (0, 500),  # Tactile frequency response (Hz)
                "resolution": self._phi_scale_resolution(dimensions, LAMBDA * LAMBDA),
                "phi_factor": LAMBDA * LAMBDA,  # Compressed further
                "transformations": self._generate_tactile_transformations()
            },
            "emotional": {
                "channels": 5,  # Basic emotional dimensions
                "frequency_range": (0, 1),  # Normalized intensity
                "resolution": self._phi_scale_resolution(dimensions, PHI),
                "phi_factor": PHI,  # Expanded by phi
                "transformations": self._generate_emotional_transformations()
            },
            "intuitive": {
                "channels": 7,  # Higher intuitive dimensions
                "frequency_range": (0, 1),  # Normalized intensity
                "resolution": self._phi_scale_resolution(dimensions, PHI_SQUARED),
                "phi_factor": PHI_SQUARED,  # Expanded further
                "transformations": self._generate_intuitive_transformations()
            }
        }
        
        # Phi-harmonic frequency mapping
        self.frequency_map = {
            "unity": SACRED_FREQUENCIES["unity"],       # 432 Hz - Ground frequency
            "love": SACRED_FREQUENCIES["love"],         # 528 Hz - Creation/healing
            "cascade": SACRED_FREQUENCIES["cascade"],   # 594 Hz - Heart resonance
            "truth": SACRED_FREQUENCIES["truth"],       # 672 Hz - Voice expression
            "vision": SACRED_FREQUENCIES["vision"],     # 720 Hz - Vision clarity
            "oneness": SACRED_FREQUENCIES["oneness"],   # 768 Hz - Unity consciousness
            
            # Derived frequencies
            "creative": 528 * LAMBDA,  # Creative expression
            "insight": 720 * LAMBDA,   # Insight frequency
            "harmony": 432 * PHI,      # Harmonic connection
            "wisdom": 768 * LAMBDA,    # Wisdom transmission
            "balance": 594 * PHI,      # Balance point
            "transcend": 768 * PHI,    # Transcendence
        }
    
    def _phi_scale_resolution(self, dimensions: Tuple[int, ...], factor: float) -> Tuple[int, ...]:
        """
        Scale resolution by a phi-related factor.
        
        Args:
            dimensions: Original dimensions
            factor: Scaling factor
            
        Returns:
            Phi-scaled dimensions
        """
        # Ensure minimum size of 2 for each dimension
        return tuple(max(2, int(d * factor)) for d in dimensions)
    
    def _generate_visual_transformations(self) -> Dict[str, Callable]:
        """
        Generate transformation functions for visual modality.
        
        Returns:
            Dictionary of transformation functions
        """
        transformations = {}
        
        # Generate color mapping function (field value -> color)
        def phi_harmonic_color_map(field_data: np.ndarray) -> np.ndarray:
            """Map field values to phi-harmonic colors."""
            # Normalize field data to 0-1 range
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Create RGB output array
            rgb_output = np.zeros(normalized.shape + (3,), dtype=np.float32)
            
            # Generate phi-harmonic colors
            for idx in np.ndindex(normalized.shape):
                value = normalized[idx]
                
                # Use phi-weighted HSV to RGB conversion for harmony
                h = (value * PHI) % 1.0  # Hue
                s = 0.5 + 0.5 * np.sin(value * PHI_SQUARED * np.pi)  # Saturation
                v = 0.5 + 0.5 * np.sin(value * PHI_CUBED * np.pi)  # Value
                
                # Convert HSV to RGB
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                rgb_output[idx] = (r, g, b)
            
            return rgb_output
        
        # Generate luminance enhancement function
        def phi_enhance_luminance(field_data: np.ndarray) -> np.ndarray:
            """Enhance luminance patterns with phi-harmonic scaling."""
            # Apply phi-weighted gamma correction
            gamma = 1.0 / PHI
            enhanced = field_data ** gamma
            
            # Apply phi-harmonic contrast enhancement
            mean_val = np.mean(enhanced)
            contrast = enhanced - mean_val
            enhanced = mean_val + contrast * PHI
            
            # Clip to valid range
            return np.clip(enhanced, 0, 1)
        
        # Generate edge detection function for enhanced visual perception
        def phi_edge_detection(field_data: np.ndarray) -> np.ndarray:
            """Detect edges with phi-harmonic weighting."""
            # Calculate gradients
            grads = np.gradient(field_data)
            
            # Combine gradients with phi-weighting
            grad_magnitude = np.sqrt(
                grads[0]**2 * PHI + 
                grads[1]**2 * 1.0 + 
                grads[2]**2 * LAMBDA
            )
            
            # Normalize
            if np.max(grad_magnitude) > 0:
                grad_magnitude /= np.max(grad_magnitude)
            
            return grad_magnitude
        
        # Store transformations
        transformations["color_map"] = phi_harmonic_color_map
        transformations["enhance_luminance"] = phi_enhance_luminance
        transformations["edge_detection"] = phi_edge_detection
        
        return transformations
    
    def _generate_auditory_transformations(self) -> Dict[str, Callable]:
        """
        Generate transformation functions for auditory modality.
        
        Returns:
            Dictionary of transformation functions
        """
        transformations = {}
        
        # Generate frequency mapping function (field value -> frequency)
        def phi_frequency_map(field_data: np.ndarray, base_freq: float = 432.0) -> np.ndarray:
            """Map field values to phi-harmonic frequencies."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Create phi-harmonic frequency mapping
            # Each octave is mapped to phi multiples of the base frequency
            frequency_output = base_freq * (PHI ** (normalized * 3 - 1))
            
            return frequency_output
        
        # Generate amplitude mapping function
        def phi_amplitude_map(field_data: np.ndarray) -> np.ndarray:
            """Map field values to amplitudes with phi-harmonic scaling."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Apply phi-harmonic response curve
            # This creates a more natural-sounding amplitude response
            amplitude = normalized ** (1.0 / PHI)
            
            return amplitude
        
        # Generate stereo panning function
        def phi_stereo_panning(field_data: np.ndarray) -> np.ndarray:
            """Create stereo panning based on field spatial distribution."""
            # Calculate center of mass in X direction
            if len(field_data.shape) >= 3:
                x_indices = np.arange(field_data.shape[0])
                x_weights = np.sum(field_data, axis=(1, 2))
                if np.sum(x_weights) > 0:
                    x_center = np.sum(x_indices * x_weights) / np.sum(x_weights)
                else:
                    x_center = field_data.shape[0] / 2
                
                # Normalize to -1 to 1 range (left to right)
                pan = 2 * (x_center / field_data.shape[0]) - 1
            else:
                pan = 0.0  # Center if field is not 3D
            
            # Apply phi-weighting for more natural panning
            pan = np.sin(pan * np.pi/2 * PHI)
            
            return pan
        
        # Store transformations
        transformations["frequency_map"] = phi_frequency_map
        transformations["amplitude_map"] = phi_amplitude_map
        transformations["stereo_panning"] = phi_stereo_panning
        
        return transformations
    
    def _generate_tactile_transformations(self) -> Dict[str, Callable]:
        """
        Generate transformation functions for tactile modality.
        
        Returns:
            Dictionary of transformation functions
        """
        transformations = {}
        
        # Generate intensity mapping function
        def phi_intensity_map(field_data: np.ndarray) -> np.ndarray:
            """Map field values to tactile intensity with phi-harmonic scaling."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Apply phi-harmonic intensity curve
            intensity = normalized ** LAMBDA
            
            return intensity
        
        # Generate frequency mapping for tactile vibration
        def phi_vibration_frequency(field_data: np.ndarray, 
                                  min_freq: float = 20.0, 
                                  max_freq: float = 200.0) -> np.ndarray:
            """Map field values to tactile vibration frequencies."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Apply phi-harmonic mapping to frequency range
            log_min = np.log(min_freq)
            log_max = np.log(max_freq)
            log_range = log_max - log_min
            
            # Map with phi-weighted curve
            log_freq = log_min + normalized ** PHI * log_range
            frequency = np.exp(log_freq)
            
            return frequency
        
        # Generate texture mapping
        def phi_texture_pattern(field_data: np.ndarray, pattern_scale: float = 1.0) -> np.ndarray:
            """Generate tactile texture patterns from field data."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Create phi-harmonic texture pattern
            # This simulates surface texture variation
            texture = np.zeros_like(normalized)
            
            # Apply multiple frequency components with phi-harmonic relationships
            for i in range(1, 5):
                freq = pattern_scale * (PHI ** i)
                
                # Different dimensional weights for balanced texture
                phase = np.random.uniform(0, 2 * np.pi)
                
                if len(normalized.shape) >= 3:
                    x, y, z = np.meshgrid(
                        np.linspace(0, 1, normalized.shape[0]),
                        np.linspace(0, 1, normalized.shape[1]),
                        np.linspace(0, 1, normalized.shape[2]),
                        indexing='ij'
                    )
                    
                    # Apply phi-weighted pattern
                    weight = 1.0 / (i * PHI)
                    texture += weight * np.sin(
                        freq * 2 * np.pi * (
                            x * PHI + 
                            y * 1.0 + 
                            z * LAMBDA
                        ) + phase
                    )
                else:
                    # Handle lower-dimensional data
                    coords = np.linspace(0, 1, normalized.size)
                    pattern = np.sin(freq * 2 * np.pi * coords + phase)
                    texture += pattern.reshape(normalized.shape) / (i * PHI)
            
            # Normalize texture pattern
            if np.max(texture) > np.min(texture):
                texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture))
            
            # Blend with field data using phi-weighted average
            blended = (normalized * PHI + texture) / (PHI + 1)
            
            return blended
        
        # Store transformations
        transformations["intensity_map"] = phi_intensity_map
        transformations["vibration_frequency"] = phi_vibration_frequency
        transformations["texture_pattern"] = phi_texture_pattern
        
        return transformations
    
    def _generate_emotional_transformations(self) -> Dict[str, Callable]:
        """
        Generate transformation functions for emotional modality.
        
        Returns:
            Dictionary of transformation functions
        """
        transformations = {}
        
        # Define emotional dimensions with phi-harmonic relationships
        emotional_dimensions = {
            "joy": {"frequency": 528.0, "phi_factor": PHI},
            "peace": {"frequency": 432.0, "phi_factor": 1.0},
            "love": {"frequency": 594.0, "phi_factor": PHI_SQUARED},
            "insight": {"frequency": 720.0, "phi_factor": PHI_CUBED},
            "unity": {"frequency": 768.0, "phi_factor": PHI ** 4}
        }
        
        # Generate emotional resonance mapping
        def phi_emotional_map(field_data: np.ndarray) -> Dict[str, float]:
            """Map field data to emotional resonance patterns."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Compute FFT for frequency analysis
            fft_data = np.fft.fftn(normalized)
            fft_mag = np.abs(fft_data)
            
            # Normalize FFT magnitude
            if np.max(fft_mag) > 0:
                fft_mag /= np.max(fft_mag)
            
            # Map to emotional dimensions
            emotional_values = {}
            
            for emotion, properties in emotional_dimensions.items():
                # Calculate resonance with this emotional frequency
                freq = properties["frequency"]
                phi_factor = properties["phi_factor"]
                
                # Create frequency coordinates
                freq_coords = [np.fft.fftfreq(d) * d for d in normalized.shape]
                freq_grids = np.meshgrid(*freq_coords, indexing='ij')
                
                # Calculate frequency magnitude
                freq_mag = np.sqrt(sum(grid**2 for grid in freq_grids))
                
                # Find resonance at phi-harmonic points
                resonance_points = []
                
                for harmonic in range(1, 4):
                    harmonic_freq = freq * (harmonic / phi_factor)
                    
                    # Find points close to this frequency
                    mask = np.abs(freq_mag - harmonic_freq) < (harmonic_freq * 0.1)
                    resonance = np.mean(fft_mag[mask]) if np.any(mask) else 0.0
                    resonance_points.append(resonance)
                
                # Combine resonance points with phi-weighting
                weights = [PHI**(-i) for i in range(len(resonance_points))]
                weighted_sum = sum(r * w for r, w in zip(resonance_points, weights))
                weight_sum = sum(weights)
                
                emotional_values[emotion] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
            return emotional_values
        
        # Generate emotional coloration function
        def phi_emotional_coloration(field_data: np.ndarray) -> np.ndarray:
            """Apply emotional coloration to field data."""
            # Get emotional mapping
            emotions = phi_emotional_map(field_data)
            
            # Create colored version of field data
            colored = field_data.copy()
            
            # Apply emotional coloration with phi-weighted transformation
            for emotion, value in emotions.items():
                properties = emotional_dimensions[emotion]
                phi_factor = properties["phi_factor"]
                
                # Skip if negligible resonance
                if value < 0.1:
                    continue
                
                # Apply transformation based on emotion
                if emotion == "joy":
                    # Joy - amplify high values
                    colored = colored * (1.0 - value) + (colored ** LAMBDA) * value
                
                elif emotion == "peace":
                    # Peace - smooth the field
                    smoothed = np.copy(colored)
                    for _ in range(3):
                        smoothed = np.array([
                            np.roll(smoothed, 1, axis=i) + 
                            np.roll(smoothed, -1, axis=i) 
                            for i in range(len(colored.shape))
                        ]).mean(axis=0) / 2
                    
                    colored = colored * (1.0 - value) + smoothed * value
                
                elif emotion == "love":
                    # Love - coherent patterns
                    x, y, z = np.meshgrid(
                        np.linspace(0, 1, colored.shape[0]),
                        np.linspace(0, 1, colored.shape[1]),
                        np.linspace(0, 1, colored.shape[2]) if len(colored.shape) > 2 else [0],
                        indexing='ij'
                    )
                    
                    r = np.sqrt(
                        (x - 0.5)**2 + (y - 0.5)**2 + 
                        (z - 0.5)**2 if len(colored.shape) > 2 else 0
                    )
                    
                    pattern = np.sin(r * 15 * PHI_PHI) * 0.5 + 0.5
                    colored = colored * (1.0 - value) + pattern * colored * value
                
                elif emotion == "insight":
                    # Insight - enhance edge patterns
                    edges = np.zeros_like(colored)
                    for i in range(len(colored.shape)):
                        edges += np.abs(np.diff(colored, axis=i, append=0))
                    edges /= len(colored.shape)
                    
                    colored = colored * (1.0 - value) + (colored + edges * PHI) * value
                
                elif emotion == "unity":
                    # Unity - phi-harmonic unification
                    mean_val = np.mean(colored)
                    
                    # Create unified field with phi-harmonic balance
                    unified = mean_val + (colored - mean_val) * PHI
                    
                    colored = colored * (1.0 - value) + unified * value
            
            # Ensure valid range
            if np.max(colored) > np.min(colored):
                colored = (colored - np.min(colored)) / (np.max(colored) - np.min(colored))
            
            return colored
        
        # Store transformations
        transformations["emotional_map"] = phi_emotional_map
        transformations["emotional_coloration"] = phi_emotional_coloration
        
        return transformations
    
    def _generate_intuitive_transformations(self) -> Dict[str, Callable]:
        """
        Generate transformation functions for intuitive modality.
        
        Returns:
            Dictionary of transformation functions
        """
        transformations = {}
        
        # Generate intuitive pattern extraction
        def phi_intuitive_patterns(field_data: np.ndarray) -> Dict[str, np.ndarray]:
            """Extract intuitive patterns from field data."""
            # Normalize field data
            if np.max(field_data) > np.min(field_data):
                normalized = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
            else:
                normalized = np.zeros_like(field_data)
            
            # Extract various intuitive patterns using phi-harmonic principles
            patterns = {}
            
            # Coherence pattern - overall field harmony
            grad = np.gradient(normalized)
            grad_mag = np.sqrt(sum(g**2 for g in grad))
            if np.max(grad_mag) > 0:
                grad_mag /= np.max(grad_mag)
            patterns["coherence"] = 1.0 - grad_mag
            
            # Flow pattern - directional tendencies
            flow = np.zeros_like(normalized)
            for i in range(len(normalized.shape)):
                flow_component = np.roll(normalized, -1, axis=i) - normalized
                flow += flow_component ** 2
            flow = np.sqrt(flow)
            if np.max(flow) > 0:
                flow /= np.max(flow)
            patterns["flow"] = flow
            
            # Resonance pattern - phi-harmonic resonance
            resonance = np.zeros_like(normalized)
            for phi_scale in [PHI, PHI_SQUARED, PHI_CUBED]:
                # Create phi-scaled pattern
                x, y, z = np.meshgrid(
                    np.linspace(0, 1, normalized.shape[0]),
                    np.linspace(0, 1, normalized.shape[1]),
                    np.linspace(0, 1, normalized.shape[2]) if len(normalized.shape) > 2 else [0],
                    indexing='ij'
                )
                
                r = np.sqrt(
                    (x - 0.5)**2 + (y - 0.5)**2 + 
                    (z - 0.5)**2 if len(normalized.shape) > 2 else 0
                )
                
                pattern = np.sin(r * 10 * phi_scale) * 0.5 + 0.5
                
                # Calculate correlation
                corr = (normalized - np.mean(normalized)) * (pattern - np.mean(pattern))
                corr_norm = np.sqrt(np.mean((normalized - np.mean(normalized))**2) * 
                                  np.mean((pattern - np.mean(pattern))**2))
                
                if corr_norm > 0:
                    resonance += np.abs(np.mean(corr) / corr_norm) * (1 / phi_scale)
            
            patterns["resonance"] = resonance / sum(1 / p for p in [PHI, PHI_SQUARED, PHI_CUBED])
            
            # Intention pattern - directionality of field
            intention = np.zeros_like(normalized)
            for i in range(len(normalized.shape)):
                # Calculate directional tendency
                forward = np.roll(normalized, -1, axis=i)
                backward = np.roll(normalized, 1, axis=i)
                
                # Directional preference (positive = forward, negative = backward)
                direction = (forward - normalized) - (normalized - backward)
                
                # Weight by phi factor based on dimension
                if i == 0:  # X dimension
                    weight = PHI
                elif i == 1:  # Y dimension
                    weight = 1.0
                else:  # Z dimension
                    weight = LAMBDA
                
                intention += direction * weight
            
            # Normalize intention
            intention_min = np.min(intention)
            intention_max = np.max(intention)
            if intention_max > intention_min:
                intention = (intention - intention_min) / (intention_max - intention_min)
            
            patterns["intention"] = intention
            
            return patterns
        
        # Generate intuitive dimension enhancement
        def phi_enhance_intuition(field_data: np.ndarray, 
                               pattern_name: str = "resonance",
                               intensity: float = 0.5) -> np.ndarray:
            """Enhance field data based on intuitive pattern."""
            # Get intuitive patterns
            patterns = phi_intuitive_patterns(field_data)
            
            # Use requested pattern
            if pattern_name not in patterns:
                pattern_name = "resonance"  # Default
            
            pattern = patterns[pattern_name]
            
            # Enhance the field with this pattern
            enhanced = field_data * (1.0 - intensity) + (field_data * pattern * PHI) * intensity
            
            # Normalize result
            if np.max(enhanced) > np.min(enhanced):
                enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
            
            return enhanced
        
        # Generate multidimensional perception function
        def phi_multidimensional_perception(field_data: np.ndarray) -> Dict[str, np.ndarray]:
            """Generate multidimensional perception patterns from field data."""
            # Extract intuitive patterns
            patterns = phi_intuitive_patterns(field_data)
            
            # Create multi-dimensional projection
            projections = {}
            
            # Base projection (3D)
            projections["3d"] = field_data
            
            # 4D projection (time dimension)
            time_projection = np.zeros(field_data.shape, dtype=field_data.dtype)
            
            # Create time evolution based on field flow
            flow = patterns["flow"]
            for t in range(5):  # Generate 5 time steps
                # Calculate evolution step
                step = flow * 0.1 * (t+1) * LAMBDA
                
                # Apply step to create evolved field
                evolved = field_data + step * (field_data - 0.5)
                
                # Normalize evolved field
                if np.max(evolved) > np.min(evolved):
                    evolved = (evolved - np.min(evolved)) / (np.max(evolved) - np.min(evolved))
                
                # Add to time projection with phi-weighted temporal decay
                weight = PHI ** (-t)
                time_projection += evolved * weight
            
            # Normalize time projection
            weight_sum = sum(PHI ** (-t) for t in range(5))
            time_projection /= weight_sum
            
            projections["4d_time"] = time_projection
            
            # 5D projection (consciousness dimension)
            consciousness_projection = patterns["coherence"] * field_data
            
            # Normalize consciousness projection
            if np.max(consciousness_projection) > np.min(consciousness_projection):
                consciousness_projection = (consciousness_projection - np.min(consciousness_projection)) / (
                    np.max(consciousness_projection) - np.min(consciousness_projection))
            
            projections["5d_consciousness"] = consciousness_projection
            
            # 6D projection (intention dimension)
            intention_projection = patterns["intention"] * field_data
            
            # Normalize intention projection
            if np.max(intention_projection) > np.min(intention_projection):
                intention_projection = (intention_projection - np.min(intention_projection)) / (
                    np.max(intention_projection) - np.min(intention_projection))
            
            projections["6d_intention"] = intention_projection
            
            # 7D projection (unified field dimension)
            # Combined aspect of all dimensions with phi-harmonic weighting
            unified_projection = (
                field_data * 1.0 + 
                time_projection * LAMBDA + 
                consciousness_projection * PHI + 
                intention_projection * PHI_SQUARED
            ) / (1.0 + LAMBDA + PHI + PHI_SQUARED)
            
            projections["7d_unified"] = unified_projection
            
            return projections
        
        # Store transformations
        transformations["intuitive_patterns"] = phi_intuitive_patterns
        transformations["enhance_intuition"] = phi_enhance_intuition
        transformations["multidimensional_perception"] = phi_multidimensional_perception
        
        return transformations
    
    def get_modality_profile(self) -> Dict[str, Dict]:
        """
        Get a complete profile of all sensory modalities.
        
        Returns:
            Dictionary of modality profiles
        """
        profile = {}
        
        for modality_name, properties in self.modalities.items():
            # Create a copy of the properties without the transformations
            modality_profile = {k: v for k, v in properties.items() if k != "transformations"}
            
            # Add transformation names
            modality_profile["transformations"] = list(properties["transformations"].keys())
            
            profile[modality_name] = modality_profile
        
        return profile
    
    def get_transformation(self, modality: str, transformation: str) -> Optional[Callable]:
        """
        Get a specific transformation function.
        
        Args:
            modality: Sensory modality name
            transformation: Transformation function name
            
        Returns:
            Transformation function or None if not found
        """
        if modality not in self.modalities:
            return None
        
        transformations = self.modalities[modality]["transformations"]
        return transformations.get(transformation)
    
    def apply_transformation(self, 
                           field_data: np.ndarray,
                           modality: str, 
                           transformation: str,
                           **kwargs) -> np.ndarray:
        """
        Apply a specific transformation to field data.
        
        Args:
            field_data: Input field data
            modality: Sensory modality name
            transformation: Transformation function name
            **kwargs: Additional parameters for the transformation
            
        Returns:
            Transformed data
        """
        transform_func = self.get_transformation(modality, transformation)
        
        if transform_func is None:
            # Return unchanged if transformation not found
            return field_data
        
        # Apply the transformation with any additional parameters
        return transform_func(field_data, **kwargs)


class SensoryTranslator:
    """
    Translates quantum field patterns across different sensory modalities.
    
    This class provides the core functionality for translating fields
    between visual, auditory, tactile, emotional, and intuitive domains
    while preserving phi-harmonic relationships.
    """
    
    def __init__(self, dimensions=(21, 21, 21)):
        """
        Initialize the sensory translator.
        
        Args:
            dimensions: Base dimensions for the field
        """
        self.dimensions = dimensions
        self.modality_map = ModalityMap(dimensions)
        
        # Current field data
        self.field_data = None
        
        # Cached sensory representations
        self.sensory_cache = {}
    
    def load_field(self, field_data: np.ndarray) -> None:
        """
        Load field data for translation.
        
        Args:
            field_data: Input field data
        """
        self.field_data = field_data
        
        # Clear the cache
        self.sensory_cache = {}
    
    def get_visual_representation(self, mode: str = "color_map") -> np.ndarray:
        """
        Get visual representation of the field.
        
        Args:
            mode: Visual transformation mode
            
        Returns:
            Visual representation array
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = f"visual_{mode}"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        visual_data = self.modality_map.apply_transformation(
            self.field_data, "visual", mode
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = visual_data
        
        return visual_data
    
    def get_auditory_representation(self, 
                                  mode: str = "frequency_map",
                                  base_freq: float = 432.0) -> np.ndarray:
        """
        Get auditory representation of the field.
        
        Args:
            mode: Auditory transformation mode
            base_freq: Base frequency for mapping
            
        Returns:
            Auditory representation array
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = f"auditory_{mode}_{base_freq}"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        auditory_data = self.modality_map.apply_transformation(
            self.field_data, "auditory", mode, base_freq=base_freq
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = auditory_data
        
        return auditory_data
    
    def get_tactile_representation(self, mode: str = "intensity_map") -> np.ndarray:
        """
        Get tactile representation of the field.
        
        Args:
            mode: Tactile transformation mode
            
        Returns:
            Tactile representation array
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = f"tactile_{mode}"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        tactile_data = self.modality_map.apply_transformation(
            self.field_data, "tactile", mode
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = tactile_data
        
        return tactile_data
    
    def get_emotional_representation(self) -> Dict[str, float]:
        """
        Get emotional representation of the field.
        
        Returns:
            Dictionary of emotional values
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = "emotional_map"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        emotional_data = self.modality_map.apply_transformation(
            self.field_data, "emotional", "emotional_map"
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = emotional_data
        
        return emotional_data
    
    def get_intuitive_representation(self, mode: str = "intuitive_patterns") -> Dict:
        """
        Get intuitive representation of the field.
        
        Args:
            mode: Intuitive transformation mode
            
        Returns:
            Dictionary of intuitive patterns
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = f"intuitive_{mode}"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        intuitive_data = self.modality_map.apply_transformation(
            self.field_data, "intuitive", mode
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = intuitive_data
        
        return intuitive_data
    
    def get_multidimensional_projections(self) -> Dict[str, np.ndarray]:
        """
        Get multidimensional projections of the field.
        
        Returns:
            Dictionary of dimensional projections
        """
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Check cache
        cache_key = "multidimensional_projections"
        if cache_key in self.sensory_cache:
            return self.sensory_cache[cache_key]
        
        # Apply transformation
        projections = self.modality_map.apply_transformation(
            self.field_data, "intuitive", "multidimensional_perception"
        )
        
        # Cache the result
        self.sensory_cache[cache_key] = projections
        
        return projections
    
    def translate_between_modalities(self, 
                                   source_modality: str,
                                   target_modality: str,
                                   source_data: np.ndarray = None) -> np.ndarray:
        """
        Translate data between sensory modalities.
        
        Args:
            source_modality: Source sensory modality
            target_modality: Target sensory modality
            source_data: Source data (uses loaded field data if None)
            
        Returns:
            Translated data in target modality
        """
        # Use loaded field data if source_data not provided
        if source_data is None:
            if self.field_data is None:
                raise ValueError("No field data loaded")
            source_data = self.field_data
        
        # Validate modalities
        if source_modality not in self.modality_map.modalities:
            raise ValueError(f"Unknown source modality: {source_modality}")
        
        if target_modality not in self.modality_map.modalities:
            raise ValueError(f"Unknown target modality: {target_modality}")
        
        # Get phi factors for source and target
        source_phi = self.modality_map.modalities[source_modality]["phi_factor"]
        target_phi = self.modality_map.modalities[target_modality]["phi_factor"]
        
        # Calculate phi-scaled transformation ratio
        if source_phi == 0 or target_phi == 0:
            phi_ratio = 1.0
        else:
            phi_ratio = target_phi / source_phi
        
        # Normalize source data
        if np.max(source_data) > np.min(source_data):
            normalized = (source_data - np.min(source_data)) / (np.max(source_data) - np.min(source_data))
        else:
            normalized = np.zeros_like(source_data)
        
        # Apply appropriate transformations based on target modality
        if target_modality == "visual":
            # Transform to visual
            return self.modality_map.apply_transformation(
                normalized, "visual", "color_map"
            )
        
        elif target_modality == "auditory":
            # Transform to auditory
            # Use phi-scaled base frequency
            base_freq = 432.0 * phi_ratio
            return self.modality_map.apply_transformation(
                normalized, "auditory", "frequency_map", base_freq=base_freq
            )
        
        elif target_modality == "tactile":
            # Transform to tactile
            return self.modality_map.apply_transformation(
                normalized, "tactile", "intensity_map"
            )
        
        elif target_modality == "emotional":
            # Transform to emotional
            # Apply emotional coloration
            return self.modality_map.apply_transformation(
                normalized, "emotional", "emotional_coloration"
            )
        
        elif target_modality == "intuitive":
            # Transform to intuitive
            # Enhance intuition based on dominant pattern
            pattern_name = "resonance"
            return self.modality_map.apply_transformation(
                normalized, "intuitive", "enhance_intuition",
                pattern_name=pattern_name, intensity=phi_ratio
            )
        
        # Default case
        return normalized
    
    def create_synchronized_experience(self, field_data: np.ndarray = None) -> Dict:
        """
        Create a synchronized multi-sensory experience from field data.
        
        Args:
            field_data: Input field data (uses loaded field if None)
            
        Returns:
            Dictionary with synchronized sensory representations
        """
        # Use loaded field data if not provided
        if field_data is not None:
            self.load_field(field_data)
        
        if self.field_data is None:
            raise ValueError("No field data loaded")
        
        # Create synchronized experience with phi-harmonic relationships
        experience = {}
        
        # Visual representation
        experience["visual"] = self.get_visual_representation(mode="color_map")
        
        # Auditory representation
        experience["auditory"] = {
            "frequency": self.get_auditory_representation(mode="frequency_map"),
            "amplitude": self.get_auditory_representation(mode="amplitude_map"),
            "panning": self.modality_map.apply_transformation(
                self.field_data, "auditory", "stereo_panning"
            )
        }
        
        # Tactile representation
        experience["tactile"] = {
            "intensity": self.get_tactile_representation(mode="intensity_map"),
            "frequency": self.modality_map.apply_transformation(
                self.field_data, "tactile", "vibration_frequency"
            ),
            "texture": self.modality_map.apply_transformation(
                self.field_data, "tactile", "texture_pattern"
            )
        }
        
        # Emotional representation
        experience["emotional"] = self.get_emotional_representation()
        
        # Intuitive representation
        experience["intuitive"] = self.get_intuitive_representation()
        
        # Multidimensional projections
        experience["dimensions"] = self.get_multidimensional_projections()
        
        return experience