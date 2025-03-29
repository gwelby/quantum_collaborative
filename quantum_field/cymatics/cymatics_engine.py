"""
Cymatics Engine Module

Integration of all cymatic components for direct manifestation of quantum fields 
into physical matter through sound-based cymatics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
sys.path.append('/mnt/d/projects/Python')
from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES, 
    calculate_phi_resonance, phi_harmonic
)

from .cymatic_field import CymaticField
from .frequency_modulator import FrequencyModulator
from .pattern_generator import PatternGenerator, StandingWavePattern
from .resonance_chamber import (
    MaterialResonator, WaterResonator, CrystalResonator, MetalResonator
)
from ..toroidal import ToroidalField

class CymaticsEngine:
    """
    A unified engine for cymatic pattern materialization that enables 
    real-time translation of consciousness into material form through
    sound geometry and phi-harmonic frequency modulation.
    """
    
    def __init__(self, name: str = "Cymatic Materialization Engine"):
        """
        Initialize a new CymaticsEngine.
        
        Args:
            name: Name for this engine instance
        """
        self.name = name
        
        # Core components
        self.cymatic_field = CymaticField()
        self.frequency_modulator = FrequencyModulator()
        self.pattern_generator = PatternGenerator()
        
        # Resonators for different materials
        self.resonators = {
            'water': WaterResonator(),
            'crystal': CrystalResonator(),
            'metal': MetalResonator()
        }
        
        # Current state
        self.active_frequency = SACRED_FREQUENCIES['unity']  # 432 Hz
        self.active_material = 'water'
        self.active_pattern = None
        self.system_coherence = 1.0
        self.energy_level = 0.5
        
        # Performance metrics
        self.manifestation_efficiency = 0.0
        self.field_stability = 0.0
        self.phase_coherence = 0.0
        
        # Pattern memory
        self.pattern_memory = {}
        
        # System configuration
        self.config = {
            'phi_alignment': 1.0,
            'consciousness_integration': 0.8,
            'material_coupling': 0.9,
            'auto_resonance': True,
            'energy_conservation': True
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the cymatics engine system."""
        # Set base frequency
        self.set_frequency(self.active_frequency)
        
        # Initialize pattern memory with basic patterns
        self._initialize_pattern_memory()
        
        # Calculate initial performance metrics
        self._update_performance_metrics()
    
    def _initialize_pattern_memory(self):
        """Initialize pattern memory with fundamental patterns."""
        # Add basic patterns for each sacred frequency
        for name, freq in SACRED_FREQUENCIES.items():
            # Create pattern
            pattern = self.pattern_generator.create_sacred_frequency_pattern(
                frequency_name=name,
                pattern_type='CIRCULAR',
                symmetry=6
            )
            
            # Store pattern data
            pattern_data = pattern.get_pattern()
            
            # Store in memory
            self.pattern_memory[f"{name}_basic"] = {
                'frequency': freq,
                'pattern': pattern_data,
                'type': 'basic',
                'coherence': 0.9
            }
    
    def set_frequency(self, frequency: float):
        """
        Set the active frequency for the cymatics engine.
        
        Args:
            frequency: Frequency in Hz
        """
        self.active_frequency = frequency
        
        # Update all components
        self.cymatic_field.set_frequency(frequency)
        self.frequency_modulator.set_base_frequency(frequency)
        
        # Update active resonator
        if self.active_material in self.resonators:
            self.resonators[self.active_material].apply_frequency(frequency)
            
        # Update performance metrics
        self._update_performance_metrics()
    
    def set_sacred_frequency(self, name: str):
        """
        Set the active frequency using a sacred frequency name.
        
        Args:
            name: Name of the sacred frequency (e.g., 'unity', 'love', 'truth')
        """
        if name in SACRED_FREQUENCIES:
            self.set_frequency(SACRED_FREQUENCIES[name])
    
    def set_active_material(self, material: str):
        """
        Set the active material for cymatic pattern formation.
        
        Args:
            material: Material name ('water', 'crystal', 'metal')
        """
        if material in self.resonators:
            self.active_material = material
            
            # Apply current frequency to the new active resonator
            self.resonators[material].apply_frequency(self.active_frequency)
            
            # Update pattern based on new material
            self._update_active_pattern()
            
            # Update performance metrics
            self._update_performance_metrics()
    
    def generate_pattern(
        self,
        pattern_type: str = 'CIRCULAR',
        symmetry: int = 6,
        resolution: Tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Generate a cymatic pattern for the current configuration.
        
        Args:
            pattern_type: Type of pattern to generate
            symmetry: Symmetry order for the pattern
            resolution: Resolution of the pattern
            
        Returns:
            2D numpy array with the generated pattern
        """
        # Create pattern
        pattern = self.pattern_generator.create_pattern(
            name=f"Pattern_{pattern_type}_{symmetry}",
            frequencies=[self.active_frequency],
            pattern_type=pattern_type,
            symmetry=symmetry,
            resolution=resolution
        )
        
        # Get pattern data
        pattern_data = pattern.get_pattern()
        
        # Apply material-specific effects
        if self.active_material in self.resonators:
            # Get resonator response
            resonator = self.resonators[self.active_material]
            response = resonator.apply_frequency(self.active_frequency)
            
            # Apply material-specific transformation
            if self.active_material == 'water':
                # Water patterns have smoother transitions
                smoothed = self._apply_water_smoothing(pattern_data)
                pattern_data = 0.7 * pattern_data + 0.3 * smoothed
                
            elif self.active_material == 'crystal':
                # Crystal patterns have more defined structure
                structured = self._apply_crystal_structure(pattern_data)
                pattern_data = 0.4 * pattern_data + 0.6 * structured
                
            elif self.active_material == 'metal':
                # Metal patterns have sharp nodal lines
                chladni = self._apply_metal_nodal_lines(pattern_data)
                pattern_data = 0.3 * pattern_data + 0.7 * chladni
        
        # Store as active pattern
        self.active_pattern = pattern_data
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return pattern_data
    
    def _apply_water_smoothing(self, pattern: np.ndarray) -> np.ndarray:
        """
        Apply water-specific smoothing to a pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Smoothed pattern
        """
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        sigma = 1.0 + 1.0 * LAMBDA  # Phi-based smoothing
        
        smoothed = gaussian_filter(pattern, sigma=sigma)
        
        # Enhance central regions (water forms stronger patterns at center)
        center_y, center_x = [s // 2 for s in pattern.shape]
        Y, X = np.indices(pattern.shape)
        
        # Calculate distance from center
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        R_norm = R / np.max(R)
        
        # Create center enhancement mask
        center_mask = np.exp(-R_norm**2 / (2 * LAMBDA**2))
        
        # Apply center enhancement
        enhanced = smoothed * (1.0 + center_mask * LAMBDA)
        
        # Normalize
        if np.max(enhanced) > 0:
            enhanced = enhanced / np.max(enhanced)
            
        return enhanced
    
    def _apply_crystal_structure(self, pattern: np.ndarray) -> np.ndarray:
        """
        Apply crystal-specific structural enhancement to a pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Enhanced pattern with crystalline structure
        """
        # Get crystal resonator
        if 'crystal' not in self.resonators:
            return pattern
            
        resonator = self.resonators['crystal']
        
        # Enhance pattern structure based on crystalline properties
        grad_y, grad_x = np.gradient(pattern)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Enhance edges (crystals form along energy gradients)
        edge_enhanced = gradient_magnitude * (pattern > 0.5)
        
        # Combine with original pattern
        structured = pattern * (1.0 - edge_enhanced * 0.5) + edge_enhanced * 0.5
        
        # Add crystalline faceting
        if hasattr(resonator, 'facet_count') and resonator.facet_count > 0:
            # Create facet pattern
            center_y, center_x = [s // 2 for s in pattern.shape]
            Y, X = np.indices(pattern.shape)
            
            # Calculate angle from center
            theta = np.arctan2(Y - center_y, X - center_x)
            
            # Create facet pattern
            facets = np.cos(theta * resonator.facet_count) ** 2
            
            # Apply faceting
            structured = structured * (1.0 - 0.3 * facets) + 0.3 * facets * structured
            
        # Normalize
        if np.max(structured) > 0:
            structured = structured / np.max(structured)
            
        return structured
    
    def _apply_metal_nodal_lines(self, pattern: np.ndarray) -> np.ndarray:
        """
        Apply metal-specific nodal line enhancement to a pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Enhanced pattern with Chladni-like nodal lines
        """
        # For Chladni patterns, the nodes (zeros) create the patterns
        
        # Find regions near zero-crossings
        chladni = 1.0 - np.abs(pattern) / (np.max(np.abs(pattern)) + 1e-10)
        
        # Apply threshold to highlight nodal lines
        threshold = 0.8
        nodal_lines = (chladni > threshold).astype(float)
        
        # Smooth lines slightly
        from scipy.ndimage import gaussian_filter
        nodal_lines = gaussian_filter(nodal_lines, sigma=0.5)
        
        # Normalize
        if np.max(nodal_lines) > 0:
            nodal_lines = nodal_lines / np.max(nodal_lines)
            
        return nodal_lines
    
    def apply_frequency_modulation(
        self,
        mod_rate: float = 1.0,
        mod_depth: float = 0.3,
        waveform: str = 'sine',
        duration: float = 3.0,
        phi_weight: float = 0.8
    ) -> Dict[str, Any]:
        """
        Apply frequency modulation and track the resulting patterns.
        
        Args:
            mod_rate: Modulation rate in Hz
            mod_depth: Modulation depth (0 to 1)
            waveform: Modulation waveform ('sine', 'triangle', 'square')
            duration: Duration of modulation in seconds
            phi_weight: Weight of phi-based modulation (0 to 1)
            
        Returns:
            Dictionary with modulation results
        """
        # Set up modulation
        self.frequency_modulator.set_modulation_parameters(
            rate=mod_rate,
            depth=mod_depth,
            waveform=waveform,
            phi_weight=phi_weight
        )
        
        # Number of time steps
        steps = int(duration / self.frequency_modulator.time_step)
        
        # Track results
        results = {
            'frequencies': [],
            'times': [],
            'responses': [],
            'coherence': [],
            'patterns': []
        }
        
        # Run modulation
        for i in range(steps):
            # Update modulator
            self.frequency_modulator.update()
            
            # Get current frequency
            freq = self.frequency_modulator.get_current_frequency()
            
            # Apply to resonator
            if self.active_material in self.resonators:
                response = self.resonators[self.active_material].apply_frequency(freq)
            else:
                response = 0.0
                
            # Apply to cymatic field
            self.cymatic_field.set_frequency(freq)
            
            # Get current pattern
            pattern = self.cymatic_field.visualize_pattern()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Record results
            results['frequencies'].append(freq)
            results['times'].append(i * self.frequency_modulator.time_step)
            results['responses'].append(response)
            results['coherence'].append(self.system_coherence)
            
            # Store pattern every 10 steps to save memory
            if i % 10 == 0:
                results['patterns'].append(pattern)
                
        # Store most interesting patterns to memory
        self._store_interesting_patterns(results)
        
        return results
    
    def _store_interesting_patterns(self, modulation_results: Dict[str, Any]):
        """
        Store the most interesting patterns from a modulation sequence.
        
        Args:
            modulation_results: Results from frequency modulation
        """
        frequencies = modulation_results['frequencies']
        responses = modulation_results['responses']
        coherence = modulation_results['coherence']
        patterns = modulation_results['patterns']
        
        # Find peaks in response
        peak_indices = []
        for i in range(1, len(responses) - 1):
            if responses[i] > responses[i-1] and responses[i] > responses[i+1]:
                # This is a local peak
                peak_indices.append(i)
        
        # If no clear peaks, find highest responses
        if not peak_indices or len(peak_indices) < 3:
            # Sort by response
            sorted_indices = np.argsort(responses)
            
            # Take top 3
            peak_indices = sorted_indices[-3:]
        
        # Find corresponding pattern indices
        pattern_step = 10  # We stored patterns every 10 steps
        pattern_indices = [i // pattern_step for i in peak_indices if i // pattern_step < len(patterns)]
        
        # Store interesting patterns
        for i, pattern_idx in enumerate(pattern_indices):
            if pattern_idx < len(patterns):
                freq = frequencies[peak_indices[i]]
                pattern = patterns[pattern_idx]
                
                # Create name
                name = f"mod_{self.active_material}_{freq:.1f}Hz"
                
                # Store in memory
                self.pattern_memory[name] = {
                    'frequency': freq,
                    'pattern': pattern,
                    'type': 'modulation',
                    'coherence': coherence[peak_indices[i]],
                    'material': self.active_material
                }
    
    def align_with_consciousness(self, consciousness_state: int, intensity: float = 1.0):
        """
        Align the cymatics engine with a specific consciousness state.
        
        Args:
            consciousness_state: The consciousness state (0-5)
            intensity: Intensity of the alignment (0 to 1)
        """
        # Align cymatic field
        self.cymatic_field.align_with_consciousness(consciousness_state, intensity)
        
        # Set sacred frequency based on consciousness state
        consciousness_frequencies = {
            0: 'unity',      # BE state
            1: 'love',       # DO state
            2: 'truth',      # WITNESS state
            3: 'love',       # CREATE state
            4: 'cascade',    # INTEGRATE state
            5: 'vision',     # TRANSCEND state
        }
        
        if consciousness_state in consciousness_frequencies:
            freq_name = consciousness_frequencies[consciousness_state]
            # Blend current frequency with target frequency
            current_freq = self.active_frequency
            target_freq = SACRED_FREQUENCIES[freq_name]
            
            new_freq = (1 - intensity) * current_freq + intensity * target_freq
            self.set_frequency(new_freq)
            
        # Generate appropriate pattern
        pattern = self.pattern_generator.create_consciousness_state_pattern(
            state=consciousness_state,
            resolution=(100, 100)
        )
        
        self.active_pattern = pattern
        
        # Select optimal material based on consciousness state
        consciousness_materials = {
            0: 'water',      # BE state - fluid, adaptable
            1: 'water',      # DO state - motion, flow
            2: 'metal',      # WITNESS state - clear structure
            3: 'crystal',    # CREATE state - crystallization
            4: 'crystal',    # INTEGRATE state - structured harmony
            5: 'metal',      # TRANSCEND state - resonant clarity
        }
        
        if consciousness_state in consciousness_materials:
            material = consciousness_materials[consciousness_state]
            if intensity > 0.5:  # Only change material if intensity is high enough
                self.set_active_material(material)
                
        # Update performance metrics with consciousness factor
        consciousness_factor = 0.7 + 0.3 * intensity
        self.config['consciousness_integration'] = consciousness_factor
        
        # Update metrics
        self._update_performance_metrics()
    
    def connect_with_toroidal_field(self, toroidal_field: ToroidalField, connection_strength: float = 0.8):
        """
        Connect the cymatics engine with a toroidal field for enhanced coherence.
        
        Args:
            toroidal_field: ToroidalField instance
            connection_strength: Strength of the connection (0 to 1)
        """
        # Link cymatic field to toroidal field
        self.cymatic_field = CymaticField(
            base_frequency=self.active_frequency,
            toroidal_base=toroidal_field
        )
        
        # Extract toroidal field properties
        toroidal_coherence = toroidal_field.get_coherence()
        toroidal_flow_balance = toroidal_field.get_flow_balance()
        toroidal_freq = 432.0 * toroidal_field.torus_ratio  # Base frequency scaled by torus ratio
        
        # Apply toroidal influence to frequency
        if connection_strength > 0.1:
            # Blend frequencies
            blended_freq = (1 - connection_strength) * self.active_frequency + connection_strength * toroidal_freq
            self.set_frequency(blended_freq)
            
        # Apply toroidal influence to cymatic patterns
        if connection_strength > 0.1:
            # Apply phi-harmonic modulation influenced by toroidal field
            self.cymatic_field.apply_phi_harmonic_modulation(
                intensity=connection_strength * toroidal_coherence
            )
            
        # Update active pattern
        self.active_pattern = self.cymatic_field.visualize_pattern()
        
        # Update performance metrics based on toroidal connection
        self.config['phi_alignment'] = max(self.config['phi_alignment'], toroidal_coherence * connection_strength)
        self.system_coherence = max(self.system_coherence, toroidal_coherence * connection_strength)
        
        # Update metrics
        self._update_performance_metrics()
    
    def _update_active_pattern(self):
        """Update the active pattern based on current settings."""
        # Get current pattern from the resonator or cymatic field
        if self.active_material in self.resonators:
            resonator = self.resonators[self.active_material]
            self.active_pattern = resonator.get_2d_pattern_slice()
        else:
            self.active_pattern = self.cymatic_field.visualize_pattern()
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        # Calculate manifestation efficiency
        if self.active_material in self.resonators:
            material_response = self.resonators[self.active_material].apply_frequency(
                self.active_frequency, self.energy_level
            )
        else:
            material_response = 0.5
            
        material_coupling = self.config['material_coupling']
        phi_alignment = self.config['phi_alignment']
        
        self.manifestation_efficiency = (
            material_response * PHI + 
            material_coupling * 1.0 + 
            phi_alignment * LAMBDA
        ) / (PHI + 1.0 + LAMBDA)
        
        # Calculate field stability
        if hasattr(self.frequency_modulator, 'get_frequency_stability'):
            frequency_stability = self.frequency_modulator.get_frequency_stability()
        else:
            frequency_stability = 0.9
            
        pattern_stability = self._calculate_pattern_stability()
        
        self.field_stability = (frequency_stability + pattern_stability) / 2.0
        
        # Calculate phase coherence
        if self.active_material in self.resonators:
            resonator_coherence = self.resonators[self.active_material].coherence
        else:
            resonator_coherence = 0.8
            
        field_coherence = self.cymatic_field._calculate_coherence()
        
        self.phase_coherence = (
            resonator_coherence * PHI + 
            field_coherence * 1.0
        ) / (PHI + 1.0)
        
        # Calculate overall system coherence
        consciousness_factor = self.config['consciousness_integration']
        
        self.system_coherence = (
            self.manifestation_efficiency * 1.0 + 
            self.field_stability * LAMBDA + 
            self.phase_coherence * PHI + 
            consciousness_factor * PHI_PHI
        ) / (1.0 + LAMBDA + PHI + PHI_PHI)
    
    def _calculate_pattern_stability(self) -> float:
        """
        Calculate the stability of the current pattern.
        
        Returns:
            Stability value between 0 and 1
        """
        if self.active_pattern is None:
            return 0.5
            
        # Calculate pattern properties
        grad_y, grad_x = np.gradient(self.active_pattern)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate smoothness - lower gradient means more stable
        avg_gradient = np.mean(gradient_magnitude)
        smoothness = 1.0 / (1.0 + avg_gradient * 5.0)
        
        # Calculate structure clarity - measure of distinct features
        # Use FFT to analyze frequency components
        fft = np.fft.fft2(self.active_pattern)
        fft_magnitude = np.abs(fft)
        
        # Normalize
        if np.max(fft_magnitude) > 0:
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)
            
        # Calculate energy concentration
        sorted_magnitudes = np.sort(fft_magnitude.flatten())[::-1]  # Sort descending
        top_energy = np.sum(sorted_magnitudes[:100]) / np.sum(sorted_magnitudes)
        
        structure_clarity = 0.2 + 0.8 * top_energy  # More concentrated energy means clearer structure
        
        # Combine metrics
        stability = (
            smoothness * LAMBDA + 
            structure_clarity * PHI
        ) / (LAMBDA + PHI)
        
        return min(max(stability, 0.0), 1.0)
    
    def estimate_materialization_potential(self) -> Dict[str, float]:
        """
        Estimate the potential for materializing cymatic patterns.
        
        Returns:
            Dictionary with materialization metrics
        """
        # Update metrics
        self._update_performance_metrics()
        
        # Calculate material-specific potential
        material_potentials = {}
        
        for material, resonator in self.resonators.items():
            # Calculate resonator response
            response = resonator.apply_frequency(self.active_frequency)
            
            # Calculate field coherence with material
            if material == 'water':
                field_influence = self.cymatic_field.calculate_material_influence('water')
            elif material == 'crystal':
                field_influence = self.cymatic_field.calculate_material_influence('crystal')
            elif material == 'metal':
                field_influence = self.cymatic_field.calculate_material_influence('metal')
            else:
                field_influence = 0.5
                
            # Calculate phi-resonance
            phi_resonance = calculate_phi_resonance(
                self.active_frequency, 
                resonator.resonant_frequency
            )
            
            # Calculate overall potential
            potential = (
                response * 1.0 + 
                field_influence * PHI + 
                phi_resonance * LAMBDA + 
                self.system_coherence * 0.5
            ) / (1.0 + PHI + LAMBDA + 0.5)
            
            material_potentials[material] = potential
            
        # Calculate overall potential
        if self.active_material in material_potentials:
            active_potential = material_potentials[self.active_material]
        else:
            active_potential = 0.5
            
        overall_potential = (
            active_potential * PHI + 
            self.system_coherence * 1.0 + 
            self.manifestation_efficiency * LAMBDA
        ) / (PHI + 1.0 + LAMBDA)
        
        # Return all metrics
        return {
            'overall_potential': overall_potential,
            'material_potentials': material_potentials,
            'system_coherence': self.system_coherence,
            'manifestation_efficiency': self.manifestation_efficiency,
            'field_stability': self.field_stability,
            'phase_coherence': self.phase_coherence
        }
    
    def find_optimal_frequency(
        self, 
        min_freq: float = 20.0,
        max_freq: float = 2000.0,
        steps: int = 20
    ) -> Dict[str, float]:
        """
        Find the optimal frequency for the current material and configuration.
        
        Args:
            min_freq: Minimum frequency to search (Hz)
            max_freq: Maximum frequency to search (Hz)
            steps: Number of frequency steps to test
            
        Returns:
            Dictionary with optimal frequency information
        """
        # Generate test frequencies (phi-logarithmic spacing)
        start_log = np.log(min_freq) / np.log(PHI)
        end_log = np.log(max_freq) / np.log(PHI)
        
        log_steps = np.linspace(start_log, end_log, steps)
        test_frequencies = [PHI ** log_val for log_val in log_steps]
        
        # Track results
        results = []
        
        # Test each frequency
        for freq in test_frequencies:
            # Save current frequency
            orig_freq = self.active_frequency
            
            # Set test frequency
            self.set_frequency(freq)
            
            # Calculate materialization potential
            potential = self.estimate_materialization_potential()
            
            # Store result
            results.append({
                'frequency': freq,
                'potential': potential['overall_potential'],
                'coherence': self.system_coherence
            })
            
            # Restore original frequency
            self.set_frequency(orig_freq)
            
        # Find optimal frequency
        optimal = max(results, key=lambda x: x['potential'])
        
        # Get nearest sacred frequency
        nearest_sacred = None
        min_distance = float('inf')
        
        for name, sacred_freq in SACRED_FREQUENCIES.items():
            distance = abs(optimal['frequency'] - sacred_freq)
            if distance < min_distance:
                min_distance = distance
                nearest_sacred = {
                    'name': name,
                    'frequency': sacred_freq,
                    'distance': distance
                }
                
        # Add to result
        optimal['nearest_sacred'] = nearest_sacred
        
        return optimal
    
    def store_current_pattern(self, name: str):
        """
        Store the current pattern in memory.
        
        Args:
            name: Name for the stored pattern
        """
        if self.active_pattern is not None:
            # Create pattern entry
            self.pattern_memory[name] = {
                'frequency': self.active_frequency,
                'pattern': self.active_pattern.copy(),
                'type': 'manual',
                'coherence': self.system_coherence,
                'material': self.active_material
            }
    
    def recall_pattern(self, name: str) -> bool:
        """
        Recall a stored pattern from memory.
        
        Args:
            name: Name of the pattern to recall
            
        Returns:
            True if pattern was found and recalled, False otherwise
        """
        if name in self.pattern_memory:
            # Get pattern data
            pattern_data = self.pattern_memory[name]
            
            # Set frequency
            self.set_frequency(pattern_data['frequency'])
            
            # Set active pattern
            self.active_pattern = pattern_data['pattern'].copy()
            
            # Set material if specified
            if 'material' in pattern_data and pattern_data['material'] in self.resonators:
                self.set_active_material(pattern_data['material'])
                
            return True
            
        return False
    
    def combine_patterns(
        self, 
        pattern_names: List[str],
        weights: Optional[List[float]] = None
    ) -> bool:
        """
        Combine multiple stored patterns.
        
        Args:
            pattern_names: List of pattern names to combine
            weights: Optional weights for each pattern
            
        Returns:
            True if patterns were successfully combined, False otherwise
        """
        # Check if all patterns exist
        patterns = []
        for name in pattern_names:
            if name in self.pattern_memory:
                patterns.append(self.pattern_memory[name])
            else:
                return False
                
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
            
        # Check pattern dimensions
        pattern_shapes = [p['pattern'].shape for p in patterns]
        if not all(shape == pattern_shapes[0] for shape in pattern_shapes):
            # Patterns have different dimensions, need to resize
            # For simplicity, just use the first pattern's dimensions
            target_shape = pattern_shapes[0]
            
            # Resize patterns
            for i, pattern in enumerate(patterns):
                if pattern['pattern'].shape != target_shape:
                    # Simple resize using numpy - in practice would use proper interpolation
                    from scipy.ndimage import zoom
                    
                    # Calculate zoom factors
                    zoom_factors = [target_shape[i] / pattern['pattern'].shape[i] 
                                   for i in range(len(target_shape))]
                    
                    # Resize
                    patterns[i]['pattern'] = zoom(pattern['pattern'], zoom_factors)
                    
        # Combine patterns
        combined_pattern = np.zeros_like(patterns[0]['pattern'])
        
        for pattern, weight in zip(patterns, weights):
            combined_pattern += pattern['pattern'] * weight
            
        # Calculate combined frequency (weighted average)
        combined_freq = sum(p['frequency'] * w for p, w in zip(patterns, weights))
        
        # Calculate combined coherence (phi-weighted average)
        coherence_values = [p.get('coherence', 0.8) for p in patterns]
        phi_weights = [PHI ** -i for i in range(len(coherence_values))]
        weight_sum = sum(phi_weights)
        combined_coherence = sum(c * w for c, w in zip(coherence_values, phi_weights)) / weight_sum
        
        # Store combined pattern
        combined_name = '_'.join(pattern_names[:2]) + '_combined'
        self.pattern_memory[combined_name] = {
            'frequency': combined_freq,
            'pattern': combined_pattern,
            'type': 'combined',
            'coherence': combined_coherence,
            'source_patterns': pattern_names,
            'material': self.active_material
        }
        
        # Set as active pattern
        self.active_pattern = combined_pattern
        self.set_frequency(combined_freq)
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics for the cymatic engine.
        
        Returns:
            Dictionary with performance metrics
        """
        # Update metrics
        self._update_performance_metrics()
        
        # Return metrics
        return {
            'manifestation_efficiency': self.manifestation_efficiency,
            'field_stability': self.field_stability,
            'phase_coherence': self.phase_coherence,
            'system_coherence': self.system_coherence,
            'energy_level': self.energy_level,
            'active_frequency': self.active_frequency,
            'material_response': self.resonators[self.active_material].apply_frequency(
                self.active_frequency) if self.active_material in self.resonators else 0.0
        }
    
    def apply_cymatics_to_consciousness(
        self, 
        consciousness_state: int,
        intention: str,
        intensity: float = 0.8
    ) -> Dict[str, Any]:
        """
        Apply cymatic patterns to influence consciousness through resonance.
        
        Args:
            consciousness_state: Target consciousness state (0-5)
            intention: Intention/purpose of the influence
            intensity: Intensity of the influence (0 to 1)
            
        Returns:
            Dictionary with influence metrics
        """
        # Map consciousness states to optimal configurations
        consciousness_configs = {
            0: {  # BE state - Unity/Ground
                'frequency': 'unity',
                'pattern_type': 'CIRCULAR',
                'material': 'water',
                'base_effectiveness': 0.9,
                'phi_factor': 0.8
            },
            1: {  # DO state
                'frequency': 'love',
                'pattern_type': 'SPIRAL',
                'material': 'water',
                'base_effectiveness': 0.85,
                'phi_factor': 0.7
            },
            2: {  # WITNESS state
                'frequency': 'truth',
                'pattern_type': 'MANDALA',
                'material': 'metal',
                'base_effectiveness': 0.8,
                'phi_factor': 0.75
            },
            3: {  # CREATE state
                'frequency': 'love',
                'pattern_type': 'FLOWER',
                'material': 'crystal',
                'base_effectiveness': 0.9,
                'phi_factor': 0.85
            },
            4: {  # INTEGRATE state
                'frequency': 'cascade',
                'pattern_type': 'FRACTAL',
                'material': 'crystal',
                'base_effectiveness': 0.85,
                'phi_factor': 0.9
            },
            5: {  # TRANSCEND state
                'frequency': 'vision',
                'pattern_type': 'HEXAGONAL',
                'material': 'metal',
                'base_effectiveness': 0.8,
                'phi_factor': 0.95
            }
        }
        
        # Get configuration for this state
        if consciousness_state not in consciousness_configs:
            consciousness_state = 0
            
        config = consciousness_configs[consciousness_state]
        
        # Apply configuration
        self.set_sacred_frequency(config['frequency'])
        self.set_active_material(config['material'])
        
        # Generate pattern
        pattern = self.generate_pattern(
            pattern_type=config['pattern_type'],
            symmetry=6
        )
        
        # Apply phi-harmonic modulation
        self.cymatic_field.apply_phi_harmonic_modulation(intensity * config['phi_factor'])
        
        # Calculate effectiveness
        phi_alignment = self.config['phi_alignment']
        pattern_coherence = self._calculate_pattern_stability()
        system_coherence = self.system_coherence
        
        # Base effectiveness from configuration
        base_effectiveness = config['base_effectiveness']
        
        # Calculate intention alignment (simple string matching for demo)
        intention_alignment = 0.8  # Default alignment
        
        # Keywords for different states
        state_keywords = {
            0: ['ground', 'peace', 'calm', 'be', 'center', 'balance'],
            1: ['do', 'action', 'move', 'flow', 'work', 'create'],
            2: ['witness', 'observe', 'truth', 'clarity', 'see', 'perceive'],
            3: ['create', 'manifest', 'form', 'generate', 'produce', 'design'],
            4: ['integrate', 'combine', 'unify', 'balance', 'harmonize', 'connect'],
            5: ['transcend', 'expand', 'vision', 'beyond', 'higher', 'cosmic']
        }
        
        # Check intention against keywords
        if consciousness_state in state_keywords:
            keywords = state_keywords[consciousness_state]
            intention_lower = intention.lower()
            
            # Count matching keywords
            matches = sum(keyword in intention_lower for keyword in keywords)
            
            # Calculate alignment
            if len(keywords) > 0:
                intention_alignment = 0.5 + 0.5 * min(matches / 3, 1.0)
                
        # Calculate overall effectiveness
        effectiveness = (
            base_effectiveness * 1.0 + 
            phi_alignment * PHI + 
            pattern_coherence * LAMBDA + 
            system_coherence * PHI_INVERSE +
            intention_alignment * 0.5
        ) / (1.0 + PHI + LAMBDA + PHI_INVERSE + 0.5)
        
        # Scale by intensity
        effectiveness = effectiveness * intensity
        
        # Return metrics
        return {
            'effectiveness': effectiveness,
            'state': consciousness_state,
            'frequency': self.active_frequency,
            'pattern_coherence': pattern_coherence,
            'system_coherence': system_coherence,
            'intention_alignment': intention_alignment,
            'recommended_duration': 5.0 + 5.0 * effectiveness  # 5-10 minutes depending on effectiveness
        }