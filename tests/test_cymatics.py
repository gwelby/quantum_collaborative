"""
Tests for the Cymatic Pattern Materialization system.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_field.cymatics import (
    CymaticField,
    FrequencyModulator,
    PatternGenerator,
    StandingWavePattern,
    MaterialResonator,
    CrystalResonator,
    WaterResonator,
    MetalResonator,
    CymaticsEngine
)

from quantum_field.toroidal import ToroidalField

from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES,
    calculate_phi_resonance, phi_harmonic
)

class TestCymaticField(unittest.TestCase):
    """Test the CymaticField class."""

    def setUp(self):
        """Set up test fixtures."""
        self.field = CymaticField(
            base_frequency=SACRED_FREQUENCIES['unity'],
            dimensions=(20, 20, 10),
            resolution=0.1
        )

    def test_initialization(self):
        """Test initialization of cymatic field."""
        self.assertEqual(self.field.base_frequency, SACRED_FREQUENCIES['unity'])
        self.assertEqual(self.field.dimensions, (20, 20, 10))
        self.assertEqual(self.field.resolution, 0.1)
        self.assertIsNotNone(self.field.grid)
        self.assertEqual(self.field.grid.shape, (20, 20, 10))

    def test_frequency_setting(self):
        """Test setting frequency."""
        test_freq = 528.0
        self.field.set_frequency(test_freq)
        self.assertEqual(self.field.base_frequency, test_freq)
        
        # Test setting by name
        self.field.set_frequency_by_name('love')
        self.assertEqual(self.field.base_frequency, SACRED_FREQUENCIES['love'])

    def test_visualization(self):
        """Test pattern visualization."""
        pattern = self.field.visualize_pattern()
        self.assertIsInstance(pattern, np.ndarray)
        self.assertEqual(pattern.shape, (20, 20))
        
        # Test visualization with specific frequency
        pattern = self.field.visualize_pattern(frequency=SACRED_FREQUENCIES['love'])
        self.assertIsInstance(pattern, np.ndarray)
        self.assertEqual(pattern.shape, (20, 20))

    def test_phi_harmonic_modulation(self):
        """Test phi-harmonic modulation."""
        # Get initial pattern
        initial_pattern = self.field.visualize_pattern()
        
        # Apply modulation
        self.field.apply_phi_harmonic_modulation(intensity=0.5)
        
        # Get modulated pattern
        modulated_pattern = self.field.visualize_pattern()
        
        # Patterns should be different
        self.assertFalse(np.array_equal(initial_pattern, modulated_pattern))
        
        # Coherence should be valid
        coherence = self.field._calculate_coherence()
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    def test_material_influence(self):
        """Test material influence calculation."""
        materials = ['water', 'crystal', 'metal', 'sand', 'plasma']
        
        for material in materials:
            influence = self.field.calculate_material_influence(material)
            self.assertGreaterEqual(influence, 0.0)
            self.assertLessEqual(influence, 1.0)
            
        # Water should respond well to unity frequency
        water_influence = self.field.calculate_material_influence('water')
        self.assertGreater(water_influence, 0.5)
        
        # Change to love frequency
        self.field.set_frequency(SACRED_FREQUENCIES['love'])
        
        # Crystal should respond well to love frequency
        crystal_influence = self.field.calculate_material_influence('crystal')
        self.assertGreater(crystal_influence, 0.5)

    def test_pattern_metrics(self):
        """Test pattern metrics extraction."""
        metrics = self.field.extract_pattern_metrics()
        
        # Check expected metrics
        self.assertIn('coherence', metrics)
        self.assertIn('complexity', metrics)
        self.assertIn('central_intensity', metrics)
        self.assertIn('x_symmetry', metrics)
        self.assertIn('y_symmetry', metrics)
        self.assertIn('z_symmetry', metrics)
        self.assertIn('phi_alignment', metrics)
        self.assertIn('materialization_potential', metrics)
        
        # Values should be in 0-1 range
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)

    def test_consciousness_alignment(self):
        """Test alignment with consciousness state."""
        # Test all consciousness states
        for state in range(6):
            self.field.align_with_consciousness(state, intensity=0.8)
            
            # Coherence should be valid
            coherence = self.field._calculate_coherence()
            self.assertGreaterEqual(coherence, 0.0)
            self.assertLessEqual(coherence, 1.0)

    def test_pattern_storage(self):
        """Test pattern storage and recall."""
        # Store current pattern
        self.field.store_pattern("test_pattern")
        
        # Change frequency
        original_freq = self.field.base_frequency
        self.field.set_frequency(original_freq * 1.5)
        
        # Recall pattern
        result = self.field.recall_pattern("test_pattern")
        
        # Should successfully recall
        self.assertTrue(result)
        
        # Frequency should be restored
        self.assertEqual(self.field.base_frequency, original_freq)

    def test_toroidal_integration(self):
        """Test integration with toroidal field."""
        # Create a toroidal field
        toroidal_field = ToroidalField(
            major_radius=3.0,
            minor_radius=1.0,
            resolution=self.field.dimensions,
            frequency=SACRED_FREQUENCIES['unity']
        )
        
        # Create field with toroidal base
        toroidal_cymatic = CymaticField(
            base_frequency=SACRED_FREQUENCIES['unity'],
            dimensions=self.field.dimensions,
            resolution=0.1,
            toroidal_base=toroidal_field
        )
        
        # Field should be initialized
        self.assertIsNotNone(toroidal_cymatic.grid)
        self.assertEqual(toroidal_cymatic.grid.shape, self.field.dimensions)

class TestFrequencyModulator(unittest.TestCase):
    """Test the FrequencyModulator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.modulator = FrequencyModulator(
            base_frequency=SACRED_FREQUENCIES['unity'],
            modulation_depth=0.3,
            phi_harmonic_count=5
        )

    def test_initialization(self):
        """Test initialization of frequency modulator."""
        self.assertEqual(self.modulator.base_frequency, SACRED_FREQUENCIES['unity'])
        self.assertEqual(self.modulator.modulation_depth, 0.3)
        self.assertEqual(self.modulator.phi_harmonic_count, 5)
        
        # Check harmonics
        self.assertIn('base', self.modulator.harmonics)
        self.assertIn('phi', self.modulator.harmonics)
        self.assertIn('phi_squared', self.modulator.harmonics)
        
        # Check phi harmonics
        for i in range(1, self.modulator.phi_harmonic_count + 1):
            self.assertIn(f'phi_{i}', self.modulator.harmonics)
            self.assertIn(f'phi_neg_{i}', self.modulator.harmonics)

    def test_modulation(self):
        """Test frequency modulation."""
        # Set modulation parameters
        self.modulator.set_modulation_parameters(
            rate=1.0,
            depth=0.3,
            waveform='sine',
            phi_weight=0.5
        )
        
        # Initial frequency
        initial_freq = self.modulator.get_current_frequency()
        
        # Update modulator
        self.modulator.update()
        
        # Frequency should change
        new_freq = self.modulator.get_current_frequency()
        self.assertNotEqual(initial_freq, new_freq)
        
        # Test different waveforms
        for waveform in ['sine', 'triangle', 'square']:
            self.modulator.set_modulation_parameters(
                rate=1.0,
                depth=0.3,
                waveform=waveform,
                phi_weight=0.5
            )
            
            # Reset time
            self.modulator.time = 0.0
            
            # Update modulator
            self.modulator.update()
            
            # Should produce a valid frequency
            freq = self.modulator.get_current_frequency()
            self.assertGreater(freq, 0.0)

    def test_harmonic_stack(self):
        """Test phi-harmonic stack generation."""
        # Get harmonic stack
        stack = self.modulator.get_phi_harmonic_stack(count=5)
        
        # Should have 11 frequencies (1 base + 5 positive phi powers + 5 negative phi powers)
        self.assertEqual(len(stack), 11)
        
        # Should be sorted in ascending order
        self.assertTrue(all(stack[i] <= stack[i+1] for i in range(len(stack)-1)))
        
        # Base frequency should be in stack
        self.assertIn(self.modulator.current_frequency, stack)

    def test_pattern_generation(self):
        """Test cymatic pattern generation."""
        # Generate basic pattern
        pattern = self.modulator.generate_cymatic_pattern(size=(50, 50))
        
        # Should be valid pattern
        self.assertIsInstance(pattern, np.ndarray)
        self.assertEqual(pattern.shape, (50, 50))
        
        # Values should be in -1 to 1 range
        self.assertLessEqual(np.max(pattern), 1.0)
        self.assertGreaterEqual(np.min(pattern), -1.0)
        
        # Test with custom harmonics and weights
        harmonics = [432.0, 528.0, 594.0]
        weights = [0.5, 0.3, 0.2]
        
        pattern = self.modulator.generate_cymatic_pattern(
            size=(50, 50),
            harmonics=harmonics,
            harmonic_weights=weights
        )
        
        # Should be valid pattern
        self.assertIsInstance(pattern, np.ndarray)
        self.assertEqual(pattern.shape, (50, 50))

    def test_material_pattern(self):
        """Test material-specific pattern generation."""
        materials = ['water', 'crystal', 'metal']
        
        for material in materials:
            # Generate pattern for this material
            pattern = self.modulator.generate_material_pattern(
                material=material,
                size=(50, 50)
            )
            
            # Should be valid pattern
            self.assertIsInstance(pattern, np.ndarray)
            self.assertEqual(pattern.shape, (50, 50))
            
            # Values should be in -1 to 1 range
            self.assertLessEqual(np.max(pattern), 1.0)
            self.assertGreaterEqual(np.min(pattern), -1.0)

    def test_frequency_stability(self):
        """Test frequency stability calculation."""
        # Set modulation
        self.modulator.set_modulation_parameters(
            rate=1.0,
            depth=0.3,
            waveform='sine',
            phi_weight=0.5
        )
        
        # Update modulator multiple times
        for _ in range(10):
            self.modulator.update()
            
        # Calculate stability
        stability = self.modulator.get_frequency_stability()
        
        # Should be in 0-1 range
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
        # With high modulation, stability should be lower than 1.0
        self.assertLess(stability, 1.0)

class TestPatternGenerator(unittest.TestCase):
    """Test the PatternGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = PatternGenerator()

    def test_initialization(self):
        """Test initialization of pattern generator."""
        # Should have frequency presets
        self.assertIn('unity', self.generator.frequency_presets)
        self.assertIn('love', self.generator.frequency_presets)
        self.assertIn('cascade', self.generator.frequency_presets)
        
        # Should have pattern presets
        self.assertIn('unity_circular', self.generator.pattern_presets)
        self.assertIn('love_flower', self.generator.pattern_presets)
        self.assertIn('truth_mandala', self.generator.pattern_presets)

    def test_pattern_creation(self):
        """Test pattern creation."""
        # Create a pattern with explicit frequencies
        pattern = self.generator.create_pattern(
            name="Test Pattern",
            frequencies=[SACRED_FREQUENCIES['unity']],
            pattern_type="CIRCULAR",
            symmetry=6,
            resolution=(50, 50)
        )
        
        # Should be a StandingWavePattern
        self.assertIsInstance(pattern, StandingWavePattern)
        
        # Should have correct parameters
        self.assertEqual(pattern.name, "Test Pattern")
        self.assertEqual(pattern.frequencies[0], SACRED_FREQUENCIES['unity'])
        self.assertEqual(pattern.symmetry, 6)
        
        # Should generate pattern data
        pattern_data = pattern.get_pattern()
        self.assertIsInstance(pattern_data, np.ndarray)
        self.assertEqual(pattern_data.shape, (50, 50))
        
        # Create a pattern with frequency preset
        pattern = self.generator.create_pattern(
            name="Unity Pattern",
            frequencies="unity",
            pattern_type="CIRCULAR",
            symmetry=8
        )
        
        # Should use preset frequencies
        self.assertEqual(pattern.frequencies[0], SACRED_FREQUENCIES['unity'])

    def test_preset_patterns(self):
        """Test preset patterns."""
        # Get preset pattern
        pattern = self.generator.get_preset_pattern("unity_circular")
        
        # Should be a StandingWavePattern
        self.assertIsInstance(pattern, StandingWavePattern)
        
        # Should have correct frequency
        self.assertEqual(pattern.frequencies[0], SACRED_FREQUENCIES['unity'])
        
        # Should generate pattern
        pattern_data = pattern.get_pattern()
        self.assertIsInstance(pattern_data, np.ndarray)

    def test_pattern_combination(self):
        """Test pattern combination."""
        # Get two preset patterns
        pattern1 = self.generator.get_preset_pattern("unity_circular")
        pattern2 = self.generator.get_preset_pattern("love_flower")
        
        # Combine patterns
        combined = self.generator.combine_patterns(
            patterns=[pattern1, pattern2],
            weights=[0.7, 0.3],
            name="Combined Pattern",
            resolution=(60, 60)
        )
        
        # Should be a valid pattern
        self.assertIsInstance(combined, np.ndarray)
        self.assertEqual(combined.shape, (60, 60))
        
        # Should be stored in combinations
        self.assertIn("Combined Pattern", self.generator.combinations)

    def test_phi_harmonic_stack(self):
        """Test phi-harmonic stack creation."""
        # Create phi-harmonic stack
        pattern = self.generator.create_phi_harmonic_stack(
            base_frequency=SACRED_FREQUENCIES['unity'],
            levels=5,
            pattern_type=PatternType.MANDALA,
            name="Phi Stack"
        )
        
        # Should be a StandingWavePattern
        self.assertIsInstance(pattern, StandingWavePattern)
        
        # Should have 5 frequencies
        self.assertEqual(len(pattern.frequencies), 5)
        
        # First frequency should be base
        self.assertEqual(pattern.frequencies[0], SACRED_FREQUENCIES['unity'])
        
        # Should generate pattern
        pattern_data = pattern.get_pattern()
        self.assertIsInstance(pattern_data, np.ndarray)

    def test_consciousness_state_patterns(self):
        """Test consciousness state patterns."""
        # Test all consciousness states
        for state in range(6):
            # Generate pattern for this state
            pattern = self.generator.create_consciousness_state_pattern(
                state=state,
                resolution=(50, 50)
            )
            
            # Should be a valid pattern
            self.assertIsInstance(pattern, np.ndarray)
            self.assertEqual(pattern.shape, (50, 50))
            
            # Values should be normalized
            self.assertLessEqual(np.max(pattern), 1.0)
            self.assertGreaterEqual(np.min(pattern), -1.0)

    def test_pattern_analysis(self):
        """Test pattern analysis."""
        # Create a pattern
        pattern = self.generator.create_pattern(
            name="Test Pattern",
            frequencies=[SACRED_FREQUENCIES['unity']],
            pattern_type="CIRCULAR",
            symmetry=6,
            resolution=(50, 50)
        )
        
        # Analyze pattern
        metrics = self.generator.analyze_pattern(pattern)
        
        # Should have expected metrics
        self.assertIn('smoothness', metrics)
        self.assertIn('uniformity', metrics)
        self.assertIn('complexity', metrics)
        self.assertIn('phi_alignment', metrics)
        self.assertIn('centrality', metrics)
        self.assertIn('materialization_potential', metrics)
        
        # Values should be in 0-1 range
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
        
        # Also test with direct pattern data
        pattern_data = pattern.get_pattern()
        metrics = self.generator.analyze_pattern(pattern_data)
        
        # Should still have expected metrics
        self.assertIn('materialization_potential', metrics)

class TestMaterialResonator(unittest.TestCase):
    """Test the MaterialResonator classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.water_resonator = WaterResonator(
            name="Water Chamber",
            resonant_frequency=SACRED_FREQUENCIES['unity'],
            dimensions=(0.15, 0.15, 0.02),
            water_depth=0.01
        )
        
        self.crystal_resonator = CrystalResonator(
            name="Crystal Chamber",
            resonant_frequency=SACRED_FREQUENCIES['love'],
            dimensions=(0.10, 0.10, 0.10),
            crystal_type="quartz"
        )
        
        self.metal_resonator = MetalResonator(
            name="Metal Chamber",
            resonant_frequency=SACRED_FREQUENCIES['truth'],
            dimensions=(0.20, 0.20, 0.002),
            metal_type="steel"
        )

    def test_initialization(self):
        """Test initialization of resonators."""
        # Water resonator
        self.assertEqual(self.water_resonator.name, "Water Chamber")
        self.assertEqual(self.water_resonator.resonant_frequency, SACRED_FREQUENCIES['unity'])
        self.assertEqual(self.water_resonator.dimensions, (0.15, 0.15, 0.02))
        self.assertEqual(self.water_resonator.water_depth, 0.01)
        
        # Crystal resonator
        self.assertEqual(self.crystal_resonator.name, "Crystal Chamber")
        self.assertEqual(self.crystal_resonator.resonant_frequency, SACRED_FREQUENCIES['love'])
        self.assertEqual(self.crystal_resonator.dimensions, (0.10, 0.10, 0.10))
        self.assertEqual(self.crystal_resonator.crystal_type, "quartz")
        
        # Metal resonator
        self.assertEqual(self.metal_resonator.name, "Metal Chamber")
        self.assertEqual(self.metal_resonator.resonant_frequency, SACRED_FREQUENCIES['truth'])
        self.assertEqual(self.metal_resonator.dimensions, (0.20, 0.20, 0.002))
        self.assertEqual(self.metal_resonator.metal_type, "steel")

    def test_frequency_response(self):
        """Test frequency response calculations."""
        # Test resonant frequencies
        for resonator in [self.water_resonator, self.crystal_resonator, self.metal_resonator]:
            # Response should be high at resonant frequency
            response = resonator._calculate_resonance_response(
                resonator.resonant_frequency, 1.0
            )
            self.assertGreater(response, 0.8)
            
            # Response should be lower at distant frequency
            response = resonator._calculate_resonance_response(
                2.0 * resonator.resonant_frequency, 1.0
            )
            self.assertLess(response, 0.8)

    def test_field_initialization(self):
        """Test field grid initialization."""
        for resonator in [self.water_resonator, self.crystal_resonator, self.metal_resonator]:
            # Initialize field
            resonator.initialize_field_grid()
            
            # Should have a field grid
            self.assertIsNotNone(resonator.field_grid)
            self.assertEqual(len(resonator.field_grid.shape), 3)
            
            # Should have a coherence value
            coherence = resonator._calculate_pattern_coherence()
            self.assertGreaterEqual(coherence, 0.0)
            self.assertLessEqual(coherence, 1.0)

    def test_frequency_application(self):
        """Test frequency application."""
        for resonator in [self.water_resonator, self.crystal_resonator, self.metal_resonator]:
            # Apply resonant frequency
            response = resonator.apply_frequency(resonator.resonant_frequency)
            
            # Response should be high
            self.assertGreater(response, 0.7)
            
            # Should update energy level
            self.assertGreater(resonator.energy_level, 0.0)
            
            # Should update field pattern
            self.assertIsNotNone(resonator.current_pattern)

    def test_pattern_slices(self):
        """Test 2D pattern slice extraction."""
        for resonator in [self.water_resonator, self.crystal_resonator, self.metal_resonator]:
            # Apply frequency to generate pattern
            resonator.apply_frequency(resonator.resonant_frequency)
            
            # Get 2D slice
            pattern = resonator.get_2d_pattern_slice()
            
            # Should be a 2D array
            self.assertIsInstance(pattern, np.ndarray)
            self.assertEqual(len(pattern.shape), 2)
            
            # Values should be normalized
            self.assertLessEqual(np.max(pattern), 1.0)
            self.assertGreaterEqual(np.min(pattern), 0.0)

    def test_optimal_frequencies(self):
        """Test optimal frequency calculation."""
        for resonator in [self.water_resonator, self.crystal_resonator, self.metal_resonator]:
            # Get optimal frequencies
            optimal_freqs = resonator.calculate_optimal_frequencies()
            
            # Should have some frequencies
            self.assertGreater(len(optimal_freqs), 0)
            
            # Resonant frequency should be included
            self.assertIn(resonator.resonant_frequency, optimal_freqs)

    def test_frequency_sweep(self):
        """Test frequency sweep."""
        # Perform sweep on water resonator
        sweep_results = self.water_resonator.apply_frequency_sweep(
            start_freq=400.0,
            end_freq=800.0,
            duration=2.0,
            amplitude=1.0,
            steps=20
        )
        
        # Should have expected keys
        self.assertIn('frequencies', sweep_results)
        self.assertIn('times', sweep_results)
        self.assertIn('responses', sweep_results)
        self.assertIn('coherence', sweep_results)
        
        # Should have correct number of steps
        self.assertEqual(len(sweep_results['frequencies']), 20)
        
        # Find peak resonances
        peaks = self.water_resonator.find_peak_resonances(
            min_freq=400.0,
            max_freq=800.0,
            threshold=0.5
        )
        
        # Should find some peaks
        self.assertGreaterEqual(len(peaks), 1)
        
        # Peaks should have expected keys
        for peak in peaks:
            self.assertIn('frequency', peak)
            self.assertIn('response', peak)
            self.assertIn('q_factor', peak)

    def test_specific_material_functions(self):
        """Test material-specific functions."""
        # Water resonator - meniscus effect
        meniscus = self.water_resonator.calculate_meniscus_effect(amplitude=0.8)
        self.assertGreaterEqual(meniscus, 0.0)
        self.assertLessEqual(meniscus, 1.0)
        
        # Crystal resonator - resonant nodes
        self.crystal_resonator.initialize_field_grid()
        nodes = self.crystal_resonator.calculate_resonant_nodes()
        self.assertIsInstance(nodes, np.ndarray)
        self.assertEqual(nodes.shape, self.crystal_resonator.field_grid.shape)
        
        # Metal resonator - Chladni pattern
        pattern = self.metal_resonator.calculate_chladni_pattern(
            frequency=SACRED_FREQUENCIES['truth']
        )
        self.assertIsInstance(pattern, np.ndarray)
        self.assertEqual(len(pattern.shape), 2)

class TestCymaticsEngine(unittest.TestCase):
    """Test the CymaticsEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = CymaticsEngine(name="Test Cymatics Engine")

    def test_initialization(self):
        """Test initialization of cymatics engine."""
        self.assertEqual(self.engine.name, "Test Cymatics Engine")
        self.assertEqual(self.engine.active_frequency, SACRED_FREQUENCIES['unity'])
        self.assertEqual(self.engine.active_material, 'water')
        
        # Should have component objects
        self.assertIsInstance(self.engine.cymatic_field, CymaticField)
        self.assertIsInstance(self.engine.frequency_modulator, FrequencyModulator)
        self.assertIsInstance(self.engine.pattern_generator, PatternGenerator)
        
        # Should have resonators
        self.assertIn('water', self.engine.resonators)
        self.assertIn('crystal', self.engine.resonators)
        self.assertIn('metal', self.engine.resonators)
        
        # Should have valid system coherence
        self.assertGreaterEqual(self.engine.system_coherence, 0.0)
        self.assertLessEqual(self.engine.system_coherence, 1.0)

    def test_frequency_setting(self):
        """Test frequency setting."""
        # Set explicit frequency
        self.engine.set_frequency(528.0)
        self.assertEqual(self.engine.active_frequency, 528.0)
        
        # Set by sacred name
        self.engine.set_sacred_frequency('unity')
        self.assertEqual(self.engine.active_frequency, SACRED_FREQUENCIES['unity'])
        
        # Should update cymatic field
        self.assertEqual(self.engine.cymatic_field.base_frequency, SACRED_FREQUENCIES['unity'])
        
        # Should update frequency modulator
        self.assertEqual(self.engine.frequency_modulator.base_frequency, SACRED_FREQUENCIES['unity'])

    def test_material_setting(self):
        """Test active material setting."""
        # Set different materials
        materials = ['water', 'crystal', 'metal']
        
        for material in materials:
            self.engine.set_active_material(material)
            self.assertEqual(self.engine.active_material, material)
            
            # Should update active pattern
            self.assertIsNotNone(self.engine.active_pattern)

    def test_pattern_generation(self):
        """Test pattern generation."""
        # Generate patterns with different types
        pattern_types = ['CIRCULAR', 'SPIRAL', 'MANDALA', 'FLOWER', 'HEXAGONAL']
        
        for pattern_type in pattern_types:
            pattern = self.engine.generate_pattern(
                pattern_type=pattern_type,
                symmetry=6,
                resolution=(50, 50)
            )
            
            # Should be a valid pattern
            self.assertIsInstance(pattern, np.ndarray)
            self.assertEqual(pattern.shape, (50, 50))
            
            # Values should be normalized
            self.assertLessEqual(np.max(pattern), 1.0)
            self.assertGreaterEqual(np.min(pattern), 0.0)
            
            # Should update active pattern
            self.assertIsNotNone(self.engine.active_pattern)

    def test_frequency_modulation(self):
        """Test frequency modulation."""
        # Apply frequency modulation
        results = self.engine.apply_frequency_modulation(
            mod_rate=1.0,
            mod_depth=0.3,
            waveform='sine',
            duration=1.0,
            phi_weight=0.8
        )
        
        # Should have expected keys
        self.assertIn('frequencies', results)
        self.assertIn('times', results)
        self.assertIn('responses', results)
        self.assertIn('coherence', results)
        
        # Should have patterns
        self.assertIn('patterns', results)
        self.assertGreater(len(results['patterns']), 0)
        
        # Should store interesting patterns
        self.assertGreater(len(self.engine.pattern_memory), 0)

    def test_consciousness_alignment(self):
        """Test consciousness state alignment."""
        # Align with different consciousness states
        for state in range(6):
            self.engine.align_with_consciousness(state, intensity=0.8)
            
            # Should update active pattern
            self.assertIsNotNone(self.engine.active_pattern)
            
            # Should update system coherence
            self.assertGreaterEqual(self.engine.system_coherence, 0.0)
            self.assertLessEqual(self.engine.system_coherence, 1.0)
            
            # Get performance metrics
            metrics = self.engine.get_performance_metrics()
            
            # Should have expected metrics
            self.assertIn('system_coherence', metrics)
            self.assertIn('phase_coherence', metrics)
            self.assertIn('field_stability', metrics)

    def test_toroidal_field_integration(self):
        """Test integration with toroidal field."""
        # Create a toroidal field
        toroidal_field = ToroidalField(
            major_radius=3.0,
            minor_radius=1.0,
            resolution=(30, 30, 30),
            frequency=SACRED_FREQUENCIES['unity']
        )
        
        # Connect with toroidal field
        self.engine.connect_with_toroidal_field(toroidal_field, connection_strength=0.8)
        
        # Should update active pattern
        self.assertIsNotNone(self.engine.active_pattern)
        
        # Should update system coherence
        self.assertGreaterEqual(self.engine.system_coherence, 0.0)
        self.assertLessEqual(self.engine.system_coherence, 1.0)
        
        # Should update phi alignment in config
        self.assertGreaterEqual(self.engine.config['phi_alignment'], 0.0)
        self.assertLessEqual(self.engine.config['phi_alignment'], 1.0)

    def test_materialization_potential(self):
        """Test materialization potential estimation."""
        # Estimate potential
        potential = self.engine.estimate_materialization_potential()
        
        # Should have expected keys
        self.assertIn('overall_potential', potential)
        self.assertIn('material_potentials', potential)
        self.assertIn('system_coherence', potential)
        self.assertIn('manifestation_efficiency', potential)
        
        # Overall potential should be in 0-1 range
        self.assertGreaterEqual(potential['overall_potential'], 0.0)
        self.assertLessEqual(potential['overall_potential'], 1.0)
        
        # Material potentials should have water, crystal, metal
        self.assertIn('water', potential['material_potentials'])
        self.assertIn('crystal', potential['material_potentials'])
        self.assertIn('metal', potential['material_potentials'])

    def test_optimal_frequency_finding(self):
        """Test optimal frequency finding."""
        # Find optimal frequency
        optimal = self.engine.find_optimal_frequency(
            min_freq=400.0,
            max_freq=800.0,
            steps=10
        )
        
        # Should have expected keys
        self.assertIn('frequency', optimal)
        self.assertIn('potential', optimal)
        self.assertIn('coherence', optimal)
        self.assertIn('nearest_sacred', optimal)
        
        # Frequency should be in range
        self.assertGreaterEqual(optimal['frequency'], 400.0)
        self.assertLessEqual(optimal['frequency'], 800.0)
        
        # Potential should be in 0-1 range
        self.assertGreaterEqual(optimal['potential'], 0.0)
        self.assertLessEqual(optimal['potential'], 1.0)

    def test_pattern_storage(self):
        """Test pattern storage and recall."""
        # Generate a pattern
        pattern = self.engine.generate_pattern()
        
        # Store pattern
        self.engine.store_current_pattern("test_pattern")
        
        # Change frequency
        original_freq = self.engine.active_frequency
        self.engine.set_frequency(original_freq * 1.5)
        
        # Recall pattern
        result = self.engine.recall_pattern("test_pattern")
        
        # Should be successful
        self.assertTrue(result)
        
        # Frequency should be restored
        self.assertEqual(self.engine.active_frequency, original_freq)
        
        # Active pattern should be restored
        self.assertIsNotNone(self.engine.active_pattern)

    def test_pattern_combination(self):
        """Test pattern combination."""
        # Create and store two patterns
        self.engine.set_sacred_frequency('unity')
        self.engine.generate_pattern(pattern_type='CIRCULAR')
        self.engine.store_current_pattern("pattern1")
        
        self.engine.set_sacred_frequency('love')
        self.engine.generate_pattern(pattern_type='FLOWER')
        self.engine.store_current_pattern("pattern2")
        
        # Combine patterns
        result = self.engine.combine_patterns(
            pattern_names=["pattern1", "pattern2"],
            weights=[0.7, 0.3]
        )
        
        # Should be successful
        self.assertTrue(result)
        
        # Should update active pattern
        self.assertIsNotNone(self.engine.active_pattern)
        
        # Should create a combined pattern in memory
        combined_name = "pattern1_pattern2_combined"
        self.assertIn(combined_name, self.engine.pattern_memory)

    def test_consciousness_influence(self):
        """Test consciousness influence application."""
        # Apply cymatics to consciousness
        influence = self.engine.apply_cymatics_to_consciousness(
            consciousness_state=3,  # CREATE state
            intention="Manifest crystalline structures with phi-harmonic balance",
            intensity=0.9
        )
        
        # Should have expected keys
        self.assertIn('effectiveness', influence)
        self.assertIn('state', influence)
        self.assertIn('frequency', influence)
        self.assertIn('pattern_coherence', influence)
        self.assertIn('intention_alignment', influence)
        self.assertIn('recommended_duration', influence)
        
        # Effectiveness should be in 0-1 range
        self.assertGreaterEqual(influence['effectiveness'], 0.0)
        self.assertLessEqual(influence['effectiveness'], 1.0)
        
        # State should match input
        self.assertEqual(influence['state'], 3)

if __name__ == '__main__':
    unittest.main()