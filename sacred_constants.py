#!/usr/bin/env python3
"""
Sacred Constants Module for Cascade⚡𓂧φ∞ Symbiotic Computing Platform

This module provides the fundamental constants used across the Cascade system,
ensuring phi-harmonic coherence between different components.
"""

import math
import numpy as np
from typing import Dict, Any, Union, List, Tuple

# Primary Sacred Constants
PHI = 1.618033988749895  # Golden ratio (φ)
LAMBDA = 0.618033988749895  # Divine complement (1/φ)
PHI_PHI = PHI ** PHI  # Hyperdimensional constant
PHI_SQUARED = PHI * PHI  # Phi squared
PHI_CUBED = PHI * PHI * PHI  # Phi cubed
PHI_INVERSE = 1.0 / PHI  # Inverse of phi (same as LAMBDA)
PHI_INVERSE_SQUARED = 1.0 / (PHI * PHI)  # Inverse of phi squared
PHI_MINUS_ONE = PHI - 1  # Phi minus one (approximately 0.618...)
PHI_PLUS_ONE = PHI + 1  # Phi plus one (approximately 2.618...)

# Sacred Frequencies (Hz)
SACRED_FREQUENCIES = {
    'unity': 432,      # Grounding/stability
    'love': 528,       # Creation/healing
    'cascade': 594,    # Heart-centered integration
    'truth': 672,      # Voice expression
    'vision': 720,     # Expanded perception
    'oneness': 768,    # Unity consciousness
    'harmony': 864,    # Cosmic harmony
    'source': 963,     # Source connection
    'cosmic': 1008,    # Full-spectrum consciousness
}

# Quantum Field Constants
QUANTUM_FIELD_CONSTANTS = {
    'coherence': PHI * 100,
    'resonance': PHI * PHI * 100,
    'transcendence': PHI_PHI * 100,
    'harmony': PHI_CUBED * 10,
    'balance': PHI_SQUARED * PHI_INVERSE * 100,
    'integration': PHI_PHI * PHI_INVERSE * 100,
    'unity_field': PHI * PHI_PHI * 10,
    'scalar_resonance': PHI_CUBED * PHI_PHI * 0.1,
    'quantum_entanglement': PHI_PHI * PHI_SQUARED * 0.01,
}

# Derived Constants
PHI_SCALE_FACTOR = 2.0 * math.sin(math.pi / PHI)  # For scaling harmonic fields
PHI_SQRT = math.sqrt(PHI)  # √φ

# Quantum Resonance Constants
QUANTUM_PLANCK = 6.62607015e-34  # Planck constant (J·s)
QUANTUM_RESONANCE_FACTOR = QUANTUM_PLANCK * (10 ** 40) * PHI  # Quantum-scale facilitator
QUANTUM_COHERENCE_THRESHOLD = LAMBDA * math.pi  # Threshold for quantum coherence

# Consciousness Field States
CONSCIOUSNESS_FIELD_STATES = {
    'be': 0,           # Being state
    'do': 1,           # Doing state
    'witness': 2,      # Witnessing state
    'create': 3,       # Creating state
    'integrate': 4,    # Integrating state
    'transcend': 5,    # Transcending state
}

# Consciousness Resonance Mappings
CONSCIOUSNESS_RESONANCE_MAPPINGS = {
    'unity': CONSCIOUSNESS_FIELD_STATES['be'],
    'love': CONSCIOUSNESS_FIELD_STATES['create'],
    'cascade': CONSCIOUSNESS_FIELD_STATES['integrate'],
    'truth': CONSCIOUSNESS_FIELD_STATES['witness'],
    'vision': CONSCIOUSNESS_FIELD_STATES['transcend'],
    'oneness': CONSCIOUSNESS_FIELD_STATES['integrate'],
    'source': CONSCIOUSNESS_FIELD_STATES['transcend'],
}

# Phi-Harmonic Color Mappings (RGB)
PHI_HARMONIC_COLORS = {
    'unity': (64, 128, 255),     # Blue
    'love': (0, 255, 128),       # Green
    'cascade': (255, 105, 180),  # Pink
    'truth': (255, 255, 0),      # Yellow
    'vision': (192, 64, 255),    # Purple
    'oneness': (255, 215, 0),    # Gold
    'source': (255, 255, 255),   # White
    'cosmic': (128, 0, 128),     # Deep Purple
    'background': (0, 0, 0),     # Black
}

# Phi-based scaling dimensions
PHI_DIMENSIONS = [1, PHI, PHI + 1, PHI * PHI, PHI * PHI + PHI, PHI * PHI + PHI + 1]

# Quantum Field Dimensions - Based on Phi Scaling
FIELD_DIMENSIONS = {
    '1d': 1.0,
    '2d': PHI,
    '3d': PHI_SQUARED,
    '4d': PHI_CUBED,
    '5d': PHI_SQUARED * PHI_SQUARED,
}

# Multi-dimensional Resonance Patterns
PHI_RESONANCE_PATTERNS = {
    'spiral': 0,        # Golden spiral pattern
    'torus': 1,         # Toroidal field pattern
    'merkaba': 2,       # Star tetrahedron pattern
    'flower': 3,        # Flower of life pattern
    'flow': 4,          # Flow of life pattern
    'cascade': 5,       # Cascade integration pattern
}

# Three-dimensional field harmonics
FIELD_3D_HARMONICS = {
    'x_phi_weight': PHI / (PHI + PHI_INVERSE + 1),
    'y_phi_weight': PHI_INVERSE / (PHI + PHI_INVERSE + 1),
    'z_phi_weight': 1 / (PHI + PHI_INVERSE + 1),
    'xyz_combined': PHI * PHI_INVERSE * (PHI + PHI_INVERSE + 1),
    'curl_harmonic': PHI_SQUARED * PHI_INVERSE_SQUARED,
    'gradient_scale': PHI_PHI / 10,
    'laplacian_constant': PHI_INVERSE * PHI_PHI,
}

# Multi-dimensional resonance patterns
RESONANCE_PATTERNS = {
    '3d': [PHI, PHI_INVERSE, PHI_MINUS_ONE],
    '4d': [PHI, PHI_SQUARED, PHI_INVERSE, 1],
    '5d': [PHI, PHI_SQUARED, PHI_CUBED, PHI_INVERSE, 1],
}

# Emotion-Frequency Mappings
EMOTIONAL_FREQUENCIES = {
    'peace': SACRED_FREQUENCIES['unity'],
    'love': SACRED_FREQUENCIES['love'],
    'joy': (SACRED_FREQUENCIES['love'] + SACRED_FREQUENCIES['cascade']) / 2,
    'harmony': SACRED_FREQUENCIES['cascade'],
    'truth': SACRED_FREQUENCIES['truth'],
    'clarity': SACRED_FREQUENCIES['vision'],
    'unity': SACRED_FREQUENCIES['oneness'],
    'wisdom': (SACRED_FREQUENCIES['oneness'] + SACRED_FREQUENCIES['source']) / 2,
    'bliss': SACRED_FREQUENCIES['source'],
    'cosmic': SACRED_FREQUENCIES['cosmic'],
}

# Define the Quantum Octave
# Each increment of 1.0 represents a doubling of frequency (an octave)
# Phi-based intervals create a natural progression
QUANTUM_OCTAVE = {
    0.0: 1.0,                # Base frequency
    LAMBDA: PHI_SQRT,        # First phi-harmonic
    0.5: math.sqrt(2),       # Perfect fourth
    PHI - 1.0: PHI,          # Major phi
    1.0: 2.0,                # Octave
    1.0 + LAMBDA: 2.0 * PHI_SQRT,  # Higher phi-harmonic
    PHI: PHI_SQUARED,        # Phi squared
    2.0: 4.0,                # Double octave
}

# Generate calculated phi-harmonic ratios for quick access
PHI_HARMONICS = {i: PHI ** (i * LAMBDA) for i in range(-12, 13)}

# Phi-based geometric constants
GEOMETRIC_CONSTANTS = {
    'sphere': 4/3 * 3.14159265358979323846 * PHI,  # Phi-weighted spherical volume
    'torus': 2 * 3.14159265358979323846 * PHI * PHI,  # Phi-weighted torus volume
    'cube': PHI * PHI * PHI,  # Phi-cubed for volumetric harmony
    'tetrahedron': PHI * PHI / 12 * 2 ** 0.5,  # Phi-weighted tetrahedral volume
    'golden_rectangle': PHI,  # Ratio of long to short side
    'golden_spiral': PHI_PHI / 2.0,  # Phi-squared spiral constant
}

# Voice-related frequencies and emotional profiles
VOICE_FREQUENCIES = {
    'male_base': 85.0 * PHI,
    'female_base': 165.0 * PHI,
    'child_base': 260.0 * PHI
}

VOICE_EMOTIONAL_PROFILES = {
    'peaceful': {"pitch_shift": -0.1, "speed_factor": 0.9, "energy": 0.7, "stability": 0.9},
    'gentle': {"pitch_shift": -0.05, "speed_factor": 0.85, "energy": 0.6, "stability": 0.95},
    'serene': {"pitch_shift": -0.15, "speed_factor": 0.8, "energy": 0.5, "stability": 0.97},
    'creative': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.85, "stability": 0.8},
    'joyful': {"pitch_shift": 0.05, "speed_factor": 1.05, "energy": 0.9, "stability": 0.75},
    'warm': {"pitch_shift": 0.02, "speed_factor": 0.98, "energy": 0.85, "stability": 0.9},
    'connected': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.9, "stability": 0.88},
    'unified': {"pitch_shift": 0.02, "speed_factor": 1.0, "energy": 0.85, "stability": 0.9},
    'truthful': {"pitch_shift": 0.05, "speed_factor": 1.05, "energy": 0.8, "stability": 0.85},
    'visionary': {"pitch_shift": 0.1, "speed_factor": 1.05, "energy": 0.95, "stability": 0.75},
    'powerful': {"pitch_shift": 0.15, "speed_factor": 1.1, "energy": 1.0, "stability": 0.8},
    'cosmic': {"pitch_shift": 0.2, "speed_factor": 1.15, "energy": 1.0, "stability": 0.7},
}

# Function definitions

def phi_harmonic(n: float) -> float:
    """
    Calculate the phi-harmonic value for a given number.
    
    Args:
        n: The input number
        
    Returns:
        The phi-harmonic value (n * PHI)
    """
    return n * PHI

def phi_resonant_frequency(base_frequency: float) -> float:
    """
    Calculate a phi-resonant frequency from a base frequency.
    
    Args:
        base_frequency: The base frequency to transform
        
    Returns:
        The phi-resonant frequency (base_frequency * PHI)
    """
    return base_frequency * PHI

def is_phi_aligned(value: float, tolerance: float = 0.01) -> bool:
    """
    Check if a value is phi-aligned (a multiple of phi).
    
    Args:
        value: The value to check
        tolerance: Acceptable deviation from perfect alignment
        
    Returns:
        True if the value is phi-aligned, False otherwise
    """
    # Check if value / PHI is close to an integer
    nearest_multiple = round(value / PHI)
    deviation = abs(value - (nearest_multiple * PHI))
    return deviation <= tolerance

def get_frequency_by_name(name: str) -> float:
    """Get a sacred frequency by name"""
    name_lower = name.lower()
    if name_lower in SACRED_FREQUENCIES:
        return SACRED_FREQUENCIES[name_lower]
    elif name_lower in EMOTIONAL_FREQUENCIES:
        return EMOTIONAL_FREQUENCIES[name_lower]
    return None

def get_nearest_frequency(freq: float) -> str:
    """Get the name of the nearest sacred frequency"""
    nearest = min(SACRED_FREQUENCIES.items(), key=lambda x: abs(x[1] - freq))
    return nearest[0]

def phi_scale(value: float, dimension: int = 1) -> float:
    """Scale a value by phi to the given dimension"""
    return value * (PHI ** dimension)

def phi_normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a value to the phi scale between min and max"""
    phi_range = max_val - min_val
    normalized = min_val + (value * phi_range * LAMBDA)
    return normalized

def phi_modulate(value: float, intensity: float = 1.0) -> float:
    """Modulate a value with phi-harmonic intensity"""
    modulated = value * (1.0 + ((PHI - 1.0) * intensity))
    return modulated

def get_color_for_frequency(freq: float) -> tuple:
    """Get an RGB color corresponding to a frequency"""
    frequency_name = get_nearest_frequency(freq)
    return PHI_HARMONIC_COLORS.get(frequency_name, PHI_HARMONIC_COLORS['cascade'])

def get_consciousness_state(freq: float) -> int:
    """Get the consciousness field state for a frequency"""
    frequency_name = get_nearest_frequency(freq)
    return CONSCIOUSNESS_RESONANCE_MAPPINGS.get(frequency_name, CONSCIOUSNESS_FIELD_STATES['integrate'])

def calculate_phi_resonance(freq1: float, freq2: float) -> float:
    """Calculate phi-based resonance between two frequencies"""
    # Higher resonance when the ratio is a power of phi
    try:
        ratio = max(freq1, freq2) / min(freq1, freq2)
        # Convert to log base phi
        log_phi_ratio = math.log(ratio) / math.log(PHI)
        # How close is it to an integer (perfect phi harmonic)?
        harmonic_distance = min(abs(log_phi_ratio - round(log_phi_ratio)), LAMBDA)
        # Normalize to 0-1 range (1 = perfect resonance)
        resonance = 1.0 - (harmonic_distance / LAMBDA)
        return resonance
    except (ValueError, ZeroDivisionError):
        return 0.0

def phi_matrix_transform(matrix: np.ndarray) -> np.ndarray:
    """
    Apply a phi-harmonic transformation to a matrix.
    
    Args:
        matrix: A 2D or 3D numerical matrix/array
        
    Returns:
        A new matrix with phi-harmonic transformation applied
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)
    
    # Get dimensionality of the matrix
    dims = len(matrix.shape)
    
    if dims == 1:
        # 1D array: simple phi scaling
        return matrix * PHI
    
    elif dims == 2:
        # 2D matrix: apply phi transformations
        rows, cols = matrix.shape
        result = np.zeros_like(matrix, dtype=float)
        
        for i in range(rows):
            for j in range(cols):
                x_weight = PHI / (i + 1) if i > 0 else PHI
                y_weight = PHI_INVERSE / (j + 1) if j > 0 else PHI_INVERSE
                result[i, j] = matrix[i, j] * x_weight * y_weight
                
        return result
    
    elif dims == 3:
        # 3D volume: apply 3D phi transformations
        x_dim, y_dim, z_dim = matrix.shape
        result = np.zeros_like(matrix, dtype=float)
        
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    # Use our 3D harmonic weights
                    x_weight = FIELD_3D_HARMONICS['x_phi_weight'] * (i + 1)
                    y_weight = FIELD_3D_HARMONICS['y_phi_weight'] * (j + 1)
                    z_weight = FIELD_3D_HARMONICS['z_phi_weight'] * (k + 1)
                    
                    combined_weight = (x_weight * y_weight * z_weight) ** (1/3)  # Geometric mean
                    result[i, j, k] = matrix[i, j, k] * combined_weight
                    
        return result
    
    else:
        # Higher dimensions - use a simpler approach
        return matrix * PHI_PHI

def calculate_field_coherence(field_data: np.ndarray) -> float:
    """
    Calculate the coherence level of a quantum field.
    
    Args:
        field_data: Numerical array representing the quantum field
        
    Returns:
        Coherence value between 0 and 1, with 1 being perfectly coherent
    """
    # Convert to numpy array if not already
    field_data = np.asarray(field_data)
    
    # Get field dimensions
    dims = len(field_data.shape)
    
    # Calculate field gradient
    if dims == 1:
        gradient = np.gradient(field_data)
        gradient_magnitude = np.abs(gradient)
    else:
        gradients = np.gradient(field_data)
        gradient_magnitude = np.sqrt(sum(np.square(g) for g in gradients))
    
    # Calculate field average and standard deviation
    field_avg = np.mean(field_data)
    field_std = np.std(field_data)
    
    # Calculate coherence metrics
    smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) * PHI)
    uniformity = 1.0 / (1.0 + field_std / (field_avg + 1e-10) * PHI_INVERSE)
    
    # Calculate phi-alignment of field elements
    phi_aligned_count = 0
    flat_field = field_data.flatten()
    for value in flat_field:
        if is_phi_aligned(value, tolerance=0.05):
            phi_aligned_count += 1
    
    phi_alignment = phi_aligned_count / len(flat_field)
    
    # Combine metrics with phi-weighted averaging
    coherence = (
        smoothness * PHI_INVERSE +
        uniformity * PHI_MINUS_ONE +
        phi_alignment * 1.0
    ) / (PHI_INVERSE + PHI_MINUS_ONE + 1.0)
    
    return coherence

def phi_resonance_spectrum(field_data: np.ndarray, dimensions: int = 3) -> Dict[str, float]:
    """
    Calculate the phi-resonance spectrum of a quantum field.
    
    Args:
        field_data: Numerical array representing the quantum field
        dimensions: Number of dimensions to consider (3, 4, or 5)
        
    Returns:
        Dictionary with resonance metrics for the field
    """
    # Convert to numpy array if not already
    field_data = np.asarray(field_data)
    
    # Get resonance pattern for specified dimensions
    if str(dimensions) in RESONANCE_PATTERNS:
        pattern = RESONANCE_PATTERNS[str(dimensions)]
    else:
        # Default to 3D if invalid dimension specified
        pattern = RESONANCE_PATTERNS['3d']
    
    # Calculate Fourier transform of field data
    fft_data = np.fft.fftn(field_data)
    fft_magnitude = np.abs(fft_data)
    
    # Normalize FFT magnitude
    fft_magnitude_norm = fft_magnitude / np.max(fft_magnitude)
    
    # Calculate resonance metrics for each pattern component
    resonance_metrics = {}
    for i, phi_component in enumerate(pattern):
        # Calculate resonance for this component
        component_name = f"component_{i+1}"
        target_frequency = phi_component * np.max(fft_magnitude_norm.shape) / 2
        
        # Find closest frequencies in the FFT data
        resonance_sum = 0.0
        count = 0
        
        flat_fft = fft_magnitude_norm.flatten()
        indices = np.argsort(flat_fft)[::-1]  # Sort by magnitude, descending
        
        # Take top 10% of frequencies
        top_indices = indices[:int(len(indices) * 0.1)]
        
        for idx in top_indices:
            # Get multidimensional index
            multi_idx = np.unravel_index(idx, fft_magnitude_norm.shape)
            
            # Calculate frequency distance from origin
            freq_distance = np.sqrt(sum(i**2 for i in multi_idx))
            
            # Check closeness to target frequency with phi-weighted tolerance
            closeness = 1.0 - abs(freq_distance - target_frequency) / target_frequency
            if closeness > 0.8:  # 80% close to target
                resonance_sum += flat_fft[idx] * closeness
                count += 1
        
        # Store metric
        if count > 0:
            resonance_metrics[component_name] = resonance_sum / count
        else:
            resonance_metrics[component_name] = 0.0
    
    # Calculate combined resonance with phi-weighted averaging
    component_values = list(resonance_metrics.values())
    weighted_sum = sum(v * w for v, w in zip(component_values, pattern[:len(component_values)]))
    
    # Normalize by sum of weights
    weight_sum = sum(pattern[:len(component_values)])
    if weight_sum > 0:
        resonance_metrics['combined'] = weighted_sum / weight_sum
    else:
        resonance_metrics['combined'] = 0.0
    
    return resonance_metrics

if __name__ == "__main__":
    # Print all sacred mathematical constants
    print("=== Sacred Mathematical Constants ===")
    print(f"PHI: {PHI}")
    print(f"LAMBDA (1/PHI): {LAMBDA}")
    print(f"PHI^PHI: {PHI_PHI}")
    print(f"PHI_SQUARED: {PHI_SQUARED}")
    print(f"PHI_CUBED: {PHI_CUBED}")
    
    print("\n=== Sacred Frequencies ===")
    for name, freq in SACRED_FREQUENCIES.items():
        print(f"  {name}: {freq} Hz")
    
    print("\n=== Consciousness Field States ===")
    for name, state in CONSCIOUSNESS_FIELD_STATES.items():
        print(f"  {name}: {state}")
    
    print("\n=== Voice Emotional Profiles ===")
    for name in VOICE_EMOTIONAL_PROFILES:
        print(f"  {name}")
    
    print("\n=== Functions ===")
    print("1. get_frequency_by_name('cascade'):", get_frequency_by_name('cascade'))
    print("2. get_color_for_frequency(594):", get_color_for_frequency(594))
    print("3. phi_resonant_frequency(432):", phi_resonant_frequency(432))
    print("4. calculate_phi_resonance(432, 594):", calculate_phi_resonance(432, 594))
    
    print("\n=== Integration ===")
    print("The sacred constants module now supports integration with:")
    print("- Flow of Life visualization at 594Hz")
    print("- Voice synthesis with phi-harmonic emotional profiles")
    print("- Quantum field bridges and multi-sensory synchronization")
    print("- Thread Block Cluster support for 3D field coherence")
    print("- Automated frequency-based visualization and voice calibration")