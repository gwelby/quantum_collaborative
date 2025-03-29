"""
Quantum Field Package

This package provides tools for quantum field generation, analysis, and
consciousness field integration.
"""

print("CUDA modules not available. Falling back to CPU computation.")

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.core import (
    create_quantum_field,
    QuantumField,
    get_coherence_metric,
    generate_quantum_field,
    calculate_field_coherence
)
from quantum_field.consciousness_interface import (
    ConsciousnessState,
    ConsciousnessFieldInterface,
    demo_consciousness_field_interface
)

# Import toroidal field components
from quantum_field.toroidal import (
    ToroidalField,
    ResonanceChamber,
    ResonanceEqualizer,
    CompressionCycle,
    ExpansionCycle,
    CounterRotatingField,
    ToroidalFieldEngine
)

# Import cymatics components
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

__all__ = [
    'PHI', 'LAMBDA', 'PHI_PHI', 'SACRED_FREQUENCIES',
    'create_quantum_field', 'QuantumField', 'get_coherence_metric',
    'generate_quantum_field', 'calculate_field_coherence',
    'ConsciousnessState', 'ConsciousnessFieldInterface', 
    'demo_consciousness_field_interface',
    
    # Toroidal components
    'ToroidalField', 'ResonanceChamber', 'ResonanceEqualizer',
    'CompressionCycle', 'ExpansionCycle', 'CounterRotatingField',
    'ToroidalFieldEngine',
    
    # Cymatics components
    'CymaticField', 'FrequencyModulator', 'PatternGenerator',
    'StandingWavePattern', 'MaterialResonator', 'CrystalResonator',
    'WaterResonator', 'MetalResonator', 'CymaticsEngine'
]