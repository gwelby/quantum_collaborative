"""
Tests for the Toroidal Field Dynamics system.
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.toroidal import (
    ToroidalField,
    ToroidalFieldEngine,
    ResonanceChamber,
    CompressionCycle,
    CounterRotatingField,
)

def test_toroidal_field_creation():
    """Test basic creation of toroidal field."""
    # Create field with default parameters
    field = ToroidalField(dimensions=(16, 16, 16))
    
    # Check field dimensions
    assert field.data.shape == (16, 16, 16)
    
    # Check non-zero data
    assert np.sum(np.abs(field.data)) > 0
    
    # Calculate metrics
    metrics = field.calculate_flow_metrics()
    
    # Check metrics exist
    assert 'coherence' in metrics
    assert 'flow_rate' in metrics
    assert 'balance_factor' in metrics
    
    # Check coherence is reasonable
    assert 0 <= metrics['coherence'] <= 1

def test_toroidal_field_flow():
    """Test toroidal field flow evolution."""
    # Create field
    field = ToroidalField(dimensions=(16, 16, 16))
    
    # Store initial data
    initial_data = field.data.copy()
    
    # Apply flow
    field.apply_flow(time_factor=0.1)
    
    # Check data has changed
    assert not np.array_equal(field.data, initial_data)

def test_resonance_chamber():
    """Test resonance chamber functionality."""
    # Create field and chamber
    field = ToroidalField(dimensions=(16, 16, 16))
    chamber = ResonanceChamber(toroidal_field=field)
    
    # Store initial coherence
    initial_coherence = field.coherence
    
    # Apply resonance
    chamber.apply_resonance(strength=0.5)
    
    # Check coherence has changed
    assert field.coherence != initial_coherence

def test_compression_cycle():
    """Test compression cycle functionality."""
    # Create field and compression cycle
    field = ToroidalField(dimensions=(16, 16, 16))
    cycle = CompressionCycle(toroidal_field=field)
    
    # Store initial radii
    initial_major = field.major_radius
    initial_minor = field.minor_radius
    
    # Compress field
    cycle.compress(amount=0.2)
    
    # Check radii have changed
    assert field.minor_radius < initial_minor
    
    # Reset to original
    cycle.reset()
    
    # Check radii restored
    assert abs(field.minor_radius - initial_minor) < 1e-5

def test_counter_rotation():
    """Test counter-rotating field functionality."""
    # Create field and counter-rotation
    field = ToroidalField(dimensions=(16, 16, 16))
    cr_field = CounterRotatingField(toroidal_field=field)
    
    # Initialize component fields
    cr_field.split_field(separation=0.5)
    
    # Check component fields exist
    assert cr_field.clockwise_field is not None
    assert cr_field.counterclockwise_field is not None
    
    # Store initial stability
    initial_stability = cr_field.stability_factor
    
    # Apply rotation
    cr_field.rotate_components(time_factor=0.2)
    
    # Check stability has changed
    assert cr_field.stability_factor != initial_stability

def test_toroidal_field_engine():
    """Test the complete toroidal field engine."""
    # Create engine
    engine = ToroidalFieldEngine(dimensions=(16, 16, 16), frequency_name='unity')
    
    # Check metrics
    metrics = engine.update_metrics()
    assert 'coherence' in metrics
    assert 'balance' in metrics
    assert 'stability' in metrics
    
    # Store initial coherence
    initial_coherence = metrics['coherence']
    
    # Optimize coherence
    new_coherence = engine.optimize_field_coherence(target_coherence=0.8)
    
    # Check coherence has improved
    assert new_coherence >= initial_coherence
    
    # Run a toroidal cycle
    cycle_metrics = engine.run_toroidal_cycle(steps=5, time_factor=0.1)
    
    # Check cycle produced metrics
    assert len(cycle_metrics) > 0

if __name__ == "__main__":
    # Run tests manually
    test_toroidal_field_creation()
    test_toroidal_field_flow()
    test_resonance_chamber()
    test_compression_cycle()
    test_counter_rotation()
    test_toroidal_field_engine()
    
    print("All tests passed!")