#!/usr/bin/env python3
"""
Test script for CascadeOS - Verifies the system is working properly
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path if needed
CASCADE_PATH = Path(__file__).parent.resolve()
if CASCADE_PATH not in sys.path:
    sys.path.append(str(CASCADE_PATH))

# Import CascadeOS components
try:
    from CascadeOS import (
        QuantumField,
        ConsciousnessState,
        ConsciousnessFieldInterface,
        create_quantum_field,
        field_to_ascii,
        print_field,
        CascadeSystem,
        PHI, LAMBDA, PHI_PHI,
        SACRED_FREQUENCIES
    )
    print("✓ Successfully imported CascadeOS")
except ImportError as e:
    print(f"✗ Error importing CascadeOS: {e}")
    print("  Make sure CascadeOS.py is in the current directory")
    sys.exit(1)

def test_quantum_field():
    """Test basic quantum field creation and operations"""
    print("\n== Testing Quantum Field ==")
    
    # Test field creation
    try:
        print("  Creating quantum field...", end="")
        field = create_quantum_field((21, 21), frequency_name='love')
        print(f" ✓ (shape: {field.shape}, coherence: {field.coherence:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test phi modulation
    try:
        print("  Testing phi modulation...", end="")
        initial_coherence = field.coherence
        field.apply_phi_modulation(intensity=0.5)
        print(f" ✓ (coherence changed: {initial_coherence:.4f} → {field.coherence:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test visualization
    try:
        print("  Testing field visualization...", end="")
        ascii_art = field_to_ascii(field.data)
        # Just check if we got something
        if ascii_art and len(ascii_art) > 10:
            print(" ✓")
        else:
            print(f" ✗ (empty or too small visualization)")
            return False
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    return True

def test_consciousness_interface():
    """Test consciousness interface functionality"""
    print("\n== Testing Consciousness Interface ==")
    
    # Create field and interface
    try:
        print("  Creating consciousness interface...", end="")
        field = create_quantum_field((21, 21), frequency_name='unity')
        interface = ConsciousnessFieldInterface(field)
        print(" ✓")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test biofeedback update
    try:
        print("  Testing biofeedback update...", end="")
        initial_coherence = interface.get_field_coherence()
        interface.update_consciousness_state(
            heart_rate=60,
            breath_rate=6.18,
            skin_conductance=3,
            eeg_alpha=12,
            eeg_theta=7.4
        )
        updated_coherence = interface.get_field_coherence()
        print(f" ✓ (coherence changed: {initial_coherence:.4f} → {updated_coherence:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test intention application
    try:
        print("  Testing intention application...", end="")
        interface.state.intention = 0.9
        interface._apply_intention()
        intention_coherence = interface.get_field_coherence()
        change = intention_coherence - updated_coherence
        print(f" ✓ (coherence change: {change:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test emotional influence
    try:
        print("  Testing emotional influence...", end="")
        for emotion in interface.state.emotional_states:
            interface.state.emotional_states[emotion] = 0.1
        interface.state.emotional_states["love"] = 0.9
        interface._apply_emotional_influence()
        emotion_coherence = interface.get_field_coherence()
        change = emotion_coherence - intention_coherence
        print(f" ✓ (coherence change: {change:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test phi-resonance profile
    try:
        print("  Testing phi-resonance profile...", end="")
        # Add some feedback history
        for i in range(5):
            interface.update_consciousness_state(
                heart_rate=65 - i * 2,
                breath_rate=10 - i,
                skin_conductance=5 - i * 0.5,
                eeg_alpha=8 + i,
                eeg_theta=4 + i * 0.5
            )
        
        profile = interface.create_phi_resonance_profile(interface.feedback_history)
        if not profile:
            print(" ✗ (empty profile)")
            return False
            
        interface.apply_phi_resonance_profile()
        profile_coherence = interface.get_field_coherence()
        print(f" ✓ (final coherence: {profile_coherence:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    return True

def test_cascade_system():
    """Test the full CascadeSystem"""
    print("\n== Testing CascadeSystem ==")
    
    # Create and initialize system
    try:
        print("  Creating CascadeSystem...", end="")
        system = CascadeSystem()
        system.initialize()
        print(" ✓")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Activate system
    try:
        print("  Activating system...", end="")
        system.activate()
        print(f" ✓ (system coherence: {system.system_coherence:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Update system
    try:
        print("  Updating system with consciousness data...", end="")
        status = system.update({
            "heart_rate": 60,
            "breath_rate": 6.18,
            "skin_conductance": 3,
            "eeg_alpha": 12,
            "eeg_theta": 7.4
        })
        print(f" ✓ (system coherence: {status['system_coherence']:.4f})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Test visualization
    try:
        print("  Testing system visualization...", end="")
        system._visualize_fields()
        print(" ✓")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    # Deactivate system
    try:
        print("  Deactivating system...", end="")
        result = system.deactivate()
        print(f" ✓ (active: {not result})")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all test functions"""
    print("\n===== CascadeOS Test Suite =====")
    
    # Check sacred constants
    print("\n== Checking Sacred Constants ==")
    print(f"  PHI = {PHI}")
    print(f"  LAMBDA = {LAMBDA}")
    print(f"  PHI_PHI = {PHI_PHI}")
    
    # Check sacred frequencies
    print("\n== Checking Sacred Frequencies ==")
    for name, freq in SACRED_FREQUENCIES.items():
        print(f"  {name}: {freq} Hz")
    
    # Run tests
    results = {}
    
    results["quantum_field"] = test_quantum_field()
    results["consciousness_interface"] = test_consciousness_interface()
    results["cascade_system"] = test_cascade_system()
    
    # Print summary
    print("\n===== Test Results Summary =====")
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    # Return overall success
    return all(results.values())

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✓✓ All tests passed! CascadeOS is working properly.")
        sys.exit(0)
    else:
        print("\n✗✗ Some tests failed. See above for details.")
        sys.exit(1)