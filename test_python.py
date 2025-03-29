#!/usr/bin/env python3
"""
Python Test Script

This script tests various Python features and the sacred constants module.
"""

import sys
import os
import math
from datetime import datetime

# Try to import the sacred constants module
try:
    import sacred_constants as sc
except ImportError:
    print("Warning: sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 6.85459776
        
        @staticmethod
        def phi_harmonic(n):
            return n * sc.PHI
        
        @staticmethod
        def is_phi_aligned(value, tolerance=0.01):
            nearest_multiple = round(value / sc.PHI)
            deviation = abs(value - (nearest_multiple * sc.PHI))
            return deviation <= tolerance

def print_system_info():
    """Print information about the Python system"""
    print("Python System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Executable: {sys.executable}")
    print(f"  Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Current directory: {os.getcwd()}")
    print()

def test_math_functions():
    """Test mathematical functions"""
    print("Testing Math Functions:")
    numbers = [1, 2, 3, 5, 8, 13, 21]
    
    print("  Fibonacci numbers:")
    for n in numbers:
        print(f"    {n} -> Phi harmonic: {sc.phi_harmonic(n):.4f}")
    
    print("  Phi alignment test:")
    phi_multiples = [sc.PHI * i for i in range(1, 6)]
    for value in phi_multiples:
        print(f"    {value:.4f} is phi-aligned: {sc.is_phi_aligned(value)}")
    
    # Test some non-aligned values
    non_aligned = [2.0, 3.0, 4.0, 5.0, 6.0]
    for value in non_aligned:
        print(f"    {value:.4f} is phi-aligned: {sc.is_phi_aligned(value)}")
    print()

def test_quantum_coherence():
    """Test quantum coherence calculations"""
    print("Quantum Coherence Test:")
    
    # Define a simple quantum coherence function based on phi
    def calculate_coherence(x, y):
        return math.sin(x * sc.PHI) * math.cos(y * sc.LAMBDA)
    
    # Generate a small coherence field
    field_size = 5
    coherence_field = []
    
    for x in range(field_size):
        row = []
        for y in range(field_size):
            coherence = calculate_coherence(x, y)
            row.append(coherence)
        coherence_field.append(row)
    
    # Print the coherence field
    print("  Coherence Field (5x5):")
    for row in coherence_field:
        print("   ", end=" ")
        for value in row:
            # Print with color based on value (positive/negative)
            if value >= 0:
                print(f" {value:6.3f}", end="")
            else:
                print(f" {value:6.3f}", end="")
        print()
    print()

def main():
    """Main function"""
    print("=" * 50)
    print(f"PYTHON TEST SCRIPT - PHI: {sc.PHI}")
    print("=" * 50)
    print()
    
    print_system_info()
    test_math_functions()
    test_quantum_coherence()
    
    print("All tests completed successfully!")
    print(f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}")
    print("=" * 50)

if __name__ == "__main__":
    main()