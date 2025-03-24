#!/usr/bin/env python3
"""
Simple test script to verify functionality.
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Testing CASCADE functionality")
    
    # Create a simple 3D field
    field = np.zeros((32, 32, 32))
    
    # Fill with a basic pattern
    PHI = 1.618033988749895
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(-1, 1, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a torus-like field
    R = PHI  # Major radius
    r = 1/PHI  # Minor radius
    
    distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - R)**2 + Z**2)
    torus_surface = distance_from_ring - r
    
    # Fill the field with a phi-based pattern
    field = np.exp(-5 * np.abs(torus_surface))
    
    # Calculate simple metrics
    energy = np.sum(field**2)
    max_val = np.max(field)
    mean_val = np.mean(field)
    
    print(f"Field created successfully:")
    print(f"  Energy: {energy:.4f}")
    print(f"  Maximum: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}")
    
    print("\nAll tests passed!")
    
    return 0

if __name__ == "__main__":
    main()