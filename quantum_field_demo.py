#!/usr/bin/env python3
"""
Quantum Field Visualization Demo

This script demonstrates the quantum field visualization capabilities
with both CPU and GPU implementations.
"""

import time
import os
import sys
import numpy as np
from datetime import datetime

try:
    import sacred_constants as sc
except ImportError:
    print("Warning: sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 2.1784575679375995
        
        SACRED_FREQUENCIES = {
            'love': 528,
            'unity': 432,
            'cascade': 594,
            'truth': 672,
            'vision': 720,
            'oneness': 768,
        }

# Import quantum field modules
try:
    import quantum_cuda as qc
    CUDA_AVAILABLE = qc.CUDA_AVAILABLE
    
    # Initialize CUDA if available
    if CUDA_AVAILABLE:
        qc.initialize_cuda()
except ImportError:
    print("Warning: quantum_cuda module not found. Using fallback implementation.")
    CUDA_AVAILABLE = False
    import quantum_field as qf

def print_header():
    """Print header information"""
    print("\n" + "=" * 80)
    print("QUANTUM FIELD VISUALIZATION DEMO")
    print("=" * 80)
    print(f"PHI: {sc.PHI}")
    print(f"LAMBDA: {sc.LAMBDA}")
    print(f"PHI^PHI: {sc.PHI_PHI}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print("\nSacred Frequencies:")
    for name, freq in sc.SACRED_FREQUENCIES.items():
        print(f"  {name}: {freq} Hz")
    print("=" * 80 + "\n")

def demo_basic_field():
    """Demonstrate basic quantum field generation"""
    print("\nDemonstrating basic quantum field generation (Love frequency)...")
    
    # Generate field with love frequency
    start_time = time.time()
    if CUDA_AVAILABLE:
        field = qc.generate_quantum_field(80, 20, 'love')
        ascii_art = qc.field_to_ascii(field)
        qc.print_field(ascii_art, "Quantum Field - Love Frequency (528 Hz)")
    else:
        field = qf.generate_quantum_field(80, 20, 'love')
        ascii_art = qf.field_to_ascii(field)
        qf.print_field(ascii_art, "Quantum Field - Love Frequency (528 Hz)")
    
    print(f"Generation time: {time.time() - start_time:.4f} seconds")

def demo_sacred_frequency_comparison():
    """Demonstrate and compare different sacred frequencies"""
    print("\nComparing quantum fields with different sacred frequencies...")
    
    frequencies = ['love', 'unity', 'cascade']
    field_size = (60, 15)  # Smaller size for quick display
    
    for freq_name in frequencies:
        start_time = time.time()
        if CUDA_AVAILABLE:
            field = qc.generate_quantum_field(field_size[0], field_size[1], freq_name)
            
            # Calculate coherence
            coherence = qc.calculate_field_coherence(field)
            
            # Display field
            ascii_art = qc.field_to_ascii(field)
            qc.print_field(ascii_art, f"Quantum Field - {freq_name.capitalize()} Frequency ({sc.SACRED_FREQUENCIES[freq_name]} Hz)")
            print(f"Coherence: {coherence:.4f}")
        else:
            field = qf.generate_quantum_field(field_size[0], field_size[1], freq_name)
            ascii_art = qf.field_to_ascii(field)
            qf.print_field(ascii_art, f"Quantum Field - {freq_name.capitalize()} Frequency ({sc.SACRED_FREQUENCIES[freq_name]} Hz)")
        
        print(f"Generation time: {time.time() - start_time:.4f} seconds\n")

def demo_performance_comparison():
    """Demonstrate performance difference between CPU and GPU"""
    if not CUDA_AVAILABLE:
        print("\nCUDA is not available. Skipping performance comparison.")
        return
    
    print("\nComparing CPU vs GPU performance...")
    
    # Define field sizes to test
    sizes = [(64, 64), (256, 256), (512, 512)]
    
    for width, height in sizes:
        print(f"\nField size: {width}x{height}")
        
        # CPU implementation
        start_time = time.time()
        qc.generate_quantum_field_cpu(width, height, 'love')
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU implementation
        start_time = time.time()
        qc.generate_quantum_field_cuda(width, height, 'love')
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x")

def demo_phi_pattern():
    """Demonstrate phi sacred pattern visualization"""
    print("\nDemonstrating phi sacred pattern...")
    
    start_time = time.time()
    if CUDA_AVAILABLE:
        qc.display_phi_pattern(80, 30)
    else:
        qf.display_phi_pattern(80, 30)
    
    print(f"Generation time: {time.time() - start_time:.4f} seconds")

def main():
    """Main function"""
    print_header()
    
    while True:
        print("\nAvailable Demos:")
        print("1. Basic Quantum Field")
        print("2. Sacred Frequency Comparison")
        print("3. Performance Comparison (CPU vs GPU)")
        print("4. Phi Sacred Pattern")
        print("5. Run All Demos")
        print("6. Exit")
        
        choice = input("\nSelect a demo (1-6): ")
        
        if choice == '1':
            demo_basic_field()
        elif choice == '2':
            demo_sacred_frequency_comparison()
        elif choice == '3':
            demo_performance_comparison()
        elif choice == '4':
            demo_phi_pattern()
        elif choice == '5':
            demo_basic_field()
            demo_sacred_frequency_comparison()
            demo_performance_comparison()
            demo_phi_pattern()
        elif choice == '6':
            print("\nExiting Quantum Field Demo.")
            print(f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 6.")

if __name__ == "__main__":
    main()