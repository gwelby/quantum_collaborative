#!/usr/bin/env python3
"""
Test script for the multi-accelerator backend architecture
"""

import os
import time
import numpy as np
from typing import Dict, Any, List, Optional

def main():
    """Main function to test the backend architecture"""
    try:
        # Import the backend system
        from quantum_field.backends import (
            detect_backends, 
            get_available_backends,
            get_backend
        )
        
        print("\n=== Quantum Field Accelerator Backend Test ===\n")
        
        # Re-detect backends
        print("Detecting available backends...")
        detect_backends()
        
        # Get all available backends
        backends = get_available_backends()
        print(f"\nFound {len(backends)} available backends:")
        
        for name, backend in backends.items():
            info = backend.get_info()
            print(f"\n- {name.upper()} Backend:")
            print(f"  Priority: {backend.priority}")
            print(f"  Initialized: {info['initialized']}")
            print("  Capabilities:")
            for cap, enabled in info['capabilities'].items():
                status = "✓" if enabled else "✗"
                print(f"    {status} {cap}")
        
        # Get the best available backend
        print("\n=== Testing Best Available Backend ===\n")
        backend = get_backend()
        print(f"Selected backend: {backend.name}")
        
        # Test field generation
        print("\nGenerating small field (128x128)...")
        start_time = time.time()
        field_small = backend.generate_quantum_field(128, 128, 'love', 0)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Field shape: {field_small.shape}")
        print(f"Field range: [{field_small.min():.4f}, {field_small.max():.4f}]")
        
        # Test larger field
        print("\nGenerating medium field (512x512)...")
        start_time = time.time()
        field_medium = backend.generate_quantum_field(512, 512, 'unity', 0.5)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Field shape: {field_medium.shape}")
        print(f"Field range: [{field_medium.min():.4f}, {field_medium.max():.4f}]")
        
        # Test coherence calculation
        print("\nCalculating field coherence...")
        start_time = time.time()
        coherence = backend.calculate_field_coherence(field_medium)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Coherence: {coherence:.4f}")
        
        # Test phi pattern generation
        print("\nGenerating phi pattern...")
        start_time = time.time()
        pattern = backend.generate_phi_pattern(256, 256)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Pattern shape: {pattern.shape}")
        print(f"Pattern range: [{pattern.min():.4f}, {pattern.max():.4f}]")
        
        # Test with specific backends
        if len(backends) > 1:
            print("\n=== Testing Specific Backends ===")
            for name, backend in backends.items():
                if name == get_backend().name:
                    continue  # Skip the default one we already tested
                
                print(f"\nTesting {name.upper()} backend...")
                backend = get_backend(name)
                
                # Generate small field
                try:
                    start_time = time.time()
                    field = backend.generate_quantum_field(128, 128, 'love', 0)
                    end_time = time.time()
                    print(f"  Field generation: {end_time - start_time:.4f} seconds")
                    print(f"  Field shape: {field.shape}")
                except Exception as e:
                    print(f"  Error in field generation: {e}")
        
        print("\n=== Test Completed Successfully ===")
        
    except ImportError:
        print("Error: Could not import quantum_field.backends")
        print("Make sure the backend architecture is implemented and available")
    except Exception as e:
        print(f"Error during testing: {e}")


# Alternative simplified test for direct core API testing
def test_core_api():
    """Test the core API (which will use the backend architecture internally)"""
    try:
        from quantum_field.core import (
            generate_quantum_field,
            calculate_field_coherence,
            generate_phi_pattern
        )
        
        print("\n=== Testing Core API with Backend Architecture ===\n")
        
        # Test field generation
        print("Generating quantum field...")
        start_time = time.time()
        field = generate_quantum_field(256, 256, 'love', 0)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Field shape: {field.shape}")
        print(f"Field range: [{field.min():.4f}, {field.max():.4f}]")
        
        # Test coherence calculation
        print("\nCalculating field coherence...")
        start_time = time.time()
        coherence = calculate_field_coherence(field)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Coherence: {coherence:.4f}")
        
        # Test phi pattern generation
        print("\nGenerating phi pattern...")
        start_time = time.time()
        pattern = generate_phi_pattern(256, 256)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Pattern shape: {pattern.shape}")
        print(f"Pattern range: [{pattern.min():.4f}, {pattern.max():.4f}]")
        
        print("\n=== Core API Test Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during core API testing: {e}")


if __name__ == "__main__":
    # Run both tests
    main()
    print("\n" + "=" * 50 + "\n")
    test_core_api()