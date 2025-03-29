#!/usr/bin/env python3
"""
Quantum Field 3D Visualization Demo

This script demonstrates the 3D quantum field generation capabilities with various
backends, visualizing the fields in different ways.
"""

import sys
import os
import numpy as np
import time
import argparse
from typing import Optional, Union, List, Tuple

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import quantum field modules
from quantum_field.backends import get_backend, get_available_backends
from quantum_field.visualization3d import (
    generate_3d_quantum_field,
    calculate_3d_field_coherence,
    visualize_3d_slices,
    visualize_3d_isosurface,
    visualize_3d_volume,
    animate_3d_field
)
from quantum_field.constants import SACRED_FREQUENCIES


def benchmark_backends(width: int, height: int, depth: int, frequency_name: str = 'love'):
    """Benchmark all available backends for 3D field generation"""
    backends = get_available_backends()
    results = []
    
    print(f"Benchmarking 3D quantum field generation ({width}x{height}x{depth})...")
    
    # Test CPU generation first (as baseline)
    cpu_backend = get_backend("cpu")
    start_time = time.time()
    cpu_field = cpu_backend.generate_3d_quantum_field(width, height, depth, frequency_name)
    cpu_time = time.time() - start_time
    
    coherence = cpu_backend.calculate_3d_field_coherence(cpu_field)
    print(f"CPU Backend: {cpu_time:.4f} seconds, coherence: {coherence:.4f}")
    results.append(("CPU", cpu_time))
    
    # Test other backends
    for name, backend in backends.items():
        if name == "cpu" or not backend.capabilities.get("3d_fields", False):
            continue
        
        try:
            start_time = time.time()
            field = backend.generate_3d_quantum_field(width, height, depth, frequency_name)
            backend_time = time.time() - start_time
            
            coherence = backend.calculate_3d_field_coherence(field)
            print(f"{name.upper()} Backend: {backend_time:.4f} seconds, coherence: {coherence:.4f}")
            results.append((name.upper(), backend_time))
            
            # Calculate speedup vs CPU
            speedup = cpu_time / backend_time
            print(f"  {speedup:.2f}x faster than CPU")
            
            # Verify fields match approximately
            if field.shape != cpu_field.shape:
                print(f"  Warning: Field shapes don't match! {field.shape} vs {cpu_field.shape}")
            else:
                # Check a few sample points
                samples = 5
                max_diff = 0
                for i in range(samples):
                    z = np.random.randint(0, depth)
                    y = np.random.randint(0, height)
                    x = np.random.randint(0, width)
                    diff = abs(field[z, y, x] - cpu_field[z, y, x])
                    max_diff = max(max_diff, diff)
                print(f"  Max sample difference: {max_diff:.6f}")
                
        except Exception as e:
            print(f"{name.upper()} Backend failed: {e}")
    
    return results


def demo_visualizations(width: int, height: int, depth: int, frequency_name: str = 'love',
                       backend_name: Optional[str] = None):
    """Demonstrate different 3D visualization techniques"""
    backend = get_backend(backend_name) if backend_name else get_backend()
    print(f"Using {backend.name.upper()} backend for 3D field generation")
    
    # Generate 3D field
    print(f"Generating 3D quantum field ({width}x{height}x{depth}) with frequency '{frequency_name}'...")
    start_time = time.time()
    field = backend.generate_3d_quantum_field(width, height, depth, frequency_name)
    gen_time = time.time() - start_time
    
    # Calculate coherence
    coherence = backend.calculate_3d_field_coherence(field)
    print(f"Generation time: {gen_time:.4f} seconds")
    print(f"Field coherence: {coherence:.4f}")
    print(f"Field shape: {field.shape}, min: {np.min(field):.4f}, max: {np.max(field):.4f}")
    
    # Visualize using slices
    print("\nGenerating slice visualization...")
    try:
        import matplotlib.pyplot as plt
        fig = visualize_3d_slices(field)
        plt.savefig("quantum_field_3d_slices.png")
        plt.close(fig)
        print("Saved slice visualization to 'quantum_field_3d_slices.png'")
    except Exception as e:
        print(f"Slice visualization failed: {e}")
    
    # Try isosurface visualization if libraries available
    print("\nGenerating isosurface visualization...")
    try:
        import plotly
        import pyvista
        
        fig = visualize_3d_isosurface(field)
        fig.write_html("quantum_field_3d_isosurface.html")
        print("Saved isosurface visualization to 'quantum_field_3d_isosurface.html'")
    except ImportError:
        print("Isosurface visualization requires plotly and pyvista")
    except Exception as e:
        print(f"Isosurface visualization failed: {e}")
    
    # Try volume rendering if plotly is available
    print("\nGenerating volume visualization...")
    try:
        import plotly
        
        fig = visualize_3d_volume(field)
        fig.write_html("quantum_field_3d_volume.html")
        print("Saved volume visualization to 'quantum_field_3d_volume.html'")
    except ImportError:
        print("Volume visualization requires plotly")
    except Exception as e:
        print(f"Volume visualization failed: {e}")
    
    return field


def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="Quantum Field 3D Visualization Demo")
    parser.add_argument("--size", type=int, default=64, help="Field size (N for NxNxN field)")
    parser.add_argument("--frequency", default="love", choices=list(SACRED_FREQUENCIES.keys()),
                       help="Sacred frequency to use")
    parser.add_argument("--backend", default=None, help="Backend to use (cuda, cpu, etc.)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark of all backends")
    parser.add_argument("--animate", action="store_true", help="Generate animation")
    
    args = parser.parse_args()
    
    # Print available backends
    backends = get_available_backends()
    print("Available quantum field backends:")
    for name, backend in backends.items():
        has_3d = "✓" if backend.capabilities.get("3d_fields", False) else "✗"
        print(f"  - {name.upper()}: 3D capabilities: {has_3d}")
    
    # Run benchmarks if requested
    if args.benchmark:
        benchmark_backends(args.size, args.size, args.size, args.frequency)
    
    # Run visualization demo
    field = demo_visualizations(args.size, args.size, args.size, args.frequency, args.backend)
    
    # Generate animation if requested
    if args.animate:
        print("\nGenerating animation...")
        try:
            small_size = min(32, args.size)  # Use smaller size for animation
            animate_3d_field(
                small_size, small_size, small_size, 
                args.frequency, 
                frames=15,
                output_path="quantum_field_3d_animation.html"
            )
            print("Saved animation to 'quantum_field_3d_animation.html'")
        except Exception as e:
            print(f"Animation failed: {e}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()