#!/usr/bin/env python3
"""
3D Quantum Field Visualization Demo

This script demonstrates the 3D visualization capabilities of the quantum field package.
It shows how to generate and visualize 3D quantum fields using various techniques.

Usage:
    python 3d_visualization_demo.py [--mode MODE] [--frequency FREQ] [--size SIZE] [--output FILE]

Options:
    --mode MODE         Visualization mode: slices, volume, isosurface, or all [default: all]
    --frequency FREQ    Sacred frequency to use: love, unity, etc. [default: love]
    --size SIZE         Field size in voxels [default: 64]
    --output FILE       Save visualization to file(s) instead of displaying
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import quantum_field
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import quantum field modules
try:
    from quantum_field.visualization3d import (
        generate_3d_quantum_field,
        calculate_3d_field_coherence,
        visualize_3d_slices,
        visualize_3d_volume,
        visualize_3d_isosurface,
        animate_3d_field
    )
    from quantum_field.constants import SACRED_FREQUENCIES
except ImportError:
    print("Error importing quantum_field package")
    sys.exit(1)

# Check for required packages
try:
    import plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not installed. Some visualizations won't be available.")
    print("Install with: pip install plotly")

try:
    import pyvista
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not installed. Isosurface extraction will use fallback method.")
    print("Install with: pip install pyvista")


def main():
    """Main function to run the demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="3D Quantum Field Visualization Demo")
    parser.add_argument("--mode", choices=["slices", "volume", "isosurface", "all"], 
                      default="all", help="Visualization mode")
    parser.add_argument("--frequency", default="love", 
                      help="Sacred frequency to use")
    parser.add_argument("--size", type=int, default=64, 
                      help="Field size in voxels")
    parser.add_argument("--output", help="Save visualization to file(s) instead of displaying")
    parser.add_argument("--animate", action="store_true", 
                      help="Create animation instead of static visualization")
    args = parser.parse_args()
    
    # Check frequency is valid
    if args.frequency not in SACRED_FREQUENCIES:
        print(f"Unknown frequency: {args.frequency}")
        print(f"Available frequencies: {', '.join(SACRED_FREQUENCIES.keys())}")
        sys.exit(1)
    
    # Check visualization mode requirements
    if (args.mode in ["volume", "isosurface", "all"] or args.animate) and not HAS_PLOTLY:
        print(f"Plotly is required for {args.mode} visualization.")
        print("Install with: pip install plotly")
        sys.exit(1)
    
    # Print demo information
    print(f"3D Quantum Field Visualization Demo")
    print(f"-----------------------------------")
    print(f"Frequency: {args.frequency} ({SACRED_FREQUENCIES[args.frequency]} Hz)")
    print(f"Field size: {args.size}x{args.size}x{args.size} voxels")
    print(f"Visualization mode: {args.mode}")
    print(f"Animation: {'Yes' if args.animate else 'No'}")
    print()
    
    if args.animate:
        # Create animation
        print(f"Generating animation with {args.size}x{args.size}x{args.size} field...")
        animation_frames = 30
        output_path = args.output if args.output else None
        
        # Create animation
        animation = animate_3d_field(
            args.size, args.size, args.size,
            args.frequency,
            frames=animation_frames,
            mode=args.mode if args.mode != "all" else "isosurface",
            output_path=output_path
        )
        
        if output_path:
            print(f"Animation saved to {output_path}")
        else:
            # Display animation
            import plotly.io as pio
            pio.show(animation)
        
        return
    
    # Generate 3D quantum field
    print(f"Generating {args.size}x{args.size}x{args.size} field...")
    field = generate_3d_quantum_field(
        args.size, args.size, args.size,
        frequency_name=args.frequency,
        time_factor=0.0
    )
    
    # Calculate field coherence
    coherence = calculate_3d_field_coherence(field)
    print(f"Field coherence: {coherence:.4f}")
    
    # Visualize field based on selected mode
    if args.mode == "slices" or args.mode == "all":
        print("Creating slice visualization...")
        slice_fig = visualize_3d_slices(
            field,
            title=f"3D Quantum Field Slices ({args.frequency.capitalize()} Frequency)"
        )
        
        if args.output and args.mode != "all":
            # Save to file
            output_file = args.output
            slice_fig.savefig(output_file)
            print(f"Slice visualization saved to {output_file}")
        elif args.mode != "all":
            # Display
            plt.show()
    
    if (args.mode == "volume" or args.mode == "all") and HAS_PLOTLY:
        print("Creating volume visualization...")
        volume_fig = visualize_3d_volume(
            field,
            title=f"3D Quantum Field Volume ({args.frequency.capitalize()} Frequency)"
        )
        
        if args.output and args.mode != "all":
            # Save to file
            output_file = args.output
            volume_fig.write_html(output_file)
            print(f"Volume visualization saved to {output_file}")
        elif args.mode != "all":
            # Display
            import plotly.io as pio
            pio.show(volume_fig)
    
    if (args.mode == "isosurface" or args.mode == "all") and HAS_PLOTLY:
        print("Creating isosurface visualization...")
        iso_fig = visualize_3d_isosurface(
            field,
            title=f"3D Quantum Field Isosurfaces ({args.frequency.capitalize()} Frequency)"
        )
        
        if args.output and args.mode != "all":
            # Save to file
            output_file = args.output
            iso_fig.write_html(output_file)
            print(f"Isosurface visualization saved to {output_file}")
        elif args.mode != "all":
            # Display
            import plotly.io as pio
            pio.show(iso_fig)
    
    # If all modes and output specified, save each to a different file
    if args.mode == "all" and args.output:
        base_name, ext = os.path.splitext(args.output)
        
        # Save slice visualization
        slice_output = f"{base_name}_slices.png"
        slice_fig.savefig(slice_output)
        print(f"Slice visualization saved to {slice_output}")
        
        if HAS_PLOTLY:
            # Save volume visualization
            volume_output = f"{base_name}_volume.html"
            volume_fig.write_html(volume_output)
            print(f"Volume visualization saved to {volume_output}")
            
            # Save isosurface visualization
            iso_output = f"{base_name}_isosurface.html"
            iso_fig.write_html(iso_output)
            print(f"Isosurface visualization saved to {iso_output}")
    
    # If all modes and no output, display each sequentially
    if args.mode == "all" and not args.output:
        # Show slice visualization
        plt.show()
        
        if HAS_PLOTLY:
            # Show volume visualization
            import plotly.io as pio
            pio.show(volume_fig)
            
            # Show isosurface visualization
            pio.show(iso_fig)
    
    print("Visualization complete.")


if __name__ == "__main__":
    main()