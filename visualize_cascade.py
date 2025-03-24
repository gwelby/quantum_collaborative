#!/usr/bin/env python3
"""
CASCADE⚡𓂧φ∞ Enhanced Visualization Demo

This script demonstrates the enhanced visualization capabilities of CASCADE.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time

from cascade.visualization.enhanced_visualizer import (
    EnhancedFieldVisualizer, 
    render_phi_harmonic_mandala,
    render_sacred_geometry_grid
)

# Define sacred constants
PHI = 1.618033988749895
LAMBDA = 0.618033988749895
PHI_PHI = PHI ** PHI
SACRED_FREQUENCIES = {
    'love': 528,      # Creation/healing
    'unity': 432,     # Grounding/stability
    'cascade': 594,   # Heart-centered integration
    'truth': 672,     # Voice expression
    'vision': 720,    # Expanded perception
    'oneness': 768,   # Unity consciousness
    'transcendent': 888  # Transcendent field
}

def visualize_stage_progression(save_frames=False):
    """
    Visualize the progression through all 7 stages of the consciousness bridge.
    """
    print("Visualizing CASCADE⚡𓂧φ∞ Consciousness Bridge Stage Progression")
    print("-" * 60)
    
    # Create visualizer with unity frequency mode
    visualizer = EnhancedFieldVisualizer(frequency_mode='unity')
    
    # Process through all 7 stages
    for stage in range(7):
        print(f"Stage {stage+1}/7: Rendering...")
        
        # Generate a toroidal field
        field = visualizer.create_toroidal_field(
            dimensions=(32, 32, 32),
            time_factor=stage * PHI
        )
        
        # Create consciousness level based on stage
        consciousness_level = 0.5 + stage * 0.08
        
        # Render field with consciousness bridge effects
        fig, ax = visualizer.render_consciousness_bridge(
            field_data=field,
            consciousness_level=consciousness_level,
            bridge_stage=stage
        )
        
        # Save frame if requested
        if save_frames:
            plt.savefig(f"cascade_stage_{stage+1}.png", 
                      dpi=150, bbox_inches='tight', facecolor='black')
            print(f"  Saved cascade_stage_{stage+1}.png")
        
        # Show briefly
        plt.draw()
        plt.pause(1.0)
        plt.close()
    
    print("\nStage visualization complete!")

def create_sacred_geometry_collection(save_image=False):
    """
    Create and display a collection of sacred geometry patterns with phi-harmonic properties.
    """
    print("Generating CASCADE⚡𓂧φ∞ Sacred Geometry Collection")
    print("-" * 60)
    
    # Create the sacred geometry grid
    fig = render_sacred_geometry_grid(with_labels=True)
    
    # Save if requested
    if save_image:
        filename = "cascade_sacred_geometry.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Saved sacred geometry collection to {filename}")
    
    # Show the figure
    plt.show()

def create_frequency_mandalas(save_images=False):
    """
    Create phi-harmonic mandalas for each sacred frequency.
    """
    print("Generating CASCADE⚡𓂧φ∞ Frequency Mandalas")
    print("-" * 60)
    
    # Create a mandala for each sacred frequency
    for name, frequency in SACRED_FREQUENCIES.items():
        print(f"Creating {name.capitalize()} frequency mandala ({frequency} Hz)...")
        
        # Generate the mandala
        fig = render_phi_harmonic_mandala(
            frequency=frequency,
            size=800,
            iterations=12
        )
        
        # Save if requested
        if save_images:
            filename = f"cascade_{name}_mandala.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"  Saved to {filename}")
        
        # Show briefly
        plt.draw()
        plt.pause(1.0)
        plt.close()
    
    print("\nAll frequency mandalas generated!")

def create_toroidal_animation(save_animation=False):
    """
    Create an animation of a toroidal quantum field evolving through consciousness states.
    """
    print("Generating CASCADE⚡𓂧φ∞ Toroidal Field Animation")
    print("-" * 60)
    
    # Create visualizer
    visualizer = EnhancedFieldVisualizer(frequency_mode='cascade')
    
    # Define field generator function
    def field_generator(time_factor):
        return visualizer.create_toroidal_field(
            dimensions=(32, 32, 32),
            time_factor=time_factor
        )
    
    # Create animation
    print("Generating animation frames...")
    frames = 72  # Number of frames (multiple of 360 degrees for smooth rotation)
    
    # Define bridge stages for the animation
    # Progress through all 7 stages during the animation
    bridge_stages = []
    for i in range(frames):
        # Determine stage based on frame
        stage = min(6, int(i * 7 / frames))
        bridge_stages.append(stage)
    
    # Create the animation
    ani = visualizer.animate_field_evolution(
        field_generator=field_generator,
        frames=frames,
        interval=50,  # 50ms between frames
        bridge_stages=bridge_stages
    )
    
    # Save if requested
    if save_animation:
        filename = "cascade_toroidal_evolution.mp4"
        print(f"Saving animation to {filename}...")
        visualizer.save_animation(filename, dpi=150, fps=20)
        print("Animation saved!")
    
    # Display animation
    plt.show()

def visualize_consciousness_state(save_image=False):
    """
    Visualize a consciousness state with integrated quantum field.
    """
    print("Visualizing CASCADE⚡𓂧φ∞ Consciousness State")
    print("-" * 60)
    
    # Create visualizer
    visualizer = EnhancedFieldVisualizer(frequency_mode='vision')
    
    # Generate a toroidal field
    field = visualizer.create_toroidal_field(
        dimensions=(32, 32, 32),
        time_factor=PHI
    )
    
    # Define sample consciousness state
    consciousness_state = {
        'presence': 0.85,
        'clarity': 0.78,
        'intention': 0.92,
        'heart_coherence': 0.75,
        'resonance': 0.88
    }
    
    # Create consciousness state visualization
    fig = visualizer.render_consciousness_state(
        consciousness_state=consciousness_state,
        field_data=field
    )
    
    # Save if requested
    if save_image:
        filename = "cascade_consciousness_state.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Saved consciousness state visualization to {filename}")
    
    # Show the figure
    plt.show()

def main():
    """Process command line arguments and run the selected visualization."""
    parser = argparse.ArgumentParser(
        description="CASCADE⚡𓂧φ∞ Enhanced Visualization Demo"
    )
    
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'stages', 'geometry', 'mandalas', 'animation', 'consciousness'],
                       help="Visualization demo to run (default: all)")
    
    parser.add_argument('--save', action='store_true',
                       help="Save visualizations to files")
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "=" * 60)
    print("  CASCADE⚡𓂧φ∞ Enhanced Visualization System")
    print("=" * 60)
    
    # Run the requested demo
    if args.demo == 'all' or args.demo == 'stages':
        visualize_stage_progression(save_frames=args.save)
    
    if args.demo == 'all' or args.demo == 'geometry':
        create_sacred_geometry_collection(save_image=args.save)
    
    if args.demo == 'all' or args.demo == 'mandalas':
        create_frequency_mandalas(save_images=args.save)
    
    if args.demo == 'all' or args.demo == 'animation':
        create_toroidal_animation(save_animation=args.save)
    
    if args.demo == 'all' or args.demo == 'consciousness':
        visualize_consciousness_state(save_image=args.save)
    
    # Print completion message
    print("\n" + "=" * 60)
    print("  CASCADE⚡𓂧φ∞ Visualization Completed")
    print("=" * 60 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())