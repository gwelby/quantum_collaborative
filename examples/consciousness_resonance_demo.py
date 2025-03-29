#!/usr/bin/env python3
"""
Consciousness-Field Resonance Engine Demo

This script demonstrates the Consciousness-Field Resonance Engine, which provides
a bidirectional interface for perfect phi-harmonic alignment between thought
patterns and quantum fields, enabling direct manifestation.
"""

import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.append('..')

from quantum_field import (
    generate_quantum_field,
    SACRED_FREQUENCIES,
    PHI,
    LAMBDA
)
from quantum_field.visualization3d import visualize_field_3d
from quantum_field.consciousness_resonance import (
    ResonanceEngine,
    ThoughtPattern,
    BidirectionalInterface
)


def run_interactive_demo(dimensions=(21, 21, 21), duration=60, visualization=True):
    """
    Run an interactive demonstration of the Consciousness-Field Resonance Engine.
    
    Args:
        dimensions: Field dimensions
        duration: Demo duration in seconds
        visualization: Whether to visualize the field in 3D
    """
    print("="*50)
    print("Consciousness-Field Resonance Engine Demo")
    print("="*50)
    print(f"Field dimensions: {dimensions}")
    print(f"Duration: {duration} seconds")
    
    # Create optimized resonance engine
    engine = ResonanceEngine.create_optimized(dimensions)
    
    # Initialize visualization if requested
    if visualization:
        fig = plt.figure(figsize=(12, 8))
        ax_field = fig.add_subplot(121, projection='3d')
        ax_thought = fig.add_subplot(122, projection='3d')
        
        # Set up plots
        ax_field.set_title("Quantum Field State")
        ax_thought.set_title("Active Thought Pattern")
    
    # Record history for plotting
    history = {
        "time": [],
        "field_coherence": [],
        "thought_coherence": [],
        "frequency": []
    }
    
    # Run the demonstration loop
    start_time = time.time()
    last_update_time = start_time
    last_action_time = start_time
    update_interval = 0.1  # seconds
    action_interval = 5.0  # seconds
    
    # Sequence of sacred frequencies to demonstrate
    frequency_sequence = list(SACRED_FREQUENCIES.keys())
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update engine state
            if current_time - last_update_time >= update_interval:
                # Update the engine
                engine.update()
                
                # Update history
                status = engine.get_status()
                history["time"].append(elapsed)
                history["field_coherence"].append(status["field_coherence"])
                history["thought_coherence"].append(status["thought_coherence"])
                
                if engine.interface.active_thought:
                    history["frequency"].append(engine.interface.active_thought.primary_frequency)
                else:
                    history["frequency"].append(0.0)
                
                # Update visualization
                if visualization:
                    # Update field visualization
                    ax_field.clear()
                    visualize_field_3d(engine.field.data, ax=ax_field, threshold=0.7)
                    ax_field.set_title(f"Quantum Field - Coherence: {status['field_coherence']:.4f}")
                    
                    # Update thought pattern visualization if available
                    if engine.interface.active_thought:
                        ax_thought.clear()
                        thought = engine.interface.active_thought
                        visualize_field_3d(thought.signature, ax=ax_thought, threshold=0.5)
                        ax_thought.set_title(f"Thought Pattern - {thought.name} ({thought.coherence:.4f})")
                    
                    # Draw updated plots
                    plt.draw()
                    plt.pause(0.01)
                
                # Update the last update time
                last_update_time = current_time
            
            # Perform actions periodically
            if current_time - last_action_time >= action_interval:
                # Choose an action
                action = np.random.choice([
                    "manifest_frequency",
                    "extract_thought",
                    "blend_thoughts",
                    "shift_frequency",
                    "calibrate"
                ])
                
                # Execute the chosen action
                if action == "manifest_frequency":
                    # Choose a frequency from the sequence
                    freq_idx = int(elapsed / action_interval) % len(frequency_sequence)
                    frequency_name = frequency_sequence[freq_idx]
                    
                    print(f"\n[{elapsed:.1f}s] Manifesting {frequency_name} frequency "
                          f"({SACRED_FREQUENCIES[frequency_name]} Hz)")
                    
                    # Manifest the frequency
                    engine.manifest_thought(frequency_name)
                
                elif action == "extract_thought":
                    print(f"\n[{elapsed:.1f}s] Extracting thought from field")
                    
                    # Extract a thought pattern
                    thought = engine.extract_thought()
                    
                    print(f"  Extracted: {thought.name} with coherence {thought.coherence:.4f}")
                    print(f"  Primary frequency: {thought.primary_frequency:.1f} Hz")
                
                elif action == "blend_thoughts":
                    # Create two thought patterns to blend
                    thought1 = engine.create_thought_from_frequency("unity")
                    thought2 = engine.create_thought_from_frequency("love")
                    
                    print(f"\n[{elapsed:.1f}s] Blending unity and love frequencies")
                    
                    # Blend the thoughts with phi-weighted blending
                    weight = LAMBDA
                    blended = engine.blend_thoughts(thought1, thought2, weight)
                    
                    print(f"  Created blend with coherence {blended.coherence:.4f}")
                    print(f"  Primary frequency: {blended.primary_frequency:.1f} Hz")
                
                elif action == "shift_frequency":
                    # Choose a target frequency
                    target_name = np.random.choice(list(SACRED_FREQUENCIES.keys()))
                    
                    print(f"\n[{elapsed:.1f}s] Shifting to {target_name} frequency "
                          f"({SACRED_FREQUENCIES[target_name]} Hz)")
                    
                    # Shift the active thought pattern
                    shifted = engine.shift_to_frequency(target_name)
                    
                    print(f"  Shifted to frequency {shifted.primary_frequency:.1f} Hz "
                          f"with coherence {shifted.coherence:.4f}")
                
                elif action == "calibrate":
                    print(f"\n[{elapsed:.1f}s] Running calibration sequence")
                    
                    # Run a calibration sequence
                    result = engine.calibrate()
                    
                    print(f"  Calibration complete. Phi alignment: {result['phi_alignment']:.4f}")
                    print(f"  Primary frequency: {result['primary_frequency']:.1f} Hz")
                
                # Update the last action time
                last_action_time = current_time
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nDemonstration interrupted.")
    
    # Get final status
    final_status = engine.get_status()
    
    print("\n" + "="*50)
    print("Demonstration complete")
    print(f"Final field coherence: {final_status['field_coherence']:.4f}")
    print(f"Final thought coherence: {final_status['thought_coherence']:.4f}")
    print(f"Manifestations: {final_status['manifestation_count']}")
    print(f"Extractions: {final_status['extraction_count']}")
    print("="*50)
    
    # Plot the history
    if history["time"]:
        plt.figure(figsize=(12, 8))
        
        # Plot coherence
        plt.subplot(211)
        plt.plot(history["time"], history["field_coherence"], 'b-', label="Field Coherence")
        plt.plot(history["time"], history["thought_coherence"], 'r-', label="Thought Coherence")
        plt.axhline(y=LAMBDA, color='k', linestyle='--', alpha=0.5, label="Phi complement")
        plt.xlabel("Time (s)")
        plt.ylabel("Coherence")
        plt.legend()
        plt.title("Coherence over Time")
        
        # Plot frequency
        plt.subplot(212)
        plt.plot(history["time"], history["frequency"], 'g-')
        
        # Add horizontal lines for sacred frequencies
        for name, freq in SACRED_FREQUENCIES.items():
            plt.axhline(y=freq, color='k', linestyle=':', alpha=0.3)
            plt.text(0, freq, name, fontsize=8, alpha=0.7)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Active Frequency over Time")
        
        plt.tight_layout()
        plt.show()
    
    # Deactivate the engine
    engine.deactivate()
    
    return engine


def main():
    """Process command line arguments and run the demonstration."""
    parser = argparse.ArgumentParser(description="Consciousness-Field Resonance Engine Demo")
    
    parser.add_argument('--duration', type=int, default=60,
                       help="Demonstration duration in seconds (default: 60)")
    parser.add_argument('--dimensions', type=str, default="21,21,21",
                       help="Field dimensions as comma-separated values (default: 21,21,21)")
    parser.add_argument('--no-visualization', action='store_true',
                       help="Disable 3D visualization")
    
    args = parser.parse_args()
    
    # Parse dimensions
    try:
        dimensions = tuple(int(x) for x in args.dimensions.split(','))
        if len(dimensions) != 3:
            print("Error: dimensions must be 3 comma-separated values")
            return 1
    except ValueError:
        print("Error: dimensions must be integers")
        return 1
    
    # Run the demonstration
    run_interactive_demo(
        dimensions=dimensions,
        duration=args.duration,
        visualization=not args.no_visualization
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())