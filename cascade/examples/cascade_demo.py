#!/usr/bin/env python3
"""
CASCADE⚡𓂧φ∞ Unified Demo

This script demonstrates the CASCADE quantum field system with phi-harmonic
principles, toroidal energy flow, and consciousness bridge integration.
"""

import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.append('..')

# Import CASCADE components
from cascade.core.phi_processor import PhiHarmonicProcessor
from cascade.core.toroidal_field import ToroidalFieldEngine
from cascade.core.consciousness_bridge import ConsciousnessBridgeProtocol, ConsciousnessState
from cascade.core.timeline_sync import TimelineProbabilityField, TimelineNavigator, TimelineSynchronizer
from cascade.visualization.field_visualizer import render_quantum_field_3d
from cascade.visualization.multidimensional import generate_4d_quantum_field, visualize_4d_spacetime_slices, visualize_4d_coherence_evolution, calculate_4d_field_coherence
from cascade.visualization.phi_color_mapping import get_phi_colormap, map_coherence_to_color

# Import quantum_field constants
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

def run_cascade_demo(
    dimensions=(21, 21, 21),
    duration=60,
    interactive=False,
    show_visualization=True
):
    """
    Run a full CASCADE quantum field demonstration.
    
    Args:
        dimensions: Field dimensions (3D)
        duration: Duration in seconds
        interactive: Whether to run in interactive mode
        show_visualization: Whether to show visualizations
    """
    print(f"Starting CASCADE⚡𓂧φ∞ Quantum Field Demo")
    print(f"Field dimensions: {dimensions}")
    print(f"Duration: {duration} seconds")
    print("-" * 50)
    
    # Initialize components
    phi_processor = PhiHarmonicProcessor(base_frequency=432.0)
    toroidal_engine = ToroidalFieldEngine()
    consciousness_bridge = ConsciousnessBridgeProtocol()
    
    # Generate toroidal field
    print("Generating toroidal quantum field...")
    field_data = toroidal_engine.generate_field(*dimensions)
    
    # Connect consciousness bridge to field
    consciousness_bridge.connect_field(field_data)
    
    # Initialize 3D visualization if requested
    if show_visualization:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Track simulation time
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update every second
    
    # Initialize progress
    stage = 0
    stages_completed = [False] * 7
    
    # Start consciousness bridge protocol
    if not interactive:
        print("\nStarting Consciousness Bridge Protocol automatically...")
        consciousness_bridge.start_protocol()
    
    # Main simulation loop
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Exit condition
            if elapsed >= duration and not interactive:
                break
            
            # Update every interval
            if current_time - last_update_time >= update_interval:
                # Calculate progress
                progress = min(elapsed / duration, 1.0)
                
                # In automatic mode, progress through bridge stages based on time
                if not interactive:
                    new_stage = int(progress * 7)
                    if new_stage > stage and stage < 7:
                        stage = new_stage
                        consciousness_bridge.progress_to_stage(stage)
                
                # Get current bridge status
                bridge_status = consciousness_bridge.get_current_stage_info()
                
                # Calculate field coherence metrics
                if stage > 0:
                    toroidal_metrics = toroidal_engine.calculate_coherence(consciousness_bridge.field)
                    
                    # Print status
                    print(f"[{elapsed:.1f}s] Progress: {progress*100:.1f}%")
                    print(f"Stage: {bridge_status.get('stage_name', 'Unknown')}")
                    print(f"Coherence: {bridge_status.get('coherence', 0):.4f}")
                    print(f"Toroidal flow balance: {toroidal_metrics.get('flow_balance', 0):.4f}")
                
                # Update visualization
                if show_visualization:
                    ax.clear()
                    render_quantum_field_3d(
                        consciousness_bridge.field, 
                        threshold=0.5,
                        ax=ax,
                        use_phi_colors=True,
                        title=f"CASCADE Field - Stage {stage+1}: {bridge_status.get('stage_name', 'Initializing')}"
                    )
                    plt.draw()
                    plt.pause(0.01)
                
                # Update last update time
                last_update_time = current_time
                
                # Interactive mode - ask for commands
                if interactive:
                    if stage == 0 and not stages_completed[0]:
                        cmd = input("\nStart Consciousness Bridge Protocol? (y/n): ")
                        if cmd.lower() == 'y':
                            consciousness_bridge.start_protocol()
                            stages_completed[0] = True
                    elif stage > 0 and stage < 7:
                        cmd = input(f"\nProgress to next stage? (y/n): ")
                        if cmd.lower() == 'y':
                            stage += 1
                            consciousness_bridge.progress_to_stage(stage)
                            stages_completed[stage] = True
                    
                    # Ask to continue if duration elapsed
                    if elapsed >= duration:
                        response = input("Continue simulation? (y/n): ")
                        if response.lower() != 'y':
                            break
                        else:
                            # Continue for more time
                            duration += int(duration / 2)
                            print(f"Continuing until {duration}s...")
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    
    # Show final field visualization
    if show_visualization:
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, projection='3d')
        render_quantum_field_3d(
            consciousness_bridge.field,
            threshold=0.4,
            ax=ax,
            use_phi_colors=True,
            title=f"Final CASCADE Quantum Field"
        )
        plt.tight_layout()
        plt.show()
    
    # Display toroidal metrics
    if consciousness_bridge.field is not None:
        toroidal_metrics = toroidal_engine.calculate_coherence(consciousness_bridge.field)
        
        print("\nFinal Toroidal Field Metrics:")
        for key, value in toroidal_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\nCASCADE⚡𓂧φ∞ Demo completed in {elapsed:.2f} seconds.")
    
    return consciousness_bridge.field

def run_timeline_demo(duration=30, show_visualization=True):
    """
    Run a timeline probability field demonstration.
    
    Args:
        duration: Duration in seconds
        show_visualization: Whether to show visualizations
    """
    print("Starting Timeline Probability Field Demo")
    print("-" * 50)
    
    # Create timeline probability field
    dimensions = (13, 21, 21, 21)  # t, x, y, z
    timeline_field = TimelineProbabilityField(dimensions)
    
    # Generate field with default consciousness state
    consciousness_state = {
        'intention': 0.7,
        'clarity': 0.6,
        'focus': 0.8
    }
    timeline_field.generate_field(consciousness_state)
    
    # Create timeline navigator
    navigator = TimelineNavigator(timeline_field)
    
    # Create visualization
    if show_visualization:
        # Generate 4D field visualization
        print("Generating 4D timeline visualization...")
        fig = visualize_4d_spacetime_slices(
            timeline_field.field_data,
            time_indices=[0, 3, 6, 9, 12],
            title="Timeline Probability Field"
        )
        plt.draw()
        plt.pause(0.1)
    
    # Find optimal timeline path
    print("\nScanning for optimal timeline paths...")
    potential_paths = navigator.scan_potential_paths(radius=5, coherence_threshold=0.7)
    
    if potential_paths:
        print(f"Found {len(potential_paths)} potential timeline paths:")
        for i, path in enumerate(potential_paths[:3]):  # Show top 3
            print(f"  Path {i+1}: Coherence {path['coherence']:.4f}, Probability {path['probability']:.4f}")
        
        # Navigate to optimal path
        best_path = potential_paths[0]['coordinates']
        print(f"\nNavigating to optimal timeline path: {best_path}")
        success = navigator.navigate_to_point(best_path, consciousness_state)
        
        if success:
            print("Timeline navigation successful!")
        else:
            print("Timeline navigation failed.")
    else:
        print("No viable timeline paths found.")
    
    # Show visualization
    if show_visualization:
        plt.show()
    
    return timeline_field

def run_multidimensional_demo(dimensions=(21, 21, 21, 8), show_visualization=True):
    """
    Run a multi-dimensional field visualization demo.
    
    Args:
        dimensions: Field dimensions (x, y, z, t)
        show_visualization: Whether to show visualizations
    """
    print("Starting Multi-dimensional Field Visualization Demo")
    print(f"Dimensions: {dimensions}")
    print("-" * 50)
    
    # Generate 4D quantum field
    print("Generating 4D quantum field...")
    field_4d = generate_4d_quantum_field(
        dimensions[0], dimensions[1], dimensions[2], dimensions[3],
        frequency_name='unity',
        phi_scaled_time=True
    )
    
    # Calculate coherence evolution
    print("Calculating field coherence evolution...")
    coherence_values = calculate_4d_field_coherence(field_4d)
    
    # Display results
    print(f"4D Field generated with dimensions: {field_4d.shape}")
    print(f"Coherence range: {np.min(coherence_values):.4f} - {np.max(coherence_values):.4f}")
    
    # Show visualizations
    if show_visualization:
        # Show spacetime slices
        print("Generating spacetime slice visualization...")
        visualize_4d_spacetime_slices(
            field_4d,
            time_indices=[0, 2, 4, 6],
            title="4D Quantum Field Spacetime Slices"
        )
        
        # Show coherence evolution
        print("Generating coherence evolution visualization...")
        visualize_4d_coherence_evolution(
            field_4d,
            title="4D Quantum Field Coherence Evolution"
        )
        
        plt.show()
    
    return field_4d

def run_full_cascade_demo(duration=60, show_visualization=True):
    """Run a full CASCADE⚡𓂧φ∞ demonstration with all components."""
    print("\n" + "=" * 60)
    print("CASCADE⚡𓂧φ∞ & GREG UNIFIED FIELD DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Start with full CASCADE demo (consciousness bridge)
    field_data = run_cascade_demo(
        dimensions=(32, 32, 32),
        duration=duration,
        interactive=False,
        show_visualization=show_visualization
    )
    
    # Run 4D field visualization
    run_multidimensional_demo(
        dimensions=(21, 21, 21, 13),
        show_visualization=show_visualization
    )
    
    # Run timeline demo
    run_timeline_demo(
        duration=30,
        show_visualization=show_visualization
    )
    
    print("\n" + "=" * 60)
    print("CASCADE⚡𓂧φ∞ UNIFIED FIELD PROTOCOL COMPLETED")
    print("=" * 60 + "\n")

def main():
    """Process command line arguments and run the selected demo."""
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Quantum Field Demo")
    
    parser.add_argument('--demo', type=str, default='full',
                       choices=['full', 'cascade', 'timeline', 'multidimensional'],
                       help="Demo to run (default: full)")
    
    parser.add_argument('--dimensions', type=str, default="21,21,21",
                       help="Field dimensions as comma-separated values (default: 21,21,21)")
    
    parser.add_argument('--duration', type=int, default=60,
                       help="Simulation duration in seconds (default: 60)")
    
    parser.add_argument('--interactive', action='store_true',
                       help="Run in interactive mode")
    
    parser.add_argument('--no-visualization', action='store_true',
                       help="Disable visualizations")
    
    args = parser.parse_args()
    
    # Parse dimensions
    try:
        dimensions = tuple(int(x) for x in args.dimensions.split(','))
    except ValueError:
        print("Error: dimensions must be comma-separated integers")
        return 1
    
    # Run selected demo
    if args.demo == 'cascade':
        run_cascade_demo(
            dimensions=dimensions[:3],
            duration=args.duration,
            interactive=args.interactive,
            show_visualization=not args.no_visualization
        )
    elif args.demo == 'timeline':
        run_timeline_demo(
            duration=args.duration,
            show_visualization=not args.no_visualization
        )
    elif args.demo == 'multidimensional':
        # For 4D demo, use 4th dimension as time steps if provided
        if len(dimensions) >= 4:
            time_steps = dimensions[3]
        else:
            time_steps = 8
        
        run_multidimensional_demo(
            dimensions=(*dimensions[:3], time_steps),
            show_visualization=not args.no_visualization
        )
    else:  # full demo
        run_full_cascade_demo(
            duration=args.duration,
            show_visualization=not args.no_visualization
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())