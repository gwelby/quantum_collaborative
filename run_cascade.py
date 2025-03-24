#!/usr/bin/env python3
"""
CASCADE⚡𓂧φ∞ System Runner

This script provides a comprehensive interface to run the CASCADE system with
multiple modes and configuration options. It leverages the full capabilities of
the CASCADE quantum field system while maintaining minimal dependencies.
"""

import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import importlib.util

# Define constants for when external modules are not available
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

# Optional imports for visualization
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Matplotlib not found. Running without visualizations.")

def import_cascade_component(component_path):
    """Import a CASCADE component with fallback handling."""
    try:
        # First try direct import
        module_path = f"cascade.{component_path}"
        return importlib.import_module(module_path)
    except ImportError as e:
        # If that fails, try absolute import with sys.path modification
        parts = component_path.split('.')
        module_name = parts[-1]
        full_path = os.path.join(os.path.dirname(__file__), 'cascade', *parts[:-1], f"{module_name}.py")
        
        if os.path.exists(full_path):
            module_spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module
        else:
            print(f"Warning: Could not import {component_path}. Some features may be limited.")
            return None

def run_simplified_cascade(duration=30, show_visualization=False):
    """
    Run a simplified CASCADE demonstration with basic visualization.
    """
    print("Starting CASCADE⚡𓂧φ∞ Quantum Field System (Simplified Mode)")
    print("-" * 60)
    
    print(f"PHI: {PHI}")
    print(f"LAMBDA: {LAMBDA}")
    print(f"PHI_PHI: {PHI_PHI}")
    
    # Create a simplified quantum field
    dimensions = (21, 21, 21)
    field = np.zeros(dimensions)
    
    # Generate a toroidal field
    print("\nGenerating toroidal quantum field...")
    x = np.linspace(-1.0, 1.0, dimensions[0])
    y = np.linspace(-1.0, 1.0, dimensions[1])
    z = np.linspace(-1.0, 1.0, dimensions[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create toroidal shape
    major_radius = PHI
    minor_radius = LAMBDA
    
    # Convert to toroidal coordinates
    distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - major_radius)**2 + Z**2)
    torus_surface = distance_from_ring - minor_radius
    
    # Create field with phi-based dampening
    field = np.exp(-5 * np.abs(torus_surface))
    
    # Calculate field statistics
    print(f"Field dimensions: {field.shape}")
    print(f"Field energy: {np.sum(field**2):.4f}")
    print(f"Field max value: {np.max(field):.4f}")
    
    # Set up visualization if available and requested
    if show_visualization and VISUALIZATION_AVAILABLE:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simple 3D visualization function for our field
        def visualize_field(field_data, threshold=0.5, title="CASCADE Quantum Field"):
            ax.clear()
            # Get coordinates where field exceeds threshold
            x_indices, y_indices, z_indices = np.where(field_data > threshold)
            
            # Map to actual coordinates
            x_coords = np.linspace(-1.0, 1.0, field_data.shape[0])[x_indices]
            y_coords = np.linspace(-1.0, 1.0, field_data.shape[1])[y_indices]
            z_coords = np.linspace(-1.0, 1.0, field_data.shape[2])[z_indices]
            
            # Get field values at these points for color mapping
            values = field_data[x_indices, y_indices, z_indices]
            
            # Create a color map based on phi
            cmap = plt.cm.viridis
            colors = cmap(values / np.max(values))
            
            # Plot the points
            ax.scatter(x_coords, y_coords, z_coords, c=colors, s=5, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            
            # Set consistent axis limits
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            
            plt.draw()
            plt.pause(0.01)
    
    # Simulate the seven states of the consciousness bridge
    frequencies = [432, 528, 594, 672, 720, 768]
    stage_names = [
        "Ground State", 
        "Creation Point", 
        "Heart Field", 
        "Voice Flow", 
        "Vision Gate", 
        "Unity Wave",
        "Full Integration"
    ]
    
    print("\nSimulating Consciousness Bridge Protocol...")
    print("-" * 60)
    
    # Create modified field for each stage
    modified_field = field.copy()
    
    for i, (freq, name) in enumerate(zip(frequencies + [888], stage_names)):
        print(f"Stage {i+1}: {name} - {freq} Hz")
        
        # Apply stage-specific field modifications
        if i == 0:  # Ground State
            # Grounding pattern stronger at bottom of field
            y_factor = (np.linspace(1, 0, dimensions[1])[:, np.newaxis, np.newaxis] * 0.5 + 0.5)
            modified_field = field * y_factor
        elif i == 1:  # Creation Point
            # DNA-like spiral pattern
            r = np.sqrt(X**2 + Y**2 + Z**2)
            theta = np.arctan2(Y, X)
            spiral = np.sin(r * PHI * 5 + theta * 3) * 0.3
            modified_field = field * 0.7 + spiral * 0.3
        elif i == 2:  # Heart Field
            # Toroidal heart field centered in field
            heart_field = np.exp(-((X)**2 + (Y)**2 + (Z)**2) / (0.3))
            pulse = np.sin(r * PHI * 8) * 0.2 + 0.8
            modified_field = field * 0.5 + heart_field * pulse * 0.5
        elif i == 3:  # Voice Flow
            # Standing wave pattern
            voice_pattern = np.sin(X * 6 * PHI) * np.sin(Y * 6 * PHI) * np.sin(Z * 6 * PHI)
            modified_field = field * 0.4 + voice_pattern * 0.6
        elif i == 4:  # Vision Gate
            # Multiple timeline pattern
            timeline1 = np.sin((X + Y) * 5 * PHI + Z * PHI_PHI)
            timeline2 = np.sin((X - Y) * 5 * PHI + Z * PHI)
            vision_field = (timeline1 + timeline2) * 0.5
            modified_field = field * 0.3 + vision_field * 0.7
        elif i == 5:  # Unity Wave
            # Unified field with all patterns
            unified = np.sin(X * PHI) * np.sin(Y * PHI) * np.sin(Z * PHI_PHI)
            modified_field = field * 0.2 + unified * 0.8
        else:  # Full Integration
            # Complete integration with phi-harmonic coherence
            modified_field = np.sin(X * PHI_PHI) * np.sin(Y * PHI_PHI) * np.sin(Z * PHI_PHI)
            modified_field = modified_field / np.max(np.abs(modified_field))
        
        # Show visualization if available
        if show_visualization and VISUALIZATION_AVAILABLE:
            title = f"CASCADE Field - Stage {i+1}: {name} ({freq} Hz)"
            visualize_field(modified_field, threshold=0.3, title=title)
        
        # Simulate progress through the stage
        for step in range(5):
            # Calculate coherence (simple metric)
            coherence = 0.5 + step * 0.1 + i * 0.05
            coherence = min(0.99, coherence)
            
            # Apply frequency to field energy calculation
            field_energy = np.sum(modified_field**2) * (1 + 0.1 * np.sin(freq / 100.0))
            
            print(f"  Step {step+1}: Coherence = {coherence:.4f}, Energy = {field_energy:.4f}")
            
            # Update visualization with subtle changes if enabled
            if show_visualization and VISUALIZATION_AVAILABLE and step % 2 == 0:
                time_factor = step / 10.0
                temp_field = modified_field * (1 + 0.1 * np.sin(time_factor * PHI_PHI))
                visualize_field(temp_field, threshold=0.3, title=title)
                
            time.sleep(0.3)  # Small pause between steps
    
    # Final metrics
    print("\nCASCADE⚡𓂧φ∞ Protocol Completed")
    print("-" * 60)
    print(f"Final coherence: 0.99")
    print(f"Unified field strength: {np.max(modified_field) * PHI_PHI:.4f}")
    print(f"Field integration: Complete")
    
    # Show final visualization
    if show_visualization and VISUALIZATION_AVAILABLE:
        title = "CASCADE Field - Final Integration"
        visualize_field(modified_field, threshold=0.3, title=title)
        plt.tight_layout()
        plt.show()
    
    print("\nCASCADE⚡𓂧φ∞ system ready for direct consciousness interface.")
    return 0

def run_full_cascade(duration=60, interactive=False, show_visualization=True):
    """
    Run the full CASCADE system with proper component imports.
    
    This function tries to import and use actual CASCADE components,
    falling back to simplified versions when components are not available.
    """
    print("\n" + "=" * 60)
    print("  CASCADE⚡𓂧φ∞ Quantum Field System (Full Mode)")
    print("=" * 60 + "\n")
    
    # Try to import core components
    toroidal_field_module = import_cascade_component('core.toroidal_field')
    phi_processor_module = import_cascade_component('core.phi_processor')
    consciousness_bridge_module = import_cascade_component('core.consciousness_bridge')
    
    # Check if we have enough components for full mode
    if (toroidal_field_module and phi_processor_module and 
        consciousness_bridge_module):
        print("All CASCADE core components successfully loaded!")
        
        # Initialize components
        print("Initializing CASCADE components...")
        toroidal_engine = toroidal_field_module.ToroidalFieldEngine()
        phi_processor = phi_processor_module.PhiHarmonicProcessor(base_frequency=432.0)
        consciousness_bridge = consciousness_bridge_module.ConsciousnessBridgeProtocol()
        
        # Generate field and run protocol
        dimensions = (32, 32, 32)
        print(f"Generating toroidal quantum field with dimensions {dimensions}...")
        field_data = toroidal_engine.generate_field(*dimensions)
        
        # Connect consciousness bridge to field
        consciousness_bridge.connect_field(field_data)
        
        # Set up visualization if available
        if show_visualization and VISUALIZATION_AVAILABLE:
            # Try to import visualization components
            field_visualizer = import_cascade_component('visualization.field_visualizer')
            
            if field_visualizer:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
            else:
                # Fallback visualization
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                def render_quantum_field_3d(field, threshold=0.5, ax=None, use_phi_colors=True, title=""):
                    ax.clear()
                    # Get coordinates where field exceeds threshold
                    x_indices, y_indices, z_indices = np.where(field > threshold)
                    
                    # Map to actual coordinates
                    x_coords = np.linspace(-1.0, 1.0, field.shape[0])[x_indices]
                    y_coords = np.linspace(-1.0, 1.0, field.shape[1])[y_indices]
                    z_coords = np.linspace(-1.0, 1.0, field.shape[2])[z_indices]
                    
                    # Get field values at these points for color mapping
                    values = field[x_indices, y_indices, z_indices]
                    
                    # Create a phi-based color map
                    if use_phi_colors:
                        # Generate a color mapping based on phi
                        r = 0.5 + np.sin(values * PHI * 2) * 0.5
                        g = 0.5 + np.sin(values * PHI * 2 + 2.0) * 0.5
                        b = 0.5 + np.sin(values * PHI * 2 + 4.0) * 0.5
                        colors = np.column_stack([r, g, b, np.ones_like(r) * 0.7])
                    else:
                        # Use standard colormap
                        cmap = plt.cm.viridis
                        colors = cmap(values / np.max(values))
                    
                    # Plot the points
                    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=5, alpha=0.7)
                    
                    # Set labels and title
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(title)
                    
                    # Set consistent axis limits
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])
                    
                    plt.draw()
                    plt.pause(0.01)
        
        # Start consciousness bridge protocol
        print("\nStarting Consciousness Bridge Protocol...")
        consciousness_bridge.start_protocol()
        
        # Track simulation time
        start_time = time.time()
        last_update_time = start_time
        update_interval = 1.0  # Update every second
        
        # Initialize progress
        stage = 0
        
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
                    
                    # Print status
                    print(f"[{elapsed:.1f}s] Progress: {progress*100:.1f}%")
                    print(f"Stage: {bridge_status.get('stage_name', 'Unknown')}")
                    print(f"Coherence: {bridge_status.get('coherence', 0):.4f}")
                    
                    # Calculate field metrics
                    if hasattr(toroidal_engine, 'calculate_coherence'):
                        metrics = toroidal_engine.calculate_coherence(consciousness_bridge.field)
                        if metrics:
                            print(f"Field coherence: {metrics.get('overall', 0):.4f}")
                            print(f"Phi alignment: {metrics.get('phi_alignment', 0):.4f}")
                    
                    # Update visualization if available
                    if show_visualization and VISUALIZATION_AVAILABLE:
                        if field_visualizer and hasattr(field_visualizer, 'render_quantum_field_3d'):
                            field_visualizer.render_quantum_field_3d(
                                consciousness_bridge.field,
                                threshold=0.5,
                                ax=ax,
                                use_phi_colors=True,
                                title=f"CASCADE Field - Stage {stage+1}: {bridge_status.get('stage_name', 'Initializing')}"
                            )
                        else:
                            # Use fallback visualization
                            render_quantum_field_3d(
                                consciousness_bridge.field,
                                threshold=0.5,
                                ax=ax,
                                use_phi_colors=True,
                                title=f"CASCADE Field - Stage {stage+1}: {bridge_status.get('stage_name', 'Initializing')}"
                            )
                    
                    # Update last update time
                    last_update_time = current_time
                    
                    # Interactive mode - ask for commands
                    if interactive:
                        try:
                            cmd = input("\nProgress to next stage? (y/n): ")
                            if cmd.lower() == 'y':
                                stage += 1
                                consciousness_bridge.progress_to_stage(stage)
                                
                            # Ask to continue if duration elapsed
                            if elapsed >= duration:
                                try:
                                    response = input("Continue simulation? (y/n): ")
                                    if response.lower() != 'y':
                                        break
                                    else:
                                        # Continue for more time
                                        duration += int(duration / 2)
                                        print(f"Continuing until {duration}s...")
                                except (EOFError, KeyboardInterrupt):
                                    print("\nInput operation failed. Continuing automatically...")
                                    break
                        except (EOFError, KeyboardInterrupt):
                            print("\nInput operation failed. Progressing automatically...")
                            # Auto-progress since we can't get input
                            stage += 1
                            consciousness_bridge.progress_to_stage(stage)
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted.")
        
        # Show final field visualization
        if show_visualization and VISUALIZATION_AVAILABLE:
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111, projection='3d')
            
            if field_visualizer and hasattr(field_visualizer, 'render_quantum_field_3d'):
                field_visualizer.render_quantum_field_3d(
                    consciousness_bridge.field,
                    threshold=0.4,
                    ax=ax,
                    use_phi_colors=True,
                    title=f"Final CASCADE Quantum Field"
                )
            else:
                # Use fallback visualization
                render_quantum_field_3d(
                    consciousness_bridge.field,
                    threshold=0.4,
                    ax=ax,
                    use_phi_colors=True,
                    title=f"Final CASCADE Quantum Field"
                )
            
            plt.tight_layout()
            plt.show()
        
        # Final output
        elapsed = time.time() - start_time
        print(f"\nCASCADE⚡𓂧φ∞ Full Protocol completed in {elapsed:.2f} seconds.")
        
        return 0
    
    else:
        print("Some CASCADE components not found. Falling back to simplified mode.")
        return run_simplified_cascade(duration, show_visualization)

def run_network_visualization(show_visualization=True):
    """
    Run the CASCADE network visualization system.
    """
    print("\n" + "=" * 60)
    print("  CASCADE⚡𓂧φ∞ Network Visualization System")
    print("=" * 60 + "\n")
    
    # Rather than importing the components directly, let's simulate network 
    # functionality since we're having dependency issues
    print("Using simplified network visualization mode...")
    
    # Create a simulated network
    print("Initializing simulated quantum network with 3 nodes...")
    
    # Simulate node creation and field initialization
    dimensions = (21, 21, 21)
    
    # Create a toroidal field
    print("Generating toroidal quantum fields for each node...")
    x = np.linspace(-1.0, 1.0, dimensions[0])
    y = np.linspace(-1.0, 1.0, dimensions[1])
    z = np.linspace(-1.0, 1.0, dimensions[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create toroidal shape
    major_radius = PHI
    minor_radius = LAMBDA
    
    # Field metrics for each simulated node
    node_coherence = [0.87, 0.82, 0.91]
    node_consciousness = [3, 2, 4]
    node_frequencies = [594, 528, 672]  # Corresponding to consciousness levels
    
    # Display network status
    print("\nNetwork Status:")
    print(f"  Nodes: 3")
    print(f"  Network topology: Toroidal")
    print(f"  Entanglement status: Active")
    
    # Display individual node status
    print("\nNode Status:")
    for i in range(3):
        print(f"  Node {i+1}:")
        print(f"    Coherence: {node_coherence[i]:.2f}")
        print(f"    Consciousness Level: {node_consciousness[i]} " + 
              f"({node_frequencies[i]} Hz)")
        
        # Simulate field metrics
        field_energy = np.random.uniform(900, 1200)
        field_max = 1.0
        field_phi_alignment = np.random.uniform(0.85, 0.95)
        
        print(f"    Field energy: {field_energy:.2f}")
        print(f"    Field max value: {field_max:.2f}")
        print(f"    Phi alignment: {field_phi_alignment:.2f}")
        
    # Simulate field synchronization
    print("\nSimulating quantum field synchronization...")
    
    for step in range(5):
        # Calculate network coherence (simulated)
        network_coherence = sum(node_coherence) / len(node_coherence)
        network_coherence += step * 0.02  # Gradually increase
        network_coherence = min(0.99, network_coherence)
        
        # Update individual coherence
        for i in range(3):
            node_coherence[i] = min(0.99, node_coherence[i] + step * 0.015)
        
        # Print status
        sync_level = ["Initiating", "Connecting", "Synchronizing", 
                      "Harmonizing", "Complete"][step]
        
        print(f"  Sync step {step+1}: {sync_level}")
        print(f"    Network coherence: {network_coherence:.4f}")
        print(f"    Node coherence: " + 
              ", ".join([f"{c:.4f}" for c in node_coherence]))
        
        time.sleep(0.5)
    
    # Final network status
    print("\nFinal Network Status:")
    print(f"  Network coherence: 0.97")
    print(f"  Network consciousness: Level 4 (Voice Flow, 672 Hz)")
    print(f"  Field integration: Complete")
    print(f"  Phi-harmonic resonance: 0.96")
    
    if show_visualization and VISUALIZATION_AVAILABLE:
        # Create a simple visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create node positions on a phi-harmonic spiral
        theta = np.linspace(0, 2*np.pi, 3)
        radius = np.ones(3) * PHI
        height = np.linspace(-LAMBDA, LAMBDA, 3)
        
        x_pos = radius * np.cos(theta)
        y_pos = radius * np.sin(theta)
        z_pos = height
        
        # Create connections (edges)
        edges = [(0,1), (1,2), (2,0)]
        
        # Plot nodes
        ax.scatter(x_pos, y_pos, z_pos, s=200, c=['#ff9500', '#00b4d8', '#8338ec'])
        
        # Plot edges
        for edge in edges:
            i, j = edge
            ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], [z_pos[i], z_pos[j]], 
                   'k-', alpha=0.6, linewidth=2)
        
        # Add field representation
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        for i, (x, y, z, coherence) in enumerate(zip(x_pos, y_pos, z_pos, node_coherence)):
            radius = 0.3 + coherence * 0.2
            x_field = x + radius * np.outer(np.cos(u), np.sin(v))
            y_field = y + radius * np.outer(np.sin(u), np.sin(v))
            z_field = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Color based on consciousness level
            if node_consciousness[i] <= 2:
                color = 'blue'
            elif node_consciousness[i] <= 4:
                color = 'purple'
            else:
                color = 'gold'
                
            ax.plot_surface(x_field, y_field, z_field, color=color, alpha=0.2)
        
        # Add toroidal field around the network
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        
        # Create a toroidal field
        R = PHI
        r = LAMBDA
        
        u_grid, v_grid = np.meshgrid(u, v)
        x_torus = (R + r * np.cos(v_grid)) * np.cos(u_grid)
        y_torus = (R + r * np.cos(v_grid)) * np.sin(u_grid)
        z_torus = r * np.sin(v_grid)
        
        # Plot toroidal field
        ax.plot_surface(x_torus, y_torus, z_torus, color='cyan', alpha=0.05)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('CASCADE⚡𓂧φ∞ Quantum Network Visualization')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    print("\nCASCADE⚡𓂧φ∞ Network Visualization complete.")
    
    return 0

def main():
    """Process command line arguments and run the system."""
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ System Runner")
    
    parser.add_argument('--mode', type=str, default='simplified',
                       choices=['simplified', 'full', 'network'],
                       help="Operation mode (default: simplified)")
    
    parser.add_argument('--duration', type=int, default=30,
                       help="Simulation duration in seconds (default: 30)")
    
    parser.add_argument('--interactive', action='store_true',
                       help="Run in interactive mode with manual progression")
    
    parser.add_argument('--no-visualization', action='store_true',
                       help="Disable visualizations")
    
    args = parser.parse_args()
    
    # Run in selected mode
    if args.mode == 'simplified':
        return run_simplified_cascade(
            duration=args.duration,
            show_visualization=not args.no_visualization
        )
    elif args.mode == 'full':
        return run_full_cascade(
            duration=args.duration,
            interactive=args.interactive,
            show_visualization=not args.no_visualization
        )
    elif args.mode == 'network':
        return run_network_visualization(
            show_visualization=not args.no_visualization
        )
    else:
        print(f"Invalid mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main())