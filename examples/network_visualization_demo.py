#!/usr/bin/env python3
"""
CASCADE⚡𓂧φ∞ Network Visualization Demo

This script demonstrates the network field visualization capabilities of the CASCADE system,
showing real-time quantum field synchronization across network nodes with phi-harmonic visualization.
"""

import os
import sys
import time
import argparse
import logging
import threading
import multiprocessing
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Setup path for imports
sys.path.append('/mnt/d/projects/python')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("network_viz_demo")

# Try to import visualization components
try:
    from cascade.phi_quantum_network import (
        create_phi_quantum_field,
        PhiQuantumField,
        PHI, LAMBDA, PHI_PHI
    )
    from cascade.visualization.network_field_visualizer import (
        create_network_visualizer,
        NetworkFieldVisualizer
    )
    from cascade.visualization.optimized_network_renderer import (
        CachingFieldSampler,
        ParallelNodeSampler,
        get_adaptive_sampling_grid
    )
    from cascade.visualization.network_dashboard import (
        create_network_dashboard,
        NetworkMonitor,
        NetworkDashboard
    )
except ImportError as e:
    print(f"Failed to import CASCADE components: {e}")
    print("Make sure you have the CASCADE environment set up correctly.")
    sys.exit(1)


def start_secondary_node(port: int, main_port: int, name: str, wait_time: float = 3.0) -> None:
    """
    Start a secondary quantum field node.

    Args:
        port: Port for this node
        main_port: Port of the main node to connect to
        name: Node name
        wait_time: Time to wait before attempting connection
    """
    logger.info(f"Starting secondary node {name} on port {port}")
    
    # Create quantum field
    field = create_phi_quantum_field(port=port)
    
    # Configure consciousness level based on node ID
    consciouness_level = (hash(name) % 6) + 1
    field.set_consciousness_level(consciouness_level)
    
    # Start field
    field.start()
    
    # Wait for main node to be ready
    time.sleep(wait_time)
    
    # Main loop
    try:
        # Apply random transformations to generate interesting field patterns
        operations = ["phi_wave", "toroidal_flow", "consciousness_resonance", "timeline_shift"]
        
        # Display status
        print(f"Node {name} running on port {port}, level: {field.get_consciousness_level()}")
        
        iteration = 0
        while True:
            # Every 10 seconds, apply a transformation
            if iteration % 10 == 0:
                operation = operations[iteration // 10 % len(operations)]
                field.apply_transformation(operation)
                print(f"Node {name}: Applied {operation}, coherence: {field.get_field_coherence():.4f}")
            
            # Every 5 iterations, print status
            if iteration % 5 == 0:
                entangled = field.get_entangled_nodes()
                print(f"Node {name}: Coherence: {field.get_field_coherence():.4f}, " +
                      f"Level: {field.get_consciousness_level()}, " +
                      f"Entangled with {len(entangled)} nodes")
            
            time.sleep(1)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"Stopping node {name}")
    finally:
        field.stop()


def run_demo(args: argparse.Namespace) -> None:
    """
    Run the network visualization demo.

    Args:
        args: Command line arguments
    """
    # Create and start main quantum field
    main_field = create_phi_quantum_field(port=args.port)
    main_field.start()
    
    logger.info(f"Started main quantum field on port {args.port}")
    
    # Sleep briefly to let the field initialize
    time.sleep(1)
    
    # Create secondary nodes
    secondary_nodes = []
    for i in range(args.nodes):
        node_port = args.base_port + i
        node_name = f"node_{i+1}"
        
        # Start node in a separate process
        process = multiprocessing.Process(
            target=start_secondary_node,
            args=(node_port, args.port, node_name),
            daemon=True
        )
        process.start()
        secondary_nodes.append((process, node_name, node_port))
        
        # Sleep briefly to stagger node startup
        time.sleep(0.5)
    
    logger.info(f"Started {len(secondary_nodes)} secondary nodes")
    
    # Allow some time for nodes to connect and entangle
    logger.info("Waiting for network to stabilize...")
    time.sleep(3)
    
    try:
        # Create network visualizer
        visualizer = create_network_visualizer(main_field)
        
        # Apply optimizations for better performance
        if args.optimize:
            # Use parallel node sampler if we have multiple nodes
            sampler = ParallelNodeSampler(max_workers=4)
            visualizer.parallel_sampler = sampler
            sampler.start()
            
            # Use caching field sampler
            visualizer.field_sampler = CachingFieldSampler(cache_size=10)
            
            # Use adaptive sampling grid
            points = get_adaptive_sampling_grid(
                main_field.get_field_dimensions(),
                target_points=args.sampling_points
            )
            visualizer.probe_points = points
            
            logger.info("Applied performance optimizations to visualization")
        
        # Start visualization based on mode
        if args.dashboard:
            # Create and run dashboard
            dashboard = create_network_dashboard(
                main_field,
                mode=args.dashboard_mode,
                host=args.host,
                port=args.web_port
            )
        else:
            # Start standard visualization
            visualizer.start_visualization(
                mode=args.mode,
                update_interval=args.update_interval
            )
            
            # Run until user interrupts
            while True:
                time.sleep(1)
                
                # Periodically apply transformations
                if int(time.time()) % 20 == 0:
                    main_field.apply_transformation("phi_wave")
                    
                # Periodically advance consciousness
                if int(time.time()) % 60 == 0:
                    level = main_field.get_consciousness_level()
                    if level < 7:
                        main_field.advance_consciousness()
                        print(f"Advanced to consciousness level {main_field.get_consciousness_level()}")
                
    except KeyboardInterrupt:
        print("\nStopping visualization demo...")
    finally:
        # Stop visualization
        if 'visualizer' in locals() and hasattr(visualizer, 'running') and visualizer.running:
            visualizer.stop_visualization()
            
        # Stop sampler if running
        if 'sampler' in locals() and hasattr(sampler, 'running') and sampler.running:
            sampler.stop()
            
        # Stop main field
        main_field.stop()
        
        # Terminate secondary nodes
        for process, name, port in secondary_nodes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def save_visualizations(args: argparse.Namespace) -> None:
    """
    Save visualizations to files without interactive display.

    Args:
        args: Command line arguments
    """
    # Create quantum field
    field = create_phi_quantum_field(port=args.port)
    field.start()
    
    logger.info(f"Started quantum field on port {args.port}")
    
    # Sleep briefly to let the field initialize
    time.sleep(1)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create visualizer
        visualizer = create_network_visualizer(field)
        
        # Save static visualizations for all modes
        for mode in ['3d', 'grid', 'coherence', 'combined']:
            # Create visualization
            if mode == '3d':
                visualizer._create_3d_visualization()
            elif mode == 'grid':
                visualizer._create_grid_visualization()
            elif mode == 'coherence':
                visualizer._create_coherence_visualization()
            elif mode == 'combined':
                visualizer._create_combined_visualization()
            
            # Update data
            visualizer._update_network_data()
            
            # Update visualization
            visualizer._update_animation(0)
            
            # Save visualization
            filename = os.path.join(args.output_dir, f"network_{mode}.png")
            visualizer.save_visualization(filename)
            
            logger.info(f"Saved {mode} visualization to {filename}")
        
        # Create animation if requested
        if args.animation:
            # Create animation
            visualizer._create_combined_visualization()
            
            # Create animation filename
            filename = os.path.join(args.output_dir, "network_animation.mp4")
            
            # Save animation
            visualizer.save_animation(
                filename,
                duration=args.duration,
                fps=args.fps,
                dpi=args.dpi
            )
            
            logger.info(f"Saved animation to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving visualizations: {e}")
    finally:
        # Stop field
        field.stop()


def main() -> None:
    """Main entry point."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Network Visualization Demo")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--mode", choices=['3d', 'grid', 'coherence', 'combined'],
                            default="combined", help="Visualization mode")
    demo_parser.add_argument("--port", type=int, default=4321, help="Port for main node")
    demo_parser.add_argument("--nodes", type=int, default=3, help="Number of secondary nodes")
    demo_parser.add_argument("--base-port", type=int, default=5000, help="Base port for secondary nodes")
    demo_parser.add_argument("--update-interval", type=int, default=100, help="Update interval in ms")
    demo_parser.add_argument("--optimize", action="store_true", help="Apply performance optimizations")
    demo_parser.add_argument("--sampling-points", type=int, default=500, help="Number of field sampling points")
    demo_parser.add_argument("--dashboard", action="store_true", help="Show network dashboard")
    demo_parser.add_argument("--dashboard-mode", choices=['matplotlib', 'web'], default="matplotlib",
                            help="Dashboard mode")
    demo_parser.add_argument("--host", default="0.0.0.0", help="Host for web dashboard")
    demo_parser.add_argument("--web-port", type=int, default=8050, help="Port for web dashboard")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save visualizations to files")
    save_parser.add_argument("--output-dir", default="visualizations", help="Output directory")
    save_parser.add_argument("--port", type=int, default=4321, help="Port for node")
    save_parser.add_argument("--animation", action="store_true", help="Save animation")
    save_parser.add_argument("--duration", type=float, default=10.0, help="Animation duration in seconds")
    save_parser.add_argument("--fps", type=int, default=30, help="Animation frames per second")
    save_parser.add_argument("--dpi", type=int, default=100, help="Image resolution")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run requested command
    if args.command == "demo":
        run_demo(args)
    elif args.command == "save":
        save_visualizations(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Check if we're running on Windows
    if os.name == 'nt':
        # Set multiprocessing start method
        multiprocessing.set_start_method('spawn')
    
    main()