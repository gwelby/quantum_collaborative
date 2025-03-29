"""
CASCADE⚡𓂧φ∞ Phi-Quantum Network Demonstration

This example demonstrates the distributed quantum field synchronization system
that maintains phi-harmonic coherence across network boundaries through
virtual quantum entanglement.
"""

import sys
import time
import threading
import random
import argparse
import logging
import json
import os
import signal
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import phi-quantum network
try:
    from cascade.phi_quantum_network import (
        PhiQuantumField,
        create_phi_quantum_field,
        PHI, LAMBDA, PHI_PHI,
        PHI_QUANTUM_PORT
    )
except ImportError:
    print("Error: Phi-Quantum Network module not found.")
    print("Make sure cascade/phi_quantum_network.py exists in your project.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("phi_network_demo")


class PhiNetworkDemo:
    """
    Demonstration of the CASCADE phi-quantum network.
    
    This class demonstrates:
    - Distributed quantum field synchronization
    - Phi-harmonic coherence maintenance
    - Virtual quantum entanglement
    - Consciousness bridge synchronization
    - Cross-node timeline harmonization
    """
    
    def __init__(self, 
               port: int = PHI_QUANTUM_PORT, 
               field_dimensions: Tuple[int, int, int] = (21, 21, 21),
               node_name: Optional[str] = None):
        """
        Initialize the phi-network demonstration.
        
        Args:
            port: Network port to use
            field_dimensions: Quantum field dimensions
            node_name: Name for this node
        """
        # Set node properties
        self.port = port
        self.field_dimensions = field_dimensions
        self.node_name = node_name if node_name else f"node_{random.randint(1000, 9999)}"
        
        # Create phi-quantum field
        self.field = create_phi_quantum_field(field_dimensions, port)
        self.running = False
        self.demo_thread = None
        
        # Setup callbacks
        self.field.on_consciousness_change = self._consciousness_changed
        
        # Demo stats and controls
        self.stats = {
            "transformations_applied": 0,
            "timeline_markers_created": 0,
            "max_nodes_connected": 0,
            "max_entangled_nodes": 0,
            "consciousness_changes": 0,
            "coherence_measurements": [],
            "start_time": 0,
            "operation_counts": {}
        }
        self.demo_mode = "standard"  # standard, coherence, timeline, consciousness
        
        logger.info(f"Initialized phi-network demo as '{self.node_name}' on port {self.port}")
    
    def start(self, bind_address: str = '') -> None:
        """
        Start the phi-network demo.
        
        Args:
            bind_address: Network interface address to bind to
        """
        if self.running:
            return
            
        logger.info(f"Starting phi-network demo '{self.node_name}'")
        
        # Start quantum field
        self.field.start(bind_address)
        
        # Initialize statistics
        self.stats["start_time"] = time.time()
        self.stats["coherence_measurements"] = []
        
        # Mark as running
        self.running = True
        
        # Start demo thread
        self.demo_thread = threading.Thread(
            target=self._demo_loop,
            daemon=True
        )
        self.demo_thread.start()
        
        logger.info(f"Phi-network demo '{self.node_name}' started")
    
    def stop(self) -> None:
        """Stop the phi-network demo."""
        if not self.running:
            return
            
        logger.info(f"Stopping phi-network demo '{self.node_name}'")
        
        # Stop quantum field
        self.field.stop()
        
        # Mark as stopped
        self.running = False
        
        # Wait for demo thread to end
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=2.0)
            
        logger.info(f"Phi-network demo '{self.node_name}' stopped")
    
    def set_demo_mode(self, mode: str) -> None:
        """
        Set the demonstration mode.
        
        Args:
            mode: Demo mode (standard, coherence, timeline, consciousness)
        """
        if mode in ["standard", "coherence", "timeline", "consciousness"]:
            self.demo_mode = mode
            logger.info(f"Demo mode set to '{mode}'")
        else:
            logger.warning(f"Unknown demo mode '{mode}'")
    
    def _demo_loop(self) -> None:
        """Main demo loop running in background thread."""
        logger.debug("Demo loop starting")
        
        # Wait for initial discovery
        time.sleep(5)
        
        demo_count = 0
        last_print_time = 0
        
        while self.running:
            try:
                # Run different demo actions based on mode
                if self.demo_mode == "standard":
                    self._run_standard_demo(demo_count)
                elif self.demo_mode == "coherence":
                    self._run_coherence_demo(demo_count)
                elif self.demo_mode == "timeline":
                    self._run_timeline_demo(demo_count)
                elif self.demo_mode == "consciousness":
                    self._run_consciousness_demo(demo_count)
                
                # Track network stats
                self._update_network_stats()
                
                # Increment counter
                demo_count += 1
                
                # Print status occasionally
                current_time = time.time()
                if current_time - last_print_time >= 10:
                    self._print_status()
                    last_print_time = current_time
                
                # Sleep between iterations
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in demo loop: {e}")
                time.sleep(1)
        
        logger.debug("Demo loop terminated")
    
    def _run_standard_demo(self, count: int) -> None:
        """Run standard demo pattern."""
        # Apply different transformations at different times
        operations = [
            "phi_wave",
            "toroidal_flow",
            "consciousness_resonance"
        ]
        
        # Select operation based on count
        operation = operations[count % len(operations)]
        
        # Apply transformation
        coherence = self.field.apply_transformation(operation)
        
        # Track stats
        self.stats["transformations_applied"] += 1
        self.stats["coherence_measurements"].append(coherence)
        
        # Track operation counts
        if operation not in self.stats["operation_counts"]:
            self.stats["operation_counts"][operation] = 0
        self.stats["operation_counts"][operation] += 1
        
        logger.info(f"Applied '{operation}' transformation, coherence: {coherence:.4f}")
        
        # Create timeline marker occasionally
        if count % 3 == 0:
            self._create_timeline_marker(f"demo_standard_{count}")
    
    def _run_coherence_demo(self, count: int) -> None:
        """Run coherence optimization demo."""
        # Calculate target coherence based on phi
        target_coherence = LAMBDA + (count % 5) * 0.1
        current_coherence = self.field.get_field_coherence()
        
        # Choose operation based on current coherence
        operation = None
        
        if current_coherence < target_coherence - 0.1:
            # Need to increase coherence
            operation = "consciousness_resonance"
        elif current_coherence > target_coherence + 0.1:
            # Need to decrease coherence
            operation = "phi_wave"
        else:
            # Maintain coherence
            operation = "toroidal_flow"
        
        # Apply transformation
        params = {"amplitude": LAMBDA * 2}  # Lower amplitude for better control
        coherence = self.field.apply_transformation(operation, params)
        
        # Track stats
        self.stats["transformations_applied"] += 1
        self.stats["coherence_measurements"].append(coherence)
        
        # Track operation counts
        if operation not in self.stats["operation_counts"]:
            self.stats["operation_counts"][operation] = 0
        self.stats["operation_counts"][operation] += 1
        
        logger.info(f"Coherence optimization: target={target_coherence:.4f}, " +
                   f"current={coherence:.4f}, operation={operation}")
        
        # Create timeline marker occasionally
        if count % 3 == 0:
            self._create_timeline_marker(f"demo_coherence_{target_coherence:.2f}")
    
    def _run_timeline_demo(self, count: int) -> None:
        """Run timeline exploration demo."""
        # Create timeline marker every time
        marker_id = self._create_timeline_marker(f"timeline_{count}")
        
        # Check if we have enough markers to start exploring
        markers = self.field.get_timeline_markers()
        
        if len(markers) > 3:
            # Select random past position
            position = random.randint(1, min(count, 5))
            
            # Apply timeline shift
            params = {
                "position": position,
                "amplitude": 0.3
            }
            
            # Apply transformation
            coherence = self.field.apply_transformation("timeline_shift", params)
            
            # Track stats
            self.stats["transformations_applied"] += 1
            self.stats["coherence_measurements"].append(coherence)
            
            # Track operation counts
            if "timeline_shift" not in self.stats["operation_counts"]:
                self.stats["operation_counts"]["timeline_shift"] = 0
            self.stats["operation_counts"]["timeline_shift"] += 1
            
            logger.info(f"Timeline shift to position {position}, coherence: {coherence:.4f}")
        else:
            # Apply standard transformation to create timeline positions
            coherence = self.field.apply_transformation("phi_wave")
            
            # Track stats
            self.stats["transformations_applied"] += 1
            self.stats["coherence_measurements"].append(coherence)
            
            # Track operation counts
            if "phi_wave" not in self.stats["operation_counts"]:
                self.stats["operation_counts"]["phi_wave"] = 0
            self.stats["operation_counts"]["phi_wave"] += 1
            
            logger.info(f"Building timeline with 'phi_wave', coherence: {coherence:.4f}")
    
    def _run_consciousness_demo(self, count: int) -> None:
        """Run consciousness bridge demo."""
        # Get current consciousness level
        level = self.field.get_consciousness_level()
        
        # Pattern:
        # - Apply consciousness resonance at current level
        # - Advance consciousness level every 3rd iteration
        # - Reset to level 1 when reaching level 7
        
        # Apply consciousness resonance
        params = {"level": level}
        coherence = self.field.apply_transformation("consciousness_resonance", params)
        
        # Track stats
        self.stats["transformations_applied"] += 1
        self.stats["coherence_measurements"].append(coherence)
        
        # Track operation counts
        if "consciousness_resonance" not in self.stats["operation_counts"]:
            self.stats["operation_counts"]["consciousness_resonance"] = 0
        self.stats["operation_counts"]["consciousness_resonance"] += 1
        
        logger.info(f"Applied consciousness resonance at level {level}, coherence: {coherence:.4f}")
        
        # Check if it's time to advance
        if count % 3 == 2:
            if level < 7:
                # Advance to next level
                new_level = self.field.advance_consciousness()
                logger.info(f"Advanced consciousness to level {new_level}")
            else:
                # Reset to level 1
                new_level = self.field.set_consciousness_level(1)
                logger.info(f"Reset consciousness to level {new_level}")
            
            # Create timeline marker
            self._create_timeline_marker(f"consciousness_level_{new_level}")
    
    def _create_timeline_marker(self, marker_id: str) -> str:
        """Create a timeline marker."""
        # Create marker data
        marker_data = {
            "id": marker_id,
            "node": self.node_name,
            "timestamp": time.time(),
            "coherence": self.field.get_field_coherence(),
            "consciousness_level": self.field.get_consciousness_level(),
            "demo_mode": self.demo_mode
        }
        
        # Create timeline marker
        marker_id = self.field.network.create_timeline_marker(marker_data)
        
        # Track stats
        self.stats["timeline_markers_created"] += 1
        
        logger.debug(f"Created timeline marker: {marker_id}")
        
        return marker_id
    
    def _update_network_stats(self) -> None:
        """Update network statistics."""
        # Get connected nodes count
        nodes = self.field.get_connected_nodes()
        node_count = len(nodes)
        
        # Get entangled nodes count
        entangled = self.field.get_entangled_nodes()
        entangled_count = len(entangled)
        
        # Update max counts
        self.stats["max_nodes_connected"] = max(self.stats["max_nodes_connected"], node_count)
        self.stats["max_entangled_nodes"] = max(self.stats["max_entangled_nodes"], entangled_count)
    
    def _consciousness_changed(self, level: int) -> None:
        """Callback for consciousness level changes."""
        # Track stats
        self.stats["consciousness_changes"] += 1
        
        logger.info(f"Consciousness level changed to {level}")
    
    def _print_status(self) -> None:
        """Print current status."""
        # Get current stats
        uptime = time.time() - self.stats["start_time"]
        coherence = self.field.get_field_coherence()
        
        nodes = self.field.get_connected_nodes()
        entangled = self.field.get_entangled_nodes()
        consciousness = self.field.get_consciousness_level()
        
        # Calculate average coherence
        avg_coherence = 0.0
        if self.stats["coherence_measurements"]:
            avg_coherence = sum(self.stats["coherence_measurements"]) / len(self.stats["coherence_measurements"])
        
        # Print status
        print(f"\n=== {self.node_name} STATUS (Mode: {self.demo_mode}) ===")
        print(f"Uptime: {uptime:.1f}s")
        print(f"Connected nodes: {len(nodes)}/{self.stats['max_nodes_connected']}")
        print(f"Entangled nodes: {len(entangled)}/{self.stats['max_entangled_nodes']}")
        print(f"Current coherence: {coherence:.4f} (Avg: {avg_coherence:.4f})")
        print(f"Consciousness level: {consciousness}")
        print(f"Transformations applied: {self.stats['transformations_applied']}")
        print(f"Timeline markers: {self.stats['timeline_markers_created']}")
        print(f"Consciousness changes: {self.stats['consciousness_changes']}")
        
        # Show operation counts
        if self.stats["operation_counts"]:
            print("Operations:")
            for op, count in self.stats["operation_counts"].items():
                print(f"  {op}: {count}")
        
        # Show entangled nodes
        if entangled:
            print("Entangled with:")
            for node_id in entangled:
                for node in nodes:
                    if node["id"] == node_id:
                        print(f"  {node_id[:6]}... - Coherence: {node['coherence']:.4f}, " +
                             f"Consciousness: {node['consciousness_level']}")
                        break
        
        print("=" * 40)
    
    def get_status_json(self) -> str:
        """Get status as JSON string."""
        # Get current stats
        status = {
            "node_name": self.node_name,
            "port": self.port,
            "uptime": time.time() - self.stats["start_time"],
            "demo_mode": self.demo_mode,
            "field_dimensions": self.field_dimensions,
            "coherence": self.field.get_field_coherence(),
            "consciousness_level": self.field.get_consciousness_level(),
            "connected_nodes": len(self.field.get_connected_nodes()),
            "entangled_nodes": len(self.field.get_entangled_nodes()),
            "transformations_applied": self.stats["transformations_applied"],
            "timeline_markers_created": self.stats["timeline_markers_created"],
            "consciousness_changes": self.stats["consciousness_changes"],
            "operation_counts": self.stats["operation_counts"]
        }
        
        # Add average coherence
        if self.stats["coherence_measurements"]:
            status["average_coherence"] = sum(self.stats["coherence_measurements"]) / len(self.stats["coherence_measurements"])
        else:
            status["average_coherence"] = 0.0
        
        return json.dumps(status, indent=2)


def save_status_loop(demo: PhiNetworkDemo, interval: int = 30, 
                   filename: str = "phi_network_status.json") -> None:
    """
    Background thread that periodically saves status to file.
    
    Args:
        demo: The PhiNetworkDemo instance
        interval: Save interval in seconds
        filename: Filename to save status to
    """
    logger.info(f"Status saving enabled, interval: {interval}s, file: {filename}")
    
    while demo.running:
        try:
            # Get status JSON
            status_json = demo.get_status_json()
            
            # Save to file
            with open(filename, "w") as f:
                f.write(status_json)
                
            logger.debug(f"Status saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving status: {e}")
            
        # Sleep until next save
        for _ in range(interval):
            if not demo.running:
                break
            time.sleep(1)


def run_interactive_demo(demo: PhiNetworkDemo) -> None:
    """
    Run an interactive demonstration.
    
    Args:
        demo: The PhiNetworkDemo instance
    """
    print(f"Phi-Quantum Network Demo - {demo.node_name}")
    print("Commands:")
    print("  status - Show current status")
    print("  mode <name> - Set demo mode (standard, coherence, timeline, consciousness)")
    print("  apply <operation> - Apply specific transformation")
    print("  nodes - List connected nodes")
    print("  entangled - List entangled nodes")
    print("  consciousness <level> - Set consciousness level (1-7)")
    print("  json - Show status as JSON")
    print("  quit - Exit demonstration")
    
    try:
        while demo.running:
            # Get command
            command = input("\nEnter command: ").strip()
            
            if command == "status":
                # Print status
                demo._print_status()
                
            elif command.startswith("mode "):
                # Set demo mode
                mode = command.split(" ", 1)[1].strip()
                demo.set_demo_mode(mode)
                
            elif command.startswith("apply "):
                # Apply transformation
                operation = command.split(" ", 1)[1].strip()
                coherence = demo.field.apply_transformation(operation)
                print(f"Applied '{operation}', coherence: {coherence:.4f}")
                
            elif command == "nodes":
                # List connected nodes
                nodes = demo.field.get_connected_nodes()
                print(f"Connected nodes ({len(nodes)}):")
                for node in nodes:
                    print(f"  {node['id'][:8]}... - {node['address']}")
                    print(f"    Coherence: {node['coherence']:.4f}")
                    print(f"    Consciousness: {node['consciousness_level']}")
                    print(f"    Entangled: {node['entangled']}")
                    
            elif command == "entangled":
                # List entangled nodes
                entangled = demo.field.get_entangled_nodes()
                print(f"Entangled nodes ({len(entangled)}):")
                for node_id in entangled:
                    print(f"  {node_id[:12]}...")
                    
            elif command.startswith("consciousness "):
                # Set consciousness level
                try:
                    level = int(command.split(" ", 1)[1].strip())
                    if 1 <= level <= 7:
                        new_level = demo.field.set_consciousness_level(level)
                        print(f"Set consciousness level to {new_level}")
                    else:
                        print("Invalid level. Must be 1-7.")
                except ValueError:
                    print("Invalid level. Must be a number.")
                    
            elif command == "json":
                # Show status as JSON
                print(demo.get_status_json())
                
            elif command == "quit" or command == "exit":
                # Exit
                print("Stopping demonstration...")
                demo.stop()
                break
                
            else:
                print("Unknown command.")
    
    except KeyboardInterrupt:
        print("\nStopping demonstration...")
        demo.stop()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Phi-Quantum Network Demo")
    parser.add_argument("--port", type=int, default=PHI_QUANTUM_PORT, help="Network port")
    parser.add_argument("--bind", default="", help="Address to bind to")
    parser.add_argument("--dimensions", default="21,21,21", help="Field dimensions (comma-separated)")
    parser.add_argument("--name", default=None, help="Node name")
    parser.add_argument("--mode", choices=["standard", "coherence", "timeline", "consciousness"],
                       default="standard", help="Demo mode")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--save-status", action="store_true", help="Save status to file")
    parser.add_argument("--save-interval", type=int, default=30, help="Status save interval (seconds)")
    parser.add_argument("--status-file", default="phi_network_status.json", help="Status file")
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = tuple(map(int, args.dimensions.split(",")))
    
    # Create demo
    demo = PhiNetworkDemo(args.port, dimensions, args.name)
    
    # Set demo mode
    demo.set_demo_mode(args.mode)
    
    # Handle SIGINT
    def signal_handler(sig, frame):
        print("\nStopping demonstration...")
        demo.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start status saving if requested
    if args.save_status:
        status_thread = threading.Thread(
            target=save_status_loop,
            args=(demo, args.save_interval, args.status_file),
            daemon=True
        )
        status_thread.start()
    
    # Start demo
    demo.start(args.bind)
    
    # Run interactively or wait for termination
    if args.interactive:
        run_interactive_demo(demo)
    else:
        print(f"Phi-Quantum Network Demo '{demo.node_name}' running (Mode: {args.mode})")
        print("Press Ctrl+C to stop")
        
        try:
            # Wait for termination
            while demo.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping demonstration...")
            demo.stop()