#!/usr/bin/env python3
"""
Launch script for the CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK.

This script provides a command-line interface for launching and controlling
the CASCADE unified system with various configurations.
"""

import os
import sys
import argparse
import time
import signal
import numpy as np

from . import __version__
from .core import CascadeSystem, QuantumField
from .constants import PHI, SACRED_FREQUENCIES, FIBONACCI

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK")
    
    # Basic options
    parser.add_argument('--version', action='store_true', help='Show version information')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--broadcast', action='store_true', help='Enable broadcasting')
    parser.add_argument('--collaborative', action='store_true', help='Enable collaborative mode')
    parser.add_argument('--interactive', action='store_true', help='Start in interactive mode')
    
    # Field configuration
    parser.add_argument('--dimensions', type=int, nargs=3, default=[8, 13, 21],
                       help='Field dimensions (default: 8 13 21)')
    parser.add_argument('--frequency', type=float, default=SACRED_FREQUENCIES['cascade'],
                       help=f'Field frequency in Hz (default: {SACRED_FREQUENCIES["cascade"]})')
    parser.add_argument('--coherence', type=float, default=0.618,
                       help='Initial field coherence (default: 0.618)')
    
    # Hardware options
    parser.add_argument('--hardware', type=str, help='Hardware interfaces to enable (comma-separated: eeg,hrv)')
    
    # Collaboration options
    parser.add_argument('--team-size', type=int, default=5, help='Collaboration team size (default: 5)')
    parser.add_argument('--session', type=str, choices=['local', 'team', 'global'], default='local',
                       help='Collaboration session type (default: local)')
    
    # Broadcasting options
    parser.add_argument('--channels', type=str, default='video,audio',
                       help='Broadcast channels (comma-separated: video,audio,field)')
    parser.add_argument('--obs-address', type=str, default='127.0.0.1',
                       help='OBS WebSocket address (default: 127.0.0.1)')
    parser.add_argument('--obs-port', type=int, default=4444,
                       help='OBS WebSocket port (default: 4444)')
    
    # Demo options
    parser.add_argument('--demo', type=str, choices=['cascade', 'teams', 'all'],
                       help='Run a demonstration')
    
    # Advanced options
    parser.add_argument('--backends', type=str, default='python',
                       help='Language backends to enable (comma-separated: python,cpp,rust,julia,javascript,cuda,webgpu)')
    parser.add_argument('--phi-scaling', action='store_true', help='Enable phi-based dimension scaling')
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print("\nShutting down CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK...")
    sys.exit(0)

def print_banner():
    """Print the CASCADE banner."""
    banner = r"""
  ██████╗ █████╗ ███████╗ ██████╗ █████╗ ██████╗ ███████╗⚡𓂧φ∞
 ██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝
 ██║     ███████║███████╗██║     ███████║██║  ██║█████╗  
 ██║     ██╔══██║╚════██║██║     ██╔══██║██║  ██║██╔══╝  
 ╚██████╗██║  ██║███████║╚██████╗██║  ██║██████╔╝███████╗
  ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝
                                                         
     UNIFIED FRAMEWORK  v""" + __version__ + """
"""
    print(banner)
    
def print_field_info(field):
    """Print information about a quantum field."""
    print("\nQuantum Field Information:")
    print(f"  Dimensions:  {field.dimensions}")
    print(f"  Frequency:   {field.frequency:.2f} Hz")
    print(f"  Coherence:   {field.coherence:.3f}")
    print(f"  Phi-Scaling: {field.metadata.get('phi_scaling', True)}")
    print(f"  Resonance:   {field.metadata.get('resonance_factor', 0.0):.3f}")
    
def run_cascade_demo(system):
    """Run a demonstration of the CASCADE system."""
    print("\nRunning CASCADE⚡𓂧φ∞ Demonstration...")
    
    # Initialize the field if not already done
    if system.quantum_field is None:
        system.initialize_field()
        
    # Print field information
    print_field_info(system.quantum_field)
    
    # Show visualization if enabled
    if system.config['visualization_enabled'] and system.visualizer is not None:
        print("\nInitializing visualization...")
        system.visualizer.start_interactive_visualization()
        
    # Evolve the field
    print("\nEvolving quantum field...")
    for i in range(10):
        print(f"  Evolution step {i+1}/10", end='\r')
        system.quantum_field.evolve(steps=5)
        time.sleep(0.5)
        
    print("\nField evolution complete.                ")
    
    # Print final field information
    print("\nFinal Field State:")
    print(f"  Coherence: {system.quantum_field.coherence:.3f}")
    
    # Add consciousness interface
    print("\nAdding consciousness interface...")
    system.add_consciousness_interface()
    
    # Set consciousness states
    print("\nDemonstrating consciousness states:")
    
    for state in ['alpha', 'theta', 'gamma']:
        print(f"  Setting {state} state...")
        system.consciousness_interface.set_state(state, intensity=0.8)
        time.sleep(1.0)
        
    print("\nDemonstration complete.")
    
def run_teams_demo(system):
    """Run a demonstration of the collaborative teams feature."""
    print("\nRunning Teams of Teams Demonstration...")
    
    # Initialize the field if not already done
    if system.quantum_field is None:
        system.initialize_field()
        
    # Enable collaborative mode
    print("\nEnabling collaborative mode...")
    system.enable_collaborative_mode(team_size=5)
    
    # Add simulated team members
    print("\nAdding team members...")
    for i in range(3):
        member_id = system.team_interface.add_member(name=f"Member_{i+1}")
        system.team_interface.set_contribution_weight(member_id, 0.5 + i * 0.1)
        
    # Connect to team network
    print("\nConnecting to team network...")
    system.team_interface.connect()
    
    # Show team status
    team_status = system.team_interface.get_team_status()
    print("\nTeam Status:")
    print(f"  Active Members: {team_status['active_members']}/{team_status['team_size']}")
    print(f"  Team Coherence: {team_status['team_coherence']:.3f}")
    
    # Share field
    print("\nSharing field with team...")
    system.team_interface.share_field()
    
    # Show global network if enabled
    if hasattr(system, 'global_network'):
        print("\nConnecting to global network...")
        system.global_network.connect()
        
        # Show global network status
        network_status = system.global_network.get_network_status()
        print("\nGlobal Network Status:")
        print(f"  Active Nodes: {network_status['active_nodes']}/{network_status['total_nodes']}")
        print(f"  Participants: {network_status['total_participants']}")
        print(f"  Network Coherence: {network_status['network_coherence']:.3f}")
        
        # Contribute to global network
        print("\nContributing to global network...")
        system.global_network.contribute_field()
        
    print("\nDemonstration complete.")
    
def run_interactive_mode(system):
    """Run the system in interactive mode."""
    print("\nStarting CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK in interactive mode.")
    print("Type 'help' for a list of commands, or 'exit' to quit.")
    
    # Start the system
    system.start()
    
    # Interactive command loop
    while True:
        try:
            cmd = input("\ncascade> ").strip().lower()
            
            if cmd == 'exit' or cmd == 'quit':
                break
                
            elif cmd == 'help':
                print("\nAvailable commands:")
                print("  help              - Show this help message")
                print("  info              - Show system information")
                print("  field             - Show field information")
                print("  evolve [steps]    - Evolve the field")
                print("  state <state>     - Set consciousness state")
                print("  broadcast start   - Start broadcasting")
                print("  broadcast stop    - Stop broadcasting")
                print("  record [seconds]  - Record field state")
                print("  team connect      - Connect to team network")
                print("  team disconnect   - Disconnect from team network")
                print("  global connect    - Connect to global network")
                print("  global disconnect - Disconnect from global network")
                print("  exit/quit         - Exit the system")
                
            elif cmd == 'info':
                print("\nSystem Information:")
                print(f"  Version:        {__version__}")
                print(f"  Running:        {system.status['running']}")
                print(f"  Field Active:   {system.status['field_active']}")
                print(f"  Broadcasting:   {system.status['broadcasting']}")
                print(f"  Hardware:       {', '.join(system.status['connected_hardware']) if system.status['connected_hardware'] else 'None'}")
                print(f"  Backends:       {', '.join(system.status['active_backends'])}")
                print(f"  Collaborative:  {system.status['collaborative_active']}")
                
            elif cmd == 'field':
                if system.quantum_field:
                    print_field_info(system.quantum_field)
                else:
                    print("No active field. Initializing...")
                    system.initialize_field()
                    print_field_info(system.quantum_field)
                    
            elif cmd.startswith('evolve'):
                parts = cmd.split()
                steps = int(parts[1]) if len(parts) > 1 else 1
                
                if system.quantum_field:
                    print(f"Evolving field for {steps} steps...")
                    system.quantum_field.evolve(steps=steps)
                    print(f"New coherence: {system.quantum_field.coherence:.3f}")
                else:
                    print("No active field. Initialize first.")
                    
            elif cmd.startswith('state'):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: state <state_name>")
                    continue
                    
                state = parts[1]
                
                if not hasattr(system, 'consciousness_interface') or system.consciousness_interface is None:
                    print("Adding consciousness interface...")
                    system.add_consciousness_interface()
                    
                print(f"Setting consciousness state to {state}...")
                success = system.consciousness_interface.set_state(state, intensity=0.8)
                
                if not success:
                    print(f"Invalid state: {state}")
                    print("Available states: delta, theta, alpha, beta, gamma, lambda, epsilon")
                    
            elif cmd == 'broadcast start':
                if not system.config['broadcast_enabled']:
                    print("Enabling broadcasting...")
                    system.config['broadcast_enabled'] = True
                    
                if not hasattr(system, 'broadcast_engine') or system.broadcast_engine is None:
                    print("Setting up broadcast engine...")
                    system.setup_broadcasting()
                    
                print("Starting broadcasting...")
                system.broadcast_engine.start_broadcasting()
                system.status['broadcasting'] = True
                
            elif cmd == 'broadcast stop':
                if hasattr(system, 'broadcast_engine') and system.broadcast_engine is not None:
                    print("Stopping broadcasting...")
                    system.broadcast_engine.stop_broadcasting()
                    system.status['broadcasting'] = False
                else:
                    print("Broadcasting not active.")
                    
            elif cmd.startswith('record'):
                parts = cmd.split()
                duration = float(parts[1]) if len(parts) > 1 else 10.0
                
                if hasattr(system, 'broadcast_engine') and system.broadcast_engine is not None:
                    print(f"Recording field state for {duration} seconds...")
                    name = system.broadcast_engine.record_field_state(duration=duration)
                    print(f"Recording saved as {name}")
                else:
                    print("Broadcasting not active. Start broadcasting first.")
                    
            elif cmd == 'team connect':
                if not hasattr(system, 'team_interface') or system.team_interface is None:
                    print("Enabling collaborative mode...")
                    system.enable_collaborative_mode()
                    
                print("Connecting to team network...")
                system.team_interface.connect()
                
                # Show team status
                team_status = system.team_interface.get_team_status()
                print("\nTeam Status:")
                print(f"  Active Members: {team_status['active_members']}/{team_status['team_size']}")
                print(f"  Team Coherence: {team_status['team_coherence']:.3f}")
                
            elif cmd == 'team disconnect':
                if hasattr(system, 'team_interface') and system.team_interface is not None:
                    print("Disconnecting from team network...")
                    system.team_interface.disconnect()
                else:
                    print("Team interface not active.")
                    
            elif cmd == 'global connect':
                if not hasattr(system, 'global_network') or system.global_network is None:
                    if not hasattr(system, 'team_interface') or system.team_interface is None:
                        print("Enabling collaborative mode...")
                        system.enable_collaborative_mode()
                        
                    from .collaborative import GlobalNetwork
                    print("Creating global network interface...")
                    system.global_network = GlobalNetwork(system.team_interface)
                    
                print("Connecting to global network...")
                system.global_network.connect()
                
                # Show global network status
                network_status = system.global_network.get_network_status()
                print("\nGlobal Network Status:")
                print(f"  Active Nodes: {network_status['active_nodes']}/{network_status['total_nodes']}")
                print(f"  Network Coherence: {network_status['network_coherence']:.3f}")
                
            elif cmd == 'global disconnect':
                if hasattr(system, 'global_network') and system.global_network is not None:
                    print("Disconnecting from global network...")
                    system.global_network.disconnect()
                else:
                    print("Global network not active.")
                    
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for a list of commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            
    # Stop the system
    print("\nStopping CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK...")
    system.stop()
    
def main():
    """Main entry point for the CASCADE unified system."""
    # Parse command-line arguments
    args = parse_args()
    
    # Show version and exit if requested
    if args.version:
        print(f"CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK v{__version__}")
        return
        
    # Print banner
    print_banner()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Apply phi-scaling to dimensions if requested
    if args.phi_scaling:
        dims = args.dimensions.copy()
        for i in range(1, len(dims)):
            dims[i] = int(dims[0] * (PHI ** i))
        print(f"Phi-scaled dimensions: {dims}")
    else:
        dims = args.dimensions
        
    # Configure the system
    config = {
        'field_dimensions': dims,
        'base_frequency': args.frequency,
        'coherence_target': args.coherence,
        'visualization_enabled': args.visualize,
        'broadcast_enabled': args.broadcast,
        'collaborative_mode': args.collaborative,
        'language_backends': args.backends.split(',') if args.backends else ['python']
    }
    
    # Create the system
    system = CascadeSystem(config)
    
    # Register language backends
    for backend in config['language_backends']:
        system.register_language_backend(backend)
        
    # Add hardware interfaces if specified
    if args.hardware:
        hardware_interfaces = args.hardware.split(',')
        for hw in hardware_interfaces:
            system.add_hardware_interface(hw.strip())
            
    # Set up broadcasting if enabled
    if args.broadcast:
        channels = args.channels.split(',')
        system.setup_broadcasting(channels=channels)
        
        # Configure OBS connection if using video
        if 'video' in channels:
            if hasattr(system, 'broadcast_engine'):
                system.broadcast_engine.connect_to_obs(
                    address=args.obs_address,
                    port=args.obs_port
                )
                
    # Set up collaborative mode if enabled
    if args.collaborative:
        system.enable_collaborative_mode(team_size=args.team_size)
        
        # Connect to global network if requested
        if args.session == 'global':
            from .collaborative import GlobalNetwork
            system.global_network = GlobalNetwork(system.team_interface)
            
    # Run in the appropriate mode
    if args.interactive:
        run_interactive_mode(system)
    elif args.demo:
        if args.demo == 'cascade' or args.demo == 'all':
            run_cascade_demo(system)
        if args.demo == 'teams' or args.demo == 'all':
            run_teams_demo(system)
    else:
        # Start the system
        system.start(visualize=args.visualize, broadcast=args.broadcast)
        
        # Keep the system running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK...")
        finally:
            system.stop()
            
if __name__ == '__main__':
    main()