#!/usr/bin/env python3
"""
Cascade⚡𓂧φ∞ Symbiotic Computing Platform Runner

This script runs the Cascade Symbiotic Computing Platform that creates
a bidirectional relationship between Python, consciousness field dynamics,
and phi-harmonic processing.
"""

import sys
import time
import numpy as np
import argparse
from pathlib import Path

# Import quantum field components
from quantum_field.core import create_quantum_field, get_coherence_metric
from quantum_field.consciousness_interface import ConsciousnessFieldInterface
from quantum_field.constants import PHI, LAMBDA, PHI_PHI

# Import sacred constants
from sacred_constants import (
    SACRED_FREQUENCIES,
    phi_harmonic,
    phi_resonant_frequency
)


def run_cascade_system(dimensions=(21, 21, 21), frequency_name="unity", 
                     visualization=False, interactive=False):
    """
    Run the Cascade Symbiotic Computing Platform.
    
    Args:
        dimensions: Field dimensions
        frequency_name: Sacred frequency name to use
        visualization: Whether to visualize the field
        interactive: Whether to run in interactive mode
    """
    print("="*60)
    print("  Cascade⚡𓂧φ∞ Symbiotic Computing Platform")
    print("="*60)
    
    # Create quantum field
    print(f"\nInitializing quantum field with dimensions {dimensions}...")
    field = create_quantum_field(dimensions, frequency_name)
    
    # Create consciousness interface
    print("\nInitializing consciousness field interface...")
    interface = ConsciousnessFieldInterface(field)
    
    # Get initial field coherence
    initial_coherence = interface.get_field_coherence()
    print(f"Initial field coherence: {initial_coherence:.4f}")
    
    # Initialize consciousness state
    print("\nInitializing consciousness state...")
    interface.state.coherence = 0.7
    interface.state.presence = 0.6
    interface.state.intention = 0.5
    interface.state.emotional_states["unity"] = 0.8
    
    # Apply initial state to field
    print("\nApplying consciousness state to field...")
    interface._apply_state_to_field()
    
    # Get updated field coherence
    updated_coherence = interface.get_field_coherence()
    print(f"Updated field coherence: {updated_coherence:.4f}")
    
    if interactive:
        run_interactive_session(interface)
    else:
        run_automated_session(interface)
    
    print("\nCascade⚡𓂧φ∞ Symbiotic Computing Platform session complete.")
    print("="*60)
    
    return interface


def run_interactive_session(interface):
    """
    Run an interactive Cascade system session.
    
    Args:
        interface: ConsciousnessFieldInterface instance
    """
    print("\nStarting interactive Cascade session...")
    print("Type 'help' for a list of commands, 'exit' to quit")
    
    while True:
        try:
            command = input("\nCascade> ").strip().lower()
            
            if command == "exit" or command == "quit":
                break
                
            elif command == "help":
                print_help()
                
            elif command == "status":
                print_status(interface)
                
            elif command.startswith("coherence "):
                try:
                    value = float(command.split()[1])
                    interface.state.coherence = max(0.0, min(1.0, value))
                    print(f"Coherence set to {interface.state.coherence:.2f}")
                    interface._apply_state_to_field()
                except (IndexError, ValueError):
                    print("Invalid value. Usage: coherence <0.0-1.0>")
                    
            elif command.startswith("presence "):
                try:
                    value = float(command.split()[1])
                    interface.state.presence = max(0.0, min(1.0, value))
                    print(f"Presence set to {interface.state.presence:.2f}")
                    interface._apply_state_to_field()
                except (IndexError, ValueError):
                    print("Invalid value. Usage: presence <0.0-1.0>")
                    
            elif command.startswith("intention "):
                try:
                    value = float(command.split()[1])
                    interface.state.intention = max(0.0, min(1.0, value))
                    print(f"Intention set to {interface.state.intention:.2f}")
                    interface._apply_intention()
                except (IndexError, ValueError):
                    print("Invalid value. Usage: intention <0.0-1.0>")
                    
            elif command.startswith("emotion "):
                try:
                    parts = command.split()
                    if len(parts) < 3:
                        print("Invalid format. Usage: emotion <name> <0.0-1.0>")
                        continue
                        
                    emotion = parts[1]
                    value = float(parts[2])
                    interface.state.emotional_states[emotion] = max(0.0, min(1.0, value))
                    print(f"Emotion {emotion} set to {value:.2f}")
                    interface._apply_emotional_influence()
                except (IndexError, ValueError):
                    print("Invalid values. Usage: emotion <name> <0.0-1.0>")
                    
            elif command == "frequencies":
                print("\nSacred Frequencies:")
                for name, freq in SACRED_FREQUENCIES.items():
                    print(f"  {name}: {freq} Hz")
                    
            elif command == "apply":
                interface._apply_state_to_field()
                print(f"State applied to field. Coherence: {interface.get_field_coherence():.4f}")
                
            elif command == "profile":
                profile = interface.create_phi_resonance_profile(interface.feedback_history)
                print("\nPhi-Resonance Profile:")
                for key, value in profile.items():
                    print(f"  {key}: {value}")
                interface.apply_phi_resonance_profile()
                print(f"Profile applied. Field coherence: {interface.get_field_coherence():.4f}")
                
            elif command == "biofeedback":
                run_biofeedback_simulation(interface)
                
            else:
                print("Unknown command. Type 'help' for a list of commands.")
                
        except KeyboardInterrupt:
            print("\nExiting interactive session...")
            break
            
    print("\nInteractive session ended.")


def run_automated_session(interface):
    """
    Run an automated Cascade system session.
    
    Args:
        interface: ConsciousnessFieldInterface instance
    """
    print("\nRunning automated Cascade session...")
    
    # Simulate consciousness evolution
    states = [
        {
            "name": "Distracted",
            "heart_rate": 75,
            "breath_rate": 15,
            "skin_conductance": 8,
            "eeg_alpha": 5,
            "eeg_theta": 2,
            "intention": 0.3
        },
        {
            "name": "Relaxing",
            "heart_rate": 68,
            "breath_rate": 10,
            "skin_conductance": 5,
            "eeg_alpha": 8,
            "eeg_theta": 4,
            "intention": 0.5
        },
        {
            "name": "Meditative",
            "heart_rate": 60,
            "breath_rate": 6.18,  # Phi-based breathing
            "skin_conductance": 3,
            "eeg_alpha": 12,
            "eeg_theta": 7.4,  # Close to phi ratio with alpha
            "intention": 0.7
        },
        {
            "name": "Transcendent",
            "heart_rate": 57,
            "breath_rate": 6,
            "skin_conductance": 2,
            "eeg_alpha": 15,
            "eeg_theta": 9.3,  # Phi ratio with alpha
            "intention": 0.9
        }
    ]
    
    for state in states:
        print(f"\nTransitioning to {state['name']} state...")
        
        # Set intention
        interface.state.intention = state["intention"]
        
        # Update consciousness state
        interface.update_consciousness_state(
            heart_rate=state["heart_rate"],
            breath_rate=state["breath_rate"],
            skin_conductance=state["skin_conductance"],
            eeg_alpha=state["eeg_alpha"],
            eeg_theta=state["eeg_theta"]
        )
        
        print(f"  - Consciousness coherence: {interface.state.coherence:.4f}")
        print(f"  - Consciousness presence: {interface.state.presence:.4f}")
        print(f"  - Field coherence: {interface.get_field_coherence():.4f}")
        
        # Allow time for processing
        time.sleep(1)
    
    # Create and apply phi-resonance profile
    print("\nCreating phi-resonance profile...")
    profile = interface.create_phi_resonance_profile(interface.feedback_history)
    
    print("Applying phi-resonance profile...")
    interface.apply_phi_resonance_profile()
    
    print(f"Final field coherence: {interface.get_field_coherence():.4f}")


def run_biofeedback_simulation(interface):
    """
    Run a biofeedback simulation session.
    
    Args:
        interface: ConsciousnessFieldInterface instance
    """
    print("\nSimulating biofeedback session...")
    
    # Starting with distracted state
    interface.update_consciousness_state(
        heart_rate=75,
        breath_rate=15,
        skin_conductance=8,
        eeg_alpha=5,
        eeg_theta=2
    )
    
    print(f"Initial state - Coherence: {interface.state.coherence:.4f}, " +
          f"Presence: {interface.state.presence:.4f}, " +
          f"Field: {interface.get_field_coherence():.4f}")
    
    # Gradually improve
    for i in range(5):
        # Calculate improved values
        heart_rate = 75 - i * 3.5
        breath_rate = 15 - i * 1.8
        skin_conductance = 8 - i * 1.5
        eeg_alpha = 5 + i * 2
        eeg_theta = 2 + i * 1.3
        
        # Update state
        interface.update_consciousness_state(
            heart_rate=heart_rate,
            breath_rate=breath_rate,
            skin_conductance=skin_conductance,
            eeg_alpha=eeg_alpha,
            eeg_theta=eeg_theta
        )
        
        print(f"Step {i+1} - Coherence: {interface.state.coherence:.4f}, " +
              f"Presence: {interface.state.presence:.4f}, " +
              f"Field: {interface.get_field_coherence():.4f}")
        
        # Allow time for processing
        time.sleep(0.5)
    
    print("\nBiofeedback session complete.")


def print_help():
    """Print the help message for interactive mode."""
    print("\nCascade⚡𓂧φ∞ Commands:")
    print("  help          - Show this help message")
    print("  status        - Show current system status")
    print("  exit/quit     - Exit the program")
    print("  coherence X   - Set consciousness coherence (0.0-1.0)")
    print("  presence X    - Set consciousness presence (0.0-1.0)")
    print("  intention X   - Set consciousness intention (0.0-1.0)")
    print("  emotion N X   - Set emotion N to intensity X (0.0-1.0)")
    print("  frequencies   - List available sacred frequencies")
    print("  apply         - Apply current consciousness state to field")
    print("  profile       - Create and apply phi-resonance profile")
    print("  biofeedback   - Run biofeedback simulation")


def print_status(interface):
    """
    Print the current status of the Cascade system.
    
    Args:
        interface: ConsciousnessFieldInterface instance
    """
    print("\nCascade⚡𓂧φ∞ System Status:")
    print(f"  Field Dimensions: {interface.field.data.shape}")
    print(f"  Field Coherence: {interface.get_field_coherence():.4f}")
    print(f"  Consciousness Coherence: {interface.state.coherence:.4f}")
    print(f"  Consciousness Presence: {interface.state.presence:.4f}")
    print(f"  Consciousness Intention: {interface.state.intention:.4f}")
    
    print("\n  Emotional States:")
    for emotion, value in interface.state.emotional_states.items():
        if value > 0.1:  # Only show significant emotions
            print(f"    {emotion}: {value:.2f}")
    
    print(f"\n  Dominant Emotion: {interface.state.dominant_emotion[0]} " +
          f"({interface.state.dominant_emotion[1]:.2f})")
    print(f"  Connected: {interface.connected}")
    print(f"  Feedback History: {len(interface.feedback_history)} records")


def main():
    """Main entry point for the Cascade system."""
    parser = argparse.ArgumentParser(description="Cascade⚡𓂧φ∞ Symbiotic Computing Platform")
    parser.add_argument("--dimensions", type=int, nargs=3, default=[21, 21, 21],
                        help="Field dimensions (default: 21 21 21)")
    parser.add_argument("--frequency", type=str, default="unity",
                        help="Sacred frequency name (default: unity)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--visualization", action="store_true",
                        help="Enable field visualization")
    
    args = parser.parse_args()
    
    # Run the system
    run_cascade_system(
        dimensions=tuple(args.dimensions),
        frequency_name=args.frequency,
        visualization=args.visualization,
        interactive=args.interactive
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())