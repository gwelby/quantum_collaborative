#!/usr/bin/env python3
"""
φFlow Demo - Quantum Field State Transitions

This demo shows how to use the φFlow DSL to define and execute
state transitions for quantum fields.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import phiflow
parent_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import the phiflow module
from languages.phiflow.src import parse_phiflow, PhiFlowInterpreter, compile_phiflow

# Import sacred constants
try:
    sys.path.append(str(parent_dir.parent))
    import sacred_constants as sc
    PHI = sc.PHI
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895

def generate_test_field(width, height, frequency=528, time_factor=0):
    """Generate a test quantum field."""
    field = np.zeros((height, width), dtype=np.float32)
    
    # Scale frequency
    freq_factor = frequency / 1000.0 * PHI
    
    # Calculate the center of the field
    center_x = width / 2
    center_y = height / 2
    
    # Generate the field values
    for y in range(height):
        for x in range(width):
            # Calculate distance from center (normalized)
            dx = (x - center_x) / (width / 2)
            dy = (y - center_y) / (height / 2)
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate the field value using phi-harmonics
            angle = np.arctan2(dy, dx) * PHI
            time_value = time_factor * (1.0 / PHI)
            
            # Create an interference pattern
            value = (
                np.sin(distance * freq_factor + time_value) * 
                np.cos(angle * PHI) * 
                np.exp(-distance / PHI)
            )
            
            field[y, x] = value
    
    return field

def visualize_field(field, title="Quantum Field", show=True, save_path=None):
    """Visualize a quantum field."""
    plt.figure(figsize=(10, 8))
    plt.imshow(field, cmap='viridis')
    plt.colorbar(label='Field Value')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def interpret_phiflow_demo():
    """Demonstrate using the φFlow interpreter."""
    print("φFlow Interpreter Demo")
    print("=====================")
    
    # Load φFlow code
    phi_file = Path(__file__).parent / "field_state_machine.phi"
    with open(phi_file, 'r') as f:
        phiflow_code = f.read()
    
    # Parse the code
    states, transitions = parse_phiflow(phiflow_code)
    
    print(f"Parsed {len(states)} states and {len(transitions)} transitions")
    
    # List states
    print("\nStates:")
    for name, props in states.items():
        freq = props.get("frequency", "love")
        coherence = props.get("min_coherence", 0.0)
        compression = props.get("compression", 1.0)
        print(f"  {name}: freq={freq}, coherence>={coherence}, compression={compression}")
    
    # List transitions
    print("\nTransitions:")
    for transition in transitions:
        ops = ", ".join([op.operation_type for op in transition.operations])
        cond = f" when {transition.condition}" if transition.condition else ""
        print(f"  {transition.from_state} -> {transition.to_state}{cond} [{ops}]")
    
    # Create a test field
    field = generate_test_field(100, 100)
    
    # Create an interpreter
    interpreter = PhiFlowInterpreter()
    interpreter.load(phiflow_code)
    interpreter.set_field(field)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Visualize initial field
    visualize_field(
        interpreter.field_data, 
        f"Initial Field (State: {interpreter.current_state}, Coherence: {interpreter.coherence:.4f})",
        show=False,
        save_path=output_dir / "phiflow_initial.png"
    )
    
    # Run some transitions
    for i in range(5):
        # Get available transitions
        available = interpreter.get_available_transitions()
        if not available:
            print(f"Step {i+1}: No available transitions")
            break
        
        # Select the first transition
        selected = available[0]
        print(f"Step {i+1}: Applying transition {selected}")
        
        # Apply the transition
        new_field, coherence = interpreter.apply_transition(selected)
        
        # Visualize the new field
        visualize_field(
            new_field, 
            f"Step {i+1}: {selected.from_state} -> {selected.to_state} (Coherence: {coherence:.4f})",
            show=False,
            save_path=output_dir / f"phiflow_step_{i+1}.png"
        )
    
    print(f"\nFinal state: {interpreter.current_state}")
    print(f"Final coherence: {interpreter.coherence:.4f}")
    print(f"Output images saved to: {output_dir}")

def compiled_phiflow_demo():
    """Demonstrate using the compiled φFlow code."""
    print("\nCompiled φFlow Demo")
    print("===================")
    
    # Load φFlow code
    phi_file = Path(__file__).parent / "field_state_machine.phi"
    with open(phi_file, 'r') as f:
        phiflow_code = f.read()
    
    # Compile the code
    python_code = compile_phiflow(phiflow_code)
    
    # Save the compiled code
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    compiled_file = output_dir / "compiled_phiflow.py"
    with open(compiled_file, 'w') as f:
        f.write(python_code)
    
    print(f"Compiled φFlow code saved to: {compiled_file}")
    
    # Create a test field
    field = generate_test_field(100, 100)
    
    # Create output dir for compiled output
    compiled_output_dir = output_dir / "compiled"
    compiled_output_dir.mkdir(exist_ok=True)
    
    # Execute the compiled code (in this case, we'd typically import it)
    # For demo purposes, we'll use the interpreter again
    interpreter = PhiFlowInterpreter()
    interpreter.load(phiflow_code)
    interpreter.set_field(field)
    
    # Run auto transitions
    print("\nRunning compiled state machine:")
    for i in range(5):
        transition = interpreter.run_auto_transition()
        if not transition:
            print(f"Step {i+1}: No available transitions")
            break
        
        print(f"Step {i+1}: Applied {transition} (coherence: {interpreter.coherence:.4f})")
        
        # Visualize the field
        visualize_field(
            interpreter.field_data, 
            f"Compiled Step {i+1}: State = {interpreter.current_state} (Coherence: {interpreter.coherence:.4f})",
            show=False,
            save_path=compiled_output_dir / f"compiled_step_{i+1}.png"
        )
    
    print(f"\nFinal state: {interpreter.current_state}")
    print(f"Final coherence: {interpreter.coherence:.4f}")
    print(f"Compiled output images saved to: {compiled_output_dir}")

if __name__ == "__main__":
    interpret_phiflow_demo()
    compiled_phiflow_demo()
    print("\nDone! Check the output directory for visualization images.")