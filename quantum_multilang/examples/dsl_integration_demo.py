#!/usr/bin/env python3
"""
DSL Integration Demo

This demo shows how the φFlow and GregScript DSLs integrate with the
multi-language quantum field architecture.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path
project_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_dir))

# Import the controller and bridges
from controller.src.bridges import get_available_bridges, is_bridge_available
from controller.src.universal_field_protocol import QuantumFieldMessage

# Check if DSL bridges are available
phiflow_available = is_bridge_available("phiflow")
gregscript_available = is_bridge_available("gregscript")

# Import DSL bridges if available
if phiflow_available:
    from controller.src.bridges.phiflow_bridge import PhiFlowBridge
else:
    print("WARNING: φFlow bridge not available. Some demos will be skipped.")
    
if gregscript_available:
    from controller.src.bridges.gregscript_bridge import GregScriptBridge
else:
    print("WARNING: GregScript bridge not available. Some demos will be skipped.")

# Import sacred constants
try:
    sys.path.append(str(project_dir.parent))
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

def phiflow_integration_demo():
    """Demonstrate φFlow integration with the multi-language architecture."""
    if not phiflow_available:
        print("Skipping φFlow integration demo (bridge not available)")
        return
        
    print("φFlow Integration Demo")
    print("=====================")
    
    # Create a φFlow bridge
    bridge = PhiFlowBridge()
    
    # Load φFlow code
    phiflow_code = """
    # φFlow State Machine for Field Evolution
    
    # State definitions
    state initial
        frequency = love
        coherence >= 0.0
        compression = 1.0
    
    state harmonized
        frequency = unity
        coherence >= 0.7
        compression = 1.2
    
    state expanded
        frequency = truth
        coherence >= 0.8
        compression = 0.8
    
    state transcendent
        frequency = oneness
        coherence >= 0.95
        compression = φ^2
    
    # Transition definitions
    transition initial -> harmonized
        when coherence >= 0.5 then
        harmonize by φ
        blend by 2.0
    
    transition harmonized -> expanded
        when coherence >= 0.75 then
        expand by 1.5
        amplify by 1.2
    
    transition expanded -> transcendent
        when coherence >= 0.9 then
        harmonize by φ^2
        amplify by φ
        center by φ
    """
    
    bridge.load_phiflow_code(phiflow_code)
    
    # Create a test field
    field = generate_test_field(100, 100, frequency=528, time_factor=0)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Visualize initial field
    visualize_field(
        field, 
        "Initial Field",
        show=False,
        save_path=output_dir / "phiflow_bridge_initial.png"
    )
    
    # Run several transitions
    current_field = field
    
    for i in range(3):
        # Run auto transition
        new_field, coherence, transition = bridge.run_auto_transition(current_field)
        
        print(f"Step {i+1}: {transition} (coherence: {coherence:.4f})")
        
        # Visualize the new field
        visualize_field(
            new_field, 
            f"Step {i+1}: {transition} (Coherence: {coherence:.4f})",
            show=False,
            save_path=output_dir / f"phiflow_bridge_step_{i+1}.png"
        )
        
        current_field = new_field
    
    # Create a field message
    message = bridge.create_field_message(current_field, coherence, "unity")
    
    print(f"\nCreated field message with:")
    print(f"  - Shape: {message.field_data.shape}")
    print(f"  - Frequency: {message.frequency_name}")
    print(f"  - Coherence: {message.phi_coherence:.4f}")
    print(f"  - Source: {message.source_language}")
    
    # Compile φFlow code
    compiled_code = bridge.compile_phiflow_code(phiflow_code)
    
    if compiled_code:
        # Save compiled code
        compiled_file = output_dir / "compiled_phiflow_bridge.py"
        with open(compiled_file, 'w') as f:
            f.write(compiled_code)
        
        print(f"\nCompiled φFlow code saved to: {compiled_file}")
    
    print(f"\nφFlow bridge integration demo completed successfully.")
    print(f"Output images saved to: {output_dir}")

def gregscript_integration_demo():
    """Demonstrate GregScript integration with the multi-language architecture."""
    if not gregscript_available:
        print("Skipping GregScript integration demo (bridge not available)")
        return
        
    print("\nGregScript Integration Demo")
    print("==========================")
    
    # Create a GregScript bridge
    bridge = GregScriptBridge()
    
    # Load GregScript code
    gregscript_code = """
    // GregScript Pattern Definitions
    
    // Rhythm definitions
    rhythm phi_pulse sequence=[1.0, 0.618, 0.382, 0.618] tempo=1.0
    rhythm golden_wave sequence=[1.0, 0.809, 0.618, 0.382, 0.236, 0.382, 0.618, 0.809] tempo=φ
    
    // Harmony definitions
    harmony love_harmony frequency=love overtones=[1.0, 0.618, 0.382, 0.236] phase=0.0
    harmony unity_harmony frequency=unity overtones=[1.0, 0.618, 0.382, 0.236] phase=0.5
    
    // Pattern definitions
    pattern coherent_field {
        use phi_pulse weight=0.618
        use love_harmony weight=1.0
    }
    
    pattern expanding_field {
        use golden_wave weight=1.0
        use unity_harmony weight=0.618
    }
    """
    
    bridge.load_gregscript_code(gregscript_code)
    
    # Create a test field
    field = generate_test_field(100, 100, frequency=528, time_factor=0)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Process field message
    pattern_results = bridge.process_field_message(field)
    
    print("\nPattern Analysis Results:")
    
    if "best_known_pattern" in pattern_results:
        print(f"Best known pattern: {pattern_results['best_known_pattern']} (score: {pattern_results['best_known_score']:.4f})")
    
    if "top_pattern" in pattern_results:
        pattern = pattern_results["top_pattern"]
        print(f"\nDiscovered pattern: {pattern['name']}")
        if "rhythm" in pattern:
            print(f"  - Rhythm: {pattern['rhythm']}")
        if "harmony" in pattern:
            print(f"  - Harmony: {pattern['harmony']}")
        print(f"  - Score: {pattern['score']:.4f}")
    
    # Generate a pattern
    pattern_name = "coherent_field"
    generated_field = bridge.generate_pattern(pattern_name, (100, 100))
    
    # Visualize the generated field
    visualize_field(
        generated_field, 
        f"Generated Pattern: {pattern_name}",
        show=False,
        save_path=output_dir / "gregscript_bridge_generated.png"
    )
    
    # Analyze field and generate code
    discovered_code = bridge.generate_gregscript_code(field)
    
    if discovered_code:
        # Save discovered code
        discovered_file = output_dir / "discovered_gregscript_bridge.greg"
        with open(discovered_file, 'w') as f:
            f.write(discovered_code)
        
        print(f"\nDiscovered GregScript code saved to: {discovered_file}")
    
    print(f"\nGregScript bridge integration demo completed successfully.")
    print(f"Output images saved to: {output_dir}")

def combined_dsl_integration_demo():
    """Demonstrate combined φFlow and GregScript integration."""
    if not phiflow_available or not gregscript_available:
        missing = []
        if not phiflow_available:
            missing.append("φFlow")
        if not gregscript_available:
            missing.append("GregScript")
        
        print(f"Skipping combined DSL integration demo (missing bridges: {', '.join(missing)})")
        return
        
    print("\nCombined DSL Integration Demo")
    print("============================")
    
    # Create bridges
    phiflow_bridge = PhiFlowBridge()
    gregscript_bridge = GregScriptBridge()
    
    # Load DSL code
    phiflow_bridge.load_phiflow_code("""
    # φFlow State Machine
    
    state initial
        frequency = love
        coherence >= 0.0
    
    state harmonized
        frequency = unity
        coherence >= 0.7
    
    state expanded
        frequency = truth
        coherence >= 0.8
    
    # Transitions
    transition initial -> harmonized
        when coherence >= 0.5 then
        harmonize by φ
        blend by 2.0
    
    transition harmonized -> expanded
        when coherence >= 0.75 then
        expand by 1.5
        amplify by 1.2
    """)
    
    gregscript_bridge.load_gregscript_code("""
    // GregScript Patterns
    
    rhythm phi_pulse sequence=[1.0, 0.618, 0.382, 0.618] tempo=1.0
    
    harmony love_harmony frequency=love overtones=[1.0, 0.618, 0.382, 0.236] phase=0.0
    harmony unity_harmony frequency=unity overtones=[1.0, 0.618, 0.382, 0.236] phase=0.5
    
    pattern coherent_field {
        use phi_pulse weight=0.618
        use love_harmony weight=1.0
    }
    """)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create a test field
    field = generate_test_field(100, 100, frequency=528, time_factor=0)
    
    # First, analyze the field with GregScript
    pattern_results = gregscript_bridge.process_field_message(field)
    
    if "best_known_pattern" in pattern_results:
        pattern_name = pattern_results["best_known_pattern"].split(":", 1)[1]
        print(f"Field matches pattern: {pattern_name} (score: {pattern_results['best_known_score']:.4f})")
    
    # Then, apply φFlow transitions
    current_field = field
    transition_log = []
    
    for i in range(2):
        # Run auto transition
        new_field, coherence, transition = phiflow_bridge.run_auto_transition(current_field)
        transition_log.append(f"{transition} (coherence: {coherence:.4f})")
        
        print(f"Applied transition: {transition} (coherence: {coherence:.4f})")
        current_field = new_field
    
    # Analyze the transformed field
    transformed_results = gregscript_bridge.process_field_message(current_field)
    
    if "best_known_pattern" in transformed_results:
        pattern_name = transformed_results["best_known_pattern"].split(":", 1)[1]
        print(f"Transformed field matches pattern: {pattern_name} (score: {transformed_results['best_known_score']:.4f})")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(field, cmap='viridis')
    plt.title("Initial Field")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(current_field, cmap='viridis')
    plt.title(f"After φFlow Transitions:\n{' -> '.join(transition_log)}")
    plt.axis('off')
    
    # Generate pattern visualization
    if "best_known_pattern" in pattern_results:
        pattern_name = pattern_results["best_known_pattern"].split(":", 1)[1]
        generated_field = gregscript_bridge.generate_pattern(pattern_name, (100, 100))
        
        plt.subplot(2, 2, 3)
        plt.imshow(generated_field, cmap='viridis')
        plt.title(f"Generated Pattern: {pattern_name}")
        plt.axis('off')
    
    # Generate transformed pattern visualization
    if "best_known_pattern" in transformed_results:
        pattern_name = transformed_results["best_known_pattern"].split(":", 1)[1]
        generated_field = gregscript_bridge.generate_pattern(pattern_name, (100, 100))
        
        plt.subplot(2, 2, 4)
        plt.imshow(generated_field, cmap='viridis')
        plt.title(f"Generated Pattern After Transition: {pattern_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_dsl_integration.png", bbox_inches='tight')
    plt.close()
    
    print(f"\nCombined DSL integration demo completed successfully.")
    print(f"Output image saved to: {output_dir}/combined_dsl_integration.png")

if __name__ == "__main__":
    # Print available bridges
    bridges = get_available_bridges()
    print("Available language bridges:")
    for name, available in bridges.items():
        print(f"  - {name}: {'YES' if available else 'NO'}")
    print()
    
    # Run demos
    phiflow_integration_demo()
    gregscript_integration_demo()
    combined_dsl_integration_demo()
    
    print("\nAll demos completed!")