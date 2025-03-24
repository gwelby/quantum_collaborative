#!/usr/bin/env python3
"""
CASCADE⚡𓂧φ∞ Quantum Pattern Recognition Demo

This script demonstrates the pattern recognition capabilities of the CASCADE
quantum field system, including pattern matching, field state recognition, and
phi-harmonic pattern analysis.
"""

import sys
import os
import time
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import CASCADE components
# First try to import components directly, then handle potential import failures
try:
    from cascade.pattern_recognition import QuantumPatternEngine, PhiPatternMatcher, FieldStateRecognizer
    from cascade.core.toroidal_field import ToroidalFieldEngine
    from cascade.core.consciousness_bridge import ConsciousnessBridgeProtocol
    
    COMPONENTS_LOADED = True
except ImportError as e:
    print(f"Warning: Failed to import some CASCADE components: {e}")
    print("Falling back to minimal imports")
    COMPONENTS_LOADED = False

# Constants for when modules aren't available
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
}

# Optional visualization support
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Running without visualizations.")
    VISUALIZATION_AVAILABLE = False


def create_test_fields() -> Dict[str, np.ndarray]:
    """Create test fields for pattern recognition."""
    fields = {}
    
    # Generate using toroidal field engine if available
    if COMPONENTS_LOADED:
        try:
            toroidal_engine = ToroidalFieldEngine()
            
            # Ground state (432 Hz)
            field = toroidal_engine.generate_field(32, 32, 32)
            fields['ground_state'] = field
            
            # Create consciousness bridge
            bridge = ConsciousnessBridgeProtocol()
            bridge.connect_field(field.copy())
            bridge.start_protocol()
            
            # Progress through stages
            for stage in range(1, 7):
                bridge.progress_to_stage(stage)
                fields[f'stage_{stage+1}'] = bridge.field.copy()
                
            return fields
            
        except Exception as e:
            print(f"Error using CASCADE components: {e}")
            print("Falling back to simple field generation")
    
    # Fallback: Create simplified test fields
    
    # Create coordinate grids for 3D field
    size = 32
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Ground state (concentrated at bottom)
    ground = np.exp(-((X)**2 + (Y)**2 + (Z+0.5)**2) / 0.3)
    fields['ground_state'] = ground
    
    # Creation state (spiral pattern)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arctan2(Y, X)
    spiral = np.sin(r * PHI * 5 + theta * 3) * 0.3
    creation = ground * 0.7 + spiral * 0.3
    fields['stage_2'] = creation
    
    # Heart state (toroidal)
    heart_field = np.exp(-((X)**2 + (Y)**2 + (Z)**2) / 0.3)
    pulse = np.sin(r * PHI * 8) * 0.2 + 0.8
    heart = creation * 0.5 + heart_field * pulse * 0.5
    fields['stage_3'] = heart
    
    # Voice state (standing wave)
    voice_pattern = np.sin(X * 6 * PHI) * np.sin(Y * 6 * PHI) * np.sin(Z * 6 * PHI)
    voice = heart * 0.4 + voice_pattern * 0.6
    fields['stage_4'] = voice
    
    # Vision state (multiple timelines)
    timeline1 = np.sin((X + Y) * 5 * PHI + Z * PHI_PHI)
    timeline2 = np.sin((X - Y) * 5 * PHI + Z * PHI)
    vision_field = (timeline1 + timeline2) * 0.5
    vision = voice * 0.3 + vision_field * 0.7
    fields['stage_5'] = vision
    
    # Unity state (unified field)
    unified = np.sin(X * PHI) * np.sin(Y * PHI) * np.sin(Z * PHI_PHI)
    unity = vision * 0.2 + unified * 0.8
    fields['stage_6'] = unity
    
    # Transcendent state (phi-harmonic coherence)
    transcendent = np.sin(X * PHI_PHI) * np.sin(Y * PHI_PHI) * np.sin(Z * PHI_PHI)
    transcendent = transcendent / np.max(np.abs(transcendent))
    fields['stage_7'] = transcendent
    
    return fields


def visualize_field_3d(field: np.ndarray,
                       threshold: float = 0.5,
                       title: str = "Quantum Field",
                       use_phi_colors: bool = True) -> None:
    """
    Visualize a 3D quantum field.
    
    Args:
        field: 3D NumPy array containing the field
        threshold: Value threshold for visualization
        title: Plot title
        use_phi_colors: Whether to use phi-based color mapping
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available (matplotlib not installed)")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
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
    
    plt.tight_layout()
    plt.show()


def test_pattern_engine() -> None:
    """Test the quantum pattern engine with example fields."""
    print("\n=== Testing Quantum Pattern Engine ===\n")
    
    # Try to import components first
    if COMPONENTS_LOADED:
        try:
            # Create pattern engine
            pattern_engine = QuantumPatternEngine()
            
            # Create test fields
            test_fields = create_test_fields()
            
            # Register some patterns
            print("Registering patterns...")
            pattern_engine.register_pattern(
                "ground_state", 
                test_fields['ground_state'],
                metadata={"frequency": 432, "stage": 1}
            )
            
            pattern_engine.register_pattern(
                "heart_field", 
                test_fields['stage_3'],
                metadata={"frequency": 594, "stage": 3}
            )
            
            pattern_engine.register_pattern(
                "unity_field", 
                test_fields['stage_6'],
                metadata={"frequency": 768, "stage": 6}
            )
            
            # Test pattern matching
            print("\nTesting pattern matching...")
            
            for name, field in test_fields.items():
                print(f"\nAnalyzing field: {name}")
                
                # Match patterns
                matches = pattern_engine.match_pattern(field, threshold=0.7)
                
                # Print results
                if matches:
                    print("Pattern matches:")
                    for match in matches:
                        print(f"  - {match['name']}: {match['similarity']:.4f} similarity")
                else:
                    print("No pattern matches found.")
                
                # Analyze field
                metrics = pattern_engine._calculate_phi_metrics(field)
                
                print("Phi-harmonic metrics:")
                print(f"  - Phi alignment: {metrics.get('phi_alignment', 0):.4f}")
                
                if 'frequency_resonance' in metrics:
                    print("  - Frequency resonance:")
                    for freq_name, resonance in metrics['frequency_resonance'].items():
                        print(f"    - {freq_name} ({SACRED_FREQUENCIES[freq_name]} Hz): {resonance:.4f}")
                
                if 'toroidal_metrics' in metrics:
                    torus = metrics['toroidal_metrics']
                    print("  - Toroidal metrics:")
                    print(f"    - Flow balance: {torus['flow_balance']:.4f}")
                    print(f"    - Circulation: {torus['circulation']:.4f}")
                    print(f"    - Toroidal coherence: {torus['toroidal_coherence']:.4f}")
                
                # Visualize
                if VISUALIZATION_AVAILABLE and name in ['ground_state', 'stage_3', 'stage_6']:
                    visualize_field_3d(
                        field,
                        threshold=0.3,
                        title=f"Quantum Field - {name}",
                        use_phi_colors=True
                    )
            
            # Save pattern library
            library_path = os.path.join(project_root, "cascade", "pattern_recognition", "data", "pattern_library.pkl")
            os.makedirs(os.path.dirname(library_path), exist_ok=True)
            
            print(f"\nSaving pattern library to {library_path}")
            pattern_engine.save_pattern_library(library_path)
            
            return
        
        except Exception as e:
            print(f"Error in pattern engine test: {e}")
            print("Falling back to simplified test")
    
    # Simplified fallback test
    print("Running simplified pattern recognition test...")
    
    # Create test fields
    test_fields = create_test_fields()
    
    # Analyze key fields
    for name in ['ground_state', 'stage_3', 'stage_6']:
        if name in test_fields:
            field = test_fields[name]
            
            print(f"\nAnalyzing field: {name}")
            
            # Calculate basic metrics
            field_mean = np.mean(field)
            field_max = np.max(field)
            
            # Calculate phi alignment (simplified)
            flat_field = field.flatten()
            phi_powers = [PHI ** i for i in range(-2, 3)]
            
            min_distances = []
            for val in flat_field[:1000]:  # Sample for efficiency
                distances = [abs(val - p) for p in phi_powers]
                min_distances.append(min(distances))
            
            phi_alignment = 1.0 - np.mean(min_distances) / PHI
            
            print(f"Field mean: {field_mean:.4f}")
            print(f"Field max: {field_max:.4f}")
            print(f"Phi alignment: {phi_alignment:.4f}")
            
            # Visualize
            if VISUALIZATION_AVAILABLE:
                visualize_field_3d(
                    field,
                    threshold=0.3,
                    title=f"Quantum Field - {name}",
                    use_phi_colors=True
                )


def test_phi_matcher() -> None:
    """Test the PhiPatternMatcher with example fields."""
    print("\n=== Testing Phi Pattern Matcher ===\n")
    
    # Try to import components first
    if COMPONENTS_LOADED:
        try:
            # Create phi matcher
            phi_matcher = PhiPatternMatcher(use_sacred_geometry=True)
            
            # Show available patterns
            print("Available phi patterns:")
            patterns = phi_matcher.get_available_patterns()
            for pattern in patterns:
                print(f"  - {pattern['name']} ({pattern['type']}): {pattern['description']}")
            
            # Generate some patterns
            print("\nGenerating phi patterns...")
            
            # Create field dimensions
            dimensions_2d = (64, 64)
            dimensions_3d = (32, 32, 32)
            
            # Generate 2D patterns
            phi_spiral = phi_matcher.generate_pattern("phi_spiral", dimensions_2d)
            flower_of_life = phi_matcher.generate_pattern("flower_of_life", dimensions_2d)
            sri_yantra = phi_matcher.generate_pattern("sri_yantra", dimensions_2d)
            
            # Generate 3D patterns
            phi_torus = phi_matcher.generate_pattern("phi_torus", dimensions_3d)
            merkaba = phi_matcher.generate_pattern("merkaba", dimensions_3d)
            
            # Visualize 2D patterns
            if VISUALIZATION_AVAILABLE:
                # Visualize 2D patterns
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(phi_spiral, cmap='viridis')
                axes[0].set_title("Phi Spiral")
                axes[0].axis('off')
                
                axes[1].imshow(flower_of_life, cmap='viridis')
                axes[1].set_title("Flower of Life")
                axes[1].axis('off')
                
                axes[2].imshow(sri_yantra, cmap='viridis')
                axes[2].set_title("Sri Yantra")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Visualize 3D patterns
                visualize_field_3d(
                    phi_torus,
                    threshold=0.3,
                    title="Phi Torus Pattern",
                    use_phi_colors=True
                )
                
                visualize_field_3d(
                    merkaba,
                    threshold=0.3,
                    title="Merkaba Pattern",
                    use_phi_colors=True
                )
            
            # Test fields for pattern detection
            print("\nDetecting phi patterns in test fields...")
            
            test_fields = create_test_fields()
            
            for name, field in test_fields.items():
                # Only test a subset of fields
                if name not in ['ground_state', 'stage_3', 'stage_6']:
                    continue
                    
                print(f"\nAnalyzing field: {name}")
                
                # Detect patterns
                patterns = phi_matcher.detect_phi_patterns(field)
                
                # Print results
                if patterns:
                    print("Detected patterns:")
                    for pattern in patterns:
                        print(f"  - {pattern['pattern_name']}: {pattern['confidence']:.4f} confidence")
                else:
                    print("No patterns detected.")
                
                # Calculate phi metrics
                metrics = phi_matcher.get_phi_metrics(field)
                
                print("Phi metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  - {metric}: {value:.4f}")
            
            return
            
        except Exception as e:
            print(f"Error in phi matcher test: {e}")
            print("Falling back to simplified test")
    
    # Simplified fallback test
    print("Running simplified phi pattern test...")
    
    if VISUALIZATION_AVAILABLE:
        # Create a simple phi spiral pattern
        size = 64
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Create phi spiral
        spiral = np.exp(-R / PHI) * np.sin(Theta * PHI + R * PHI_PHI * 5)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(spiral, cmap='viridis')
        plt.title("Phi Spiral Pattern")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def test_field_recognizer() -> None:
    """Test the FieldStateRecognizer with example fields."""
    print("\n=== Testing Field State Recognizer ===\n")
    
    # Try to import components first
    if COMPONENTS_LOADED:
        try:
            # Create field recognizer
            recognizer = FieldStateRecognizer()
            
            # Show built-in states
            print("Built-in field states:")
            for name, state in recognizer.state_definitions.items():
                print(f"  - {name}: {state['description']}")
                print(f"    Frequency: {state['frequencies'][0]} Hz")
                print(f"    Coherence range: {state['coherence_range'][0]}-{state['coherence_range'][1]}")
            
            # Test recognition on example fields
            print("\nTesting field state recognition...")
            
            test_fields = create_test_fields()
            
            for name, field in test_fields.items():
                print(f"\nAnalyzing field: {name}")
                
                # Recognize state
                state_probs = recognizer.recognize_state(field)
                
                # Print top states
                print("State recognition results:")
                sorted_states = sorted(state_probs.items(), key=lambda x: x[1], reverse=True)
                
                for i, (state_name, prob) in enumerate(sorted_states[:3]):
                    freq = recognizer.state_definitions[state_name]['frequencies'][0]
                    print(f"  {i+1}. {state_name}: {prob:.4f} ({freq} Hz)")
                
                # Extract and show field features
                features = recognizer.extract_field_state_features(field)
                
                print("\nField features:")
                print(f"  Phi alignment: {features.get('phi_alignment', 0):.4f}")
                
                if 'energy_center' in features:
                    center = features['energy_center']
                    print(f"  Energy center: {center}")
                
                if 'center_energy_ratio' in features:
                    print(f"  Center energy ratio: {features['center_energy_ratio']:.4f}")
                
                if 'bottom_energy_ratio' in features:
                    print(f"  Bottom energy ratio: {features['bottom_energy_ratio']:.4f}")
                
                if 'torus_alignment' in features:
                    print(f"  Torus alignment: {features['torus_alignment']:.4f}")
                
                # Only visualize a few key fields
                if VISUALIZATION_AVAILABLE and name in ['ground_state', 'stage_3', 'stage_6', 'stage_7']:
                    # Extract top state
                    top_state = sorted_states[0][0]
                    top_prob = sorted_states[0][1]
                    state_desc = recognizer.state_definitions[top_state]['description']
                    
                    visualize_field_3d(
                        field,
                        threshold=0.3,
                        title=f"Field: {name}\nRecognized as: {top_state} ({top_prob:.2f})\n{state_desc}",
                        use_phi_colors=True
                    )
            
            # Train recognizer on test fields
            print("\nTraining field recognizer on example fields...")
            
            # Organize training examples
            training_examples = {
                'ground': [test_fields['ground_state']],
                'creation': [test_fields['stage_2']],
                'heart': [test_fields['stage_3']],
                'voice': [test_fields['stage_4']],
                'vision': [test_fields['stage_5']],
                'unity': [test_fields['stage_6']],
                'transcendent': [test_fields['stage_7']]
            }
            
            recognizer.train_on_examples(training_examples, method='template')
            
            print("Training complete. Testing on samples...")
            
            # Test on the same fields for demonstration
            for name, field in test_fields.items():
                if name not in ['ground_state', 'stage_6']:  # Just test a couple
                    continue
                    
                print(f"\nAnalyzing field: {name} (after training)")
                
                # Recognize state
                state_probs = recognizer.recognize_state(field)
                
                # Print top states
                print("State recognition results:")
                sorted_states = sorted(state_probs.items(), key=lambda x: x[1], reverse=True)
                
                for i, (state_name, prob) in enumerate(sorted_states[:3]):
                    freq = recognizer.state_definitions[state_name]['frequencies'][0]
                    print(f"  {i+1}. {state_name}: {prob:.4f} ({freq} Hz)")
            
            # Save trained model
            model_path = os.path.join(project_root, "cascade", "pattern_recognition", "data", "field_state_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            print(f"\nSaving trained model to {model_path}")
            recognizer.save_model(model_path)
            
            return
        
        except Exception as e:
            print(f"Error in field recognizer test: {e}")
            print("Falling back to simplified test")
    
    # Simplified fallback test
    print("Running simplified field state recognition test...")
    
    # Create test fields
    test_fields = create_test_fields()
    
    # Define simple state frequencies
    state_freqs = {
        'ground_state': 432,
        'stage_2': 528,
        'stage_3': 594,
        'stage_4': 672,
        'stage_5': 720,
        'stage_6': 768,
        'stage_7': 888
    }
    
    # Analyze a few fields
    for name in ['ground_state', 'stage_3', 'stage_6']:
        if name in test_fields:
            field = test_fields[name]
            
            print(f"\nAnalyzing field: {name}")
            print(f"Associated frequency: {state_freqs.get(name, 'unknown')} Hz")
            
            # Calculate simple metrics
            field_mean = np.mean(field)
            field_std = np.std(field)
            
            # Calculate entropy
            hist, _ = np.histogram(field, bins=10)
            hist_norm = hist / np.sum(hist)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            
            print(f"Field mean: {field_mean:.4f}")
            print(f"Field std: {field_std:.4f}")
            print(f"Field entropy: {entropy:.4f}")
            
            # Visualize
            if VISUALIZATION_AVAILABLE:
                visualize_field_3d(
                    field,
                    threshold=0.3,
                    title=f"Field: {name}\nFrequency: {state_freqs.get(name, 'unknown')} Hz",
                    use_phi_colors=True
                )


def main():
    """Process command line arguments and run the demo."""
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Quantum Pattern Recognition Demo")
    
    parser.add_argument('--test', type=str, default='all',
                      choices=['all', 'pattern', 'phi', 'state'],
                      help="Test to run (default: all)")
    
    parser.add_argument('--no-visualization', action='store_true',
                      help="Disable visualizations")
    
    args = parser.parse_args()
    
    # Handle visualization setting
    global VISUALIZATION_AVAILABLE
    if args.no_visualization:
        VISUALIZATION_AVAILABLE = False
    
    print("CASCADE⚡𓂧φ∞ Quantum Pattern Recognition Demo")
    print("=" * 60)
    
    if COMPONENTS_LOADED:
        print("CASCADE components loaded successfully.")
    else:
        print("CASCADE components not fully loaded. Running in simplified mode.")
    
    print(f"Visualization: {'enabled' if VISUALIZATION_AVAILABLE else 'disabled'}")
    print("-" * 60)
    
    # Run tests
    try:
        if args.test in ['all', 'pattern']:
            test_pattern_engine()
        
        if args.test in ['all', 'phi']:
            test_phi_matcher()
        
        if args.test in ['all', 'state']:
            test_field_recognizer()
            
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())