#!/usr/bin/env python3
"""
Multi-dimensional Perception Framework Demo

This script demonstrates the Multi-dimensional Perception Framework, which 
expands sensory capabilities beyond 3D into phi-scaled dimensions, translating 
higher-dimensional patterns into synchronized audio-visual-tactile experiences.
"""

import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.append('..')

from quantum_field import (
    generate_quantum_field,
    SACRED_FREQUENCIES,
    PHI,
    LAMBDA
)
from quantum_field.visualization3d import visualize_field_3d
from quantum_field.multidimensional import (
    PerceptionEngine,
    HyperField,
    SensoryTranslator,
    ModalityMap,
    PhiDimensionalScaling
)


# Create phi-harmonic colormap
def create_phi_harmonic_cmap():
    """Create a custom colormap with phi-harmonic color transitions."""
    # Define phi-harmonic color points
    phi_colors = [
        (0.0, (0.1, 0.1, 0.3)),          # Deep blue-violet (unity)
        (0.5 - LAMBDA/2, (0.0, 0.2, 0.5)), # Deep blue (peace)
        (0.5 - LAMBDA/4, (0.0, 0.5, 0.8)), # Blue (truth)
        (0.5, (0.0, 0.7, 0.7)),          # Teal (balance)
        (0.5 + LAMBDA/4, (0.3, 0.8, 0.3)), # Green (love)
        (0.5 + LAMBDA/2, (0.7, 0.7, 0.0)), # Gold (wisdom)
        (0.618, (0.9, 0.5, 0.0)),        # Orange (creativity)
        (0.786, (1.0, 0.0, 0.0)),        # Red (passion)
        (0.888, (0.7, 0.0, 0.7)),        # Purple (insight)
        (1.0, (0.3, 0.0, 0.5))           # Deep purple (unity)
    ]
    
    # Create colormap
    return LinearSegmentedColormap.from_list('phi_harmonic', phi_colors)


class DemoVisualizer:
    """Visualization handler for the multidimensional perception demo."""
    
    def __init__(self, engine):
        """
        Initialize the visualizer.
        
        Args:
            engine: PerceptionEngine instance
        """
        self.engine = engine
        self.phi_cmap = create_phi_harmonic_cmap()
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 9))
        
        # 3D visualizations
        self.ax_3d = self.fig.add_subplot(231, projection='3d')
        self.ax_dim = self.fig.add_subplot(232, projection='3d')
        
        # Sensory visualizations
        self.ax_visual = self.fig.add_subplot(234)
        self.ax_auditory = self.fig.add_subplot(235)
        self.ax_emotional = self.fig.add_subplot(233)
        
        # Dimensional analysis
        self.ax_analysis = self.fig.add_subplot(236)
        
        # Set up plots
        self.setup_plots()
        
        # Initialize last update time
        self.last_update = time.time()
        self.update_interval = 0.2  # seconds
    
    def setup_plots(self):
        """Set up plot layouts and labels."""
        # 3D Field
        self.ax_3d.set_title("3D Quantum Field")
        
        # Dimension Projection
        self.ax_dim.set_title("Dimensional Projection")
        
        # Visual Representation
        self.ax_visual.set_title("Visual Perception")
        self.ax_visual.axis('off')
        
        # Auditory Representation
        self.ax_auditory.set_title("Auditory Perception")
        self.ax_auditory.set_xlabel("Time")
        self.ax_auditory.set_ylabel("Frequency (Hz)")
        
        # Emotional Representation
        self.ax_emotional.set_title("Emotional Perception")
        self.ax_emotional.axis('equal')
        self.ax_emotional.set_xlim(-1.1, 1.1)
        self.ax_emotional.set_ylim(-1.1, 1.1)
        
        # Dimensional Analysis
        self.ax_analysis.set_title("Dimensional Profile")
        self.ax_analysis.set_xlabel("Dimension")
        self.ax_analysis.set_ylabel("Energy")
        
        # Set tight layout
        self.fig.tight_layout()
    
    def update(self, active_dimension=3):
        """
        Update all visualizations.
        
        Args:
            active_dimension: Currently active dimension
        """
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Get data from engine
        field_3d = self.engine.hyperfield.project_to_3d()
        dim_projection = self.engine.get_dimension_projection(active_dimension)
        
        # Update 3D field visualization
        self.ax_3d.clear()
        self.ax_3d.set_title(f"3D Quantum Field (φ={self.engine.coherence:.3f})")
        visualize_field_3d(field_3d, ax=self.ax_3d, threshold=0.6)
        
        # Update dimension projection
        self.ax_dim.clear()
        dim_name = self.engine.phi_scaling.dimension_name(active_dimension)
        self.ax_dim.set_title(f"{dim_name.capitalize()} Dimension ({active_dimension}D)")
        visualize_field_3d(dim_projection, ax=self.ax_dim, threshold=0.5, 
                         colormap=self.phi_cmap)
        
        # Update visual representation
        self.ax_visual.clear()
        self.ax_visual.set_title("Visual Perception")
        self.ax_visual.axis('off')
        
        visual_data = self.engine.get_sensory_experience("visual")
        if len(visual_data.shape) == 3:
            # Get a slice for 2D visualization
            slice_idx = visual_data.shape[2] // 2
            self.ax_visual.imshow(visual_data[:, :, slice_idx])
        elif len(visual_data.shape) == 4:
            # Use RGB data
            slice_idx = visual_data.shape[2] // 2
            self.ax_visual.imshow(visual_data[:, :, slice_idx, :])
        
        # Update auditory representation
        self.ax_auditory.clear()
        self.ax_auditory.set_title("Auditory Perception")
        self.ax_auditory.set_xlabel("Time")
        self.ax_auditory.set_ylabel("Frequency (Hz)")
        
        auditory_data = self.engine.get_sensory_experience("auditory")
        if isinstance(auditory_data, dict) and "frequency" in auditory_data:
            # Get a central slice
            slice_idx = auditory_data["frequency"].shape[1] // 2
            freq_data = auditory_data["frequency"][:, slice_idx, :]
            
            # Create time scale
            time_scale = np.linspace(0, 1, freq_data.shape[1])
            
            # Plot spectrogram-like visualization
            extent = [0, 1, 0, 1000]
            self.ax_auditory.imshow(freq_data.T, aspect='auto', 
                                  origin='lower', extent=extent,
                                  cmap=self.phi_cmap)
            
            # Add frequency reference lines
            for name, freq in SACRED_FREQUENCIES.items():
                if freq <= 1000:  # Only show frequencies in range
                    rel_pos = freq / 1000
                    self.ax_auditory.axhline(y=rel_pos, color='white', 
                                          alpha=0.5, linestyle='--')
                    self.ax_auditory.text(1.01, rel_pos, name, 
                                        color='white', alpha=0.7,
                                        verticalalignment='center',
                                        fontsize=8)
        
        # Update emotional representation
        self.ax_emotional.clear()
        self.ax_emotional.set_title("Emotional Perception")
        self.ax_emotional.set_xlim(-1.1, 1.1)
        self.ax_emotional.set_ylim(-1.1, 1.1)
        self.ax_emotional.axis('equal')
        
        emotional_data = self.engine.get_sensory_experience("emotional")
        if isinstance(emotional_data, dict):
            # Create a radar chart of emotional values
            emotions = list(emotional_data.keys())
            values = [emotional_data[e] for e in emotions]
            
            # Calculate radar coordinates
            angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            values += values[:1]  # Close the loop
            
            # Draw radar
            self.ax_emotional.plot(
                [np.sin(a) for a in angles], 
                [np.cos(a) for a in angles], 
                'k-', alpha=0.3
            )
            
            self.ax_emotional.fill(
                [v * np.sin(a) for a, v in zip(angles, values)],
                [v * np.cos(a) for a, v in zip(angles, values)],
                alpha=0.4, color='g'
            )
            
            # Add emotion labels
            for i, emotion in enumerate(emotions):
                angle = angles[i]
                self.ax_emotional.text(
                    1.1 * np.sin(angle), 1.1 * np.cos(angle),
                    emotion, horizontalalignment='center',
                    verticalalignment='center'
                )
        
        # Update dimensional analysis
        self.ax_analysis.clear()
        self.ax_analysis.set_title("Dimensional Profile")
        self.ax_analysis.set_xlabel("Dimension")
        self.ax_analysis.set_ylabel("Energy")
        
        # Get dimensional profile
        dim_profile = self.engine.hyperfield.get_dimensional_profile()
        dimensions = []
        energies = []
        
        for key, value in dim_profile.items():
            if key.startswith("dim") and key.endswith("_energy"):
                dim = int(key.split("_")[0][3:])  # Extract dimension number
                dimensions.append(dim)
                energies.append(value)
        
        # Sort by dimension
        dim_energy = sorted(zip(dimensions, energies))
        if dim_energy:
            dimensions, energies = zip(*dim_energy)
            
            # Plot dimensional energies with phi-harmonic colors
            phi_colors = [self.phi_cmap(d / max(dimensions)) for d in dimensions]
            self.ax_analysis.bar(dimensions, energies, color=phi_colors)
            
            # Highlight active dimension
            if active_dimension in dimensions:
                idx = dimensions.index(active_dimension)
                self.ax_analysis.bar([active_dimension], [energies[idx]], 
                                  color='yellow', alpha=0.7)
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Return active dimension for reference
        return active_dimension


def perception_update_callback(event_name, data):
    """Callback for perception updates."""
    if event_name == "perception_update":
        coherence = data.get("coherence", 0)
        print(f"Perception update: coherence = {coherence:.4f}")


def run_interactive_demo(dimensions=(21, 21, 21), duration=60, ndim=5):
    """
    Run an interactive demonstration of the Multi-dimensional Perception Framework.
    
    Args:
        dimensions: Field dimensions
        duration: Demo duration in seconds
        ndim: Number of dimensions to use
    """
    print("="*60)
    print("Multi-dimensional Perception Framework Demo")
    print("="*60)
    print(f"Field dimensions: {dimensions}")
    print(f"Number of dimensions: {ndim}")
    print(f"Duration: {duration} seconds")
    
    # Create optimized perception engine
    engine = PerceptionEngine.create_optimized(dimensions, ndim)
    
    # Register callback
    engine.register_callback("perception_update", perception_update_callback)
    
    # Create visualizer
    visualizer = DemoVisualizer(engine)
    
    # Run the demonstration loop
    start_time = time.time()
    last_action_time = start_time
    action_interval = 5.0  # seconds
    active_dimension = 3  # Start with 3D
    
    # Sacred frequencies to demonstrate
    frequency_sequence = list(SACRED_FREQUENCIES.keys())
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update visualization
            active_dimension = visualizer.update(active_dimension)
            
            # Perform actions periodically
            if current_time - last_action_time >= action_interval:
                # Choose an action
                action = np.random.choice([
                    "shift_dimension",
                    "shift_frequency",
                    "amplify_modality",
                    "cross_modal_mapping",
                    "calibrate_phi"
                ])
                
                # Execute the chosen action
                if action == "shift_dimension":
                    # Choose a random dimension to shift to
                    new_dimension = np.random.randint(3, ndim+1)
                    intensity = 0.6
                    
                    print(f"\n[{elapsed:.1f}s] Shifting to {new_dimension}D perception "
                          f"({engine.phi_scaling.dimension_name(new_dimension)})")
                    
                    # Shift dimension
                    result = engine.shift_to_dimension(new_dimension, intensity)
                    active_dimension = new_dimension
                    
                    print(f"  New dimensional weights: " + 
                          ", ".join([f"{d}D: {w:.3f}" for d, w in 
                                   result["new_weights"].items()]))
                
                elif action == "shift_frequency":
                    # Choose a frequency from the sequence
                    freq_idx = int(elapsed / action_interval) % len(frequency_sequence)
                    frequency_name = frequency_sequence[freq_idx]
                    
                    print(f"\n[{elapsed:.1f}s] Shifting to {frequency_name} frequency "
                          f"({SACRED_FREQUENCIES[frequency_name]} Hz)")
                    
                    # Shift frequency
                    result = engine.shift_to_frequency(frequency_name)
                    
                    print(f"  Frequency: {result['frequency_value']} Hz")
                    print(f"  Field coherence: {result['coherence']:.4f}")
                
                elif action == "amplify_modality":
                    # Choose a modality to amplify
                    modality = np.random.choice(["visual", "auditory", "tactile", 
                                              "emotional", "intuitive"])
                    amplification = 0.5
                    
                    print(f"\n[{elapsed:.1f}s] Amplifying {modality} perception")
                    
                    # Amplify modality
                    result = engine.amplify_sensory_modality(modality, amplification)
                    
                    print(f"  New modality weights: " + 
                          ", ".join([f"{m}: {w:.3f}" for m, w in 
                                   result["new_weights"].items()]))
                
                elif action == "cross_modal_mapping":
                    # Choose random source and target modalities
                    modalities = ["visual", "auditory", "tactile", "emotional", "intuitive"]
                    source = np.random.choice(modalities)
                    target = np.random.choice([m for m in modalities if m != source])
                    
                    print(f"\n[{elapsed:.1f}s] Creating {source}→{target} cross-modal mapping")
                    
                    # Create mapping
                    result = engine.create_cross_modal_mapping(source, target)
                    
                    print(f"  Phi relationship: {result['phi_relationship']}")
                    print(f"  Field coherence: {engine.coherence:.4f}")
                
                elif action == "calibrate_phi":
                    print(f"\n[{elapsed:.1f}s] Calibrating phi alignment")
                    
                    # Run calibration
                    result = engine.calibrate_phi_alignment()
                    
                    print(f"  Initial coherence: {result['initial_coherence']:.4f}")
                    print(f"  Final coherence: {result['final_coherence']:.4f}")
                    print(f"  Improvement: {result['improvement']:.4f}")
                
                # Update the last action time
                last_action_time = current_time
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nDemonstration interrupted.")
    
    # Get final status
    final_status = engine.update()
    
    print("\n" + "="*60)
    print("Demonstration complete")
    print(f"Final field coherence: {final_status['coherence']:.4f}")
    print("="*60)
    
    # Keep the visualization window open
    plt.ioff()
    plt.show()
    
    return engine


def main():
    """Process command line arguments and run the demonstration."""
    parser = argparse.ArgumentParser(
        description="Multi-dimensional Perception Framework Demo")
    
    parser.add_argument('--duration', type=int, default=60,
                       help="Demonstration duration in seconds (default: 60)")
    parser.add_argument('--dimensions', type=str, default="21,21,21",
                       help="Field dimensions as comma-separated values (default: 21,21,21)")
    parser.add_argument('--ndim', type=int, default=5,
                       help="Number of dimensions (3-7, default: 5)")
    
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
    
    # Validate number of dimensions
    ndim = max(3, min(7, args.ndim))
    
    # Run the demonstration
    run_interactive_demo(
        dimensions=dimensions,
        duration=args.duration,
        ndim=ndim
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())