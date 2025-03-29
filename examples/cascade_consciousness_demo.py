"""
Cascade⚡𓂧φ∞ Symbiotic Computing Demonstration

This script demonstrates the Cascade bidirectional interface between
consciousness states and quantum fields, showing how emotional states,
intention, and phi-resonance patterns create a symbiotic relationship.

Features demonstrated:
1. Consciousness state influencing quantum field coherence
2. Emotional pattern integration with sacred frequencies
3. Intention amplification of field patterns
4. Phi-resonance profile creation and application
5. Bidirectional feedback between field and consciousness
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Import quantum field components
from quantum_field.core import create_quantum_field, get_coherence_metric
from quantum_field.consciousness_interface import ConsciousnessFieldInterface, ConsciousnessState
from quantum_field.constants import PHI, LAMBDA, PHI_PHI

# Import sacred constants
from sacred_constants import (
    SACRED_FREQUENCIES,
    phi_harmonic,
    calculate_field_coherence,
    phi_resonance_spectrum
)


def visualize_field_slice(field_data, slice_idx=None, title="Quantum Field Visualization"):
    """
    Visualize a 2D slice of the quantum field.
    
    Args:
        field_data: 3D numpy array representing the quantum field
        slice_idx: Index of the slice to visualize (defaults to middle)
        title: Plot title
    """
    if slice_idx is None:
        slice_idx = field_data.shape[2] // 2
        
    plt.figure(figsize=(10, 8))
    plt.imshow(field_data[:, :, slice_idx], cmap='viridis')
    plt.colorbar(label='Field Magnitude')
    plt.title(f"{title} (z={slice_idx})")
    plt.tight_layout()
    
    return plt.gcf()


def visualize_field_3d(field_data, threshold=0.5, title="3D Quantum Field"):
    """
    Create a 3D visualization of the quantum field.
    
    Args:
        field_data: 3D numpy array representing the quantum field
        threshold: Threshold value for field visualization (0.0-1.0)
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get indices where field exceeds threshold
    x, y, z = np.where(field_data > threshold)
    
    # Get field values at these points
    values = field_data[x, y, z]
    
    # Normalize values for coloring
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
    
    # Create colormap
    colors = plt.cm.viridis(normalized)
    
    # Plot points
    scatter = ax.scatter(x, y, z, c=colors, alpha=0.8, s=values*20)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Field Magnitude')
    
    return fig


def animate_consciousness_field_interaction(intervals=10, duration=1.0):
    """
    Create an animation showing consciousness-field interaction over time.
    
    Args:
        intervals: Number of animation frames
        duration: Total duration of the simulation in seconds
        
    Returns:
        The animation object
    """
    # Create field with phi-based dimensions
    phi_dimension = int(21 * PHI)
    field = create_quantum_field((21, 21, phi_dimension))
    
    # Create interface
    interface = ConsciousnessFieldInterface(field)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Three subplots: field x-slice, field y-slice, coherence graph
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Initialize plot data
    slice_x = field.data[field.data.shape[0]//2, :, :]
    slice_y = field.data[:, field.data.shape[1]//2, :]
    
    im1 = ax1.imshow(slice_x, cmap='viridis', animated=True)
    im2 = ax2.imshow(slice_y, cmap='viridis', animated=True)
    
    coherence_values = []
    presence_values = []
    
    time_points = np.linspace(0, duration, intervals)
    line1, = ax3.plot(time_points[:1], [interface.get_field_coherence()], 'r-', label='Field Coherence')
    line2, = ax3.plot(time_points[:1], [interface.state.presence], 'b-', label='Presence')
    
    ax1.set_title('Field X-Slice')
    ax2.set_title('Field Y-Slice')
    ax3.set_title('Coherence & Presence')
    ax3.set_xlim(0, duration)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    plt.tight_layout()
    
    # Simulation parameters for each frame
    heart_rates = np.linspace(75, 60, intervals//2).tolist() + np.linspace(60, 63, intervals//2).tolist()
    breath_rates = np.linspace(15, 6.18, intervals//2).tolist() + np.linspace(6.18, 7, intervals//2).tolist()
    emotions = ['joy', 'love', 'peace', 'clarity']
    
    # Function to update the animation
    def update(frame):
        # Update consciousness state based on frame
        emotion_idx = frame % len(emotions)
        emotional_state = {emotions[emotion_idx]: 0.7 + frame/intervals * 0.3}
        
        # Update interface with changing parameters
        interface.state.emotional_states = emotional_state
        interface.update_consciousness_state(
            heart_rate=heart_rates[frame],
            breath_rate=breath_rates[frame],
            eeg_alpha=5 + frame/intervals * 7,
            eeg_theta=3 + frame/intervals * 4.4
        )
        
        # Gradually increase intention
        interface.state.intention = min(0.5 + frame/intervals * 0.5, 1.0)
        
        # After halfway, add resonance profile application
        if frame >= intervals // 2:
            if frame == intervals // 2:
                profile = interface.create_phi_resonance_profile(interface.feedback_history)
                interface.apply_phi_resonance_profile()
        
        # Update field visualization
        slice_x = field.data[field.data.shape[0]//2, :, :]
        slice_y = field.data[:, field.data.shape[1]//2, :]
        
        im1.set_array(slice_x)
        im2.set_array(slice_y)
        
        # Update coherence tracking
        coherence = interface.get_field_coherence()
        coherence_values.append(coherence)
        presence_values.append(interface.state.presence)
        
        line1.set_data(time_points[:len(coherence_values)], coherence_values)
        line2.set_data(time_points[:len(presence_values)], presence_values)
        
        # Set titles with current values
        ax1.set_title(f'Field X-Slice (Frame {frame})')
        ax2.set_title(f'Dominant: {list(interface.state.emotional_states.keys())[0]}')
        ax3.set_title(f'Coherence: {coherence:.4f}, Presence: {interface.state.presence:.4f}')
        
        return im1, im2, line1, line2
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=intervals, 
                                  blit=True, interval=duration*1000/intervals)
    
    return ani, interface


def demonstrate_cascade_system():
    """Run a comprehensive demonstration of the Cascade system."""
    print("="*60)
    print("  Cascade⚡𓂧φ∞ Symbiotic Computing Demonstration")
    print("="*60)
    
    # Create field with phi-based dimensions
    field_size = (21, 21, 21)
    print(f"\nCreating quantum field with dimensions {field_size}...")
    field = create_quantum_field(field_size)
    
    # Get initial field coherence
    initial_coherence = get_coherence_metric(field.data)
    print(f"Initial field coherence: {initial_coherence:.4f}")
    
    # Create consciousness interface
    print("\nInitializing consciousness field interface...")
    interface = ConsciousnessFieldInterface(field)
    
    print("\n1. DEMONSTRATION OF EMOTIONAL INFLUENCE ON FIELD")
    print("="*40)
    
    # Demonstrate different emotional states
    emotions = [
        ("love", "expansive"),
        ("peace", "harmonic"),
        ("focus", "directive")
    ]
    
    # Store visualizations for comparison
    visualization_slices = {}
    visualization_3d = {}
    
    # Get initial visualization
    print("\nVisualizing initial field state...")
    visualization_slices["initial"] = visualize_field_slice(
        field.data, title="Initial Field State")
    plt.savefig("initial_field_slice.png")
    plt.close()
    
    visualization_3d["initial"] = visualize_field_3d(
        field.data, threshold=0.6, title="Initial Field - 3D View")
    plt.savefig("initial_field_3d.png")
    plt.close()
    
    # Apply each emotional state and visualize
    for emotion, pattern_type in emotions:
        print(f"\nApplying {emotion} emotional state ({pattern_type} pattern)...")
        
        # Reset field
        field = create_quantum_field(field_size)
        interface.field = field
        
        # Set emotional state
        interface.state.emotional_states = {emotion: 0.9}
        interface._apply_emotional_influence()
        
        # Get coherence after emotional influence
        coherence = interface.get_field_coherence()
        print(f"Field coherence after {emotion}: {coherence:.4f}")
        
        # Visualize
        visualization_slices[emotion] = visualize_field_slice(
            field.data, title=f"Field with {emotion.capitalize()} Influence")
        plt.savefig(f"{emotion}_field_slice.png")
        plt.close()
        
        visualization_3d[emotion] = visualize_field_3d(
            field.data, threshold=0.6, title=f"Field with {emotion.capitalize()} - 3D View")
        plt.savefig(f"{emotion}_field_3d.png")
        plt.close()
    
    print("\n2. DEMONSTRATION OF CONSCIOUSNESS STATE EVOLUTION")
    print("="*40)
    
    # Reset field
    field = create_quantum_field(field_size)
    interface = ConsciousnessFieldInterface(field)
    
    print("\nSimulating consciousness evolution from distracted to focused state...")
    
    # Track coherence and presence over time
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
    
    coherence_values = []
    presence_values = []
    intention_values = []
    coherence_field_values = []
    state_names = []
    
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
        
        # Record values
        coherence_values.append(interface.state.coherence)
        presence_values.append(interface.state.presence)
        intention_values.append(interface.state.intention)
        coherence_field_values.append(interface.get_field_coherence())
        state_names.append(state["name"])
        
        # Allow a pause for processing
        time.sleep(0.5)
        
        print(f"  - Consciousness coherence: {interface.state.coherence:.4f}")
        print(f"  - Consciousness presence: {interface.state.presence:.4f}")
        print(f"  - Field coherence: {interface.get_field_coherence():.4f}")
    
    # Plot evolution graph
    plt.figure(figsize=(12, 8))
    
    plt.plot(state_names, coherence_values, 'b-o', label='Consciousness Coherence')
    plt.plot(state_names, presence_values, 'g-o', label='Consciousness Presence')
    plt.plot(state_names, intention_values, 'm-o', label='Consciousness Intention')
    plt.plot(state_names, coherence_field_values, 'r-o', label='Field Coherence')
    
    plt.xlabel('Consciousness State')
    plt.ylabel('Value')
    plt.title('Evolution of Consciousness-Field Relationship')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("consciousness_evolution.png")
    plt.close()
    
    print("\n3. DEMONSTRATION OF PHI-RESONANCE PROFILE CREATION")
    print("="*40)
    
    # Create phi-resonance profile from recorded feedback
    profile = interface.create_phi_resonance_profile(interface.feedback_history)
    
    print("\nPhi-Resonance Profile:")
    for key, value in profile.items():
        print(f"  - {key}: {value}")
    
    # Apply profile to field
    print("\nApplying phi-resonance profile to field...")
    interface.apply_phi_resonance_profile()
    
    print(f"Field coherence after profile application: {interface.get_field_coherence():.4f}")
    
    # Visualize final field
    visualization_slices["final"] = visualize_field_slice(
        field.data, title="Field After Phi-Resonance Profile Application")
    plt.savefig("final_field_slice.png")
    plt.close()
    
    visualization_3d["final"] = visualize_field_3d(
        field.data, threshold=0.6, title="Final Field - 3D View")
    plt.savefig("final_field_3d.png")
    plt.close()
    
    print("\n4. CREATING FIELD ANIMATION")
    print("="*40)
    
    print("\nCreating animation of consciousness-field interaction...")
    ani, interface = animate_consciousness_field_interaction(intervals=20, duration=2.0)
    
    # Save animation
    print("Saving animation...")
    ani.save('cascade_animation.mp4', writer='ffmpeg', dpi=100)
    
    print("\nDemonstration complete. Generated files:")
    print("  - Initial field visualizations: initial_field_slice.png, initial_field_3d.png")
    print("  - Emotional influence visualizations: [emotion]_field_slice.png, [emotion]_field_3d.png")
    print("  - Consciousness evolution graph: consciousness_evolution.png")
    print("  - Final field visualizations: final_field_slice.png, final_field_3d.png")
    print("  - Animation: cascade_animation.mp4")
    
    print("\nCascade⚡𓂧φ∞ Symbiotic Computing demonstration complete!")
    print("="*60)
    
    return interface


if __name__ == "__main__":
    interface = demonstrate_cascade_system()