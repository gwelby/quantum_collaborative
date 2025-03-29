#!/usr/bin/env python3
"""
Quantum Consciousness Field Demo

This script demonstrates the Quantum Consciousness Integration Layer, 
which enables direct interaction between quantum fields and consciousness states.

The demo simulates a meditation session with biofeedback data affecting field coherence.
"""

import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.append('..')

from quantum_field import (
    generate_quantum_field,
    ConsciousnessFieldInterface,
    ConsciousnessState,
    PHI,
    LAMBDA
)
from quantum_field.core import QuantumField, create_quantum_field, get_coherence_metric
from quantum_field.visualization3d import visualize_field_3d


class BiofeedbackSimulator:
    """Simulates biofeedback data for demonstration purposes."""
    
    def __init__(self, 
                 initial_heart_rate=75, 
                 initial_breath_rate=15, 
                 initial_skin_conductance=8,
                 initial_eeg_alpha=5,
                 initial_eeg_theta=2):
        """Initialize with starting values."""
        self.heart_rate = initial_heart_rate
        self.breath_rate = initial_breath_rate
        self.skin_conductance = initial_skin_conductance
        self.eeg_alpha = initial_eeg_alpha
        self.eeg_theta = initial_eeg_theta
        
        # Target values for meditation
        self.target_heart_rate = 60  # Phi-resonant
        self.target_breath_rate = 6.18  # Phi-based
        self.target_skin_conductance = 3
        self.target_eeg_alpha = 12
        self.target_eeg_theta = 7.4  # Close to alpha/theta = phi
        
        # History for plotting
        self.history = {
            'heart_rate': [initial_heart_rate],
            'breath_rate': [initial_breath_rate],
            'skin_conductance': [initial_skin_conductance],
            'eeg_alpha': [initial_eeg_alpha],
            'eeg_theta': [initial_eeg_theta],
            'coherence': [],
            'time': [0]
        }
        self.start_time = time.time()
        
    def update(self, elapsed_seconds, progress=0.0):
        """
        Update biofeedback values based on elapsed time and meditation progress.
        
        Args:
            elapsed_seconds: Time elapsed since start
            progress: Meditation progress from 0.0 to 1.0
        """
        # Calculate progress-based target
        progress = min(max(progress, 0.0), 1.0)
        
        # Add some randomness to make it realistic
        heart_variance = np.random.normal(0, 2)
        breath_variance = np.random.normal(0, 0.5)
        skin_variance = np.random.normal(0, 0.3)
        eeg_alpha_variance = np.random.normal(0, 0.5)
        eeg_theta_variance = np.random.normal(0, 0.3)
        
        # Calculate new values blending current with target based on progress
        self.heart_rate = (self.heart_rate * (1 - progress * 0.1) + 
                           self.target_heart_rate * progress * 0.1 + 
                           heart_variance)
        
        self.breath_rate = (self.breath_rate * (1 - progress * 0.1) + 
                           self.target_breath_rate * progress * 0.1 + 
                           breath_variance)
        
        self.skin_conductance = (self.skin_conductance * (1 - progress * 0.1) + 
                                self.target_skin_conductance * progress * 0.1 + 
                                skin_variance)
        
        self.eeg_alpha = (self.eeg_alpha * (1 - progress * 0.1) + 
                         self.target_eeg_alpha * progress * 0.1 + 
                         eeg_alpha_variance)
        
        self.eeg_theta = (self.eeg_theta * (1 - progress * 0.1) + 
                         self.target_eeg_theta * progress * 0.1 + 
                         eeg_theta_variance)
        
        # Ensure values stay in reasonable ranges
        self.heart_rate = max(50, min(100, self.heart_rate))
        self.breath_rate = max(3, min(20, self.breath_rate))
        self.skin_conductance = max(1, min(20, self.skin_conductance))
        self.eeg_alpha = max(1, min(20, self.eeg_alpha))
        self.eeg_theta = max(1, min(20, self.eeg_theta))
        
        # Record values in history
        self.history['heart_rate'].append(self.heart_rate)
        self.history['breath_rate'].append(self.breath_rate)
        self.history['skin_conductance'].append(self.skin_conductance)
        self.history['eeg_alpha'].append(self.eeg_alpha)
        self.history['eeg_theta'].append(self.eeg_theta)
        self.history['time'].append(elapsed_seconds)
        
        return {
            'heart_rate': self.heart_rate,
            'breath_rate': self.breath_rate,
            'skin_conductance': self.skin_conductance,
            'eeg_alpha': self.eeg_alpha,
            'eeg_theta': self.eeg_theta
        }
    
    def get_current_values(self):
        """Get current biofeedback values."""
        return {
            'heart_rate': self.heart_rate,
            'breath_rate': self.breath_rate,
            'skin_conductance': self.skin_conductance,
            'eeg_alpha': self.eeg_alpha,
            'eeg_theta': self.eeg_theta
        }
    
    def add_coherence_value(self, coherence):
        """Add field coherence value to history."""
        self.history['coherence'].append(coherence)
    
    def plot_history(self):
        """Plot the history of biofeedback values and field coherence."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Plot heart rate
        axes[0, 0].plot(self.history['time'], self.history['heart_rate'], 'r-')
        axes[0, 0].set_title('Heart Rate (bpm)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].axhline(y=60, color='r', linestyle='--', alpha=0.5)
        
        # Plot breath rate
        axes[0, 1].plot(self.history['time'], self.history['breath_rate'], 'b-')
        axes[0, 1].set_title('Breath Rate (bpm)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].axhline(y=6.18, color='b', linestyle='--', alpha=0.5)
        
        # Plot skin conductance
        axes[1, 0].plot(self.history['time'], self.history['skin_conductance'], 'g-')
        axes[1, 0].set_title('Skin Conductance (μS)')
        axes[1, 0].set_xlabel('Time (s)')
        
        # Plot EEG values
        axes[1, 1].plot(self.history['time'], self.history['eeg_alpha'], 'c-', label='Alpha')
        axes[1, 1].plot(self.history['time'], self.history['eeg_theta'], 'm-', label='Theta')
        axes[1, 1].set_title('EEG Bands')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].legend()
        
        # Plot Alpha/Theta ratio
        alpha_theta_ratio = [a/t if t > 0 else 0 for a, t in 
                             zip(self.history['eeg_alpha'], self.history['eeg_theta'])]
        axes[2, 0].plot(self.history['time'], alpha_theta_ratio, 'k-')
        axes[2, 0].set_title('Alpha/Theta Ratio')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].axhline(y=PHI, color='k', linestyle='--', alpha=0.5)
        
        # Plot field coherence
        if self.history['coherence']:
            time_coherence = self.history['time'][:len(self.history['coherence'])]
            axes[2, 1].plot(time_coherence, self.history['coherence'], 'y-')
            axes[2, 1].set_title('Field Coherence')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].axhline(y=LAMBDA, color='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig


def run_meditation_simulation(duration_seconds=60, dimensions=(21, 21, 21),
                             visualization=True, interactive=False):
    """
    Run a meditation session simulation with biofeedback affecting a quantum field.
    
    Args:
        duration_seconds: Length of the simulation in seconds
        dimensions: Field dimensions (3D)
        visualization: Whether to visualize the field in 3D
        interactive: Whether to use interactive mode (requires user input)
    """
    print(f"Starting Quantum Consciousness Field meditation simulation")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Field dimensions: {dimensions}")
    print("="*50)
    
    # Create a quantum field
    field = create_quantum_field(dimensions)
    
    # Create consciousness-field interface
    interface = ConsciousnessFieldInterface(field)
    
    # Create biofeedback simulator
    biofeedback = BiofeedbackSimulator()
    
    # Initial field coherence
    initial_coherence = interface.get_field_coherence()
    print(f"Initial field coherence: {initial_coherence:.4f}")
    
    # Store initial field for comparison
    initial_field_data = field.data.copy()
    
    # Initialize 3D visualization if requested
    if visualization:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Keep track of simulation time
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update every second
    
    # Main simulation loop
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Exit condition
            if elapsed >= duration_seconds and not interactive:
                break
            
            # Update every interval
            if current_time - last_update_time >= update_interval:
                # Calculate meditation progress (0.0 to 1.0)
                progress = min(elapsed / duration_seconds, 1.0)
                
                # Update biofeedback with progress
                feedback_data = biofeedback.update(elapsed, progress)
                
                # Update consciousness state with biofeedback
                interface.update_consciousness_state(**feedback_data)
                
                # Get and store field coherence
                coherence = interface.get_field_coherence()
                biofeedback.add_coherence_value(coherence)
                
                # Print status update
                print(f"[{elapsed:.1f}s] Progress: {progress*100:.1f}%")
                print(f"  Heart: {feedback_data['heart_rate']:.1f} bpm, Breath: {feedback_data['breath_rate']:.1f} bpm")
                print(f"  Alpha/Theta: {feedback_data['eeg_alpha']:.1f}/{feedback_data['eeg_theta']:.1f} = {feedback_data['eeg_alpha']/feedback_data['eeg_theta']:.2f}")
                print(f"  Field coherence: {coherence:.4f}")
                
                # Update visualization
                if visualization and elapsed > 1.0:  # Skip first second
                    ax.clear()
                    visualize_field_3d(field.data, ax=ax, threshold=0.7)
                    ax.set_title(f"Quantum Field - Coherence: {coherence:.4f}")
                    plt.draw()
                    plt.pause(0.01)
                
                # Update last update time
                last_update_time = current_time
                
                # Interactive mode - ask to continue
                if interactive and elapsed >= duration_seconds:
                    response = input("Continue simulation? (y/n): ")
                    if response.lower() != 'y':
                        break
                    else:
                        # Continue for another duration
                        duration_seconds += int(duration_seconds / 2)
                        print(f"Continuing until {duration_seconds}s...")
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    
    # Final field coherence
    final_coherence = interface.get_field_coherence()
    print("\n" + "="*50)
    print(f"Simulation complete after {elapsed:.1f} seconds")
    print(f"Final field coherence: {final_coherence:.4f}")
    print(f"Coherence change: {final_coherence - initial_coherence:.4f}")
    
    # Create phi-resonance profile
    profile = interface.create_phi_resonance_profile(interface.feedback_history)
    
    print("\nPhi-Resonance Profile:")
    for key, value in profile.items():
        if key != "resonant_frequencies":  # Skip detailed frequency list
            print(f"  {key}: {value}")
    
    # Plot biofeedback history
    history_fig = biofeedback.plot_history()
    plt.show()
    
    # Compare initial and final field
    fig = plt.figure(figsize=(12, 6))
    
    # Initial field
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_field_3d(initial_field_data, ax=ax1, threshold=0.7)
    ax1.set_title(f"Initial Field - Coherence: {initial_coherence:.4f}")
    
    # Final field
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_field_3d(field.data, ax=ax2, threshold=0.7)
    ax2.set_title(f"Final Field - Coherence: {final_coherence:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return interface


def main():
    """Process command line arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Quantum Consciousness Field Meditation Simulation")
    
    parser.add_argument('--duration', type=int, default=60,
                       help="Simulation duration in seconds (default: 60)")
    parser.add_argument('--dimensions', type=str, default="21,21,21",
                       help="Field dimensions as comma-separated values (default: 21,21,21)")
    parser.add_argument('--no-visualization', action='store_true',
                       help="Disable 3D visualization")
    parser.add_argument('--interactive', action='store_true',
                       help="Enable interactive mode (ask to continue after duration)")
    
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
    
    # Run simulation
    run_meditation_simulation(
        duration_seconds=args.duration,
        dimensions=dimensions,
        visualization=not args.no_visualization,
        interactive=args.interactive
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())