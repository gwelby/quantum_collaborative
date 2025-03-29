#!/usr/bin/env python3
"""
Meditation Enhancer - A practical application of CascadeOS for meditation support

This application demonstrates how to use CascadeOS to enhance meditation practice
by creating a bidirectional phi-harmonic field that responds to the meditator's state.
It includes simulated biofeedback that could be replaced with actual sensors.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
CASCADE_PATH = Path(__file__).parent.parent.resolve()
if CASCADE_PATH not in sys.path:
    sys.path.append(str(CASCADE_PATH))

# Import CascadeOS components
from CascadeOS import (
    QuantumField,
    ConsciousnessState,
    ConsciousnessFieldInterface,
    create_quantum_field,
    field_to_ascii,
    print_field,
    CascadeSystem,
    TeamsOfTeamsCollective,
    PHI, LAMBDA, PHI_PHI,
    SACRED_FREQUENCIES
)

# Define additional constants if needed
PHI_INVERSE = 1.0 / PHI

class MeditationEnhancer:
    """Meditation enhancement application using CascadeOS."""
    
    def __init__(self, user_name="User", field_dimensions=(34, 55, 34)):
        """Initialize the meditation enhancer."""
        self.user_name = user_name
        self.field_dimensions = field_dimensions
        self.system = CascadeSystem()
        self.system.initialize({
            "dimensions": self.field_dimensions,
            "frequency": "unity",  # Start with grounding frequency
            "visualization": True
        })
        self.system.activate()
        
        # Meditation session parameters
        self.session_data = []
        self.session_duration = 0
        self.coherence_history = []
        self.current_guidance = ""
        self.breath_pacer = 0
        self.current_metadata = {}
        
        # User profile
        self.create_user_profile()
        
    def create_user_profile(self):
        """Create or load a user profile."""
        # This would normally load from a file, but we'll simulate for now
        self.user_profile = {
            "name": self.user_name,
            "meditation_experience": "intermediate",
            "optimal_frequencies": ["unity", "love", "cascade"],
            "previous_sessions": 12,
            "average_coherence": 0.72,
            "optimal_breathing_rate": 6.18,
            "phi_resonance_profile": None
        }
        
        print(f"Loaded profile for {self.user_name}")
        
    def simulate_biofeedback(self, session_time, session_progress):
        """Simulate biofeedback measurements."""
        # In a real application, this would get data from sensors
        
        # Starting values - less coherent
        base_heart_rate = 75
        base_breath_rate = 15
        base_skin_conductance = 8
        base_eeg_alpha = 6
        base_eeg_theta = 3
        
        # Target values - more coherent
        target_heart_rate = 60  # ~1 per second
        target_breath_rate = self.user_profile["optimal_breathing_rate"]  # PHI breaths per minute
        target_skin_conductance = 3
        target_eeg_alpha = 12
        target_eeg_theta = 7.4  # alpha/theta ~= PHI
        
        # Calculate gradual improvement based on session progress
        heart_rate = base_heart_rate - (base_heart_rate - target_heart_rate) * session_progress
        breath_rate = base_breath_rate - (base_breath_rate - target_breath_rate) * session_progress
        skin_conductance = base_skin_conductance - (base_skin_conductance - target_skin_conductance) * session_progress
        eeg_alpha = base_eeg_alpha + (target_eeg_alpha - base_eeg_alpha) * session_progress
        eeg_theta = base_eeg_theta + (target_eeg_theta - base_eeg_theta) * session_progress
        
        # Add some variation to make it more realistic
        # Use phi-based oscillation for natural variability
        variation = 0.05 * np.sin(session_time * PHI) + 0.03 * np.sin(session_time * PHI_PHI)
        
        heart_rate += variation * 5
        breath_rate += variation * 0.5
        skin_conductance += variation
        eeg_alpha += variation * 2
        eeg_theta += variation
        
        # Store metadata separately from the biofeedback data that will be sent to the system
        result = {
            "heart_rate": heart_rate,
            "breath_rate": breath_rate,
            "skin_conductance": skin_conductance,
            "eeg_alpha": eeg_alpha,
            "eeg_theta": eeg_theta
        }
        
        # Store metadata in the object for tracking
        self.current_metadata = {
            "timestamp": session_time,
            "progress": session_progress
        }
        
        return result
    
    def update_breath_pacer(self, breath_rate):
        """Update breath pacer visualization."""
        # This would connect to a visual or audio breath pacer
        cycle_time = 60.0 / breath_rate  # seconds per breath cycle
        
        # Calculate breath phase (0-1)
        phase = (time.time() % cycle_time) / cycle_time
        
        # Inhale from 0-0.382, hold from 0.382-0.5, exhale from 0.5-0.882, hold from 0.882-1.0
        # (using phi proportions)
        if phase < LAMBDA * 0.618:
            message = "Inhale..."
        elif phase < 0.5:
            message = "Hold..."
        elif phase < 0.5 + LAMBDA * 0.618:
            message = "Exhale..."
        else:
            message = "Hold..."
            
        self.breath_pacer = phase
        return message
    
    def run_session(self, duration_minutes=10, guidance_level="high"):
        """Run a meditation session."""
        print(f"\n{'-'*80}")
        print(f"Starting {duration_minutes}-minute meditation session for {self.user_name}")
        print(f"{'-'*80}")
        
        # Session parameters
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        update_interval = 2  # seconds between updates
        display_interval = 15  # seconds between status displays
        last_update = last_display = start_time
        
        # Session data
        self.session_data = []
        self.session_duration = duration_minutes
        self.coherence_history = []
        
        try:
            while time.time() < end_time:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = end_time - current_time
                session_progress = elapsed / (duration_minutes * 60)
                
                # Update system at regular intervals
                if current_time - last_update >= update_interval:
                    # Get simulated biofeedback
                    biofeedback = self.simulate_biofeedback(elapsed, session_progress)
                    
                    # Store data with metadata
                    data_point = {**biofeedback, **self.current_metadata}
                    self.session_data.append(data_point)
                    
                    # Update system with biofeedback
                    status = self.system.update(biofeedback)
                    self.coherence_history.append(status["system_coherence"])
                    
                    # Get meditation guidance
                    if guidance_level != "off":
                        breath_message = self.update_breath_pacer(biofeedback["breath_rate"])
                        self.current_guidance = self.get_meditation_guidance(status, guidance_level)
                    
                    last_update = current_time
                
                # Display status at regular intervals
                if current_time - last_display >= display_interval:
                    self.display_session_status(elapsed, remaining, session_progress)
                    last_display = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
        
        # Complete session
        self.complete_session()
        return self.session_summary()
    
    def display_session_status(self, elapsed, remaining, progress):
        """Display current session status."""
        minutes_elapsed = int(elapsed / 60)
        seconds_elapsed = int(elapsed % 60)
        minutes_remaining = int(remaining / 60)
        seconds_remaining = int(remaining % 60)
        
        coherence = self.coherence_history[-1] if self.coherence_history else 0
        
        print(f"\n{'-'*40}")
        print(f"Session Progress: {progress*100:.1f}%")
        print(f"Time: {minutes_elapsed}:{seconds_elapsed:02d} elapsed, {minutes_remaining}:{seconds_remaining:02d} remaining")
        print(f"System Coherence: {coherence:.4f}")
        
        if self.current_guidance:
            print(f"\nGuidance: {self.current_guidance}")
            
        # Show breath pacer visualization
        if self.breath_pacer is not None:
            self.display_breath_pacer()
    
    def display_breath_pacer(self):
        """Display a simple breath pacer visualization."""
        width = 40
        position = int(self.breath_pacer * width)
        
        pacer = "─" * width
        pacer = pacer[:position] + "●" + pacer[position+1:]
        
        print("\nBreath Pacer:")
        print(f"Inhale {pacer} Exhale")
    
    def get_meditation_guidance(self, status, level="medium"):
        """Generate meditation guidance based on system status."""
        coherence = status["consciousness_state"]["coherence"]
        presence = status["consciousness_state"]["presence"]
        system_coherence = status["system_coherence"]
        
        # Basic guidance based on coherence level
        if coherence < 0.4:
            basic_guidance = "Focus on your breath. Allow thoughts to pass without attachment."
        elif coherence < 0.6:
            basic_guidance = "Good. Deepen your breath slightly and relax your body further."
        elif coherence < 0.8:
            basic_guidance = "Excellent focus. Maintain awareness without straining."
        else:
            basic_guidance = "Perfect meditation state. Simply rest in awareness."
        
        # Return appropriate level of guidance
        if level == "low":
            return basic_guidance
        elif level == "medium":
            if presence < 0.5:
                return basic_guidance + " Bring more awareness to the present moment."
            else:
                return basic_guidance + " Your presence is strong, maintain this state."
        else:  # high
            # Include system coherence and specific recommendations
            if system_coherence < 0.5:
                return f"{basic_guidance} The field is still harmonizing with your state. " + \
                       "Try visualizing a golden light surrounding you."
            elif system_coherence < 0.7:
                return f"{basic_guidance} The field is resonating well. " + \
                       "Feel the connection between your awareness and the space around you."
            else:
                return f"{basic_guidance} Perfect field resonance achieved. " + \
                       "You and the field are in complete harmony."
    
    def complete_session(self):
        """Complete the meditation session."""
        print(f"\n{'-'*80}")
        print(f"Meditation session complete: {self.session_duration} minutes")
        
        # Create phi-resonance profile
        profile = self.system.interface.create_phi_resonance_profile(
            self.system.interface.feedback_history
        )
        self.system.interface.apply_phi_resonance_profile()
        
        # Save to user profile
        self.user_profile["phi_resonance_profile"] = profile
        self.user_profile["previous_sessions"] += 1
        
        # Calculate average coherence
        avg_coherence = np.mean(self.coherence_history) if self.coherence_history else 0
        self.user_profile["average_coherence"] = (
            self.user_profile["average_coherence"] * 0.8 + avg_coherence * 0.2
        )
        
        print(f"Final system coherence: {self.coherence_history[-1]:.4f}")
        print(f"Average session coherence: {avg_coherence:.4f}")
        print(f"Session data recorded: {len(self.session_data)} points")
        print(f"{'-'*80}")
        
        # Visualize session results
        self.visualize_session_results()
    
    def visualize_session_results(self):
        """Visualize the session results."""
        if not self.session_data or not self.coherence_history:
            print("No session data to visualize")
            return
        
        # Visualize fields
        self.system._visualize_fields()
        
        # Create plots if matplotlib is available
        try:
            # Extract data
            timestamps = [d["timestamp"] for d in self.session_data]
            minutes = [t / 60 for t in timestamps]
            heart_rates = [d["heart_rate"] for d in self.session_data]
            breath_rates = [d["breath_rate"] for d in self.session_data]
            skin_conductances = [d["skin_conductance"] for d in self.session_data]
            eeg_alphas = [d["eeg_alpha"] for d in self.session_data]
            eeg_thetas = [d["eeg_theta"] for d in self.session_data]
            
            # Create plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Plot biofeedback metrics
            axes[0].plot(minutes, heart_rates, 'r-', label='Heart Rate (bpm)')
            axes[0].plot(minutes, breath_rates, 'b-', label='Breath Rate (bpm)')
            axes[0].set_ylabel('Rate (per minute)')
            axes[0].set_title('Meditation Session Biofeedback')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot EEG data
            axes[1].plot(minutes, eeg_alphas, 'g-', label='Alpha Waves')
            axes[1].plot(minutes, eeg_thetas, 'm-', label='Theta Waves')
            axes[1].plot(minutes, [a/t if t > 0 else 0 for a, t in zip(eeg_alphas, eeg_thetas)], 
                       'k--', label='Alpha/Theta Ratio')
            axes[1].axhline(y=PHI, color='orange', linestyle=':', label=f'Phi ({PHI:.3f})')
            axes[1].set_ylabel('Amplitude / Ratio')
            axes[1].legend()
            axes[1].grid(True)
            
            # Plot coherence
            axes[2].plot(minutes, self.coherence_history, 'b-', label='System Coherence')
            axes[2].axhline(y=LAMBDA, color='orange', linestyle=':', label=f'Lambda ({LAMBDA:.3f})')
            axes[2].axhline(y=PHI_INVERSE, color='green', linestyle=':', label=f'Phi Inverse ({PHI_INVERSE:.3f})')
            axes[2].set_xlabel('Time (minutes)')
            axes[2].set_ylabel('Coherence')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"meditation_session_{int(time.time())}.png")
            print("Session visualization saved as PNG file")
            
        except ImportError:
            print("Matplotlib not available for visualization")
            return
    
    def session_summary(self):
        """Return a summary of the meditation session."""
        if not self.session_data or not self.coherence_history:
            return {"status": "No session data available"}
        
        # Collect summary data
        avg_coherence = np.mean(self.coherence_history)
        max_coherence = max(self.coherence_history)
        final_coherence = self.coherence_history[-1]
        
        # Calculate improvements
        first_biofeedback = self.session_data[0]
        last_biofeedback = self.session_data[-1]
        
        improvements = {
            "heart_rate_change": last_biofeedback["heart_rate"] - first_biofeedback["heart_rate"],
            "breath_rate_change": last_biofeedback["breath_rate"] - first_biofeedback["breath_rate"],
            "alpha_wave_change": last_biofeedback["eeg_alpha"] - first_biofeedback["eeg_alpha"],
            "alpha_theta_ratio_start": first_biofeedback["eeg_alpha"] / first_biofeedback["eeg_theta"] 
                                      if first_biofeedback["eeg_theta"] > 0 else 0,
            "alpha_theta_ratio_end": last_biofeedback["eeg_alpha"] / last_biofeedback["eeg_theta"]
                                    if last_biofeedback["eeg_theta"] > 0 else 0,
        }
        
        return {
            "status": "complete",
            "duration_minutes": self.session_duration,
            "data_points": len(self.session_data),
            "average_coherence": avg_coherence,
            "maximum_coherence": max_coherence,
            "final_coherence": final_coherence,
            "improvements": improvements,
            "user_profile_update": {
                "previous_sessions": self.user_profile["previous_sessions"],
                "average_coherence": self.user_profile["average_coherence"]
            }
        }


def main():
    """Main function to run the meditation enhancer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Meditation Enhancer")
    parser.add_argument("--user", type=str, default="User", 
                      help="User name for personalized meditation")
    parser.add_argument("--duration", type=int, default=5, 
                      help="Session duration in minutes")
    parser.add_argument("--guidance", type=str, choices=["off", "low", "medium", "high"],
                      default="medium", help="Level of meditation guidance")
    parser.add_argument("--dimensions", type=int, nargs=3, default=[34, 55, 34],
                      help="Field dimensions (default: 34 55 34)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Cascade⚡𓂧φ∞ Meditation Enhancer")
    print("=" * 80)
    
    # Create and run enhancer
    enhancer = MeditationEnhancer(
        user_name=args.user,
        field_dimensions=tuple(args.dimensions)
    )
    
    enhancer.run_session(
        duration_minutes=args.duration,
        guidance_level=args.guidance
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())