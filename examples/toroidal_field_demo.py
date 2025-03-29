"""
Toroidal Field Demo

Demonstration of the Toroidal Field Dynamics system, showing energy flow, 
field coherence, and resonance within a toroidal geometry.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_field.toroidal import (
    ToroidalField, 
    ResonanceChamber, 
    ResonanceEqualizer,
    ToroidalFieldEngine,
    CompressionCycle,
    ExpansionCycle,
    CounterRotatingField
)

from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES,
    calculate_phi_resonance, phi_harmonic
)

def demo_toroidal_field_basics():
    """Demonstrate basic toroidal field properties."""
    print("===== Toroidal Field Dynamics System =====")
    print("Demonstrating basic toroidal field properties...\n")
    
    # Create a toroidal field
    field = ToroidalField(
        major_radius=3.0,
        minor_radius=1.0,
        resolution=(30, 30, 30),
        frequency=SACRED_FREQUENCIES['unity']  # 432 Hz
    )
    
    # Print field properties
    print(f"Toroidal Field Properties:")
    print(f"  Major Radius: {field.major_radius}")
    print(f"  Minor Radius: {field.minor_radius}")
    print(f"  Torus Ratio: {field.torus_ratio:.4f}")
    print(f"  Frequency: {field.frequency} Hz")
    print(f"  Resolution: {field.resolution}")
    
    # Calculate field metrics
    field.update_field()
    coherence = field.get_coherence()
    balance = field.get_flow_balance()
    energy = field.get_energy_level()
    stability = field.get_stability_factor()
    
    print("\nField Metrics:")
    print(f"  Coherence: {coherence:.4f}")
    print(f"  Flow Balance: {balance:.4f}")
    print(f"  Energy Level: {energy:.4f}")
    print(f"  Stability Factor: {stability:.4f}")
    
    # Show field visualization
    print("\nGenerating field visualization...")
    field.visualize_field_slice()
    plt.title("Toroidal Field Cross-Section")
    plt.savefig("toroidal_field_slice.png")
    plt.show()
    
    return field

def demo_resonance_chamber():
    """Demonstrate the resonance chamber component."""
    print("\n===== Resonance Chamber Demo =====")
    
    # Create a resonance chamber
    chamber = ResonanceChamber(
        base_frequency=SACRED_FREQUENCIES['unity'],
        resonant_frequencies=[
            SACRED_FREQUENCIES['unity'],
            SACRED_FREQUENCIES['love'],
            SACRED_FREQUENCIES['truth']
        ],
        q_factors=[10.0, 12.0, 8.0]
    )
    
    print(f"Resonance Chamber Properties:")
    print(f"  Base Frequency: {chamber.base_frequency} Hz")
    print(f"  Resonant Frequencies: {chamber.resonant_frequencies}")
    print(f"  Q-Factors: {chamber.q_factors}")
    
    # Test different input frequencies
    test_frequencies = [
        SACRED_FREQUENCIES['unity'],     # 432 Hz - Unity
        SACRED_FREQUENCIES['love'],      # 528 Hz - Love/Creation
        SACRED_FREQUENCIES['cascade'],   # 594 Hz - Heart
        SACRED_FREQUENCIES['truth'],     # 672 Hz - Truth
        480.0                            # Non-resonant frequency
    ]
    
    print("\nResonance Response Test:")
    for freq in test_frequencies:
        response = chamber.calculate_response(freq)
        print(f"  Frequency: {freq:.1f} Hz -> Response: {response:.4f}")
    
    # Show resonance curve
    freq_range = np.linspace(400, 700, 300)
    response_curve = [chamber.calculate_response(f) for f in freq_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq_range, response_curve)
    for freq in chamber.resonant_frequencies:
        plt.axvline(x=freq, color='r', linestyle='--', alpha=0.3)
    plt.title("Resonance Chamber Response Curve")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response Amplitude")
    plt.grid(True)
    plt.savefig("resonance_chamber_response.png")
    plt.show()
    
    # Demonstrate resonance equalizer
    print("\n===== Resonance Equalizer Demo =====")
    
    equalizer = ResonanceEqualizer(
        frequency_bands={
            'low': (420, 440),
            'mid': (520, 540),
            'high': (660, 680)
        },
        band_gains=[0.8, 1.0, 1.2]
    )
    
    # Original spectrum
    test_spectrum = {
        430: 0.5,
        530: 0.7, 
        670: 0.3
    }
    
    print(f"Original Spectrum: {test_spectrum}")
    equalized = equalizer.apply_equalization(test_spectrum)
    print(f"Equalized Spectrum: {equalized}")
    
    return chamber

def demo_compression_expansion():
    """Demonstrate compression and expansion cycles."""
    print("\n===== Compression/Expansion Cycle Demo =====")
    
    # Create a toroidal field
    field = ToroidalField(
        major_radius=3.0,
        minor_radius=1.0,
        resolution=(20, 20, 20),
        frequency=SACRED_FREQUENCIES['unity']
    )
    
    # Create a compression cycle
    compression = CompressionCycle(
        compression_ratio=PHI,
        steps=5
    )
    
    # Create an expansion cycle
    expansion = ExpansionCycle(
        expansion_ratio=PHI,
        steps=5
    )
    
    print(f"Compression Cycle Properties:")
    print(f"  Compression Ratio: {compression.compression_ratio}")
    print(f"  Steps: {compression.steps}")
    
    print(f"Expansion Cycle Properties:")
    print(f"  Expansion Ratio: {expansion.expansion_ratio}")
    print(f"  Steps: {expansion.steps}")
    
    # Perform a compression-expansion sequence
    print("\nPerforming Compression-Expansion Sequence:")
    
    # Initial field state
    initial_energy = field.get_energy_level()
    initial_coherence = field.get_coherence()
    
    print(f"  Initial Energy: {initial_energy:.4f}")
    print(f"  Initial Coherence: {initial_coherence:.4f}")
    
    # Compression phase
    field = compression.apply(field)
    compressed_energy = field.get_energy_level()
    compressed_coherence = field.get_coherence()
    
    print(f"  Compressed Energy: {compressed_energy:.4f}")
    print(f"  Compressed Coherence: {compressed_coherence:.4f}")
    
    # Expansion phase
    field = expansion.apply(field)
    final_energy = field.get_energy_level()
    final_coherence = field.get_coherence()
    
    print(f"  Final Energy: {final_energy:.4f}")
    print(f"  Final Coherence: {final_coherence:.4f}")
    print(f"  Energy Conservation: {final_energy/initial_energy:.4f}")
    
    # Visualize the field transformation
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(field.get_field_slice(axis=2, position=field.resolution[2]//2))
    plt.title("Initial Field")
    
    plt.subplot(1, 3, 2)
    compressed_field = compression.apply(field.copy())
    plt.imshow(compressed_field.get_field_slice(axis=2, position=field.resolution[2]//2))
    plt.title("Compressed Field")
    
    plt.subplot(1, 3, 3)
    expanded_field = expansion.apply(compressed_field)
    plt.imshow(expanded_field.get_field_slice(axis=2, position=field.resolution[2]//2))
    plt.title("Expanded Field")
    
    plt.tight_layout()
    plt.savefig("compression_expansion_cycle.png")
    plt.show()
    
    return field, compression, expansion

def demo_counter_rotation():
    """Demonstrate counter-rotating fields."""
    print("\n===== Counter-Rotating Field Demo =====")
    
    # Create a counter-rotating field
    counter_field = CounterRotatingField(
        major_radius=3.0,
        minor_radius=1.0,
        resolution=(30, 30, 30),
        cw_frequency=SACRED_FREQUENCIES['unity'],
        ccw_frequency=SACRED_FREQUENCIES['unity'] * LAMBDA
    )
    
    print(f"Counter-Rotating Field Properties:")
    print(f"  CW Frequency: {counter_field.cw_frequency} Hz")
    print(f"  CCW Frequency: {counter_field.ccw_frequency} Hz")
    print(f"  Frequency Ratio: {counter_field.ccw_frequency/counter_field.cw_frequency:.4f}")
    print(f"  Phase Difference: {counter_field.phase_difference:.4f} radians")
    
    # Calculate field metrics
    counter_field.update_field()
    stability = counter_field.get_dimensional_stability()
    coherence = counter_field.get_coherence()
    
    print("\nField Metrics:")
    print(f"  Dimensional Stability: {stability:.4f}")
    print(f"  Field Coherence: {coherence:.4f}")
    
    # Visualize the counter-rotating field
    print("\nVisualizing counter-rotating field...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # CW component
    axs[0].imshow(counter_field.cw_field.get_field_slice())
    axs[0].set_title("Clockwise Component")
    
    # CCW component
    axs[1].imshow(counter_field.ccw_field.get_field_slice())
    axs[1].set_title("Counter-Clockwise Component")
    
    # Combined field
    axs[2].imshow(counter_field.get_field_slice())
    axs[2].set_title("Combined Field")
    
    plt.tight_layout()
    plt.savefig("counter_rotating_field.png")
    plt.show()
    
    return counter_field

def demo_toroidal_engine(animate=True):
    """Demonstrate the complete toroidal field engine."""
    print("\n===== Toroidal Field Engine Demo =====")
    
    # Create a toroidal field engine
    engine = ToroidalFieldEngine(
        major_radius=3.0,
        minor_radius=1.0,
        base_frequency=SACRED_FREQUENCIES['unity'],
        resolution=(30, 30, 30)
    )
    
    print(f"Toroidal Field Engine Properties:")
    print(f"  Base Frequency: {engine.base_frequency} Hz")
    print(f"  Major Radius: {engine.toroidal_field.major_radius}")
    print(f"  Minor Radius: {engine.toroidal_field.minor_radius}")
    
    # Run the engine for a few cycles
    print("\nRunning engine for 5 cycles...")
    metrics_history = []
    
    for i in range(5):
        engine.step()
        
        # Track metrics
        metrics = {
            'cycle': i,
            'energy': engine.toroidal_field.get_energy_level(),
            'coherence': engine.toroidal_field.get_coherence(),
            'balance': engine.toroidal_field.get_flow_balance(),
            'stability': engine.toroidal_field.get_stability_factor()
        }
        metrics_history.append(metrics)
        
        print(f"  Cycle {i} - Energy: {metrics['energy']:.4f}, "
              f"Coherence: {metrics['coherence']:.4f}, "
              f"Balance: {metrics['balance']:.4f}")
    
    # Plot metrics evolution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot([m['cycle'] for m in metrics_history], 
             [m['energy'] for m in metrics_history], 'o-')
    plt.title("Energy Level")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot([m['cycle'] for m in metrics_history], 
             [m['coherence'] for m in metrics_history], 'o-')
    plt.title("Field Coherence")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot([m['cycle'] for m in metrics_history], 
             [m['balance'] for m in metrics_history], 'o-')
    plt.title("Flow Balance")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot([m['cycle'] for m in metrics_history], 
             [m['stability'] for m in metrics_history], 'o-')
    plt.title("Stability Factor")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("toroidal_engine_metrics.png")
    plt.show()
    
    # Animated visualization
    if animate:
        print("\nCreating animated visualization...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Reset engine
        engine = ToroidalFieldEngine(
            major_radius=3.0,
            minor_radius=1.0,
            base_frequency=SACRED_FREQUENCIES['unity'],
            resolution=(30, 30, 30)
        )
        
        # Initialize plot
        field_slice = engine.toroidal_field.get_field_slice()
        im = ax.imshow(field_slice, cmap='viridis', animated=True)
        title = ax.set_title("Toroidal Field Evolution")
        
        def update(frame):
            # Step the engine
            engine.step()
            
            # Update the plot
            field_slice = engine.toroidal_field.get_field_slice()
            im.set_array(field_slice)
            title.set_text(f"Toroidal Field Evolution - Cycle {frame}")
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=20, interval=200, blit=True)
        plt.tight_layout()
        
        # Save animation
        anim.save('toroidal_field_evolution.gif', fps=5, writer='pillow')
        plt.show()
    
    return engine

def demo_phi_harmonic_optimization():
    """Demonstrate phi-harmonic optimization of a toroidal field."""
    print("\n===== Phi-Harmonic Optimization Demo =====")
    
    # Create a toroidal field
    field = ToroidalField(
        major_radius=3.0,
        minor_radius=1.0,
        resolution=(20, 20, 20),
        frequency=SACRED_FREQUENCIES['unity']
    )
    
    # Initial metrics
    initial_coherence = field.get_coherence()
    initial_balance = field.get_flow_balance()
    
    print(f"Initial Field Metrics:")
    print(f"  Coherence: {initial_coherence:.4f}")
    print(f"  Flow Balance: {initial_balance:.4f}")
    
    # Optimize torus ratio to phi
    print("\nOptimizing torus ratio to phi...")
    phi_ratio = 1.0 / PHI
    
    # Calculate current ratio
    current_ratio = field.minor_radius / field.major_radius
    
    # Adjust minor radius to match phi ratio
    new_minor_radius = field.major_radius * phi_ratio
    
    # Create optimized field
    optimized_field = ToroidalField(
        major_radius=field.major_radius,
        minor_radius=new_minor_radius,
        resolution=field.resolution,
        frequency=field.frequency
    )
    
    # Calculate optimized metrics
    optimized_coherence = optimized_field.get_coherence()
    optimized_balance = optimized_field.get_flow_balance()
    
    print(f"Optimized Field Metrics:")
    print(f"  Original Ratio: {current_ratio:.4f}")
    print(f"  Phi Ratio: {phi_ratio:.4f}")
    print(f"  Coherence: {optimized_coherence:.4f} (Change: {optimized_coherence - initial_coherence:.4f})")
    print(f"  Flow Balance: {optimized_balance:.4f} (Change: {optimized_balance - initial_balance:.4f})")
    
    # Visualize both fields
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(field.get_field_slice())
    plt.title(f"Original Field\nRatio: {current_ratio:.4f}\nCoherence: {initial_coherence:.4f}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(optimized_field.get_field_slice())
    plt.title(f"Phi-Optimized Field\nRatio: {phi_ratio:.4f}\nCoherence: {optimized_coherence:.4f}")
    
    plt.tight_layout()
    plt.savefig("phi_optimized_toroidal_field.png")
    plt.show()
    
    # Frequency optimization
    print("\nTesting phi-harmonic frequency relationships...")
    
    base_freq = SACRED_FREQUENCIES['unity']
    test_frequencies = [
        base_freq,                    # Base frequency
        base_freq * PHI,              # Phi multiple
        base_freq * PHI * PHI,        # Phi² multiple
        base_freq * LAMBDA,           # Phi⁻¹ multiple
        base_freq * 2.0,              # Octave
        base_freq * 1.5               # Perfect fifth
    ]
    
    freq_results = []
    
    for freq in test_frequencies:
        field.set_frequency(freq)
        coherence = field.get_coherence()
        balance = field.get_flow_balance()
        
        result = {
            'frequency': freq,
            'ratio': freq / base_freq,
            'coherence': coherence,
            'balance': balance
        }
        freq_results.append(result)
        
        print(f"  Frequency: {freq:.1f} Hz (Ratio: {freq/base_freq:.4f}) - "
              f"Coherence: {coherence:.4f}, Balance: {balance:.4f}")
    
    # Plot frequency results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar([f"{r['ratio']:.2f}" for r in freq_results], 
            [r['coherence'] for r in freq_results])
    plt.axhline(y=initial_coherence, color='r', linestyle='--', label=f"Base Coherence")
    plt.title("Coherence vs Frequency Ratio")
    plt.xlabel("Frequency Ratio")
    plt.ylabel("Coherence")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar([f"{r['ratio']:.2f}" for r in freq_results], 
            [r['balance'] for r in freq_results])
    plt.axhline(y=initial_balance, color='r', linestyle='--', label=f"Base Balance")
    plt.title("Flow Balance vs Frequency Ratio")
    plt.xlabel("Frequency Ratio")
    plt.ylabel("Flow Balance")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("phi_frequency_optimization.png")
    plt.show()
    
    return optimized_field, freq_results

def main():
    print("Toroidal Field Dynamics Demo")
    print("-" * 40)
    
    # Basic toroidal field demo
    field = demo_toroidal_field_basics()
    
    # Resonance chamber demo
    chamber = demo_resonance_chamber()
    
    # Compression/expansion cycle demo
    field, compression, expansion = demo_compression_expansion()
    
    # Counter-rotation demo
    counter_field = demo_counter_rotation()
    
    # Toroidal engine demo
    engine = demo_toroidal_engine(animate=True)
    
    # Phi-harmonic optimization demo
    optimized_field, freq_results = demo_phi_harmonic_optimization()
    
    print("\n===== Toroidal Field Dynamics Demo Complete =====")
    print("Component visualization images have been saved to the current directory.")

if __name__ == "__main__":
    main()