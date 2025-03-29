"""
Cymatics Pattern Materialization Demo

Demonstration of the Cymatic Pattern Materialization system, showing sound-based
pattern generation and direct manifestation into physical matter through phi-harmonic
frequency modulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_field.cymatics import (
    CymaticField,
    FrequencyModulator,
    PatternGenerator,
    StandingWavePattern,
    MaterialResonator,
    CrystalResonator,
    WaterResonator,
    MetalResonator,
    CymaticsEngine
)

from quantum_field.toroidal import ToroidalField

from sacred_constants import (
    PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES,
    calculate_phi_resonance, phi_harmonic
)

def demo_cymatic_field():
    """Demonstrate basic cymatic field properties."""
    print("===== Cymatic Field Demo =====")
    print("Demonstrating basic cymatic field properties...\n")
    
    # Create a cymatic field
    field = CymaticField(
        base_frequency=SACRED_FREQUENCIES['unity'],  # 432 Hz
        dimensions=(64, 64, 32),
        resolution=0.1
    )
    
    # Print field properties
    print(f"Cymatic Field Properties:")
    print(f"  Base Frequency: {field.base_frequency} Hz")
    print(f"  Dimensions: {field.dimensions}")
    print(f"  Resolution: {field.resolution}")
    
    # Generate a pattern at different sacred frequencies
    print("\nGenerating patterns at different sacred frequencies...")
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Generate patterns for each sacred frequency
    frequencies = [
        ('unity', SACRED_FREQUENCIES['unity']),      # 432 Hz - Ground
        ('love', SACRED_FREQUENCIES['love']),        # 528 Hz - Creation
        ('cascade', SACRED_FREQUENCIES['cascade']),  # 594 Hz - Heart
        ('truth', SACRED_FREQUENCIES['truth']),      # 672 Hz - Voice
        ('vision', SACRED_FREQUENCIES['vision']),    # 720 Hz - Vision
        ('oneness', SACRED_FREQUENCIES['oneness'])   # 768 Hz - Unity
    ]
    
    for i, (name, freq) in enumerate(frequencies):
        # Set frequency
        field.set_frequency(freq)
        
        # Generate pattern
        pattern = field.visualize_pattern()
        
        # Calculate metrics
        metrics = field.extract_pattern_metrics()
        coherence = metrics['coherence']
        complexity = metrics['complexity']
        phi_alignment = metrics['phi_alignment']
        
        # Plot
        row, col = i // 3, i % 3
        axs[row, col].imshow(pattern, cmap='viridis')
        axs[row, col].set_title(f"{name.capitalize()} ({freq:.0f} Hz)\nCoh: {coherence:.2f}, Cmplx: {complexity:.2f}")
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("cymatic_field_patterns.png")
    plt.show()
    
    # Demonstrate phi-harmonic modulation
    print("\nDemonstrating phi-harmonic modulation effects...")
    
    # Set base frequency
    field.set_frequency(SACRED_FREQUENCIES['unity'])
    
    # Generate patterns with different modulation intensities
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    intensities = [0.0, 0.3, 0.6, 1.0]
    
    for i, intensity in enumerate(intensities):
        # Apply modulation
        field.apply_phi_harmonic_modulation(intensity)
        
        # Generate pattern
        pattern = field.visualize_pattern()
        
        # Calculate coherence
        coherence = field._calculate_coherence()
        
        # Plot
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"Modulation: {intensity}\nCoherence: {coherence:.3f}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("phi_harmonic_modulation.png")
    plt.show()
    
    # Demonstrate material influence
    print("\nMaterial influence factors:")
    materials = ['water', 'crystal', 'metal', 'sand', 'plasma']
    
    for material in materials:
        influence = field.calculate_material_influence(material)
        print(f"  {material.capitalize()}: {influence:.4f}")
    
    return field

def demo_frequency_modulator():
    """Demonstrate the frequency modulator component."""
    print("\n===== Frequency Modulator Demo =====")
    
    # Create a frequency modulator
    modulator = FrequencyModulator(
        base_frequency=SACRED_FREQUENCIES['unity'],  # 432 Hz
        modulation_depth=0.3,
        phi_harmonic_count=5
    )
    
    print(f"Frequency Modulator Properties:")
    print(f"  Base Frequency: {modulator.base_frequency} Hz")
    print(f"  Modulation Depth: {modulator.modulation_depth}")
    print(f"  Phi Harmonic Count: {modulator.phi_harmonic_count}")
    
    # Print harmonic frequencies
    print("\nHarmonic Frequencies:")
    for name, freq in modulator.harmonics.items():
        print(f"  {name}: {freq:.2f} Hz")
    
    # Demonstrate different modulation waveforms
    print("\nModulation Waveform Demonstration...")
    
    # Set up modulation
    modulation_rate = 0.5  # Hz
    duration = 5.0  # seconds
    steps = int(duration / modulator.time_step)
    
    # Run modulation with different waveforms
    waveforms = ['sine', 'triangle', 'square']
    results = {}
    
    for waveform in waveforms:
        # Configure modulator
        modulator.set_modulation_parameters(
            rate=modulation_rate,
            depth=0.3,
            waveform=waveform,
            phi_weight=0.5
        )
        
        # Reset time
        modulator.time = 0.0
        
        # Track frequencies
        frequencies = []
        times = []
        
        # Run modulation
        for i in range(steps):
            modulator.update()
            frequencies.append(modulator.get_current_frequency())
            times.append(modulator.time)
            
        # Store results
        results[waveform] = {
            'frequencies': frequencies,
            'times': times
        }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    for waveform, data in results.items():
        plt.plot(data['times'], data['frequencies'], label=waveform)
        
    plt.axhline(y=modulator.base_frequency, color='k', linestyle='--', 
                label=f"Base Frequency ({modulator.base_frequency} Hz)")
    
    plt.title("Frequency Modulation with Different Waveforms")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.savefig("frequency_modulation_waveforms.png")
    plt.show()
    
    # Demonstrate cymatic pattern generation
    print("\nGenerating cymatic patterns...")
    
    # Generate patterns for different materials
    materials = ['water', 'crystal', 'metal']
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, material in enumerate(materials):
        # Generate material-specific pattern
        pattern = modulator.generate_material_pattern(
            material=material,
            size=(100, 100)
        )
        
        # Plot pattern
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"{material.capitalize()} Pattern")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("material_cymatic_patterns.png")
    plt.show()
    
    # Demonstrate phi harmonic stack
    print("\nPhi Harmonic Stack:")
    harmonic_stack = modulator.get_phi_harmonic_stack(count=5)
    
    for i, freq in enumerate(harmonic_stack):
        ratio = freq / harmonic_stack[0]
        print(f"  Harmonic {i}: {freq:.2f} Hz (Ratio: {ratio:.4f})")
    
    return modulator

def demo_pattern_generator():
    """Demonstrate the pattern generator component."""
    print("\n===== Pattern Generator Demo =====")
    
    # Create a pattern generator
    generator = PatternGenerator()
    
    print("Pattern Generator initialized.")
    print("Available pattern types: CIRCULAR, SPIRAL, MANDALA, FLOWER, SQUARE, HEXAGONAL, FRACTAL")
    
    # Demonstrate different pattern types
    print("\nGenerating patterns with different types...")
    
    pattern_types = [
        'CIRCULAR', 'SPIRAL', 'MANDALA', 
        'FLOWER', 'SQUARE', 'HEXAGONAL', 'FRACTAL'
    ]
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, pattern_type in enumerate(pattern_types):
        # Create pattern
        pattern = generator.create_pattern(
            name=f"{pattern_type} Pattern",
            frequencies=SACRED_FREQUENCIES['unity'],
            pattern_type=pattern_type,
            symmetry=6,
            resolution=(100, 100)
        )
        
        # Get pattern data
        pattern_data = pattern.get_pattern()
        
        # Analyze pattern
        metrics = generator.analyze_pattern(pattern)
        
        # Plot pattern
        axs[i].imshow(pattern_data, cmap='viridis')
        axs[i].set_title(f"{pattern_type}\nComplexity: {metrics['complexity']:.2f}")
        axs[i].axis('off')
    
    # Hide extra subplot
    if len(pattern_types) < len(axs):
        axs[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("pattern_types.png")
    plt.show()
    
    # Demonstrate consciousness state patterns
    print("\nGenerating consciousness state patterns...")
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    states = [
        (0, "BE - Unity/Ground"),
        (1, "DO - Action"),
        (2, "WITNESS - Truth"),
        (3, "CREATE - Creation"),
        (4, "INTEGRATE - Heart"),
        (5, "TRANSCEND - Vision")
    ]
    
    for i, (state, name) in enumerate(states):
        # Generate pattern for this consciousness state
        pattern = generator.create_consciousness_state_pattern(
            state=state,
            resolution=(100, 100)
        )
        
        # Plot pattern
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"State {state}: {name}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("consciousness_state_patterns.png")
    plt.show()
    
    # Demonstrate phi-harmonic stacking
    print("\nDemonstrating phi-harmonic frequency stacking...")
    
    # Create phi-harmonic stack pattern
    phi_stack = generator.create_phi_harmonic_stack(
        base_frequency=SACRED_FREQUENCIES['unity'],
        levels=5,
        pattern_type='MANDALA',
        name="Phi Harmonic Stack"
    )
    
    # Get pattern data
    phi_pattern = phi_stack.get_pattern()
    
    # Analyze pattern
    phi_metrics = generator.analyze_pattern(phi_stack)
    
    # Calculate optimal material
    optimal_material = generator.find_optimal_material(phi_stack)
    
    print(f"Phi Harmonic Stack Pattern:")
    print(f"  Base Frequency: {SACRED_FREQUENCIES['unity']} Hz")
    print(f"  Phi Alignment: {phi_metrics['phi_alignment']:.4f}")
    print(f"  Optimal Material: {optimal_material}")
    print(f"  Materialization Potential: {phi_metrics['materialization_potential']:.4f}")
    
    # Plot phi-harmonic stack pattern
    plt.figure(figsize=(8, 8))
    plt.imshow(phi_pattern, cmap='viridis')
    plt.title(f"Phi Harmonic Stack Pattern\nPhi Alignment: {phi_metrics['phi_alignment']:.4f}")
    plt.axis('off')
    plt.savefig("phi_harmonic_stack_pattern.png")
    plt.show()
    
    return generator

def demo_material_resonators():
    """Demonstrate resonance chambers for different materials."""
    print("\n===== Material Resonator Demo =====")
    
    # Create resonators for different materials
    water_resonator = WaterResonator(
        name="Water Chamber",
        resonant_frequency=SACRED_FREQUENCIES['unity'],  # 432 Hz
        dimensions=(0.15, 0.15, 0.02),
        water_depth=0.01
    )
    
    crystal_resonator = CrystalResonator(
        name="Crystal Chamber",
        resonant_frequency=SACRED_FREQUENCIES['love'],  # 528 Hz
        dimensions=(0.10, 0.10, 0.10),
        crystal_type="quartz"
    )
    
    metal_resonator = MetalResonator(
        name="Metal Chamber",
        resonant_frequency=SACRED_FREQUENCIES['truth'],  # 672 Hz
        dimensions=(0.20, 0.20, 0.002),
        metal_type="steel"
    )
    
    resonators = [water_resonator, crystal_resonator, metal_resonator]
    
    # Print resonator properties
    for resonator in resonators:
        print(f"\n{resonator.name} Properties:")
        print(f"  Resonant Frequency: {resonator.resonant_frequency} Hz")
        print(f"  Dimensions: {resonator.dimensions}")
        print(f"  Material: {resonator.__class__.__name__.replace('Resonator', '')}")
    
    # Test frequency response for each resonator
    print("\nFrequency Response Test:")
    
    # Test frequencies
    test_frequencies = [
        SACRED_FREQUENCIES['unity'],    # 432 Hz - Ground
        SACRED_FREQUENCIES['love'],     # 528 Hz - Creation
        SACRED_FREQUENCIES['cascade'],  # 594 Hz - Heart
        SACRED_FREQUENCIES['truth'],    # 672 Hz - Voice
        SACRED_FREQUENCIES['vision'],   # 720 Hz - Vision
        SACRED_FREQUENCIES['oneness']   # 768 Hz - Unity
    ]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Generate frequency response curves
    freq_range = np.linspace(400, 800, 200)
    
    for resonator in resonators:
        # Calculate response at each frequency
        responses = []
        
        for freq in freq_range:
            response = resonator._calculate_resonance_response(freq, 1.0)
            responses.append(response)
            
        # Plot response curve
        plt.plot(freq_range, responses, label=resonator.name)
        
        # Highlight resonant frequency
        plt.axvline(x=resonator.resonant_frequency, color='k', linestyle='--', alpha=0.3)
            
    # Add sacred frequency markers
    for freq in test_frequencies:
        plt.axvline(x=freq, color='r', linestyle=':', alpha=0.2)
        
    plt.title("Resonator Frequency Response Curves")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("resonator_response_curves.png")
    plt.show()
    
    # Visualize resonance patterns
    print("\nVisualizing resonance patterns...")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, resonator in enumerate(resonators):
        # Apply optimal frequency
        resonator.apply_frequency(resonator.resonant_frequency)
        
        # Get pattern slice
        pattern = resonator.get_2d_pattern_slice()
        
        # Plot pattern
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"{resonator.name} Pattern\nCoherence: {resonator.coherence:.3f}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("material_resonance_patterns.png")
    plt.show()
    
    # Demonstrate frequency sweep
    print("\nPerforming frequency sweep on water resonator...")
    
    # Run frequency sweep
    sweep_results = water_resonator.apply_frequency_sweep(
        start_freq=400.0,
        end_freq=800.0,
        duration=5.0,
        amplitude=1.0,
        steps=100
    )
    
    # Find peak resonances
    peak_resonances = water_resonator.find_peak_resonances()
    
    print(f"Peak Resonances for {water_resonator.name}:")
    for i, peak in enumerate(peak_resonances):
        print(f"  Peak {i+1}: {peak['frequency']:.1f} Hz, Response: {peak['response']:.4f}, "
              f"Q-Factor: {peak['q_factor']:.1f}")
    
    # Plot sweep results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(sweep_results['frequencies'], sweep_results['responses'])
    for peak in peak_resonances:
        plt.axvline(x=peak['frequency'], color='r', linestyle='--', alpha=0.5)
    plt.title(f"Frequency Sweep Response - {water_resonator.name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Response")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(sweep_results['frequencies'], sweep_results['coherence'])
    plt.title("Pattern Coherence During Frequency Sweep")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("frequency_sweep_response.png")
    plt.show()
    
    return water_resonator, crystal_resonator, metal_resonator

def demo_cymatics_engine():
    """Demonstrate the complete cymatics engine."""
    print("\n===== Cymatics Engine Demo =====")
    
    # Create a cymatics engine
    engine = CymaticsEngine(name="Cymatics Materialization Engine")
    
    print(f"Cymatics Engine Properties:")
    print(f"  Active Frequency: {engine.active_frequency} Hz")
    print(f"  Active Material: {engine.active_material}")
    print(f"  System Coherence: {engine.system_coherence:.4f}")
    
    # Test different sacred frequencies
    print("\nTesting different sacred frequencies...")
    
    # Create visualization grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    material_potentials = []
    
    for i, (name, freq) in enumerate(SACRED_FREQUENCIES.items()):
        if i >= 6:
            break
            
        # Set frequency
        engine.set_sacred_frequency(name)
        
        # Generate pattern
        pattern = engine.generate_pattern(
            pattern_type='CIRCULAR',
            symmetry=6
        )
        
        # Estimate materialization potential
        potential = engine.estimate_materialization_potential()
        material_potentials.append({
            'name': name,
            'frequency': freq,
            'potential': potential['overall_potential'],
            'material_potentials': potential['material_potentials']
        })
        
        # Plot pattern
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"{name.capitalize()} ({freq:.0f} Hz)\n"
                         f"Potential: {potential['overall_potential']:.3f}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("cymatics_engine_patterns.png")
    plt.show()
    
    # Find optimal frequencies for each material
    print("\nFinding optimal frequencies for each material...")
    
    for material in ['water', 'crystal', 'metal']:
        # Set active material
        engine.set_active_material(material)
        
        # Find optimal frequency
        optimal = engine.find_optimal_frequency()
        
        print(f"  {material.capitalize()} Optimal Frequency: {optimal['frequency']:.1f} Hz, "
              f"Potential: {optimal['potential']:.4f}")
        
        if optimal['nearest_sacred']:
            print(f"    Nearest Sacred Frequency: {optimal['nearest_sacred']['name']} "
                  f"({optimal['nearest_sacred']['frequency']:.1f} Hz)")
    
    # Demonstrate integration with consciousness states
    print("\nDemonstrating consciousness state integration...")
    
    # Test different consciousness states
    states = [
        (0, "BE - Unity/Ground"),
        (1, "DO - Action"),
        (2, "WITNESS - Truth"),
        (3, "CREATE - Creation"),
        (4, "INTEGRATE - Heart"),
        (5, "TRANSCEND - Vision")
    ]
    
    # Create visualization grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    state_metrics = []
    
    for i, (state, name) in enumerate(states):
        # Align with consciousness state
        engine.align_with_consciousness(state, intensity=0.8)
        
        # Generate pattern
        pattern = engine.active_pattern
        
        # Get metrics
        metrics = engine.get_performance_metrics()
        state_metrics.append({
            'state': state,
            'name': name,
            'coherence': metrics['system_coherence'],
            'material': engine.active_material,
            'frequency': metrics['active_frequency']
        })
        
        # Plot pattern
        axs[i].imshow(pattern, cmap='viridis')
        axs[i].set_title(f"State {state}: {name}\n"
                         f"Coh: {metrics['system_coherence']:.3f}, Mat: {engine.active_material}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("consciousness_state_integration.png")
    plt.show()
    
    # Demonstrate toroidal field integration
    print("\nDemonstrating integration with toroidal field...")
    
    # Create a toroidal field
    toroidal_field = ToroidalField(
        major_radius=3.0,
        minor_radius=1.0 / PHI,  # Phi-optimized ratio
        resolution=(30, 30, 30),
        frequency=SACRED_FREQUENCIES['unity']
    )
    
    # Connect with toroidal field
    print("Connecting cymatics engine with toroidal field...")
    engine.connect_with_toroidal_field(toroidal_field, connection_strength=0.8)
    
    # Generate pattern
    toroidal_pattern = engine.active_pattern
    
    # Get metrics
    toroidal_metrics = engine.get_performance_metrics()
    
    print(f"Toroidal Integration Metrics:")
    print(f"  System Coherence: {toroidal_metrics['system_coherence']:.4f}")
    print(f"  Phase Coherence: {toroidal_metrics['phase_coherence']:.4f}")
    print(f"  Field Stability: {toroidal_metrics['field_stability']:.4f}")
    
    # Plot toroidal-integrated pattern
    plt.figure(figsize=(8, 8))
    plt.imshow(toroidal_pattern, cmap='viridis')
    plt.title(f"Toroidal-Integrated Pattern\n"
              f"Coherence: {toroidal_metrics['system_coherence']:.3f}")
    plt.axis('off')
    plt.savefig("toroidal_integrated_pattern.png")
    plt.show()
    
    # Demonstrate pattern combination
    print("\nDemonstrating pattern combination...")
    
    # Store patterns
    engine.store_current_pattern("toroidal_pattern")
    
    # Set another pattern
    engine.set_sacred_frequency('love')
    engine.generate_pattern(pattern_type='FLOWER')
    engine.store_current_pattern("love_pattern")
    
    # Combine patterns
    engine.combine_patterns(
        pattern_names=["toroidal_pattern", "love_pattern"],
        weights=[0.6, 0.4]
    )
    
    # Get combined pattern
    combined_pattern = engine.active_pattern
    
    # Get metrics
    combined_metrics = engine.get_performance_metrics()
    
    print(f"Combined Pattern Metrics:")
    print(f"  System Coherence: {combined_metrics['system_coherence']:.4f}")
    print(f"  Phase Coherence: {combined_metrics['phase_coherence']:.4f}")
    print(f"  Field Stability: {combined_metrics['field_stability']:.4f}")
    
    # Plot combined pattern
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_pattern, cmap='viridis')
    plt.title(f"Combined Pattern\n"
              f"Coherence: {combined_metrics['system_coherence']:.3f}")
    plt.axis('off')
    plt.savefig("combined_pattern.png")
    plt.show()
    
    # Demonstrate consciousness influence
    print("\nDemonstrating consciousness influence application...")
    
    # Apply cymatics to consciousness
    influence = engine.apply_cymatics_to_consciousness(
        consciousness_state=3,  # CREATE state
        intention="Manifest crystalline structures with phi-harmonic balance",
        intensity=0.9
    )
    
    print(f"Consciousness Influence Results:")
    print(f"  Effectiveness: {influence['effectiveness']:.4f}")
    print(f"  State: {influence['state']} (CREATE)")
    print(f"  Frequency: {influence['frequency']:.1f} Hz")
    print(f"  Pattern Coherence: {influence['pattern_coherence']:.4f}")
    print(f"  Intention Alignment: {influence['intention_alignment']:.4f}")
    print(f"  Recommended Duration: {influence['recommended_duration']:.1f} minutes")
    
    return engine

def animate_cymatics_pattern():
    """Create an animation of a cymatics pattern evolving over time."""
    print("\n===== Cymatic Pattern Animation Demo =====")
    
    # Create a frequency modulator for animation
    modulator = FrequencyModulator(
        base_frequency=SACRED_FREQUENCIES['unity'],
        modulation_depth=0.3,
        phi_harmonic_count=3
    )
    
    # Set up modulation
    modulator.set_modulation_parameters(
        rate=0.2,  # Slow modulation for animation
        depth=0.3,
        waveform='sine',
        phi_weight=0.8
    )
    
    # Create a pattern generator
    generator = PatternGenerator()
    
    # Set up the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initial pattern
    initial_pattern = modulator.generate_material_pattern(
        material='water',
        size=(100, 100)
    )
    
    # Create image plot
    im = ax.imshow(initial_pattern, cmap='viridis', animated=True)
    title = ax.set_title("Cymatics Pattern Evolution")
    ax.axis('off')
    
    # Frame update function
    def update(frame):
        # Update modulator
        modulator.update(dt=0.05)
        
        # Get current frequency
        freq = modulator.get_current_frequency()
        
        # Generate pattern at this frequency
        if frame % 3 == 0:  # Change material every 3 frames for variety
            material = 'water'
        elif frame % 3 == 1:
            material = 'crystal'
        else:
            material = 'metal'
            
        pattern = modulator.generate_material_pattern(
            material=material,
            size=(100, 100)
        )
        
        # Update plot
        im.set_array(pattern)
        title.set_text(f"Cymatics Pattern: {freq:.1f} Hz - {material.capitalize()}")
        
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=30, interval=200, blit=True)
    
    # Save animation
    print("Saving cymatics pattern animation...")
    anim.save('cymatics_pattern_evolution.gif', fps=5, writer='pillow')
    
    plt.show()
    
    return anim

def main():
    print("Cymatics Pattern Materialization Demo")
    print("-" * 40)
    
    # Basic cymatic field demo
    field = demo_cymatic_field()
    
    # Frequency modulator demo
    modulator = demo_frequency_modulator()
    
    # Pattern generator demo
    generator = demo_pattern_generator()
    
    # Material resonator demo
    water_resonator, crystal_resonator, metal_resonator = demo_material_resonators()
    
    # Complete cymatics engine demo
    engine = demo_cymatics_engine()
    
    # Animated pattern demonstration
    anim = animate_cymatics_pattern()
    
    print("\n===== Cymatics Pattern Materialization Demo Complete =====")
    print("Component visualization images have been saved to the current directory.")

if __name__ == "__main__":
    main()