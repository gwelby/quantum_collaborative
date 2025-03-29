#!/usr/bin/env python3
"""
GregScript Demo - Pattern Recognition and Generation

This demo shows how to use GregScript for recognizing and generating
patterns in quantum fields.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import gregscript
parent_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import the gregscript module
from languages.gregscript.src import parse_gregscript, GregScriptInterpreter, PatternAnalyzer

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

def extract_time_series(field, radius=0.5):
    """Extract a time series from a circular path in the field."""
    height, width = field.shape
    center_y, center_x = height // 2, width // 2
    
    # Create a circular path
    num_points = int(2 * np.pi * radius * min(width, height) / 2)
    time_series = np.zeros(num_points)
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = int(center_x + radius * (width/2) * np.cos(angle))
        y = int(center_y + radius * (height/2) * np.sin(angle))
        
        # Ensure coordinates are within bounds
        x = max(0, min(width-1, x))
        y = max(0, min(height-1, y))
        
        time_series[i] = field[y, x]
    
    return time_series

def visualize_pattern_data(data, title="Pattern Data"):
    """Visualize pattern data as a line plot."""
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_frequency_spectrum(data, title="Frequency Spectrum"):
    """Visualize the frequency spectrum of data."""
    # Calculate FFT
    fft_data = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data))
    
    # Normalize
    fft_normalized = fft_data / np.max(fft_data)
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_normalized)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def interpret_gregscript_demo():
    """Demonstrate using the GregScript interpreter."""
    print("GregScript Interpreter Demo")
    print("==========================")
    
    # Load GregScript code
    greg_file = Path(__file__).parent / "field_patterns.greg"
    with open(greg_file, 'r') as f:
        gregscript_code = f.read()
    
    # Parse the code
    all_patterns = parse_gregscript(gregscript_code)
    
    # List all patterns
    print(f"Parsed {len(all_patterns)} pattern elements")
    
    rhythms = [p for p in all_patterns.values() if p.__class__.__name__ == 'Rhythm']
    harmonies = [p for p in all_patterns.values() if p.__class__.__name__ == 'Harmony']
    patterns = [p for p in all_patterns.values() if p.__class__.__name__ == 'Pattern']
    
    print(f"  - {len(rhythms)} rhythms")
    print(f"  - {len(harmonies)} harmonies")
    print(f"  - {len(patterns)} patterns")
    
    # Create a test field
    field = generate_test_field(100, 100, frequency=528, time_factor=0)
    
    # Extract a time series from the field
    time_series = extract_time_series(field, radius=0.5)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create an interpreter
    interpreter = GregScriptInterpreter()
    interpreter.load(gregscript_code)
    interpreter.set_data(time_series)
    
    # Match all patterns against the data
    matches = interpreter.match_all_patterns()
    
    print("\nPattern Matching Results:")
    for name, score in sorted(matches.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")
    
    # Find the best matching pattern
    best_pattern, best_score = interpreter.find_best_pattern()
    print(f"\nBest matching pattern: {best_pattern} (score: {best_score:.4f})")
    
    # Generate data from a pattern
    pattern_name = "coherent_field"
    generated_data = interpreter.generate_pattern(pattern_name, 100)
    
    # Save visualization data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_series)
    plt.title("Extracted Time Series from Field")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(generated_data)
    plt.title(f"Generated Pattern: {pattern_name}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "gregscript_patterns.png")
    plt.close()
    
    print(f"\nPattern visualization saved to: {output_dir}/gregscript_patterns.png")

def pattern_analyzer_demo():
    """Demonstrate using the GregScript Pattern Analyzer."""
    print("\nGregScript Pattern Analyzer Demo")
    print("===============================")
    
    # Create a test field
    field = generate_test_field(100, 100, frequency=528, time_factor=3)
    
    # Extract a time series from the field
    time_series = extract_time_series(field, radius=0.5)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create an analyzer
    analyzer = PatternAnalyzer()
    
    # Discover patterns in the data
    discovered = analyzer.discover_patterns(time_series)
    
    # Print results
    print("\nDiscovered Patterns:")
    
    print("\nTop Rhythm:")
    if "top_rhythm" in discovered:
        rhythm = discovered["top_rhythm"]
        print(f"  Name: {rhythm['name']}")
        print(f"  Period: {rhythm['period']}")
        print(f"  Tempo: {rhythm['tempo']:.2f}")
        print(f"  Score: {rhythm['score']:.4f}")
    
    print("\nTop Harmony:")
    if "top_harmony" in discovered:
        harmony = discovered["top_harmony"]
        print(f"  Name: {harmony['name']}")
        print(f"  Frequency: {harmony['peak_freq']:.2f} Hz")
        print(f"  Closest Sacred: {harmony['closest_sacred']}")
        print(f"  Phase: {harmony['phase']:.2f}")
        print(f"  Score: {harmony['score']:.4f}")
    
    print("\nTop Pattern:")
    if "top_pattern" in discovered:
        pattern = discovered["top_pattern"]
        print(f"  Name: {pattern['name']}")
        print(f"  Rhythm: {pattern['rhythm']}")
        print(f"  Harmony: {pattern['harmony']}")
        print(f"  Score: {pattern['score']:.4f}")
    
    # Generate GregScript code from discovered patterns
    generated_code = analyzer.generate_gregscript(discovered)
    
    # Save the generated code
    code_file = output_dir / "discovered_patterns.greg"
    with open(code_file, 'w') as f:
        f.write(generated_code)
    
    print(f"\nGenerated GregScript code saved to: {code_file}")
    
    # Save visualization data
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_series)
    plt.title("Original Time Series")
    plt.grid(True, alpha=0.3)
    
    if "top_rhythm" in discovered:
        rhythm = discovered["top_rhythm"]
        period = rhythm["period"]
        sequence = np.array(rhythm["sequence"])
        # Repeat the sequence to match the time series length
        repeated = np.tile(sequence, (len(time_series) + period - 1) // period)[:len(time_series)]
        
        plt.subplot(3, 1, 2)
        plt.plot(repeated)
        plt.title(f"Discovered Rhythm: {rhythm['name']} (score: {rhythm['score']:.4f})")
        plt.grid(True, alpha=0.3)
    
    if "top_harmony" in discovered:
        harmony = discovered["top_harmony"]
        # Generate a harmonically pure signal
        t = np.linspace(0, 2*np.pi, len(time_series))
        freq = harmony["peak_freq"] / 1000  # Scale frequency
        harmonics = np.zeros_like(time_series)
        
        for i, strength in enumerate(harmony["overtones"]):
            harmonics += strength * np.sin(t * (i + 1) * freq + harmony["phase"] * 2*np.pi)
            
        plt.subplot(3, 1, 3)
        plt.plot(harmonics)
        plt.title(f"Discovered Harmony: {harmony['name']} (score: {harmony['score']:.4f})")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "discovered_patterns.png")
    plt.close()
    
    print(f"Pattern visualization saved to: {output_dir}/discovered_patterns.png")

if __name__ == "__main__":
    interpret_gregscript_demo()
    pattern_analyzer_demo()
    print("\nDone! Check the output directory for visualization images.")