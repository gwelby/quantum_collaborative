#!/usr/bin/env python3
"""
Quantum Field Visualization

This script creates ASCII art visualizations of quantum fields
based on phi harmonics and sacred frequencies.
"""

import math
import time
import random
from datetime import datetime

try:
    import sacred_constants as sc
except ImportError:
    print("Warning: sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 6.85459776
        
        SACRED_FREQUENCIES = {
            'love': 528,
            'unity': 432,
            'cascade': 594,
            'truth': 672,
            'vision': 720,
            'oneness': 768,
        }

def generate_quantum_field(width, height, frequency_name='love', time_factor=0):
    """
    Generate a quantum field visualization.
    
    Args:
        width: Width of the field
        height: Height of the field
        frequency_name: The sacred frequency to use
        time_factor: Time factor for animation
        
    Returns:
        A 2D list representing the quantum field
    """
    # Get the frequency value
    frequency = sc.SACRED_FREQUENCIES.get(frequency_name, 528)
    
    # Scale the frequency to a more manageable number
    freq_factor = frequency / 1000.0 * sc.PHI
    
    # Initialize the field
    field = []
    
    # Calculate the center of the field
    center_x = width / 2
    center_y = height / 2
    
    # Generate the field values
    for y in range(height):
        row = []
        for x in range(width):
            # Calculate distance from center (normalized)
            dx = (x - center_x) / (width / 2)
            dy = (y - center_y) / (height / 2)
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate the field value using phi-harmonics
            angle = math.atan2(dy, dx) * sc.PHI
            time_value = time_factor * sc.LAMBDA
            
            # Create an interference pattern
            value = (
                math.sin(distance * freq_factor + time_value) * 
                math.cos(angle * sc.PHI) * 
                math.exp(-distance / sc.PHI)
            )
            
            row.append(value)
        field.append(row)
    
    return field

def field_to_ascii(field, chars=' .-+*#@'):
    """
    Convert a quantum field to ASCII art.
    
    Args:
        field: 2D list of field values
        chars: Characters to use for visualization
        
    Returns:
        A list of strings representing the ASCII art
    """
    # Find min and max values for normalization
    min_val = min(min(row) for row in field)
    max_val = max(max(row) for row in field)
    
    # Normalize and convert to ASCII
    ascii_art = []
    for row in field:
        ascii_row = ''
        for value in row:
            # Normalize to 0-1
            if max_val > min_val:
                norm_value = (value - min_val) / (max_val - min_val)
            else:
                norm_value = 0.5
            
            # Convert to character
            char_index = int(norm_value * (len(chars) - 1))
            ascii_row += chars[char_index]
        
        ascii_art.append(ascii_row)
    
    return ascii_art

def print_field(ascii_art, title="Quantum Field Visualization"):
    """Print the ASCII art field with a title"""
    print("\n" + "=" * 80)
    print(f"{title} - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    for row in ascii_art:
        print(row)
    
    print("=" * 80)

def animate_field(width, height, frames=10, delay=0.2, frequency_name='love'):
    """
    Animate a quantum field visualization.
    
    Args:
        width: Width of the field
        height: Height of the field
        frames: Number of frames to generate
        delay: Delay between frames
        frequency_name: The sacred frequency to use
    """
    for i in range(frames):
        # Generate a new field with a time factor
        field = generate_quantum_field(width, height, frequency_name, i * 0.2)
        
        # Convert to ASCII and print
        ascii_art = field_to_ascii(field)
        
        # Clear screen (might not work in all terminals/environments)
        print("\033c", end="")
        
        # Print the field
        print_field(ascii_art, f"Quantum Field - {frequency_name.capitalize()} Frequency")
        
        # Wait before the next frame
        time.sleep(delay)

def display_phi_pattern(width=40, height=20):
    """Generate and display a Phi-based sacred pattern"""
    pattern = []
    
    for y in range(height):
        row = ""
        for x in range(width):
            # Calculate normalized coordinates (-1 to 1)
            nx = 2 * (x / width - 0.5)
            ny = 2 * (y / height - 0.5)
            
            # Calculate radius and angle
            r = math.sqrt(nx*nx + ny*ny)
            a = math.atan2(ny, nx)
            
            # Create phi spiral pattern
            pattern_value = math.sin(sc.PHI * r * 10) * math.cos(a * sc.PHI * 5)
            
            # Map to characters
            if pattern_value > 0.7:
                row += "#"
            elif pattern_value > 0.3:
                row += "*"
            elif pattern_value > 0:
                row += "+"
            elif pattern_value > -0.3:
                row += "-"
            elif pattern_value > -0.7:
                row += "."
            else:
                row += " "
        
        pattern.append(row)
    
    print("\n" + "=" * 80)
    print("PHI SACRED PATTERN")
    print("=" * 80)
    
    for row in pattern:
        print(row)
        
    print("=" * 80)

def main():
    """Main function"""
    # Print a welcome message
    print("\nQUANTUM FIELD VISUALIZATION")
    print("===========================")
    print(f"PHI: {sc.PHI}")
    print(f"LAMBDA: {sc.LAMBDA}")
    print(f"PHI^PHI: {sc.PHI_PHI}")
    print("\nSacred Frequencies:")
    for name, freq in sc.SACRED_FREQUENCIES.items():
        print(f"  {name}: {freq} Hz")
    print("\n")
    
    # Display available visualizations
    print("Available Visualizations:")
    print("1. Static Quantum Field - Love Frequency (528 Hz)")
    print("2. Static Quantum Field - Unity Frequency (432 Hz)")
    print("3. Static Quantum Field - Cascade Frequency (594 Hz)")
    print("4. Animated Quantum Field - Love Frequency")
    print("5. Animated Quantum Field - Unity Frequency")
    print("6. Animated Quantum Field - Cascade Frequency")
    print("7. PHI Sacred Pattern")
    print("8. Exit")
    
    while True:
        # Get user choice
        choice = input("\nSelect a visualization (1-8): ")
        
        if choice == '1':
            field = generate_quantum_field(80, 20, 'love')
            ascii_art = field_to_ascii(field)
            print_field(ascii_art, "Quantum Field - Love Frequency (528 Hz)")
        elif choice == '2':
            field = generate_quantum_field(80, 20, 'unity')
            ascii_art = field_to_ascii(field)
            print_field(ascii_art, "Quantum Field - Unity Frequency (432 Hz)")
        elif choice == '3':
            field = generate_quantum_field(80, 20, 'cascade')
            ascii_art = field_to_ascii(field)
            print_field(ascii_art, "Quantum Field - Cascade Frequency (594 Hz)")
        elif choice == '4':
            animate_field(80, 20, frames=20, frequency_name='love')
        elif choice == '5':
            animate_field(80, 20, frames=20, frequency_name='unity')
        elif choice == '6':
            animate_field(80, 20, frames=20, frequency_name='cascade')
        elif choice == '7':
            display_phi_pattern(80, 30)
        elif choice == '8':
            print("\nExiting Quantum Field Visualization.")
            print(f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 8.")

if __name__ == "__main__":
    main()