# Quantum Field Visualization Demo Script

## Overview

This script outlines the key features to showcase in the Quantum Field Visualization demo video. The demo will highlight the core capabilities of the library, focusing on its unique features, performance, and ease of use.

## Setup

* Make sure OBS Studio is installed and configured
* Install all dependencies: `pip install "quantum-field[all]"`
* Have a terminal window ready with a clean environment
* Make sure visualization windows can be captured properly

## Demo Flow

### 1. Introduction (30 seconds)

* Introduce the Quantum Field Visualization library
* Explain that it's a Python package for generating and visualizing quantum fields based on phi-harmonic principles
* Mention that it supports multiple hardware accelerators

### 2. Basic Quantum Field Generation (1 minute)

```python
# Import the library
from quantum_field import generate_quantum_field, calculate_field_coherence
import matplotlib.pyplot as plt
import numpy as np

# Generate a basic quantum field with the "love" frequency (528 Hz)
field = generate_quantum_field(width=512, height=512, frequency_name='love')

# Calculate field coherence
coherence = calculate_field_coherence(field)
print(f"Field coherence: {coherence:.4f}")

# Visualize the field
plt.figure(figsize=(8, 8))
plt.imshow(field, cmap='viridis')
plt.colorbar(label='Field Intensity')
plt.title(f'Quantum Field - Love Frequency (528 Hz) - Coherence: {coherence:.4f}')
plt.show()
```

### 3. Sacred Frequencies Demo (1 minute)

```python
# Import the constants
from quantum_field.constants import SACRED_FREQUENCIES

# Show all available frequencies
print("Available sacred frequencies:")
for name, freq in SACRED_FREQUENCIES.items():
    print(f"- {name}: {freq} Hz")

# Generate fields with different frequencies
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, freq) in enumerate(SACRED_FREQUENCIES.items()):
    field = generate_quantum_field(width=256, height=256, frequency_name=name)
    coherence = calculate_field_coherence(field)
    
    axes[i].imshow(field, cmap='viridis')
    axes[i].set_title(f'{name.capitalize()} ({freq} Hz)\nCoherence: {coherence:.4f}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### 4. Hardware Acceleration Demo (1 minute)

```python
# Import backend capabilities
from quantum_field.backends import get_backend, get_available_backends
import time

# Show available backends
backends = get_available_backends()
print("Available backends:")
for name, backend in backends.items():
    print(f"- {name.upper()}")

# Benchmark field generation with different backends
sizes = [512, 1024, 2048]

print("\nPerformance comparison:")
print("----------------------")

# Use CPU as baseline
cpu_backend = get_backend("cpu")
for size in sizes:
    # CPU timing
    start = time.time()
    cpu_backend.generate_quantum_field(width=size, height=size, frequency_name='love')
    cpu_time = time.time() - start
    print(f"Size: {size}x{size}, CPU: {cpu_time:.4f} seconds")
    
    # Try other hardware accelerators
    for name, backend in backends.items():
        if name == "cpu":
            continue
        
        try:
            start = time.time()
            backend.generate_quantum_field(width=size, height=size, frequency_name='love')
            backend_time = time.time() - start
            speedup = cpu_time / backend_time
            print(f"Size: {size}x{size}, {name.upper()}: {backend_time:.4f} seconds ({speedup:.2f}x faster)")
        except Exception as e:
            print(f"Size: {size}x{size}, {name.upper()}: Failed - {str(e)}")
```

### 5. 3D Quantum Field Visualization (1 minute)

```python
# Import 3D visualization
from quantum_field.visualization3d import (
    generate_3d_quantum_field,
    visualize_3d_slices,
    visualize_3d_isosurface
)

# Generate a 3D quantum field
size = 64
field_3d = generate_3d_quantum_field(width=size, height=size, depth=size, frequency_name='cascade')
print(f"Generated 3D field with shape: {field_3d.shape}")

# Show slices
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(field_3d[size//2, :, :], cmap='viridis')
plt.title('XY Slice (Z middle)')

plt.subplot(132)
plt.imshow(field_3d[:, size//2, :], cmap='viridis')
plt.title('XZ Slice (Y middle)')

plt.subplot(133)
plt.imshow(field_3d[:, :, size//2], cmap='viridis')
plt.title('YZ Slice (X middle)')

plt.tight_layout()
plt.show()

# Generate isosurface (if plotly is available)
try:
    fig = visualize_3d_isosurface(field_3d)
    fig.show()
except ImportError:
    print("Isosurface visualization requires plotly and pyvista")
```

### 6. Animation Demo (1 minute)

```python
# Import animation functionality
from quantum_field import animate_field

# Generate animation
animate_field(width=256, height=256, frames=30, 
             frequency_name='unity', output_file='quantum_field_animation.gif')

print("Animation saved to 'quantum_field_animation.gif'")
```

### 7. Command Line Interface Demo (30 seconds)

```bash
# Show help
python -m quantum_field --help

# Generate a field
python -m quantum_field generate --width 256 --height 256 --frequency love --output love_field.png

# Show information about sacred constants
python -m quantum_field info

# Run a benchmark
python -m quantum_field benchmark --width 512 --height 512 --iterations 3
```

### 8. Real-World Applications (1 minute)

* Briefly mention potential applications:
  * Scientific visualization
  * Consciousness research
  * Meditation aids
  * Generative art
  * Educational tools

### 9. Conclusion (30 seconds)

* Recap key features
* Mention that the library is open-source
* Invite viewers to try it out and contribute
* Show GitHub link and documentation site

## Recording Notes

* Maintain a clear, steady pace
* Add voice narration explaining each step
* Ensure all visualizations are clearly visible
* For any slow operations, consider speeding up in post-processing
* Use OBS Studio's scene transitions for a professional look
* Consider adding background music at a low volume

## Post-Production

* Add title screen and outro
* Add captions for code sections
* Add lower-third for important information
* Export in 1080p or 4K resolution
* Upload to YouTube and embed on documentation site