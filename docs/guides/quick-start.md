# Quick Start Guide

This guide will help you get started with the Quantum Field Visualization library quickly. You'll learn how to generate basic quantum fields, visualize them, and explore different hardware acceleration options.

## Basic Field Generation

Start by importing the necessary modules and generating your first quantum field:

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_field import generate_quantum_field, calculate_field_coherence

# Generate a quantum field with the "love" frequency (528 Hz)
field = generate_quantum_field(width=256, height=256, frequency_name='love')

# Calculate the coherence of the field
coherence = calculate_field_coherence(field)
print(f"Field coherence: {coherence:.4f}")

# Visualize the field
plt.figure(figsize=(8, 8))
plt.imshow(field, cmap='viridis')
plt.colorbar(label='Field Intensity')
plt.title(f'Quantum Field - Love Frequency (528 Hz)\nCoherence: {coherence:.4f}')
plt.show()
```

## Exploring Different Frequencies

The library provides several sacred frequencies to work with:

```python
from quantum_field.constants import SACRED_FREQUENCIES

# Print all available frequencies
print("Available sacred frequencies:")
for name, freq in SACRED_FREQUENCIES.items():
    print(f"- {name}: {freq} Hz")

# Compare fields with different frequencies
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

## Time-Varying Fields

You can generate time-varying fields by adjusting the time factor:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quantum_field import generate_quantum_field

# Create figure for animation
fig, ax = plt.subplots(figsize=(8, 8))
plt.close()  # Don't display immediately

# Initialize with empty data
img = ax.imshow(np.zeros((256, 256)), cmap='viridis')
ax.set_title('Time-Varying Quantum Field (Unity Frequency: 432 Hz)')
fig.colorbar(img, label='Field Intensity')

# Update function for animation
def update(frame):
    # Generate field with time factor (normalized to 0-2π)
    time_factor = frame / 30 * 2 * np.pi
    field = generate_quantum_field(
        width=256, height=256, 
        frequency_name='unity',
        time_factor=time_factor
    )
    img.set_array(field)
    return [img]

# Create animation
anim = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
plt.show()
```

## Using Different Hardware Backends

The library automatically selects the best available backend, but you can also explicitly choose one:

```python
from quantum_field.backends import get_backend, get_available_backends
import time

# List available backends
backends = get_available_backends()
print("Available backends:")
for name, backend in backends.items():
    print(f"- {name.upper()}")

# Compare performance across backends
size = 1024  # Larger fields show more performance difference

# CPU baseline
cpu_backend = get_backend("cpu")
start = time.time()
cpu_field = cpu_backend.generate_quantum_field(width=size, height=size, frequency_name='love')
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.4f} seconds")

# Try other backends
for name, backend in backends.items():
    if name == "cpu":
        continue
        
    try:
        start = time.time()
        field = backend.generate_quantum_field(width=size, height=size, frequency_name='love')
        backend_time = time.time() - start
        speedup = cpu_time / backend_time
        print(f"{name.upper()}: {backend_time:.4f} seconds ({speedup:.2f}x faster than CPU)")
    except Exception as e:
        print(f"{name.upper()}: Failed - {str(e)}")
```

## 3D Field Visualization

The library also supports 3D quantum fields:

```python
from quantum_field.visualization3d import (
    generate_3d_quantum_field,
    visualize_3d_slices
)

# Generate a 3D quantum field
field_3d = generate_3d_quantum_field(width=64, height=64, depth=64, frequency_name='cascade')
print(f"3D field shape: {field_3d.shape}")

# Visualize slices of the 3D field
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(field_3d[32, :, :], cmap='viridis')
plt.title('XY Slice (middle of Z axis)')
plt.colorbar()

plt.subplot(132)
plt.imshow(field_3d[:, 32, :], cmap='viridis')
plt.title('XZ Slice (middle of Y axis)')
plt.colorbar()

plt.subplot(133)
plt.imshow(field_3d[:, :, 32], cmap='viridis')
plt.title('YZ Slice (middle of X axis)')
plt.colorbar()

plt.tight_layout()
plt.show()

# For more advanced 3D visualization, use the visualization3d module
fig = visualize_3d_slices(field_3d)
plt.show()
```

## Command Line Interface

The library also provides a command-line interface:

```bash
# Show help
python -m quantum_field --help

# Generate a field and save to file
python -m quantum_field generate --width 512 --height 512 --frequency love --output quantum_field.png

# Compare performance across backends
python -m quantum_field benchmark --width 512 --height 512 --iterations 3

# Display sacred constants information
python -m quantum_field info

# Create an animation
python -m quantum_field animate --width 256 --height 256 --frames 30 --frequency cascade
```

## Next Steps

Now that you've seen the basics, you can:

1. Explore the [User Guide](user_guide.md) for more detailed information
2. Check the [API Reference](api_reference.md) for complete documentation
3. Try the [Examples](examples/) for more complex usage scenarios
4. Experiment with different hardware backends