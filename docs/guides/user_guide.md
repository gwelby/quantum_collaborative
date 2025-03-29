# Quantum Field Visualization: User Guide

## Introduction

Welcome to the Quantum Field Visualization library! This guide will help you get started with generating, manipulating, and visualizing quantum fields using our package. Whether you're interested in sacred geometry, quantum physics visualization, or simply creating beautiful mathematical patterns, this library provides efficient tools to explore these concepts.

The library uses "sacred constants" like PHI (golden ratio), LAMBDA (divine complement), and PHI_PHI for generating mathematically harmonious fields that exhibit interesting patterns and coherence properties.

## Installation

To install the quantum field visualization package:

```bash
pip install quantum-field
```

For GPU acceleration, additional packages may be required:

```bash
# For NVIDIA GPU support
pip install quantum-field[cuda]

# For AMD GPU support
pip install quantum-field[rocm]

# For Intel GPU support (preliminary)
pip install quantum-field[oneapi]

# For all ML framework integrations
pip install quantum-field[ml]
```

## Quick Start

Here's a simple example to generate and visualize a quantum field:

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_field import generate_quantum_field

# Generate a quantum field with "love" frequency
field = generate_quantum_field(width=512, height=512, frequency_name="love", time_factor=0.0)

# Plot the field
plt.figure(figsize=(8, 8))
plt.imshow(field, cmap='viridis')
plt.colorbar(label='Field Intensity')
plt.title('Quantum Field Visualization (Love Frequency: 528Hz)')
plt.show()
```

## Core Concepts

### Sacred Constants

The library is built around sacred constants that appear throughout nature and mathematics:

```python
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# PHI = 1.618033988749895 (Golden ratio)
# LAMBDA = 0.618033988749895 (Divine complement)
# PHI_PHI = 2.1784575679375995 (Phi raised to power of phi)

# Available frequencies:
# SACRED_FREQUENCIES = {
#     'love': 528,      # Creation/healing
#     'unity': 432,     # Grounding/stability
#     'cascade': 594,   # Heart-centered integration
#     'truth': 672,     # Voice expression
#     'vision': 720,    # Expanded perception
#     'oneness': 768,   # Unity consciousness
# }
```

### Field Generation

Fields can be generated with different frequencies and time factors:

```python
from quantum_field import generate_quantum_field

# Generate fields with different frequencies
love_field = generate_quantum_field(512, 512, "love", 0.0)
unity_field = generate_quantum_field(512, 512, "unity", 0.0)
custom_field = generate_quantum_field(512, 512, custom_frequency=396, time_factor=0.5)

# Generate a series of fields with time evolution
fields = [generate_quantum_field(512, 512, "love", t) for t in np.linspace(0, 2*np.pi, 60)]
```

### Field Coherence

The library provides tools to analyze field coherence:

```python
from quantum_field import calculate_field_coherence

# Calculate coherence value (0.0 to 1.0)
coherence = calculate_field_coherence(field)
print(f"Field coherence: {coherence:.4f}")

# Track coherence over time
coherence_values = [calculate_field_coherence(generate_quantum_field(512, 512, "love", t)) 
                    for t in np.linspace(0, 2*np.pi, 100)]

plt.plot(np.linspace(0, 2*np.pi, 100), coherence_values)
plt.xlabel('Time Factor')
plt.ylabel('Coherence')
plt.title('Field Coherence Over Time')
plt.show()
```

### Sacred Geometry Patterns

Generate phi-based sacred geometry patterns:

```python
from quantum_field import generate_phi_pattern

# Generate a phi-based pattern
pattern = generate_phi_pattern(512, 512)

plt.figure(figsize=(8, 8))
plt.imshow(pattern, cmap='plasma')
plt.title('Phi-based Sacred Geometry Pattern')
plt.show()
```

## Hardware Acceleration

### Using Different Backends

The library automatically selects the best available backend, but you can manually select a specific one:

```python
from quantum_field.backends import get_backend, list_available_backends

# List all available backends with their priorities
backends = list_available_backends()
for backend in backends:
    print(f"{backend.name}: Priority {backend.priority}, Available: {backend.is_available()}")

# Get the default (highest priority available) backend
backend = get_backend()
print(f"Using backend: {backend.name}")

# Request a specific backend
cuda_backend = get_backend("cuda")
cpu_backend = get_backend("cpu")
```

### Backend Capabilities

Each backend reports its capabilities:

```python
# Check backend capabilities
backend = get_backend()
capabilities = backend.get_capabilities()

# Print capabilities
for capability, supported in capabilities.items():
    print(f"{capability}: {'✓' if supported else '✗'}")

# Check specific capabilities
if capabilities.get("multi_device", False):
    print("Multi-device support is available!")
if capabilities.get("half_precision", False):
    print("Half-precision (FP16) support is available!")
```

## Advanced Features

### Multi-GPU Processing

For large fields, the library can distribute work across multiple GPUs:

```python
from quantum_field.multi_gpu import generate_field_multi_gpu

# Generate a large field using multiple GPUs if available
large_field = generate_field_multi_gpu(4096, 4096, "love", 0.0)
```

### Thread Block Clusters

For newer NVIDIA GPUs (Compute Capability 9.0+), thread block clusters provide additional performance:

```python
from quantum_field.thread_block_cluster import generate_field_with_clusters

# Check if thread block clusters are supported
backend = get_backend("cuda")
if backend.capabilities.get("thread_block_clusters", False):
    field = generate_field_with_clusters(2048, 2048, "unity", 0.5)
```

### CUDA Graphs

For repetitive operations like animations, CUDA Graphs provide significant performance improvements:

```python
from quantum_field.backends.cuda import CUDAGraphsManager

# Create a CUDA graph for repeated field generation
graph_manager = CUDAGraphsManager()
graph_manager.create_field_generation_graph(512, 512, "love")

# Generate frames using the captured graph
frames = []
for t in np.linspace(0, 2*np.pi, 100):
    # Update the time factor without CPU overhead
    frame = graph_manager.run_graph_with_time_factor(t)
    frames.append(frame)

# Clean up
graph_manager.shutdown()
```

## Machine Learning Integration

### DLPack Integration

The library supports DLPack for zero-copy sharing of tensors with ML frameworks:

```python
import torch
from quantum_field.backends import get_backend

# Generate a quantum field
backend = get_backend()
field = backend.generate_quantum_field(512, 512, "love", 0.0)

# Convert to PyTorch tensor (zero-copy if on same device)
dlpack_tensor = backend.to_dlpack(field)
torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)

print(f"PyTorch tensor shape: {torch_tensor.shape}")
print(f"On CUDA: {torch_tensor.is_cuda}")
```

### PyTorch Integration Example

Process quantum fields with PyTorch models:

```python
import torch
import torch.nn as nn
from quantum_field.backends import get_backend

# Create a simple CNN model
class FieldPatternCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.upsample(x)
        return x.squeeze()

# Generate a field
backend = get_backend()
field = backend.generate_quantum_field(512, 512, "love", 0.0)

# Convert to PyTorch tensor
dlpack_tensor = backend.to_dlpack(field)
torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)

# Process with model
model = FieldPatternCNN()
with torch.no_grad():
    processed_field = model(torch_tensor)

# Convert back to NumPy if needed
processed_dlpack = torch.utils.dlpack.to_dlpack(processed_field)
processed_numpy = backend.from_dlpack(processed_dlpack)

# Visualize original and processed fields
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(field, cmap='viridis')
ax1.set_title('Original Field')
ax2.imshow(processed_numpy, cmap='viridis')
ax2.set_title('CNN Processed Field')
plt.show()
```

### TensorFlow Integration Example

```python
import tensorflow as tf
from quantum_field.backends import get_backend

# Generate a field
backend = get_backend()
field = backend.generate_quantum_field(512, 512, "vision", 0.5)

# Convert to PyTorch tensor first (as a bridge)
import torch
dlpack_tensor = backend.to_dlpack(field)
torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)

# Then to TensorFlow
tf_tensor = tf.experimental.dlpack.from_dlpack(
    torch.utils.dlpack.to_dlpack(torch_tensor))

print(f"TensorFlow tensor shape: {tf_tensor.shape}")
```

## Visualization Techniques

### Basic Static Visualization

```python
import matplotlib.pyplot as plt
from quantum_field import generate_quantum_field

field = generate_quantum_field(512, 512, "unity", 0.0)

# Basic plot
plt.figure(figsize=(10, 8))
plt.imshow(field, cmap='viridis')
plt.colorbar(label='Field Intensity')
plt.title('Quantum Field (Unity Frequency)')
plt.show()

# Custom colormaps for different frequencies
frequency_cmaps = {
    'love': 'plasma',
    'unity': 'viridis',
    'cascade': 'magma',
    'truth': 'cividis',
    'vision': 'inferno',
    'oneness': 'twilight'
}

# Create fields with different frequencies
frequencies = list(frequency_cmaps.keys())
fields = [generate_quantum_field(256, 256, freq, 0.0) for freq in frequencies]

# Plot all fields
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (field, freq) in enumerate(zip(fields, frequencies)):
    im = axes[i].imshow(field, cmap=frequency_cmaps[freq])
    axes[i].set_title(f"{freq.capitalize()} Frequency")
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()
```

### Animated Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quantum_field import generate_quantum_field

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))
plt.close()  # Don't display the empty initial figure

# Generate initial field
field = generate_quantum_field(256, 256, "love", 0.0)
img = ax.imshow(field, cmap='plasma', animated=True)
title = ax.set_title("Quantum Field Animation (t=0.0)")
fig.colorbar(img, ax=ax, label='Field Intensity')

# Animation update function
def update(frame):
    time_factor = frame / 50.0 * 2 * np.pi
    field = generate_quantum_field(256, 256, "love", time_factor)
    img.set_array(field)
    title.set_text(f"Quantum Field Animation (t={time_factor:.2f})")
    return [img, title]

# Create animation
anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Display the animation
plt.show()

# Or save it
# anim.save('quantum_field_animation.gif', writer='pillow', fps=20)
```

## Error Handling

The library includes custom exceptions for better error handling:

```python
from quantum_field.exceptions import QuantumError, FieldCoherenceError

try:
    # Try to generate a field with invalid parameters
    field = generate_quantum_field(-10, -10, "invalid_frequency", 0.0)
except QuantumError as e:
    print(f"Quantum error occurred: {e}")
```

## Performance Tips

1. **Choose the right backend**: For small fields, the CPU backend might be faster due to reduced overhead. For large fields, the GPU backends offer significant speedups.

2. **Reuse backends**: Getting a backend has minimal overhead, but initializing it can be expensive, especially for GPU backends. Reuse the same backend instance.

```python
backend = get_backend()
fields = [backend.generate_quantum_field(512, 512, "love", t) 
          for t in np.linspace(0, 1, 10)]
```

3. **Use CUDA Graphs**: For repetitive operations like animations or parameter sweeps, CUDA Graphs provide significant performance improvements.

4. **Batch processing**: When processing multiple fields, batch them for better performance.

5. **Thread Block Clusters**: For large fields on newer NVIDIA GPUs, thread block clusters provide additional performance.

## API Reference

For a complete API reference, see the [API Documentation](api_reference.md).

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related issues:

1. Check that your NVIDIA drivers are up to date
2. Verify that CuPy is installed correctly: `pip install cupy-cuda12x` (adjust version for your CUDA installation)
3. Try falling back to the CPU backend: `backend = get_backend("cpu")`

### ROCm Issues

For AMD GPU issues:

1. Ensure you have ROCm installed and configured correctly
2. PyTorch with ROCm support is required for this backend
3. Check for compatible versions: `torch.__version__` and `torch.version.hip`

### General Performance Issues

- Try reducing field size for better performance
- Enable capability reporting to understand limitations: `print(backend.get_capabilities())`
- Check if your hardware is supported at the expected performance level

## Next Steps

Now that you're familiar with the basics, explore these advanced topics:

1. **Custom Frequency Patterns**: Create your own frequency patterns beyond the predefined sacred frequencies
2. **Advanced Visualization**: Combine multiple fields or apply post-processing for more complex visualizations
3. **Machine Learning**: Use quantum fields as inputs to neural networks for pattern recognition or field evolution prediction
4. **Coherence Analysis**: Study how field coherence changes with different parameters and configurations

## Getting Help

If you encounter issues or have questions:

- Report bugs on GitHub: [Issues Page](https://github.com/example/quantum-field/issues)
- Join the community discussion: [Discussions](https://github.com/example/quantum-field/discussions)
- Check the FAQ: [Frequently Asked Questions](faq.md)