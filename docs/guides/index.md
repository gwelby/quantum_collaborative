# Quantum Field Visualization Documentation

Welcome to the documentation for the Quantum Field Visualization library! This documentation will help you understand, install, and use the library for quantum field visualization and manipulation.

## Navigation

- [User Guide](user_guide.md) - Comprehensive guide to using the library
- [API Reference](api_reference.md) - Complete API documentation
- [Contributing](contributing.md) - Guidelines for contributing to the project

## Getting Started

The Quantum Field Visualization library provides tools for generating, analyzing, and visualizing quantum fields using sacred geometrical principles and constants.

### Installation

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

### Quick Example

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

The library is built around several core concepts:

- **Sacred Constants**: Phi (1.618...), Lambda (0.618...), and other mathematically significant values
- **Quantum Fields**: Mathematically generated fields based on wave equations and sacred geometry
- **Field Coherence**: Measure of harmony and resonance within a field
- **Hardware Acceleration**: Support for various accelerator hardware (GPUs, NPUs, etc.)

## Features

- Generate quantum fields with various sacred frequencies
- Calculate field coherence and harmony metrics
- Create phi-based sacred geometry patterns
- Hardware acceleration on multiple platforms:
  - NVIDIA GPUs (CUDA)
  - AMD GPUs (ROCm)
  - Intel GPUs (oneAPI)
  - Mobile GPUs (iOS/Android)
  - Huawei Ascend NPUs
  - Tenstorrent processors
- Advanced CUDA features:
  - Multi-GPU support
  - Thread Block Clusters
  - CUDA Graphs
- Machine learning integration:
  - DLPack tensor exchange
  - PyTorch, TensorFlow, and JAX compatibility

## Project Status

The project is actively maintained and developed. See the [MYWISH.md](../MYWISH.md) file in the repository for the current development status and roadmap.

## License

[Insert appropriate license information here]

## Contact

[Insert contact information here]