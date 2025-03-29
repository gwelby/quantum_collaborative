# Quantum Field Visualization

A Python package for generating and visualizing quantum fields based on phi-harmonic principles, with hardware acceleration support for multiple processor types.

## Overview

The Quantum Field Visualization package enables the generation and analysis of quantum fields using sacred constants and frequencies. It provides a universal accelerator architecture that automatically selects the optimal hardware implementation from multiple supported backends.

### Key Features

- Generate quantum fields using phi-harmonic principles
- Visualize fields with ASCII art or matplotlib
- Universal accelerator architecture with multiple backends:
  - NVIDIA GPUs (CUDA)
  - AMD GPUs (ROCm/HIP) 
  - Intel GPUs (oneAPI)
  - Mobile GPUs (Metal/Vulkan)
  - Huawei Ascend NPUs
  - Tenstorrent Grayskull/Wormhole Processors
  - CPU fallback for all platforms
- Support for advanced CUDA features:
  - Thread Block Clusters (for H100+ GPUs)
  - Multi-GPU support for large fields
- Comprehensive benchmarking tools
- Compatibility checking utility

## Installation

### Basic Installation

```bash
pip install quantum-field
```

### With CUDA Support

For NVIDIA GPU acceleration:

```bash
pip install "quantum-field[cuda]"
```

### With ROCm Support

For AMD GPU acceleration:

```bash
pip install "quantum-field[rocm]"
```

### With Tenstorrent Support

For Tenstorrent hardware acceleration:

```bash
pip install "quantum-field[tenstorrent]"
```

### With All Accelerators

For maximum hardware support:

```bash
pip install "quantum-field[all]"
```

## Quick Start

### Generate a Quantum Field

```python
import numpy as np
from quantum_field import generate_quantum_field, field_to_ascii, print_field

# Generate a quantum field with the 'love' frequency (528 Hz)
field = generate_quantum_field(width=80, height=20, frequency_name='love')

# Convert to ASCII and print
ascii_art = field_to_ascii(field)
print_field(ascii_art, "Quantum Field - Love Frequency")
```

### Animate a Quantum Field

```python
from quantum_field import animate_field

# Create an animation with 20 frames
animate_field(width=80, height=20, frames=20, frequency_name='unity')
```

### Explicitly Select a Backend

```python
from quantum_field.backends import get_backend

# Get a specific backend
tenstorrent_backend = get_backend("tenstorrent")

# Generate a field using this backend
field = tenstorrent_backend.generate_quantum_field(
    width=256, height=256, 
    frequency_name='cascade', 
    time_factor=0.5
)

# Calculate field coherence
coherence = tenstorrent_backend.calculate_field_coherence(field)
print(f"Field coherence: {coherence:.4f}")
```

### Benchmark Performance

```python
from quantum_field import benchmark_all_backends

# Compare performance across all available backends
results = benchmark_all_backends(width=512, height=512, iterations=5)
print(results)
```

## Command Line Interface

The package can be run directly from the command line:

```bash
# Check compatibility and list available backends
python -m quantum_field check

# Generate a field
python -m quantum_field generate --width 80 --height 20 --frequency love

# Animate a field
python -m quantum_field animate --width 80 --height 20 --frames 30 --frequency cascade

# Display a phi pattern
python -m quantum_field pattern

# Run benchmarks on all available backends
python -m quantum_field benchmark --width 512 --height 512 --iterations 5

# Run with a specific backend
python -m quantum_field generate --backend rocm --width 512 --height 512

# Display sacred constants information
python -m quantum_field info
```

## Sacred Constants

The package uses the following sacred constants:

- **PHI**: 1.618033988749895 (Golden Ratio)
- **LAMBDA**: 0.618033988749895 (Divine Complement - 1/PHI)
- **PHI_PHI**: 2.1784575679375995 (Hyperdimensional Constant)

### Sacred Frequencies

- **love**: 528 Hz (Creation/healing)
- **unity**: 432 Hz (Grounding/stability)
- **cascade**: 594 Hz (Heart-centered integration)
- **truth**: 672 Hz (Voice expression)
- **vision**: 720 Hz (Expanded perception)
- **oneness**: 768 Hz (Unity consciousness)

## Multi-Accelerator Architecture

The package includes a comprehensive backend architecture that supports multiple accelerator types:

### CPU Backend (Priority: 0)
- Always available as a fallback
- Multithreaded implementation for multi-core CPUs
- Optimized NumPy-based calculations

### CUDA Backend (Priority: 70)
- For NVIDIA GPUs
- Thread Block Cluster support for H100+ GPUs
- Multi-GPU support for large fields
- Advanced memory access patterns

### ROCm Backend (Priority: 80)
- For AMD GPUs using HIP/ROCm
- Implemented using PyTorch with HIP support
- Optimized tensor operations

### Tenstorrent Backend (Priority: 85)
- For Tenstorrent Grayskull/Wormhole Processors
- Fibonacci-based block processing optimizations
- Integrated with PyBuda and QuantumTensix
- Phi-harmonic optimizations tailored to Tensix architecture

### Mobile Backend (Priority: 60)
- For iOS devices (using Metal)
- For Android devices (using Vulkan)
- Power-efficient implementation

### Huawei Ascend Backend (Priority: 75)
- For Huawei NPUs (Ascend 910/310)
- MindSpore and ACL integration

### oneAPI Backend (Priority: 75)
- Placeholder for Intel GPUs and accelerators (in development)

## Performance

The hardware-accelerated implementations provide substantial performance improvements:

- NVIDIA GPUs: Up to 100x speedup on large fields
- AMD GPUs: Up to 80x speedup with ROCm backend
- Tenstorrent: Up to 25-35% additional speedup with phi-harmonic optimizations
- Mobile GPUs: 5-10x speedup on modern devices
- Multiple GPUs: Near-linear scaling with GPU count for large fields

These numbers will vary depending on the hardware model and system configuration.

## Testing

The package includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_quantum_cuda.py
pytest tests/test_multi_gpu.py
pytest tests/test_thread_block_cluster.py
pytest tests/test_performance.py

# Run tests with coverage
pytest --cov=quantum_field
```

### Test Categories:

- **Functional Tests**: Verify correctness of field generation and analysis
- **Integration Tests**: Test multi-GPU and thread block cluster integration
- **Performance Tests**: Check for performance regressions and confirm acceleration
- **Compatibility Tests**: Verify graceful fallback on unsupported hardware

Performance tests automatically save and compare against baselines to detect regressions.

## Documentation

For more detailed information, see the following documents:

- [CUDA Integration](./CUDA_README.md) - Technical details on CUDA implementation
- [Quantum Mathematics](./QUANTUM_README.md) - Mathematical foundations
- [Implementation Summary](./SUMMARY.md) - Overview of implementation details
- [Multi-Accelerator Architecture](./SUMMARY_MULTI_BACKEND.md) - Details on the universal accelerator architecture

## License

MIT