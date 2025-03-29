# Installation Guide

This guide covers how to install the Quantum Field Visualization library on different platforms and with various hardware acceleration options.

## Prerequisites

Before installing, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) NVIDIA CUDA toolkit 11.0+ for CUDA acceleration
- (Optional) ROCm platform 4.0+ for AMD GPU acceleration
- (Optional) oneAPI Base Toolkit for Intel GPU acceleration

## Basic Installation

The simplest way to install the library is via pip:

```bash
pip install quantum-field
```

This installs the core library with NumPy and Matplotlib dependencies.

## Hardware Acceleration

### NVIDIA GPU Acceleration

To enable NVIDIA GPU acceleration with CUDA:

```bash
pip install "quantum-field[cuda]"
```

This will install the necessary CUDA dependencies, including CuPy.

#### CUDA Requirements

- NVIDIA GPU (Compute Capability 3.5+, Pascal or newer recommended)
- CUDA Toolkit 11.0 or higher
- NVIDIA drivers 450.80.02 or higher

### AMD GPU Acceleration

For AMD GPU acceleration with ROCm:

```bash
pip install "quantum-field[rocm]"
```

#### ROCm Requirements

- ROCm-supported AMD GPU
- ROCm platform 4.0 or higher
- AMD drivers compatible with your ROCm version

### Intel GPU Acceleration

For Intel GPU acceleration with oneAPI:

```bash
pip install "quantum-field[oneapi]"
```

#### oneAPI Requirements

- Intel GPU with Xe architecture
- oneAPI Base Toolkit
- Intel GPU drivers

### All Accelerators

To install all acceleration options (for development or testing):

```bash
pip install "quantum-field[all]"
```

## Development Installation

For development purposes, install from the source repository:

```bash
# Clone the repository
git clone https://github.com/yourname/quantum-field.git
cd quantum-field

# Install in development mode with development extras
pip install -e ".[dev]"
```

The `dev` extra includes:
- pytest and pytest-cov for testing
- black and flake8 for code formatting and linting
- mypy for static type checking
- build and twine for package building and publishing

## Verifying Installation

After installation, verify everything is working correctly:

```python
# Check basic functionality
python -c "from quantum_field import generate_quantum_field; print(generate_quantum_field(10, 10).shape)"

# Check version
python -c "from quantum_field.version import get_version; print(get_version())"

# Check available backends
python -c "from quantum_field.backends import get_available_backends; print(get_available_backends())"
```

## Platform-Specific Notes

### Windows

- For CUDA support, ensure you have the Microsoft Visual C++ Build Tools installed
- Some visualization features may require additional packages: `pip install pyvista plotly`

### Linux

- For ROCm support on Linux, ensure your kernel is compatible with your ROCm version
- For better performance, consider installing the Python packages from your distribution (e.g., `python3-numpy`, `python3-matplotlib`)

### macOS

- CUDA is not supported on macOS
- For Apple Silicon (M1/M2), ensure you use Python 3.9+ with ARM64 support
- Consider using Miniforge/Conda for optimized packages on Apple Silicon

## Docker Installation

A ready-to-use Docker image is available:

```bash
# Pull the image
docker pull yourname/quantum-field:latest

# Run with basic CPU support
docker run -it --rm yourname/quantum-field python -c "from quantum_field import generate_quantum_field; print(generate_quantum_field(10, 10).shape)"

# Run with GPU support (NVIDIA)
docker run --gpus all -it --rm yourname/quantum-field:cuda python -c "from quantum_field.backends import get_backend; backend = get_backend('cuda'); print(backend.generate_quantum_field(10, 10).shape)"
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'quantum_field'**
   - Verify the package is installed: `pip list | grep quantum-field`
   - Check your Python environment (virtualenv, conda, etc.)

2. **No CUDA-capable device is detected**
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Ensure your GPU is supported by your CUDA version

3. **No ROCm-capable device is detected**
   - Verify ROCm installation: `rocminfo`
   - Check that your AMD GPU is supported by ROCm

4. **Package not found: quantum-field**
   - Check your internet connection
   - Try using a different PyPI mirror: `pip install --index-url https://pypi.org/simple quantum-field`

### Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/yourname/quantum-field/issues) for similar problems and solutions
2. Join our [Discord server](https://discord.gg/yourname) for community support
3. Create a new GitHub issue with your system information and error messages