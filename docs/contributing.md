# Contributing to Quantum Field Visualization

Thank you for your interest in contributing to the Quantum Field Visualization project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Project Structure](#project-structure)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Submitting Changes](#submitting-changes)
9. [Adding New Backends](#adding-new-backends)
10. [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to:

- Be respectful and considerate of others
- Use inclusive language
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/yourusername/quantum-field.git
   cd quantum-field
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/quantum-field.git
   ```
4. **Create a new branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Setting Up

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Recommended Tools

- **IDE**: Visual Studio Code with Python extensions
- **Linting**: flake8, pylint
- **Formatting**: black
- **Type checking**: mypy
- **Testing**: pytest

## Project Structure

```
quantum-field/
├── quantum_field/               # Main package
│   ├── __init__.py              # Package initialization
│   ├── constants.py             # Sacred constants and frequencies
│   ├── core.py                  # Core functionality
│   ├── backends/                # Backend implementations
│   │   ├── __init__.py          # Backend registration
│   │   ├── cpu.py               # CPU backend
│   │   ├── cuda.py              # CUDA backend
│   │   └── ...                  # Other backends
│   ├── multi_gpu.py             # Multi-GPU support
│   └── thread_block_cluster.py  # Thread block cluster support
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_core.py
│   └── ...                      # Other test files
├── docs/                        # Documentation
│   ├── user_guide.md
│   ├── api_reference.md
│   └── contributing.md          # This file
├── examples/                    # Usage examples
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
```

## Coding Standards

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with some additional guidelines:

### Structure & Organization
- **Imports order**: stdlib → third-party → local
- **Group imports** by module and sort alphabetically
- **Maximum line length**: 120 characters
- **Indentation**: 4 spaces (no tabs)
- **Use absolute imports** over relative imports

### Naming & Documentation
- **Classes**: `CamelCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private attributes**: `_leading_underscore`
- **Include docstrings** for all public modules, classes, and functions
- **Type hint** all function parameters and return values

### Principles & Patterns
- **Prefer clarity over cleverness**
- **Follow composition over inheritance**
- **Be explicit rather than implicit**
- **Use specific exception types** with context in error messages
- **Log at appropriate levels** (debug, info, warning, error)
- **Validate inputs** at function boundaries
- **Break complex operations** into φ-sized, testable functions

Run code quality checks before submitting:
```bash
# Format code
black .

# Check for style issues
flake8 .

# Run type checker
mypy .
```

## Testing

All new features and bug fixes should include tests. This project uses pytest for testing.

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_specific_file.py

# Run tests with coverage
pytest --cov=quantum_field tests/
```

### Writing Tests

- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`
- Use fixtures for setup and teardown
- Test both normal operation and error cases
- Include tests for edge cases

Example test structure:

```python
import pytest
import numpy as np
from quantum_field import generate_quantum_field

class TestFieldGeneration:
    def test_basic_generation(self):
        """Test basic field generation with default parameters"""
        field = generate_quantum_field(64, 64, "love", 0.0)
        assert field.shape == (64, 64)
        assert np.isfinite(field).all()
        assert -1.0 <= field.min() <= field.max() <= 1.0
        
    def test_invalid_parameters(self):
        """Test error handling with invalid parameters"""
        with pytest.raises(ValueError):
            generate_quantum_field(-10, 64, "love", 0.0)
```

## Documentation

Good documentation is crucial for this project. Please follow these guidelines:

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of what the function does.
    
    More detailed explanation if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```

### User Documentation

When adding new features, please update the relevant documentation:

- User Guide (docs/user_guide.md) for how to use the feature
- API Reference (docs/api_reference.md) for detailed API information

## Submitting Changes

1. **Commit your changes** with clear commit messages:
   ```
   Feature/component: Brief description of what changed
   
   More detailed explanation of why the change was made and how it works.
   Any breaking changes or caveats should be mentioned here.
   ```

2. **Keep your branch updated** with the upstream repository:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a pull request** via the GitHub interface.
   - Include a clear description of your changes
   - Reference any related issues
   - Make sure all tests pass
   - Ensure documentation is updated
   
5. **Respond to feedback** and make necessary changes.

## Adding New Backends

To add a new accelerator backend to the project:

1. **Create a new file** in the `quantum_field/backends/` directory
2. **Implement the `AcceleratorBackend` interface**:
   ```python
   from ..backends import AcceleratorBackend
   
   class NewBackend(AcceleratorBackend):
       name = "new_backend_name"
       priority = 75  # Set appropriate priority (0-100)
       
       def __init__(self):
           super().__init__()
           # Backend-specific initialization
           
       def is_available(self) -> bool:
           # Check if required dependencies and hardware are available
           
       def get_capabilities(self) -> Dict[str, bool]:
           return {
               "feature_1": True,
               "feature_2": False,
               # ...
           }
           
       def generate_quantum_field(self, width, height, frequency_name, time_factor, custom_frequency=None):
           # Implement field generation
           
       def calculate_field_coherence(self, field_data):
           # Implement coherence calculation
           
       def generate_phi_pattern(self, width, height):
           # Implement phi pattern generation
           
       def to_dlpack(self, field_data):
           # Implement DLPack conversion (if supported)
           
       def from_dlpack(self, dlpack_tensor, shape=None):
           # Implement DLPack conversion (if supported)
           
       def shutdown(self):
           # Clean up resources
   ```

3. **Register the backend** in `quantum_field/backends/__init__.py`:
   ```python
   from .new_backend import NewBackend
   register_backend(NewBackend)
   ```

4. **Add tests** in `tests/test_new_backend.py`

5. **Update documentation** to include the new backend

## Release Process

The project follows [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

The release process:

1. Update the version number in `quantum_field/version.py`
2. Update the CHANGELOG.md file with the changes
3. Create a git tag with the version number
4. Build and publish the package to PyPI
5. Create a GitHub release with release notes

## Thank You!

Your contributions are valuable to this project. If you have any questions or need help, please open an issue on GitHub or contact the project maintainers.