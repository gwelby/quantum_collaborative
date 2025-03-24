# Setting Up a GitHub Repository for Quantum Field Visualization

This guide provides step-by-step instructions for creating and configuring a GitHub repository for the Quantum Field Visualization project. Following these steps will give you a professional repository with all the necessary components for open-source collaboration.

## 1. Create the Repository

1. Sign in to your GitHub account
2. Click the "+" icon in the top-right corner and select "New repository"
3. Fill in the repository details:
   - **Repository name**: quantum-field
   - **Description**: A Python package for generating and visualizing quantum fields based on phi-harmonic principles
   - **Visibility**: Public
   - **Initialize with**:
     - README file
     - .gitignore (Python template)
     - Choose a license (MIT recommended)
4. Click "Create repository"

## 2. Clone the Repository Locally

```bash
git clone https://github.com/yourusername/quantum-field.git
cd quantum-field
```

## 3. Set Up the Initial Directory Structure

Create the basic directory structure:

```bash
# Create main package directory
mkdir -p quantum_field
touch quantum_field/__init__.py

# Create docs directory
mkdir -p docs/assets docs/stylesheets

# Create tests directory
mkdir -p tests
touch tests/__init__.py

# Create GitHub workflows directory
mkdir -p .github/workflows

# Create examples directory
mkdir -p examples

# Create other necessary directories
mkdir -p .github/ISSUE_TEMPLATE
```

## 4. Create Essential Files

### setup.py

Create a setup.py file in the root directory:

```python
#!/usr/bin/env python3
"""
Setup script for quantum_field package
"""

import os
from setuptools import setup, find_packages

# Get version from version.py
with open(os.path.join("quantum_field", "version.py"), "r") as f:
    exec(f.read())

# Read the README.md for the long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="quantum-field",
    version=__version__,
    description="Quantum Field Visualization with Hardware Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/quantum-field",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "cuda": [
            "cuda-python>=12.0.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "rocm": [
            "torch>=2.0.0",
        ],
        "oneapi": [
            "intel-extension-for-pytorch>=1.13.0",
        ],
        "all": [
            "cuda-python>=12.0.0",
            "cupy-cuda12x>=12.0.0",
            "torch>=2.0.0",
            "tensorflow>=2.10.0",
            "jax>=0.4.0",
            "intel-extension-for-pytorch>=1.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "quantum-field=quantum_field.__main__:main",
        ],
    },
)
```

### Version File

Create a version.py file in the quantum_field directory:

```python
"""
Version information for quantum_field package
"""

__version__ = "0.1.0"

def get_version():
    """
    Get the current version of the library.
    
    Returns:
        Version string in format "X.Y.Z"
    """
    return __version__
```

### Main Module File

Create `quantum_field/__main__.py`:

```python
#!/usr/bin/env python3
"""
Command-line interface for quantum_field package
"""

import sys
import argparse
from .version import get_version

def main():
    """Main entry point for the command line interface"""
    parser = argparse.ArgumentParser(
        description="Quantum Field Visualization with Hardware Acceleration"
    )
    
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Version command
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
        
    args = parser.parse_args()
    
    if args.version:
        print(f"Quantum Field Visualization v{get_version()}")
        return 0
        
    # Placeholder for actual command handling
    print(f"Command '{args.command}' not implemented yet")
    return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Update README.md

Update the README.md with more comprehensive content (see the template we created earlier in this session).

### Create a CHANGELOG.md

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - YYYY-MM-DD

### Added
- Initial package structure
- Basic quantum field generation functionality
- CPU backend implementation
- Sacred constants definition
- Command-line interface
```

### Create a CONTRIBUTING.md

```markdown
# Contributing to Quantum Field Visualization

Thank you for considering contributing to Quantum Field Visualization! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template when creating a new issue
- Include detailed steps to reproduce the bug
- Include system information (OS, Python version, hardware details)

### Suggesting Enhancements

- Check if the enhancement has already been suggested
- Use the feature request template
- Explain why this enhancement would be useful to most users

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Guidelines

### Setting Up Your Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-field.git
cd quantum-field

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode with development dependencies
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints to functions and methods
- Write docstrings for all public modules, functions, classes, and methods

### Testing

- Write tests for all new features and bug fixes
- Run the test suite before submitting a PR:

```bash
pytest
```

- Check code coverage:

```bash
pytest --cov=quantum_field
```

### Documentation

- Update documentation for all changes
- Include examples for new features
- Follow Google-style docstrings

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
```

### Create .github/ISSUE_TEMPLATE/bug_report.md

```markdown
---
name: Bug report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Install with '...'
2. Run code '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code Sample**
If applicable, add a minimal code sample to reproduce the issue.

**System Information:**
 - OS: [e.g., Ubuntu 22.04, Windows 11]
 - Python version: [e.g., 3.10.4]
 - Package version: [e.g., 0.1.0]
 - Hardware details: [e.g., NVIDIA RTX 3080, AMD Radeon RX 6800]

**Additional context**
Add any other context about the problem here.
```

### Create .github/ISSUE_TEMPLATE/feature_request.md

```markdown
---
name: Feature request
about: Suggest an idea for this project
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### Create GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        pip install -e ".[dev]"
    - name: Test with pytest
      run: |
        pytest --cov=quantum_field
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        fail_ci_if_error: false
```

## 5. Initial Commit and Push

After creating all these files, commit and push them:

```bash
# Add all files
git add .

# Commit
git commit -m "Initial project structure"

# Push to GitHub
git push origin main
```

## 6. Set Up GitHub Pages for Documentation

1. Go to your repository on GitHub
2. Click on "Settings"
3. Scroll down to "GitHub Pages" section
4. Select the source as "GitHub Actions"
5. Create the necessary GitHub Actions workflow (see the docs.yml we created earlier)

## 7. Configure Repository Settings

1. Go to "Settings" > "Options"
2. Scroll down to "Features" section:
   - Enable "Issues"
   - Enable "Discussions"
   - Enable "Wikis" (optional)
   - Enable "Sponsorships" (optional)
3. Go to "Settings" > "Branches":
   - Add branch protection rule for `main`
   - Require pull request reviews before merging
   - Require status checks to pass before merging

## 8. Create an Initial Release

1. Go to "Releases" in your repository
2. Click "Create a new release"
3. Set the tag to "v0.1.0"
4. Set the title to "Initial Release"
5. Add release notes based on your CHANGELOG
6. Publish the release

## 9. Set Up PyPI Publishing

1. Create an account on PyPI if you don't have one
2. Generate an API token in your PyPI account settings
3. Add the token as a secret in your GitHub repository settings (name it `PYPI_API_TOKEN`)
4. Create the GitHub Actions workflow for PyPI publishing (see the publish.yml we created earlier)

## 10. Next Steps

After setting up the repository, you can:

1. Implement the core functionality of the package
2. Write comprehensive tests
3. Complete the documentation
4. Create example scripts
5. Publicize your project on relevant platforms