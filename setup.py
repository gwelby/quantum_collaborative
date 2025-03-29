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
    name="quantum_field",
    version=__version__,
    description="Quantum Field Visualization with CUDA Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quantum Field Team",
    author_email="example@example.com",
    url="https://github.com/example/quantum_field",
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
            # ROCm backend requires PyTorch with ROCm support
        ],
        "oneapi": [
            # Intel oneAPI backend requirements
            "intel-extension-for-pytorch>=1.13.0",
        ],
        "webgpu": [
            "pywebgpu>=0.1.0",
            "wgpu>=0.9.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "tensorflow>=2.10.0",
            "jax>=0.4.0",
        ],
        "mobile": [
            "torch>=2.0.0",
        ],
        "all": [
            "cuda-python>=12.0.0",
            "cupy-cuda12x>=12.0.0",
            "torch>=2.0.0",
            "tensorflow>=2.10.0",
            "jax>=0.4.0",
            "intel-extension-for-pytorch>=1.13.0",
            "pywebgpu>=0.1.0",
            "wgpu>=0.9.0",
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
            "quantum-field-demo=quantum_field.demo:main",
            "quantum-acceleration=quantum_field.acceleration:main",
        ],
    },
)