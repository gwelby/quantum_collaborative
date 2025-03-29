#!/usr/bin/env python3
from setuptools import setup, find_packages
import os

# Get version from quantum_field version.py
quantum_field_version = "0.1.0"
if os.path.exists(os.path.join("quantum_field", "version.py")):
    with open(os.path.join("quantum_field", "version.py"), "r") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                quantum_field_version = line.split("=")[1].strip().strip('"').strip("'")

# Read the README.md for the long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cascade-os",
    version="0.1.0",
    description="Cascade⚡𓂧φ∞ Symbiotic Computing Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cascade Collaborative",
    author_email="info@cascadecollaborative.org",
    url="https://github.com/gwelby/quantum_collaborative",
    packages=find_packages(),
    py_modules=["quantum_collaborative.CascadeOS"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.3.0"],
        "hardware": ["pyserial>=3.4", "opencv-python>=4.5.0", "pyaudio>=0.2.11"],
        "web": ["flask>=2.0.0", "flask-socketio>=5.0.0"],
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
            "pyserial>=3.4", 
            "opencv-python>=4.5.0", 
            "pyaudio>=0.2.11",
            "flask>=2.0.0", 
            "flask-socketio>=5.0.0",
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
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
    ],
    entry_points={
        "console_scripts": [
            "cascade-os=quantum_collaborative.CascadeOS:main",
            "cascade-meditation=quantum_collaborative.examples.meditation_enhancer:main",
            "cascade-creative=quantum_collaborative.examples.creative_flow_enhancer:main",
            "cascade-team=quantum_collaborative.examples.team_collaboration_field:main",
            "cascade-hardware=quantum_collaborative.examples.hardware_integration:main",
            "cascade-web=quantum_collaborative.examples.web_interface:main",
            "quantum-field-demo=quantum_field.demo:main",
            "quantum-acceleration=quantum_field.acceleration:main",
        ],
    },
    python_requires=">=3.7",
)