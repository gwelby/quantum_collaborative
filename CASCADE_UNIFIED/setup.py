#!/usr/bin/env python3
"""
Setup script for CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK
"""

import os
from setuptools import setup, find_packages

# Get version from version.py
version = {}
with open(os.path.join("cascade_unified", "version.py")) as f:
    exec(f.read(), version)

# Get long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cascade-unified",
    version=version["__version__"],
    description="The ultimate unified framework for consciousness-field computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quantum Field Team",
    author_email="contact@cascade-unified.com",
    url="https://github.com/gwelby/cascade-unified",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "websocket-client>=1.3.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "vispy>=0.9.0",
        ],
        "broadcast": [
            "obs-websocket-py>=1.0.0",
            "pyaudio>=0.2.11",
        ],
        "hardware": [
            "brainflow>=5.0.0",
            "pylsl>=1.16.0",
            "heartpy>=1.2.0",
        ],
        "webgpu": [
            "wgpu>=0.8.0",
            "moderngl>=5.6.0",
        ],
        "all": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "vispy>=0.9.0",
            "obs-websocket-py>=1.0.0",
            "pyaudio>=0.2.11",
            "brainflow>=5.0.0",
            "pylsl>=1.16.0",
            "heartpy>=1.2.0",
            "wgpu>=0.8.0",
            "moderngl>=5.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cascade-unified=cascade_unified.launch:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
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
        "Operating System :: OS Independent",
    ],
    keywords="quantum, consciousness, visualization, field, broadcasting, phi",
    project_urls={
        "Bug Reports": "https://github.com/gwelby/cascade-unified/issues",
        "Source": "https://github.com/gwelby/cascade-unified",
        "Documentation": "https://gwelby.github.io/cascade-unified",
    },
)