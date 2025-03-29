#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum_multilang",
    version="0.1.0",
    author="Quantum Field Team",
    author_email="info@quantumfield.example",
    description="Multi-language quantum field framework with Python controller",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-field/quantum-multilang",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pyzmq>=22.0.0",
        "cffi>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "mypy>=0.910",
        ],
    },
)