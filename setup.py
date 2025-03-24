#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="cascade-os",
    version="0.1.0",
    description="Cascade⚡𓂧φ∞ Symbiotic Computing Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Cascade Collaborative",
    author_email="info@cascadecollaborative.org",
    url="https://github.com/cascadecollaborative/cascade-os",
    packages=find_packages(),
    py_modules=["quantum_collaborative.CascadeOS"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.3.0"],
        "hardware": ["pyserial>=3.4", "opencv-python>=4.5.0", "pyaudio>=0.2.11"],
        "web": ["flask>=2.0.0", "flask-socketio>=5.0.0"],
        "dev": ["pytest>=6.0.0", "black>=20.8b1", "mypy>=0.800"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
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
        ],
    },
    python_requires=">=3.7",
)