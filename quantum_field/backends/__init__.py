"""
Quantum Field Backends

This module provides a unified interface for different accelerator backends.
It detects and selects the most appropriate backend based on hardware availability.
"""

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from typing import Dict, Any, Optional, List, Tuple, Union
import importlib.util
import numpy as np

# Backend registry
available_backends = {}
active_backend = None

class AcceleratorBackend:
    """
    Base class for quantum field accelerator backends
    
    All hardware-specific implementations should inherit from this class
    and implement the required methods.
    """
    
    name = "base"
    priority = 0  # Higher priority backends are preferred
    
    def __init__(self):
        self.initialized = False
        self.capabilities = {
            "thread_block_clusters": False,
            "multi_device": False,
            "async_execution": False,
            "tensor_cores": False,
            "half_precision": False,
            "dlpack_support": False,
            "3d_fields": False,
        }
    
    def initialize(self) -> bool:
        """Initialize the backend"""
        self.initialized = True
        return True
    
    def is_available(self) -> bool:
        """Check if this backend is available on the current system"""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the backend and available hardware"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "capabilities": self.capabilities
        }
    
    def generate_quantum_field(self, width: int, height: int, 
                              frequency_name: str = 'love', 
                              time_factor: float = 0) -> np.ndarray:
        """
        Generate a quantum field with this backend
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        raise NotImplementedError("Backend must implement generate_quantum_field")
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field with this backend
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A float representing the field coherence
        """
        raise NotImplementedError("Backend must implement calculate_field_coherence")
    
    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a Phi-based sacred pattern with this backend
        
        Args:
            width: Width of the field
            height: Height of the field
            
        Returns:
            A 2D NumPy array representing the pattern field
        """
        raise NotImplementedError("Backend must implement generate_phi_pattern")
    
    def shutdown(self) -> None:
        """Release resources used by this backend"""
        self.initialized = False
        
    def to_dlpack(self, field_data: np.ndarray):
        """
        Convert a field to DLPack format for interoperability with ML frameworks
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A DLPack tensor that can be imported into ML frameworks
        """
        raise NotImplementedError("Backend must implement to_dlpack")
    
    def from_dlpack(self, dlpack_tensor, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert a DLPack tensor to a quantum field array
        
        Args:
            dlpack_tensor: A DLPack tensor
            shape: Optional shape to reshape the tensor to (height, width)
            
        Returns:
            A 2D NumPy array containing the field data
        """
        raise NotImplementedError("Backend must implement from_dlpack")


def register_backend(backend_class):
    """Register a backend for automatic detection"""
    backend = backend_class()
    if backend.is_available():
        available_backends[backend.name] = backend
    return backend_class


def get_available_backends() -> Dict[str, AcceleratorBackend]:
    """Get all available backends on the current system"""
    return available_backends


def get_backend(name: Optional[str] = None) -> AcceleratorBackend:
    """
    Get a specific backend by name, or the best available backend
    
    Args:
        name: Name of the backend to get, or None for best available
        
    Returns:
        An initialized backend instance
    """
    global active_backend
    
    # If a specific backend is requested
    if name is not None:
        if name in available_backends:
            backend = available_backends[name]
            if not backend.initialized:
                backend.initialize()
            active_backend = backend
            return backend
        else:
            raise ValueError(f"Backend '{name}' not available")
    
    # If we already have an active backend
    if active_backend is not None:
        return active_backend
    
    # Find the best available backend by priority
    if available_backends:
        best_backend = max(available_backends.values(), key=lambda b: b.priority)
        if not best_backend.initialized:
            best_backend.initialize()
        active_backend = best_backend
        return best_backend
    
    # If no backends are available, use CPU backend
    from quantum_field.backends.cpu import CPUBackend
    cpu_backend = CPUBackend()
    cpu_backend.initialize()
    available_backends[cpu_backend.name] = cpu_backend
    active_backend = cpu_backend
    return cpu_backend


def detect_backends() -> None:
    """
    Detect and initialize all available backends
    This function is called during module initialization
    """
    # Import all backend modules to register them
    try:
        from quantum_field.backends.cpu import CPUBackend
        register_backend(CPUBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.cuda import CUDABackend
        register_backend(CUDABackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.rocm import ROCmBackend
        register_backend(ROCmBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.oneapi import OneAPIBackend
        register_backend(OneAPIBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.tenstorrent import TenstorrentBackend
        register_backend(TenstorrentBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.mobile import MobileBackend
        register_backend(MobileBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.huawei_ascend import HuaweiAscendBackend
        register_backend(HuaweiAscendBackend)
    except ImportError:
        pass
    
    try:
        from quantum_field.backends.webgpu import WebGPUBackend
        register_backend(WebGPUBackend)
    except ImportError:
        pass
    
    print(f"Detected {len(available_backends)} quantum field accelerator backends")
    
    # Print available backends
    if available_backends:
        for name, backend in available_backends.items():
            info = backend.get_info()
            capabilities = ', '.join(cap for cap, enabled in info['capabilities'].items() if enabled)
            print(f"  - {name} backend available{' with: ' + capabilities if capabilities else ''}")
    else:
        print("  No acceleration backends detected, will use CPU implementation")


# Automatically detect backends when this module is imported
detect_backends()