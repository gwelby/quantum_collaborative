"""
Mobile Backend for Quantum Field Generation

This module provides acceleration for mobile GPUs on Android and iOS devices.
It supports Metal on iOS and Vulkan/OpenCL on Android.
"""

import os
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from quantum_field.backends import AcceleratorBackend
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# Try to import mobile acceleration modules
try:
    # Try to import PyTorch Mobile (provides unified layer for Metal/Vulkan)
    import torch
    TORCH_MOBILE_AVAILABLE = hasattr(torch, 'is_vulkan_available') or hasattr(torch, 'is_metal_available')
except ImportError:
    TORCH_MOBILE_AVAILABLE = False

# Try to import Vulkan for Android
try:
    import vulkan
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False

# Try to import Metal for iOS
try:
    import pyobjc_framework_Metal as metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class MobileBackend(AcceleratorBackend):
    """
    Mobile GPU implementation of quantum field operations
    
    Supports:
    - Metal on iOS devices
    - Vulkan/OpenCL on Android devices
    - PyTorch Mobile as a unified backend
    """
    
    name = "mobile"
    priority = 70  # High priority on mobile devices, but lower than CUDA/ROCm
    
    def __init__(self):
        super().__init__()
        self.capabilities = {
            "thread_block_clusters": False,
            "multi_device": False,
            "async_execution": True,
            "tensor_cores": False,
            "half_precision": True,  # Most modern mobile GPUs support FP16
            "dlpack_support": TORCH_MOBILE_AVAILABLE,  # DLPack support via PyTorch
        }
        self.implementation = None
        self.device_info = {}
    
    def initialize(self) -> bool:
        """Initialize the mobile GPU backend"""
        if not self.is_available():
            return False
        
        try:
            # Determine which implementation to use
            if TORCH_MOBILE_AVAILABLE:
                self.implementation = "pytorch"
                return self._initialize_pytorch()
            elif VULKAN_AVAILABLE:
                self.implementation = "vulkan"
                return self._initialize_vulkan()
            elif METAL_AVAILABLE:
                self.implementation = "metal"
                return self._initialize_metal()
            
            return False
        except Exception as e:
            print(f"Error initializing mobile backend: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if mobile GPU acceleration is available"""
        # Check for PyTorch Mobile
        if TORCH_MOBILE_AVAILABLE:
            try:
                if hasattr(torch, 'is_vulkan_available') and torch.is_vulkan_available():
                    return True
                if hasattr(torch, 'is_metal_available') and torch.is_metal_available():
                    return True
            except:
                pass
        
        # Check for Vulkan
        if VULKAN_AVAILABLE:
            try:
                instance = vulkan.Instance(application_info=vulkan.ApplicationInfo(
                    application_name="QuantumField",
                    application_version=vulkan.make_version(1, 0, 0),
                    engine_name="QuantumField",
                    engine_version=vulkan.make_version(1, 0, 0),
                    api_version=vulkan.make_version(1, 0, 0)
                ))
                physical_devices = instance.enumerate_physical_devices()
                return len(physical_devices) > 0
            except:
                pass
        
        # Check for Metal
        if METAL_AVAILABLE:
            try:
                device = metal.MTLCreateSystemDefaultDevice()
                return device is not None
            except:
                pass
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the mobile backend"""
        info = super().get_info()
        
        info.update({
            "implementation": self.implementation,
            "device_info": self.device_info,
            "pytorch_available": TORCH_MOBILE_AVAILABLE,
            "vulkan_available": VULKAN_AVAILABLE,
            "metal_available": METAL_AVAILABLE
        })
        
        return info
    
    def _initialize_pytorch(self) -> bool:
        """Initialize using PyTorch Mobile"""
        try:
            # Initialize device
            if hasattr(torch, 'is_vulkan_available') and torch.is_vulkan_available():
                self.device = torch.device('vulkan')
                self.device_info = {
                    "name": "Vulkan via PyTorch",
                    "type": "vulkan",
                    "memory": "Unknown"  # PyTorch doesn't expose this
                }
            elif hasattr(torch, 'is_metal_available') and torch.is_metal_available():
                self.device = torch.device('mps')
                self.device_info = {
                    "name": "Metal via PyTorch",
                    "type": "metal",
                    "memory": "Unknown"  # PyTorch doesn't expose this
                }
            else:
                return False
            
            # Compile kernels as PyTorch modules
            self._compile_pytorch_modules()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing PyTorch mobile: {e}")
            return False
    
    def _initialize_vulkan(self) -> bool:
        """Initialize using Vulkan directly"""
        try:
            # Initialize Vulkan
            self.instance = vulkan.Instance(application_info=vulkan.ApplicationInfo(
                application_name="QuantumField",
                application_version=vulkan.make_version(1, 0, 0),
                engine_name="QuantumField",
                engine_version=vulkan.make_version(1, 0, 0),
                api_version=vulkan.make_version(1, 0, 0)
            ))
            
            # Get physical device
            physical_devices = self.instance.enumerate_physical_devices()
            if not physical_devices:
                return False
                
            self.physical_device = physical_devices[0]
            device_properties = self.physical_device.get_properties()
            
            # Store device info
            self.device_info = {
                "name": device_properties.device_name,
                "type": "vulkan",
                "driver_version": device_properties.driver_version,
                "api_version": device_properties.api_version,
                "device_type": device_properties.device_type
            }
            
            # TODO: Compile Vulkan shader modules
            # This would require more in-depth Vulkan setup code
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Vulkan: {e}")
            return False
    
    def _initialize_metal(self) -> bool:
        """Initialize using Metal on iOS"""
        try:
            # Initialize Metal
            self.device = metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                return False
                
            # Store device info
            self.device_info = {
                "name": self.device.name(),
                "type": "metal",
                "registry_id": self.device.registryID(),
                "headless": self.device.isHeadless(),
                "low_power": self.device.isLowPower(),
                "memory": self.device.recommendedMaxWorkingSetSize() / (1024 * 1024)  # Convert to MB
            }
            
            # TODO: Compile Metal shader modules
            # This would require more in-depth Metal setup code
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Metal: {e}")
            return False
    
    def _compile_pytorch_modules(self) -> bool:
        """Create PyTorch modules for quantum field operations"""
        # Only implement this if we're using PyTorch
        if self.implementation != "pytorch":
            return False
            
        try:
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Define a PyTorch module for quantum field generation
            class QuantumFieldGenerator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.phi = nn.Parameter(torch.tensor(PHI, dtype=torch.float32), requires_grad=False)
                    self.lambda_val = nn.Parameter(torch.tensor(LAMBDA, dtype=torch.float32), requires_grad=False)
                
                def forward(self, width, height, frequency, time_factor):
                    # Create coordinate grids
                    y_coords = torch.arange(height, dtype=torch.float32, device=self.phi.device)
                    x_coords = torch.arange(width, dtype=torch.float32, device=self.phi.device)
                    
                    # Normalize coordinates to center
                    center_y = height / 2
                    center_x = width / 2
                    y_coords = (y_coords - center_y) / (height / 2)
                    x_coords = (x_coords - center_x) / (width / 2)
                    
                    # Create meshgrid
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    
                    # Calculate distance from center
                    distance = torch.sqrt(x_grid**2 + y_grid**2)
                    
                    # Calculate angle
                    angle = torch.atan2(y_grid, x_grid) * self.phi
                    
                    # Calculate field values
                    freq_factor = frequency / 1000.0 * self.phi
                    time_value = time_factor * self.lambda_val
                    
                    # Create interference pattern
                    field = (
                        torch.sin(distance * freq_factor + time_value) * 
                        torch.cos(angle * self.phi) * 
                        torch.exp(-distance / self.phi)
                    )
                    
                    return field
            
            # Create the module and move it to device
            self.field_generator = QuantumFieldGenerator().to(self.device)
            
            # Define a module for phi pattern generation
            class PhiPatternGenerator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.phi = nn.Parameter(torch.tensor(PHI, dtype=torch.float32), requires_grad=False)
                
                def forward(self, width, height):
                    # Create normalized coordinate grids (-1 to 1)
                    y_coords = torch.arange(height, dtype=torch.float32, device=self.phi.device)
                    x_coords = torch.arange(width, dtype=torch.float32, device=self.phi.device)
                    
                    # Normalize to -1 to 1
                    y_coords = 2 * (y_coords / height - 0.5)
                    x_coords = 2 * (x_coords / width - 0.5)
                    
                    # Create meshgrid
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    
                    # Calculate radius and angle
                    r = torch.sqrt(x_grid**2 + y_grid**2)
                    a = torch.atan2(y_grid, x_grid)
                    
                    # Create phi spiral pattern
                    pattern = torch.sin(self.phi * r * 10) * torch.cos(a * self.phi * 5)
                    
                    return pattern
            
            # Create the module and move it to device
            self.pattern_generator = PhiPatternGenerator().to(self.device)
            
            return True
        except Exception as e:
            print(f"Error compiling PyTorch modules: {e}")
            return False
    
    def generate_quantum_field(self, width: int, height: int, 
                              frequency_name: str = 'love', 
                              time_factor: float = 0) -> np.ndarray:
        """
        Generate a quantum field using mobile GPU
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        if not self.initialized:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_quantum_field(width, height, frequency_name, time_factor)
        
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Generate using the appropriate implementation
        if self.implementation == "pytorch":
            return self._generate_field_pytorch(width, height, frequency, time_factor)
        elif self.implementation == "vulkan":
            return self._generate_field_vulkan(width, height, frequency, time_factor)
        elif self.implementation == "metal":
            return self._generate_field_metal(width, height, frequency, time_factor)
        else:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_quantum_field(width, height, frequency_name, time_factor)
    
    def _generate_field_pytorch(self, width: int, height: int, 
                               frequency: float, time_factor: float) -> np.ndarray:
        """Generate quantum field using PyTorch"""
        try:
            with torch.no_grad():
                # Generate field using PyTorch module
                field = self.field_generator(width, height, frequency, time_factor)
                
                # Convert to NumPy array
                return field.cpu().numpy()
        except Exception as e:
            print(f"Error in PyTorch field generation: {e}")
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_quantum_field(width, height, 
                                                  next(k for k, v in SACRED_FREQUENCIES.items() if v == frequency), 
                                                  time_factor)
    
    def _generate_field_vulkan(self, width: int, height: int, 
                              frequency: float, time_factor: float) -> np.ndarray:
        """Generate quantum field using Vulkan"""
        # This would require a full Vulkan implementation with compute shaders
        # For now, we'll fall back to the CPU implementation
        print("Direct Vulkan implementation not available yet, falling back to CPU")
        from quantum_field.backends.cpu import CPUBackend
        cpu_backend = CPUBackend()
        cpu_backend.initialize()
        return cpu_backend.generate_quantum_field(width, height, 
                                              next(k for k, v in SACRED_FREQUENCIES.items() if v == frequency), 
                                              time_factor)
    
    def _generate_field_metal(self, width: int, height: int, 
                             frequency: float, time_factor: float) -> np.ndarray:
        """Generate quantum field using Metal"""
        # This would require a full Metal implementation with compute shaders
        # For now, we'll fall back to the CPU implementation
        print("Direct Metal implementation not available yet, falling back to CPU")
        from quantum_field.backends.cpu import CPUBackend
        cpu_backend = CPUBackend()
        cpu_backend.initialize()
        return cpu_backend.generate_quantum_field(width, height, 
                                              next(k for k, v in SACRED_FREQUENCIES.items() if v == frequency), 
                                              time_factor)
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A float representing the field coherence
        """
        # Mobile implementations often benefit from falling back to CPU for 
        # this operation, as the data transfer overhead can outweigh benefits
        # for smaller computations like coherence calculation
        from quantum_field.backends.cpu import CPUBackend
        cpu_backend = CPUBackend()
        cpu_backend.initialize()
        return cpu_backend.calculate_field_coherence(field_data)
    
    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a Phi-based sacred pattern
        
        Args:
            width: Width of the field
            height: Height of the field
            
        Returns:
            A 2D NumPy array representing the pattern field
        """
        if not self.initialized:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_phi_pattern(width, height)
        
        # Generate using the appropriate implementation
        if self.implementation == "pytorch":
            try:
                with torch.no_grad():
                    # Generate pattern using PyTorch module
                    pattern = self.pattern_generator(width, height)
                    
                    # Convert to NumPy array
                    return pattern.cpu().numpy()
            except Exception as e:
                print(f"Error in PyTorch pattern generation: {e}")
        
        # Fall back to CPU for other implementations or on error
        from quantum_field.backends.cpu import CPUBackend
        cpu_backend = CPUBackend()
        cpu_backend.initialize()
        return cpu_backend.generate_phi_pattern(width, height)
    
    def shutdown(self) -> None:
        """Release resources used by this backend"""
        try:
            if self.implementation == "pytorch":
                # Clear PyTorch caches
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Delete PyTorch modules
                if hasattr(self, 'field_generator'):
                    del self.field_generator
                if hasattr(self, 'pattern_generator'):
                    del self.pattern_generator
            elif self.implementation == "vulkan":
                # Cleanup Vulkan resources
                if hasattr(self, 'instance'):
                    del self.instance
            elif self.implementation == "metal":
                # Cleanup Metal resources
                if hasattr(self, 'device'):
                    del self.device
        except Exception as e:
            print(f"Error shutting down mobile backend: {e}")
        
        self.initialized = False