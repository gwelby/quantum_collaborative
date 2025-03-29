"""
ROCm Backend for Quantum Field Generation
This module provides GPU acceleration via AMD GPUs using HIP/ROCm.
"""
import os
import math
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
import logging

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.backends import AcceleratorBackend

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ROCm libraries
try:
    import torch
    ROCM_AVAILABLE = torch.cuda.is_available() and torch.version.hip is not None
    if ROCM_AVAILABLE:
        logger.info(f"ROCm backend available. PyTorch HIP version: {torch.version.hip}")
    else:
        logger.warning("ROCm not available. PyTorch HIP support not detected.")
except ImportError:
    logger.warning("PyTorch with ROCm support not available.")
    ROCM_AVAILABLE = False

class ROCmBackend(AcceleratorBackend):
    """
    ROCm Backend for Quantum Field Generation
    Leverages AMD GPUs for accelerated field generation and processing.
    """
    name = "rocm"
    priority = 80  # High priority but below Tenstorrent
    
    def __init__(self, device_id: int = 0):
        """
        Initialize the ROCm backend.
        
        Args:
            device_id: ID of the AMD GPU to use
        """
        super().__init__()
        self.device_id = device_id
        self._initialized = False
        self._device = None
        self._device_info = None
        
        # Cache for computed fields
        self._field_cache = {}
        self._coherence_cache = {}
        
        # Check availability
        self.available = ROCM_AVAILABLE

    def _initialize(self) -> bool:
        """Initialize the ROCm backend."""
        if self._initialized:
            return True
            
        if not ROCM_AVAILABLE:
            logger.warning("ROCm not available. Cannot initialize ROCm backend.")
            return False
            
        try:
            # Get device information
            if self.device_id >= torch.cuda.device_count():
                logger.error(f"Device ID {self.device_id} out of range. Available devices: {torch.cuda.device_count()}")
                return False
                
            # Set the device
            self._device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self.device_id)
            
            # Get device properties
            props = torch.cuda.get_device_properties(self.device_id)
            self._device_info = {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count
            }
            
            logger.info(f"Initialized ROCm backend on {props.name}")
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing ROCm backend: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the ROCm backend is available."""
        return self.available

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get the capabilities of this backend.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "thread_block_clusters": False,  # Not applicable for ROCm
            "multi_device": True,  # Multi-GPU is supported
            "async_execution": True,
            "tensor_cores": True,  # ROCm has matrix cores
            "half_precision": True
        }

    def generate_quantum_field(self, 
                           width: int, 
                           height: int, 
                           frequency_name: str = "love", 
                           time_factor: float = 0.0) -> np.ndarray:
        """
        Generate a quantum field using AMD GPU (ROCm).
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: Sacred frequency name
            time_factor: Time evolution factor
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        # Ensure the backend is initialized
        if not self._initialized and not self._initialize():
            logger.warning("ROCm backend not initialized. Falling back to CPU implementation.")
            return self._generate_field_cpu(width, height, frequency_name, time_factor)
            
        # Check cache first
        cache_key = (width, height, frequency_name, time_factor)
        if cache_key in self._field_cache:
            return self._field_cache[cache_key].copy()
            
        try:
            # Get the frequency value
            if frequency_name in SACRED_FREQUENCIES:
                frequency = SACRED_FREQUENCIES[frequency_name]
            else:
                frequency = 432.0  # Default to Ground frequency
            
            # Generate the field using GPU
            field = self._generate_field_gpu(width, height, frequency, time_factor)
            
            # Cache the result
            self._field_cache[cache_key] = field.copy()
            
            return field
        except Exception as e:
            logger.error(f"Error generating quantum field with ROCm: {e}")
            # Fallback to CPU implementation
            return self._generate_field_cpu(width, height, frequency_name, time_factor)

    def _generate_field_gpu(self, 
                         width: int, 
                         height: int, 
                         frequency: float, 
                         time_factor: float) -> np.ndarray:
        """
        Generate a quantum field using AMD GPU.
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency: Sacred frequency value
            time_factor: Time evolution factor
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        logger.info(f"Generating quantum field using ROCm: {width}x{height}, freq={frequency}")
        
        # Create coordinate tensors - directly on GPU
        y = torch.linspace(-1.0, 1.0, height, device=self._device)
        x = torch.linspace(-1.0, 1.0, width, device=self._device)
        
        # Create a meshgrid
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate distance from center with phi-weighting
        distance = torch.sqrt(x_grid*x_grid + y_grid*y_grid) * PHI
        angle = torch.atan2(y_grid, x_grid)
        
        # Apply frequency and time factor
        wave = torch.sin(distance * frequency * 0.01 + angle * PHI + time_factor * PHI_PHI)
        
        # Apply phi-harmonic dampening
        dampening = torch.exp(-distance * LAMBDA)
        
        # Combine wave and dampening
        field = wave * dampening
        
        # Convert to NumPy array
        field_np = field.cpu().numpy()
        
        return field_np

    def _generate_field_cpu(self, 
                          width: int, 
                          height: int, 
                          frequency_name: str, 
                          time_factor: float) -> np.ndarray:
        """
        Generate a quantum field using CPU (fallback).
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: Sacred frequency name
            time_factor: Time evolution factor
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        logger.info(f"Generating quantum field using CPU: {width}x{height}, freq={frequency_name}")
        
        # Get the frequency value
        if frequency_name in SACRED_FREQUENCIES:
            frequency = SACRED_FREQUENCIES[frequency_name]
        else:
            frequency = 432.0  # Default to Ground frequency
        
        # Create the field
        field = np.zeros((height, width), dtype=np.float32)
        
        # Initialize with a phi-harmonic pattern
        for y in range(height):
            ny = 2.0 * y / height - 1.0
            for x in range(width):
                nx = 2.0 * x / width - 1.0
                
                # Calculate distance from center with phi-weighting
                distance = math.sqrt(nx * nx + ny * ny) * PHI
                angle = math.atan2(ny, nx)
                
                # Apply frequency and time factor
                wave = math.sin(distance * frequency * 0.01 + angle * PHI + time_factor * PHI_PHI)
                
                # Apply phi-harmonic dampening
                dampening = math.exp(-distance * LAMBDA)
                
                field[y, x] = wave * dampening
        
        return field

    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field using AMD GPU.
        
        Args:
            field_data: 2D array of field data
            
        Returns:
            Coherence level (0.0-1.0)
        """
        # Ensure the backend is initialized
        if not self._initialized and not self._initialize():
            logger.warning("ROCm backend not initialized. Falling back to CPU implementation.")
            return self._calculate_coherence_cpu(field_data)
            
        # Check cache first
        cache_key = hash(field_data.tobytes())
        if cache_key in self._coherence_cache:
            return self._coherence_cache[cache_key]
            
        try:
            # Calculate coherence using GPU
            coherence = self._calculate_coherence_gpu(field_data)
            
            # Cache the result
            self._coherence_cache[cache_key] = coherence
            
            return coherence
        except Exception as e:
            logger.error(f"Error calculating field coherence with ROCm: {e}")
            # Fallback to CPU implementation
            return self._calculate_coherence_cpu(field_data)

    def _calculate_coherence_gpu(self, field_data: np.ndarray) -> float:
        """
        Calculate field coherence using AMD GPU.
        
        Args:
            field_data: 2D array of field data
            
        Returns:
            Coherence level (0.0-1.0)
        """
        logger.info(f"Calculating field coherence using ROCm: shape={field_data.shape}")
        
        # Convert field data to tensor
        field_tensor = torch.tensor(field_data, dtype=torch.float32, device=self._device)
        
        # Calculate field statistics
        field_mean = torch.mean(field_tensor)
        field_std = torch.std(field_tensor)
        
        if field_std == 0:
            return 1.0  # Perfect coherence if all values are identical
        
        # Normalize field data
        field_norm = (field_tensor - field_mean) / field_std
        
        # Calculate spatial coherence using convolution
        # This is much faster than the explicit double loop in the CPU version
        height, width = field_data.shape
        
        # Create correlation kernels
        kernel_h = torch.tensor([[-PHI, PHI]], dtype=torch.float32, device=self._device)
        kernel_v = torch.tensor([[-PHI], [PHI]], dtype=torch.float32, device=self._device)
        
        # Reshape for 2D convolution
        field_norm_reshaped = field_norm.view(1, 1, height, width)
        kernel_h_reshaped = kernel_h.view(1, 1, 1, 2)
        kernel_v_reshaped = kernel_v.view(1, 1, 2, 1)
        
        # Apply convolution for horizontal and vertical correlations
        h_corr = torch.nn.functional.conv2d(field_norm_reshaped, kernel_h_reshaped, padding=(0, 1))
        v_corr = torch.nn.functional.conv2d(field_norm_reshaped, kernel_v_reshaped, padding=(1, 0))
        
        # Calculate correlation
        correlation = torch.abs(h_corr) + torch.abs(v_corr)
        correlation = correlation * LAMBDA
        
        # Calculate coherence
        coherence_sum = torch.sum(correlation)
        
        # Normalize by field size
        coherence = coherence_sum / (height * width)
        
        # Scale to 0-1 range
        coherence = min(1.0, coherence.item())
        
        return coherence

    def _calculate_coherence_cpu(self, field_data: np.ndarray) -> float:
        """
        Calculate field coherence using CPU (fallback).
        
        Args:
            field_data: 2D array of field data
            
        Returns:
            Coherence level (0.0-1.0)
        """
        logger.info(f"Calculating field coherence using CPU: shape={field_data.shape}")
        
        # Calculate field statistics
        field_mean = np.mean(field_data)
        field_std = np.std(field_data)
        
        if field_std == 0:
            return 1.0  # Perfect coherence if all values are identical
        
        # Normalize field data
        field_norm = (field_data - field_mean) / field_std
        
        # Calculate spatial coherence
        height, width = field_data.shape
        coherence_sum = 0.0
        
        # Calculate auto-correlation
        for y in range(1, height):
            for x in range(1, width):
                # Calculate correlation with neighboring points
                correlation = field_norm[y, x] * field_norm[y-1, x] * PHI
                correlation += field_norm[y, x] * field_norm[y, x-1] * PHI
                correlation *= LAMBDA
                
                coherence_sum += abs(correlation)
        
        # Normalize by field size
        coherence = coherence_sum / (height * width)
        
        # Scale to 0-1 range
        coherence = min(1.0, coherence)
        
        return coherence

    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a phi pattern using AMD GPU.
        
        Args:
            width: Width of the pattern
            height: Height of the pattern
            
        Returns:
            A 2D NumPy array representing the pattern
        """
        # Ensure the backend is initialized
        if not self._initialized and not self._initialize():
            logger.warning("ROCm backend not initialized. Falling back to CPU implementation.")
            return self._generate_phi_pattern_cpu(width, height)
            
        # Check cache
        cache_key = (width, height)
        if cache_key in self._field_cache:
            return self._field_cache[cache_key].copy()
            
        try:
            # Generate the pattern using GPU
            pattern = self._generate_phi_pattern_gpu(width, height)
            
            # Cache the result
            self._field_cache[cache_key] = pattern.copy()
            
            return pattern
        except Exception as e:
            logger.error(f"Error generating phi pattern with ROCm: {e}")
            # Fallback to CPU implementation
            return self._generate_phi_pattern_cpu(width, height)

    def _generate_phi_pattern_gpu(self, width: int, height: int) -> np.ndarray:
        """
        Generate a phi pattern using AMD GPU.
        
        Args:
            width: Width of the pattern
            height: Height of the pattern
            
        Returns:
            A 2D NumPy array representing the pattern
        """
        logger.info(f"Generating phi pattern using ROCm: {width}x{height}")
        
        # Create coordinate tensors - directly on GPU
        y = torch.linspace(-1.0, 1.0, height, device=self._device)
        x = torch.linspace(-1.0, 1.0, width, device=self._device)
        
        # Create a meshgrid
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate spiral pattern based on phi
        angle = torch.atan2(y_grid, x_grid)
        radius = torch.sqrt(x_grid*x_grid + y_grid*y_grid)
        
        # Phi spiral formula
        spiral = torch.sin(radius * PHI_PHI * 10.0 + angle * PHI)
        
        # Convert to NumPy array
        pattern_np = spiral.cpu().numpy()
        
        return pattern_np

    def _generate_phi_pattern_cpu(self, width: int, height: int) -> np.ndarray:
        """
        Generate a phi pattern using CPU (fallback).
        
        Args:
            width: Width of the pattern
            height: Height of the pattern
            
        Returns:
            A 2D NumPy array representing the pattern
        """
        logger.info(f"Generating phi pattern using CPU: {width}x{height}")
        
        # Create pattern data
        pattern = np.zeros((height, width), dtype=np.float32)
        
        # Fill with phi-harmonic spiral pattern
        for y in range(height):
            ny = 2.0 * y / height - 1.0
            for x in range(width):
                nx = 2.0 * x / width - 1.0
                
                # Calculate spiral pattern based on phi
                angle = math.atan2(ny, nx)
                radius = math.sqrt(nx * nx + ny * ny)
                
                # Phi spiral formula
                spiral = math.sin(radius * PHI_PHI * 10.0 + angle * PHI)
                
                pattern[y, x] = spiral
        
        return pattern

    def shutdown(self):
        """Release GPU resources."""
        if self._initialized:
            logger.info("Shutting down ROCm backend...")
            torch.cuda.empty_cache()
            self._initialized = False