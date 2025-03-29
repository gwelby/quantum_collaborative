"""
CPU Backend for Quantum Field Generation

This module provides a CPU implementation of quantum field operations,
serving both as a fallback when hardware acceleration is unavailable
and as a reference implementation for other backends.
"""

import os
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor

from quantum_field.backends import AcceleratorBackend
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES


class CPUBackend(AcceleratorBackend):
    """
    CPU implementation of quantum field operations
    
    Uses multi-threading for improved performance on multi-core CPUs.
    """
    
    name = "cpu"
    priority = 0  # Lowest priority - always available as fallback
    
    def __init__(self):
        super().__init__()
        self.num_threads = max(1, os.cpu_count() or 4)
        self.executor = None
        
        # Check if PyTorch is available for DLPack support
        try:
            import torch
            has_dlpack = hasattr(torch.utils, 'dlpack')
        except ImportError:
            has_dlpack = False
            
        self.capabilities = {
            "thread_block_clusters": False,
            "multi_device": False,
            "async_execution": False,
            "tensor_cores": False,
            "half_precision": False,
            "dlpack_support": has_dlpack,
            "3d_fields": True,
        }
    
    def initialize(self) -> bool:
        """Initialize the CPU backend"""
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.initialized = True
        return True
    
    def is_available(self) -> bool:
        """CPU backend is always available"""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the CPU backend"""
        info = super().get_info()
        info.update({
            "num_threads": self.num_threads,
            "architecture": "x86_64",  # This would be more accurately determined at runtime
            "simd_support": self._check_simd_support()
        })
        return info
    
    def _check_simd_support(self) -> Dict[str, bool]:
        """Detect CPU SIMD instruction support"""
        # In a real implementation, we would check for AVX, SSE, etc.
        # Here we'll just simulate some basic detection
        simd = {
            "sse": True,
            "sse2": True,
            "avx": True,
            "avx2": False,
            "avx512": False
        }
        return simd
    
    def _generate_field_chunk(self, 
                              start_row: int, 
                              end_row: int, 
                              width: int, 
                              height: int, 
                              frequency: float, 
                              time_factor: float) -> np.ndarray:
        """Generate a chunk of the quantum field (for multi-threading)"""
        # Scale the frequency to a more manageable number
        freq_factor = frequency / 1000.0 * PHI
        
        # Initialize the chunk
        chunk = np.zeros((end_row - start_row, width), dtype=np.float32)
        
        # Calculate the center of the field
        center_x = width / 2
        center_y = height / 2
        
        # Generate the field values for this chunk
        for y_rel in range(end_row - start_row):
            y = y_rel + start_row
            for x in range(width):
                # Calculate distance from center (normalized)
                dx = (x - center_x) / (width / 2)
                dy = (y - center_y) / (height / 2)
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Calculate the field value using phi-harmonics
                angle = math.atan2(dy, dx) * PHI
                time_value = time_factor * LAMBDA
                
                # Create an interference pattern
                value = (
                    math.sin(distance * freq_factor + time_value) * 
                    math.cos(angle * PHI) * 
                    math.exp(-distance / PHI)
                )
                
                chunk[y_rel, x] = value
        
        return chunk
    
    def generate_quantum_field(self, width: int, height: int, 
                              frequency_name: str = 'love', 
                              time_factor: float = 0,
                              depth: Optional[int] = None) -> np.ndarray:
        """
        Generate a quantum field using multiple CPU threads
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            depth: Optional depth for 3D fields
            
        Returns:
            A NumPy array representing the quantum field (2D or 3D)
        """
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Initialize the field
        field = np.zeros((height, width), dtype=np.float32)
        
        # If field is small or we only have one thread, generate directly
        if height * width < 10000 or self.num_threads == 1:
            return self._generate_field_chunk(0, height, width, height, frequency, time_factor)
        
        # Divide the field into chunks for multi-threading
        chunk_size = (height + self.num_threads - 1) // self.num_threads
        futures = []
        
        # Submit tasks to the thread pool
        for i in range(self.num_threads):
            start_row = i * chunk_size
            end_row = min(start_row + chunk_size, height)
            
            if start_row >= end_row:
                continue
                
            future = self.executor.submit(
                self._generate_field_chunk,
                start_row, end_row, width, height, frequency, time_factor
            )
            futures.append((start_row, future))
        
        # Collect results
        for start_row, future in futures:
            chunk = future.result()
            end_row = start_row + chunk.shape[0]
            field[start_row:end_row, :] = chunk
        
        return field

    def _calculate_coherence_chunk(self, 
                                  field_data: np.ndarray, 
                                  sample_points: List[Tuple[int, int]]) -> float:
        """Calculate coherence for a subset of sample points"""
        height, width = field_data.shape
        
        # Calculate alignment with phi
        alignments = []
        for x, y in sample_points:
            if 0 <= x < width and 0 <= y < height:
                value = field_data[y, x]
                nearest_phi_multiple = round(value / PHI)
                deviation = abs(value - (nearest_phi_multiple * PHI))
                alignment = 1.0 - min(1.0, deviation / (PHI * 0.1))
                alignments.append(alignment)
        
        # Return average alignment for this chunk
        if alignments:
            return sum(alignments) / len(alignments)
        return 0.0
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field
        
        Args:
            field_data: A NumPy array containing the field data (can be 1D, 2D, or 3D)
            
        Returns:
            A float representing the field coherence
        """
        if field_data.size == 0:
            return 0.0
        
        # Handle different dimensionality
        dims = field_data.shape
        if len(dims) == 1:
            width = dims[0]
            height = 1
        elif len(dims) == 2:
            height, width = dims
        elif len(dims) == 3:
            depth, height, width = dims
            # For 3D fields, flatten to 2D by taking the middle slice
            field_data = field_data[depth//2, :, :]
        else:
            # For higher dimensions, reshape to 2D
            field_data = field_data.reshape(-1, field_data.shape[-1])
        
        # Generate sample points
        np.random.seed(42)  # For reproducible results
        num_samples = min(400, width * height // 100)  # Cap at 400 samples
        sample_points = []
        
        for _ in range(num_samples):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            sample_points.append((x, y))
        
        # If field is small or we only have one thread, calculate directly
        if num_samples < 50 or self.num_threads == 1:
            coherence = self._calculate_coherence_chunk(field_data, sample_points)
            return coherence * PHI
        
        # Divide samples into chunks for multi-threading
        chunk_size = (num_samples + self.num_threads - 1) // self.num_threads
        futures = []
        
        # Submit tasks to the thread pool
        for i in range(self.num_threads):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            
            if start_idx >= end_idx:
                continue
                
            chunk_points = sample_points[start_idx:end_idx]
            future = self.executor.submit(
                self._calculate_coherence_chunk,
                field_data, chunk_points
            )
            futures.append(future)
        
        # Collect and combine results
        alignments = []
        for future in futures:
            alignments.append(future.result())
        
        # Calculate overall coherence
        coherence = sum(alignments) / len(alignments) if alignments else 0.0
        return coherence * PHI
    
    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a Phi-based sacred pattern
        
        Args:
            width: Width of the field
            height: Height of the field
            
        Returns:
            A 2D NumPy array representing the pattern field
        """
        field = np.zeros((height, width), dtype=np.float32)
        
        # Calculate the pattern values
        for y in range(height):
            for x in range(width):
                # Calculate normalized coordinates (-1 to 1)
                nx = 2 * (x / width - 0.5)
                ny = 2 * (y / height - 0.5)
                
                # Calculate radius and angle
                r = math.sqrt(nx*nx + ny*ny)
                a = math.atan2(ny, nx)
                
                # Create phi spiral pattern
                pattern_value = math.sin(PHI * r * 10) * math.cos(a * PHI * 5)
                field[y, x] = pattern_value
        
        return field
    
    def to_dlpack(self, field_data: np.ndarray):
        """
        Convert a field to DLPack format for interoperability with ML frameworks
        
        For CPU backend, we need numpy with DLPack support or to use another library.
        Since NumPy doesn't have native DLPack support, we'll provide a fallback
        using PyTorch if available.
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A DLPack tensor that can be imported into ML frameworks
        """
        try:
            # Try to use PyTorch as a bridge for DLPack
            import torch
            
            # Convert NumPy array to PyTorch tensor
            torch_tensor = torch.from_numpy(field_data.copy())
            
            # Convert to DLPack format
            dlpack_tensor = torch.utils.dlpack.to_dlpack(torch_tensor)
            return dlpack_tensor
        except ImportError:
            # If PyTorch is not available, try to use NumPy's __dlpack__ if available
            if hasattr(field_data, "__dlpack__"):
                return field_data.__dlpack__()
            else:
                raise RuntimeError("DLPack conversion not available: requires PyTorch or NumPy with DLPack support")
    
    def from_dlpack(self, dlpack_tensor, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert a DLPack tensor to a quantum field array
        
        Args:
            dlpack_tensor: A DLPack tensor
            shape: Optional shape to reshape the tensor to (height, width)
            
        Returns:
            A 2D NumPy array containing the field data
        """
        try:
            # Try to use PyTorch as a bridge for DLPack
            import torch
            
            # Convert DLPack to PyTorch tensor
            torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
            
            # Reshape if needed
            if shape is not None:
                torch_tensor = torch_tensor.reshape(shape)
            
            # Convert to NumPy
            return torch_tensor.numpy()
        except ImportError:
            # Try CuPy if available
            try:
                import cupy as cp
                cupy_array = cp.fromDlpack(dlpack_tensor)
                if shape is not None:
                    cupy_array = cupy_array.reshape(shape)
                return cp.asnumpy(cupy_array)
            except ImportError:
                raise RuntimeError("DLPack conversion not available: requires PyTorch or CuPy")
    
    def _generate_3d_field_chunk(self,
                              start_layer: int,
                              end_layer: int,
                              width: int,
                              height: int,
                              depth: int,
                              frequency: float,
                              time_factor: float) -> np.ndarray:
        """Generate a chunk of a 3D quantum field (for multi-threading)"""
        # Initialize the chunk
        chunk = np.zeros((end_layer - start_layer, height, width), dtype=np.float32)
        
        # Calculate the center of the field
        center_x = width / 2
        center_y = height / 2
        center_z = depth / 2
        
        # Generate the field values for this chunk
        for z_rel in range(end_layer - start_layer):
            z = z_rel + start_layer
            for y in range(height):
                for x in range(width):
                    # Calculate normalized coordinates
                    dx = (x - center_x) / (width / 2)
                    dy = (y - center_y) / (height / 2)
                    dz = (z - center_z) / (depth / 2)
                    
                    # Calculate distance from center
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz) * PHI
                    
                    # 3D angular components
                    theta = math.atan2(math.sqrt(dx*dx + dy*dy), dz)  # Polar angle
                    phi_angle = math.atan2(dy, dx)  # Azimuthal angle
                    
                    # Generate field with phi-harmonic wave equations
                    value = math.sin(distance * frequency * 0.01 + 
                                    theta * PHI + 
                                    phi_angle * PHI_PHI + 
                                    time_factor * PHI_PHI)
                    
                    # Apply phi-based dampening
                    dampening = math.exp(-distance * LAMBDA)
                    
                    # Combine wave and dampening
                    chunk[z_rel, y, x] = value * dampening
        
        return chunk
    
    def generate_3d_quantum_field(self, width: int, height: int, depth: int,
                                 frequency_name: str = 'love',
                                 time_factor: float = 0.0,
                                 custom_frequency: Optional[float] = None) -> np.ndarray:
        """
        Generate a 3D quantum field using multiple CPU threads
        
        Args:
            width: Width of the field in voxels
            height: Height of the field in voxels
            depth: Depth of the field in voxels
            frequency_name: Name of the sacred frequency to use
            time_factor: Time evolution factor (0.0 to 2π)
            custom_frequency: Custom frequency value (used if frequency_name is None)
            
        Returns:
            3D NumPy array containing the quantum field values
        """
        # Determine frequency
        if frequency_name is not None:
            if frequency_name not in SACRED_FREQUENCIES:
                raise ValueError(f"Unknown frequency name: {frequency_name}")
            frequency = SACRED_FREQUENCIES[frequency_name]
        elif custom_frequency is not None:
            frequency = custom_frequency
        else:
            raise ValueError("Either frequency_name or custom_frequency must be provided")
        
        # Initialize the field
        field = np.zeros((depth, height, width), dtype=np.float32)
        
        # If field is small or we only have one thread, generate directly
        if width * height * depth < 100000 or self.num_threads == 1:
            chunk = self._generate_3d_field_chunk(0, depth, width, height, depth, frequency, time_factor)
            field[:, :, :] = chunk
            return field
        
        # Divide the field into chunks for multi-threading (along the depth dimension)
        chunk_size = (depth + self.num_threads - 1) // self.num_threads
        futures = []
        
        # Submit tasks to the thread pool
        for i in range(self.num_threads):
            start_layer = i * chunk_size
            end_layer = min(start_layer + chunk_size, depth)
            
            if start_layer >= end_layer:
                continue
                
            future = self.executor.submit(
                self._generate_3d_field_chunk,
                start_layer, end_layer, width, height, depth, frequency, time_factor
            )
            futures.append((start_layer, future))
        
        # Collect results
        for start_layer, future in futures:
            chunk = future.result()
            end_layer = start_layer + chunk.shape[0]
            field[start_layer:end_layer, :, :] = chunk
        
        return field
    
    def calculate_3d_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a 3D quantum field
        
        Args:
            field_data: 3D NumPy array containing the field
            
        Returns:
            Coherence factor between 0.0 and 1.0
        """
        if field_data.ndim != 3:
            raise ValueError("Field data must be a 3D array")
        
        depth, height, width = field_data.shape
        
        # Calculate gradient in 3D
        grad_x, grad_y, grad_z = np.gradient(field_data)
        
        # Calculate gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Calculate vorticity (curl)
        curl_x = np.gradient(grad_z, axis=1) - np.gradient(grad_y, axis=2)
        curl_y = np.gradient(grad_x, axis=2) - np.gradient(grad_z, axis=0)
        curl_z = np.gradient(grad_y, axis=0) - np.gradient(grad_x, axis=1)
        
        # Calculate curl magnitude
        curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        
        # Calculate divergence
        div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1) + np.gradient(grad_z, axis=2)
        
        # Calculate field energy (squared amplitude)
        energy = field_data**2
        
        # Calculate coherence metrics based on field properties
        gradient_uniformity = 1.0 - np.std(grad_mag) / np.mean(grad_mag) if np.mean(grad_mag) > 0 else 0.0
        vorticity_factor = 1.0 - np.mean(curl_mag) / (np.mean(grad_mag) + 1e-10)
        divergence_factor = 1.0 - np.mean(np.abs(div)) / (np.mean(grad_mag) + 1e-10)
        phi_resonance = np.abs(np.corrcoef(
            energy.flatten(), 
            np.exp(-PHI * np.arange(energy.size) / energy.size)
        )[0, 1])
        
        # Combine metrics with phi-weighted formula
        coherence = (
            gradient_uniformity * 0.3 +
            vorticity_factor * 0.2 +
            divergence_factor * 0.2 +
            phi_resonance * 0.3
        )
        
        # Ensure result is in [0, 1] range
        coherence = max(0.0, min(1.0, coherence))
        
        return coherence
    
    def shutdown(self) -> None:
        """Release resources used by this backend"""
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        self.initialized = False