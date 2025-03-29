"""
Core functionality for quantum field visualization

This module provides the main API for quantum field generation and analysis.
It automatically selects the best available implementation (CUDA or CPU).
"""

import os
import sys
import math
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any, Sequence

# Import constants
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES


class QuantumField:
    """
    Represents a quantum field with phi-harmonic properties.
    
    This class encapsulates quantum field data and operations, providing
    a unified interface for field manipulation, regardless of the 
    underlying implementation (CPU, CUDA, Multi-GPU, etc.).
    """
    
    def __init__(self, data: np.ndarray, frequency_name: str = 'love'):
        """
        Initialize a quantum field.
        
        Args:
            data: NumPy array containing field data
            frequency_name: The sacred frequency associated with the field
        """
        self.data = data
        self.frequency_name = frequency_name
        self.creation_time = datetime.now()
        self.phi_resonance = PHI
        
        # Calculate field properties
        self._update_properties()
    
    def _update_properties(self):
        """Update field properties based on current data."""
        self.shape = self.data.shape
        self.dimensions = len(self.shape)
        self.coherence = get_coherence_metric(self.data)
    
    @property
    def size(self) -> int:
        """Get the total number of elements in the field."""
        return self.data.size
    
    def update(self, data: Optional[np.ndarray] = None, 
              apply_phi_transform: bool = False) -> None:
        """
        Update the field data.
        
        Args:
            data: New data to replace current field data
            apply_phi_transform: Whether to apply phi transform after update
        """
        if data is not None:
            self.data = data
        
        if apply_phi_transform:
            # Import here to avoid circular import
            from sacred_constants import phi_matrix_transform
            self.data = phi_matrix_transform(self.data)
        
        self._update_properties()
    
    def get_slice(self, axis: int = 0, index: Optional[int] = None) -> np.ndarray:
        """
        Get a slice of the field along specified axis.
        
        Args:
            axis: Axis along which to take the slice
            index: Index of the slice (defaults to middle of dimension)
            
        Returns:
            NumPy array representing the slice
        """
        if index is None:
            index = self.shape[axis] // 2
            
        slices = tuple(index if i == axis else slice(None) for i in range(self.dimensions))
        return self.data[slices]
    
    def apply_phi_modulation(self, intensity: float = 1.0) -> None:
        """
        Apply phi-based modulation to the field.
        
        Args:
            intensity: Intensity of the modulation (0.0-1.0)
        """
        # Get frequency from name
        frequency = SACRED_FREQUENCIES.get(self.frequency_name, 528)
        
        # Apply modulation based on field dimensionality
        if self.dimensions == 1:
            self._apply_phi_modulation_1d(frequency, intensity)
        elif self.dimensions == 2:
            self._apply_phi_modulation_2d(frequency, intensity)
        elif self.dimensions == 3:
            self._apply_phi_modulation_3d(frequency, intensity)
        else:
            self._apply_phi_modulation_nd(frequency, intensity)
        
        self._update_properties()
    
    def _apply_phi_modulation_1d(self, frequency: float, intensity: float) -> None:
        """Apply phi modulation to 1D field."""
        n = self.shape[0]
        x = np.linspace(-1, 1, n)
        
        # Create modulation pattern
        pattern = np.sin(frequency * PHI * x) * np.exp(-np.abs(x) / PHI)
        
        # Apply modulation
        self.data = self.data * (1.0 - intensity) + pattern * intensity
    
    def _apply_phi_modulation_2d(self, frequency: float, intensity: float) -> None:
        """Apply phi modulation to 2D field."""
        height, width = self.shape
        center_x = width / 2
        center_y = height / 2
        
        # Create coordinate grid
        y, x = np.ogrid[:height, :width]
        dx = (x - center_x) / center_x
        dy = (y - center_y) / center_y
        
        # Calculate radius and angle
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Create modulation pattern
        pattern = (
            np.sin(frequency * PHI * r / 1000) * 
            np.cos(theta * PHI) * 
            np.exp(-r / PHI)
        )
        
        # Apply modulation
        self.data = self.data * (1.0 - intensity) + pattern * intensity
    
    def _apply_phi_modulation_3d(self, frequency: float, intensity: float) -> None:
        """Apply phi modulation to 3D field."""
        depth, height, width = self.shape
        center_x = width / 2
        center_y = height / 2
        center_z = depth / 2
        
        # Create coordinate grid
        z, y, x = np.ogrid[:depth, :height, :width]
        dx = (x - center_x) / center_x
        dy = (y - center_y) / center_y
        dz = (z - center_z) / center_z
        
        # Calculate radius
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Create modulation pattern using 3D spherical harmonics approximation
        pattern = (
            np.sin(frequency * PHI * r / 1000) * 
            np.exp(-r / PHI)
        )
        
        # Apply modulation
        self.data = self.data * (1.0 - intensity) + pattern * intensity
    
    def _apply_phi_modulation_nd(self, frequency: float, intensity: float) -> None:
        """Apply phi modulation to N-dimensional field."""
        # For higher dimensions, apply FFT-based modulation
        # Use Fourier approach to modulate in frequency domain
        
        # Get FFT of the field
        fft_data = np.fft.fftn(self.data)
        
        # Create frequency domain modulation mask
        mask = np.ones_like(fft_data)
        
        # Enhance phi-resonant frequencies
        freq_factor = frequency / 1000.0 * PHI
        phi_bands = [PHI**i for i in range(5)]  # Geometric progression of phi powers
        
        # Define coordinates in frequency domain
        coords = [np.fft.fftfreq(dim) for dim in self.shape]
        grid = np.meshgrid(*coords, indexing='ij')
        
        # Calculate frequency domain distance from origin
        r_squared = sum(coord**2 for coord in grid)
        r = np.sqrt(r_squared)
        
        # Apply enhancement to phi-resonant frequencies
        for phi_band in phi_bands:
            band_center = freq_factor * phi_band
            band_width = band_center * LAMBDA / 5
            
            # Create Gaussian-like band around target frequency
            band_mask = np.exp(-((r - band_center) / band_width)**2)
            mask = mask + band_mask * intensity
        
        # Apply mask to frequency domain data
        modulated_fft = fft_data * mask
        
        # Convert back to spatial domain
        self.data = np.real(np.fft.ifftn(modulated_fft))


def create_quantum_field(dimensions: Union[Tuple[int, ...], List[int]], 
                        frequency_name: str = 'love',
                        initialization: str = 'phi-harmonic') -> QuantumField:
    """
    Create a new quantum field with specified dimensions.
    
    Args:
        dimensions: Tuple or list of dimensions (1D, 2D, or 3D)
        frequency_name: The sacred frequency to use
        initialization: Method to initialize the field
            
    Returns:
        A QuantumField object
    """
    # Check dimensions
    if not isinstance(dimensions, (tuple, list)) or len(dimensions) == 0:
        raise ValueError("Dimensions must be a non-empty tuple or list of integers")
    
    # Handle different dimensionality
    if len(dimensions) == 1:
        # 1D field
        width = dimensions[0]
        if initialization == 'zeros':
            data = np.zeros(width, dtype=np.float32)
        elif initialization == 'random':
            data = np.random.random(width).astype(np.float32) * 2 - 1
        else:  # phi-harmonic
            data = np.zeros(width, dtype=np.float32)
            x = np.linspace(-1, 1, width)
            frequency = SACRED_FREQUENCIES.get(frequency_name, 528) / 1000.0
            data = np.sin(frequency * PHI * x) * np.exp(-np.abs(x) / PHI)
    
    elif len(dimensions) == 2:
        # 2D field
        height, width = dimensions
        if initialization == 'zeros':
            data = np.zeros((height, width), dtype=np.float32)
        elif initialization == 'random':
            data = np.random.random((height, width)).astype(np.float32) * 2 - 1
        else:  # phi-harmonic
            data = generate_quantum_field(width, height, frequency_name)
    
    elif len(dimensions) == 3:
        # 3D field
        depth, height, width = dimensions
        if initialization == 'zeros':
            data = np.zeros((depth, height, width), dtype=np.float32)
        elif initialization == 'random':
            data = np.random.random((depth, height, width)).astype(np.float32) * 2 - 1
        else:  # phi-harmonic
            # Start with base 2D field
            base_field = generate_quantum_field(width, height, frequency_name)
            
            # Extend to 3D with phi-harmonic variation along depth
            data = np.zeros((depth, height, width), dtype=np.float32)
            frequency = SACRED_FREQUENCIES.get(frequency_name, 528) / 1000.0
            
            for z in range(depth):
                # Calculate depth factor with phi-based modulation
                z_norm = (z / depth - 0.5) * 2  # -1 to 1
                depth_factor = np.sin(z_norm * PHI * frequency * 10) * 0.5 + 0.5
                
                # Modulate base field
                data[z] = base_field * depth_factor
    
    else:
        # Higher dimensions - create with zeros and apply phi modulation
        data = np.zeros(dimensions, dtype=np.float32)
        field = QuantumField(data, frequency_name)
        field.apply_phi_modulation(intensity=0.9)
        return field
    
    # Create and return field
    field = QuantumField(data, frequency_name)
    return field


def get_coherence_metric(field_data: np.ndarray) -> float:
    """
    Calculate the coherence metric of a field.
    
    Args:
        field_data: A NumPy array containing the field data
        
    Returns:
        A float representing the field coherence (0.0-1.0)
    """
    return calculate_field_coherence(field_data)

# Check for CUDA availability
try:
    import quantum_cuda as qc
    CUDA_AVAILABLE = qc.CUDA_AVAILABLE
    
    # Initialize CUDA if available
    if CUDA_AVAILABLE:
        qc.initialize_cuda()
except ImportError:
    print("Warning: quantum_cuda module not found. Using CPU implementation.")
    CUDA_AVAILABLE = False
    qc = None

# Try importing multi-GPU support
try:
    from quantum_field.multi_gpu import (
        get_multi_gpu_manager, 
        generate_quantum_field_multi_gpu,
        calculate_field_coherence_multi_gpu
    )
    
    # Get the manager to check if multi-GPU is available
    multi_gpu_manager = get_multi_gpu_manager()
    MULTI_GPU_AVAILABLE = multi_gpu_manager.available and len(multi_gpu_manager.devices) > 1
except ImportError:
    MULTI_GPU_AVAILABLE = False

# Try importing thread block cluster support
try:
    from quantum_field.thread_block_cluster import (
        check_thread_block_cluster_support,
        initialize_thread_block_clusters,
        generate_quantum_field_tbc
    )
    
    # Check if thread block clusters are supported
    THREAD_BLOCK_CLUSTER_AVAILABLE = check_thread_block_cluster_support()
except ImportError:
    THREAD_BLOCK_CLUSTER_AVAILABLE = False

# Print available acceleration methods
print("Quantum Field Acceleration:")
print(f"- CUDA Available: {CUDA_AVAILABLE}")
print(f"- Multi-GPU Available: {MULTI_GPU_AVAILABLE}")
print(f"- Thread Block Clusters Available: {THREAD_BLOCK_CLUSTER_AVAILABLE}")


# CPU implementation for quantum field generation
def generate_quantum_field_cpu(width: int, height: int, frequency_name: str = 'love', time_factor: float = 0) -> np.ndarray:
    """
    CPU implementation for quantum field generation.
    
    Args:
        width: Width of the field
        height: Height of the field
        frequency_name: The sacred frequency to use
        time_factor: Time factor for animation
        
    Returns:
        A 2D NumPy array representing the quantum field
    """
    # Get the frequency value
    frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
    
    # Scale the frequency to a more manageable number
    freq_factor = frequency / 1000.0 * PHI
    
    # Initialize the field
    field = np.zeros((height, width), dtype=np.float32)
    
    # Calculate the center of the field
    center_x = width / 2
    center_y = height / 2
    
    # Generate the field values
    for y in range(height):
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
            
            field[y, x] = value
    
    return field


def generate_phi_pattern_cpu(width: int, height: int) -> np.ndarray:
    """
    CPU implementation for phi pattern generation.
    
    Args:
        width: Width of the field
        height: Height of the field
        
    Returns:
        A 2D NumPy array representing the pattern field
    """
    field = np.zeros((height, width), dtype=np.float32)
    
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


def calculate_field_coherence_cpu(field_data: np.ndarray) -> float:
    """
    CPU implementation for field coherence calculation.
    
    Args:
        field_data: A 2D NumPy array containing the field data
        
    Returns:
        A float representing the field coherence
    """
    # Coherence is related to alignment with phi harmonics
    if field_data.size == 0:
        return 0.0
        
    # Sample points for phi alignment
    height, width = field_data.shape
    sample_points = []
    np.random.seed(42)  # For reproducible results
    for _ in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        sample_points.append((x, y))
        
    # Calculate alignment with phi
    alignments = []
    for x, y in sample_points:
        value = field_data[y, x]
        nearest_phi_multiple = round(value / PHI)
        deviation = abs(value - (nearest_phi_multiple * PHI))
        alignment = 1.0 - min(1.0, deviation / (PHI * 0.1))
        alignments.append(alignment)
        
    coherence = np.mean(alignments) * PHI
    return coherence


# Public interface functions
def generate_quantum_field(width: int, height: int, frequency_name: str = 'love', time_factor: float = 0) -> np.ndarray:
    """
    Generate a quantum field with the best available implementation.
    
    Args:
        width: Width of the field
        height: Height of the field
        frequency_name: The sacred frequency to use
        time_factor: Time factor for animation
        
    Returns:
        A 2D NumPy array representing the quantum field
    """
    try:
        # Use the unified backend architecture if available
        from quantum_field.backends import get_backend
        backend = get_backend()
        return backend.generate_quantum_field(width, height, frequency_name, time_factor)
    except ImportError:
        # Fall back to legacy implementation if backends module is not available
        # Use thread block clusters if available (for very large fields)
        if THREAD_BLOCK_CLUSTER_AVAILABLE and width * height >= 1048576:  # 1024x1024 or larger
            try:
                return generate_quantum_field_tbc(width, height, frequency_name, time_factor)
            except Exception as e:
                print(f"Thread Block Cluster error: {e}")
                # Fall through to next method
        
        # Use multi-GPU if available (for large fields)
        if MULTI_GPU_AVAILABLE and width * height >= 262144:  # 512x512 or larger
            try:
                return generate_quantum_field_multi_gpu(width, height, frequency_name, time_factor)
            except Exception as e:
                print(f"Multi-GPU error: {e}")
                # Fall through to next method
        
        # Use single-GPU CUDA if available
        if CUDA_AVAILABLE and qc is not None:
            try:
                return qc.generate_quantum_field(width, height, frequency_name, time_factor)
            except Exception as e:
                print(f"CUDA error: {e}")
                # Fall through to CPU implementation
        
        # CPU implementation (fallback)
        return generate_quantum_field_cpu(width, height, frequency_name, time_factor)


def calculate_field_coherence(field_data: np.ndarray) -> float:
    """
    Calculate the coherence of a quantum field with the best available implementation.
    
    Args:
        field_data: A 2D NumPy array containing the field data
        
    Returns:
        A float representing the field coherence
    """
    try:
        # Use the unified backend architecture if available
        from quantum_field.backends import get_backend
        backend = get_backend()
        return backend.calculate_field_coherence(field_data)
    except ImportError:
        # Fall back to legacy implementation if backends module is not available
        # For large fields, try multi-GPU
        height, width = field_data.shape
        if MULTI_GPU_AVAILABLE and width * height >= 262144:  # 512x512 or larger
            try:
                return calculate_field_coherence_multi_gpu(field_data)
            except Exception as e:
                print(f"Multi-GPU coherence error: {e}")
                # Fall through to next method
        
        # Use single-GPU CUDA if available
        if CUDA_AVAILABLE and qc is not None:
            try:
                return qc.calculate_field_coherence(field_data)
            except Exception as e:
                print(f"CUDA coherence error: {e}")
                # Fall through to CPU implementation
        
        # CPU implementation (fallback)
        return calculate_field_coherence_cpu(field_data)


def generate_phi_pattern(width: int, height: int) -> np.ndarray:
    """
    Generate a Phi-based sacred pattern with the best available implementation.
    
    Args:
        width: Width of the field
        height: Height of the field
        
    Returns:
        A 2D NumPy array representing the pattern field
    """
    try:
        # Use the unified backend architecture if available
        from quantum_field.backends import get_backend
        backend = get_backend()
        return backend.generate_phi_pattern(width, height)
    except ImportError:
        # Fall back to legacy implementation if backends module is not available
        if CUDA_AVAILABLE and qc is not None:
            try:
                if hasattr(qc, 'generate_phi_pattern_cuda'):
                    return qc.generate_phi_pattern_cuda(width, height)
                elif hasattr(qc, 'generate_phi_pattern'):
                    return qc.generate_phi_pattern(width, height)
            except Exception as e:
                print(f"CUDA phi pattern error: {e}")
                # Fall through to CPU implementation
        
        # CPU implementation (fallback)
        return generate_phi_pattern_cpu(width, height)


def field_to_ascii(field: np.ndarray, chars: str = ' .-+*#@') -> List[str]:
    """
    Convert a quantum field to ASCII art.
    
    Args:
        field: 2D NumPy array of field values
        chars: Characters to use for visualization
        
    Returns:
        A list of strings representing the ASCII art
    """
    # Find min and max values for normalization
    min_val = field.min()
    max_val = field.max()
    
    # Normalize and convert to ASCII
    ascii_art = []
    for row in field:
        ascii_row = ''
        for value in row:
            # Normalize to 0-1
            if max_val > min_val:
                norm_value = (value - min_val) / (max_val - min_val)
            else:
                norm_value = 0.5
            
            # Convert to character
            char_index = int(norm_value * (len(chars) - 1))
            ascii_row += chars[char_index]
        
        ascii_art.append(ascii_row)
    
    return ascii_art


def print_field(ascii_art: List[str], title: str = "Quantum Field Visualization") -> None:
    """
    Print the ASCII art field with a title.
    
    Args:
        ascii_art: List of strings representing the ASCII art
        title: Title to display above the field
    """
    print("\n" + "=" * 80)
    print(f"{title} - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    for row in ascii_art:
        print(row)
    
    print("=" * 80)


def display_phi_pattern(width: int = 40, height: int = 20) -> None:
    """
    Generate and display a Phi-based sacred pattern.
    
    Args:
        width: Width of the pattern
        height: Height of the pattern
    """
    # Generate the pattern field
    field = generate_phi_pattern(width, height)
    
    # Convert to ASCII
    pattern = []
    for y in range(height):
        row = ""
        for x in range(width):
            # Map to characters
            value = field[y, x]
            if value > 0.7:
                row += "#"
            elif value > 0.3:
                row += "*"
            elif value > 0:
                row += "+"
            elif value > -0.3:
                row += "-"
            elif value > -0.7:
                row += "."
            else:
                row += " "
        
        pattern.append(row)
    
    print("\n" + "=" * 80)
    print("PHI SACRED PATTERN")
    print("=" * 80)
    
    for row in pattern:
        print(row)
        
    print("=" * 80)


def animate_field(width: int, height: int, frames: int = 10, delay: float = 0.2, frequency_name: str = 'love') -> None:
    """
    Animate a quantum field visualization.
    
    Args:
        width: Width of the field
        height: Height of the field
        frames: Number of frames to generate
        delay: Delay between frames
        frequency_name: The sacred frequency to use
    """
    for i in range(frames):
        # Generate a new field with a time factor
        field = generate_quantum_field(width, height, frequency_name, i * 0.2)
        
        # Convert to ASCII and print
        ascii_art = field_to_ascii(field)
        
        # Clear screen (might not work in all terminals/environments)
        print("\033c", end="")
        
        # Print the field
        print_field(ascii_art, f"Quantum Field - {frequency_name.capitalize()} Frequency")
        
        # Wait before the next frame
        time.sleep(delay)


def benchmark_performance(width: int, height: int, iterations: int = 10) -> Dict[str, Any]:
    """
    Benchmark different implementation methods.
    
    Args:
        width: Width of the field
        height: Height of the field
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "size": f"{width}x{height}",
        "methods": [],
        "times": [],
        "speedups": []
    }
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Always benchmark CPU implementation as baseline
    results["methods"].append("CPU")
    start_time = time.time()
    for i in range(iterations):
        generate_quantum_field_cpu(width, height, 'love', i*0.1)
    cpu_time = time.time() - start_time
    results["times"].append(cpu_time)
    results["speedups"].append(1.0)  # CPU is baseline
    
    print(f"CPU Implementation: {cpu_time:.4f} seconds for {iterations} iterations")
    print(f"CPU Average per iteration: {cpu_time/iterations:.4f} seconds")
    
    # Benchmark CUDA if available
    if CUDA_AVAILABLE and qc is not None:
        results["methods"].append("CUDA")
        start_time = time.time()
        for i in range(iterations):
            qc.generate_quantum_field(width, height, 'love', i*0.1)
        cuda_time = time.time() - start_time
        results["times"].append(cuda_time)
        
        # Calculate speedup
        speedup = cpu_time / cuda_time if cuda_time > 0 else 0
        results["speedups"].append(speedup)
        
        print(f"CUDA Implementation: {cuda_time:.4f} seconds for {iterations} iterations")
        print(f"CUDA Average per iteration: {cuda_time/iterations:.4f} seconds")
        print(f"CUDA Speedup: {speedup:.2f}x")
    
    # Benchmark Thread Block Clusters if available
    if THREAD_BLOCK_CLUSTER_AVAILABLE:
        results["methods"].append("Thread Block Clusters")
        start_time = time.time()
        for i in range(iterations):
            generate_quantum_field_tbc(width, height, 'love', i*0.1)
        tbc_time = time.time() - start_time
        results["times"].append(tbc_time)
        
        # Calculate speedup
        speedup = cpu_time / tbc_time if tbc_time > 0 else 0
        results["speedups"].append(speedup)
        
        print(f"Thread Block Cluster Implementation: {tbc_time:.4f} seconds for {iterations} iterations")
        print(f"Thread Block Cluster Average per iteration: {tbc_time/iterations:.4f} seconds")
        print(f"Thread Block Cluster Speedup: {speedup:.2f}x")
    
    # Benchmark Multi-GPU if available
    if MULTI_GPU_AVAILABLE:
        results["methods"].append("Multi-GPU")
        start_time = time.time()
        for i in range(iterations):
            generate_quantum_field_multi_gpu(width, height, 'love', i*0.1)
        multi_gpu_time = time.time() - start_time
        results["times"].append(multi_gpu_time)
        
        # Calculate speedup
        speedup = cpu_time / multi_gpu_time if multi_gpu_time > 0 else 0
        results["speedups"].append(speedup)
        
        print(f"Multi-GPU Implementation: {multi_gpu_time:.4f} seconds for {iterations} iterations")
        print(f"Multi-GPU Average per iteration: {multi_gpu_time/iterations:.4f} seconds")
        print(f"Multi-GPU Speedup: {speedup:.2f}x")
    
    print("=" * 80)
    return results