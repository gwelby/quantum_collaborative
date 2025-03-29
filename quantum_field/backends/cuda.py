"""
CUDA Backend for Quantum Field Generation

This module provides GPU acceleration via NVIDIA CUDA.
It includes support for advanced features like Multi-GPU, Thread Block Clusters,
and CUDA Graphs for optimizing repetitive workflows.
"""

import os
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from quantum_field.backends import AcceleratorBackend
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# Try to import CUDA modules with fallback
try:
    from cuda.core.experimental import Device, Stream, Program, ProgramOptions
    from cuda.core.experimental import LaunchConfig, launch
    from cuda.core.experimental import Graph, GraphExec, GraphNode
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class CUDABackend(AcceleratorBackend):
    """
    CUDA implementation of quantum field operations for NVIDIA GPUs
    
    Supports advanced features like Multi-GPU, Thread Block Clusters on H100+,
    and CUDA Graphs for optimizing repetitive workflows.
    """
    
    name = "cuda"
    priority = 90  # High priority, but not the highest
    
    def __init__(self):
        super().__init__()
        self.devices = []
        self.streams = []
        self.modules = {}
        self.tbc_module = None
        self.current_device = None
        
        # Graphs support
        self.graphs = {}  # Dictionary to store created graphs
        self.graph_execs = {}  # Dictionary to store executable graphs
        self.cuda_graphs_available = False
        
        # Check for advanced features
        self.multi_gpu_available = False
        self.thread_block_clusters_available = False
        self.has_3d_capability = False
        
    def initialize(self) -> bool:
        """Initialize the CUDA backend"""
        if not CUDA_AVAILABLE:
            return False
        
        try:
            # Get available devices
            device_count = Device.get_device_count()
            if device_count == 0:
                print("No CUDA devices available")
                return False
            
            # Initialize each device
            for i in range(device_count):
                device = Device(i)
                device.set_current()
                
                # Get device information
                device_info = {
                    "id": i,
                    "name": device.get_name(),
                    "compute_capability": device.compute_capability,
                    "total_memory": device.total_memory
                }
                
                # Create a stream for this device
                stream = device.create_stream()
                
                self.devices.append(device_info)
                self.streams.append(stream)
                
                # Compile kernels for this device
                self._compile_device_kernels(i, device_info)
            
            # Set current device to the first one
            if self.devices:
                Device(0).set_current()
                self.current_device = 0
                
            # Set capabilities based on available hardware
            self.multi_gpu_available = len(self.devices) > 1
            
            # Thread Block Clusters require Compute Capability 9.0+ (H100+)
            if self.devices and self.devices[0]["compute_capability"][0] >= 9:
                self.thread_block_clusters_available = self._compile_tbc_kernels()
            
            # Check for CUDA Graphs support (requires Compute Capability 6.0+)
            self.cuda_graphs_available = any(d["compute_capability"][0] >= 6 for d in self.devices)
            
            # Update capabilities
            self.capabilities = {
                "thread_block_clusters": self.thread_block_clusters_available,
                "multi_device": self.multi_gpu_available,
                "async_execution": True,
                "tensor_cores": any(d["compute_capability"][0] >= 7 for d in self.devices),
                "half_precision": True,
                "dlpack_support": True,
                "cuda_graphs": self.cuda_graphs_available,
                "3d_fields": True,
            }
            
            # Enable 3D capability
            self.has_3d_capability = True
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing CUDA backend: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if CUDA is available"""
        if not CUDA_AVAILABLE:
            return False
        
        try:
            device_count = Device.get_device_count()
            return device_count > 0
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the CUDA backend"""
        info = super().get_info()
        
        if self.devices:
            info.update({
                "devices": self.devices,
                "device_count": len(self.devices),
                "current_device": self.current_device,
                "cuda_version": self._get_cuda_version(),
                "multi_gpu_available": self.multi_gpu_available,
                "thread_block_clusters_available": self.thread_block_clusters_available,
                "cuda_graphs_available": self.cuda_graphs_available,
                "active_graphs": len(self.graphs),
                "cupy_available": self._is_cupy_available()
            })
        
        return info
    
    def _get_cuda_version(self) -> str:
        """Get the CUDA runtime version"""
        try:
            if hasattr(cp, 'cuda') and hasattr(cp.cuda, 'runtime'):
                version = cp.cuda.runtime.runtimeGetVersion()
                major = version // 1000
                minor = (version % 1000) // 10
                return f"{major}.{minor}"
        except:
            pass
        return "Unknown"
    
    def _is_cupy_available(self) -> bool:
        """Check if CuPy is available"""
        return 'cp' in globals() and cp is not None
    
    def _compile_device_kernels(self, device_id: int, device_info: Dict[str, Any]) -> None:
        """Compile standard kernels for a device"""
        try:
            # Get device compute capability
            cc_major, cc_minor = device_info["compute_capability"]
            arch_str = f"{cc_major}{cc_minor}"
            
            # Define kernel source
            kernel_source = """
            extern "C" __global__ void generate_quantum_field(
                float *field, int width, int height, float frequency, float phi, float lambda, float time_factor
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    // Calculate center coordinates
                    float center_x = width / 2.0f;
                    float center_y = height / 2.0f;
                    
                    // Calculate normalized coordinates
                    float dx = (idx - center_x) / (width / 2.0f);
                    float dy = (idy - center_y) / (height / 2.0f);
                    float distance = sqrtf(dx*dx + dy*dy);
                    
                    // Calculate field value using phi-harmonics
                    float angle = atan2f(dy, dx) * phi;
                    float time_value = time_factor * lambda;
                    float freq_factor = frequency / 1000.0f * phi;
                    
                    // Create interference pattern
                    float value = sinf(distance * freq_factor + time_value) * 
                                 cosf(angle * phi) * 
                                 expf(-distance / phi);
                    
                    // Store the result
                    field[idy * width + idx] = value;
                }
            }
            
            extern "C" __global__ void generate_3d_quantum_field(
                float *field, int width, int height, int depth, float frequency, float phi, float lambda, float time_factor
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                int idz = blockIdx.z * blockDim.z + threadIdx.z;
                
                if (idx < width && idy < height && idz < depth) {
                    // Calculate center coordinates
                    float center_x = width / 2.0f;
                    float center_y = height / 2.0f;
                    float center_z = depth / 2.0f;
                    
                    // Calculate normalized coordinates
                    float dx = (idx - center_x) / (width / 2.0f);
                    float dy = (idy - center_y) / (height / 2.0f);
                    float dz = (idz - center_z) / (depth / 2.0f);
                    
                    // Calculate distance from center
                    float distance = sqrtf(dx*dx + dy*dy + dz*dz) * phi;
                    
                    // 3D angular components
                    float theta = atan2f(sqrtf(dx*dx + dy*dy), dz);  // Polar angle
                    float phi_angle = atan2f(dy, dx);  // Azimuthal angle
                    
                    // Calculate field value using 3D phi-harmonics
                    float freq_factor = frequency * 0.01f;
                    
                    // Generate field with phi-harmonic wave equations
                    float value = sinf(distance * freq_factor + 
                                      theta * phi + 
                                      phi_angle * lambda + 
                                      time_factor * lambda);
                    
                    // Apply phi-based dampening
                    float dampening = expf(-distance * lambda);
                    
                    // Combine wave and dampening
                    value = value * dampening;
                    
                    // Store the result (3D indexing: x + y*width + z*width*height)
                    field[idx + idy * width + idz * width * height] = value;
                }
            }
            
            extern "C" __global__ void calculate_field_coherence(
                float *field, int width, int height, float phi, float *result, int *count
            ) {
                // Use shared memory for reduction
                extern __shared__ float shared_mem[];
                float *alignment_sum = shared_mem;
                int *counter = (int*)(alignment_sum + blockDim.x);
                
                int tid = threadIdx.x;
                if (tid == 0) {
                    counter[0] = 0;
                }
                __syncthreads();
                
                alignment_sum[tid] = 0.0f;
                
                // Each thread samples some random points
                for (int i = 0; i < 4; i++) {
                    // Use a simple hash function to generate "random" coordinates
                    int hash = (blockIdx.x * blockDim.x + tid) * 1664525 + 1013904223 + i * 22695477;
                    int x = hash % width;
                    int y = (hash / width) % height;
                    
                    float value = field[y * width + x];
                    float nearest_phi_multiple = roundf(value / phi);
                    float deviation = fabsf(value - (nearest_phi_multiple * phi));
                    float alignment = 1.0f - fminf(1.0f, deviation / (phi * 0.1f));
                    alignment_sum[tid] += alignment;
                    
                    atomicAdd(&counter[0], 1);
                }
                
                __syncthreads();
                
                // Parallel reduction to sum alignments
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        alignment_sum[tid] += alignment_sum[tid + s];
                    }
                    __syncthreads();
                }
                
                // First thread writes the result
                if (tid == 0) {
                    result[blockIdx.x] = alignment_sum[0];
                    count[blockIdx.x] = counter[0];
                }
            }
            
            extern "C" __global__ void calculate_3d_field_coherence(
                float *field, int width, int height, int depth, float phi, float lambda, float *result, int *count
            ) {
                // Use shared memory for reduction
                extern __shared__ float shared_mem[];
                float *alignment_sum = shared_mem;
                int *counter = (int*)(alignment_sum + blockDim.x);
                
                int tid = threadIdx.x;
                if (tid == 0) {
                    counter[0] = 0;
                }
                __syncthreads();
                
                alignment_sum[tid] = 0.0f;
                
                // Each thread samples some random points in 3D space
                for (int i = 0; i < 8; i++) {  // More samples for 3D
                    // Use a simple hash function to generate "random" coordinates
                    int hash = (blockIdx.x * blockDim.x + tid) * 1664525 + 1013904223 + i * 22695477;
                    int x = hash % width;
                    int y = (hash / width) % height;
                    int z = (hash / (width * height)) % depth;
                    
                    // Get field value at this point
                    float value = field[x + y * width + z * width * height];
                    
                    // Calculate phi-based coherence metrics
                    float nearest_phi_multiple = roundf(value / phi);
                    float deviation = fabsf(value - (nearest_phi_multiple * phi));
                    float alignment = 1.0f - fminf(1.0f, deviation / (phi * 0.1f));
                    
                    // Calculate gradient around this point
                    float gradient_sum = 0.0f;
                    float curl_sum = 0.0f;
                    
                    // We can't compute full gradient/curl without more complex sampling,
                    // but we can estimate it by sampling neighboring points
                    if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && z > 0 && z < depth - 1) {
                        // Calculate simple gradient approximation in all directions
                        float dx = (field[(x+1) + y*width + z*width*height] - field[(x-1) + y*width + z*width*height]) / 2.0f;
                        float dy = (field[x + (y+1)*width + z*width*height] - field[x + (y-1)*width + z*width*height]) / 2.0f;
                        float dz = (field[x + y*width + (z+1)*width*height] - field[x + y*width + (z-1)*width*height]) / 2.0f;
                        
                        // Gradient magnitude
                        gradient_sum = sqrtf(dx*dx + dy*dy + dz*dz);
                        
                        // Crude curl estimation (partial)
                        float curl_x = dy - dz;
                        float curl_y = dz - dx;
                        float curl_z = dx - dy;
                        curl_sum = sqrtf(curl_x*curl_x + curl_y*curl_y + curl_z*curl_z);
                    }
                    
                    // Combine metrics with phi-weighted formula
                    float coherence_contribution = alignment * 0.5f + 
                                                 (1.0f - fminf(1.0f, gradient_sum)) * 0.25f + 
                                                 (1.0f - fminf(1.0f, curl_sum)) * 0.25f;
                    
                    alignment_sum[tid] += coherence_contribution;
                    atomicAdd(&counter[0], 1);
                }
                
                __syncthreads();
                
                // Parallel reduction to sum coherence values
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        alignment_sum[tid] += alignment_sum[tid + s];
                    }
                    __syncthreads();
                }
                
                // First thread writes the result
                if (tid == 0) {
                    result[blockIdx.x] = alignment_sum[0];
                    count[blockIdx.x] = counter[0];
                }
            }
            
            extern "C" __global__ void generate_phi_pattern(
                float *pattern, int width, int height, float phi
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    // Calculate normalized coordinates (-1 to 1)
                    float nx = 2.0f * (float(idx) / width - 0.5f);
                    float ny = 2.0f * (float(idy) / height - 0.5f);
                    
                    // Calculate radius and angle
                    float r = sqrtf(nx*nx + ny*ny);
                    float a = atan2f(ny, nx);
                    
                    // Create phi spiral pattern
                    float pattern_value = sinf(phi * r * 10.0f) * cosf(a * phi * 5.0f);
                    pattern[idy * width + idx] = pattern_value;
                }
            }
            """
            
            # Compile program
            program_options = ProgramOptions(
                std="c++17", 
                arch=f"sm_{arch_str}",
                flags=["--use_fast_math"]
            )
            program = Program(kernel_source, code_type="c++", options=program_options)
            
            # Compile module
            module = program.compile(
                "cubin", 
                name_expressions=[
                    "generate_quantum_field",
                    "generate_3d_quantum_field",
                    "calculate_field_coherence",
                    "calculate_3d_field_coherence",
                    "generate_phi_pattern"
                ]
            )
            
            self.modules[device_id] = module
        except Exception as e:
            print(f"Error compiling kernels for device {device_id}: {e}")
    
    def _compile_tbc_kernels(self) -> bool:
        """Compile Thread Block Cluster kernels"""
        if not self.devices or self.devices[0]["compute_capability"][0] < 9:
            return False
        
        try:
            # Get device compute capability
            cc_major, cc_minor = self.devices[0]["compute_capability"]
            arch_str = f"{cc_major}{cc_minor}"
            
            # Define kernel source with Thread Block Cluster support
            kernel_source = """
            extern "C" __global__ void generate_quantum_field_tbc(
                float *field, int width, int height, float frequency, float phi, float lambda, float time_factor
            ) {
                // Get block cluster ranks
                unsigned int cluster_rank_x = cooperative_groups::block_rank_in_cluster_x();
                unsigned int cluster_rank_y = cooperative_groups::block_rank_in_cluster_y();
                
                // Calculate block offset within the cluster
                unsigned int block_offset_x = cluster_rank_x * blockDim.x;
                unsigned int block_offset_y = cluster_rank_y * blockDim.y;
                
                // Calculate global thread position
                int idx = blockIdx.x * (blockDim.x * 2) + block_offset_x + threadIdx.x;
                int idy = blockIdx.y * (blockDim.y * 2) + block_offset_y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    // Calculate center coordinates
                    float center_x = width / 2.0f;
                    float center_y = height / 2.0f;
                    
                    // Calculate normalized coordinates
                    float dx = (idx - center_x) / (width / 2.0f);
                    float dy = (idy - center_y) / (height / 2.0f);
                    float distance = sqrtf(dx*dx + dy*dy);
                    
                    // Calculate field value using phi-harmonics
                    float angle = atan2f(dy, dx) * phi;
                    float time_value = time_factor * lambda;
                    float freq_factor = frequency / 1000.0f * phi;
                    
                    // Create interference pattern
                    float value = sinf(distance * freq_factor + time_value) * 
                                cosf(angle * phi) * 
                                expf(-distance / phi);
                    
                    // Store the result
                    field[idy * width + idx] = value;
                }
            }
            
            extern "C" __global__ void generate_3d_quantum_field_tbc(
                float *field, int width, int height, int depth, float frequency, float phi, float lambda, float time_factor
            ) {
                // Get block cluster ranks
                unsigned int cluster_rank_x = cooperative_groups::block_rank_in_cluster_x();
                unsigned int cluster_rank_y = cooperative_groups::block_rank_in_cluster_y();
                unsigned int cluster_rank_z = cooperative_groups::block_rank_in_cluster_z();
                
                // Calculate block offset within the cluster
                unsigned int block_offset_x = cluster_rank_x * blockDim.x;
                unsigned int block_offset_y = cluster_rank_y * blockDim.y;
                unsigned int block_offset_z = cluster_rank_z * blockDim.z;
                
                // Calculate global thread position
                int idx = blockIdx.x * (blockDim.x * 2) + block_offset_x + threadIdx.x;
                int idy = blockIdx.y * (blockDim.y * 2) + block_offset_y + threadIdx.y;
                int idz = blockIdx.z * (blockDim.z * 2) + block_offset_z + threadIdx.z;
                
                if (idx < width && idy < height && idz < depth) {
                    // Calculate center coordinates
                    float center_x = width / 2.0f;
                    float center_y = height / 2.0f;
                    float center_z = depth / 2.0f;
                    
                    // Calculate normalized coordinates
                    float dx = (idx - center_x) / (width / 2.0f);
                    float dy = (idy - center_y) / (height / 2.0f);
                    float dz = (idz - center_z) / (depth / 2.0f);
                    
                    // Calculate distance from center
                    float distance = sqrtf(dx*dx + dy*dy + dz*dz) * phi;
                    
                    // 3D angular components
                    float theta = atan2f(sqrtf(dx*dx + dy*dy), dz);  // Polar angle
                    float phi_angle = atan2f(dy, dx);  // Azimuthal angle
                    
                    // Calculate field value using 3D phi-harmonics
                    float freq_factor = frequency * 0.01f;
                    
                    // Generate field with phi-harmonic wave equations
                    float value = sinf(distance * freq_factor + 
                                      theta * phi + 
                                      phi_angle * lambda + 
                                      time_factor * lambda);
                    
                    // Apply phi-based dampening
                    float dampening = expf(-distance * lambda);
                    
                    // Combine wave and dampening
                    value = value * dampening;
                    
                    // Store the result (3D indexing: x + y*width + z*width*height)
                    field[idx + idy * width + idz * width * height] = value;
                }
            }
            """
            
            # Compile program with thread block cluster support
            program_options = ProgramOptions(
                std="c++17", 
                arch=f"sm_{arch_str}",
                flags=["--extended-lambda", "--use_fast_math", "--threads-per-thread-block=1024"]
            )
            program = Program(kernel_source, code_type="c++", options=program_options)
            
            # Compile module
            self.tbc_module = program.compile(
                "cubin", 
                name_expressions=[
                    "generate_quantum_field_tbc",
                    "generate_3d_quantum_field_tbc"
                ]
            )
            
            return self.tbc_module is not None
        except Exception as e:
            print(f"Error compiling Thread Block Cluster kernels: {e}")
            return False
    
    def _set_device(self, device_id: int) -> bool:
        """Set the current device"""
        if device_id < 0 or device_id >= len(self.devices):
            return False
        
        try:
            Device(device_id).set_current()
            self.current_device = device_id
            return True
        except Exception as e:
            print(f"Error setting device {device_id}: {e}")
            return False
    
    def generate_quantum_field(self, width: int, height: int, 
                              frequency_name: str = 'love', 
                              time_factor: float = 0) -> np.ndarray:
        """
        Generate a quantum field using CUDA
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        if not self.initialized or not self.modules:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_quantum_field(width, height, frequency_name, time_factor)
        
        # Determine the best implementation based on field size and hardware
        field_size = width * height
        
        # Use Thread Block Clusters for very large fields (if available)
        if self.thread_block_clusters_available and field_size >= 1048576:  # 1024x1024 or larger
            try:
                return self._generate_field_tbc(width, height, frequency_name, time_factor)
            except Exception as e:
                print(f"Thread Block Cluster error: {e}")
                # Fall through to next method
        
        # Use Multi-GPU for large fields (if available)
        if self.multi_gpu_available and field_size >= 262144:  # 512x512 or larger
            try:
                return self._generate_field_multi_gpu(width, height, frequency_name, time_factor)
            except Exception as e:
                print(f"Multi-GPU error: {e}")
                # Fall through to next method
        
        # Standard CUDA implementation
        try:
            return self._generate_field_single_gpu(width, height, frequency_name, time_factor)
        except Exception as e:
            print(f"CUDA error: {e}")
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_quantum_field(width, height, frequency_name, time_factor)
    
    def _generate_field_single_gpu(self, width: int, height: int, 
                                  frequency_name: str = 'love', 
                                  time_factor: float = 0) -> np.ndarray:
        """Generate a quantum field using a single GPU"""
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Set the first device as current
        self._set_device(0)
        
        # Create a CuPy array for output
        output = cp.empty((height, width), dtype=cp.float32)
        
        # Set up grid and block dimensions
        block_dim = (16, 16, 1)
        grid_dim = (
            (width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1],
            1
        )
        
        # Create launch config
        config = LaunchConfig(grid=grid_dim, block=block_dim)
        
        # Get the kernel
        kernel = self.modules[0].get_kernel("generate_quantum_field")
        
        # Launch the kernel
        launch(
            self.streams[0], 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height, 
            frequency, 
            PHI, 
            LAMBDA, 
            time_factor
        )
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy
        return cp.asnumpy(output)
    
    def _generate_field_multi_gpu(self, width: int, height: int, 
                                 frequency_name: str = 'love', 
                                 time_factor: float = 0) -> np.ndarray:
        """Generate a quantum field distributed across multiple GPUs"""
        if not self.multi_gpu_available:
            return self._generate_field_single_gpu(width, height, frequency_name, time_factor)
        
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Create output array
        output = np.zeros((height, width), dtype=np.float32)
        
        # Divide the field among GPUs (row-wise partitioning)
        num_gpus = len(self.devices)
        rows_per_gpu = (height + num_gpus - 1) // num_gpus
        
        # Set block dimensions for each GPU
        block_dim = (16, 16, 1)
        
        # List to store output arrays from each GPU
        gpu_outputs = []
        
        # Launch kernels on each GPU
        for i in range(num_gpus):
            # Calculate the rows for this GPU
            start_row = i * rows_per_gpu
            end_row = min(start_row + rows_per_gpu, height)
            shard_height = end_row - start_row
            
            if shard_height <= 0:
                continue  # Skip if this GPU has no work
            
            # Set this device as current
            self._set_device(i)
            
            # Create a CuPy array for this GPU's output
            with cp.cuda.Device(i):
                gpu_output = cp.empty((shard_height, width), dtype=cp.float32)
                
                # Set up grid dimensions
                grid_dim = (
                    (width + block_dim[0] - 1) // block_dim[0],
                    (shard_height + block_dim[1] - 1) // block_dim[1],
                    1
                )
                
                # Create launch config
                config = LaunchConfig(grid=grid_dim, block=block_dim)
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("generate_quantum_field")
                
                # Launch the kernel with modified parameters to process just this shard
                # We need to pass the full height for center calculation, but adjust the loop bounds
                launch(
                    self.streams[i], 
                    config, 
                    kernel, 
                    gpu_output.data.ptr, 
                    width, 
                    height,  # Full height for center calculation
                    frequency, 
                    PHI, 
                    LAMBDA, 
                    time_factor
                )
                
                # Add to list for later collection
                gpu_outputs.append((i, start_row, end_row, gpu_output))
        
        # Synchronize all streams
        for stream in self.streams:
            stream.sync()
        
        # Collect results from all GPUs
        for i, start_row, end_row, gpu_output in gpu_outputs:
            with cp.cuda.Device(i):
                # Copy this shard to the output array
                shard = cp.asnumpy(gpu_output)
                output[start_row:end_row, :] = shard
        
        return output
    
    def _generate_field_tbc(self, width: int, height: int, 
                           frequency_name: str = 'love', 
                           time_factor: float = 0) -> np.ndarray:
        """Generate a quantum field using Thread Block Clusters"""
        if not self.thread_block_clusters_available or self.tbc_module is None:
            return self._generate_field_single_gpu(width, height, frequency_name, time_factor)
        
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Set the first device as current
        self._set_device(0)
        
        # Create a CuPy array for output
        output = cp.empty((height, width), dtype=cp.float32)
        
        # Set up cluster and block dimensions (2x2 cluster of 16x16 blocks)
        block_dim = (16, 16, 1)
        cluster_shape = (2, 2, 1)  # 2x2 blocks per cluster
        grid_dim = (
            (width + block_dim[0] * cluster_shape[0] - 1) // (block_dim[0] * cluster_shape[0]),
            (height + block_dim[1] * cluster_shape[1] - 1) // (block_dim[1] * cluster_shape[1]),
            1
        )
        
        # Create thread block cluster launch config
        config = LaunchConfig(
            grid=grid_dim, 
            block=block_dim,
            cluster_shape=cluster_shape  # 2x2 blocks per cluster
        )
        
        # Get the kernel
        kernel = self.tbc_module.get_kernel("generate_quantum_field_tbc")
        
        # Launch the kernel with thread block clusters
        launch(
            self.streams[0], 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height, 
            frequency, 
            PHI, 
            LAMBDA, 
            time_factor
        )
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy
        return cp.asnumpy(output)
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A float representing the field coherence
        """
        if not self.initialized or not self.modules:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.calculate_field_coherence(field_data)
        
        # For small fields, use CPU implementation (less overhead)
        height, width = field_data.shape
        if height * width < 10000:
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.calculate_field_coherence(field_data)
        
        # For large fields, try multi-GPU
        if self.multi_gpu_available and width * height >= 262144:  # 512x512 or larger
            try:
                return self._calculate_coherence_multi_gpu(field_data)
            except Exception as e:
                print(f"Multi-GPU coherence error: {e}")
                # Fall through to next method
        
        # Standard CUDA implementation
        try:
            return self._calculate_coherence_single_gpu(field_data)
        except Exception as e:
            print(f"CUDA coherence error: {e}")
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.calculate_field_coherence(field_data)
    
    def _calculate_coherence_single_gpu(self, field_data: np.ndarray) -> float:
        """Calculate field coherence using a single GPU"""
        height, width = field_data.shape
        
        # Set the first device as current
        self._set_device(0)
        
        # Copy field to GPU
        d_field = cp.array(field_data)
        
        # Prepare reduction arrays
        num_blocks = 32  # Use 32 blocks for reduction
        d_result = cp.zeros(num_blocks, dtype=cp.float32)
        d_count = cp.zeros(num_blocks, dtype=cp.int32)
        
        # Set up kernel launch parameters
        block_dim = (256, 1, 1)  # 256 threads per block
        grid_dim = (num_blocks, 1, 1)
        shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
        
        # Create launch config
        config = LaunchConfig(
            grid=grid_dim,
            block=block_dim,
            shmem_size=shmem_size
        )
        
        # Get the kernel
        kernel = self.modules[0].get_kernel("calculate_field_coherence")
        
        # Launch the kernel
        launch(
            self.streams[0], 
            config, 
            kernel, 
            d_field.data.ptr, 
            width, 
            height, 
            PHI, 
            d_result.data.ptr,
            d_count.data.ptr
        )
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Copy results back to host
        h_result = cp.asnumpy(d_result)
        h_count = cp.asnumpy(d_count)
        
        # Calculate overall coherence
        total_samples = np.sum(h_count)
        if total_samples > 0:
            coherence = np.sum(h_result) / total_samples * PHI
            return coherence
        
        return 0.0
    
    def _calculate_coherence_multi_gpu(self, field_data: np.ndarray) -> float:
        """Calculate field coherence using multiple GPUs"""
        if not self.multi_gpu_available:
            return self._calculate_coherence_single_gpu(field_data)
        
        height, width = field_data.shape
        
        # Divide the field among GPUs (row-wise partitioning)
        num_gpus = len(self.devices)
        rows_per_gpu = (height + num_gpus - 1) // num_gpus
        
        # Prepare arrays to collect results from all GPUs
        all_results = []
        all_counts = []
        
        # Process each shard on a different GPU
        for i in range(num_gpus):
            # Calculate the rows for this GPU
            start_row = i * rows_per_gpu
            end_row = min(start_row + rows_per_gpu, height)
            shard_height = end_row - start_row
            
            if shard_height <= 0:
                continue  # Skip if this GPU has no work
            
            # Extract the shard for this GPU
            shard = field_data[start_row:end_row, :]
            
            # Set this device as current
            self._set_device(i)
            
            with cp.cuda.Device(i):
                # Copy shard to GPU
                d_shard = cp.array(shard)
                
                # Prepare reduction arrays
                num_blocks = 32  # Use 32 blocks for reduction
                d_result = cp.zeros(num_blocks, dtype=cp.float32)
                d_count = cp.zeros(num_blocks, dtype=cp.int32)
                
                # Set up kernel launch parameters
                block_dim = (256, 1, 1)  # 256 threads per block
                grid_dim = (num_blocks, 1, 1)
                shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config
                config = LaunchConfig(
                    grid=grid_dim,
                    block=block_dim,
                    shmem_size=shmem_size
                )
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("calculate_field_coherence")
                
                # Launch the kernel
                launch(
                    self.streams[i], 
                    config, 
                    kernel, 
                    d_shard.data.ptr, 
                    width, 
                    shard_height,  # Use shard height here
                    PHI, 
                    d_result.data.ptr,
                    d_count.data.ptr
                )
                
                # Collect results
                all_results.append(cp.asnumpy(d_result))
                all_counts.append(cp.asnumpy(d_count))
        
        # Synchronize all streams
        for stream in self.streams:
            stream.sync()
        
        # Combine results from all GPUs
        if all_results and all_counts:
            combined_results = np.concatenate(all_results)
            combined_counts = np.concatenate(all_counts)
            
            total_samples = np.sum(combined_counts)
            if total_samples > 0:
                coherence = np.sum(combined_results) / total_samples * PHI
                return coherence
        
        # If no valid results, fall back to CPU implementation
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
        if not self.initialized or not self.modules:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_phi_pattern(width, height)
        
        try:
            # Set the first device as current
            self._set_device(0)
            
            # Create a CuPy array for output
            output = cp.empty((height, width), dtype=cp.float32)
            
            # Set up grid and block dimensions
            block_dim = (16, 16, 1)
            grid_dim = (
                (width + block_dim[0] - 1) // block_dim[0],
                (height + block_dim[1] - 1) // block_dim[1],
                1
            )
            
            # Create launch config
            config = LaunchConfig(grid=grid_dim, block=block_dim)
            
            # Get the kernel
            kernel = self.modules[0].get_kernel("generate_phi_pattern")
            
            # Launch the kernel
            launch(
                self.streams[0], 
                config, 
                kernel, 
                output.data.ptr, 
                width, 
                height,
                PHI
            )
            
            # Synchronize the stream
            self.streams[0].sync()
            
            # Convert CuPy array to NumPy
            return cp.asnumpy(output)
        except Exception as e:
            print(f"CUDA phi pattern error: {e}")
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_phi_pattern(width, height)
    
    # CUDA Graphs API
    
    def create_cuda_graph(self, graph_name: str, width: int, height: int, 
                          frequency_name: str = 'love', use_tbc: bool = None, 
                          use_multi_gpu: bool = None, depth: int = None) -> bool:
        """
        Create a CUDA graph for generating a quantum field with varying time factors
        
        Args:
            graph_name: Unique name for the graph (used for later execution)
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            use_tbc: Whether to use Thread Block Clusters. If None, auto-detect.
            use_multi_gpu: Whether to use Multi-GPU. If None, auto-detect.
            depth: If provided, creates a 3D field graph (depth, height, width)
            
        Returns:
            True if graph creation was successful, False otherwise
        """
        if not self.initialized or not self.cuda_graphs_available:
            return False
        
        # Clean up existing graph with the same name
        if graph_name in self.graphs:
            self.destroy_cuda_graph(graph_name)
        
        # Determine if this is a 3D field or 2D field
        is_3d = depth is not None and depth > 0
        
        if is_3d:
            # 3D field graph
            return self._create_3d_field_graph(
                graph_name, width, height, depth, frequency_name, use_multi_gpu, use_tbc
            )
        else:
            # 2D field graph - existing functionality
            # Determine the best implementation based on field size and hardware
            field_size = width * height
            
            # Auto-detect Thread Block Clusters if not specified
            if use_tbc is None:
                use_tbc = self.thread_block_clusters_available and field_size >= 1048576  # 1024x1024 or larger
            
            # Auto-detect Multi-GPU if not specified
            if use_multi_gpu is None:
                use_multi_gpu = self.multi_gpu_available and field_size >= 262144  # 512x512 or larger
            
            # Enforce capability constraints
            if use_tbc and not self.thread_block_clusters_available:
                use_tbc = False
            
            if use_multi_gpu and not self.multi_gpu_available:
                use_multi_gpu = False
            
            # Thread Block Clusters and Multi-GPU are mutually exclusive in graph mode
            # Thread Block Clusters take precedence
            if use_tbc:
                use_multi_gpu = False
            
            try:
                # Get the frequency value
                frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
                
                # Create a graph based on the selected implementation
                if use_tbc:
                    return self._create_tbc_graph(
                        graph_name, width, height, frequency, frequency_name
                    )
                elif use_multi_gpu:
                    return self._create_multi_gpu_graph(
                        graph_name, width, height, frequency, frequency_name
                    )
                else:
                    return self._create_basic_graph(
                        graph_name, width, height, frequency, frequency_name
                    )
            except Exception as e:
                print(f"Error creating CUDA graph '{graph_name}': {e}")
                return False
    
    def _create_basic_graph(self, graph_name: str, width: int, height: int, 
                           frequency: float, frequency_name: str) -> bool:
        """Create a basic CUDA graph using a single GPU"""
        # Set the first device as current
        self._set_device(0)
        
        # Create a stream for graph capture
        capture_stream = self.streams[0]
        
        # Create output memory that will be reused by the graph
        output = cp.empty((height, width), dtype=cp.float32)
        
        # Set up grid and block dimensions
        block_dim = (16, 16, 1)
        grid_dim = (
            (width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1],
            1
        )
        
        # Create launch config
        config = LaunchConfig(grid=grid_dim, block=block_dim)
        
        # Get the kernel
        kernel = self.modules[0].get_kernel("generate_quantum_field")
        
        # Begin graph capture
        graph = Graph(capture_stream)
        graph.begin_capture()
        
        # Launch the kernel with a placeholder time factor
        # The time factor will be updated each time we execute the graph
        launch(
            capture_stream, 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height, 
            frequency, 
            PHI, 
            LAMBDA, 
            0.0  # Placeholder time factor
        )
        
        # End graph capture
        graph.end_capture()
        
        # Instantiate the graph to get an executable
        graph_exec = graph.instantiate()
        
        # Store the graph and its executable
        self.graphs[graph_name] = {
            "graph": graph,
            "output": output,
            "width": width,
            "height": height,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "basic"
        }
        self.graph_execs[graph_name] = graph_exec
        
        return True
    
    def _create_tbc_graph(self, graph_name: str, width: int, height: int, 
                         frequency: float, frequency_name: str) -> bool:
        """Create a CUDA graph using Thread Block Clusters"""
        if not self.thread_block_clusters_available or self.tbc_module is None:
            return self._create_basic_graph(
                graph_name, width, height, frequency, frequency_name
            )
        
        # Set the first device as current
        self._set_device(0)
        
        # Create a stream for graph capture
        capture_stream = self.streams[0]
        
        # Create output memory that will be reused by the graph
        output = cp.empty((height, width), dtype=cp.float32)
        
        # Set up cluster and block dimensions (2x2 cluster of 16x16 blocks)
        block_dim = (16, 16, 1)
        cluster_shape = (2, 2, 1)  # 2x2 blocks per cluster
        grid_dim = (
            (width + block_dim[0] * cluster_shape[0] - 1) // (block_dim[0] * cluster_shape[0]),
            (height + block_dim[1] * cluster_shape[1] - 1) // (block_dim[1] * cluster_shape[1]),
            1
        )
        
        # Create thread block cluster launch config
        config = LaunchConfig(
            grid=grid_dim, 
            block=block_dim,
            cluster_shape=cluster_shape  # 2x2 blocks per cluster
        )
        
        # Get the kernel
        kernel = self.tbc_module.get_kernel("generate_quantum_field_tbc")
        
        # Begin graph capture
        graph = Graph(capture_stream)
        graph.begin_capture()
        
        # Launch the kernel with thread block clusters
        launch(
            capture_stream, 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height, 
            frequency, 
            PHI, 
            LAMBDA, 
            0.0  # Placeholder time factor
        )
        
        # End graph capture
        graph.end_capture()
        
        # Instantiate the graph to get an executable
        graph_exec = graph.instantiate()
        
        # Store the graph and its executable
        self.graphs[graph_name] = {
            "graph": graph,
            "output": output,
            "width": width,
            "height": height,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "tbc"
        }
        self.graph_execs[graph_name] = graph_exec
        
        return True
    
    def _create_multi_gpu_graph(self, graph_name: str, width: int, height: int, 
                               frequency: float, frequency_name: str) -> bool:
        """Create a CUDA graph using multiple GPUs
        
        Note: CUDA Graphs with Multi-GPU have limitations. Each graph is bound to
        a specific device, so we create a separate sub-graph for each GPU and
        manage their coordination at execution time.
        """
        if not self.multi_gpu_available or len(self.devices) <= 1:
            return self._create_basic_graph(
                graph_name, width, height, frequency, frequency_name
            )
        
        # Create output array to store the final result
        full_output = cp.empty((height, width), dtype=cp.float32)
        
        # Divide the field among GPUs (row-wise partitioning)
        num_gpus = min(len(self.devices), 4)  # Limit to 4 GPUs for graphs
        rows_per_gpu = (height + num_gpus - 1) // num_gpus
        
        # Set block dimensions for each GPU
        block_dim = (16, 16, 1)
        
        # Store GPU-specific graph information
        gpu_graphs = []
        gpu_outputs = []
        
        # Create a graph for each GPU
        for i in range(num_gpus):
            # Calculate the rows for this GPU
            start_row = i * rows_per_gpu
            end_row = min(start_row + rows_per_gpu, height)
            shard_height = end_row - start_row
            
            if shard_height <= 0:
                continue  # Skip if this GPU has no work
            
            # Set this device as current
            self._set_device(i)
            
            # Create a stream for graph capture
            capture_stream = self.streams[i]
            
            # Create a CuPy array for this GPU's output
            with cp.cuda.Device(i):
                gpu_output = cp.empty((shard_height, width), dtype=cp.float32)
                
                # Set up grid dimensions
                grid_dim = (
                    (width + block_dim[0] - 1) // block_dim[0],
                    (shard_height + block_dim[1] - 1) // block_dim[1],
                    1
                )
                
                # Create launch config
                config = LaunchConfig(grid=grid_dim, block=block_dim)
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("generate_quantum_field")
                
                # Begin graph capture
                sub_graph = Graph(capture_stream)
                sub_graph.begin_capture()
                
                # Launch the kernel
                launch(
                    capture_stream, 
                    config, 
                    kernel, 
                    gpu_output.data.ptr, 
                    width, 
                    height,  # Full height for center calculation
                    frequency, 
                    PHI, 
                    LAMBDA, 
                    0.0  # Placeholder time factor
                )
                
                # End graph capture
                sub_graph.end_capture()
                
                # Instantiate the graph
                sub_graph_exec = sub_graph.instantiate()
                
                # Store this GPU's graph info
                gpu_graphs.append({
                    "device_id": i,
                    "graph": sub_graph,
                    "graph_exec": sub_graph_exec,
                    "start_row": start_row,
                    "end_row": end_row,
                    "shard_height": shard_height
                })
                
                # Store the output array
                gpu_outputs.append(gpu_output)
        
        # Store the graph information
        self.graphs[graph_name] = {
            "output": full_output,
            "width": width,
            "height": height,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "multi_gpu",
            "gpu_graphs": gpu_graphs,
            "gpu_outputs": gpu_outputs,
            "num_gpus": num_gpus
        }
        
        # No need to store in graph_execs for multi-GPU, we handle them differently
        
        return True
    
    def execute_cuda_graph(self, graph_name: str, time_factor: float = 0) -> Optional[np.ndarray]:
        """
        Execute a previously created CUDA graph with a new time factor
        
        Args:
            graph_name: Name of the graph to execute
            time_factor: Time factor for animation
            
        Returns:
            A 2D or 3D NumPy array representing the quantum field, or None if error
        """
        if not self.initialized or graph_name not in self.graphs:
            return None
        
        try:
            # Get graph info
            graph_info = self.graphs[graph_name]
            
            # Check if this is a 3D field graph
            is_3d = graph_info.get("is_3d", False)
            
            if is_3d:
                return self._execute_3d_field_graph(graph_name, time_factor)
            else:
                # 2D field graph - existing functionality
                graph_type = graph_info.get("type", "basic")
                
                if graph_type == "multi_gpu":
                    return self._execute_multi_gpu_graph(graph_name, time_factor)
                else:  # basic or tbc
                    return self._execute_single_gpu_graph(graph_name, time_factor)
        except Exception as e:
            print(f"Error executing CUDA graph '{graph_name}': {e}")
            return None
    
    def _execute_single_gpu_graph(self, graph_name: str, time_factor: float = 0) -> np.ndarray:
        """Execute a graph running on a single GPU (basic or tbc)"""
        if graph_name not in self.graph_execs:
            raise ValueError(f"Graph executable not found for '{graph_name}'")
        
        # Get graph info
        graph_info = self.graphs[graph_name]
        graph_exec = self.graph_execs[graph_name]
        output = graph_info["output"]
        
        # Set the device
        self._set_device(0)  # Single GPU graphs use device 0
        
        # Set the time factor parameter
        graph_exec.set_params(
            output.data.ptr,
            graph_info["width"],
            graph_info["height"],
            graph_info["frequency"],
            PHI,
            LAMBDA,
            time_factor  # Updated time factor
        )
        
        # Launch the graph
        graph_exec.launch(self.streams[0])
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy
        return cp.asnumpy(output)
    
    def _execute_multi_gpu_graph(self, graph_name: str, time_factor: float = 0) -> np.ndarray:
        """Execute a graph distributed across multiple GPUs"""
        # Get graph info
        graph_info = self.graphs[graph_name]
        full_output = graph_info["output"]
        gpu_graphs = graph_info.get("gpu_graphs", [])
        gpu_outputs = graph_info.get("gpu_outputs", [])
        
        if not gpu_graphs or len(gpu_graphs) != len(gpu_outputs):
            raise ValueError(f"Invalid multi-GPU graph configuration for '{graph_name}'")
        
        # Launch each sub-graph on its respective GPU
        for i, gpu_graph in enumerate(gpu_graphs):
            # Set the device for this sub-graph
            device_id = gpu_graph["device_id"]
            self._set_device(device_id)
            
            # Get the sub-graph executable and output buffer
            sub_graph_exec = gpu_graph["graph_exec"]
            gpu_output = gpu_outputs[i]
            
            # Set the time factor parameter
            sub_graph_exec.set_params(
                gpu_output.data.ptr,
                graph_info["width"],
                graph_info["height"],  # Full height for center calculation
                graph_info["frequency"],
                PHI,
                LAMBDA,
                time_factor  # Updated time factor
            )
            
            # Launch the sub-graph
            sub_graph_exec.launch(self.streams[device_id])
        
        # Synchronize all streams
        for i in range(len(gpu_graphs)):
            device_id = gpu_graphs[i]["device_id"]
            self.streams[device_id].sync()
        
        # Collect results from all GPUs
        for i, gpu_graph in enumerate(gpu_graphs):
            device_id = gpu_graph["device_id"]
            self._set_device(device_id)
            
            # Get this GPU's output array and shard information
            gpu_output = gpu_outputs[i]
            start_row = gpu_graph["start_row"]
            end_row = gpu_graph["end_row"]
            
            # Copy this shard to the output array
            with cp.cuda.Device(device_id):
                # Use CuPy's built-in slice assignment to combine results
                full_output[start_row:end_row, :] = gpu_output
        
        # Return the combined result
        return cp.asnumpy(full_output)
    
    def destroy_cuda_graph(self, graph_name: str) -> bool:
        """
        Destroy a previously created CUDA graph and free resources
        
        Args:
            graph_name: Name of the graph to destroy
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or graph_name not in self.graphs:
            return False
        
        try:
            # Get graph info
            graph_info = self.graphs[graph_name]
            graph_type = graph_info.get("type", "basic")
            is_3d = graph_info.get("is_3d", False)
            
            # Handle multi-GPU graphs specially
            if graph_type in ["multi_gpu", "3d_multi_gpu"]:
                # Clean up GPU-specific resources
                gpu_graphs = graph_info.get("gpu_graphs", [])
                gpu_outputs = graph_info.get("gpu_outputs", [])
                
                # Delete all GPU outputs
                for gpu_output in gpu_outputs:
                    del gpu_output
                
                # Delete the full output array
                del graph_info["output"]
                
                # Remove the graph from our records
                del self.graphs[graph_name]
            else:
                # Clean up single-GPU graph resources
                if graph_name in self.graph_execs:
                    del self.graph_execs[graph_name]
                
                # Delete the output memory
                del graph_info["output"]
                
                # Remove from dictionary
                del self.graphs[graph_name]
            
            return True
        except Exception as e:
            print(f"Error destroying CUDA graph '{graph_name}': {e}")
            return False
    
    def _create_3d_field_graph(self, graph_name: str, width: int, height: int, depth: int,
                                 frequency_name: str, use_multi_gpu: bool = None,
                                 use_tbc: bool = None) -> bool:
        """
        Create a CUDA graph for generating 3D quantum fields
        
        Args:
            graph_name: Unique name for the graph
            width: Width of the 3D field
            height: Height of the 3D field
            depth: Depth of the 3D field
            frequency_name: The sacred frequency to use
            use_multi_gpu: Whether to use multiple GPUs (if available)
            use_tbc: Whether to use Thread Block Clusters (if available)
            
        Returns:
            True if graph creation was successful
        """
        if not self.initialized or not self.cuda_graphs_available or not self.has_3d_capability:
            return False
            
        # Get the frequency value
        frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
        
        # Determine field size
        field_size = width * height * depth
        
        # Auto-detect Thread Block Clusters if not specified
        if use_tbc is None:
            use_tbc = self.thread_block_clusters_available and field_size >= 2_000_000  # Large 3D fields
            
        # Auto-detect Multi-GPU if not specified
        if use_multi_gpu is None:
            use_multi_gpu = self.multi_gpu_available and field_size >= 8_000_000  # Very large 3D fields
            
        # Enforce capability constraints
        if use_tbc and not self.thread_block_clusters_available:
            use_tbc = False
            
        if use_multi_gpu and not self.multi_gpu_available:
            use_multi_gpu = False
            
        # Thread Block Clusters and Multi-GPU are mutually exclusive in graph mode
        # Thread Block Clusters take precedence
        if use_tbc:
            use_multi_gpu = False
            
        # Create the appropriate type of 3D graph
        if use_tbc:
            return self._create_3d_tbc_graph(
                graph_name, width, height, depth, frequency, frequency_name
            )
        elif use_multi_gpu:
            return self._create_3d_multi_gpu_graph(
                graph_name, width, height, depth, frequency, frequency_name
            )
        else:
            return self._create_3d_single_gpu_graph(
                graph_name, width, height, depth, frequency, frequency_name
            )
    
    def _create_3d_single_gpu_graph(self, graph_name: str, width: int, height: int, depth: int,
                                   frequency: float, frequency_name: str) -> bool:
        """Create a 3D CUDA graph using a single GPU"""
        # Set the first device as current
        self._set_device(0)
        
        # Create a stream for graph capture
        capture_stream = self.streams[0]
        
        # Create output memory that will be reused by the graph
        # Note: We use (width, height, depth) for CUDA memory layout
        output = cp.empty((width, height, depth), dtype=cp.float32)
        
        # Set up grid and block dimensions - 8x8x8 threads per block
        block_dim = (8, 8, 8)
        grid_dim = (
            (width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1],
            (depth + block_dim[2] - 1) // block_dim[2]
        )
        
        # Create launch config
        config = LaunchConfig(grid=grid_dim, block=block_dim)
        
        # Get the kernel
        kernel = self.modules[0].get_kernel("generate_3d_quantum_field")
        
        # Begin graph capture
        graph = Graph(capture_stream)
        graph.begin_capture()
        
        # Launch the kernel with a placeholder time factor
        launch(
            capture_stream, 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height,
            depth,
            frequency, 
            PHI, 
            LAMBDA, 
            0.0  # Placeholder time factor
        )
        
        # End graph capture
        graph.end_capture()
        
        # Instantiate the graph to get an executable
        graph_exec = graph.instantiate()
        
        # Store the graph and its executable
        self.graphs[graph_name] = {
            "graph": graph,
            "output": output,
            "width": width,
            "height": height,
            "depth": depth,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "3d_single_gpu",
            "is_3d": True
        }
        self.graph_execs[graph_name] = graph_exec
        
        return True
    
    def _create_3d_multi_gpu_graph(self, graph_name: str, width: int, height: int, depth: int,
                                  frequency: float, frequency_name: str) -> bool:
        """Create a 3D CUDA graph using multiple GPUs"""
        if not self.multi_gpu_available or len(self.devices) <= 1:
            return self._create_3d_single_gpu_graph(
                graph_name, width, height, depth, frequency, frequency_name
            )
        
        # Create output array to store the final result
        full_output = cp.empty((width, height, depth), dtype=cp.float32)
        
        # Divide along depth dimension for multi-GPU processing
        num_gpus = min(len(self.devices), 4)  # Limit to 4 GPUs for graphs
        depth_per_gpu = (depth + num_gpus - 1) // num_gpus
        
        # Set block dimensions for each GPU
        block_dim = (8, 8, 8)
        
        # Store GPU-specific graph information
        gpu_graphs = []
        gpu_outputs = []
        
        # Create a graph for each GPU
        for i in range(num_gpus):
            # Calculate the depth range for this GPU
            start_depth = i * depth_per_gpu
            end_depth = min(start_depth + depth_per_gpu, depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Set this device as current
            self._set_device(i)
            
            # Create a stream for graph capture
            capture_stream = self.streams[i]
            
            # Create a CuPy array for this GPU's output
            with cp.cuda.Device(i):
                gpu_output = cp.empty((width, height, shard_depth), dtype=cp.float32)
                
                # Set up grid dimensions
                grid_dim = (
                    (width + block_dim[0] - 1) // block_dim[0],
                    (height + block_dim[1] - 1) // block_dim[1],
                    (shard_depth + block_dim[2] - 1) // block_dim[2]
                )
                
                # Create launch config
                config = LaunchConfig(grid=grid_dim, block=block_dim)
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("generate_3d_quantum_field")
                
                # Begin graph capture
                sub_graph = Graph(capture_stream)
                sub_graph.begin_capture()
                
                # Launch the kernel with placeholder time factor
                launch(
                    capture_stream, 
                    config, 
                    kernel, 
                    gpu_output.data.ptr, 
                    width, 
                    height, 
                    shard_depth,
                    frequency, 
                    PHI, 
                    LAMBDA, 
                    0.0  # Placeholder time factor
                )
                
                # End graph capture
                sub_graph.end_capture()
                
                # Instantiate the graph
                sub_graph_exec = sub_graph.instantiate()
                
                # Store this GPU's graph info
                gpu_graphs.append({
                    "device_id": i,
                    "graph": sub_graph,
                    "graph_exec": sub_graph_exec,
                    "start_depth": start_depth,
                    "end_depth": end_depth,
                    "shard_depth": shard_depth
                })
                
                # Store the output array
                gpu_outputs.append(gpu_output)
        
        # Store the graph information
        self.graphs[graph_name] = {
            "output": full_output,
            "width": width,
            "height": height,
            "depth": depth,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "3d_multi_gpu",
            "gpu_graphs": gpu_graphs,
            "gpu_outputs": gpu_outputs,
            "num_gpus": num_gpus,
            "is_3d": True
        }
        
        return True
    
    def _create_3d_tbc_graph(self, graph_name: str, width: int, height: int, depth: int,
                                frequency: float, frequency_name: str) -> bool:
        """Create a 3D CUDA graph using Thread Block Clusters"""
        if not self.thread_block_clusters_available or self.tbc_module is None:
            return self._create_3d_single_gpu_graph(
                graph_name, width, height, depth, frequency, frequency_name
            )
        
        # Set the first device as current (TBC requires a device with compute capability 9.0+)
        self._set_device(0)
        
        # Create a stream for graph capture
        capture_stream = self.streams[0]
        
        # Create output memory that will be reused by the graph
        output = cp.empty((width, height, depth), dtype=cp.float32)
        
        # Set up cluster and block dimensions (2x2x2 cluster of 4x4x4 blocks)
        block_dim = (4, 4, 4)  # 64 threads per block
        cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster
        
        # Calculate grid dimensions
        grid_dim = (
            (width + block_dim[0] * cluster_shape[0] - 1) // (block_dim[0] * cluster_shape[0]),
            (height + block_dim[1] * cluster_shape[1] - 1) // (block_dim[1] * cluster_shape[1]),
            (depth + block_dim[2] * cluster_shape[2] - 1) // (block_dim[2] * cluster_shape[2])
        )
        
        # Create thread block cluster launch config
        config = LaunchConfig(
            grid=grid_dim, 
            block=block_dim,
            cluster_shape=cluster_shape
        )
        
        # Get the kernel
        kernel = self.tbc_module.get_kernel("generate_3d_quantum_field_tbc")
        
        # Begin graph capture
        graph = Graph(capture_stream)
        graph.begin_capture()
        
        # Launch the kernel with a placeholder time factor
        launch(
            capture_stream, 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height,
            depth,
            frequency, 
            PHI, 
            LAMBDA, 
            0.0  # Placeholder time factor
        )
        
        # End graph capture
        graph.end_capture()
        
        # Instantiate the graph to get an executable
        graph_exec = graph.instantiate()
        
        # Store the graph and its executable
        self.graphs[graph_name] = {
            "graph": graph,
            "output": output,
            "width": width,
            "height": height,
            "depth": depth,
            "frequency": frequency,
            "frequency_name": frequency_name,
            "type": "3d_tbc",
            "is_3d": True
        }
        self.graph_execs[graph_name] = graph_exec
        
        return True
    
    def _execute_3d_field_graph(self, graph_name: str, time_factor: float = 0) -> np.ndarray:
        """
        Execute a 3D quantum field graph with the given time factor
        
        Args:
            graph_name: Name of the graph to execute
            time_factor: Time evolution parameter
            
        Returns:
            3D NumPy array with the field values
        """
        if graph_name not in self.graphs:
            raise ValueError(f"Graph not found: {graph_name}")
            
        # Get graph info
        graph_info = self.graphs[graph_name]
        graph_type = graph_info.get("type", "3d_single_gpu")
        
        if graph_type == "3d_multi_gpu":
            return self._execute_3d_multi_gpu_graph(graph_name, time_factor)
        elif graph_type == "3d_tbc":
            return self._execute_3d_single_gpu_graph(graph_name, time_factor)  # TBC graph execution is same as single-GPU
        else:
            return self._execute_3d_single_gpu_graph(graph_name, time_factor)
    
    def _execute_3d_single_gpu_graph(self, graph_name: str, time_factor: float = 0) -> np.ndarray:
        """Execute a 3D graph on a single GPU"""
        if graph_name not in self.graph_execs:
            raise ValueError(f"Graph executable not found for '{graph_name}'")
        
        # Get graph info
        graph_info = self.graphs[graph_name]
        graph_exec = self.graph_execs[graph_name]
        output = graph_info["output"]
        
        # Set the device
        self._set_device(0)
        
        # Set the time factor parameter
        graph_exec.set_params(
            output.data.ptr,
            graph_info["width"],
            graph_info["height"],
            graph_info["depth"],
            graph_info["frequency"],
            PHI,
            LAMBDA,
            time_factor  # Updated time factor
        )
        
        # Launch the graph
        graph_exec.launch(self.streams[0])
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy and transpose to expected order (depth, height, width)
        return cp.asnumpy(output).transpose(2, 1, 0)
    
    def _execute_3d_multi_gpu_graph(self, graph_name: str, time_factor: float = 0) -> np.ndarray:
        """Execute a 3D graph distributed across multiple GPUs"""
        # Get graph info
        graph_info = self.graphs[graph_name]
        full_output = graph_info["output"]
        gpu_graphs = graph_info.get("gpu_graphs", [])
        gpu_outputs = graph_info.get("gpu_outputs", [])
        
        if not gpu_graphs or len(gpu_graphs) != len(gpu_outputs):
            raise ValueError(f"Invalid multi-GPU graph configuration for '{graph_name}'")
        
        # Launch each sub-graph on its respective GPU
        for i, gpu_graph in enumerate(gpu_graphs):
            # Set the device for this sub-graph
            device_id = gpu_graph["device_id"]
            self._set_device(device_id)
            
            # Get the sub-graph executable and output buffer
            sub_graph_exec = gpu_graph["graph_exec"]
            gpu_output = gpu_outputs[i]
            shard_depth = gpu_graph["shard_depth"]
            
            # Set the time factor parameter
            sub_graph_exec.set_params(
                gpu_output.data.ptr,
                graph_info["width"],
                graph_info["height"],
                shard_depth,  # Use this GPU's shard depth
                graph_info["frequency"],
                PHI,
                LAMBDA,
                time_factor  # Updated time factor
            )
            
            # Launch the sub-graph
            sub_graph_exec.launch(self.streams[device_id])
        
        # Synchronize all streams
        for i in range(len(gpu_graphs)):
            device_id = gpu_graphs[i]["device_id"]
            self.streams[device_id].sync()
        
        # Collect results from all GPUs
        for i, gpu_graph in enumerate(gpu_graphs):
            device_id = gpu_graph["device_id"]
            self._set_device(device_id)
            
            # Get this GPU's output array and shard information
            gpu_output = gpu_outputs[i]
            start_depth = gpu_graph["start_depth"]
            end_depth = gpu_graph["end_depth"]
            
            # Copy this shard to the output array
            with cp.cuda.Device(device_id):
                # Use CuPy's built-in slice assignment to combine results
                full_output[:, :, start_depth:end_depth] = gpu_output
        
        # Return the combined result, transposed to match expected order (depth, height, width)
        return cp.asnumpy(full_output).transpose(2, 1, 0)
    
    def list_cuda_graphs(self) -> List[Dict[str, Any]]:
        """
        List all created CUDA graphs
        
        Returns:
            A list of dictionaries with information about each graph
        """
        result = []
        for name, info in self.graphs.items():
            graph_type = info.get("type", "basic")
            is_3d = info.get("is_3d", False)
            
            graph_data = {
                "name": name,
                "width": info["width"],
                "height": info["height"],
                "frequency_name": info["frequency_name"],
                "frequency": info["frequency"],
                "type": graph_type,
                "is_3d": is_3d
            }
            
            # Add 3D-specific information
            if is_3d:
                graph_data["depth"] = info.get("depth", 0)
                
            # Add additional information based on graph type
            if graph_type in ["multi_gpu", "3d_multi_gpu"]:
                graph_data["num_gpus"] = info.get("num_gpus", 0)
                graph_data["gpu_shards"] = len(info.get("gpu_graphs", []))
            elif graph_type == "tbc":
                graph_data["using_thread_block_clusters"] = True
            
            result.append(graph_data)
        
        return result
    
    def generate_3d_quantum_field(self, width: int, height: int, depth: int,
                              frequency_name: str = 'love',
                              time_factor: float = 0.0,
                              custom_frequency: Optional[float] = None) -> np.ndarray:
        """
        Generate a 3D quantum field using CUDA
        
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
        if not self.initialized or not self.modules or not self.has_3d_capability:
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_3d_quantum_field(width, height, depth, frequency_name, 
                                                        time_factor, custom_frequency)
        
        # Determine frequency
        if frequency_name is not None:
            if frequency_name not in SACRED_FREQUENCIES:
                raise ValueError(f"Unknown frequency name: {frequency_name}")
            frequency = SACRED_FREQUENCIES[frequency_name]
        elif custom_frequency is not None:
            frequency = custom_frequency
        else:
            raise ValueError("Either frequency_name or custom_frequency must be provided")
            
        # Determine the best implementation based on field size and hardware
        field_size = width * height * depth
        
        # Use Thread Block Clusters for very large 3D fields (if available)
        if self.thread_block_clusters_available and field_size >= 2_000_000:  # e.g., 128x128x128 or larger
            try:
                return self._generate_3d_field_tbc(width, height, depth, frequency, time_factor)
            except Exception as e:
                print(f"Thread Block Cluster 3D field error: {e}")
                # Fall through to next method
        
        # Use Multi-GPU for large 3D fields (if available)
        if self.multi_gpu_available and field_size >= 8_000_000:  # e.g., 200x200x200 or larger
            try:
                return self._generate_3d_field_multi_gpu(width, height, depth, frequency, time_factor)
            except Exception as e:
                print(f"Multi-GPU 3D field error: {e}")
                # Fall through to next method
                
        # Standard CUDA implementation for 3D fields
        try:
            return self._generate_3d_field_single_gpu(width, height, depth, frequency, time_factor)
        except Exception as e:
            print(f"CUDA 3D field error: {e}")
            # Fall back to CPU implementation
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.generate_3d_quantum_field(width, height, depth, frequency_name, 
                                                        time_factor, custom_frequency)
    
    def _generate_3d_field_single_gpu(self, width: int, height: int, depth: int, 
                                     frequency: float, time_factor: float = 0.0) -> np.ndarray:
        """Generate a 3D quantum field using a single GPU"""
        # Set the first device as current
        self._set_device(0)
        
        # Create a CuPy array for output
        output = cp.empty((width, height, depth), dtype=cp.float32)
        
        # Set up grid and block dimensions
        # Use 2D blocks with 1D grid-z to optimize for most 3D fields
        block_dim = (8, 8, 8)  # 8x8x8 threads per block = 512 threads
        grid_dim = (
            (width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1],
            (depth + block_dim[2] - 1) // block_dim[2]
        )
        
        # Create launch config
        config = LaunchConfig(grid=grid_dim, block=block_dim)
        
        # Get the kernel
        kernel = self.modules[0].get_kernel("generate_3d_quantum_field")
        
        # Launch the kernel
        launch(
            self.streams[0], 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height,
            depth,
            frequency, 
            PHI, 
            LAMBDA, 
            time_factor
        )
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy and transpose to match expected order (depth, height, width)
        return cp.asnumpy(output).transpose(2, 1, 0)
        
    def _generate_3d_field_tbc(self, width: int, height: int, depth: int, 
                               frequency: float, time_factor: float = 0.0) -> np.ndarray:
        """Generate a 3D quantum field using Thread Block Clusters"""
        if not self.thread_block_clusters_available or self.tbc_module is None:
            return self._generate_3d_field_single_gpu(width, height, depth, frequency, time_factor)
        
        # Set the first device as current (TBC requires a single GPU with compute capability 9.0+)
        self._set_device(0)
        
        # Create a CuPy array for output
        output = cp.empty((width, height, depth), dtype=cp.float32)
        
        # Set up cluster and block dimensions (2x2x2 cluster of 4x4x4 blocks)
        # This configuration creates more threads per SM for better occupancy
        block_dim = (4, 4, 4)  # 64 threads per block
        cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster = 8 blocks
        
        # Calculate grid dimensions
        # Each cluster handles 2x block size in each dimension
        grid_dim = (
            (width + block_dim[0] * cluster_shape[0] - 1) // (block_dim[0] * cluster_shape[0]),
            (height + block_dim[1] * cluster_shape[1] - 1) // (block_dim[1] * cluster_shape[1]),
            (depth + block_dim[2] * cluster_shape[2] - 1) // (block_dim[2] * cluster_shape[2])
        )
        
        # Create thread block cluster launch config
        config = LaunchConfig(
            grid=grid_dim, 
            block=block_dim,
            cluster_shape=cluster_shape
        )
        
        # Get the kernel
        kernel = self.tbc_module.get_kernel("generate_3d_quantum_field_tbc")
        
        # Launch the kernel with thread block clusters
        launch(
            self.streams[0], 
            config, 
            kernel, 
            output.data.ptr, 
            width, 
            height, 
            depth,
            frequency, 
            PHI, 
            LAMBDA, 
            time_factor
        )
        
        # Synchronize the stream
        self.streams[0].sync()
        
        # Convert CuPy array to NumPy and transpose to match expected order (depth, height, width)
        return cp.asnumpy(output).transpose(2, 1, 0)
    
    def _create_3d_coherence_graph(self, width: int, height: int, depth: int, use_tbc: bool = False) -> Dict:
        """
        Create a CUDA graph for 3D field coherence calculation.
        
        This creates a reusable graph that can be executed repeatedly to calculate 
        the coherence of 3D fields with the same dimensions.
        
        Args:
            width: Width of the field
            height: Height of the field
            depth: Depth of the field
            use_tbc: Whether to use Thread Block Clusters (for H100+ GPUs)
            
        Returns:
            A dictionary containing the graph and related resources
        """
        if not self.initialized or not self.has_3d_capability:
            raise RuntimeError("CUDA backend not properly initialized for 3D operations")
        
        # Check if we should use Thread Block Clusters
        if use_tbc and not self.has_thread_block_cluster_support:
            print("Thread Block Clusters requested but not supported, falling back to standard implementation")
            use_tbc = False
        
        # Set the first device as current
        self._set_device(0)
        
        try:
            # Allocate memory for the field
            d_field = cp.zeros((width, height, depth), dtype=cp.float32)
            
            # Prepare reduction arrays - more blocks for 3D for better sampling
            if use_tbc:
                # Using 2x2x2 clusters, so we need 8x more result slots per block
                num_blocks = 32  # Use 32 blocks for sampling
                d_result = cp.zeros(num_blocks * 8, dtype=cp.float32)
                d_count = cp.zeros(num_blocks * 8, dtype=cp.int32)
            else:
                num_blocks = 64  # Use more blocks for standard implementation
                d_result = cp.zeros(num_blocks, dtype=cp.float32)
                d_count = cp.zeros(num_blocks, dtype=cp.int32)
            
            # Configure launch parameters
            if use_tbc:
                # Thread Block Cluster configuration
                block_dim = (256, 1, 1)  # 256 threads per block (1D)
                cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster (3D)
                grid_dim = (num_blocks, 1, 1)
                shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create thread block cluster launch config
                config = LaunchConfig(
                    grid=grid_dim, 
                    block=block_dim,
                    cluster_shape=cluster_shape,  # 2x2x2 blocks per cluster
                    shmem_size=shmem_size
                )
                
                # Get the kernel
                kernel = self.tbc_modules[0].get_kernel("calculate_3d_field_coherence_tbc<float>")
            else:
                # Standard configuration
                block_dim = (256, 1, 1)  # 256 threads per block
                grid_dim = (num_blocks, 1, 1)
                shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config
                config = LaunchConfig(
                    grid=grid_dim,
                    block=block_dim,
                    shmem_size=shmem_size
                )
                
                # Get the kernel
                kernel = self.modules[0].get_kernel("calculate_3d_field_coherence")
            
            # Create CUDA Graph
            graph = Graph()
            stream = self.streams[0]
            
            # Begin capturing
            graph.begin_capture(stream)
            
            # Launch the kernel
            launch(
                stream, 
                config, 
                kernel, 
                d_field.data.ptr, 
                width, 
                height, 
                depth,
                PHI, 
                LAMBDA,
                d_result.data.ptr,
                d_count.data.ptr
            )
            
            # End capturing
            graph.end_capture()
            
            # Instantiate the graph
            graph_exec = graph.instantiate()
            
            # Return the graph and resources
            return {
                "graph": graph,
                "graph_exec": graph_exec,
                "stream": stream,
                "d_field": d_field,
                "d_result": d_result,
                "d_count": d_count,
                "width": width,
                "height": height,
                "depth": depth,
                "use_tbc": use_tbc
            }
            
        except Exception as e:
            print(f"Error creating 3D coherence CUDA graph: {e}")
            raise RuntimeError(f"Failed to create 3D coherence CUDA graph: {e}")
    
    def _execute_3d_coherence_graph(self, graph_data: Dict, field_data: np.ndarray) -> float:
        """
        Execute a CUDA graph to calculate the coherence of a 3D field.
        
        Args:
            graph_data: The graph data returned by _create_3d_coherence_graph
            field_data: The 3D field to analyze
            
        Returns:
            Coherence factor between 0.0 and 1.0
        """
        if not self.initialized:
            raise RuntimeError("CUDA backend not initialized")
        
        # Ensure the field has the correct dimensions
        depth, height, width = field_data.shape
        if (width != graph_data["width"] or 
            height != graph_data["height"] or 
            depth != graph_data["depth"]):
            raise ValueError(f"Field dimensions ({depth}, {height}, {width}) don't match " 
                           f"graph dimensions ({graph_data['depth']}, {graph_data['height']}, {graph_data['width']})")
        
        try:
            # Set the first device as current
            self._set_device(0)
            
            # Copy the field data to the device
            d_field = graph_data["d_field"]
            cp.copyto(d_field, field_data.transpose(2, 1, 0))
            
            # Launch the graph
            graph_data["graph_exec"].launch(graph_data["stream"])
            
            # Synchronize the stream
            graph_data["stream"].sync()
            
            # Copy results back to host
            d_result = graph_data["d_result"]
            d_count = graph_data["d_count"]
            h_result = cp.asnumpy(d_result)
            h_count = cp.asnumpy(d_count)
            
            # Calculate overall coherence
            total_samples = np.sum(h_count)
            if total_samples > 0:
                coherence = np.sum(h_result) / total_samples * PHI
                # Ensure result is in [0, 1] range
                coherence = max(0.0, min(1.0, coherence))
                return coherence
            
            return 0.0
            
        except Exception as e:
            print(f"Error executing 3D coherence CUDA graph: {e}")
            raise RuntimeError(f"Failed to execute 3D coherence CUDA graph: {e}")
    
    def _calculate_3d_field_coherence_tbc(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a 3D quantum field using Thread Block Clusters.
        
        This optimized method uses CUDA thread block clusters available on H100+ GPUs
        for improved performance with large 3D fields.
        
        Args:
            field_data: 3D NumPy array containing the field
            
        Returns:
            Coherence factor between 0.0 and 1.0
        """
        if not self.initialized or not self.modules or not self.has_3d_capability:
            raise RuntimeError("CUDABackend not properly initialized for 3D operations")
        
        if not self.has_thread_block_cluster_support:
            raise RuntimeError("Thread Block Clusters not supported on this device")
        
        depth, height, width = field_data.shape
        
        try:
            # Set the first device as current
            self._set_device(0)
            
            # Copy field to GPU - making sure dimensions are in the right order for CUDA
            d_field = cp.array(field_data.transpose(2, 1, 0))
            
            # Prepare reduction arrays - more blocks for better sampling
            # Using 2x2x2 clusters, so we need 8x more result slots per block
            num_blocks = 32  # Use 32 blocks for enough sampling
            d_result = cp.zeros(num_blocks * 8, dtype=cp.float32)
            d_count = cp.zeros(num_blocks * 8, dtype=cp.int32)
            
            # Set up kernel launch parameters with thread block clusters
            block_dim = (256, 1, 1)  # 256 threads per block (1D)
            cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster (3D)
            grid_dim = (num_blocks, 1, 1)
            shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
            
            # Create thread block cluster launch config
            config = LaunchConfig(
                grid=grid_dim, 
                block=block_dim,
                cluster_shape=cluster_shape,  # 2x2x2 blocks per cluster
                shmem_size=shmem_size
            )
            
            # Get the kernel
            # Use our compiled TBC kernels
            kernel = self.tbc_modules[0].get_kernel("calculate_3d_field_coherence_tbc<float>")
            
            # Launch the kernel with thread block clusters
            launch(
                self.streams[0], 
                config, 
                kernel, 
                d_field.data.ptr, 
                width, 
                height, 
                depth,
                PHI, 
                LAMBDA,
                d_result.data.ptr,
                d_count.data.ptr
            )
            
            # Synchronize the stream
            self.streams[0].sync()
            
            # Copy results back to host
            h_result = cp.asnumpy(d_result)
            h_count = cp.asnumpy(d_count)
            
            # Calculate overall coherence
            total_samples = np.sum(h_count)
            if total_samples > 0:
                coherence = np.sum(h_result) / total_samples * PHI
                # Ensure result is in [0, 1] range
                coherence = max(0.0, min(1.0, coherence))
                return coherence
            
            return 0.0
        except Exception as e:
            raise RuntimeError(f"Error in Thread Block Cluster 3D coherence calculation: {e}")
    
    def _generate_3d_field_multi_gpu(self, width: int, height: int, depth: int, 
                                    frequency: float, time_factor: float = 0.0) -> np.ndarray:
        """Generate a 3D quantum field distributed across multiple GPUs"""
        if not self.multi_gpu_available:
            return self._generate_3d_field_single_gpu(width, height, depth, frequency, time_factor)
        
        # Create output array
        output = np.zeros((width, height, depth), dtype=np.float32)
        
        # Divide the field among GPUs (depth-wise partitioning)
        num_gpus = len(self.devices)
        depth_per_gpu = (depth + num_gpus - 1) // num_gpus
        
        # Set block dimensions for each GPU
        block_dim = (8, 8, 8)  # 8x8x8 threads per block = 512 threads
        
        # List to store output arrays from each GPU
        gpu_outputs = []
        
        # Launch kernels on each GPU
        for i in range(num_gpus):
            # Calculate the depth range for this GPU
            start_depth = i * depth_per_gpu
            end_depth = min(start_depth + depth_per_gpu, depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Set this device as current
            self._set_device(i)
            
            # Create a CuPy array for this GPU's output
            with cp.cuda.Device(i):
                gpu_output = cp.empty((width, height, shard_depth), dtype=cp.float32)
                
                # Set up grid dimensions
                grid_dim = (
                    (width + block_dim[0] - 1) // block_dim[0],
                    (height + block_dim[1] - 1) // block_dim[1],
                    (shard_depth + block_dim[2] - 1) // block_dim[2]
                )
                
                # Create launch config
                config = LaunchConfig(grid=grid_dim, block=block_dim)
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("generate_3d_quantum_field")
                
                # Launch the kernel
                launch(
                    self.streams[i], 
                    config, 
                    kernel, 
                    gpu_output.data.ptr, 
                    width, 
                    height, 
                    shard_depth,
                    frequency, 
                    PHI, 
                    LAMBDA, 
                    time_factor
                )
                
                # Add to list for later collection
                gpu_outputs.append((i, start_depth, end_depth, gpu_output))
        
        # Synchronize all streams
        for stream in self.streams:
            stream.sync()
        
        # Collect results from all GPUs
        for i, start_depth, end_depth, gpu_output in gpu_outputs:
            with cp.cuda.Device(i):
                # Copy this shard to the output array
                shard = cp.asnumpy(gpu_output)
                output[:, :, start_depth:end_depth] = shard
        
        # Transpose to match expected order (depth, height, width)
        return output.transpose(2, 1, 0)
    
    def calculate_3d_field_coherence_with_graph(self, fields: List[np.ndarray], use_tbc: bool = None) -> List[float]:
        """
        Calculate the coherence of multiple 3D quantum fields using CUDA Graphs.
        
        This optimized method is ideal for calculating coherence across multiple fields
        of the same dimensions, such as for animation frames or time-series data.
        It uses CUDA Graphs to minimize kernel launch overhead.
        
        Args:
            fields: List of 3D NumPy arrays, all with the same dimensions
            use_tbc: Whether to use Thread Block Clusters (None = automatic)
            
        Returns:
            List of coherence factors between 0.0 and 1.0
        """
        if not fields:
            return []
        
        if not self.initialized or not self.modules or not self.has_3d_capability:
            # Fall back to standard calculation
            return [self.calculate_3d_field_coherence(field) for field in fields]
        
        # All fields must have the same dimensions
        first_field = fields[0]
        field_shape = first_field.shape
        for i, field in enumerate(fields[1:], 1):
            if field.shape != field_shape:
                raise ValueError(f"Field {i} has shape {field.shape}, expected {field_shape}")
        
        depth, height, width = field_shape
        
        # Determine whether to use TBC
        if use_tbc is None:
            # Automatic selection: use TBC for large fields on supported hardware
            use_tbc = (
                self.has_thread_block_cluster_support and 
                width * height * depth >= 262_144  # 64x64x64 or larger
            )
        elif use_tbc and not self.has_thread_block_cluster_support:
            print("Thread Block Clusters requested but not supported, falling back to standard implementation")
            use_tbc = False
        
        # Create graph for this field size
        try:
            graph_data = self._create_3d_coherence_graph(width, height, depth, use_tbc=use_tbc)
            
            # Process all fields with the graph
            coherence_values = []
            for field in fields:
                coherence = self._execute_3d_coherence_graph(graph_data, field)
                coherence_values.append(coherence)
                
            return coherence_values
        except Exception as e:
            print(f"Error using CUDA graph for coherence calculation: {e}")
            # Fall back to standard calculation
            return [self.calculate_3d_field_coherence(field) for field in fields]
        
    def calculate_3d_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a 3D quantum field using CUDA
        
        Args:
            field_data: 3D NumPy array containing the field
            
        Returns:
            Coherence factor between 0.0 and 1.0
        """
        if not self.initialized or not self.modules or not self.has_3d_capability:
            # Fall back to CPU implementation (using the visualization3d module's function)
            from quantum_field.backends.cpu import CPUBackend
            cpu_backend = CPUBackend()
            cpu_backend.initialize()
            return cpu_backend.calculate_3d_field_coherence(field_data)
        
        # For small fields, use CPU implementation (less overhead)
        depth, height, width = field_data.shape
        if width * height * depth < 100_000:  # e.g., less than 50x50x40
            from quantum_field.visualization3d import calculate_3d_field_coherence
            return calculate_3d_field_coherence(field_data)
        
        # For large fields on H100+ GPUs, use Thread Block Clusters
        field_size = width * height * depth
        if self.has_thread_block_cluster_support and field_size >= 262_144:  # e.g., larger than 64x64x64
            try:
                return self._calculate_3d_field_coherence_tbc(field_data)
            except Exception as e:
                print(f"Thread Block Cluster coherence error: {e}")
                # Fall through to standard implementation
        
        # For large fields, try multi-GPU implementation if available
        if self.multi_gpu_available and field_size >= 1_048_576:  # 128x128x64 or larger
            try:
                # TODO: Implement _calculate_3d_field_coherence_multi_gpu
                pass  # For now, fall through to single-GPU implementation
            except Exception as e:
                print(f"Multi-GPU coherence error: {e}")
                # Fall through to standard implementation
        
        try:
            # Set the first device as current
            self._set_device(0)
            
            # Transpose to match CUDA kernel's indexing (width, height, depth)
            # cuda_field = field_data.transpose(2, 1, 0)
            
            # Copy field to GPU - making sure dimensions are in the right order
            d_field = cp.array(field_data.transpose(2, 1, 0))
            
            # Prepare reduction arrays
            num_blocks = 64  # Use more blocks for 3D for better sampling
            d_result = cp.zeros(num_blocks, dtype=cp.float32)
            d_count = cp.zeros(num_blocks, dtype=cp.int32)
            
            # Set up kernel launch parameters
            block_dim = (256, 1, 1)  # 256 threads per block
            grid_dim = (num_blocks, 1, 1)
            shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
            
            # Create launch config
            config = LaunchConfig(
                grid=grid_dim,
                block=block_dim,
                shmem_size=shmem_size
            )
            
            # Get the kernel
            kernel = self.modules[0].get_kernel("calculate_3d_field_coherence")
            
            # Launch the kernel
            launch(
                self.streams[0], 
                config, 
                kernel, 
                d_field.data.ptr, 
                width, 
                height, 
                depth,
                PHI, 
                LAMBDA,
                d_result.data.ptr,
                d_count.data.ptr
            )
            
            # Synchronize the stream
            self.streams[0].sync()
            
            # Copy results back to host
            h_result = cp.asnumpy(d_result)
            h_count = cp.asnumpy(d_count)
            
            # Calculate overall coherence
            total_samples = np.sum(h_count)
            if total_samples > 0:
                coherence = np.sum(h_result) / total_samples * PHI
                # Ensure result is in [0, 1] range
                coherence = max(0.0, min(1.0, coherence))
                return coherence
            
            return 0.0
        except Exception as e:
            print(f"CUDA 3D coherence error: {e}")
            # Fall back to CPU implementation
            from quantum_field.visualization3d import calculate_3d_field_coherence
            return calculate_3d_field_coherence(field_data)
    
    def to_dlpack(self, field_data: np.ndarray):
        """
        Convert a field to DLPack format for interoperability with ML frameworks
        
        Args:
            field_data: A 2D or 3D NumPy array containing the field data
            
        Returns:
            A DLPack tensor that can be imported into ML frameworks
        """
        if not self.initialized or not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend not initialized")
        
        try:
            # Check if input is already a CuPy array
            if isinstance(field_data, cp.ndarray):
                cupy_array = field_data
            else:
                # Convert NumPy array to CuPy array
                cupy_array = cp.array(field_data)
            
            # Convert to DLPack format
            dlpack_tensor = cupy_array.toDlpack()
            return dlpack_tensor
        except Exception as e:
            print(f"Error converting to DLPack: {e}")
            raise
    
    def from_dlpack(self, dlpack_tensor, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert a DLPack tensor to a quantum field array
        
        Args:
            dlpack_tensor: A DLPack tensor
            shape: Optional shape to reshape the tensor to (height, width)
            
        Returns:
            A 2D NumPy array containing the field data
        """
        if not self.initialized or not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend not initialized")
        
        try:
            # Convert DLPack to CuPy array
            cupy_array = cp.fromDlpack(dlpack_tensor)
            
            # Reshape if needed
            if shape is not None:
                cupy_array = cupy_array.reshape(shape)
            
            # Convert to NumPy and return
            return cp.asnumpy(cupy_array)
        except Exception as e:
            print(f"Error converting from DLPack: {e}")
            raise
    
    def shutdown(self) -> None:
        """Release resources used by this backend"""
        try:
            # Destroy all CUDA graphs
            for graph_name in list(self.graphs.keys()):
                self.destroy_cuda_graph(graph_name)
            
            # Clean up graph dictionaries
            self.graphs.clear()
            self.graph_execs.clear()
            
            # Clean up CUDA resources
            self.modules.clear()
            self.tbc_module = None
            self.devices.clear()
            self.streams.clear()
            
            # Clean up CuPy memory pool
            if 'cp' in globals() and cp is not None and hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                
        except Exception as e:
            print(f"Error shutting down CUDA backend: {e}")
        
        self.initialized = False