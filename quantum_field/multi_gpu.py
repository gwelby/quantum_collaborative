"""
Multi-GPU support for quantum field generation and analysis

This module provides functionality for distributing quantum field computations
across multiple GPUs for improved performance with large fields.
"""

import os
import numpy as np
import math
from typing import Tuple, Dict, Optional, List, Any, Union

# Import CUDA modules with fallback
try:
    from cuda.core.experimental import Device, Stream, Program, ProgramOptions
    from cuda.core.experimental import LaunchConfig, launch
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class MultiGPUManager:
    """
    Manager for multi-GPU computations
    Distributes work across available GPUs and combines results
    """
    
    def __init__(self):
        """Initialize the Multi-GPU Manager"""
        self.devices = []
        self.streams = []
        self.modules = {}
        self.available = False
        self.has_thread_block_cluster_support = False
        
        self._initialize_gpus()
        self._check_thread_block_cluster_support()
    
    def _initialize_gpus(self) -> None:
        """Initialize all available GPUs"""
        if not CUDA_AVAILABLE:
            return
        
        try:
            # Get the number of available devices
            num_devices = Device.get_device_count()
            
            if num_devices == 0:
                print("No CUDA devices found")
                return
            
            print(f"Found {num_devices} CUDA device(s)")
            
            # Initialize each device
            for i in range(num_devices):
                device = Device(i)
                info = device.get_name()
                print(f"  Device {i}: {info}")
                
                # Create a stream for this device
                device.set_current()
                stream = device.create_stream()
                
                self.devices.append(device)
                self.streams.append(stream)
            
            self.available = len(self.devices) > 0
            
            if self.available:
                print("Multi-GPU support initialized successfully")
                self._compile_kernels()
        except Exception as e:
            print(f"Error initializing Multi-GPU support: {e}")
            self.available = False
            
    def _check_thread_block_cluster_support(self) -> None:
        """Check if any of the available GPUs support Thread Block Clusters"""
        if not self.available or not self.devices:
            return
            
        try:
            # Thread Block Clusters are supported on Compute Capability 9.0+ (H100+)
            for i, device in enumerate(self.devices):
                cc_major, cc_minor = device.compute_capability
                if cc_major >= 9:
                    print(f"Device {i} supports Thread Block Clusters (Compute Capability {cc_major}.{cc_minor})")
                    self.has_thread_block_cluster_support = True
                else:
                    print(f"Device {i} does not support Thread Block Clusters (Compute Capability {cc_major}.{cc_minor})")
            
            if self.has_thread_block_cluster_support:
                print("Thread Block Cluster support is available for compatible operations")
                self._compile_tbc_kernels()
            else:
                print("Thread Block Clusters are not supported on any available GPU (requires Compute Capability 9.0+)")
        except Exception as e:
            print(f"Error checking Thread Block Cluster support: {e}")
            self.has_thread_block_cluster_support = False
            
    def _compile_tbc_kernels(self) -> None:
        """Compile Thread Block Cluster kernels for compatible GPUs"""
        if not self.available or not self.has_thread_block_cluster_support:
            return
            
        # Define kernel source with Thread Block Cluster support
        tbc_kernel_source = """
        template<typename T>
        __global__ void generate_3d_quantum_field_tbc(
            T *field, int width, int height, int depth,
            int start_depth, int end_depth,
            float frequency, float phi, float lambda, float time_factor
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
            int idz = blockIdx.z * (blockDim.z * 2) + block_offset_z + threadIdx.z + start_depth;
            
            if (idx < width && idy < height && idz < depth && idz >= start_depth && idz < end_depth) {
                // Calculate center coordinates
                float center_x = width / 2.0f;
                float center_y = height / 2.0f;
                float center_z = depth / 2.0f;
                
                // Calculate normalized coordinates
                float dx = (idx - center_x) / (width / 2.0f);
                float dy = (idy - center_y) / (height / 2.0f);
                float dz = (idz - center_z) / (depth / 2.0f);
                float distance = sqrtf(dx*dx + dy*dy + dz*dz);
                
                // Calculate spherical coordinates
                float r = distance * phi;
                float theta = acosf(dz / fmaxf(distance, 1e-6f));
                float phi_angle = atan2f(dy, dx);
                
                // Calculate field value using phi-harmonics
                float time_value = time_factor * lambda;
                float freq_factor = frequency / 1000.0f * phi;
                
                // Create 3D interference pattern with phi-harmonics
                float radial = sinf(r * freq_factor + time_value);
                float angular1 = cosf(theta * phi);
                float angular2 = cosf(phi_angle * phi * lambda);
                
                float value = radial * angular1 * angular2 * expf(-distance / phi);
                
                // Store the result (adjust index for the shard)
                field[idx + idy * width + (idz - start_depth) * width * height] = value;
            }
        }
        
        template<typename T>
        __global__ void calculate_3d_coherence_tbc(
            const T *field, int width, int height, int depth, 
            int start_depth, int end_depth,
            float phi, float lambda, float *result, int *count
        ) {
            // Use shared memory for reduction
            extern __shared__ float shared_mem[];
            float *alignment_sum = shared_mem;
            int *counter = (int*)(alignment_sum + blockDim.x);
            
            // Get block cluster ranks
            unsigned int cluster_rank_x = cooperative_groups::block_rank_in_cluster_x();
            unsigned int cluster_rank_y = cooperative_groups::block_rank_in_cluster_y();
            unsigned int cluster_rank_z = cooperative_groups::block_rank_in_cluster_z();
            
            // Calculate cluster index for the result array
            unsigned int cluster_idx = blockIdx.x * 8 + 
                                      cluster_rank_z * 4 + 
                                      cluster_rank_y * 2 + 
                                      cluster_rank_x;
            
            int tid = threadIdx.x;
            if (tid == 0) {
                counter[0] = 0;
            }
            __syncthreads();
            
            alignment_sum[tid] = 0.0f;
            
            // Use cluster id for more diverse sampling
            unsigned int seed = blockIdx.x * 1024 + 
                              cluster_rank_x * 32 + 
                              cluster_rank_y * 128 + 
                              cluster_rank_z * 512 + 
                              tid;
            
            // Each thread samples some random points in 3D space
            for (int i = 0; i < 8; i++) {  // More samples for 3D
                // Use a simple hash function to generate "random" coordinates
                int hash = seed * 1664525 + 1013904223 + i * 22695477;
                int x = hash % width;
                int y = (hash / width) % height;
                int z = start_depth + (hash / (width * height)) % (end_depth - start_depth);
                
                // Get field value at this point
                float value = field[x + y * width + (z - start_depth) * width * height];
                
                // Calculate phi-based coherence metrics
                float nearest_phi_multiple = roundf(value / phi);
                float deviation = fabsf(value - (nearest_phi_multiple * phi));
                float alignment = 1.0f - fminf(1.0f, deviation / (phi * 0.1f));
                
                // Calculate gradient around this point with boundary handling
                float gradient_sum = 0.0f;
                float curl_sum = 0.0f;
                
                // Compute gradients with proper boundary handling
                float dx = 0.0f;
                float dy = 0.0f;
                float dz = 0.0f;
                
                // X gradient with boundary handling
                if (x > 0 && x < width - 1) {
                    dx = (field[(x+1) + y*width + (z-start_depth)*width*height] - 
                          field[(x-1) + y*width + (z-start_depth)*width*height]) / 2.0f;
                } else if (x == 0) {
                    dx = field[1 + y*width + (z-start_depth)*width*height] - 
                         field[0 + y*width + (z-start_depth)*width*height];
                } else if (x == width - 1) {
                    dx = field[width-1 + y*width + (z-start_depth)*width*height] - 
                         field[width-2 + y*width + (z-start_depth)*width*height];
                }
                
                // Y gradient with boundary handling
                if (y > 0 && y < height - 1) {
                    dy = (field[x + (y+1)*width + (z-start_depth)*width*height] - 
                          field[x + (y-1)*width + (z-start_depth)*width*height]) / 2.0f;
                } else if (y == 0) {
                    dy = field[x + 1*width + (z-start_depth)*width*height] - 
                         field[x + 0*width + (z-start_depth)*width*height];
                } else if (y == height - 1) {
                    dy = field[x + (height-1)*width + (z-start_depth)*width*height] - 
                         field[x + (height-2)*width + (z-start_depth)*width*height];
                }
                
                // Z gradient with boundary handling for both shard and field boundaries
                int local_z = z - start_depth;  // z-index within the shard
                
                if (z > start_depth && z < end_depth - 1) {
                    // Regular central difference for internal points
                    dz = (field[x + y*width + (local_z+1)*width*height] - 
                          field[x + y*width + (local_z-1)*width*height]) / 2.0f;
                } else if (z == start_depth) {
                    // Forward difference at shard start
                    if (local_z + 1 < end_depth - start_depth) {
                        dz = field[x + y*width + (local_z+1)*width*height] - 
                             field[x + y*width + local_z*width*height];
                    }
                } else if (z == end_depth - 1) {
                    // Backward difference at shard end
                    if (local_z > 0) {
                        dz = field[x + y*width + local_z*width*height] - 
                             field[x + y*width + (local_z-1)*width*height];
                    }
                }
                
                // Apply phi-harmonics to gradient computation
                float phi_weighted_dx = dx * phi;
                float phi_weighted_dy = dy * phi;
                float phi_weighted_dz = dz * phi;
                
                // Gradient magnitude with phi weighting
                gradient_sum = sqrtf(phi_weighted_dx*phi_weighted_dx + 
                                    phi_weighted_dy*phi_weighted_dy + 
                                    phi_weighted_dz*phi_weighted_dz);
                
                // Calculate curl with phi weighting
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && 
                    z > start_depth && z < end_depth - 1) {
                    float curl_x = phi_weighted_dy - phi_weighted_dz;
                    float curl_y = phi_weighted_dz - phi_weighted_dx;
                    float curl_z = phi_weighted_dx - phi_weighted_dy;
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
            
            // Parallel reduction to sum alignments
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    alignment_sum[tid] += alignment_sum[tid + s];
                }
                __syncthreads();
            }
            
            // First thread writes the result
            if (tid == 0) {
                result[cluster_idx] = alignment_sum[0];
                count[cluster_idx] = counter[0];
            }
        }
        """
            
        try:
            # Compile kernels for each device that supports Thread Block Clusters
            tbc_modules = {}
            
            for i, device in enumerate(self.devices):
                cc_major, cc_minor = device.compute_capability
                
                # Skip devices that don't support Thread Block Clusters
                if cc_major < 9:
                    continue
                    
                device.set_current()
                
                # Get device architecture string
                arch_str = f"{cc_major}{cc_minor}"
                
                # Compile and link the kernel with thread block cluster support
                program_options = ProgramOptions(
                    std="c++17", 
                    arch=f"sm_{arch_str}",
                    # Enable cooperative groups for thread block clusters
                    flags=["--extended-lambda", "--use_fast_math", "--threads-per-thread-block=1024"]
                )
                program = Program(tbc_kernel_source, code_type="c++", options=program_options)
                
                # Compile with named expressions for template instantiations
                module = program.compile(
                    "cubin", 
                    name_expressions=[
                        "generate_3d_quantum_field_tbc<float>",
                        "calculate_3d_coherence_tbc<float>"
                    ]
                )
                
                # Store in a separate dictionary to avoid confusion with non-TBC modules
                tbc_modules[i] = module
                print(f"Thread Block Cluster kernels compiled successfully for device {i}")
            
            # Add TBC modules to the manager
            self.tbc_modules = tbc_modules
            
        except Exception as e:
            print(f"Error compiling Thread Block Cluster kernels: {e}")
            self.has_thread_block_cluster_support = False
    
    def _compile_kernels(self) -> None:
        """Compile kernels for each GPU"""
        if not self.available:
            return
        
        # Define the kernel source
        kernel_source = """
        template<typename T>
        __global__ void generate_quantum_field_shard(
            T *field, int width, int height, int start_row, int end_row,
            float frequency, float phi, float lambda, float time_factor
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y + start_row;
            
            if (idx < width && idy < height && idy >= start_row && idy < end_row) {
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
                
                // Store the result (adjust index for the shard)
                field[(idy - start_row) * width + idx] = value;
            }
        }
        
        template<typename T>
        __global__ void calculate_coherence_shard(
            const T *field, int width, int height, int start_row, int end_row,
            float phi, float *result, int *count
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
            
            // Each thread samples some random points in its shard
            for (int i = 0; i < 4; i++) {
                // Use a simple hash function to generate "random" coordinates
                int hash = (blockIdx.x * blockDim.x + tid) * 1664525 + 1013904223 + i * 22695477;
                int x = hash % width;
                int y = start_row + (hash / width) % (end_row - start_row);
                
                float value = field[(y - start_row) * width + x];
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
        
        template<typename T>
        __global__ void calculate_3d_coherence_shard(
            const T *field, int width, int height, int depth, 
            int start_depth, int end_depth,
            float phi, float lambda, float *result, int *count
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
            
            // Each thread samples some random points in its 3D shard
            for (int i = 0; i < 8; i++) {  // More samples for 3D field
                // Use a simple hash function to generate "random" coordinates
                int hash = (blockIdx.x * blockDim.x + tid) * 1664525 + 1013904223 + i * 22695477;
                int x = hash % width;
                int y = (hash / width) % height;
                int z = start_depth + (hash / (width * height)) % (end_depth - start_depth);
                
                // Get field value at this point (z is in shard range)
                float value = field[x + y * width + (z - start_depth) * width * height];
                
                // Calculate phi-based coherence metrics
                float nearest_phi_multiple = roundf(value / phi);
                float deviation = fabsf(value - (nearest_phi_multiple * phi));
                float alignment = 1.0f - fminf(1.0f, deviation / (phi * 0.1f));
                
                // Calculate gradient around this point with boundary handling
                float gradient_sum = 0.0f;
                float curl_sum = 0.0f;
                
                // Compute gradients with proper boundary handling
                // First handle x and y gradients which are within the same slice
                float dx = 0.0f;
                float dy = 0.0f;
                float dz = 0.0f;
                
                // X gradient with boundary handling
                if (x > 0 && x < width - 1) {
                    dx = (field[x+1 + y*width + (z-start_depth)*width*height] - 
                          field[x-1 + y*width + (z-start_depth)*width*height]) / 2.0f;
                } else if (x == 0) {
                    dx = field[1 + y*width + (z-start_depth)*width*height] - 
                         field[0 + y*width + (z-start_depth)*width*height];
                } else if (x == width - 1) {
                    dx = field[width-1 + y*width + (z-start_depth)*width*height] - 
                         field[width-2 + y*width + (z-start_depth)*width*height];
                }
                
                // Y gradient with boundary handling
                if (y > 0 && y < height - 1) {
                    dy = (field[x + (y+1)*width + (z-start_depth)*width*height] - 
                          field[x + (y-1)*width + (z-start_depth)*width*height]) / 2.0f;
                } else if (y == 0) {
                    dy = field[x + 1*width + (z-start_depth)*width*height] - 
                         field[x + 0*width + (z-start_depth)*width*height];
                } else if (y == height - 1) {
                    dy = field[x + (height-1)*width + (z-start_depth)*width*height] - 
                         field[x + (height-2)*width + (z-start_depth)*width*height];
                }
                
                // Z gradient with boundary handling for both shard and field boundaries
                int local_z = z - start_depth;  // z-index within the shard
                
                if (z > start_depth && z < end_depth - 1) {
                    // Regular central difference for internal points
                    dz = (field[x + y*width + (local_z+1)*width*height] - 
                          field[x + y*width + (local_z-1)*width*height]) / 2.0f;
                } else if (z == start_depth) {
                    // Forward difference at shard start
                    if (local_z + 1 < end_depth - start_depth) {
                        dz = field[x + y*width + (local_z+1)*width*height] - 
                             field[x + y*width + local_z*width*height];
                    }
                } else if (z == end_depth - 1) {
                    // Backward difference at shard end
                    if (local_z > 0) {
                        dz = field[x + y*width + local_z*width*height] - 
                             field[x + y*width + (local_z-1)*width*height];
                    }
                }
                
                // Apply phi-harmonics to gradient computation for more coherence-sensitive measure
                float phi_weighted_dx = dx * phi;
                float phi_weighted_dy = dy * phi;
                float phi_weighted_dz = dz * phi;
                
                // Gradient magnitude with phi weighting
                gradient_sum = sqrtf(phi_weighted_dx*phi_weighted_dx + 
                                    phi_weighted_dy*phi_weighted_dy + 
                                    phi_weighted_dz*phi_weighted_dz);
                
                // Calculate curl with phi weighting for more accurate field rotation measures
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && 
                    z > start_depth && z < end_depth - 1) {
                    float curl_x = phi_weighted_dy - phi_weighted_dz;
                    float curl_y = phi_weighted_dz - phi_weighted_dx;
                    float curl_z = phi_weighted_dx - phi_weighted_dy;
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
        
        template<typename T>
        __global__ void generate_3d_quantum_field_shard(
            T *field, int width, int height, int depth,
            int start_depth, int end_depth,
            float frequency, float phi, float lambda, float time_factor
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            int idz = blockIdx.z * blockDim.z + threadIdx.z + start_depth;
            
            if (idx < width && idy < height && idz < depth && idz >= start_depth && idz < end_depth) {
                // Calculate center coordinates
                float center_x = width / 2.0f;
                float center_y = height / 2.0f;
                float center_z = depth / 2.0f;
                
                // Calculate normalized coordinates
                float dx = (idx - center_x) / (width / 2.0f);
                float dy = (idy - center_y) / (height / 2.0f);
                float dz = (idz - center_z) / (depth / 2.0f);
                float distance = sqrtf(dx*dx + dy*dy + dz*dz);
                
                // Calculate spherical coordinates
                float r = distance * phi;
                float theta = acosf(dz / fmaxf(distance, 1e-6f));
                float phi_angle = atan2f(dy, dx);
                
                // Calculate field value using phi-harmonics
                float time_value = time_factor * lambda;
                float freq_factor = frequency / 1000.0f * phi;
                
                // Create 3D interference pattern with phi-harmonics
                float radial = sinf(r * freq_factor + time_value);
                float angular1 = cosf(theta * phi);
                float angular2 = cosf(phi_angle * phi * lambda);
                
                float value = radial * angular1 * angular2 * expf(-distance / phi);
                
                // Store the result (adjust index for the shard)
                // Using Depth, Height, Width ordering for field data
                field[idx + idy * width + (idz - start_depth) * width * height] = value;
            }
        }
        """
        
        try:
            # Compile kernels for each device
            for i, device in enumerate(self.devices):
                device.set_current()
                
                # Get device architecture string
                arch = device.compute_capability
                arch_str = "".join(f"{i}" for i in arch)
                
                # Compile and link the kernel
                program_options = ProgramOptions(std="c++17", arch=f"sm_{arch_str}")
                program = Program(kernel_source, code_type="c++", options=program_options)
                
                # Compile with named expressions for template instantiations
                module = program.compile(
                    "cubin", 
                    name_expressions=[
                        "generate_quantum_field_shard<float>",
                        "calculate_coherence_shard<float>",
                        "calculate_3d_coherence_shard<float>",
                        "generate_3d_quantum_field_shard<float>"
                    ]
                )
                
                self.modules[i] = module
            
            print("Multi-GPU kernels compiled successfully")
        except Exception as e:
            print(f"Error compiling Multi-GPU kernels: {e}")
            self.available = False
    
    def generate_quantum_field(self, width: int, height: int, frequency_name: str = 'love', 
                              time_factor: float = 0) -> np.ndarray:
        """
        Generate a quantum field distributed across multiple GPUs.
        
        Args:
            width: Width of the field
            height: Height of the field
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            
        Returns:
            A 2D NumPy array representing the quantum field
        """
        if not self.available or not self.devices:
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import generate_quantum_field
            return generate_quantum_field(width, height, frequency_name, time_factor)
        
        try:
            # Import constants
            from quantum_field.constants import PHI, LAMBDA, SACRED_FREQUENCIES
            
            # Get frequency value
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
            for i, (device, stream) in enumerate(zip(self.devices, self.streams)):
                device.set_current()
                
                # Calculate the rows for this GPU
                start_row = i * rows_per_gpu
                end_row = min(start_row + rows_per_gpu, height)
                shard_height = end_row - start_row
                
                if shard_height <= 0:
                    continue  # Skip if this GPU has no work
                
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
                    kernel = self.modules[i].get_kernel("generate_quantum_field_shard<float>")
                    
                    # Launch the kernel
                    launch(
                        stream, 
                        config, 
                        kernel, 
                        gpu_output.data.ptr, 
                        width, 
                        height, 
                        start_row,
                        end_row,
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
        except Exception as e:
            print(f"Error in Multi-GPU computation: {e}")
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import generate_quantum_field
            return generate_quantum_field(width, height, frequency_name, time_factor)
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field using multiple GPUs.
        
        Args:
            field_data: A 2D NumPy array containing the field data
            
        Returns:
            A float representing the field coherence
        """
        if not self.available or not self.devices:
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import calculate_field_coherence
            return calculate_field_coherence(field_data)
        
        try:
            # Import constants
            from quantum_field.constants import PHI
            
            height, width = field_data.shape
            
            # Divide the field among GPUs (row-wise partitioning)
            num_gpus = len(self.devices)
            rows_per_gpu = (height + num_gpus - 1) // num_gpus
            
            # Prepare arrays to collect results from all GPUs
            all_results = []
            all_counts = []
            
            # Process each shard on a different GPU
            for i, (device, stream) in enumerate(zip(self.devices, self.streams)):
                device.set_current()
                
                # Calculate the rows for this GPU
                start_row = i * rows_per_gpu
                end_row = min(start_row + rows_per_gpu, height)
                shard_height = end_row - start_row
                
                if shard_height <= 0:
                    continue  # Skip if this GPU has no work
                
                # Extract the shard for this GPU
                shard = field_data[start_row:end_row, :]
                
                with cp.cuda.Device(i):
                    # Copy shard to GPU
                    d_shard = cp.array(shard)
                    
                    # Prepare reduction arrays
                    num_blocks = 32  # Use 32 blocks for reduction
                    d_result = cp.zeros(num_blocks, dtype=cp.float32)
                    d_count = cp.zeros(num_blocks, dtype=cp.int32)
                    
                    # Set up kernel launch parameters
                    block_dim = 256  # 256 threads per block
                    shmem_size = block_dim * 4 + 4  # float alignment_sum[256] + int counter[1]
                    
                    # Create launch config
                    config = LaunchConfig(
                        grid=(num_blocks, 1, 1),
                        block=(block_dim, 1, 1),
                        shmem_size=shmem_size
                    )
                    
                    # Get the kernel
                    kernel = self.modules[i].get_kernel("calculate_coherence_shard<float>")
                    
                    # Launch the kernel
                    launch(
                        stream, 
                        config, 
                        kernel, 
                        d_shard.data.ptr, 
                        width, 
                        height, 
                        start_row,
                        end_row,
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
            from quantum_field.core import calculate_field_coherence_cpu
            return calculate_field_coherence_cpu(field_data)
            
        except Exception as e:
            print(f"Error in Multi-GPU coherence calculation: {e}")
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import calculate_field_coherence
            return calculate_field_coherence(field_data)
            
    def generate_3d_quantum_field(self, width: int, height: int, depth: int, 
                                frequency_name: str = 'love', time_factor: float = 0) -> np.ndarray:
        """
        Generate a 3D quantum field distributed across multiple GPUs.
        
        Args:
            width: Width of the field (x-dimension)
            height: Height of the field (y-dimension)
            depth: Depth of the field (z-dimension)
            frequency_name: The sacred frequency to use
            time_factor: Time factor for animation
            
        Returns:
            A 3D NumPy array representing the quantum field with shape (depth, height, width)
        """
        if not self.available or not self.devices:
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import generate_3d_quantum_field
            return generate_3d_quantum_field(width, height, depth, frequency_name, time_factor)
        
        try:
            # Import constants
            from quantum_field.constants import PHI, LAMBDA, SACRED_FREQUENCIES
            
            # Get frequency value
            frequency = SACRED_FREQUENCIES.get(frequency_name, 528)
            
            # Create output array (depth, height, width)
            output = np.zeros((depth, height, width), dtype=np.float32)
            
            # Divide the field among GPUs (depth-wise partitioning)
            num_gpus = len(self.devices)
            depths_per_gpu = (depth + num_gpus - 1) // num_gpus
            
            # Set block dimensions for each GPU
            # Use 3D blocks for better performance
            block_dim = (8, 8, 4)
            
            # List to store output arrays from each GPU
            gpu_outputs = []
            
            # Launch kernels on each GPU
            for i, (device, stream) in enumerate(zip(self.devices, self.streams)):
                device.set_current()
                
                # Calculate the depth range for this GPU
                start_depth = i * depths_per_gpu
                end_depth = min(start_depth + depths_per_gpu, depth)
                shard_depth = end_depth - start_depth
                
                if shard_depth <= 0:
                    continue  # Skip if this GPU has no work
                
                # Create a CuPy array for this GPU's output
                # Use (width, height, shard_depth) layout for CUDA memory efficiency
                with cp.cuda.Device(i):
                    gpu_output = cp.empty((shard_depth, height, width), dtype=cp.float32)
                    
                    # Set up grid dimensions
                    grid_dim = (
                        (width + block_dim[0] - 1) // block_dim[0],
                        (height + block_dim[1] - 1) // block_dim[1],
                        (shard_depth + block_dim[2] - 1) // block_dim[2]
                    )
                    
                    # Create launch config
                    config = LaunchConfig(grid=grid_dim, block=block_dim)
                    
                    # Get the kernel
                    kernel = self.modules[i].get_kernel("generate_3d_quantum_field_shard<float>")
                    
                    # Launch the kernel
                    launch(
                        stream, 
                        config, 
                        kernel, 
                        gpu_output.data.ptr, 
                        width, 
                        height, 
                        depth,
                        start_depth,
                        end_depth,
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
                    # Transpose back to depth, height, width ordering
                    shard = cp.asnumpy(gpu_output)
                    output[start_depth:end_depth, :, :] = shard
            
            return output
        except Exception as e:
            print(f"Error in Multi-GPU 3D field generation: {e}")
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import generate_3d_quantum_field
            return generate_3d_quantum_field(width, height, depth, frequency_name, time_factor)
    
    def calculate_3d_field_coherence(self, field_data: np.ndarray, use_tiling: bool = True, 
                                  tile_size: Optional[Tuple[int, int, int]] = None,
                                  use_thread_block_clusters: bool = True) -> float:
        """
        Calculate the coherence of a 3D quantum field using multiple GPUs.
        
        Args:
            field_data: A 3D NumPy array containing the field data with shape (depth, height, width)
            use_tiling: Whether to use tiling for very large fields
            tile_size: Optional custom tile size as (depth, height, width)
            use_thread_block_clusters: Whether to use Thread Block Clusters if available
            
        Returns:
            A float representing the field coherence
        """
        if not self.available or not self.devices:
            # Fall back to single-GPU or CPU implementation
            from quantum_field.core import calculate_3d_field_coherence
            return calculate_3d_field_coherence(field_data)
        
        # Check if we should use Thread Block Clusters
        use_tbc = use_thread_block_clusters and self.has_thread_block_cluster_support
        
        if use_tbc:
            try:
                # Try to use Thread Block Clusters first
                return self._calculate_3d_field_coherence_tbc(field_data, use_tiling, tile_size)
            except Exception as e:
                print(f"Error in Thread Block Cluster coherence calculation, falling back to standard method: {e}")
                # Fall back to standard implementation
                use_tbc = False
        
        if not use_tbc:
            try:
                # Import constants
                from quantum_field.constants import PHI, LAMBDA
                
                # Get field dimensions (depth, height, width)
                depth, height, width = field_data.shape
                
                # Check if field is too large for GPU memory and needs tiling
                field_size_bytes = depth * height * width * 4  # 4 bytes per float32
                
                # Determine if we should use tiling
                use_memory_tiling = use_tiling and field_size_bytes > 4 * 1024 * 1024 * 1024  # >4GB
                
                if use_memory_tiling:
                    # Set default tile size if not provided
                    if tile_size is None:
                        # Default to reasonable tile sizes that fit in memory
                        # These can be tuned for specific hardware
                        tile_depth = min(depth, 32)  
                        tile_height = min(height, 256)
                        tile_width = min(width, 256)
                        tile_size = (tile_depth, tile_height, tile_width)
                    else:
                        tile_depth, tile_height, tile_width = tile_size
                    
                    # Process field in tiles
                    all_results = []
                    all_counts = []
                    
                    # Divide the field into tiles and process each one
                    for d_start in range(0, depth, tile_depth):
                        d_end = min(d_start + tile_depth, depth)
                        
                        for h_start in range(0, height, tile_height):
                            h_end = min(h_start + tile_height, height)
                            
                            for w_start in range(0, width, tile_width):
                                w_end = min(w_start + tile_width, width)
                                
                                # Extract the tile
                                tile = field_data[d_start:d_end, h_start:h_end, w_start:w_end]
                                
                                # Process the tile
                                tile_results, tile_counts = self._process_3d_field_tile(
                                    tile, d_start, h_start, w_start
                                )
                                
                                # Collect results
                                all_results.extend(tile_results)
                                all_counts.extend(tile_counts)
                    
                    # Combine results from all tiles
                    combined_results = np.concatenate(all_results)
                    combined_counts = np.concatenate(all_counts)
                else:
                    # Process the entire field at once
                    all_results, all_counts = self._process_3d_field_full(field_data)
                    
                    # Combine results
                    combined_results = np.concatenate(all_results)
                    combined_counts = np.concatenate(all_counts)
                
                # Calculate final coherence value
                total_samples = np.sum(combined_counts)
                if total_samples > 0:
                    coherence = np.sum(combined_results) / total_samples * PHI
                    # Ensure result is in [0, 1] range
                    coherence = max(0.0, min(1.0, coherence))
                    return coherence
                
                # If no valid results, fall back to CPU implementation
                from quantum_field.backends.cpu import CPUBackend
                cpu_backend = CPUBackend()
                cpu_backend.initialize()
                return cpu_backend.calculate_3d_field_coherence(field_data)
                
            except Exception as e:
                print(f"Error in Multi-GPU 3D coherence calculation: {e}")
                # Fall back to single-GPU or CPU implementation
                from quantum_field.core import calculate_3d_field_coherence
                return calculate_3d_field_coherence(field_data)
                
    def _calculate_3d_field_coherence_tbc(self, field_data: np.ndarray, use_tiling: bool = True,
                                     tile_size: Optional[Tuple[int, int, int]] = None) -> float:
        """
        Calculate 3D field coherence using Thread Block Clusters.
        
        Args:
            field_data: A 3D NumPy array containing the field data with shape (depth, height, width)
            use_tiling: Whether to use tiling for very large fields
            tile_size: Optional custom tile size as (depth, height, width)
            
        Returns:
            A float representing the field coherence
        """
        if not self.has_thread_block_cluster_support:
            raise RuntimeError("Thread Block Clusters are not supported on this hardware")
            
        # Find devices that support Thread Block Clusters
        tbc_devices = []
        tbc_streams = []
        tbc_device_indices = []
        
        for i, device in enumerate(self.devices):
            cc_major, cc_minor = device.compute_capability
            if cc_major >= 9 and i in self.tbc_modules:
                tbc_devices.append(device)
                tbc_streams.append(self.streams[i])
                tbc_device_indices.append(i)
        
        if not tbc_devices:
            raise RuntimeError("No devices with Thread Block Cluster support found")
            
        # Import constants
        from quantum_field.constants import PHI, LAMBDA
        
        # Get field dimensions (depth, height, width)
        depth, height, width = field_data.shape
        
        # Check if field is too large for GPU memory and needs tiling
        field_size_bytes = depth * height * width * 4  # 4 bytes per float32
        
        # Determine if we should use tiling
        use_memory_tiling = use_tiling and field_size_bytes > 4 * 1024 * 1024 * 1024  # >4GB
        
        # Process field using Thread Block Clusters
        all_results = []
        all_counts = []
        
        if use_memory_tiling:
            # Set default tile size if not provided
            if tile_size is None:
                # Default to reasonable tile sizes that fit in memory
                tile_depth = min(depth, 32)  
                tile_height = min(height, 256)
                tile_width = min(width, 256)
                tile_size = (tile_depth, tile_height, tile_width)
            else:
                tile_depth, tile_height, tile_width = tile_size
            
            # Divide the field into tiles and process each one
            for d_start in range(0, depth, tile_depth):
                d_end = min(d_start + tile_depth, depth)
                
                for h_start in range(0, height, tile_height):
                    h_end = min(h_start + tile_height, height)
                    
                    for w_start in range(0, width, tile_width):
                        w_end = min(w_start + tile_width, width)
                        
                        # Extract the tile
                        tile = field_data[d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        # Process the tile using Thread Block Clusters
                        tile_results, tile_counts = self._process_3d_field_tile_tbc(
                            tile, d_start, h_start, w_start, tbc_devices, tbc_streams, tbc_device_indices
                        )
                        
                        # Collect results
                        all_results.extend(tile_results)
                        all_counts.extend(tile_counts)
        else:
            # Process the entire field at once using Thread Block Clusters
            all_results, all_counts = self._process_3d_field_full_tbc(
                field_data, tbc_devices, tbc_streams, tbc_device_indices
            )
        
        # Combine results
        combined_results = np.concatenate(all_results)
        combined_counts = np.concatenate(all_counts)
        
        # Calculate final coherence value
        total_samples = np.sum(combined_counts)
        if total_samples > 0:
            coherence = np.sum(combined_results) / total_samples * PHI
            # Ensure result is in [0, 1] range
            coherence = max(0.0, min(1.0, coherence))
            return coherence
        
        # If no valid results, fall back to non-TBC implementation
        print("No valid results from TBC implementation, falling back to standard method")
        return self.calculate_3d_field_coherence(field_data, use_tiling, tile_size, use_thread_block_clusters=False)
    
    def _process_3d_field_tile(self, tile_data: np.ndarray, 
                              d_offset: int = 0, h_offset: int = 0, w_offset: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process a 3D field tile using multiple GPUs.
        
        Args:
            tile_data: A 3D NumPy array containing the tile data
            d_offset: Depth offset of the tile in the original field
            h_offset: Height offset of the tile in the original field
            w_offset: Width offset of the tile in the original field
            
        Returns:
            Tuple of (results, counts) lists
        """
        # Import constants
        from quantum_field.constants import PHI, LAMBDA
        
        # Get tile dimensions
        tile_depth, tile_height, tile_width = tile_data.shape
        
        # Divide the tile among GPUs (depth-wise partitioning)
        num_gpus = len(self.devices)
        depths_per_gpu = (tile_depth + num_gpus - 1) // num_gpus
        
        # Prepare arrays to collect results
        all_results = []
        all_counts = []
        
        # Transpose for CUDA memory layout efficiency
        # From (depth, height, width) to (width, height, depth)
        cuda_tile = tile_data.transpose(2, 1, 0)
        
        # Process each shard on a different GPU
        for i, (device, stream) in enumerate(zip(self.devices, self.streams)):
            device.set_current()
            
            # Calculate the depth range for this GPU
            start_depth = i * depths_per_gpu
            end_depth = min(start_depth + depths_per_gpu, tile_depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Extract the shard for this GPU from the transposed data
            # This gives us a block with shape (width, height, shard_depth)
            shard = cuda_tile[:, :, start_depth:end_depth]
            
            with cp.cuda.Device(i):
                # Copy shard to GPU
                d_shard = cp.array(shard)
                
                # Prepare reduction arrays
                num_blocks = 64  # Use more blocks for 3D fields
                d_result = cp.zeros(num_blocks, dtype=cp.float32)
                d_count = cp.zeros(num_blocks, dtype=cp.int32)
                
                # Set up kernel launch parameters
                block_dim = 256  # 256 threads per block
                shmem_size = block_dim * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config
                config = LaunchConfig(
                    grid=(num_blocks, 1, 1),
                    block=(block_dim, 1, 1),
                    shmem_size=shmem_size
                )
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("calculate_3d_coherence_shard<float>")
                
                # Launch the kernel with global offsets for correct gradient calculation
                launch(
                    stream, 
                    config, 
                    kernel, 
                    d_shard.data.ptr, 
                    tile_width, 
                    tile_height, 
                    tile_depth,
                    start_depth + d_offset,  # Add global offsets
                    end_depth + d_offset,
                    PHI,
                    LAMBDA,
                    d_result.data.ptr,
                    d_count.data.ptr
                )
                
                # Collect results
                all_results.append(cp.asnumpy(d_result))
                all_counts.append(cp.asnumpy(d_count))
        
        # Synchronize all streams
        for stream in self.streams:
            stream.sync()
        
        return all_results, all_counts
    
    def _process_3d_field_full(self, field_data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process the complete 3D field using multiple GPUs.
        
        Args:
            field_data: A 3D NumPy array containing the field data
            
        Returns:
            Tuple of (results, counts) lists
        """
        # Import constants
        from quantum_field.constants import PHI, LAMBDA
        
        # Get field dimensions
        depth, height, width = field_data.shape
        
        # Divide the field among GPUs (depth-wise partitioning)
        num_gpus = len(self.devices)
        depths_per_gpu = (depth + num_gpus - 1) // num_gpus
        
        # Prepare arrays to collect results
        all_results = []
        all_counts = []
        
        # Transpose for CUDA memory layout efficiency
        # From (depth, height, width) to (width, height, depth)
        cuda_field = field_data.transpose(2, 1, 0)
        
        # Process each shard on a different GPU
        for i, (device, stream) in enumerate(zip(self.devices, self.streams)):
            device.set_current()
            
            # Calculate the depth range for this GPU
            start_depth = i * depths_per_gpu
            end_depth = min(start_depth + depths_per_gpu, depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Extract the shard for this GPU from the transposed data
            # This gives us a block with shape (width, height, shard_depth)
            shard = cuda_field[:, :, start_depth:end_depth]
            
            with cp.cuda.Device(i):
                # Copy shard to GPU
                d_shard = cp.array(shard)
                
                # Prepare reduction arrays
                num_blocks = 64  # Use more blocks for 3D fields
                d_result = cp.zeros(num_blocks, dtype=cp.float32)
                d_count = cp.zeros(num_blocks, dtype=cp.int32)
                
                # Set up kernel launch parameters
                block_dim = 256  # 256 threads per block
                shmem_size = block_dim * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config
                config = LaunchConfig(
                    grid=(num_blocks, 1, 1),
                    block=(block_dim, 1, 1),
                    shmem_size=shmem_size
                )
                
                # Get the kernel
                kernel = self.modules[i].get_kernel("calculate_3d_coherence_shard<float>")
                
                # Launch the kernel
                launch(
                    stream, 
                    config, 
                    kernel, 
                    d_shard.data.ptr, 
                    width, 
                    height, 
                    depth,
                    start_depth,
                    end_depth,
                    PHI,
                    LAMBDA,
                    d_result.data.ptr,
                    d_count.data.ptr
                )
                
                # Collect results
                all_results.append(cp.asnumpy(d_result))
                all_counts.append(cp.asnumpy(d_count))
        
        # Synchronize all streams
        for stream in self.streams:
            stream.sync()
        
        return all_results, all_counts
        
    def _process_3d_field_tile_tbc(self, tile_data: np.ndarray,
                               d_offset: int, h_offset: int, w_offset: int,
                               tbc_devices: List, tbc_streams: List, tbc_device_indices: List) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process a 3D field tile using Thread Block Clusters.
        
        Args:
            tile_data: A 3D NumPy array containing the tile data
            d_offset: Depth offset of the tile in the original field
            h_offset: Height offset of the tile in the original field
            w_offset: Width offset of the tile in the original field
            tbc_devices: List of devices that support Thread Block Clusters
            tbc_streams: List of streams for TBC-capable devices
            tbc_device_indices: List of device indices for TBC-capable devices
            
        Returns:
            Tuple of (results, counts) lists
        """
        # Import constants
        from quantum_field.constants import PHI, LAMBDA
        
        # Get tile dimensions
        tile_depth, tile_height, tile_width = tile_data.shape
        
        # Divide the tile among TBC-capable GPUs (depth-wise partitioning)
        num_tbc_gpus = len(tbc_devices)
        depths_per_gpu = (tile_depth + num_tbc_gpus - 1) // num_tbc_gpus
        
        # Prepare arrays to collect results
        all_results = []
        all_counts = []
        
        # Transpose for CUDA memory layout efficiency
        # From (depth, height, width) to (width, height, depth)
        cuda_tile = tile_data.transpose(2, 1, 0)
        
        # Process each shard on a different TBC-capable GPU
        for j, (device, stream, i) in enumerate(zip(tbc_devices, tbc_streams, tbc_device_indices)):
            device.set_current()
            
            # Calculate the depth range for this GPU
            start_depth = j * depths_per_gpu
            end_depth = min(start_depth + depths_per_gpu, tile_depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Extract the shard for this GPU from the transposed data
            # This gives us a block with shape (width, height, shard_depth)
            shard = cuda_tile[:, :, start_depth:end_depth]
            
            with cp.cuda.Device(i):
                # Copy shard to GPU
                d_shard = cp.array(shard)
                
                # Prepare reduction arrays - 8x more results for 2x2x2 Thread Block Clusters
                num_blocks = 32  # 32 grid blocks
                clusters_per_block = 8  # 2x2x2 = 8 clusters per block
                d_result = cp.zeros(num_blocks * clusters_per_block, dtype=cp.float32)
                d_count = cp.zeros(num_blocks * clusters_per_block, dtype=cp.int32)
                
                # Set up kernel launch parameters for Thread Block Clusters
                block_dim = (256, 1, 1)  # 256 threads per block (1D)
                cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster (3D)
                shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config for Thread Block Clusters
                config = LaunchConfig(
                    grid=(num_blocks, 1, 1),
                    block=block_dim,
                    cluster_shape=cluster_shape,  # 2x2x2 blocks per cluster
                    shmem_size=shmem_size
                )
                
                # Get the TBC kernel
                kernel = self.tbc_modules[i].get_kernel("calculate_3d_coherence_tbc<float>")
                
                # Add the global depth offset to start/end depths
                global_start_depth = d_offset + start_depth
                global_end_depth = d_offset + end_depth
                
                # Launch the kernel with Thread Block Clusters
                launch(
                    stream, 
                    config, 
                    kernel, 
                    d_shard.data.ptr, 
                    tile_width, 
                    tile_height, 
                    tile_depth,
                    start_depth,  # Local start depth within the tile
                    end_depth,    # Local end depth within the tile
                    PHI,
                    LAMBDA,
                    d_result.data.ptr,
                    d_count.data.ptr
                )
                
                # Collect results
                all_results.append(cp.asnumpy(d_result))
                all_counts.append(cp.asnumpy(d_count))
        
        # Synchronize all streams
        for stream in tbc_streams:
            stream.sync()
        
        return all_results, all_counts
    
    def _process_3d_field_full_tbc(self, field_data: np.ndarray,
                               tbc_devices: List, tbc_streams: List, tbc_device_indices: List) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process the complete 3D field using Thread Block Clusters.
        
        Args:
            field_data: A 3D NumPy array containing the field data
            tbc_devices: List of devices that support Thread Block Clusters
            tbc_streams: List of streams for TBC-capable devices
            tbc_device_indices: List of device indices for TBC-capable devices
            
        Returns:
            Tuple of (results, counts) lists
        """
        # Import constants
        from quantum_field.constants import PHI, LAMBDA
        
        # Get field dimensions
        depth, height, width = field_data.shape
        
        # Divide the field among TBC-capable GPUs (depth-wise partitioning)
        num_tbc_gpus = len(tbc_devices)
        depths_per_gpu = (depth + num_tbc_gpus - 1) // num_tbc_gpus
        
        # Prepare arrays to collect results
        all_results = []
        all_counts = []
        
        # Transpose for CUDA memory layout efficiency
        # From (depth, height, width) to (width, height, depth)
        cuda_field = field_data.transpose(2, 1, 0)
        
        # Process each shard on a different TBC-capable GPU
        for j, (device, stream, i) in enumerate(zip(tbc_devices, tbc_streams, tbc_device_indices)):
            device.set_current()
            
            # Calculate the depth range for this GPU
            start_depth = j * depths_per_gpu
            end_depth = min(start_depth + depths_per_gpu, depth)
            shard_depth = end_depth - start_depth
            
            if shard_depth <= 0:
                continue  # Skip if this GPU has no work
            
            # Extract the shard for this GPU from the transposed data
            # This gives us a block with shape (width, height, shard_depth)
            shard = cuda_field[:, :, start_depth:end_depth]
            
            with cp.cuda.Device(i):
                # Copy shard to GPU
                d_shard = cp.array(shard)
                
                # Prepare reduction arrays - 8x more results for 2x2x2 Thread Block Clusters
                num_blocks = 32  # 32 grid blocks
                clusters_per_block = 8  # 2x2x2 = 8 clusters per block
                d_result = cp.zeros(num_blocks * clusters_per_block, dtype=cp.float32)
                d_count = cp.zeros(num_blocks * clusters_per_block, dtype=cp.int32)
                
                # Set up kernel launch parameters for Thread Block Clusters
                block_dim = (256, 1, 1)  # 256 threads per block (1D)
                cluster_shape = (2, 2, 2)  # 2x2x2 blocks per cluster (3D)
                shmem_size = block_dim[0] * 4 + 4  # float alignment_sum[256] + int counter[1]
                
                # Create launch config for Thread Block Clusters
                config = LaunchConfig(
                    grid=(num_blocks, 1, 1),
                    block=block_dim,
                    cluster_shape=cluster_shape,  # 2x2x2 blocks per cluster
                    shmem_size=shmem_size
                )
                
                # Get the TBC kernel
                kernel = self.tbc_modules[i].get_kernel("calculate_3d_coherence_tbc<float>")
                
                # Launch the kernel with Thread Block Clusters
                launch(
                    stream, 
                    config, 
                    kernel, 
                    d_shard.data.ptr, 
                    width, 
                    height, 
                    depth,
                    start_depth,
                    end_depth,
                    PHI,
                    LAMBDA,
                    d_result.data.ptr,
                    d_count.data.ptr
                )
                
                # Collect results
                all_results.append(cp.asnumpy(d_result))
                all_counts.append(cp.asnumpy(d_count))
        
        # Synchronize all streams
        for stream in tbc_streams:
            stream.sync()
        
        return all_results, all_counts
    
    def benchmark_multi_gpu(self, max_size: int = 4096) -> Dict[str, Any]:
        """
        Benchmark multi-GPU performance against single-GPU implementation.
        
        Args:
            max_size: Maximum field dimension to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.available or len(self.devices) < 2:
            print("Multi-GPU benchmarking requires at least 2 GPUs")
            return {"error": "Insufficient GPUs available"}
        
        # Import needed modules
        import time
        from quantum_field.core import generate_quantum_field
        
        results = {
            "num_gpus": len(self.devices),
            "sizes": [],
            "single_gpu_times": [],
            "multi_gpu_times": [],
            "speedups": []
        }
        
        # Define test sizes (square fields)
        test_sizes = [
            (1024, 1024), 
            (2048, 2048)
        ]
        
        # Add max size if not already included
        if max_size > 2048:
            test_sizes.append((max_size, max_size))
        
        iterations = 3
        
        for width, height in test_sizes:
            print(f"\nBenchmarking size: {width}x{height}")
            results["sizes"].append(f"{width}x{height}")
            
            # Single-GPU implementation
            single_gpu_times = []
            for i in range(iterations):
                start_time = time.time()
                _ = generate_quantum_field(width, height, 'love')
                end_time = time.time()
                single_gpu_times.append(end_time - start_time)
                print(f"  Single-GPU iteration {i+1}/{iterations}: {single_gpu_times[-1]:.4f} seconds")
            
            avg_single_time = sum(single_gpu_times) / len(single_gpu_times)
            results["single_gpu_times"].append(avg_single_time)
            
            # Multi-GPU implementation
            multi_gpu_times = []
            for i in range(iterations):
                start_time = time.time()
                _ = self.generate_quantum_field(width, height, 'love')
                end_time = time.time()
                multi_gpu_times.append(end_time - start_time)
                print(f"  Multi-GPU iteration {i+1}/{iterations}: {multi_gpu_times[-1]:.4f} seconds")
            
            avg_multi_time = sum(multi_gpu_times) / len(multi_gpu_times)
            results["multi_gpu_times"].append(avg_multi_time)
            
            # Calculate speedup
            if avg_single_time > 0:
                speedup = avg_single_time / avg_multi_time
                results["speedups"].append(speedup)
                print(f"  Speedup: {speedup:.2f}x")
            else:
                results["speedups"].append(0)
                print("  Could not calculate speedup (single GPU time is zero)")
        
        return results
    
    def benchmark_3d_multi_gpu(self, max_size: int = 128, custom_sizes: List[Tuple[int, int, int]] = None, 
                           test_coherence: bool = True) -> Dict[str, Any]:
        """
        Benchmark 3D multi-GPU performance against single-GPU implementation.
        
        Args:
            max_size: Maximum field dimension to test for cubic fields
            custom_sizes: Optional list of custom field sizes to test as [(width, height, depth), ...]
            test_coherence: Whether to benchmark coherence calculation (in addition to field generation)
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.available or len(self.devices) < 2:
            print("Multi-GPU benchmarking requires at least 2 GPUs")
            return {"error": "Insufficient GPUs available"}
        
        # Import needed modules
        import time
        from quantum_field.backends.cuda import CUDABackend
        
        # Initialize CUDA backend for comparison
        cuda_backend = CUDABackend()
        cuda_backend.initialize()
        
        results = {
            "num_gpus": len(self.devices),
            "sizes": [],
            "single_gpu_gen_times": [],
            "multi_gpu_gen_times": [],
            "gen_speedups": [],
            "single_gpu_coh_times": [],
            "multi_gpu_coh_times": [],
            "coh_speedups": []
        }
        
        # Define test sizes (cubic and non-cubic fields)
        if custom_sizes:
            test_sizes = custom_sizes
        else:
            # Start with standard test sizes
            test_sizes = [
                (32, 32, 32),     # Small cubic field
                (64, 64, 64),     # Medium cubic field
                (128, 96, 64),    # Non-cubic field (good for real-world scenarios)
                (64, 128, 32)     # Wide but shallow field
            ]
            
            # Add max size if not already included and it's reasonable
            if max_size > 64 and max_size <= 256:  # Limit to reasonable sizes
                test_sizes.append((max_size, max_size, max_size))
        
        iterations = 3
        
        for width, height, depth in test_sizes:
            print(f"\nBenchmarking 3D size: {width}x{height}x{depth}")
            results["sizes"].append(f"{width}x{height}x{depth}")
            
            # Generate test data for coherence calculation (if needed)
            test_field = None
            if test_coherence:
                try:
                    print("  Generating test field...")
                    test_field = self.generate_3d_quantum_field(width, height, depth, 'love')
                    print("  Test field generated successfully")
                except Exception as e:
                    print(f"  Error generating test field: {e}")
                    test_coherence = False  # Skip coherence tests if generation fails
            
            # Field generation benchmark
            # Single-GPU implementation
            single_gpu_gen_times = []
            for i in range(iterations):
                try:
                    start_time = time.time()
                    _ = cuda_backend.generate_3d_quantum_field(width, height, depth, 'love')
                    end_time = time.time()
                    single_gpu_gen_times.append(end_time - start_time)
                    print(f"  Single-GPU generation {i+1}/{iterations}: {single_gpu_gen_times[-1]:.4f} seconds")
                except Exception as e:
                    print(f"  Error in single-GPU generation benchmark: {e}")
                    break
            
            if single_gpu_gen_times:
                avg_single_gen_time = sum(single_gpu_gen_times) / len(single_gpu_gen_times)
                results["single_gpu_gen_times"].append(avg_single_gen_time)
            else:
                results["single_gpu_gen_times"].append(None)
                print("  Single-GPU generation benchmark failed, possibly due to memory constraints")
            
            # Multi-GPU implementation
            multi_gpu_gen_times = []
            for i in range(iterations):
                try:
                    start_time = time.time()
                    _ = self.generate_3d_quantum_field(width, height, depth, 'love')
                    end_time = time.time()
                    multi_gpu_gen_times.append(end_time - start_time)
                    print(f"  Multi-GPU generation {i+1}/{iterations}: {multi_gpu_gen_times[-1]:.4f} seconds")
                except Exception as e:
                    print(f"  Error in multi-GPU generation benchmark: {e}")
                    break
            
            if multi_gpu_gen_times:
                avg_multi_gen_time = sum(multi_gpu_gen_times) / len(multi_gpu_gen_times)
                results["multi_gpu_gen_times"].append(avg_multi_gen_time)
            else:
                results["multi_gpu_gen_times"].append(None)
                print("  Multi-GPU generation benchmark failed")
            
            # Calculate generation speedup
            if results["single_gpu_gen_times"][-1] and results["multi_gpu_gen_times"][-1]:
                if results["single_gpu_gen_times"][-1] > 0:
                    gen_speedup = results["single_gpu_gen_times"][-1] / results["multi_gpu_gen_times"][-1]
                    results["gen_speedups"].append(gen_speedup)
                    print(f"  Generation speedup: {gen_speedup:.2f}x")
                else:
                    results["gen_speedups"].append(None)
                    print("  Could not calculate generation speedup (single GPU time is zero)")
            else:
                results["gen_speedups"].append(None)
                print("  Could not calculate generation speedup (missing data)")
            
            # Coherence calculation benchmark (if enabled and test field is available)
            if test_coherence and test_field is not None:
                print("\n  Starting coherence benchmarks...")
                
                # Single-GPU implementation
                single_gpu_coh_times = []
                for i in range(iterations):
                    try:
                        start_time = time.time()
                        _ = cuda_backend.calculate_3d_field_coherence(test_field)
                        end_time = time.time()
                        single_gpu_coh_times.append(end_time - start_time)
                        print(f"  Single-GPU coherence {i+1}/{iterations}: {single_gpu_coh_times[-1]:.4f} seconds")
                    except Exception as e:
                        print(f"  Error in single-GPU coherence benchmark: {e}")
                        break
                
                if single_gpu_coh_times:
                    avg_single_coh_time = sum(single_gpu_coh_times) / len(single_gpu_coh_times)
                    results["single_gpu_coh_times"].append(avg_single_coh_time)
                else:
                    results["single_gpu_coh_times"].append(None)
                    print("  Single-GPU coherence benchmark failed")
                
                # Multi-GPU implementation
                multi_gpu_coh_times = []
                for i in range(iterations):
                    try:
                        start_time = time.time()
                        _ = self.calculate_3d_field_coherence(test_field)
                        end_time = time.time()
                        multi_gpu_coh_times.append(end_time - start_time)
                        print(f"  Multi-GPU coherence {i+1}/{iterations}: {multi_gpu_coh_times[-1]:.4f} seconds")
                    except Exception as e:
                        print(f"  Error in multi-GPU coherence benchmark: {e}")
                        break
                
                if multi_gpu_coh_times:
                    avg_multi_coh_time = sum(multi_gpu_coh_times) / len(multi_gpu_coh_times)
                    results["multi_gpu_coh_times"].append(avg_multi_coh_time)
                else:
                    results["multi_gpu_coh_times"].append(None)
                    print("  Multi-GPU coherence benchmark failed")
                
                # Calculate coherence speedup
                if results["single_gpu_coh_times"][-1] and results["multi_gpu_coh_times"][-1]:
                    if results["single_gpu_coh_times"][-1] > 0:
                        coh_speedup = results["single_gpu_coh_times"][-1] / results["multi_gpu_coh_times"][-1]
                        results["coh_speedups"].append(coh_speedup)
                        print(f"  Coherence speedup: {coh_speedup:.2f}x")
                    else:
                        results["coh_speedups"].append(None)
                        print("  Could not calculate coherence speedup (single GPU time is zero)")
                else:
                    results["coh_speedups"].append(None)
                    print("  Could not calculate coherence speedup (missing data)")
            else:
                # Add placeholders for coherence results if skipped
                results["single_gpu_coh_times"].append(None)
                results["multi_gpu_coh_times"].append(None)
                results["coh_speedups"].append(None)
        
        # Print summary of results
        print("\nBenchmark Summary:")
        print(f"Using {len(self.devices)} GPUs")
        
        print("\nField Generation:")
        for i, size in enumerate(results["sizes"]):
            speedup = results["gen_speedups"][i]
            if speedup:
                print(f"  {size}: {speedup:.2f}x speedup")
            else:
                print(f"  {size}: No valid speedup data")
        
        if test_coherence:
            print("\nCoherence Calculation:")
            for i, size in enumerate(results["sizes"]):
                speedup = results["coh_speedups"][i]
                if speedup:
                    print(f"  {size}: {speedup:.2f}x speedup")
                else:
                    print(f"  {size}: No valid speedup data")
        
        return results
    
    def plot_multi_gpu_benchmark(self, results: Dict[str, Any]) -> None:
        """
        Plot multi-GPU benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_gpus = results.get("num_gpus", 0)
        sizes = results.get("sizes", [])
        single_times = results.get("single_gpu_times", [])
        multi_times = results.get("multi_gpu_times", [])
        speedups = results.get("speedups", [])
        
        if not sizes or not single_times or not multi_times:
            print("No benchmark data to plot")
            return
        
        # Set up the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot execution times
        x = np.arange(len(sizes))
        width = 0.35
        
        ax1.bar(x - width/2, single_times, width, label='Single GPU')
        ax1.bar(x + width/2, multi_times, width, label=f'{num_gpus} GPUs')
        
        ax1.set_xlabel('Field Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sizes)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot speedups
        ax2.bar(x, speedups, width * 1.5, color='green')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        ax2.axhline(y=num_gpus, color='b', linestyle='--', label=f'Ideal ({num_gpus}x)')
        
        ax2.set_xlabel('Field Size')
        ax2.set_ylabel('Speedup (x times)')
        ax2.set_title(f'Multi-GPU ({num_gpus} GPUs) Speedup')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sizes)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add speedup values as text
        for i, v in enumerate(speedups):
            ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('multi_gpu_benchmark.png')
        print("Multi-GPU benchmark results saved to 'multi_gpu_benchmark.png'")
        
        # Show plot if in interactive mode
        if hasattr(plt, 'show'):
            plt.show()
    
    def benchmark_3d_multi_gpu_tbc(self, max_size: int = 128, 
                                 custom_sizes: List[Tuple[int, int, int]] = None,
                                 test_standard_multi_gpu: bool = True) -> Dict[str, Any]:
        """
        Benchmark 3D multi-GPU with Thread Block Clusters performance.
        
        This benchmark compares:
        1. Standard single-GPU implementation
        2. Standard multi-GPU implementation (optional)
        3. Thread Block Cluster multi-GPU implementation
        
        Args:
            max_size: Maximum field dimension to test for cubic fields
            custom_sizes: Optional list of custom field sizes to test as [(width, height, depth), ...]
            test_standard_multi_gpu: Whether to include standard multi-GPU in the comparison
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.available or len(self.devices) < 1:
            print("Multi-GPU benchmarking requires at least 1 GPU")
            return {"error": "Insufficient GPUs available"}
            
        if not self.has_thread_block_cluster_support:
            print("Thread Block Clusters are not supported on this hardware")
            return {"error": "Thread Block Clusters not supported"}
        
        # Import needed modules
        import time
        from quantum_field.backends.cuda import CUDABackend
        
        # Initialize CUDA backend for comparison
        cuda_backend = CUDABackend()
        cuda_backend.initialize()
        
        # Check if thread_block_cluster module is available
        try:
            from quantum_field.thread_block_cluster import calculate_3d_field_coherence_tbc
            tbc_module_available = True
        except ImportError:
            tbc_module_available = False
            print("Warning: thread_block_cluster module not available, will use internal implementation")
        
        results = {
            "num_gpus": len(self.devices),
            "num_tbc_gpus": sum(1 for i, dev in enumerate(self.devices) if i in getattr(self, 'tbc_modules', {})),
            "sizes": [],
            "single_gpu_times": [],
            "multi_gpu_times": [] if test_standard_multi_gpu else None,
            "tbc_multi_gpu_times": [],
            "tbc_speedup_vs_single": [],
            "tbc_speedup_vs_multi": [] if test_standard_multi_gpu else None
        }
        
        # Define test sizes
        if custom_sizes:
            test_sizes = custom_sizes
        else:
            # Start with standard test sizes
            test_sizes = [
                (32, 32, 32),     # Small cubic field
                (64, 64, 64),     # Medium cubic field
                (128, 96, 64),    # Non-cubic field
                (64, 128, 32)     # Wide but shallow field
            ]
            
            # Add max size if not already included and it's reasonable
            if max_size > 64 and max_size <= 256:
                test_sizes.append((max_size, max_size, max_size))
        
        iterations = 3
        
        for width, height, depth in test_sizes:
            print(f"\nBenchmarking 3D size with TBC: {width}x{height}x{depth}")
            results["sizes"].append(f"{width}x{height}x{depth}")
            
            # Generate test data for coherence calculation
            try:
                print("  Generating test field...")
                test_field = self.generate_3d_quantum_field(width, height, depth, 'love')
                print(f"  Test field generated successfully: shape {test_field.shape}")
            except Exception as e:
                print(f"  Error generating test field: {e}")
                # Add placeholder results and continue to next size
                results["single_gpu_times"].append(None)
                if test_standard_multi_gpu:
                    results["multi_gpu_times"].append(None)
                results["tbc_multi_gpu_times"].append(None)
                results["tbc_speedup_vs_single"].append(None)
                if test_standard_multi_gpu:
                    results["tbc_speedup_vs_multi"].append(None)
                continue
            
            # 1. Standard single-GPU implementation
            single_gpu_times = []
            for i in range(iterations):
                try:
                    start_time = time.time()
                    _ = cuda_backend.calculate_3d_field_coherence(test_field)
                    end_time = time.time()
                    single_gpu_times.append(end_time - start_time)
                    print(f"  Single-GPU iteration {i+1}/{iterations}: {single_gpu_times[-1]:.4f} seconds")
                except Exception as e:
                    print(f"  Error in single-GPU benchmark: {e}")
                    break
            
            if single_gpu_times:
                avg_single_time = sum(single_gpu_times) / len(single_gpu_times)
                results["single_gpu_times"].append(avg_single_time)
            else:
                results["single_gpu_times"].append(None)
                print("  Single-GPU benchmark failed, possibly due to memory constraints")
            
            # 2. Standard multi-GPU implementation (if requested)
            if test_standard_multi_gpu:
                multi_gpu_times = []
                for i in range(iterations):
                    try:
                        start_time = time.time()
                        _ = self.calculate_3d_field_coherence(test_field, use_thread_block_clusters=False)
                        end_time = time.time()
                        multi_gpu_times.append(end_time - start_time)
                        print(f"  Standard Multi-GPU iteration {i+1}/{iterations}: {multi_gpu_times[-1]:.4f} seconds")
                    except Exception as e:
                        print(f"  Error in standard multi-GPU benchmark: {e}")
                        break
                
                if multi_gpu_times:
                    avg_multi_time = sum(multi_gpu_times) / len(multi_gpu_times)
                    results["multi_gpu_times"].append(avg_multi_time)
                else:
                    results["multi_gpu_times"].append(None)
                    print("  Standard multi-GPU benchmark failed")
            
            # 3. Thread Block Cluster multi-GPU implementation
            tbc_times = []
            for i in range(iterations):
                try:
                    start_time = time.time()
                    _ = self.calculate_3d_field_coherence(test_field, use_thread_block_clusters=True)
                    end_time = time.time()
                    tbc_times.append(end_time - start_time)
                    print(f"  TBC Multi-GPU iteration {i+1}/{iterations}: {tbc_times[-1]:.4f} seconds")
                except Exception as e:
                    print(f"  Error in TBC multi-GPU benchmark: {e}")
                    break
            
            if tbc_times:
                avg_tbc_time = sum(tbc_times) / len(tbc_times)
                results["tbc_multi_gpu_times"].append(avg_tbc_time)
            else:
                results["tbc_multi_gpu_times"].append(None)
                print("  TBC multi-GPU benchmark failed")
            
            # Calculate speedups
            if results["single_gpu_times"][-1] and results["tbc_multi_gpu_times"][-1]:
                if results["single_gpu_times"][-1] > 0:
                    tbc_vs_single = results["single_gpu_times"][-1] / results["tbc_multi_gpu_times"][-1]
                    results["tbc_speedup_vs_single"].append(tbc_vs_single)
                    print(f"  TBC vs. Single-GPU speedup: {tbc_vs_single:.2f}x")
                else:
                    results["tbc_speedup_vs_single"].append(None)
                    print("  Could not calculate TBC vs. Single-GPU speedup (single GPU time is zero)")
            else:
                results["tbc_speedup_vs_single"].append(None)
                print("  Could not calculate TBC vs. Single-GPU speedup (missing data)")
            
            if test_standard_multi_gpu:
                if results["multi_gpu_times"][-1] and results["tbc_multi_gpu_times"][-1]:
                    if results["multi_gpu_times"][-1] > 0:
                        tbc_vs_multi = results["multi_gpu_times"][-1] / results["tbc_multi_gpu_times"][-1]
                        results["tbc_speedup_vs_multi"].append(tbc_vs_multi)
                        print(f"  TBC vs. Standard Multi-GPU speedup: {tbc_vs_multi:.2f}x")
                    else:
                        results["tbc_speedup_vs_multi"].append(None)
                        print("  Could not calculate TBC vs. Multi-GPU speedup (multi-GPU time is zero)")
                else:
                    results["tbc_speedup_vs_multi"].append(None)
                    print("  Could not calculate TBC vs. Multi-GPU speedup (missing data)")
        
        # Print summary of results
        print("\nThread Block Cluster Benchmark Summary:")
        print(f"Using {results['num_tbc_gpus']} TBC-capable GPUs out of {len(self.devices)} total GPUs")
        
        for i, size in enumerate(results["sizes"]):
            vs_single = results["tbc_speedup_vs_single"][i]
            if vs_single:
                print(f"  {size}: {vs_single:.2f}x speedup vs. Single-GPU")
            else:
                print(f"  {size}: No valid speedup data vs. Single-GPU")
            
            if test_standard_multi_gpu:
                vs_multi = results["tbc_speedup_vs_multi"][i]
                if vs_multi:
                    print(f"    {vs_multi:.2f}x speedup vs. Standard Multi-GPU")
                else:
                    print(f"    No valid speedup data vs. Standard Multi-GPU")
        
        return results
    
    def plot_3d_multi_gpu_tbc_benchmark(self, results: Dict[str, Any]) -> None:
        """
        Plot 3D multi-GPU Thread Block Cluster benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_gpus = results.get("num_gpus", 0)
        num_tbc_gpus = results.get("num_tbc_gpus", 0)
        sizes = results.get("sizes", [])
        single_times = results.get("single_gpu_times", [])
        multi_times = results.get("multi_gpu_times", [])
        tbc_times = results.get("tbc_multi_gpu_times", [])
        tbc_vs_single = results.get("tbc_speedup_vs_single", [])
        tbc_vs_multi = results.get("tbc_speedup_vs_multi", [])
        
        test_standard_multi = multi_times is not None
        
        if not sizes or not single_times or not tbc_times:
            print("No benchmark data to plot")
            return
        
        # Set up the figure with appropriate number of subplots
        if test_standard_multi:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter out None values
        valid_indices = [i for i, (s, t) in enumerate(zip(single_times, tbc_times)) 
                       if s is not None and t is not None]
        
        valid_sizes = [sizes[i] for i in valid_indices]
        valid_single_times = [single_times[i] for i in valid_indices]
        valid_tbc_times = [tbc_times[i] for i in valid_indices]
        valid_tbc_vs_single = [tbc_vs_single[i] for i in valid_indices]
        
        if test_standard_multi:
            valid_multi_indices = [i for i in valid_indices if multi_times[i] is not None]
            valid_multi_sizes = [sizes[i] for i in valid_multi_indices]
            valid_multi_times = [multi_times[i] for i in valid_multi_indices]
            valid_tbc_multi_times = [tbc_times[i] for i in valid_multi_indices]
            valid_tbc_vs_multi = [tbc_vs_multi[i] for i in valid_multi_indices]
        
        if not valid_indices:
            print("No valid benchmark data to plot")
            return
        
        # Plot execution times
        x = np.arange(len(valid_sizes))
        width = 0.35
        
        # Plot execution times - Single vs TBC
        ax1.bar(x - width/2 if test_standard_multi else x - width/3, valid_single_times, width, label='Single GPU')
        if test_standard_multi:
            ax1.bar(x + width/2, valid_tbc_times, width, label=f'TBC Multi-GPU ({num_tbc_gpus} GPUs)')
        else:
            ax1.bar(x + width/3, valid_tbc_times, width, label=f'TBC Multi-GPU ({num_tbc_gpus} GPUs)')
        
        ax1.set_xlabel('Field Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('TBC vs Single-GPU Execution Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_sizes, rotation=30, ha='right')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot TBC vs. Single-GPU speedup
        ax2.bar(x, valid_tbc_vs_single, width * 1.5, color='green')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        ideal_speedup = num_tbc_gpus
        ax2.axhline(y=ideal_speedup, color='b', linestyle='--', 
                   label=f'Ideal ({ideal_speedup}x)')
        
        ax2.set_xlabel('Field Size')
        ax2.set_ylabel('Speedup (x times)')
        ax2.set_title(f'TBC Multi-GPU vs. Single-GPU Speedup')
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_sizes, rotation=30, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add speedup values as text
        for i, v in enumerate(valid_tbc_vs_single):
            ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        
        # Plot TBC vs. Standard Multi-GPU comparison (if requested)
        if test_standard_multi and valid_multi_indices:
            x_multi = np.arange(len(valid_multi_sizes))
            
            # Multi-GPU time comparison
            ax3.bar(x_multi - width/2, valid_multi_times, width, label=f'Standard Multi-GPU ({num_gpus} GPUs)')
            ax3.bar(x_multi + width/2, valid_tbc_multi_times, width, label=f'TBC Multi-GPU ({num_tbc_gpus} GPUs)')
            
            ax3.set_xlabel('Field Size')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('TBC vs Standard Multi-GPU Execution Time')
            ax3.set_xticks(x_multi)
            ax3.set_xticklabels(valid_multi_sizes, rotation=30, ha='right')
            ax3.legend()
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add overall title
        plt.suptitle(f'Thread Block Cluster Multi-GPU Performance', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for overall title
        plt.savefig('tbc_multi_gpu_benchmark.png', dpi=150)
        print("Thread Block Cluster multi-GPU benchmark results saved to 'tbc_multi_gpu_benchmark.png'")
        
        # Show plot if in interactive mode
        if hasattr(plt, 'show'):
            plt.show()
    
    def plot_3d_multi_gpu_benchmark(self, results: Dict[str, Any]) -> None:
        """
        Plot 3D multi-GPU benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_gpus = results.get("num_gpus", 0)
        sizes = results.get("sizes", [])
        
        # Check for the new benchmark format (with separate generation and coherence data)
        has_new_format = "single_gpu_gen_times" in results
        
        if has_new_format:
            single_gen_times = results.get("single_gpu_gen_times", [])
            multi_gen_times = results.get("multi_gpu_gen_times", [])
            gen_speedups = results.get("gen_speedups", [])
            
            single_coh_times = results.get("single_gpu_coh_times", [])
            multi_coh_times = results.get("multi_gpu_coh_times", [])
            coh_speedups = results.get("coh_speedups", [])
            
            # Determine if we have coherence data
            has_coherence_data = any(t is not None for t in single_coh_times)
            
            # Create a larger figure with more subplots if we have coherence data
            if has_coherence_data:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                ((ax1, ax2), (ax3, ax4)) = axes
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Filter out None values for generation data
            valid_gen_indices = [i for i, (s, m) in enumerate(zip(single_gen_times, multi_gen_times)) 
                             if s is not None and m is not None]
            
            valid_sizes_gen = [sizes[i] for i in valid_gen_indices]
            valid_single_gen_times = [single_gen_times[i] for i in valid_gen_indices]
            valid_multi_gen_times = [multi_gen_times[i] for i in valid_gen_indices]
            valid_gen_speedups = [gen_speedups[i] for i in valid_gen_indices]
            
            if valid_gen_indices:
                # Plot generation execution times
                x_gen = np.arange(len(valid_sizes_gen))
                width = 0.35
                
                ax1.bar(x_gen - width/2, valid_single_gen_times, width, label='Single GPU')
                ax1.bar(x_gen + width/2, valid_multi_gen_times, width, label=f'{num_gpus} GPUs')
                
                ax1.set_xlabel('Field Size')
                ax1.set_ylabel('Time (seconds)')
                ax1.set_title('3D Field Generation Time Comparison')
                ax1.set_xticks(x_gen)
                ax1.set_xticklabels(valid_sizes_gen, rotation=30, ha='right')
                ax1.legend()
                ax1.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Plot generation speedups
                ax2.bar(x_gen, valid_gen_speedups, width * 1.5, color='green')
                ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
                ax2.axhline(y=num_gpus, color='b', linestyle='--', label=f'Ideal ({num_gpus}x)')
                
                ax2.set_xlabel('Field Size')
                ax2.set_ylabel('Speedup (x times)')
                ax2.set_title(f'3D Field Generation Speedup ({num_gpus} GPUs)')
                ax2.set_xticks(x_gen)
                ax2.set_xticklabels(valid_sizes_gen, rotation=30, ha='right')
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                ax2.legend()
                
                # Add speedup values as text
                for i, v in enumerate(valid_gen_speedups):
                    ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
            
            # Plot coherence data if available
            if has_coherence_data and has_coherence_data:
                # Filter out None values for coherence data
                valid_coh_indices = [i for i, (s, m) in enumerate(zip(single_coh_times, multi_coh_times)) 
                                if s is not None and m is not None]
                
                valid_sizes_coh = [sizes[i] for i in valid_coh_indices]
                valid_single_coh_times = [single_coh_times[i] for i in valid_coh_indices]
                valid_multi_coh_times = [multi_coh_times[i] for i in valid_coh_indices]
                valid_coh_speedups = [coh_speedups[i] for i in valid_coh_indices]
                
                if valid_coh_indices:
                    # Plot coherence execution times
                    x_coh = np.arange(len(valid_sizes_coh))
                    width = 0.35
                    
                    ax3.bar(x_coh - width/2, valid_single_coh_times, width, label='Single GPU')
                    ax3.bar(x_coh + width/2, valid_multi_coh_times, width, label=f'{num_gpus} GPUs')
                    
                    ax3.set_xlabel('Field Size')
                    ax3.set_ylabel('Time (seconds)')
                    ax3.set_title('3D Field Coherence Calculation Time Comparison')
                    ax3.set_xticks(x_coh)
                    ax3.set_xticklabels(valid_sizes_coh, rotation=30, ha='right')
                    ax3.legend()
                    ax3.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Plot coherence speedups
                    ax4.bar(x_coh, valid_coh_speedups, width * 1.5, color='purple')
                    ax4.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
                    ax4.axhline(y=num_gpus, color='b', linestyle='--', label=f'Ideal ({num_gpus}x)')
                    
                    ax4.set_xlabel('Field Size')
                    ax4.set_ylabel('Speedup (x times)')
                    ax4.set_title(f'3D Field Coherence Calculation Speedup ({num_gpus} GPUs)')
                    ax4.set_xticks(x_coh)
                    ax4.set_xticklabels(valid_sizes_coh, rotation=30, ha='right')
                    ax4.grid(axis='y', linestyle='--', alpha=0.7)
                    ax4.legend()
                    
                    # Add speedup values as text
                    for i, v in enumerate(valid_coh_speedups):
                        ax4.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        else:
            # Handle the old benchmark format
            single_times = results.get("single_gpu_times", [])
            multi_times = results.get("multi_gpu_times", [])
            speedups = results.get("speedups", [])
            
            if not sizes or not single_times or not multi_times:
                print("No benchmark data to plot")
                return
            
            # Set up the figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Filter out None values
            valid_indices = [i for i, (s, m) in enumerate(zip(single_times, multi_times)) 
                            if s is not None and m is not None]
            
            valid_sizes = [sizes[i] for i in valid_indices]
            valid_single_times = [single_times[i] for i in valid_indices]
            valid_multi_times = [multi_times[i] for i in valid_indices]
            valid_speedups = [speedups[i] for i in valid_indices]
            
            if not valid_indices:
                print("No valid benchmark data to plot")
                return
            
            # Plot execution times
            x = np.arange(len(valid_sizes))
            width = 0.35
            
            ax1.bar(x - width/2, valid_single_times, width, label='Single GPU')
            ax1.bar(x + width/2, valid_multi_times, width, label=f'{num_gpus} GPUs')
            
            ax1.set_xlabel('Field Size')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('3D Field Execution Time Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(valid_sizes, rotation=30, ha='right')
            ax1.legend()
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot speedups
            ax2.bar(x, valid_speedups, width * 1.5, color='green')
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            ax2.axhline(y=num_gpus, color='b', linestyle='--', label=f'Ideal ({num_gpus}x)')
            
            ax2.set_xlabel('Field Size')
            ax2.set_ylabel('Speedup (x times)')
            ax2.set_title(f'3D Field Multi-GPU ({num_gpus} GPUs) Speedup')
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_sizes, rotation=30, ha='right')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.legend()
            
            # Add speedup values as text
            for i, v in enumerate(valid_speedups):
                ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        
        # Add overall title
        plt.suptitle(f'Multi-GPU 3D Field Performance Benchmarks ({num_gpus} GPUs)', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for overall title
        plt.savefig('multi_gpu_3d_benchmark.png', dpi=150)
        print("3D Multi-GPU benchmark results saved to 'multi_gpu_3d_benchmark.png'")
        
        # Show plot if in interactive mode
        if hasattr(plt, 'show'):
            plt.show()


# Global manager instance
_manager = None

def get_multi_gpu_manager() -> MultiGPUManager:
    """
    Get or create the global MultiGPUManager instance.
    
    Returns:
        MultiGPUManager instance
    """
    global _manager
    
    if _manager is None:
        _manager = MultiGPUManager()
    
    return _manager

def generate_quantum_field_multi_gpu(width: int, height: int, frequency_name: str = 'love', 
                                   time_factor: float = 0) -> np.ndarray:
    """
    Generate a quantum field using multiple GPUs.
    
    Args:
        width: Width of the field
        height: Height of the field
        frequency_name: The sacred frequency to use
        time_factor: Time factor for animation
        
    Returns:
        A 2D NumPy array representing the quantum field
    """
    manager = get_multi_gpu_manager()
    return manager.generate_quantum_field(width, height, frequency_name, time_factor)

def calculate_field_coherence_multi_gpu(field_data: np.ndarray) -> float:
    """
    Calculate the coherence of a quantum field using multiple GPUs.
    
    Args:
        field_data: A 2D NumPy array containing the field data
        
    Returns:
        A float representing the field coherence
    """
    manager = get_multi_gpu_manager()
    return manager.calculate_field_coherence(field_data)

def generate_3d_quantum_field_multi_gpu(width: int, height: int, depth: int, 
                                      frequency_name: str = 'love', 
                                      time_factor: float = 0) -> np.ndarray:
    """
    Generate a 3D quantum field using multiple GPUs.
    
    Args:
        width: Width of the field
        height: Height of the field
        depth: Depth of the field
        frequency_name: The sacred frequency to use
        time_factor: Time factor for animation
        
    Returns:
        A 3D NumPy array representing the quantum field
    """
    manager = get_multi_gpu_manager()
    return manager.generate_3d_quantum_field(width, height, depth, frequency_name, time_factor)

def calculate_3d_field_coherence_multi_gpu(field_data: np.ndarray) -> float:
    """
    Calculate the coherence of a 3D quantum field using multiple GPUs.
    
    Args:
        field_data: A 3D NumPy array containing the field data
        
    Returns:
        A float representing the field coherence
    """
    manager = get_multi_gpu_manager()
    return manager.calculate_3d_field_coherence(field_data)

def benchmark_multi_gpu(max_size: int = 4096) -> Dict[str, Any]:
    """
    Benchmark multi-GPU performance against single-GPU implementation.
    
    Args:
        max_size: Maximum field dimension to test
        
    Returns:
        Dictionary with benchmark results
    """
    manager = get_multi_gpu_manager()
    results = manager.benchmark_multi_gpu(max_size)
    manager.plot_multi_gpu_benchmark(results)
    return results

def benchmark_3d_multi_gpu(max_size: int = 128, 
                     custom_sizes: List[Tuple[int, int, int]] = None,
                     test_coherence: bool = True,
                     plot_results: bool = True) -> Dict[str, Any]:
    """
    Benchmark 3D multi-GPU performance against single-GPU implementation.
    
    Args:
        max_size: Maximum field dimension to test for cubic fields
        custom_sizes: Optional list of custom field sizes to test as [(width, height, depth), ...]
        test_coherence: Whether to benchmark coherence calculation (in addition to field generation)
        plot_results: Whether to generate and save a plot of the results
        
    Returns:
        Dictionary with benchmark results
    """
    manager = get_multi_gpu_manager()
    results = manager.benchmark_3d_multi_gpu(max_size, custom_sizes, test_coherence)
    
    if plot_results:
        manager.plot_3d_multi_gpu_benchmark(results)
    
    return results
    
def benchmark_3d_multi_gpu_tbc(max_size: int = 128,
                          custom_sizes: List[Tuple[int, int, int]] = None,
                          test_standard_multi_gpu: bool = True,
                          plot_results: bool = True) -> Dict[str, Any]:
    """
    Benchmark 3D multi-GPU with Thread Block Clusters performance.
    
    This benchmark compares:
    1. Standard single-GPU implementation
    2. Standard multi-GPU implementation
    3. Thread Block Cluster multi-GPU implementation
    
    Args:
        max_size: Maximum field dimension to test for cubic fields
        custom_sizes: Optional list of custom field sizes to test as [(width, height, depth), ...]
        test_standard_multi_gpu: Whether to include standard multi-GPU in the comparison
        plot_results: Whether to generate and save a plot of the results
        
    Returns:
        Dictionary with benchmark results
    """
    manager = get_multi_gpu_manager()
    
    # Check if TBC is supported first
    if not manager.has_thread_block_cluster_support:
        print("Thread Block Clusters are not supported on this hardware.")
        print("This feature requires an H100 GPU or newer (Compute Capability 9.0+)")
        return {"error": "Thread Block Clusters not supported"}
    
    # Call the manager's benchmark method
    results = manager.benchmark_3d_multi_gpu_tbc(max_size, custom_sizes, test_standard_multi_gpu)
    
    if plot_results:
        manager.plot_3d_multi_gpu_tbc_benchmark(results)
    
    return results