"""
WebGPU backend for quantum field visualization.

This backend enables quantum field generation and visualization directly in web browsers
using the WebGPU API via PyWebGPU bridge.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, Union

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from . import AcceleratorBackend

# Conditional imports
try:
    import pywebgpu as webgpu
    from pywebgpu.backend.wgpu import (
        Device, 
        ShaderModule, 
        Buffer, 
        CommandEncoder, 
        ComputePipeline
    )
    HAS_WEBGPU = True
except ImportError:
    HAS_WEBGPU = False


class WebGPUBackend(AcceleratorBackend):
    """
    WebGPU implementation of the accelerator backend.
    
    This backend enables quantum field generation directly in web browsers
    or on desktop using the WebGPU API.
    
    Attributes:
        name (str): "webgpu"
        priority (int): 65 (between CUDA and Mobile)
    """
    
    name = "webgpu"
    priority = 65  # Between CUDA (70) and Mobile (60)
    
    def __init__(self):
        """Initialize the WebGPU backend."""
        super().__init__()
        self._device = None
        self._adapter = None
        self._shader_modules = {}
        self._pipelines = {}
        self._initialized = False
        self._device_info = {}
        
        if HAS_WEBGPU:
            try:
                self._initialize()
            except Exception as e:
                print(f"WebGPU initialization failed: {str(e)}")
    
    def _initialize(self):
        """Initialize WebGPU device and resources."""
        if not HAS_WEBGPU:
            return
        
        # Request adapter (with power preference)
        self._adapter = webgpu.wgpu.request_adapter(
            power_preference=webgpu.wgpu.PowerPreference.high_performance
        )
        
        if not self._adapter:
            return
        
        # Create device
        self._device = self._adapter.request_device()
        
        if not self._device:
            return
        
        # Get device information
        self._device_info = {
            "name": self._adapter.get_info().get("name", "Unknown"),
            "backend": self._adapter.get_info().get("backend", "Unknown"),
            "is_fallback_adapter": self._adapter.get_info().get("is_fallback_adapter", False),
        }
        
        # Create shader for quantum field generation
        self._create_field_generation_shader()
        
        # Mark as initialized
        self._initialized = True
    
    def _create_field_generation_shader(self):
        """Create and compile the shader for quantum field generation."""
        if not self._device:
            return
        
        # WebGPU compute shader for quantum field generation
        shader_code = """
        @group(0) @binding(0) var<storage, read_write> output: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;
        
        struct Params {
            width: u32,
            height: u32,
            frequency: f32,
            time_factor: f32,
            phi: f32,
            lambda: f32,
            phi_phi: f32,
        }
        
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x_id = global_id.x;
            let y_id = global_id.y;
            
            if (x_id >= params.width || y_id >= params.height) {
                return;
            }
            
            // Calculate normalized coordinates (-1 to 1)
            let x = (f32(x_id) / f32(params.width)) * 2.0 - 1.0;
            let y = (f32(y_id) / f32(params.height)) * 2.0 - 1.0;
            
            // Calculate with phi-harmonic principles
            let distance = sqrt(x*x + y*y) * params.phi;
            let angle = atan2(y, x);
            
            // Apply frequency and time factor
            let wave = sin(distance * params.frequency * 0.01 + angle * params.phi + params.time_factor * params.phi_phi);
            let dampening = exp(-distance * params.lambda);
            
            // Combine wave and dampening
            let field_value = wave * dampening;
            
            // Store result in output buffer
            let index = y_id * params.width + x_id;
            output[index] = field_value;
        }
        """
        
        # Create shader module
        self._shader_modules["field_generation"] = self._device.create_shader_module(
            code=shader_code
        )
        
        # Create pipeline layout and bind group layout
        bind_group_layout = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": webgpu.wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": webgpu.wgpu.BufferBindingType.storage,
                    }
                },
                {
                    "binding": 1,
                    "visibility": webgpu.wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": webgpu.wgpu.BufferBindingType.uniform,
                    }
                }
            ]
        )
        
        pipeline_layout = self._device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        
        # Create compute pipeline
        self._pipelines["field_generation"] = self._device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": self._shader_modules["field_generation"], "entry_point": "main"}
        )
    
    def is_available(self) -> bool:
        """Check if WebGPU backend is available on the current system."""
        return HAS_WEBGPU and self._initialized
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get the capabilities of the WebGPU backend."""
        if not self.is_available():
            return {
                "web_compatible": True,
                "multi_device": False,
                "thread_block_clusters": False,
                "async_execution": False,
                "half_precision": False,
                "dlpack_support": False,
            }
        
        return {
            "web_compatible": True,
            "multi_device": False,
            "thread_block_clusters": False,
            "async_execution": True,
            "half_precision": True,  # Most WebGPU implementations support FP16
            "dlpack_support": False,  # No direct DLPack support yet
        }
    
    def generate_quantum_field(
        self, 
        width: int, 
        height: int, 
        frequency_name: Optional[str] = None, 
        time_factor: float = 0.0,
        custom_frequency: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate a quantum field using WebGPU.
        
        Args:
            width: Width of the field in pixels
            height: Height of the field in pixels
            frequency_name: Name of the frequency to use (from SACRED_FREQUENCIES)
            time_factor: Time evolution factor (0.0 to 2π)
            custom_frequency: Custom frequency value (used if frequency_name is None)
            
        Returns:
            NumPy array containing the generated field
        """
        if not self.is_available():
            raise RuntimeError("WebGPU backend is not available")
        
        # Determine frequency to use
        if frequency_name is not None:
            if frequency_name not in SACRED_FREQUENCIES:
                raise ValueError(f"Unknown frequency name: {frequency_name}")
            frequency = SACRED_FREQUENCIES[frequency_name]
        elif custom_frequency is not None:
            frequency = custom_frequency
        else:
            raise ValueError("Either frequency_name or custom_frequency must be provided")
        
        # Create output buffer
        output_size = width * height * 4  # 4 bytes per float32
        output_buffer = self._device.create_buffer(
            size=output_size,
            usage=webgpu.wgpu.BufferUsage.STORAGE | webgpu.wgpu.BufferUsage.COPY_SRC
        )
        
        # Create uniform buffer for parameters
        params = np.array([
            width, height, frequency, time_factor, 
            PHI, LAMBDA, PHI_PHI
        ], dtype=np.float32)
        
        params_buffer = self._device.create_buffer_with_data(
            data=params.tobytes(),
            usage=webgpu.wgpu.BufferUsage.UNIFORM | webgpu.wgpu.BufferUsage.COPY_DST
        )
        
        # Create bind group
        bind_group = self._device.create_bind_group(
            layout=self._pipelines["field_generation"].get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": output_buffer}},
                {"binding": 1, "resource": {"buffer": params_buffer}}
            ]
        )
        
        # Create result buffer for copying data back to CPU
        result_buffer = self._device.create_buffer(
            size=output_size,
            usage=webgpu.wgpu.BufferUsage.COPY_DST | webgpu.wgpu.BufferUsage.MAP_READ
        )
        
        # Create and execute command encoder
        encoder = self._device.create_command_encoder()
        
        # Dispatch compute pass
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipelines["field_generation"])
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(
            (width + 7) // 8,  # Round up to multiple of workgroup size
            (height + 7) // 8,
            1
        )
        compute_pass.end()
        
        # Copy output to result buffer
        encoder.copy_buffer_to_buffer(
            output_buffer, 0,
            result_buffer, 0,
            output_size
        )
        
        # Submit commands
        command_buffer = encoder.finish()
        self._device.queue.submit([command_buffer])
        
        # Map the result buffer to read data
        result_buffer.map_async(webgpu.wgpu.MapMode.READ)
        self._device.poll(True)  # Wait for mapping
        
        # Read data
        data = result_buffer.get_mapped_range()
        field_data = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        
        # Unmap the buffer
        result_buffer.unmap()
        
        return field_data
    
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum field using WebGPU.
        
        Args:
            field_data: NumPy array containing the field data
            
        Returns:
            Coherence factor between 0.0 and 1.0
        """
        # For now, use CPU implementation for coherence calculation
        # WebGPU coherence calculation could be implemented in the future
        from ..core import calculate_field_coherence as cpu_calculate_field_coherence
        return cpu_calculate_field_coherence(field_data)
    
    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """
        Generate a Phi-based sacred pattern using WebGPU.
        
        Args:
            width: Width of the pattern in pixels
            height: Height of the pattern in pixels
            
        Returns:
            NumPy array containing the generated pattern
        """
        # For now, use CPU implementation for phi pattern generation
        # WebGPU phi pattern generation could be implemented in the future
        from ..core import generate_phi_pattern as cpu_generate_phi_pattern
        return cpu_generate_phi_pattern(width, height)
    
    def to_dlpack(self, field_data: np.ndarray) -> Any:
        """
        Convert a NumPy array to DLPack format (not supported yet).
        
        Args:
            field_data: NumPy array to convert
            
        Raises:
            RuntimeError: WebGPU backend doesn't support DLPack yet
        """
        raise RuntimeError("WebGPU backend doesn't support DLPack conversion yet")
    
    def from_dlpack(self, dlpack_tensor: Any, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert a DLPack tensor to a NumPy array (not supported yet).
        
        Args:
            dlpack_tensor: DLPack tensor to convert
            shape: Optional shape to reshape the resulting array
            
        Raises:
            RuntimeError: WebGPU backend doesn't support DLPack yet
        """
        raise RuntimeError("WebGPU backend doesn't support DLPack conversion yet")
    
    def shutdown(self) -> None:
        """Release resources used by the WebGPU backend."""
        self._initialized = False
        self._device = None
        self._adapter = None
        self._shader_modules = {}
        self._pipelines = {}