#!/usr/bin/env python3
"""
Integration tests for multi-GPU functionality

These tests verify that workloads are correctly distributed across multiple GPUs
and that results are properly combined.
"""

import os
import sys
import pytest
import numpy as np
from typing import Tuple, Dict, Optional

# Skip all tests if quantum_field package is not available
try:
    import quantum_field
    from quantum_field.multi_gpu import (
        get_multi_gpu_manager,
        generate_quantum_field_multi_gpu,
        calculate_field_coherence_multi_gpu
    )
except ImportError:
    pytest.skip("quantum_field package not found", allow_module_level=True)

# Get multi-GPU manager and skip tests if multi-GPU is not available
multi_gpu_manager = get_multi_gpu_manager()
if not multi_gpu_manager.available or len(multi_gpu_manager.devices) < 2:
    pytest.skip("Multiple GPUs not available", allow_module_level=True)


@pytest.fixture
def test_field() -> Tuple[int, int]:
    """Define a test field size"""
    return (512, 512)


class TestMultiGPUInitialization:
    """Tests for multi-GPU initialization and detection"""

    def test_devices_detected(self):
        """Test that multiple devices are detected"""
        assert len(multi_gpu_manager.devices) >= 2
        assert multi_gpu_manager.available is True
        assert len(multi_gpu_manager.streams) == len(multi_gpu_manager.devices)

    def test_modules_compiled(self):
        """Test that modules are compiled for each device"""
        assert len(multi_gpu_manager.modules) == len(multi_gpu_manager.devices)
        for device_id, module in multi_gpu_manager.modules.items():
            assert module is not None
            assert device_id < len(multi_gpu_manager.devices)


class TestFieldGeneration:
    """Tests for multi-GPU field generation"""

    def test_generate_field(self, test_field):
        """Test generating a field with multi-GPU"""
        width, height = test_field
        field = generate_quantum_field_multi_gpu(width, height, 'love', 0)
        
        # Check field dimensions
        assert field.shape == (height, width)
        assert field.dtype == np.float32
        
        # Check that field values are reasonable (should be in range [-1, 1])
        assert np.min(field) >= -1.0
        assert np.max(field) <= 1.0
    
    @pytest.mark.parametrize("frequency", ['love', 'unity', 'cascade', 'truth'])
    def test_different_frequencies(self, test_field, frequency):
        """Test different frequencies with multi-GPU"""
        width, height = test_field
        field = generate_quantum_field_multi_gpu(width, height, frequency, 0)
        
        # Check field dimensions
        assert field.shape == (height, width)
        
        # Different frequencies should produce different fields
        field2 = generate_quantum_field_multi_gpu(width, height, 'love' if frequency != 'love' else 'unity', 0)
        assert not np.array_equal(field, field2)
    
    @pytest.mark.parametrize("time_factor", [0, 0.5, 1.0])
    def test_time_evolution(self, test_field, time_factor):
        """Test field evolution over time with multi-GPU"""
        width, height = test_field
        field = generate_quantum_field_multi_gpu(width, height, 'love', time_factor)
        
        # Fields at different time points should be different
        if time_factor != 0:
            field_t0 = generate_quantum_field_multi_gpu(width, height, 'love', 0)
            assert not np.array_equal(field, field_t0)


class TestFieldCoherence:
    """Tests for multi-GPU field coherence calculations"""

    def test_coherence_calculation(self, test_field):
        """Test calculating field coherence with multi-GPU"""
        width, height = test_field
        
        # Generate a field
        field = generate_quantum_field_multi_gpu(width, height, 'love', 0)
        
        # Calculate coherence
        coherence = calculate_field_coherence_multi_gpu(field)
        
        # Coherence should be a float value in a reasonable range
        assert isinstance(coherence, float)
        assert 0 <= coherence <= quantum_field.PHI * 2
    
    def test_coherence_consistency(self, test_field):
        """Test consistency of coherence calculation with multiple runs"""
        width, height = test_field
        
        # Generate a field
        field = generate_quantum_field_multi_gpu(width, height, 'love', 0)
        
        # Calculate coherence multiple times
        coherence1 = calculate_field_coherence_multi_gpu(field)
        coherence2 = calculate_field_coherence_multi_gpu(field)
        
        # Values should be close (may not be exactly equal due to random sampling)
        assert abs(coherence1 - coherence2) < 0.3


class TestMultiGPUWorkDistribution:
    """Tests for work distribution across GPUs"""

    def test_work_distribution(self, monkeypatch):
        """Test that work is distributed across GPUs"""
        # Create a side effect counter to track GPU usage
        gpu_usage_counts = [0] * len(multi_gpu_manager.devices)
        
        # Original set_current method
        original_set_current = type(multi_gpu_manager.devices[0]).set_current
        
        def set_current_with_count(self):
            # Count when each device is used
            device_id = self.id()
            if device_id < len(gpu_usage_counts):
                gpu_usage_counts[device_id] += 1
            return original_set_current(self)
            
        # Patch the set_current method
        monkeypatch.setattr(
            type(multi_gpu_manager.devices[0]), 
            'set_current', 
            set_current_with_count
        )
        
        # Generate a large field
        width, height = 1024, 1024
        _ = generate_quantum_field_multi_gpu(width, height, 'love', 0)
        
        # Check that all GPUs were used
        used_gpus = sum(1 for count in gpu_usage_counts if count > 0)
        assert used_gpus > 1, "Work was not distributed across multiple GPUs"


class TestMultiGPUErrorHandling:
    """Tests for error handling in multi-GPU code"""

    def test_fallback_on_error(self, monkeypatch):
        """Test fallback to single-GPU on errors"""
        # Make all device operations fail
        def failing_launch(*args, **kwargs):
            raise RuntimeError("Simulated launch failure")
            
        # Patch the launch function
        import cuda.core.experimental
        monkeypatch.setattr(cuda.core.experimental, 'launch', failing_launch)
        
        # Should not crash, will fall back to CPU
        width, height = 512, 512
        try:
            field = generate_quantum_field_multi_gpu(width, height, 'love', 0)
            assert field.shape == (height, width)
        except Exception as e:
            pytest.fail(f"Field generation failed with unexpected error: {e}")


class TestIntegrationWithCore:
    """Tests for integration with the core module"""

    def test_core_integration(self):
        """Test integration with the core module"""
        from quantum_field.core import generate_quantum_field
        
        # Core function should use multi-GPU for large fields
        width, height = 1024, 1024
        field1 = generate_quantum_field(width, height, 'love', 0)
        field2 = generate_quantum_field_multi_gpu(width, height, 'love', 0)
        
        # Both should produce valid fields of the same shape
        assert field1.shape == field2.shape
        assert field1.dtype == field2.dtype
        
        # Values might differ slightly due to floating-point variations
        # between implementations, but should be close
        max_diff = np.max(np.abs(field1 - field2))
        assert max_diff < 0.01 or np.allclose(field1, field2, rtol=1e-3, atol=1e-3)