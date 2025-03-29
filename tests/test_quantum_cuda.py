#!/usr/bin/env python3
"""
Tests for quantum_cuda.py module

These tests ensure that both CPU and GPU implementations
produce equivalent results, and verify the correctness of
the quantum field visualization algorithms.
"""

import os
import pytest
import numpy as np

# Import the module to test
try:
    import quantum_cuda as qc
    import sacred_constants as sc
except ImportError:
    pytest.skip("quantum_cuda module not found", allow_module_level=True)


# Define test parameters
TEST_SIZES = [(64, 32), (128, 64)]
TEST_FREQUENCIES = ['love', 'unity', 'cascade', 'truth']
TIME_FACTORS = [0, 0.5, 1.0]


def test_module_imported():
    """Verify that the module can be imported"""
    assert qc is not None
    assert hasattr(qc, 'generate_quantum_field')
    assert hasattr(qc, 'calculate_field_coherence')


def test_sacred_constants():
    """Verify that the sacred constants are defined correctly"""
    assert hasattr(sc, 'PHI')
    assert abs(sc.PHI - 1.618033988749895) < 1e-10
    assert hasattr(sc, 'LAMBDA')
    assert abs(sc.LAMBDA - 0.618033988749895) < 1e-10
    assert hasattr(sc, 'PHI_PHI')
    assert abs(sc.PHI_PHI - sc.PHI ** sc.PHI) < 1e-10
    assert hasattr(sc, 'SACRED_FREQUENCIES')
    assert 'love' in sc.SACRED_FREQUENCIES
    assert sc.SACRED_FREQUENCIES['love'] == 528


class TestQuantumField:
    """Tests for quantum field generation functions"""

    @pytest.mark.parametrize("size", TEST_SIZES)
    @pytest.mark.parametrize("frequency", TEST_FREQUENCIES)
    @pytest.mark.parametrize("time_factor", TIME_FACTORS)
    def test_generate_quantum_field_output_shape(self, size, frequency, time_factor):
        """Test that generate_quantum_field produces the correct output shape"""
        width, height = size
        field = qc.generate_quantum_field(width, height, frequency, time_factor)
        assert field.shape == (height, width)
        assert field.dtype == np.float32

    @pytest.mark.parametrize("size", TEST_SIZES)
    @pytest.mark.parametrize("frequency", TEST_FREQUENCIES)
    @pytest.mark.parametrize("time_factor", TIME_FACTORS)
    def test_cpu_gpu_equivalence(self, size, frequency, time_factor):
        """Test that CPU and GPU implementations produce similar results"""
        width, height = size
        
        # Force CPU implementation
        qc.CUDA_AVAILABLE = False
        cpu_field = qc.generate_quantum_field(width, height, frequency, time_factor)
        
        # Restore original CUDA_AVAILABLE value
        qc.CUDA_AVAILABLE = hasattr(qc, 'cp')
        
        # Only run if CUDA is available
        if qc.CUDA_AVAILABLE and qc.initialize_cuda():
            gpu_field = qc.generate_quantum_field_cuda(width, height, frequency, time_factor)
            
            # The implementations won't be exactly the same due to floating-point differences
            # between CPU and GPU, but they should be very similar
            assert gpu_field.shape == cpu_field.shape
            
            # Check if values are close (within some tolerance)
            abs_diff = np.abs(gpu_field - cpu_field)
            max_diff = np.max(abs_diff)
            assert max_diff < 0.01, f"Max difference: {max_diff}"
        else:
            pytest.skip("CUDA not available")

    def test_phi_pattern_generation(self):
        """Test phi pattern generation"""
        width, height = 64, 32
        pattern = qc.generate_phi_pattern_cpu(width, height)
        assert pattern.shape == (height, width)
        assert pattern.dtype == np.float32
        
        # Verify that the pattern has proper values (should be in range [-1, 1])
        assert np.min(pattern) >= -1.0
        assert np.max(pattern) <= 1.0


class TestFieldCoherence:
    """Tests for field coherence calculation functions"""
    
    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_field_coherence_calculation(self, size):
        """Test field coherence calculation"""
        width, height = size
        
        # Generate a test field
        field = qc.generate_quantum_field_cpu(width, height, 'love', 0)
        
        # Calculate coherence
        coherence = qc.calculate_field_coherence_cpu(field)
        
        # Coherence should be a single value
        assert isinstance(coherence, float)
        
        # Coherence should be within a reasonable range
        assert 0 <= coherence <= sc.PHI * 2

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_cpu_gpu_coherence_equivalence(self, size):
        """Test that CPU and GPU coherence calculations produce similar results"""
        width, height = size
        
        # Generate a test field
        field = qc.generate_quantum_field_cpu(width, height, 'love', 0)
        
        # Calculate coherence on CPU
        cpu_coherence = qc.calculate_field_coherence_cpu(field)
        
        # Only run if CUDA is available
        if qc.CUDA_AVAILABLE and qc.initialize_cuda():
            # Calculate coherence on GPU
            gpu_coherence = qc.calculate_field_coherence_cuda(field)
            
            # The implementations won't be exactly the same due to different
            # sampling methods, but they should be within a reasonable range
            assert abs(gpu_coherence - cpu_coherence) < 0.5, \
                   f"CPU: {cpu_coherence}, GPU: {gpu_coherence}"
        else:
            pytest.skip("CUDA not available")


class TestPerformance:
    """Performance tests for quantum field functions"""
    
    @pytest.mark.parametrize("size", [(256, 256), (512, 512)])
    def test_gpu_speedup(self, size):
        """Test that GPU implementation is faster than CPU (if available)"""
        # Skip if CUDA is not available
        if not (qc.CUDA_AVAILABLE and qc.initialize_cuda()):
            pytest.skip("CUDA not available")
            
        width, height = size
        iterations = 3
        
        # Import time module
        import time
        
        # Benchmark CPU implementation
        start_time = time.time()
        for i in range(iterations):
            qc.generate_quantum_field_cpu(width, height, 'love', i*0.1)
        cpu_time = time.time() - start_time
        
        # Benchmark CUDA implementation
        start_time = time.time()
        for i in range(iterations):
            qc.generate_quantum_field_cuda(width, height, 'love', i*0.1)
        cuda_time = time.time() - start_time
        
        # CUDA should be faster
        assert cuda_time < cpu_time, \
               f"CUDA implementation ({cuda_time:.2f}s) is slower than CPU ({cpu_time:.2f}s)"
        
        # Log the speedup (not an assertion, just for information)
        speedup = cpu_time / cuda_time
        print(f"\nCUDA Speedup for {width}x{height}: {speedup:.2f}x")


# Integration tests
def test_integration_field_to_ascii():
    """Test the field to ASCII conversion"""
    width, height = 20, 10
    field = qc.generate_quantum_field(width, height)
    
    # Convert to ASCII
    ascii_art = qc.field_to_ascii(field)
    
    # Check output
    assert len(ascii_art) == height
    assert all(len(row) == width for row in ascii_art)