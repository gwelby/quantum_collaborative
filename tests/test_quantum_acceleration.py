#!/usr/bin/env python3
"""
Tests for quantum_acceleration.py module

These tests verify the benchmarking functionality for
quantum field acceleration, comparing CPU and GPU performance.
"""

import os
import pytest
import numpy as np

# Import the module to test
try:
    import quantum_acceleration as qa
    import sacred_constants as sc
except ImportError:
    pytest.skip("quantum_acceleration module not found", allow_module_level=True)


def test_module_imported():
    """Verify that the module can be imported"""
    assert qa is not None
    assert hasattr(qa, 'benchmark_implementations')
    assert hasattr(qa, 'benchmark_frequencies')
    assert hasattr(qa, 'benchmark_thread_blocks')


class TestCPUFallback:
    """Tests for CPU fallback implementation"""
    
    def test_generate_field_cpu_fallback(self):
        """Test the CPU fallback implementation for field generation"""
        width, height = 32, 16
        field = qa.generate_field_cpu_fallback(width, height, 'love')
        
        # Check shape and type
        assert field.shape == (height, width)
        assert field.dtype == np.float32
        
        # Check value range (should be between -1 and 1)
        assert np.min(field) >= -1.0
        assert np.max(field) <= 1.0


class TestBenchmarking:
    """Tests for benchmarking functions"""
    
    def test_benchmark_implementations_return_structure(self):
        """Test that benchmark_implementations returns the correct data structure"""
        # Use small sizes for faster testing
        sizes = [(16, 8), (32, 16)]
        results = qa.benchmark_implementations(sizes, 'love', iterations=1)
        
        # Check that the results dictionary contains the expected keys
        assert 'sizes' in results
        assert 'cpu_times' in results
        assert 'speedups' in results
        
        # Check that the data has the correct shape
        assert len(results['sizes']) == len(sizes)
        assert len(results['cpu_times']) == len(sizes)
        
        # Check that CPU times are positive
        assert all(t > 0 for t in results['cpu_times'])
    
    def test_benchmark_frequencies_return_structure(self):
        """Test that benchmark_frequencies returns the correct data structure"""
        # Use small sizes for faster testing
        width, height = 16, 16
        results = qa.benchmark_frequencies(width, height, iterations=1)
        
        # Check that the results dictionary contains the expected keys
        assert 'frequencies' in results
        assert 'cpu_times' in results
        
        # Check that the data has the correct shape
        assert len(results['frequencies']) == len(sc.SACRED_FREQUENCIES)
        assert len(results['cpu_times']) == len(sc.SACRED_FREQUENCIES)
        
        # Check that CPU times are positive
        assert all(t > 0 for t in results['cpu_times'])
    
    @pytest.mark.skipif(not qa.CUDA_AVAILABLE, reason="CUDA not available")
    def test_benchmark_thread_blocks(self):
        """Test thread block benchmarking (skip if CUDA not available)"""
        # Use small sizes for faster testing
        width, height = 64, 64
        results = qa.benchmark_thread_blocks(width, height, iterations=1)
        
        # Check if the results are non-None (indicating the benchmark ran)
        assert results is not None
        
        # Check that the results dictionary contains the expected keys
        assert 'block_sizes' in results
        assert 'times' in results
        assert 'relative_performance' in results
        
        # Check that we have at least some valid measurements
        valid_times = [t for t in results['times'] if t != float('inf')]
        assert len(valid_times) > 0


class TestPlotting:
    """Tests for plotting functions (testing they don't raise exceptions)"""
    
    def test_plot_results_no_exception(self):
        """Test that plot_results doesn't raise exceptions"""
        # Create a minimal results dictionary
        results = {
            'sizes': [(16, 8), (32, 16)],
            'cpu_times': [0.1, 0.2],
            'gpu_times': [0.05, 0.1],
            'speedups': [2.0, 2.0]
        }
        
        # This should not raise an exception
        qa.plot_results(results)
        
        # Clean up the file if it was created
        if os.path.exists('quantum_acceleration_benchmark.png'):
            os.remove('quantum_acceleration_benchmark.png')
    
    def test_plot_frequency_results_no_exception(self):
        """Test that plot_frequency_results doesn't raise exceptions"""
        # Create a minimal results dictionary
        results = {
            'frequencies': ['love', 'unity'],
            'cpu_times': [0.1, 0.2],
            'gpu_times': [0.05, 0.1],
            'coherence_values': [1.5, 1.6]
        }
        
        # This should not raise an exception
        qa.plot_frequency_results(results)
        
        # Clean up the file if it was created
        if os.path.exists('frequency_benchmark.png'):
            os.remove('frequency_benchmark.png')