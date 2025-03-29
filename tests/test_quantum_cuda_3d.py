"""
Tests for the 3D field generation capabilities in the CUDA backend.
"""

import pytest
import numpy as np
from quantum_field.backends.cuda import CUDABackend
from quantum_field.visualization3d import generate_3d_quantum_field, calculate_3d_field_coherence
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# Skip all tests if CUDA is not available
try:
    from cuda.core.experimental import Device
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


@pytest.fixture
def cuda_backend():
    """Create and initialize a CUDA backend instance."""
    backend = CUDABackend()
    assert backend.initialize(), "CUDA backend initialization failed"
    yield backend
    backend.shutdown()


def test_cuda_3d_capability_flag(cuda_backend):
    """Test that the 3D capability flag is set correctly."""
    capabilities = cuda_backend.capabilities
    assert "3d_fields" in capabilities
    assert capabilities["3d_fields"] is True
    assert cuda_backend.has_3d_capability is True


def test_basic_3d_field_generation(cuda_backend):
    """Test basic 3D quantum field generation."""
    # Generate a small 3D field
    width, height, depth = 32, 32, 32
    field = cuda_backend.generate_3d_quantum_field(
        width, height, depth, frequency_name="love"
    )
    
    # Check shape and basic properties
    assert field.shape == (depth, height, width)
    assert field.dtype == np.float32
    assert not np.isnan(field).any(), "Field contains NaN values"
    assert not np.isinf(field).any(), "Field contains Inf values"
    
    # Check value range (should be between -1 and 1)
    assert np.min(field) >= -1.0, f"Minimum value ({np.min(field)}) is too low"
    assert np.max(field) <= 1.0, f"Maximum value ({np.max(field)}) is too high"


def test_cuda_vs_cpu_3d_field_generation():
    """Test that CUDA and CPU 3D field generation produce similar results."""
    # Parameters
    width, height, depth = 16, 16, 16  # Small field for quick test
    frequency_name = "love"
    
    # Generate fields with both methods
    cpu_field = generate_3d_quantum_field(
        width, height, depth, frequency_name=frequency_name, backend=None
    )
    
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    cuda_field = cuda_backend.generate_3d_quantum_field(
        width, height, depth, frequency_name=frequency_name
    )
    cuda_backend.shutdown()
    
    # Compare fields by sampling points
    # Note: Due to floating point differences, we can't expect exact equality
    # Sample points at regular intervals
    samples_x = np.linspace(0, width-1, 4, dtype=int)
    samples_y = np.linspace(0, height-1, 4, dtype=int)
    samples_z = np.linspace(0, depth-1, 4, dtype=int)
    
    for z in samples_z:
        for y in samples_y:
            for x in samples_x:
                # Allow for small floating point differences
                assert abs(cpu_field[z, y, x] - cuda_field[z, y, x]) < 0.1, \
                    f"Fields differ significantly at ({x}, {y}, {z}): CPU={cpu_field[z, y, x]}, CUDA={cuda_field[z, y, x]}"


def test_3d_field_coherence_calculation(cuda_backend):
    """Test 3D quantum field coherence calculation."""
    # Generate a 3D field
    width, height, depth = 32, 32, 32
    field = cuda_backend.generate_3d_quantum_field(
        width, height, depth, frequency_name="love"
    )
    
    # Calculate coherence
    coherence = cuda_backend.calculate_3d_field_coherence(field)
    
    # Check coherence value
    assert 0.0 <= coherence <= 1.0, f"Coherence value {coherence} is out of range [0, 1]"
    print(f"3D field coherence: {coherence}")


def test_different_frequency_effects():
    """Test how different frequencies affect 3D field patterns."""
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    
    # Parameters
    width, height, depth = 32, 32, 32
    frequency_names = ["love", "unity", "truth", "vision"]
    
    # Generate fields with different frequencies
    fields = {}
    coherence_values = {}
    
    for freq_name in frequency_names:
        field = cuda_backend.generate_3d_quantum_field(
            width, height, depth, frequency_name=freq_name
        )
        fields[freq_name] = field
        coherence_values[freq_name] = cuda_backend.calculate_3d_field_coherence(field)
    
    cuda_backend.shutdown()
    
    # Verify fields are different
    for i, name1 in enumerate(frequency_names):
        for name2 in frequency_names[i+1:]:
            difference = np.abs(fields[name1] - fields[name2]).mean()
            assert difference > 0.05, f"Fields for {name1} and {name2} are too similar"
    
    # Print coherence values
    for name, coherence in coherence_values.items():
        print(f"Frequency '{name}' coherence: {coherence}")


def test_time_evolution():
    """Test time evolution of 3D quantum fields."""
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    
    # Parameters
    width, height, depth = 32, 32, 32
    frequency_name = "love"
    time_steps = [0.0, 0.5, 1.0, 1.5]
    
    # Generate fields at different time steps
    fields = {}
    for time in time_steps:
        field = cuda_backend.generate_3d_quantum_field(
            width, height, depth, frequency_name=frequency_name, time_factor=time
        )
        fields[time] = field
    
    cuda_backend.shutdown()
    
    # Verify fields evolve over time
    for i, time1 in enumerate(time_steps):
        for time2 in time_steps[i+1:]:
            difference = np.abs(fields[time1] - fields[time2]).mean()
            assert difference > 0.01, f"Fields at time {time1} and {time2} are too similar"


def test_large_3d_field_generation():
    """Test generation of a larger 3D field that's more likely to use optimizations."""
    # Skip if there's insufficient memory
    try:
        import psutil
        available_memory = psutil.virtual_memory().available
        required_memory = 128 * 128 * 64 * 4 * 2  # Approx memory for float32 field with buffers
        if available_memory < required_memory:
            pytest.skip("Not enough memory available for large field test")
    except ImportError:
        pass  # Skip memory check if psutil not available
    
    # Initialize CUDA backend
    cuda_backend = CUDABackend()
    cuda_backend.initialize()
    
    # Generate a large 3D field
    width, height, depth = 128, 128, 64
    
    try:
        field = cuda_backend.generate_3d_quantum_field(
            width, height, depth, frequency_name="love"
        )
        
        # Basic validation
        assert field.shape == (depth, height, width)
        assert not np.isnan(field).any(), "Field contains NaN values"
        
        # If multi-GPU is available, this should have used it for this size
        if cuda_backend.multi_gpu_available and len(cuda_backend.devices) > 1:
            print(f"Large field test used multi-GPU with {len(cuda_backend.devices)} devices")
    except Exception as e:
        pytest.fail(f"Failed to generate large 3D field: {e}")
    finally:
        cuda_backend.shutdown()


if __name__ == "__main__":
    # Run simple test
    cuda_backend = CUDABackend()
    if cuda_backend.is_available():
        cuda_backend.initialize()
        # Test 3D generation
        field = cuda_backend.generate_3d_quantum_field(64, 64, 64, frequency_name="love")
        print(f"3D Field shape: {field.shape}")
        print(f"3D Field min/max: {field.min():.4f}/{field.max():.4f}")
        
        # Test coherence calculation
        coherence = cuda_backend.calculate_3d_field_coherence(field)
        print(f"3D Field coherence: {coherence:.4f}")
        
        cuda_backend.shutdown()
        print("CUDA 3D tests successful!")
    else:
        print("CUDA not available")