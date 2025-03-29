# Quantum Field Visualization: API Reference

This document provides a complete reference to all the modules, classes, and functions in the Quantum Field Visualization library.

## Core Module

### quantum_field

The main module provides simple functions for generating and analyzing quantum fields.

#### Functions

```python
def generate_quantum_field(width: int, height: int, 
                          frequency_name: Optional[str] = None, 
                          time_factor: float = 0.0,
                          custom_frequency: Optional[float] = None) -> np.ndarray:
    """
    Generate a quantum field visualization.
    
    Args:
        width: Width of the field in pixels
        height: Height of the field in pixels
        frequency_name: Name of the frequency to use (from SACRED_FREQUENCIES)
        time_factor: Time evolution factor (0.0 to 2π)
        custom_frequency: Custom frequency value (used if frequency_name is None)
        
    Returns:
        NumPy array containing the generated field
        
    Raises:
        QuantumError: If invalid parameters are provided
    """
```

```python
def calculate_field_coherence(field_data: np.ndarray) -> float:
    """
    Calculate the coherence factor of a quantum field.
    
    Args:
        field_data: NumPy array containing the field data
        
    Returns:
        Coherence factor between 0.0 and 1.0
        
    Raises:
        FieldCoherenceError: If coherence calculation fails
    """
```

```python
def generate_phi_pattern(width: int, height: int) -> np.ndarray:
    """
    Generate a Phi-based sacred geometry pattern.
    
    Args:
        width: Width of the pattern in pixels
        height: Height of the pattern in pixels
        
    Returns:
        NumPy array containing the generated pattern
    """
```

## Constants Module

### quantum_field.constants

This module provides the sacred constants used throughout the library.

#### Constants

```python
PHI = 1.618033988749895  # Golden ratio
LAMBDA = 0.618033988749895  # Divine complement (1/PHI)
PHI_PHI = 2.1784575679375995  # PHI ** PHI

SACRED_FREQUENCIES = {
    'love': 528,      # Creation/healing
    'unity': 432,     # Grounding/stability 
    'cascade': 594,   # Heart-centered integration
    'truth': 672,     # Voice expression
    'vision': 720,    # Expanded perception
    'oneness': 768,   # Unity consciousness
}
```

## Backend Architecture

### quantum_field.backends

This module provides the multi-accelerator backend architecture.

#### Functions

```python
def get_backend(name: Optional[str] = None) -> AcceleratorBackend:
    """
    Get the best available backend or a specific backend by name.
    
    Args:
        name: Optional name of the specific backend to get
        
    Returns:
        An instance of AcceleratorBackend
        
    Raises:
        ValueError: If the requested backend is not available
    """
```

```python
def list_available_backends() -> List[AcceleratorBackend]:
    """
    List all available backends sorted by priority (highest first).
    
    Returns:
        List of available backend instances
    """
```

#### Classes

```python
class AcceleratorBackend(ABC):
    """
    Abstract base class for all accelerator backends.
    
    Attributes:
        name (str): Backend name
        priority (int): Backend priority (0-100, higher is better)
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system"""
        
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Get the capabilities of this backend"""
        
    @abstractmethod
    def generate_quantum_field(self, width: int, height: int, 
                             frequency_name: Optional[str] = None,
                             time_factor: float = 0.0,
                             custom_frequency: Optional[float] = None) -> np.ndarray:
        """Generate a quantum field with this backend"""
        
    @abstractmethod
    def calculate_field_coherence(self, field_data: np.ndarray) -> float:
        """Calculate the coherence of a quantum field with this backend"""
        
    @abstractmethod
    def generate_phi_pattern(self, width: int, height: int) -> np.ndarray:
        """Generate a Phi-based sacred pattern with this backend"""
    
    def to_dlpack(self, field_data: np.ndarray) -> Any:
        """
        Convert field data to DLPack format for interoperability with ML frameworks.
        
        Args:
            field_data: NumPy array containing the field data
            
        Returns:
            DLPack capsule that can be used with ML frameworks
            
        Raises:
            RuntimeError: If DLPack conversion is not supported by this backend
        """
    
    def from_dlpack(self, dlpack_tensor: Any, 
                  shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert DLPack tensor to NumPy array.
        
        Args:
            dlpack_tensor: DLPack capsule from an ML framework
            shape: Optional shape to reshape the resulting array
            
        Returns:
            NumPy array containing the converted data
            
        Raises:
            RuntimeError: If DLPack conversion is not supported by this backend
        """
        
    def shutdown(self) -> None:
        """Release resources used by this backend"""
```

### CPU Backend

```python
class CPUBackend(AcceleratorBackend):
    """
    CPU implementation of the accelerator backend.
    
    Attributes:
        name (str): "cpu"
        priority (int): 0 (lowest priority, always available as fallback)
    """
```

### CUDA Backend

```python
class CUDABackend(AcceleratorBackend):
    """
    NVIDIA CUDA implementation of the accelerator backend.
    
    Attributes:
        name (str): "cuda" 
        priority (int): 70
    """
```

### ROCm Backend

```python
class ROCmBackend(AcceleratorBackend):
    """
    AMD ROCm implementation of the accelerator backend.
    
    Attributes:
        name (str): "rocm"
        priority (int): 80
    """
```

### Other Backends

Additional backends with similar structure:
- TenstorrentBackend (priority: 85)
- MobileBackend (priority: 60)
- HuaweiAscendBackend (priority: 75)
- OneAPIBackend (priority: 75)

## Advanced Features

### CUDA Graphs

```python
class CUDAGraphsManager:
    """
    Manager for creating and running CUDA graphs for efficient repetitive execution.
    
    Methods:
        create_field_generation_graph(width, height, frequency_name, base_time=0.0):
            Capture field generation as a CUDA graph
            
        run_graph_with_time_factor(time_factor):
            Execute the captured graph with an updated time factor
            
        shutdown():
            Release resources
    """
```

### Multi-GPU Support

```python
def generate_field_multi_gpu(width: int, height: int, 
                           frequency_name: str, 
                           time_factor: float = 0.0) -> np.ndarray:
    """
    Generate a quantum field using multiple GPUs if available.
    
    Args:
        width: Width of the field in pixels
        height: Height of the field in pixels
        frequency_name: Name of the frequency to use
        time_factor: Time evolution factor
        
    Returns:
        NumPy array containing the generated field
    """
```

### Thread Block Clusters

```python
def generate_field_with_clusters(width: int, height: int, 
                               frequency_name: str, 
                               time_factor: float = 0.0) -> np.ndarray:
    """
    Generate a quantum field using thread block clusters for newer NVIDIA GPUs.
    
    Args:
        width: Width of the field in pixels
        height: Height of the field in pixels
        frequency_name: Name of the frequency to use
        time_factor: Time evolution factor
        
    Returns:
        NumPy array containing the generated field
        
    Raises:
        RuntimeError: If thread block clusters are not supported
    """
```

## Exception Handling

```python
class QuantumError(Exception):
    """Base class for quantum-related errors"""
    pass

class FieldCoherenceError(QuantumError):
    """Error in quantum field coherence calculation"""
    pass

class BackendError(QuantumError):
    """Error related to backend operations"""
    pass

class UnsupportedDeviceError(BackendError):
    """Error when hardware is unsupported"""
    pass
```

## Version and Compatibility

```python
def check_version_compatibility(min_version: str, max_version: Optional[str] = None) -> bool:
    """
    Check if the current version is compatible with the specified range.
    
    Args:
        min_version: Minimum supported version
        max_version: Maximum supported version (optional)
        
    Returns:
        True if compatible, False otherwise
    """
```

```python
def get_version() -> str:
    """
    Get the current version of the library.
    
    Returns:
        Version string in format "X.Y.Z"
    """
```

## DLPack Integration

DLPack methods in the AcceleratorBackend classes:

```python
def to_dlpack(self, field_data: np.ndarray) -> Any:
    """
    Convert a NumPy array to DLPack format.
    
    Args:
        field_data: NumPy array to convert
        
    Returns:
        DLPack tensor that can be used with ML frameworks
        
    Raises:
        RuntimeError: If DLPack conversion is not supported
    """
```

```python
def from_dlpack(self, dlpack_tensor: Any, 
              shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convert a DLPack tensor to a NumPy array.
    
    Args:
        dlpack_tensor: DLPack tensor to convert
        shape: Optional shape to reshape the resulting array
        
    Returns:
        NumPy array containing the converted data
        
    Raises:
        RuntimeError: If DLPack conversion is not supported
    """
```

## Utility Functions

Various utility functions for internal use:

```python
def detect_available_backends() -> List[AcceleratorBackend]:
    """Detect all available backends on the current system"""
```

```python
def register_backend(backend_class: Type[AcceleratorBackend]) -> None:
    """Register a new backend class with the system"""
```

```python
def optimal_backend_for_size(width: int, height: int) -> AcceleratorBackend:
    """Select the optimal backend for a given field size"""
```