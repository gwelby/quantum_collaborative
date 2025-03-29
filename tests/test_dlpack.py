"""
Tests for DLPack integration with ML frameworks
"""

import unittest
import numpy as np

# Try to import required modules (tests will be skipped if not available)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    from cuda.core.experimental import Device
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Import quantum field modules
from quantum_field.backends import get_backend


class DLPackIntegrationTests(unittest.TestCase):
    """Test DLPack integration for interoperability with ML frameworks"""
    
    def setUp(self):
        """Set up test environment"""
        # Get backends
        self.cuda_backend = None
        self.cpu_backend = get_backend("cpu")
        
        if HAS_CUPY:
            try:
                self.cuda_backend = get_backend("cuda")
                if not self.cuda_backend.capabilities.get("dlpack_support", False):
                    self.cuda_backend = None
            except ValueError:
                pass
        
        # Skip test if no DLPack-capable backend is available
        if (self.cuda_backend is None and 
            not self.cpu_backend.capabilities.get("dlpack_support", False)):
            self.skipTest("No backend with DLPack support available")
    
    def test_to_dlpack_cpu(self):
        """Test converting NumPy array to DLPack with CPU backend"""
        if not self.cpu_backend.capabilities.get("dlpack_support", False):
            self.skipTest("CPU backend doesn't support DLPack")
        
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for DLPack testing")
        
        # Generate a field with CPU backend
        field_data = self.cpu_backend.generate_quantum_field(64, 64, "love", 0.0)
        
        # Convert to DLPack format
        dlpack_tensor = self.cpu_backend.to_dlpack(field_data)
        
        # Convert DLPack to PyTorch tensor
        torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(torch_tensor.shape, field_data.shape)
        self.assertTrue(np.allclose(torch_tensor.numpy(), field_data))
    
    def test_from_dlpack_cpu(self):
        """Test converting DLPack to NumPy array with CPU backend"""
        if not self.cpu_backend.capabilities.get("dlpack_support", False):
            self.skipTest("CPU backend doesn't support DLPack")
        
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for DLPack testing")
        
        # Create PyTorch tensor
        torch_tensor = torch.rand(64, 64)
        
        # Convert to DLPack
        dlpack_tensor = torch.utils.dlpack.to_dlpack(torch_tensor)
        
        # Convert DLPack to NumPy via CPU backend
        numpy_array = self.cpu_backend.from_dlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(numpy_array.shape, torch_tensor.shape)
        self.assertTrue(np.allclose(numpy_array, torch_tensor.numpy()))
    
    @unittest.skipIf(not HAS_CUPY, "CuPy not available")
    def test_to_dlpack_cuda(self):
        """Test converting CuPy array to DLPack with CUDA backend"""
        if self.cuda_backend is None:
            self.skipTest("CUDA backend not available")
        
        # Generate a field with CUDA backend
        field_data = self.cuda_backend.generate_quantum_field(64, 64, "love", 0.0)
        
        # Convert to DLPack format
        dlpack_tensor = self.cuda_backend.to_dlpack(field_data)
        
        # Convert DLPack back to CuPy to verify
        cupy_array = cp.fromDlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(cupy_array.shape, field_data.shape)
        self.assertTrue(np.allclose(cp.asnumpy(cupy_array), field_data))
    
    @unittest.skipIf(not HAS_CUPY, "CuPy not available")
    def test_from_dlpack_cuda(self):
        """Test converting DLPack to CuPy array with CUDA backend"""
        if self.cuda_backend is None:
            self.skipTest("CUDA backend not available")
        
        # Create CuPy array
        cupy_array = cp.random.rand(64, 64).astype(cp.float32)
        
        # Convert to DLPack
        dlpack_tensor = cupy_array.toDlpack()
        
        # Convert DLPack to NumPy via CUDA backend
        numpy_array = self.cuda_backend.from_dlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(numpy_array.shape, cupy_array.shape)
        self.assertTrue(np.allclose(numpy_array, cp.asnumpy(cupy_array)))
    
    @unittest.skipIf(not (HAS_TORCH and HAS_CUPY), "PyTorch or CuPy not available")
    def test_framework_interoperability(self):
        """Test interoperability between PyTorch and CUDA backend"""
        if self.cuda_backend is None:
            self.skipTest("CUDA backend not available")
        
        # Create PyTorch tensor on CUDA
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA not available")
        
        # Create CUDA tensor
        torch_tensor = torch.rand(64, 64, device="cuda")
        
        # Convert to DLPack
        dlpack_tensor = torch.utils.dlpack.to_dlpack(torch_tensor)
        
        # Convert DLPack to NumPy via CUDA backend
        numpy_array = self.cuda_backend.from_dlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(numpy_array.shape, torch_tensor.shape)
        self.assertTrue(np.allclose(numpy_array, torch_tensor.cpu().numpy()))
        
        # Now go back to PyTorch
        field_data = self.cuda_backend.generate_quantum_field(64, 64, "love", 0.5)
        dlpack_tensor = self.cuda_backend.to_dlpack(field_data)
        torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
        
        # Check shape and value consistency
        self.assertEqual(torch_tensor.shape, field_data.shape)
        self.assertTrue(np.allclose(torch_tensor.cpu().numpy(), field_data))
    
    def test_ml_example(self):
        """Demonstrate a simple ML use case with quantum fields"""
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for ML example")
        
        if self.cuda_backend and self.cuda_backend.capabilities.get("dlpack_support", False):
            backend = self.cuda_backend
        elif self.cpu_backend.capabilities.get("dlpack_support", False):
            backend = self.cpu_backend
        else:
            self.skipTest("No backend with DLPack support available")
        
        # Generate quantum fields with different parameters
        fields = []
        for t in np.linspace(0, 1, 10):
            field = backend.generate_quantum_field(128, 128, "love", t)
            dlpack_tensor = backend.to_dlpack(field)
            torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
            # Add batch dimension
            fields.append(torch_tensor.unsqueeze(0))
        
        # Stack tensors to create a batch
        batch = torch.cat(fields, dim=0)
        
        # Create a simple CNN model (just as an example, not for training)
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.pool = torch.nn.MaxPool2d(2)
                self.conv2 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
                self.fc = torch.nn.Linear(8 * 32 * 32, 1)
            
            def forward(self, x):
                # Add channel dimension if needed
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        # Create model instance
        model = SimpleModel()
        
        # Add channel dimension to batch
        batch = batch.unsqueeze(1)
        
        # Forward pass (just to verify everything works)
        with torch.no_grad():
            output = model(batch)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 1))
        
        # Just verify tensors are finite (not NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()