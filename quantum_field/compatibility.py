"""
Version compatibility module for quantum field visualization

This module checks compatibility with required libraries and CUDA versions,
and handles graceful degradation when requirements are not met.
"""

import os
import sys
import platform
import subprocess
import importlib.util
from typing import Dict, Tuple, List, Optional, Union, Any

# Required versions
REQUIRED_VERSIONS = {
    "numpy": "1.20.0",
    "matplotlib": "3.5.0",
    "pillow": "9.0.0",
    "cuda-python": "12.0.0",
    "cupy": "12.0.0",
}

class VersionInfo:
    """Class representing version information"""
    
    def __init__(self, major: int = 0, minor: int = 0, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    @classmethod
    def from_string(cls, version_str: str) -> 'VersionInfo':
        """Parse a version string into a VersionInfo object"""
        try:
            parts = version_str.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2].split('+')[0].split('-')[0]) if len(parts) > 2 else 0
            return cls(major, minor, patch)
        except (ValueError, IndexError):
            return cls(0, 0, 0)
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __ge__(self, other: 'VersionInfo') -> bool:
        if self.major > other.major:
            return True
        if self.major == other.major and self.minor > other.minor:
            return True
        if self.major == other.major and self.minor == other.minor and self.patch >= other.patch:
            return True
        return False


def get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package"""
    try:
        if package_name == "cuda-python":
            # Special case for cuda-python
            if importlib.util.find_spec("cuda") is not None:
                try:
                    import cuda
                    return getattr(cuda, "__version__", None) or "Unknown"
                except ImportError:
                    return None
            return None
        
        if package_name == "cupy":
            # Special case for cupy (could be installed as cupy-cudaXX)
            for variant in ["cupy", "cupy-cuda12x", "cupy-cuda11x", "cupy-cuda10x"]:
                if importlib.util.find_spec(variant) is not None:
                    try:
                        module = importlib.import_module(variant)
                        return getattr(module, "__version__", None) or "Unknown"
                    except ImportError:
                        continue
            return None
        
        # Standard case
        if importlib.util.find_spec(package_name) is not None:
            module = importlib.import_module(package_name)
            return getattr(module, "__version__", None) or "Unknown"
        return None
    except Exception:
        return None


def get_cuda_version() -> Optional[str]:
    """Get the installed CUDA toolkit version"""
    try:
        # Try using nvcc
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Example: "Cuda compilation tools, release 11.2, V11.2.152"
            for line in output.split('\n'):
                if "release" in line and "V" in line:
                    parts = line.split("release")[1].split(",")[0].strip()
                    return parts
        
        # If nvcc fails, try nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if "CUDA Version:" in line:
                    parts = line.split("CUDA Version:")[1].strip()
                    return parts
        
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_compute_capability() -> Optional[Tuple[int, int]]:
    """Get the compute capability of the installed GPU"""
    try:
        # Try to use cuda-python if available
        if importlib.util.find_spec("cuda") is not None:
            try:
                from cuda.core.experimental import Device
                device = Device(0)  # Use the first GPU
                return device.compute_capability
            except Exception:
                pass
        
        # Try to use cupy if available
        if importlib.util.find_spec("cupy") is not None:
            try:
                import cupy as cp
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                major = device_props["major"]
                minor = device_props["minor"]
                return (major, minor)
            except Exception:
                pass
        
        # Try nvidia-smi as a last resort
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_capability", "--format=csv,noheader"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if "." in output:
                major, minor = output.split(".")
                return (int(major), int(minor))
        
        return None
    except (subprocess.SubprocessError, FileNotFoundError, ImportError):
        return None


def check_compatibility() -> Dict[str, Any]:
    """
    Check compatibility with required libraries and CUDA versions
    
    Returns:
        Dictionary with compatibility information
    """
    results = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "packages": {},
        "cuda": {
            "available": False,
            "version": None,
            "compute_capability": None,
            "compute_capability_name": None,
            "thread_block_cluster_support": False,
            "multi_gpu_support": False,
        },
        "overall_compatibility": "unknown"
    }
    
    # Check package versions
    missing_packages = []
    incompatible_packages = []
    
    for package, required_version in REQUIRED_VERSIONS.items():
        actual_version = get_package_version(package)
        
        if actual_version is None:
            if package in ["cuda-python", "cupy"]:
                # These are optional packages
                results["packages"][package] = {
                    "installed": False,
                    "required": required_version,
                    "actual": None,
                    "compatible": True  # Mark as compatible since they're optional
                }
            else:
                # Required packages
                results["packages"][package] = {
                    "installed": False,
                    "required": required_version,
                    "actual": None,
                    "compatible": False
                }
                missing_packages.append(package)
        else:
            # Check if version is compatible
            required = VersionInfo.from_string(required_version)
            actual = VersionInfo.from_string(actual_version)
            compatible = actual >= required
            
            results["packages"][package] = {
                "installed": True,
                "required": str(required),
                "actual": str(actual),
                "compatible": compatible
            }
            
            if not compatible and package not in ["cuda-python", "cupy"]:
                incompatible_packages.append(f"{package} (required: {required}, found: {actual})")
    
    # Check CUDA compatibility
    cuda_version = get_cuda_version()
    if cuda_version:
        results["cuda"]["available"] = True
        results["cuda"]["version"] = cuda_version
        
        # Get compute capability
        compute_capability = get_compute_capability()
        if compute_capability:
            major, minor = compute_capability
            results["cuda"]["compute_capability"] = f"{major}.{minor}"
            
            # Map compute capability to architecture name
            arch_names = {
                (7, 0): "Volta",
                (7, 5): "Turing",
                (8, 0): "Ampere",
                (8, 6): "Ampere",
                (8, 9): "Ada Lovelace",
                (9, 0): "Hopper",
            }
            
            results["cuda"]["compute_capability_name"] = arch_names.get(
                (major, minor), 
                f"Unknown (sm_{major}{minor})"
            )
            
            # Check for thread block cluster support (requires Hopper, sm_90)
            results["cuda"]["thread_block_cluster_support"] = major >= 9
            
            # Check for multi-GPU support
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split('\n'))
                    results["cuda"]["multi_gpu_support"] = gpu_count > 1
                    results["cuda"]["gpu_count"] = gpu_count
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
    
    # Determine overall compatibility
    if missing_packages:
        results["overall_compatibility"] = "missing_required_packages"
        results["missing_packages"] = missing_packages
    elif incompatible_packages:
        results["overall_compatibility"] = "incompatible_package_versions"
        results["incompatible_packages"] = incompatible_packages
    else:
        # Core packages are compatible
        if not results["cuda"]["available"]:
            results["overall_compatibility"] = "compatible_cpu_only"
        else:
            if results["packages"]["cuda-python"]["installed"] and results["packages"]["cupy"]["installed"]:
                results["overall_compatibility"] = "compatible_full"
            else:
                results["overall_compatibility"] = "compatible_partial"
    
    return results


def print_compatibility_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted compatibility report
    
    Args:
        results: Dictionary with compatibility information from check_compatibility()
    """
    print("\n" + "=" * 80)
    print("QUANTUM FIELD COMPATIBILITY REPORT")
    print("=" * 80)
    
    print(f"System: {results['system']}")
    print(f"Python Version: {results['python_version']}")
    print("\nPackage Compatibility:")
    
    for package, info in results["packages"].items():
        if info["installed"]:
            status = "✓ " if info["compatible"] else "✗ "
            print(f"  {status}{package}: {info['actual']} (required: {info['required']})")
        else:
            if package in ["cuda-python", "cupy"]:
                print(f"  ○ {package}: Not installed (optional)")
            else:
                print(f"  ✗ {package}: Not installed (required: {info['required']})")
    
    print("\nCUDA Compatibility:")
    if results["cuda"]["available"]:
        print(f"  ✓ CUDA: {results['cuda']['version']}")
        
        if results["cuda"]["compute_capability"]:
            print(f"  ✓ Compute Capability: {results['cuda']['compute_capability']} ({results['cuda']['compute_capability_name']})")
            
            # Thread block cluster support
            if results["cuda"]["thread_block_cluster_support"]:
                print(f"  ✓ Thread Block Cluster: Supported")
            else:
                print(f"  ○ Thread Block Cluster: Not supported (requires Hopper/H100 GPU)")
            
            # Multi-GPU support
            if results["cuda"]["multi_gpu_support"]:
                print(f"  ✓ Multi-GPU: Supported ({results['cuda'].get('gpu_count', '?')} GPUs detected)")
            else:
                print(f"  ○ Multi-GPU: Not supported (only 1 GPU detected)")
    else:
        print(f"  ○ CUDA: Not available")
    
    print("\nOverall Compatibility:")
    overall = results["overall_compatibility"]
    
    if overall == "compatible_full":
        print("  ✓ FULLY COMPATIBLE: All requirements are met with GPU acceleration")
    elif overall == "compatible_partial":
        print("  ✓ PARTIALLY COMPATIBLE: Core requirements are met, but some CUDA libraries are missing")
        print("  ⓘ GPU acceleration will be limited or unavailable")
    elif overall == "compatible_cpu_only":
        print("  ✓ CPU COMPATIBLE: Core requirements are met, but CUDA is not available")
        print("  ⓘ GPU acceleration will be unavailable")
    elif overall == "missing_required_packages":
        print("  ✗ INCOMPATIBLE: Missing required packages")
        print("  ⓘ Missing packages: " + ", ".join(results["missing_packages"]))
    elif overall == "incompatible_package_versions":
        print("  ✗ INCOMPATIBLE: Incompatible package versions")
        print("  ⓘ Incompatible packages: " + ", ".join(results["incompatible_packages"]))
    else:
        print("  ? UNKNOWN: Compatibility could not be determined")
    
    print("\nRecommendations:")
    if overall.startswith("incompatible"):
        print("  • Install or update required packages:")
        if "missing_packages" in results:
            for package in results["missing_packages"]:
                req_version = results["packages"][package]["required"]
                print(f"    - pip install {package}>={req_version}")
        
        if "incompatible_packages" in results:
            for package_info in results["incompatible_packages"]:
                package = package_info.split(" ")[0]
                req_version = results["packages"][package]["required"]
                print(f"    - pip install --upgrade {package}>={req_version}")
    elif overall == "compatible_partial":
        print("  • To enable full GPU acceleration, install:")
        
        if not results["packages"]["cuda-python"]["installed"]:
            req_version = results["packages"]["cuda-python"]["required"]
            print(f"    - pip install cuda-python>={req_version}")
        
        if not results["packages"]["cupy"]["installed"]:
            req_version = results["packages"]["cupy"]["required"]
            print(f"    - pip install cupy-cuda11x>={req_version}  # Adjust cuda version as needed")
    elif overall == "compatible_cpu_only":
        print("  • To enable GPU acceleration, install CUDA toolkit and:")
        print(f"    - pip install cuda-python>={REQUIRED_VERSIONS['cuda-python']}")
        print(f"    - pip install cupy-cuda11x>={REQUIRED_VERSIONS['cupy']}  # Adjust cuda version as needed")
    
    print("=" * 80)


if __name__ == "__main__":
    results = check_compatibility()
    print_compatibility_report(results)