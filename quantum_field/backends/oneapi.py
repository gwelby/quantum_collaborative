"""Intel oneAPI Backend for Intel GPUs"""

# Placeholder for Intel GPU acceleration via oneAPI
# This will be implemented in a future update

from quantum_field.backends import AcceleratorBackend

class OneAPIBackend(AcceleratorBackend):
    name = "oneapi"
    priority = 75
    
    def is_available(self) -> bool:
        return False  # Not implemented yet
