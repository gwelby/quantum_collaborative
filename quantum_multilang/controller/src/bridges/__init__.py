"""
Language Bridge Initialization

This module initializes all available language bridges for the
quantum field multi-language architecture.
"""

import logging
import importlib.util
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridges")

# Try to import language bridges
AVAILABLE_BRIDGES = {}

def _try_import_bridge(name: str) -> bool:
    """
    Try to import a language bridge module.
    
    Args:
        name: Name of the bridge module
        
    Returns:
        True if bridge is available, False otherwise
    """
    try:
        module = importlib.import_module(f"controller.src.bridges.{name}_bridge")
        
        # Check if the module has an initialize function
        if hasattr(module, 'initialize'):
            available = module.initialize()
            AVAILABLE_BRIDGES[name] = available
            logger.info(f"{name.capitalize()} bridge {'available' if available else 'unavailable'}")
            return available
        else:
            logger.warning(f"{name.capitalize()} bridge lacks initialize() function")
            AVAILABLE_BRIDGES[name] = False
            return False
    except ImportError:
        logger.warning(f"{name.capitalize()} bridge not found")
        AVAILABLE_BRIDGES[name] = False
        return False
    except Exception as e:
        logger.error(f"Error initializing {name} bridge: {e}")
        AVAILABLE_BRIDGES[name] = False
        return False

# Initialize all bridges
def initialize_bridges():
    """Initialize all available language bridges."""
    logger.info("Initializing language bridges...")
    
    # List of bridge modules to try
    bridges = [
        "rust",
        "cpp",
        "julia",
        "go",
        "wasm",
        "zig",
        "phiflow",
        "gregscript"
    ]
    
    # Try to import each bridge
    for bridge in bridges:
        _try_import_bridge(bridge)
    
    # Log available bridges
    available = [name for name, available in AVAILABLE_BRIDGES.items() if available]
    logger.info(f"Available bridges: {', '.join(available) if available else 'none'}")
    
    return AVAILABLE_BRIDGES

# Get available bridges
def get_available_bridges() -> Dict[str, bool]:
    """
    Get available language bridges.
    
    Returns:
        Dictionary of bridge names and availability
    """
    if not AVAILABLE_BRIDGES:
        initialize_bridges()
    
    return AVAILABLE_BRIDGES

# Check if a specific bridge is available
def is_bridge_available(name: str) -> bool:
    """
    Check if a specific bridge is available.
    
    Args:
        name: Name of the bridge
        
    Returns:
        True if bridge is available, False otherwise
    """
    if not AVAILABLE_BRIDGES:
        initialize_bridges()
    
    return AVAILABLE_BRIDGES.get(name, False)

# Initialize bridges on module import
initialize_bridges()