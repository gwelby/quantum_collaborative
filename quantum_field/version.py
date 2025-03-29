"""
Version information for quantum_field package
"""

__version__ = "0.1.0"

def get_version():
    """
    Get the current version of the library.
    
    Returns:
        Version string in format "X.Y.Z"
    """
    return __version__


def check_version_compatibility(min_version: str, max_version: str = None) -> bool:
    """
    Check if the current version is compatible with the specified range.
    
    Args:
        min_version: Minimum supported version
        max_version: Maximum supported version (optional)
        
    Returns:
        True if compatible, False otherwise
    """
    def parse_version(version_str):
        """Parse version string into tuple of integers"""
        return tuple(map(int, version_str.split('.')))
    
    current = parse_version(__version__)
    minimum = parse_version(min_version)
    
    if current < minimum:
        return False
    
    if max_version:
        maximum = parse_version(max_version)
        if current > maximum:
            return False
    
    return True