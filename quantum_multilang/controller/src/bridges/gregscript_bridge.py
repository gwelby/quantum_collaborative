"""
GregScript Language Bridge

This module provides the interface between the multi-language controller and 
the GregScript language for pattern recognition and generation.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridges.gregscript")

# Import GregScript
try:
    from languages.gregscript.src import (
        parse_gregscript, 
        GregScriptInterpreter, 
        PatternAnalyzer
    )
    GREGSCRIPT_AVAILABLE = True
except ImportError:
    logger.warning("GregScript language not found. Some functionality will be disabled.")
    GREGSCRIPT_AVAILABLE = False

# Import universal field protocol
try:
    from controller.src.universal_field_protocol import (
        QuantumFieldMessage,
        UFPSerializer
    )
except ImportError:
    logger.warning("Universal Field Protocol not found. Using simplified message format.")
    QuantumFieldMessage = None
    UFPSerializer = None

class GregScriptBridge:
    """
    Bridge for GregScript language integration with the multi-language architecture.
    """
    
    def __init__(self):
        """Initialize the GregScript bridge."""
        self.interpreter = None
        self.analyzer = None
        self.serializer = UFPSerializer() if UFPSerializer else None
        
        if GREGSCRIPT_AVAILABLE:
            self.interpreter = GregScriptInterpreter()
            self.analyzer = PatternAnalyzer()
            logger.info("GregScript bridge initialized")
        else:
            logger.warning("GregScript bridge initialization failed: GregScript not available")
    
    def load_gregscript_code(self, code: str) -> bool:
        """
        Load GregScript code into the interpreter.
        
        Args:
            code: GregScript code as string
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not GREGSCRIPT_AVAILABLE or not self.interpreter:
            logger.warning("Cannot load GregScript code: GregScript not available")
            return False
            
        try:
            self.interpreter.load(code)
            return True
        except Exception as e:
            logger.error(f"Error loading GregScript code: {e}")
            return False
    
    def match_pattern(self, field_data: np.ndarray, pattern_name: str) -> float:
        """
        Match a named pattern against a field.
        
        Args:
            field_data: The quantum field data
            pattern_name: Name of the pattern to match
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not GREGSCRIPT_AVAILABLE or not self.interpreter:
            logger.warning("Cannot match pattern: GregScript not available")
            return 0.0
            
        # Extract a time series from the field
        time_series = self._extract_time_series(field_data)
            
        # Set the data
        self.interpreter.set_data(time_series)
        
        # Match the pattern
        return self.interpreter.match_pattern(pattern_name)
    
    def _extract_time_series(self, field_data: np.ndarray, radius: float = 0.5) -> np.ndarray:
        """
        Extract a time series from a field.
        
        Args:
            field_data: The quantum field data
            radius: Radius for circular path extraction (0.0 to 1.0)
            
        Returns:
            1D time series data
        """
        # If already 1D, return as is
        if len(field_data.shape) == 1:
            return field_data
            
        # If 2D, extract a circular path
        if len(field_data.shape) == 2:
            height, width = field_data.shape
            center_y, center_x = height // 2, width // 2
            
            # Create a circular path
            num_points = int(2 * np.pi * radius * min(width, height) / 2)
            time_series = np.zeros(num_points)
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = int(center_x + radius * (width/2) * np.cos(angle))
                y = int(center_y + radius * (height/2) * np.sin(angle))
                
                # Ensure coordinates are within bounds
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                time_series[i] = field_data[y, x]
                
            return time_series
            
        # If higher dimensional, flatten
        return field_data.flatten()
    
    def find_best_pattern(self, field_data: np.ndarray, min_score: float = 0.5) -> Tuple[str, float]:
        """
        Find the best matching pattern for a field.
        
        Args:
            field_data: The quantum field data
            min_score: Minimum score threshold for consideration
            
        Returns:
            Tuple of (pattern_name, score) for the best match
        """
        if not GREGSCRIPT_AVAILABLE or not self.interpreter:
            logger.warning("Cannot find best pattern: GregScript not available")
            return None, 0.0
            
        # Extract a time series from the field
        time_series = self._extract_time_series(field_data)
            
        # Set the data
        self.interpreter.set_data(time_series)
        
        # Find the best pattern
        return self.interpreter.find_best_pattern(min_score)
    
    def generate_pattern(self, pattern_name: str, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Generate a field from a pattern.
        
        Args:
            pattern_name: Name of the pattern to generate
            shape: Shape of the field to generate
            
        Returns:
            Generated field data
        """
        if not GREGSCRIPT_AVAILABLE or not self.interpreter:
            logger.warning("Cannot generate pattern: GregScript not available")
            if isinstance(shape, int):
                return np.zeros(shape)
            else:
                return np.zeros(shape)
            
        # Generate the pattern
        return self.interpreter.generate_pattern(pattern_name, shape)
    
    def discover_patterns(self, field_data: np.ndarray) -> Dict[str, Any]:
        """
        Discover patterns in a field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Dictionary with discovered patterns
        """
        if not GREGSCRIPT_AVAILABLE or not self.analyzer:
            logger.warning("Cannot discover patterns: GregScript not available")
            return {}
            
        # Extract a time series from the field
        time_series = self._extract_time_series(field_data)
            
        # Discover patterns
        return self.analyzer.discover_patterns(time_series)
    
    def generate_gregscript_code(self, field_data: np.ndarray) -> Optional[str]:
        """
        Generate GregScript code from discovered patterns in a field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Generated GregScript code as string, or None if generation failed
        """
        if not GREGSCRIPT_AVAILABLE or not self.analyzer:
            logger.warning("Cannot generate GregScript code: GregScript not available")
            return None
            
        # Extract a time series from the field
        time_series = self._extract_time_series(field_data)
            
        # Discover patterns
        discovered = self.analyzer.discover_patterns(time_series)
        
        # Generate code
        return self.analyzer.generate_gregscript(discovered)
    
    def process_field_message(self, message: Any) -> Dict[str, Any]:
        """
        Process a field message and analyze patterns.
        
        Args:
            message: Field message (either QuantumFieldMessage or raw field data)
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not GREGSCRIPT_AVAILABLE or not self.interpreter or not self.analyzer:
            logger.warning("Cannot process message: GregScript not available")
            return {}
        
        # Extract field data
        if QuantumFieldMessage and isinstance(message, QuantumFieldMessage):
            field_data = message.field_data
        elif isinstance(message, np.ndarray):
            field_data = message
        else:
            logger.warning(f"Unknown message format: {type(message)}")
            return {}
        
        # Extract a time series from the field
        time_series = self._extract_time_series(field_data)
        
        # Discover patterns
        patterns = self.analyzer.discover_patterns(time_series)
        
        # Set the data for pattern matching
        self.interpreter.set_data(time_series)
        
        # Find the best pattern in the loaded patterns
        best_pattern, best_score = self.interpreter.find_best_pattern()
        
        # Add to results
        if best_pattern:
            patterns["best_known_pattern"] = best_pattern
            patterns["best_known_score"] = best_score
        
        return patterns
    
    def create_field_message(self, field_data: np.ndarray, pattern_results: Dict[str, Any],
                           frequency_name: str = 'love') -> Any:
        """
        Create a field message with pattern analysis metadata.
        
        Args:
            field_data: The quantum field data
            pattern_results: Pattern analysis results
            frequency_name: Sacred frequency name
            
        Returns:
            Field message (QuantumFieldMessage or raw field data)
        """
        # Calculate a coherence value from pattern results
        coherence = 0.0
        
        if "top_pattern" in pattern_results and "score" in pattern_results["top_pattern"]:
            coherence = pattern_results["top_pattern"]["score"]
        elif "best_known_score" in pattern_results:
            coherence = pattern_results["best_known_score"]
        
        if QuantumFieldMessage and self.serializer:
            # Create a proper message
            message = QuantumFieldMessage(
                field_data=field_data,
                frequency_name=frequency_name,
                consciousness_level=coherence,
                phi_coherence=coherence,
                source_language="gregscript",
                metadata={"patterns": pattern_results}
            )
            return message
        else:
            # Return raw field data with metadata
            return {"field_data": field_data, "patterns": pattern_results}

# Initialize the bridge
def initialize():
    """Initialize the GregScript bridge."""
    bridge = GregScriptBridge()
    return GREGSCRIPT_AVAILABLE

# Test connection
def test_connection():
    """Test if GregScript is available."""
    return GREGSCRIPT_AVAILABLE

# Analyze patterns in a field
def analyze_field_patterns(field_data):
    """
    Analyze patterns in a field.
    
    Args:
        field_data: The quantum field data
        
    Returns:
        Dictionary with pattern analysis results
    """
    bridge = GregScriptBridge()
    return bridge.discover_patterns(field_data)