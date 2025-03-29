"""
GregScript Interpreter

This module provides the interpreter for executing GregScript code
and managing rhythm, harmony, and pattern recognition.
"""

import re
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

from .patterns import Pattern, Rhythm, Harmony
from .parser import parse_gregscript

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gregscript")

# Import sacred constants
try:
    import sacred_constants as sc
    PHI = sc.PHI
    PHI_PHI = sc.PHI_PHI
except ImportError:
    # Fallback constants
    logger.warning("sacred_constants module not found. Using default values.")
    PHI = 1.618033988749895
    PHI_PHI = 2.1784575679375995


class GregScriptInterpreter:
    """Interpreter for GregScript language."""
    
    def __init__(self):
        """Initialize the interpreter."""
        self.patterns = {}
        self.rhythms = {}
        self.harmonies = {}
        self.current_data = None
        
    def load(self, code: str):
        """
        Load GregScript code into the interpreter.
        
        Args:
            code: GregScript code as string
        """
        all_patterns = parse_gregscript(code)
        
        # Categorize patterns
        self.patterns = {}
        self.rhythms = {}
        self.harmonies = {}
        
        for name, pattern in all_patterns.items():
            if isinstance(pattern, Pattern):
                self.patterns[name] = pattern
            elif isinstance(pattern, Rhythm):
                self.rhythms[name] = pattern
            elif isinstance(pattern, Harmony):
                self.harmonies[name] = pattern
        
        logger.info(f"Loaded {len(self.patterns)} patterns, {len(self.rhythms)} rhythms, {len(self.harmonies)} harmonies")
    
    def set_data(self, data: np.ndarray):
        """
        Set the current data for pattern matching.
        
        Args:
            data: Data array to match patterns against
        """
        self.current_data = data
        logger.debug(f"Set data with shape {data.shape}")
    
    def match_pattern(self, pattern_name: str) -> float:
        """
        Match a named pattern against the current data.
        
        Args:
            pattern_name: Name of the pattern to match
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not self.current_data is not None:
            logger.warning("No data set for pattern matching")
            return 0.0
            
        # Find the pattern
        pattern = None
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
        elif pattern_name in self.rhythms:
            pattern = self.rhythms[pattern_name]
        elif pattern_name in self.harmonies:
            pattern = self.harmonies[pattern_name]
            
        if pattern is None:
            logger.warning(f"Pattern '{pattern_name}' not found")
            return 0.0
            
        # Match against data
        score = pattern.match(self.current_data)
        logger.info(f"Pattern '{pattern_name}' matched with score {score:.4f}")
        
        return score
    
    def match_all_patterns(self) -> Dict[str, float]:
        """
        Match all patterns against the current data.
        
        Returns:
            Dictionary of pattern names to match scores
        """
        if not self.current_data is not None:
            logger.warning("No data set for pattern matching")
            return {}
            
        # Match all patterns
        results = {}
        
        # Match rhythms
        for name, rhythm in self.rhythms.items():
            results[f"rhythm:{name}"] = rhythm.match(self.current_data)
            
        # Match harmonies
        for name, harmony in self.harmonies.items():
            results[f"harmony:{name}"] = harmony.match(self.current_data)
            
        # Match patterns
        for name, pattern in self.patterns.items():
            results[f"pattern:{name}"] = pattern.match(self.current_data)
            
        return results
    
    def generate_pattern(self, pattern_name: str, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Generate data from a named pattern.
        
        Args:
            pattern_name: Name of the pattern to generate
            shape: Shape of the data to generate
            
        Returns:
            Generated data
        """
        # Find the pattern
        pattern = None
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
        elif pattern_name in self.rhythms:
            pattern = self.rhythms[pattern_name]
        elif pattern_name in self.harmonies:
            pattern = self.harmonies[pattern_name]
            
        if pattern is None:
            logger.warning(f"Pattern '{pattern_name}' not found")
            if isinstance(shape, int):
                return np.zeros(shape)
            else:
                return np.zeros(shape)
            
        # Generate data
        data = pattern.generate(shape)
        logger.info(f"Generated data from pattern '{pattern_name}' with shape {data.shape}")
        
        return data
    
    def find_best_pattern(self, min_score: float = 0.5) -> Tuple[str, float]:
        """
        Find the best matching pattern for the current data.
        
        Args:
            min_score: Minimum score threshold for consideration
            
        Returns:
            Tuple of (pattern_name, score) for the best match
        """
        if not self.current_data is not None:
            logger.warning("No data set for pattern matching")
            return None, 0.0
            
        # Match all patterns
        all_matches = self.match_all_patterns()
        
        # Find the best match
        best_pattern = None
        best_score = 0.0
        
        for name, score in all_matches.items():
            if score >= min_score and score > best_score:
                best_pattern = name
                best_score = score
                
        if best_pattern:
            logger.info(f"Best matching pattern: {best_pattern} (score: {best_score:.4f})")
            return best_pattern, best_score
        else:
            logger.info(f"No patterns matched with score >= {min_score}")
            return None, 0.0
    
    def visualize_pattern(self, pattern_name: str, length: int = 100) -> Dict[str, Any]:
        """
        Generate visualization data for a pattern.
        
        Args:
            pattern_name: Name of the pattern to visualize
            length: Length of the visualization data
            
        Returns:
            Dictionary with visualization data
        """
        # Find the pattern
        pattern = None
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
        elif pattern_name in self.rhythms:
            pattern = self.rhythms[pattern_name]
        elif pattern_name in self.harmonies:
            pattern = self.harmonies[pattern_name]
            
        if pattern is None:
            logger.warning(f"Pattern '{pattern_name}' not found")
            return {"error": f"Pattern '{pattern_name}' not found"}
            
        # Generate data for visualization
        data = pattern.generate(length)
        
        # Create visualization data
        result = {
            "name": pattern_name,
            "type": pattern.__class__.__name__,
            "data": data.tolist(),
            "x_values": list(range(length)),
        }
        
        # Add type-specific data
        if isinstance(pattern, Rhythm):
            result["tempo"] = pattern.tempo
            result["sequence"] = pattern.sequence
        elif isinstance(pattern, Harmony):
            result["frequency"] = pattern.frequency
            result["overtones"] = pattern.overtones
            result["phase"] = pattern.phase
        elif isinstance(pattern, Pattern):
            result["elements"] = [e.name for e in pattern.elements]
            result["weights"] = pattern.weights
            
        return result
            

def run_gregscript(code: str, data: np.ndarray) -> Dict[str, Any]:
    """
    Run GregScript code on data.
    
    Args:
        code: GregScript code
        data: Data to match patterns against
        
    Returns:
        Dictionary with match results
    """
    interpreter = GregScriptInterpreter()
    interpreter.load(code)
    interpreter.set_data(data)
    
    # Match all patterns
    all_matches = interpreter.match_all_patterns()
    
    # Find the best match
    best_pattern, best_score = interpreter.find_best_pattern()
    
    # Prepare results
    results = {
        "matches": all_matches,
        "best_match": best_pattern,
        "best_score": best_score,
    }
    
    # Add visualization for the best pattern
    if best_pattern:
        pattern_name = best_pattern.split(":", 1)[1]  # Remove type prefix
        results["visualization"] = interpreter.visualize_pattern(pattern_name)
        
    return results