"""
PhiFlow State Transitions

This module defines the core classes for quantum field state transitions
in the PhiFlow DSL.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
import re
import sys
from pathlib import Path

# Add project root to path to import sacred constants
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Import sacred constants
try:
    import sacred_constants as sc
    PHI = sc.PHI
    PHI_PHI = sc.PHI_PHI
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895
    PHI_PHI = 2.1784575679375995

class FieldOperation:
    """
    Represents a single operation to be performed on a quantum field
    during a state transition.
    """
    
    def __init__(self, operation_type: str, value: str):
        """
        Initialize a field operation.
        
        Args:
            operation_type: Type of operation (amplify, attenuate, etc.)
            value: Value parameter for the operation
        """
        self.operation_type = operation_type
        self.raw_value = value
        
        # Parse the value, which might be a phi expression
        self.value = self._parse_value(value)
    
    def _parse_value(self, value_str: str) -> float:
        """Parse a value string that might include phi references."""
        if value_str == "PHI":
            return PHI
        elif value_str.startswith("PHI^"):
            power = float(value_str[4:])
            return PHI ** power
        else:
            try:
                return float(value_str)
            except ValueError:
                return 1.0  # Default value
    
    def apply(self, field_data: np.ndarray) -> np.ndarray:
        """
        Apply this operation to a quantum field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Modified field data
        """
        if self.operation_type == "amplify":
            return field_data * self.value
            
        elif self.operation_type == "attenuate":
            return field_data / self.value
            
        elif self.operation_type == "rotate":
            # Rotate the field pattern
            angle = self.value * np.pi / 180.0
            height, width = field_data.shape
            center_y, center_x = height // 2, width // 2
            
            # Create rotation matrix
            cos_theta, sin_theta = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            
            # Create new field
            rotated_field = np.zeros_like(field_data)
            
            # Apply rotation
            for y in range(height):
                for x in range(width):
                    # Translate to origin
                    y_centered = y - center_y
                    x_centered = x - center_x
                    
                    # Rotate
                    coords = np.array([x_centered, y_centered])
                    rotated_coords = rot_matrix @ coords
                    
                    # Translate back and round to nearest pixel
                    x_rotated = int(round(rotated_coords[0] + center_x))
                    y_rotated = int(round(rotated_coords[1] + center_y))
                    
                    # Check if within bounds
                    if 0 <= x_rotated < width and 0 <= y_rotated < height:
                        rotated_field[y, x] = field_data[y_rotated, x_rotated]
            
            return rotated_field
            
        elif self.operation_type == "harmonize":
            # Enhance phi-harmonic patterns in the field
            harmonized = np.zeros_like(field_data)
            for y in range(field_data.shape[0]):
                for x in range(field_data.shape[1]):
                    value = field_data[y, x]
                    # Find nearest phi multiple
                    nearest_phi_multiple = round(value / PHI)
                    # Pull value toward nearest phi multiple
                    harmonized[y, x] = nearest_phi_multiple * PHI * self.value + value * (1 - self.value)
            
            return harmonized
            
        elif self.operation_type == "blend":
            # Smooth the field with a phi-weighted kernel
            from scipy import ndimage
            kernel_size = int(3 * self.value)
            if kernel_size % 2 == 0:  # Ensure odd size
                kernel_size += 1
            
            # Create phi-weighted kernel
            kernel = np.ones((kernel_size, kernel_size))
            center = kernel_size // 2
            for y in range(kernel_size):
                for x in range(kernel_size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    kernel[y, x] = np.exp(-dist / PHI)
            
            # Normalize kernel
            kernel = kernel / kernel.sum()
            
            # Apply convolution
            return ndimage.convolve(field_data, kernel, mode='reflect')
            
        elif self.operation_type == "center":
            # Center the field energy around the middle
            height, width = field_data.shape
            center_y, center_x = height // 2, width // 2
            centered = np.zeros_like(field_data)
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance from center
                    dist = np.sqrt(((y - center_y) / height * 2)**2 + ((x - center_x) / width * 2)**2)
                    # Apply centering factor
                    weight = np.exp(-dist * self.value / PHI)
                    centered[y, x] = field_data[y, x] * weight
            
            return centered
            
        elif self.operation_type == "expand":
            # Expand the field outward
            height, width = field_data.shape
            center_y, center_x = height // 2, width // 2
            expanded = np.zeros_like(field_data)
            
            scale_factor = 1.0 / self.value
            
            for y in range(height):
                for x in range(width):
                    # Scale coordinates
                    y_centered = y - center_y
                    x_centered = x - center_x
                    
                    y_scaled = y_centered * scale_factor + center_y
                    x_scaled = x_centered * scale_factor + center_x
                    
                    # Bilinear interpolation
                    x0, y0 = int(x_scaled), int(y_scaled)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    if 0 <= x0 < width-1 and 0 <= y0 < height-1:
                        wx = x_scaled - x0
                        wy = y_scaled - y0
                        
                        expanded[y, x] = (1-wx)*(1-wy)*field_data[y0, x0] + \
                                         wx*(1-wy)*field_data[y0, x1] + \
                                         (1-wx)*wy*field_data[y1, x0] + \
                                         wx*wy*field_data[y1, x1]
            
            return expanded
            
        elif self.operation_type == "contract":
            # Contract the field inward
            height, width = field_data.shape
            center_y, center_x = height // 2, width // 2
            contracted = np.zeros_like(field_data)
            
            scale_factor = self.value
            
            for y in range(height):
                for x in range(width):
                    # Scale coordinates
                    y_centered = y - center_y
                    x_centered = x - center_x
                    
                    y_scaled = y_centered * scale_factor + center_y
                    x_scaled = x_centered * scale_factor + center_x
                    
                    # Bilinear interpolation
                    x0, y0 = int(x_scaled), int(y_scaled)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    if 0 <= x0 < width-1 and 0 <= y0 < height-1:
                        wx = x_scaled - x0
                        wy = y_scaled - y0
                        
                        contracted[y, x] = (1-wx)*(1-wy)*field_data[y0, x0] + \
                                           wx*(1-wy)*field_data[y0, x1] + \
                                           (1-wx)*wy*field_data[y1, x0] + \
                                           wx*wy*field_data[y1, x1]
            
            return contracted
        
        else:
            # Unknown operation, return field unchanged
            return field_data
    
    def __repr__(self):
        return f"FieldOperation({self.operation_type}, {self.raw_value})"


class TransitionOperator:
    """
    Operator for executing quantum field state transitions.
    """
    
    def __init__(self, transition_id: str, from_state: str, to_state: str):
        """
        Initialize a transition operator.
        
        Args:
            transition_id: Unique identifier for this transition
            from_state: Source state
            to_state: Target state
        """
        self.transition_id = transition_id
        self.from_state = from_state
        self.to_state = to_state
        self.operations = []
        
    def add_operation(self, operation: FieldOperation):
        """Add an operation to this transition."""
        self.operations.append(operation)
    
    def apply(self, field_data: np.ndarray) -> np.ndarray:
        """
        Apply all operations in this transition to the field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Modified field data
        """
        result = field_data.copy()
        
        for operation in self.operations:
            result = operation.apply(result)
            
        return result


class StateTransition:
    """
    Represents a transition between quantum field states in the PhiFlow DSL.
    """
    
    def __init__(self, from_state: str, to_state: str):
        """
        Initialize a state transition.
        
        Args:
            from_state: Source state name
            to_state: Target state name
        """
        self.from_state = from_state
        self.to_state = to_state
        self.condition = None
        self.operations = []
    
    def set_condition(self, condition: str):
        """Set the condition for this transition."""
        self.condition = condition
    
    def add_operation(self, operation: FieldOperation):
        """Add a field operation to this transition."""
        self.operations.append(operation)
    
    def can_transition(self, field_data: np.ndarray, coherence: float) -> bool:
        """
        Check if this transition can be executed based on its condition.
        
        Args:
            field_data: The quantum field data
            coherence: Current field coherence value
            
        Returns:
            True if transition can be executed, False otherwise
        """
        if not self.condition:
            return True
            
        # Parse and evaluate condition
        if 'coherence' in self.condition:
            threshold_match = re.search(r'coherence\s*(>=|>|==|<|<=)\s*([0-9.]+)', self.condition)
            if threshold_match:
                operator, value = threshold_match.groups()
                threshold = float(value)
                
                if operator == '>=':
                    return coherence >= threshold
                elif operator == '>':
                    return coherence > threshold
                elif operator == '==':
                    return abs(coherence - threshold) < 1e-6
                elif operator == '<':
                    return coherence < threshold
                elif operator == '<=':
                    return coherence <= threshold
        
        # If no recognizable condition or parsing failed, default to True
        return True
    
    def execute(self, field_data: np.ndarray) -> np.ndarray:
        """
        Execute this transition on a quantum field.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Modified field data
        """
        result = field_data.copy()
        
        for operation in self.operations:
            result = operation.apply(result)
            
        return result
    
    def __repr__(self):
        return f"StateTransition({self.from_state} -> {self.to_state})"