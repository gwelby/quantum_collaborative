"""
PhiFlow DSL Compiler

This module provides compilation functionality for PhiFlow DSL code,
converting it to executable Python code for optimized performance.
"""

import re
import ast
import numpy as np
from typing import Dict, List, Tuple, Any
import textwrap

from .parser import parse_philflow
from .transitions import StateTransition, FieldOperation

class PhiFlowCompiler:
    """Compiler for PhiFlow DSL."""
    
    def __init__(self):
        """Initialize the compiler."""
        pass
    
    def compile(self, philflow_code: str) -> str:
        """
        Compile PhiFlow code into Python code.
        
        Args:
            philflow_code: PhiFlow DSL code
            
        Returns:
            Python code string
        """
        # Parse the PhiFlow code
        states, transitions = parse_philflow(philflow_code)
        
        # Generate the Python code
        python_code = self._generate_python_code(states, transitions)
        
        return python_code
    
    def _generate_python_code(self, states: Dict[str, Dict[str, Any]], transitions: List[StateTransition]) -> str:
        """Generate Python code from parsed PhiFlow states and transitions."""
        # Generate imports
        code = textwrap.dedent("""
        import numpy as np
        from typing import Dict, List, Tuple, Any, Optional
        
        # Sacred constants
        PHI = 1.618033988749895
        PHI_PHI = 2.1784575679375995
        SACRED_FREQUENCIES = {
            'love': 528,
            'unity': 432,
            'cascade': 594,
            'truth': 672,
            'vision': 720,
            'oneness': 768,
        }
        
        class CompiledPhiFlow:
            \"\"\"Compiled PhiFlow state machine.\"\"\"
            
            def __init__(self):
                \"\"\"Initialize the state machine.\"\"\"
                self.current_state = None
                self.field_data = None
                self.coherence = 0.0
                
                # Initialize states
                self.states = self._initialize_states()
                
                # Set initial state
                if self.states:
                    self.current_state = list(self.states.keys())[0]
                
            def _initialize_states(self) -> Dict[str, Dict[str, Any]]:
                \"\"\"Initialize the states.\"\"\"
                states = {}
        """)
        
        # Generate state initialization
        for state_name, state_props in states.items():
            props_str = ", ".join([f"'{k}': {v}" if isinstance(v, (int, float)) else f"'{k}': '{v}'" for k, v in state_props.items()])
            code += f"        states['{state_name}'] = {{{props_str}}}\n"
            
        code += "        return states\n\n"
        
        # Generate transition methods
        for i, transition in enumerate(transitions):
            method_name = f"_transition_{i+1}"
            
            # Generate method signature
            code += f"            def {method_name}(self, field_data: np.ndarray) -> Tuple[np.ndarray, float]:\n"
            code += f'                """Transition from {transition.from_state} to {transition.to_state}."""\n'
            
            # Check condition
            if transition.condition:
                code += f"                # Check condition: {transition.condition}\n"
                if 'coherence' in transition.condition:
                    threshold_match = re.search(r'coherence\s*(>=|>|==|<|<=)\s*([0-9.]+)', transition.condition)
                    if threshold_match:
                        operator, value = threshold_match.groups()
                        code += f"                if not (self.coherence {operator} {value}):\n"
                        code += f"                    return field_data, self.coherence\n\n"
            
            # Apply operations
            code += "                result = field_data.copy()\n\n"
            
            for op in transition.operations:
                code += f"                # {op.operation_type.capitalize()} by {op.raw_value}\n"
                if op.operation_type == "amplify":
                    code += f"                result = result * {op.value}\n"
                elif op.operation_type == "attenuate":
                    code += f"                result = result / {op.value}\n"
                elif op.operation_type == "rotate":
                    code += textwrap.dedent(f"""
                    # Rotate the field pattern
                    angle = {op.value} * np.pi / 180.0
                    height, width = result.shape
                    center_y, center_x = height // 2, width // 2
                    
                    # Create rotation matrix
                    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
                    rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
                    
                    # Create new field
                    rotated_field = np.zeros_like(result)
                    
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
                                rotated_field[y, x] = result[y_rotated, x_rotated]
                    
                    result = rotated_field
                    """).replace("\n", "\n                ")
                elif op.operation_type == "harmonize":
                    code += textwrap.dedent(f"""
                    # Enhance phi-harmonic patterns in the field
                    harmonized = np.zeros_like(result)
                    for y in range(result.shape[0]):
                        for x in range(result.shape[1]):
                            value = result[y, x]
                            # Find nearest phi multiple
                            nearest_phi_multiple = round(value / PHI)
                            # Pull value toward nearest phi multiple
                            harmonized[y, x] = nearest_phi_multiple * PHI * {op.value} + value * (1 - {op.value})
                    
                    result = harmonized
                    """).replace("\n", "\n                ")
                elif op.operation_type == "blend":
                    code += textwrap.dedent(f"""
                    # Smooth the field with a phi-weighted kernel
                    from scipy import ndimage
                    kernel_size = int(3 * {op.value})
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
                    result = ndimage.convolve(result, kernel, mode='reflect')
                    """).replace("\n", "\n                ")
                elif op.operation_type == "center":
                    code += textwrap.dedent(f"""
                    # Center the field energy around the middle
                    height, width = result.shape
                    center_y, center_x = height // 2, width // 2
                    centered = np.zeros_like(result)
                    
                    for y in range(height):
                        for x in range(width):
                            # Calculate distance from center
                            dist = np.sqrt(((y - center_y) / height * 2)**2 + ((x - center_x) / width * 2)**2)
                            # Apply centering factor
                            weight = np.exp(-dist * {op.value} / PHI)
                            centered[y, x] = result[y, x] * weight
                    
                    result = centered
                    """).replace("\n", "\n                ")
                elif op.operation_type == "expand":
                    code += textwrap.dedent(f"""
                    # Expand the field outward
                    height, width = result.shape
                    center_y, center_x = height // 2, width // 2
                    expanded = np.zeros_like(result)
                    
                    scale_factor = 1.0 / {op.value}
                    
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
                                
                                expanded[y, x] = (1-wx)*(1-wy)*result[y0, x0] + \
                                                 wx*(1-wy)*result[y0, x1] + \
                                                 (1-wx)*wy*result[y1, x0] + \
                                                 wx*wy*result[y1, x1]
                    
                    result = expanded
                    """).replace("\n", "\n                ")
                elif op.operation_type == "contract":
                    code += textwrap.dedent(f"""
                    # Contract the field inward
                    height, width = result.shape
                    center_y, center_x = height // 2, width // 2
                    contracted = np.zeros_like(result)
                    
                    scale_factor = {op.value}
                    
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
                                
                                contracted[y, x] = (1-wx)*(1-wy)*result[y0, x0] + \
                                                   wx*(1-wy)*result[y0, x1] + \
                                                   (1-wx)*wy*result[y1, x0] + \
                                                   wx*wy*result[y1, x1]
                    
                    result = contracted
                    """).replace("\n", "\n                ")
            
            # Calculate new coherence
            code += textwrap.dedent("""
            # Calculate coherence
            coherence = self._calculate_coherence(result)
            
            # Update state
            self.current_state = '{to_state}'
            
            return result, coherence
            """.format(to_state=transition.to_state)).replace("\n", "\n                ")
            
            code += "\n"
        
        # Generate utility methods
        code += textwrap.dedent("""
            def _calculate_coherence(self, field_data: np.ndarray) -> float:
                \"\"\"Calculate field coherence.\"\"\"
                if field_data.size == 0:
                    return 0.0
                    
                # Sample points for phi alignment
                height, width = field_data.shape
                sample_points = []
                np.random.seed(42)  # For reproducible results
                for _ in range(100):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    sample_points.append((x, y))
                    
                # Calculate alignment with phi
                alignments = []
                for x, y in sample_points:
                    value = field_data[y, x]
                    nearest_phi_multiple = round(value / PHI)
                    deviation = abs(value - (nearest_phi_multiple * PHI))
                    alignment = 1.0 - min(1.0, deviation / (PHI * 0.1))
                    alignments.append(alignment)
                    
                coherence = np.mean(alignments) * PHI
                return coherence
                
            def set_field(self, field_data: np.ndarray, coherence: float = None):
                \"\"\"Set the current quantum field data.\"\"\"
                self.field_data = field_data
                
                if coherence is None:
                    self.coherence = self._calculate_coherence(field_data)
                else:
                    self.coherence = coherence
                    
            def get_available_transitions(self) -> List[Tuple[str, str]]:
                \"\"\"Get all transitions available from the current state.\"\"\"
                if not self.current_state or not self.field_data is not None:
                    return []
                    
                available = []
        """)
        
        # Generate transition checks
        for i, transition in enumerate(transitions):
            method_name = f"_transition_{i+1}"
            code += f"                if self.current_state == '{transition.from_state}':\n"
            
            if transition.condition:
                code += f"                    # Check condition: {transition.condition}\n"
                if 'coherence' in transition.condition:
                    threshold_match = re.search(r'coherence\s*(>=|>|==|<|<=)\s*([0-9.]+)', transition.condition)
                    if threshold_match:
                        operator, value = threshold_match.groups()
                        code += f"                    if self.coherence {operator} {value}:\n"
                        code += f"                        available.append(('{transition.from_state}', '{transition.to_state}'))\n"
            else:
                code += f"                    available.append(('{transition.from_state}', '{transition.to_state}'))\n"
                
        code += "                return available\n\n"
        
        # Generate apply_transition method
        code += textwrap.dedent("""
            def apply_transition(self, from_state: str, to_state: str) -> Tuple[np.ndarray, float]:
                \"\"\"Apply a transition to the current field.\"\"\"
                if self.field_data is None:
                    raise ValueError("No field data set")
                    
                if from_state != self.current_state:
                    raise ValueError(f"Current state {self.current_state} does not match from_state {from_state}")
        """)
        
        # Generate transition dispatch
        for i, transition in enumerate(transitions):
            method_name = f"_transition_{i+1}"
            code += f"                if from_state == '{transition.from_state}' and to_state == '{transition.to_state}':\n"
            code += f"                    return self.{method_name}(self.field_data)\n"
                
        code += "                raise ValueError(f\"No transition from {from_state} to {to_state}\")\n\n"
        
        # Generate run_auto_transition method
        code += textwrap.dedent("""
            def run_auto_transition(self) -> Optional[Tuple[str, str]]:
                \"\"\"Automatically select and apply the best available transition.\"\"\"
                available = self.get_available_transitions()
                
                if not available:
                    return None
                    
                # Select the first available transition
                selected = available[0]
                
                # Apply it
                new_field, new_coherence = self.apply_transition(*selected)
                
                # Update field and coherence
                self.field_data = new_field
                self.coherence = new_coherence
                
                return selected
        """)
        
        # Generate run_philflow function
        code += textwrap.dedent("""
        def run_philflow(field_data: np.ndarray, max_transitions: int = 10) -> Tuple[np.ndarray, List[str]]:
            \"\"\"
            Run compiled PhiFlow on a quantum field.
            
            Args:
                field_data: Initial quantum field data
                max_transitions: Maximum number of automatic transitions to apply
                
            Returns:
                Tuple of (final_field_data, transition_log)
            \"\"\"
            state_machine = CompiledPhiFlow()
            state_machine.set_field(field_data)
            
            transition_log = []
            transition_log.append(f"Initial state: {state_machine.current_state}")
            
            # Run transitions
            for i in range(max_transitions):
                transition = state_machine.run_auto_transition()
                if not transition:
                    transition_log.append(f"Step {i+1}: No available transitions, execution halted")
                    break
                    
                from_state, to_state = transition
                transition_log.append(f"Step {i+1}: Applied {from_state} -> {to_state} (coherence: {state_machine.coherence:.4f})")
            
            return state_machine.field_data, transition_log
        """)
        
        return code

def compile_philflow(philflow_code: str) -> str:
    """
    Compile PhiFlow DSL code to Python.
    
    Args:
        philflow_code: PhiFlow DSL code
        
    Returns:
        Compiled Python code
    """
    compiler = PhiFlowCompiler()
    return compiler.compile(philflow_code)