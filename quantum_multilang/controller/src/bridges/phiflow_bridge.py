"""
φFlow Language Bridge

This module provides the interface between the multi-language controller and 
the φFlow DSL for quantum field state transitions.
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
logger = logging.getLogger("bridges.phiflow")

# Import φFlow
try:
    from languages.phiflow.src import (
        parse_phiflow, 
        PhiFlowInterpreter, 
        compile_phiflow
    )
    PHIFLOW_AVAILABLE = True
except ImportError:
    logger.warning("φFlow language not found. Some functionality will be disabled.")
    PHIFLOW_AVAILABLE = False

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

class PhiFlowBridge:
    """
    Bridge for φFlow DSL integration with the multi-language architecture.
    """
    
    def __init__(self):
        """Initialize the φFlow bridge."""
        self.interpreter = None
        self.serializer = UFPSerializer() if UFPSerializer else None
        
        if PHIFLOW_AVAILABLE:
            self.interpreter = PhiFlowInterpreter()
            logger.info("φFlow bridge initialized")
        else:
            logger.warning("φFlow bridge initialization failed: φFlow not available")
    
    def load_phiflow_code(self, code: str) -> bool:
        """
        Load φFlow code into the interpreter.
        
        Args:
            code: φFlow DSL code as string
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not PHIFLOW_AVAILABLE or not self.interpreter:
            logger.warning("Cannot load φFlow code: φFlow not available")
            return False
            
        try:
            self.interpreter.load(code)
            return True
        except Exception as e:
            logger.error(f"Error loading φFlow code: {e}")
            return False
    
    def apply_transition(self, field_data: np.ndarray, transition_name: str) -> Tuple[np.ndarray, float]:
        """
        Apply a named transition to a field.
        
        Args:
            field_data: The quantum field data
            transition_name: Name of the transition (formatted as "from_state->to_state")
            
        Returns:
            Tuple of (new_field_data, new_coherence)
        """
        if not PHIFLOW_AVAILABLE or not self.interpreter:
            logger.warning("Cannot apply transition: φFlow not available")
            return field_data, 0.0
            
        # Set the field data
        self.interpreter.set_field(field_data)
        
        # Find the transition
        available = self.interpreter.get_available_transitions()
        
        # Parse transition name
        if "->" in transition_name:
            from_state, to_state = transition_name.split("->")
            from_state = from_state.strip()
            to_state = to_state.strip()
            
            # Find the matching transition
            for transition in available:
                if transition.from_state == from_state and transition.to_state == to_state:
                    return self.interpreter.apply_transition(transition)
        
        # If we get here, no matching transition was found
        logger.warning(f"Transition '{transition_name}' not found or not available")
        return field_data, self.interpreter.coherence if self.interpreter.coherence else 0.0
    
    def run_auto_transition(self, field_data: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """
        Automatically select and apply the best available transition.
        
        Args:
            field_data: The quantum field data
            
        Returns:
            Tuple of (new_field_data, new_coherence, transition_name)
        """
        if not PHIFLOW_AVAILABLE or not self.interpreter:
            logger.warning("Cannot run auto transition: φFlow not available")
            return field_data, 0.0, "none"
            
        # Set the field data
        self.interpreter.set_field(field_data)
        
        # Run auto transition
        transition = self.interpreter.run_auto_transition()
        
        if transition:
            transition_name = f"{transition.from_state}->{transition.to_state}"
            return self.interpreter.field_data, self.interpreter.coherence, transition_name
        else:
            return field_data, self.interpreter.coherence, "none"
    
    def compile_phiflow_code(self, code: str) -> Optional[str]:
        """
        Compile φFlow code to Python for optimization.
        
        Args:
            code: φFlow DSL code as string
            
        Returns:
            Compiled Python code as string, or None if compilation failed
        """
        if not PHIFLOW_AVAILABLE:
            logger.warning("Cannot compile φFlow code: φFlow not available")
            return None
            
        try:
            return compile_phiflow(code)
        except Exception as e:
            logger.error(f"Error compiling φFlow code: {e}")
            return None
    
    def process_field_message(self, message: Any) -> Tuple[np.ndarray, float, str]:
        """
        Process a field message and apply appropriate transitions.
        
        Args:
            message: Field message (either QuantumFieldMessage or raw field data)
            
        Returns:
            Tuple of (new_field_data, new_coherence, transition_name)
        """
        if not PHIFLOW_AVAILABLE or not self.interpreter:
            logger.warning("Cannot process message: φFlow not available")
            if isinstance(message, np.ndarray):
                return message, 0.0, "none"
            else:
                try:
                    return message.field_data, message.phi_coherence, "none"
                except:
                    return np.array([]), 0.0, "none"
        
        # Extract field data
        if QuantumFieldMessage and isinstance(message, QuantumFieldMessage):
            field_data = message.field_data
            coherence = message.phi_coherence
        elif isinstance(message, np.ndarray):
            field_data = message
            coherence = None
        else:
            logger.warning(f"Unknown message format: {type(message)}")
            return np.array([]), 0.0, "none"
        
        # Run auto transition
        new_field, new_coherence, transition_name = self.run_auto_transition(field_data)
        
        return new_field, new_coherence, transition_name
    
    def create_field_message(self, field_data: np.ndarray, coherence: float, 
                           frequency_name: str = 'love') -> Any:
        """
        Create a field message from processed data.
        
        Args:
            field_data: The quantum field data
            coherence: Field coherence value
            frequency_name: Sacred frequency name
            
        Returns:
            Field message (QuantumFieldMessage or raw field data)
        """
        if QuantumFieldMessage and self.serializer:
            # Create a proper message
            message = QuantumFieldMessage(
                field_data=field_data,
                frequency_name=frequency_name,
                consciousness_level=coherence, # Use coherence as consciousness level
                phi_coherence=coherence,
                source_language="phiflow"
            )
            return message
        else:
            # Return raw field data
            return field_data

# Initialize the bridge
def initialize():
    """Initialize the φFlow bridge."""
    bridge = PhiFlowBridge()
    return PHIFLOW_AVAILABLE

# Test connection
def test_connection():
    """Test if φFlow is available."""
    return PHIFLOW_AVAILABLE

# Process a field transition using φFlow
def process_field_transition(field_data, state_machine_code=None):
    """
    Process a field transition using φFlow.
    
    Args:
        field_data: The quantum field data
        state_machine_code: φFlow state machine code (optional)
        
    Returns:
        Tuple of (new_field_data, new_coherence, transition_name)
    """
    bridge = PhiFlowBridge()
    
    if state_machine_code:
        bridge.load_phiflow_code(state_machine_code)
    
    return bridge.run_auto_transition(field_data)