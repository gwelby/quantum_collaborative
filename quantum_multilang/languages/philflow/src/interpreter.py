"""
PhiFlow DSL Interpreter

This module provides the interpreter for executing PhiFlow DSL code
and managing quantum field state transitions.
"""

import re
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

from .transitions import StateTransition, TransitionOperator
from .parser import parse_philflow

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philflow")

# Import sacred constants
try:
    import sacred_constants as sc
    PHI = sc.PHI
    PHI_PHI = sc.PHI_PHI
    SACRED_FREQUENCIES = sc.SACRED_FREQUENCIES
except ImportError:
    # Fallback constants
    logger.warning("sacred_constants module not found. Using default values.")
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


class PhiFlowInterpreter:
    """Interpreter for PhiFlow DSL."""
    
    def __init__(self):
        """Initialize the interpreter."""
        self.states = {}
        self.transitions = []
        self.current_state = None
        self.field_data = None
        self.coherence = 0.0
        
    def load(self, code: str):
        """
        Load PhiFlow code into the interpreter.
        
        Args:
            code: PhiFlow DSL code as string
        """
        self.states, self.transitions = parse_philflow(code)
        
        # Set initial state if available
        if self.states:
            self.current_state = list(self.states.keys())[0]
            logger.info(f"Initial state set to: {self.current_state}")
    
    def set_field(self, field_data: np.ndarray, coherence: float = None):
        """
        Set the current quantum field data.
        
        Args:
            field_data: Quantum field data as numpy array
            coherence: Field coherence value (calculated if not provided)
        """
        self.field_data = field_data
        
        if coherence is None:
            # Calculate coherence if not provided
            self.coherence = self._calculate_coherence(field_data)
        else:
            self.coherence = coherence
            
        logger.debug(f"Field set with coherence: {self.coherence}")
    
    def _calculate_coherence(self, field_data: np.ndarray) -> float:
        """Calculate field coherence (simplified version)."""
        # This is a simplified version; in production, use the actual coherence calculation
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
    
    def set_state(self, state_name: str):
        """
        Set the current state.
        
        Args:
            state_name: Name of the state to set
            
        Returns:
            True if state changed, False if state doesn't exist
        """
        if state_name in self.states:
            self.current_state = state_name
            logger.info(f"State changed to: {state_name}")
            return True
        else:
            logger.warning(f"Attempted to set unknown state: {state_name}")
            return False
    
    def get_available_transitions(self) -> List[StateTransition]:
        """
        Get all transitions available from the current state.
        
        Returns:
            List of available transitions
        """
        if not self.current_state:
            return []
            
        available = []
        for transition in self.transitions:
            if transition.from_state == self.current_state:
                if self.field_data is not None:
                    if transition.can_transition(self.field_data, self.coherence):
                        available.append(transition)
                else:
                    available.append(transition)
                    
        return available
    
    def apply_transition(self, transition: StateTransition) -> Tuple[np.ndarray, float]:
        """
        Apply a transition to the current field.
        
        Args:
            transition: The transition to apply
            
        Returns:
            Tuple of (new_field_data, new_coherence)
        """
        if self.field_data is None:
            logger.error("Cannot apply transition: No field data set")
            return None, 0.0
            
        if transition.from_state != self.current_state:
            logger.warning(f"Transition {transition} not applicable from current state {self.current_state}")
            return self.field_data, self.coherence
            
        if not transition.can_transition(self.field_data, self.coherence):
            logger.warning(f"Transition {transition} condition not met")
            return self.field_data, self.coherence
            
        # Apply transition
        new_field = transition.execute(self.field_data)
        new_coherence = self._calculate_coherence(new_field)
        
        # Update state
        self.field_data = new_field
        self.coherence = new_coherence
        self.current_state = transition.to_state
        
        logger.info(f"Applied transition: {transition}")
        logger.info(f"New state: {self.current_state} (coherence: {self.coherence:.4f})")
        
        return new_field, new_coherence
    
    def run_auto_transition(self) -> Optional[StateTransition]:
        """
        Automatically select and apply the best available transition.
        
        Returns:
            The transition that was applied, or None if no transition was available
        """
        available = self.get_available_transitions()
        
        if not available:
            logger.info(f"No transitions available from state {self.current_state}")
            return None
            
        # Select the first available transition (in a more complex system,
        # we could implement a coherence-optimizing selection algorithm)
        selected = available[0]
        
        # Apply it
        self.apply_transition(selected)
        
        return selected
    
    def get_state_property(self, state_name: str, property_name: str) -> Any:
        """
        Get a property of a state.
        
        Args:
            state_name: Name of the state
            property_name: Name of the property
            
        Returns:
            The property value, or None if not found
        """
        if state_name not in self.states:
            return None
            
        return self.states[state_name].get(property_name)
    
    def get_current_frequency(self) -> str:
        """
        Get the frequency of the current state.
        
        Returns:
            The frequency name
        """
        if not self.current_state:
            return "love"  # Default
            
        freq = self.get_state_property(self.current_state, "frequency")
        return freq if freq else "love"
    
    def visualize_state_machine(self):
        """
        Visualize the state machine as ASCII art.
        
        Returns:
            ASCII representation of the state machine
        """
        if not self.states or not self.transitions:
            return "Empty state machine"
            
        result = []
        result.append("PhiFlow State Machine")
        result.append("=" * 40)
        
        # List states
        result.append("States:")
        for name, props in self.states.items():
            freq = props.get("frequency", "love")
            coherence = props.get("min_coherence", 0.0)
            compression = props.get("compression", 1.0)
            
            state_desc = f"  {name}: freq={freq}, coherence>={coherence}, compression={compression}"
            if name == self.current_state:
                state_desc += " (CURRENT)"
            result.append(state_desc)
            
        # List transitions
        result.append("\nTransitions:")
        for transition in self.transitions:
            ops = ", ".join([str(op) for op in transition.operations])
            cond = f" when {transition.condition}" if transition.condition else ""
            result.append(f"  {transition.from_state} -> {transition.to_state}{cond} [{ops}]")
            
        result.append("=" * 40)
        
        return "\n".join(result)
        

def run_philflow(code: str, field_data: np.ndarray, max_transitions: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Run PhiFlow code on a quantum field.
    
    Args:
        code: PhiFlow DSL code
        field_data: Initial quantum field data
        max_transitions: Maximum number of automatic transitions to apply
        
    Returns:
        Tuple of (final_field_data, transition_log)
    """
    interpreter = PhiFlowInterpreter()
    interpreter.load(code)
    interpreter.set_field(field_data)
    
    transition_log = []
    
    # Visualize initial state
    transition_log.append(interpreter.visualize_state_machine())
    
    # Run transitions
    for i in range(max_transitions):
        transition = interpreter.run_auto_transition()
        if not transition:
            transition_log.append(f"Step {i+1}: No available transitions, execution halted")
            break
            
        transition_log.append(f"Step {i+1}: Applied {transition} (coherence: {interpreter.coherence:.4f})")
    
    return interpreter.field_data, transition_log