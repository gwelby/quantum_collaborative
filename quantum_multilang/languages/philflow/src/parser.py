"""
PhiFlow DSL Parser

This module provides parsing functionality for the PhiFlow DSL,
converting the textual representation into an abstract syntax tree.
"""

import re
import ast
from typing import Dict, List, Tuple, Any, Optional, Union

from .transitions import StateTransition, TransitionOperator, FieldOperation

class PhiFlowSyntaxError(Exception):
    """Exception raised for syntax errors in PhiFlow DSL code."""
    pass

class PhiFlowParser:
    """Parser for PhiFlow DSL."""
    
    def __init__(self):
        """Initialize the parser."""
        # Regular expressions for lexical analysis
        self.token_patterns = [
            ('COMMENT', r'#.*'),
            ('STATE', r'state\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('TRANSITION', r'transition\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*->\s*([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('WHEN', r'when\s+(.+?)\s+then'),
            ('WITH', r'with\s+(.+)'),
            ('FREQ', r'frequency\s*=\s*([0-9.]+|love|unity|cascade|truth|vision|oneness)'),
            ('COHERENCE', r'coherence\s*>=\s*([0-9.]+)'),
            ('COMPRESSION', r'compression\s*=\s*([0-9.]+)'),
            ('FIELD_OP', r'(amplify|attenuate|rotate|harmonize|blend|center|expand|contract)(\s+by\s+([0-9.φΦ]+))?'),
            ('PHI_VAL', r'φ\^?([0-9.]+)?'),
            ('NUMBER', r'[0-9]+(\.[0-9]+)?'),
            ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('NEWLINE', r'\n'),
            ('WHITESPACE', r'[ \t]+'),
        ]
        
        # Compile regex patterns
        self.patterns = [(name, re.compile(pattern)) for name, pattern in self.token_patterns]
    
    def tokenize(self, code: str) -> List[Tuple[str, str]]:
        """
        Convert PhiFlow code into a list of tokens.
        
        Args:
            code: PhiFlow DSL code as string
            
        Returns:
            List of (token_type, token_value) tuples
        """
        tokens = []
        position = 0
        line = 1
        
        while position < len(code):
            match = None
            
            for token_type, pattern in self.patterns:
                match = pattern.match(code, position)
                if match:
                    value = match.group(0)
                    
                    if token_type != 'WHITESPACE':  # Skip whitespace
                        if token_type == 'NEWLINE':
                            line += 1
                        else:
                            tokens.append((token_type, value, line))
                    
                    position = match.end()
                    break
            
            if not match:
                # No match found, report error
                error_context = code[max(0, position-10):position+10]
                raise PhiFlowSyntaxError(f"Invalid syntax at line {line}, position {position}: '{error_context}'")
        
        return tokens
    
    def parse(self, code: str) -> List[Union[StateTransition, Dict[str, Any]]]:
        """
        Parse PhiFlow code into a list of state transitions and definitions.
        
        Args:
            code: PhiFlow DSL code as string
            
        Returns:
            List of StateTransition objects and state definitions
        """
        tokens = self.tokenize(code)
        
        # State tracking
        states = {}
        transitions = []
        current_state = None
        current_transition = None
        
        i = 0
        while i < len(tokens):
            token_type, value, line = tokens[i]
            
            if token_type == 'COMMENT':
                # Skip comments
                pass
            
            elif token_type == 'STATE':
                # Parse state definition
                state_name = re.match(r'state\s+([a-zA-Z_][a-zA-Z0-9_]*)', value).group(1)
                states[state_name] = {'name': state_name, 'line': line}
                current_state = state_name
                
                # Look ahead for state properties
                j = i + 1
                while j < len(tokens):
                    prop_type, prop_value, _ = tokens[j]
                    if prop_type == 'FREQ':
                        freq_match = re.match(r'frequency\s*=\s*([0-9.]+|love|unity|cascade|truth|vision|oneness)', prop_value)
                        states[state_name]['frequency'] = freq_match.group(1)
                    elif prop_type == 'COHERENCE':
                        coherence_match = re.match(r'coherence\s*>=\s*([0-9.]+)', prop_value)
                        states[state_name]['min_coherence'] = float(coherence_match.group(1))
                    elif prop_type == 'COMPRESSION':
                        compression_match = re.match(r'compression\s*=\s*([0-9.]+)', prop_value)
                        states[state_name]['compression'] = float(compression_match.group(1))
                    elif prop_type in ('STATE', 'TRANSITION'):
                        break
                    j += 1
                    
                i = j - 1  # Continue from last property
            
            elif token_type == 'TRANSITION':
                # Parse transition definition
                match = re.match(r'transition\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*->\s*([a-zA-Z_][a-zA-Z0-9_]*)', value)
                from_state, to_state = match.group(1), match.group(2)
                
                # Create transition object
                transition = StateTransition(from_state=from_state, to_state=to_state)
                transitions.append(transition)
                current_transition = transition
                
                # Look ahead for transition properties
                j = i + 1
                while j < len(tokens):
                    prop_type, prop_value, _ = tokens[j]
                    if prop_type == 'WHEN':
                        condition_match = re.match(r'when\s+(.+?)\s+then', prop_value)
                        transition.set_condition(condition_match.group(1))
                    elif prop_type == 'FIELD_OP':
                        op_match = re.match(r'(amplify|attenuate|rotate|harmonize|blend|center|expand|contract)(\s+by\s+([0-9.φΦ]+))?', prop_value)
                        op_name = op_match.group(1)
                        op_value = op_match.group(3) if op_match.group(3) else "1.0"
                        
                        # Handle phi values
                        if op_value.startswith('φ'):
                            if len(op_value) > 1 and op_value[1] == '^':
                                power = float(op_value[2:]) if len(op_value) > 2 else 1.0
                                op_value = f"PHI^{power}"
                            else:
                                op_value = "PHI"
                        
                        transition.add_operation(FieldOperation(op_name, op_value))
                    elif prop_type in ('STATE', 'TRANSITION'):
                        break
                    j += 1
                    
                i = j - 1  # Continue from last property
            
            i += 1
        
        return [states, transitions]

def parse_philflow(code: str) -> Tuple[Dict[str, Dict[str, Any]], List[StateTransition]]:
    """
    Parse PhiFlow DSL code.
    
    Args:
        code: PhiFlow DSL code as string
        
    Returns:
        Tuple of (states, transitions)
    """
    parser = PhiFlowParser()
    result = parser.parse(code)
    return result[0], result[1]