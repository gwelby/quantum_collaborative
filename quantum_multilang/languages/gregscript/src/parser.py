"""
GregScript Parser

This module provides parsing functionality for the GregScript language,
converting the textual representation into pattern objects.
"""

import re
import ast
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from .patterns import Pattern, Rhythm, Harmony

class GregScriptSyntaxError(Exception):
    """Exception raised for syntax errors in GregScript code."""
    pass

class GregScriptParser:
    """Parser for GregScript language."""
    
    def __init__(self):
        """Initialize the parser."""
        # Regular expressions for lexical analysis
        self.token_patterns = [
            ('COMMENT', r'//.*'),
            ('PATTERN', r'pattern\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('RHYTHM', r'rhythm\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('HARMONY', r'harmony\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('SEQUENCE', r'sequence\s*=\s*\[([0-9., ]+)\]'),
            ('FREQUENCY', r'frequency\s*=\s*([0-9.]+|love|unity|cascade|truth|vision|oneness)'),
            ('OVERTONES', r'overtones\s*=\s*\[([0-9., ]+)\]'),
            ('TEMPO', r'tempo\s*=\s*([0-9.φΦ\^]+)'),
            ('PHASE', r'phase\s*=\s*([0-9.]+)'),
            ('WEIGHT', r'weight\s*=\s*([0-9.φΦ\^]+)'),
            ('REFERENCE', r'use\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            ('PHI_VAL', r'φ\^?([0-9.]+)?'),
            ('NUMBER', r'[0-9]+(\.[0-9]+)?'),
            ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('OPEN_BRACE', r'\{'),
            ('CLOSE_BRACE', r'\}'),
            ('NEWLINE', r'\n'),
            ('WHITESPACE', r'[ \t]+'),
        ]
        
        # Compile regex patterns
        self.patterns = [(name, re.compile(pattern)) for name, pattern in self.token_patterns]
    
    def tokenize(self, code: str) -> List[Tuple[str, str, int]]:
        """
        Convert GregScript code into a list of tokens.
        
        Args:
            code: GregScript code as string
            
        Returns:
            List of (token_type, token_value, line_number) tuples
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
                raise GregScriptSyntaxError(f"Invalid syntax at line {line}, position {position}: '{error_context}'")
        
        return tokens
    
    def _parse_phi_value(self, value_str: str) -> float:
        """Parse a phi value string to a float."""
        if 'φ' in value_str or 'Φ' in value_str:
            value_str = value_str.replace('Φ', 'φ')  # Normalize to lowercase phi
            if '^' in value_str:
                power = float(value_str.split('^')[1])
                from .patterns import PHI
                return PHI ** power
            else:
                from .patterns import PHI
                return PHI
        else:
            try:
                return float(value_str)
            except ValueError:
                return 1.0
    
    def _parse_sequence(self, sequence_str: str) -> List[float]:
        """Parse a sequence string to a list of floats."""
        sequence = []
        
        # Clean and split sequence string
        sequence_str = sequence_str.strip('[]').strip()
        if sequence_str:
            for val in sequence_str.split(','):
                try:
                    sequence.append(float(val.strip()))
                except ValueError:
                    # Skip invalid values
                    pass
        
        return sequence
    
    def parse(self, code: str) -> Dict[str, Union[Pattern, Rhythm, Harmony]]:
        """
        Parse GregScript code into pattern objects.
        
        Args:
            code: GregScript code as string
            
        Returns:
            Dictionary of pattern objects indexed by name
        """
        tokens = self.tokenize(code)
        
        # Initialize pattern tracking
        all_patterns = {}
        current_pattern = None
        current_type = None
        in_definition = False
        
        i = 0
        while i < len(tokens):
            token_type, value, line = tokens[i]
            
            if token_type == 'COMMENT':
                # Skip comments
                pass
            
            elif token_type == 'PATTERN':
                # Parse pattern definition
                pattern_name = re.match(r'pattern\s+([a-zA-Z_][a-zA-Z0-9_]*)', value).group(1)
                current_pattern = Pattern(pattern_name)
                current_type = 'pattern'
                
                # Look for open brace
                j = i + 1
                while j < len(tokens) and tokens[j][0] != 'OPEN_BRACE':
                    j += 1
                    
                if j < len(tokens) and tokens[j][0] == 'OPEN_BRACE':
                    in_definition = True
                    all_patterns[pattern_name] = current_pattern
                    i = j
            
            elif token_type == 'RHYTHM':
                # Parse rhythm definition
                rhythm_name = re.match(r'rhythm\s+([a-zA-Z_][a-zA-Z0-9_]*)', value).group(1)
                
                # Default values
                sequence = [1.0, 0.0, 0.5, 0.0]  # Default phi-ish rhythm
                tempo = 1.0
                
                # Look ahead for rhythm properties
                j = i + 1
                while j < len(tokens) and tokens[j][0] != 'OPEN_BRACE':
                    prop_type, prop_value, _ = tokens[j]
                    
                    if prop_type == 'SEQUENCE':
                        seq_match = re.match(r'sequence\s*=\s*\[([0-9., ]+)\]', prop_value)
                        if seq_match:
                            sequence = self._parse_sequence(seq_match.group(1))
                    
                    elif prop_type == 'TEMPO':
                        tempo_match = re.match(r'tempo\s*=\s*([0-9.φΦ\^]+)', prop_value)
                        if tempo_match:
                            tempo = self._parse_phi_value(tempo_match.group(1))
                    
                    j += 1
                
                # Create the rhythm object
                rhythm = Rhythm(rhythm_name, sequence, tempo)
                all_patterns[rhythm_name] = rhythm
                
                # If we're in a pattern definition, add it to the current pattern
                if in_definition and current_type == 'pattern':
                    weight = None
                    
                    # Look for weight property
                    k = j
                    while k < len(tokens) and tokens[k][0] != 'NEWLINE':
                        if tokens[k][0] == 'WEIGHT':
                            weight_match = re.match(r'weight\s*=\s*([0-9.φΦ\^]+)', tokens[k][1])
                            if weight_match:
                                weight = self._parse_phi_value(weight_match.group(1))
                        k += 1
                    
                    current_pattern.add_element(rhythm, weight)
                
                i = j - 1  # Continue from last property
            
            elif token_type == 'HARMONY':
                # Parse harmony definition
                harmony_name = re.match(r'harmony\s+([a-zA-Z_][a-zA-Z0-9_]*)', value).group(1)
                
                # Default values
                frequency = 'love'  # Default frequency
                overtones = None  # Will use default phi-based overtones
                phase = 0.0
                
                # Look ahead for harmony properties
                j = i + 1
                while j < len(tokens) and tokens[j][0] != 'OPEN_BRACE':
                    prop_type, prop_value, _ = tokens[j]
                    
                    if prop_type == 'FREQUENCY':
                        freq_match = re.match(r'frequency\s*=\s*([0-9.]+|love|unity|cascade|truth|vision|oneness)', prop_value)
                        if freq_match:
                            frequency = freq_match.group(1)
                    
                    elif prop_type == 'OVERTONES':
                        overtones_match = re.match(r'overtones\s*=\s*\[([0-9., ]+)\]', prop_value)
                        if overtones_match:
                            overtones = self._parse_sequence(overtones_match.group(1))
                    
                    elif prop_type == 'PHASE':
                        phase_match = re.match(r'phase\s*=\s*([0-9.]+)', prop_value)
                        if phase_match:
                            phase = float(phase_match.group(1))
                    
                    j += 1
                
                # Create the harmony object
                harmony = Harmony(harmony_name, frequency, overtones, phase)
                all_patterns[harmony_name] = harmony
                
                # If we're in a pattern definition, add it to the current pattern
                if in_definition and current_type == 'pattern':
                    weight = None
                    
                    # Look for weight property
                    k = j
                    while k < len(tokens) and tokens[k][0] != 'NEWLINE':
                        if tokens[k][0] == 'WEIGHT':
                            weight_match = re.match(r'weight\s*=\s*([0-9.φΦ\^]+)', tokens[k][1])
                            if weight_match:
                                weight = self._parse_phi_value(weight_match.group(1))
                        k += 1
                    
                    current_pattern.add_element(harmony, weight)
                
                i = j - 1  # Continue from last property
            
            elif token_type == 'REFERENCE' and in_definition:
                # Add a reference to an existing pattern
                ref_name = re.match(r'use\s+([a-zA-Z_][a-zA-Z0-9_]*)', value).group(1)
                
                if ref_name in all_patterns:
                    referenced_pattern = all_patterns[ref_name]
                    
                    # Look for weight property
                    weight = None
                    j = i + 1
                    while j < len(tokens) and tokens[j][0] != 'NEWLINE':
                        if tokens[j][0] == 'WEIGHT':
                            weight_match = re.match(r'weight\s*=\s*([0-9.φΦ\^]+)', tokens[j][1])
                            if weight_match:
                                weight = self._parse_phi_value(weight_match.group(1))
                        j += 1
                    
                    current_pattern.add_element(referenced_pattern, weight)
            
            elif token_type == 'CLOSE_BRACE':
                # End of definition
                in_definition = False
                current_pattern = None
                current_type = None
            
            i += 1
        
        return all_patterns

def parse_gregscript(code: str) -> Dict[str, Union[Pattern, Rhythm, Harmony]]:
    """
    Parse GregScript code.
    
    Args:
        code: GregScript code as string
        
    Returns:
        Dictionary of pattern objects indexed by name
    """
    parser = GregScriptParser()
    return parser.parse(code)