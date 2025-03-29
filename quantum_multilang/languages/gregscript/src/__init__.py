"""
GregScript - A language for harmony, patterns, and rhythmic structures

GregScript is a specialized language for defining and recognizing patterns,
particularly focused on phi-harmonic relationships and dynamic rhythms.
"""

from .parser import parse_gregscript
from .interpreter import GregScriptInterpreter
from .patterns import Pattern, Rhythm, Harmony
from .analyzer import PatternAnalyzer

__all__ = [
    'parse_gregscript', 
    'GregScriptInterpreter', 
    'Pattern', 
    'Rhythm', 
    'Harmony',
    'PatternAnalyzer'
]