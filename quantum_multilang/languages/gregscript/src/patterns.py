"""
GregScript Pattern Structures

This module defines the core pattern structures used in GregScript,
including rhythms, harmonies, and complex patterns.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
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
    SACRED_FREQUENCIES = sc.SACRED_FREQUENCIES
except ImportError:
    # Fallback constants
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

class PatternElement:
    """Base class for all pattern elements in GregScript."""
    
    def __init__(self, name: str):
        """
        Initialize a pattern element.
        
        Args:
            name: Name of the pattern element
        """
        self.name = name
    
    def match(self, data: np.ndarray) -> float:
        """
        Match this pattern against data, returning a score.
        
        Args:
            data: Data to match against
            
        Returns:
            Match score between 0.0 and 1.0
        """
        # Base implementation always returns 0.0
        return 0.0
        
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate data based on this pattern.
        
        Args:
            shape: Shape of the data to generate
            
        Returns:
            Generated data
        """
        # Base implementation returns zeros
        return np.zeros(shape)
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Rhythm(PatternElement):
    """
    Represents a rhythm pattern in GregScript.
    
    Rhythms define sequences of pulses or oscillations based on
    phi-harmonic principles.
    """
    
    def __init__(self, name: str, sequence: List[float], tempo: float = 1.0):
        """
        Initialize a rhythm pattern.
        
        Args:
            name: Name of the rhythm
            sequence: Sequence of pulse strengths (0.0 to 1.0)
            tempo: Speed factor for the rhythm
        """
        super().__init__(name)
        self.sequence = sequence
        self.tempo = tempo
        
    def match(self, data: np.ndarray) -> float:
        """
        Match this rhythm against temporal data.
        
        Args:
            data: 1D temporal data to match against
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if len(data.shape) != 1:
            # Only match against 1D data
            return 0.0
            
        # Normalize data
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Calculate sequence points
        length = len(normalized)
        seq_length = len(self.sequence)
        
        if length < seq_length:
            # Data too short for matching
            return 0.0
            
        # Calculate pattern repetitions
        repetitions = length // seq_length
        
        # Stretch sequence to match data
        stretched_seq = np.repeat(self.sequence, repetitions)
        stretched_seq = stretched_seq[:length]  # Truncate if needed
        
        # Calculate correlation
        correlation = np.corrcoef(normalized, stretched_seq)[0, 1]
        
        # Convert to score (0.0 to 1.0)
        score = (correlation + 1.0) / 2.0
        
        return score
    
    def generate(self, length: int) -> np.ndarray:
        """
        Generate a rhythm sequence.
        
        Args:
            length: Length of the sequence to generate
            
        Returns:
            Generated rhythm sequence
        """
        # Calculate number of repetitions needed
        seq_length = len(self.sequence)
        repetitions = (length + seq_length - 1) // seq_length
        
        # Create repeated sequence
        raw_sequence = np.tile(self.sequence, repetitions)
        
        # Apply tempo variations (phi-based)
        t = np.linspace(0, length * self.tempo / PHI, length)
        phi_factor = np.sin(t * np.pi * 2) * 0.2 + 1.0
        
        # Blend original sequence with phi factor
        result = raw_sequence[:length] * phi_factor
        
        # Normalize to 0.0 - 1.0
        if np.max(result) > np.min(result):
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        
        return result


class Harmony(PatternElement):
    """
    Represents a harmonic pattern in GregScript.
    
    Harmonies define resonant relationships between different elements,
    often based on sacred frequencies and phi ratios.
    """
    
    def __init__(self, name: str, frequency: Union[str, float], 
                 overtones: List[float] = None, phase: float = 0.0):
        """
        Initialize a harmony pattern.
        
        Args:
            name: Name of the harmony
            frequency: Base frequency (either a name from SACRED_FREQUENCIES or a number)
            overtones: List of overtone strength ratios
            phase: Phase offset (0.0 to 1.0)
        """
        super().__init__(name)
        
        # Parse frequency
        if isinstance(frequency, str) and frequency in SACRED_FREQUENCIES:
            self.frequency = SACRED_FREQUENCIES[frequency]
        else:
            self.frequency = float(frequency)
        
        # Initialize overtones with default phi-harmonic series if not provided
        if overtones is None:
            self.overtones = [1.0, 1.0/PHI, 1.0/PHI_PHI, 1.0/(PHI*PHI)]
        else:
            self.overtones = overtones
            
        self.phase = phase
    
    def match(self, data: np.ndarray) -> float:
        """
        Match this harmony against frequency data.
        
        Args:
            data: Data to match against (interpreted as frequency domain)
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if len(data.shape) != 1:
            # Only match against 1D data
            return 0.0
            
        # Perform FFT if data seems to be in time domain
        if len(data) > 10:
            fft_data = np.abs(np.fft.rfft(data))
            freq_data = fft_data / np.max(fft_data)
        else:
            # Assume data is already frequency domain
            freq_data = data / np.max(data)
            
        # Generate harmony signature
        harmony_sig = self.generate_frequency_signature(len(freq_data))
        
        # Calculate correlation
        correlation = np.corrcoef(freq_data, harmony_sig)[0, 1]
        
        # Convert to score (0.0 to 1.0)
        score = (correlation + 1.0) / 2.0
        
        return score
    
    def generate_frequency_signature(self, length: int) -> np.ndarray:
        """
        Generate a frequency domain signature for this harmony.
        
        Args:
            length: Length of the signature to generate
            
        Returns:
            Frequency domain signature
        """
        signature = np.zeros(length)
        
        # Calculate frequency indices for each overtone
        base_idx = int(self.frequency * length / (44100 / 2))  # Assuming 44.1kHz sample rate
        if base_idx >= length:
            base_idx = length // 10  # Fallback
        
        # Add base frequency and overtones
        for i, strength in enumerate(self.overtones):
            idx = base_idx * (i + 1)
            if idx < length:
                signature[idx] = strength
                
        # Apply phase smoothing
        if self.phase > 0:
            # Smooth with a Gaussian kernel
            from scipy import ndimage
            sigma = self.phase * length / 20
            signature = ndimage.gaussian_filter1d(signature, sigma)
                
        return signature
    
    def generate(self, length: int) -> np.ndarray:
        """
        Generate a harmonic waveform.
        
        Args:
            length: Length of the waveform to generate
            
        Returns:
            Generated harmonic waveform
        """
        t = np.linspace(0, 2*np.pi, length)
        signal = np.zeros(length)
        
        # Add each overtone
        for i, strength in enumerate(self.overtones):
            signal += strength * np.sin(t * (i + 1) * self.frequency / 100 + self.phase * 2*np.pi)
            
        # Normalize
        if np.max(signal) > np.min(signal):
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
        return signal


class Pattern(PatternElement):
    """
    Represents a complex pattern in GregScript.
    
    Patterns can combine rhythms, harmonies, and structural elements
    to recognize and generate complex phi-harmonic signatures.
    """
    
    def __init__(self, name: str, elements: List[PatternElement] = None, 
                 weights: List[float] = None):
        """
        Initialize a complex pattern.
        
        Args:
            name: Name of the pattern
            elements: List of component pattern elements
            weights: Weight for each element (phi-based defaults if None)
        """
        super().__init__(name)
        self.elements = elements or []
        
        # Set phi-harmonic weights if not provided
        if weights is None:
            self.weights = [PHI ** (-i) for i in range(len(self.elements))]
            # Normalize weights
            weight_sum = sum(self.weights)
            if weight_sum > 0:
                self.weights = [w / weight_sum for w in self.weights]
        else:
            self.weights = weights
            
    def add_element(self, element: PatternElement, weight: float = None):
        """
        Add an element to this pattern.
        
        Args:
            element: Pattern element to add
            weight: Weight for this element (default: phi-based)
        """
        self.elements.append(element)
        
        if weight is None:
            # Set phi-harmonic weight
            weight = PHI ** (-len(self.elements) + 1)
            
        # Add weight and renormalize
        self.weights.append(weight)
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
    
    def match(self, data: np.ndarray) -> float:
        """
        Match this pattern against data.
        
        Args:
            data: Data to match against
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not self.elements:
            return 0.0
            
        # Match each element and combine with weights
        scores = []
        for element, weight in zip(self.elements, self.weights):
            score = element.match(data)
            scores.append(score * weight)
            
        return sum(scores)
    
    def generate(self, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Generate data based on this pattern.
        
        Args:
            shape: Shape of the data to generate
            
        Returns:
            Generated data
        """
        if not self.elements:
            if isinstance(shape, int):
                return np.zeros(shape)
            else:
                return np.zeros(shape)
                
        # Convert shape to tuple if needed
        if isinstance(shape, int):
            shape = (shape,)
            
        # Initialize result array
        result = np.zeros(shape)
        
        # Generate data from each element and combine with weights
        for element, weight in zip(self.elements, self.weights):
            # Handle different element types differently
            if isinstance(element, Rhythm) and len(shape) == 1:
                # Generate rhythm for 1D data
                result += element.generate(shape[0]) * weight
            elif isinstance(element, Harmony) and len(shape) == 1:
                # Generate harmony for 1D data
                result += element.generate(shape[0]) * weight
            elif isinstance(element, Pattern):
                # Recursively generate from sub-pattern
                subpattern = element.generate(shape)
                result += subpattern * weight
                
        # Normalize
        if np.max(result) > np.min(result):
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
            
        return result