"""
GregScript Pattern Analyzer

This module provides tools for analyzing data to discover patterns,
rhythms, and harmonies that may be present.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import signal, fft

from .patterns import Pattern, Rhythm, Harmony, PHI, PHI_PHI

logger = logging.getLogger("gregscript.analyzer")

class PatternAnalyzer:
    """
    Analyzer for discovering patterns in data.
    
    The Pattern Analyzer uses phi-harmonic principles to identify
    potential rhythms, harmonies, and complex patterns in data.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_rhythm(self, data: np.ndarray, min_period: int = 4, 
                       max_period: int = 32) -> List[Dict[str, Any]]:
        """
        Analyze data for rhythmic patterns.
        
        Args:
            data: 1D data array to analyze
            min_period: Minimum rhythm period to consider
            max_period: Maximum rhythm period to consider
            
        Returns:
            List of discovered rhythms with scores and properties
        """
        if len(data.shape) != 1:
            # Only analyze 1D data
            return []
            
        # Normalize data
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Calculate autocorrelation
        autocorr = signal.correlate(normalized, normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=min_period)
        
        rhythms = []
        
        # For each potential period, extract a rhythm
        for period in peaks:
            if min_period <= period <= max_period:
                # Extract the sequence for this period
                sequence = []
                for i in range(period):
                    idx = i % len(normalized)
                    sequence.append(normalized[idx])
                    
                # Normalize sequence
                sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))
                
                # Calculate confidence score
                score = autocorr[period]
                
                # Detect tempo based on peaks
                peaks_in_seq, _ = signal.find_peaks(sequence, height=0.5)
                if len(peaks_in_seq) > 0:
                    # Use phi-weighting for tempo
                    tempo = PHI * len(peaks_in_seq) / period
                else:
                    tempo = 1.0
                    
                # Quantize sequence to reduce noise
                quant_seq = np.round(sequence * 4) / 4
                
                rhythms.append({
                    "period": int(period),
                    "sequence": quant_seq.tolist(),
                    "tempo": tempo,
                    "score": float(score),
                    "name": f"r{period}_{len(peaks_in_seq)}"
                })
                
        # Sort by score
        rhythms.sort(key=lambda r: r["score"], reverse=True)
        
        return rhythms
    
    def analyze_harmony(self, data: np.ndarray, 
                       sacred_freqs: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Analyze data for harmonic patterns.
        
        Args:
            data: 1D data array to analyze
            sacred_freqs: Dictionary of sacred frequency names and values
            
        Returns:
            List of discovered harmonies with scores and properties
        """
        if len(data.shape) != 1:
            # Only analyze 1D data
            return []
            
        # Use default sacred frequencies if not provided
        if sacred_freqs is None:
            from .patterns import SACRED_FREQUENCIES
            sacred_freqs = SACRED_FREQUENCIES
            
        # Normalize data
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Calculate FFT
        fft_data = np.abs(np.fft.rfft(normalized))
        freqs = np.fft.rfftfreq(len(normalized))
        
        # Normalize FFT data
        fft_normalized = fft_data / np.max(fft_data)
        
        # Find peaks in FFT
        peaks, properties = signal.find_peaks(fft_normalized, height=0.3, distance=3)
        
        harmonies = []
        
        # For each peak, analyze harmonics
        for i, peak in enumerate(peaks):
            if peak == 0:  # Skip DC component
                continue
                
            # Find the peak frequency
            peak_freq = freqs[peak]
            peak_height = fft_normalized[peak]
            
            # Find overtones
            overtones = []
            for j in range(1, 5):
                overtone_idx = int(peak * j)
                if overtone_idx < len(fft_normalized):
                    overtones.append(fft_normalized[overtone_idx])
                else:
                    overtones.append(0.0)
            
            # Calculate phase spectrum
            phase_spectrum = np.angle(np.fft.rfft(normalized))
            phase = phase_spectrum[peak] / (2 * np.pi)  # Normalize to 0-1
            phase = (phase + 1) / 2  # Convert from -1...1 to 0...1
            
            # Find closest sacred frequency
            closest_sacred = None
            closest_distance = float('inf')
            
            for name, freq in sacred_freqs.items():
                # Scale sacred frequency to the same range as FFT frequencies
                scaled_freq = freq / 1000  # Simple scaling for demonstration
                distance = abs(peak_freq - scaled_freq)
                
                if distance < closest_distance:
                    closest_sacred = name
                    closest_distance = distance
            
            # Create harmony data
            harmony = {
                "peak_freq": float(peak_freq * 1000),  # Scale back to Hz range
                "closest_sacred": closest_sacred,
                "overtones": overtones,
                "phase": float(phase),
                "score": float(peak_height),
                "name": f"h{peak}_{closest_sacred}"
            }
            
            harmonies.append(harmony)
        
        # Sort by score
        harmonies.sort(key=lambda h: h["score"], reverse=True)
        
        return harmonies
    
    def analyze_patterns(self, data: np.ndarray, rhythms: List[Dict[str, Any]] = None, 
                       harmonies: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Analyze data for complex patterns combining rhythms and harmonies.
        
        Args:
            data: Data array to analyze
            rhythms: List of discovered rhythms
            harmonies: List of discovered harmonies
            
        Returns:
            List of discovered patterns with scores and properties
        """
        # Ensure we have rhythms and harmonies to work with
        if rhythms is None:
            rhythms = self.analyze_rhythm(data)
            
        if harmonies is None:
            harmonies = self.analyze_harmony(data)
            
        if not rhythms or not harmonies:
            return []
            
        patterns = []
        
        # Try combining the top rhythms and harmonies
        for r_idx, rhythm in enumerate(rhythms[:3]):  # Consider top 3 rhythms
            for h_idx, harmony in enumerate(harmonies[:3]):  # Consider top 3 harmonies
                # Create pattern name
                pattern_name = f"p{r_idx}_{h_idx}"
                
                # Calculate phi-weighted pattern elements
                r_weight = PHI / (r_idx + 1)
                h_weight = PHI / (h_idx + PHI)
                
                # Combined score
                combined_score = (rhythm["score"] * r_weight + harmony["score"] * h_weight) / (r_weight + h_weight)
                
                # Create pattern data
                pattern = {
                    "name": pattern_name,
                    "rhythm": rhythm["name"],
                    "harmony": harmony["name"],
                    "rhythm_weight": float(r_weight),
                    "harmony_weight": float(h_weight),
                    "score": float(combined_score),
                    "elements": [rhythm["name"], harmony["name"]]
                }
                
                patterns.append(pattern)
        
        # Sort by score
        patterns.sort(key=lambda p: p["score"], reverse=True)
        
        return patterns
    
    def discover_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Discover all types of patterns in data.
        
        Args:
            data: Data array to analyze
            
        Returns:
            Dictionary with discovered rhythms, harmonies, and patterns
        """
        # Ensure data is 1D
        if len(data.shape) > 1:
            # Flatten multi-dimensional data
            flattened = data.flatten()
        else:
            flattened = data
            
        # Analyze rhythms
        rhythms = self.analyze_rhythm(flattened)
        
        # Analyze harmonies
        harmonies = self.analyze_harmony(flattened)
        
        # Analyze complex patterns
        patterns = self.analyze_patterns(flattened, rhythms, harmonies)
        
        # Prepare results
        results = {
            "rhythms": rhythms,
            "harmonies": harmonies,
            "patterns": patterns,
        }
        
        # Add top picks for each category
        if rhythms:
            results["top_rhythm"] = rhythms[0]
        if harmonies:
            results["top_harmony"] = harmonies[0]
        if patterns:
            results["top_pattern"] = patterns[0]
            
        return results
    
    def generate_gregscript(self, discovered: Dict[str, Any]) -> str:
        """
        Generate GregScript code from discovered patterns.
        
        Args:
            discovered: Dictionary with discovered patterns
            
        Returns:
            GregScript code string
        """
        code = "// GregScript generated from pattern analysis\n\n"
        
        # Generate rhythm definitions
        if "rhythms" in discovered and discovered["rhythms"]:
            code += "// Rhythm definitions\n"
            for rhythm in discovered["rhythms"][:3]:  # Top 3 rhythms
                name = rhythm["name"]
                sequence = rhythm["sequence"]
                tempo = rhythm["tempo"]
                
                code += f"rhythm {name} sequence=[{', '.join([str(x) for x in sequence])}] tempo={tempo:.2f}\n\n"
        
        # Generate harmony definitions
        if "harmonies" in discovered and discovered["harmonies"]:
            code += "// Harmony definitions\n"
            for harmony in discovered["harmonies"][:3]:  # Top 3 harmonies
                name = harmony["name"]
                freq = harmony["peak_freq"]
                sacred = harmony["closest_sacred"]
                overtones = harmony["overtones"]
                phase = harmony["phase"]
                
                # Use sacred frequency name if close
                freq_str = sacred if sacred else str(freq)
                
                code += f"harmony {name} frequency={freq_str} "
                code += f"overtones=[{', '.join([f'{x:.2f}' for x in overtones])}] "
                code += f"phase={phase:.2f}\n\n"
        
        # Generate pattern definitions
        if "patterns" in discovered and discovered["patterns"]:
            code += "// Pattern definitions\n"
            for pattern in discovered["patterns"][:3]:  # Top 3 patterns
                name = pattern["name"]
                rhythm = pattern["rhythm"]
                harmony = pattern["harmony"]
                r_weight = pattern["rhythm_weight"]
                h_weight = pattern["harmony_weight"]
                
                code += f"pattern {name} {{\n"
                code += f"    use {rhythm} weight={r_weight:.2f}\n"
                code += f"    use {harmony} weight={h_weight:.2f}\n"
                code += "}\n\n"
        
        return code