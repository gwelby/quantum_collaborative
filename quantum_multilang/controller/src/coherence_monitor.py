#!/usr/bin/env python3
"""
Coherence Monitor for Quantum Field Multi-Language Architecture

This module monitors the coherence of quantum fields across different language
components and tracks the overall system coherence.
"""

import time
import threading
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

class CoherenceMonitor:
    """
    Monitors quantum field coherence across different language components
    and provides analytics on system-wide coherence.
    """
    
    def __init__(self, history_length=100):
        """
        Initialize the coherence monitor.
        
        Args:
            history_length: Number of coherence measurements to keep in history
        """
        self.history_length = history_length
        self.coherence_history = deque(maxlen=history_length)
        self.language_coherence = {}
        self.lock = threading.RLock()
        self.last_update = time.time()
    
    def track_field_coherence(self, coherence: float, source_language: str) -> None:
        """
        Track the coherence of a quantum field.
        
        Args:
            coherence: The field coherence value
            source_language: The language that generated the field
        """
        with self.lock:
            # Add to overall history
            self.coherence_history.append((coherence, source_language, time.time()))
            
            # Update language-specific history
            if source_language not in self.language_coherence:
                self.language_coherence[source_language] = deque(maxlen=self.history_length)
            
            self.language_coherence[source_language].append((coherence, time.time()))
            
            # Update timestamp
            self.last_update = time.time()
    
    def get_system_coherence(self) -> float:
        """
        Calculate the overall system coherence.
        
        Returns:
            A float representing the system-wide coherence
        """
        with self.lock:
            if not self.coherence_history:
                return 0.0
            
            # Calculate phi-weighted average of recent coherence values
            phi = 1.618033988749895
            lambda_val = 0.618033988749895
            
            # Extract coherence values and calculate recency weights
            coherence_values = []
            weights = []
            
            for i, (coherence, _, timestamp) in enumerate(self.coherence_history):
                # More recent values have higher weight
                recency = lambda_val ** (len(self.coherence_history) - i - 1)
                coherence_values.append(coherence)
                weights.append(recency)
            
            # Calculate weighted average
            weighted_sum = sum(c * w for c, w in zip(coherence_values, weights))
            weight_sum = sum(weights)
            
            if weight_sum > 0:
                system_coherence = weighted_sum / weight_sum * phi
            else:
                system_coherence = 0.0
            
            return system_coherence
    
    def get_language_coherence(self, language: Optional[str] = None) -> Dict[str, float]:
        """
        Get the coherence for specific languages.
        
        Args:
            language: The language to get coherence for, or None for all languages
            
        Returns:
            A dictionary mapping language names to coherence values
        """
        with self.lock:
            result = {}
            
            if language is not None:
                # Return coherence for a specific language
                if language in self.language_coherence and self.language_coherence[language]:
                    values = [c for c, _ in self.language_coherence[language]]
                    result[language] = sum(values) / len(values)
                else:
                    result[language] = 0.0
            else:
                # Return coherence for all languages
                for lang, history in self.language_coherence.items():
                    if history:
                        values = [c for c, _ in history]
                        result[lang] = sum(values) / len(values)
                    else:
                        result[lang] = 0.0
            
            return result
    
    def get_coherence_trend(self, window_size: int = 10) -> float:
        """
        Calculate the trend in system coherence over time.
        
        Args:
            window_size: Number of recent measurements to analyze
            
        Returns:
            A float representing the trend (positive = improving, negative = declining)
        """
        with self.lock:
            if len(self.coherence_history) < window_size:
                return 0.0
            
            # Get recent coherence values
            recent = list(self.coherence_history)[-window_size:]
            coherence_values = [c for c, _, _ in recent]
            
            # Calculate trend using least squares
            x = np.arange(len(coherence_values))
            y = np.array(coherence_values)
            
            if len(x) < 2:
                return 0.0
            
            # Calculate slope
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator != 0:
                slope = numerator / denominator
            else:
                slope = 0.0
            
            return slope
    
    def generate_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive coherence report.
        
        Returns:
            A dictionary with coherence metrics and analytics
        """
        with self.lock:
            report = {
                "system_coherence": self.get_system_coherence(),
                "language_coherence": self.get_language_coherence(),
                "trend": self.get_coherence_trend(),
                "last_update": self.last_update,
                "measurement_count": len(self.coherence_history),
                "languages": list(self.language_coherence.keys())
            }
            
            return report

# Simple test
if __name__ == "__main__":
    monitor = CoherenceMonitor()
    
    # Simulate some coherence values
    monitor.track_field_coherence(0.85, "python")
    monitor.track_field_coherence(0.92, "rust")
    monitor.track_field_coherence(0.88, "cpp")
    monitor.track_field_coherence(0.90, "python")
    
    # Get system coherence
    system_coherence = monitor.get_system_coherence()
    print(f"System coherence: {system_coherence:.4f}")
    
    # Get language coherence
    language_coherence = monitor.get_language_coherence()
    for lang, coherence in language_coherence.items():
        print(f"{lang} coherence: {coherence:.4f}")
    
    # Generate report
    report = monitor.generate_report()
    print(f"Coherence trend: {report['trend']:.4f}")
    print(f"Measurement count: {report['measurement_count']}")
    print(f"Languages: {', '.join(report['languages'])}")