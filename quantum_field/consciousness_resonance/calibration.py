"""
Phi Calibration System for the Consciousness-Field Resonance Engine.

This module provides tools for calibrating and optimizing the resonance
between consciousness states and quantum fields to achieve perfect phi-harmonic
alignment for manifestation.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from .patterns import ThoughtPattern, ResonancePattern
from .manifestation import ManifestationMatrix


@dataclass
class CalibrationProfile:
    """
    A calibration profile storing optimal resonance parameters for a specific user/field.
    
    The CalibrationProfile contains the optimized parameters that achieve maximum
    phi-harmonic alignment between a specific consciousness signature and field type.
    """
    # User/consciousness identification
    profile_id: str
    user_id: Optional[str] = None
    
    # Phi-harmonic calibration parameters
    phi_alignment: float = 0.8
    field_resonance: Dict[str, float] = field(default_factory=dict)
    thought_resonance: Dict[str, float] = field(default_factory=dict)
    frequency_weights: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    coherence_threshold: float = 0.618
    manifestation_efficiency: float = 0.7
    extraction_efficiency: float = 0.7
    
    # Metadata
    creation_timestamp: float = field(default_factory=time.time)
    last_update_timestamp: float = field(default_factory=time.time)
    calibration_version: str = "1.0.0"
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        # Initialize default field resonance values if empty
        if not self.field_resonance:
            self.field_resonance = {
                "energetic": 0.8,
                "spatial": 0.7,
                "temporal": 0.65,
                "intentional": 0.75,
                "harmonic": 0.85
            }
        
        # Initialize default thought resonance values if empty
        if not self.thought_resonance:
            self.thought_resonance = {
                "focus": 0.7,
                "intention": 0.75,
                "visualization": 0.8,
                "emotion": 0.65,
                "intuition": 0.7
            }
        
        # Initialize default frequency weights if empty
        if not self.frequency_weights:
            self.frequency_weights = {name: 1.0 for name in SACRED_FREQUENCIES}
    
    def to_dict(self) -> Dict:
        """Convert the profile to a dictionary for storage."""
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "phi_alignment": self.phi_alignment,
            "field_resonance": self.field_resonance,
            "thought_resonance": self.thought_resonance,
            "frequency_weights": self.frequency_weights,
            "coherence_threshold": self.coherence_threshold,
            "manifestation_efficiency": self.manifestation_efficiency,
            "extraction_efficiency": self.extraction_efficiency,
            "creation_timestamp": self.creation_timestamp,
            "last_update_timestamp": self.last_update_timestamp,
            "calibration_version": self.calibration_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationProfile':
        """Create a profile from a dictionary."""
        return cls(
            profile_id=data["profile_id"],
            user_id=data.get("user_id"),
            phi_alignment=data.get("phi_alignment", 0.8),
            field_resonance=data.get("field_resonance", {}),
            thought_resonance=data.get("thought_resonance", {}),
            frequency_weights=data.get("frequency_weights", {}),
            coherence_threshold=data.get("coherence_threshold", 0.618),
            manifestation_efficiency=data.get("manifestation_efficiency", 0.7),
            extraction_efficiency=data.get("extraction_efficiency", 0.7),
            creation_timestamp=data.get("creation_timestamp", time.time()),
            last_update_timestamp=data.get("last_update_timestamp", time.time()),
            calibration_version=data.get("calibration_version", "1.0.0")
        )
    
    def update_from_session(self, field_coherence: float, thought_coherence: float,
                          active_frequencies: Dict[str, float]) -> None:
        """
        Update the calibration profile based on a session's results.
        
        Args:
            field_coherence: Achieved field coherence
            thought_coherence: Achieved thought coherence
            active_frequencies: Dictionary of frequency names to activation strengths
        """
        # Update phi alignment with weighted average
        self.phi_alignment = self.phi_alignment * 0.7 + min(field_coherence, thought_coherence) * 0.3
        
        # Update frequency weights based on active frequencies
        for freq_name, strength in active_frequencies.items():
            if freq_name in self.frequency_weights:
                # Update with exponential moving average
                self.frequency_weights[freq_name] = (
                    self.frequency_weights[freq_name] * 0.8 + strength * 0.2
                )
        
        # Update timestamp
        self.last_update_timestamp = time.time()
    
    def optimize_for_manifestation(self) -> Dict:
        """
        Optimize the profile settings for maximum manifestation efficiency.
        
        Returns:
            Dictionary of optimized parameters
        """
        # Calculate optimal manifestation settings
        # In a real system, this would use more sophisticated algorithms
        
        # Find strongest frequency resonances
        top_frequencies = sorted(
            self.frequency_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 frequencies
        
        # Create optimized settings
        optimized = {
            "primary_frequency": SACRED_FREQUENCIES[top_frequencies[0][0]],
            "secondary_frequencies": [
                SACRED_FREQUENCIES[name] for name, _ in top_frequencies[1:3]
            ],
            "phi_alignment": min(self.phi_alignment * 1.1, 0.95),  # Boost alignment slightly
            "coherence_threshold": self.coherence_threshold * 0.9,  # Lower threshold slightly
            "intensity_factor": min(1.0, self.phi_alignment / 0.618)  # PHI-scaled intensity
        }
        
        return optimized


class PhiCalibrator:
    """
    System for calibrating phi-harmonic resonance between consciousness and fields.
    
    The PhiCalibrator provides tools for measuring, adjusting, and optimizing the
    phi-alignment between consciousness states and quantum fields to achieve perfect
    resonance for manifestation.
    """
    
    def __init__(self):
        """Initialize the phi calibrator."""
        self.profiles = {}  # Dictionary of calibration profiles
        self.active_profile_id = None
        self.calibration_history = []
        
        # Calibration parameters
        self.calibration_steps = 7  # Fibonacci-based steps
        self.min_calibration_time = 3.0  # seconds per step
        self.max_calibration_time = 21.0  # seconds per step
    
    def create_profile(self, profile_id: str, user_id: Optional[str] = None) -> CalibrationProfile:
        """
        Create a new calibration profile.
        
        Args:
            profile_id: Unique identifier for the profile
            user_id: Optional user identifier
            
        Returns:
            The created calibration profile
        """
        # Create new profile
        profile = CalibrationProfile(
            profile_id=profile_id,
            user_id=user_id,
            phi_alignment=LAMBDA,  # Start with phi complement
            coherence_threshold=LAMBDA * 0.9,
            manifestation_efficiency=LAMBDA,
            extraction_efficiency=LAMBDA
        )
        
        # Store the profile
        self.profiles[profile_id] = profile
        
        # Set as active
        self.active_profile_id = profile_id
        
        return profile
    
    def load_profile(self, profile_id: str) -> Optional[CalibrationProfile]:
        """
        Load an existing calibration profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            The loaded profile or None if not found
        """
        if profile_id in self.profiles:
            self.active_profile_id = profile_id
            return self.profiles[profile_id]
        
        return None
    
    def get_active_profile(self) -> Optional[CalibrationProfile]:
        """Get the currently active calibration profile."""
        if self.active_profile_id and self.active_profile_id in self.profiles:
            return self.profiles[self.active_profile_id]
        
        return None
    
    def calibrate_with_feedback(self, field_coherence: float, thought_coherence: float,
                              frequency_activities: Dict[str, float]) -> Dict:
        """
        Update calibration based on consciousness-field interactions.
        
        Args:
            field_coherence: Measured field coherence
            thought_coherence: Measured thought coherence
            frequency_activities: Dictionary of frequency names to activation levels
            
        Returns:
            Dictionary of updated calibration parameters
        """
        profile = self.get_active_profile()
        if not profile:
            # Create a default profile if none active
            profile = self.create_profile("default")
        
        # Update the profile
        profile.update_from_session(
            field_coherence=field_coherence,
            thought_coherence=thought_coherence,
            active_frequencies=frequency_activities
        )
        
        # Record in history
        self.calibration_history.append({
            "timestamp": time.time(),
            "profile_id": profile.profile_id,
            "field_coherence": field_coherence,
            "thought_coherence": thought_coherence,
            "phi_alignment": profile.phi_alignment
        })
        
        # Return optimized parameters
        return profile.optimize_for_manifestation()
    
    def run_calibration_sequence(self, manifestation_matrix: ManifestationMatrix,
                               callback: Optional[Callable[[float, Dict], None]] = None) -> CalibrationProfile:
        """
        Run an automated calibration sequence to optimize phi-alignment.
        
        Args:
            manifestation_matrix: The manifestation matrix to calibrate
            callback: Optional callback function to report progress
            
        Returns:
            The calibrated profile
        """
        profile = self.get_active_profile()
        if not profile:
            # Create a default profile if none active
            profile = self.create_profile("calibration_" + str(int(time.time())))
        
        # Ensure the manifestation matrix is active
        if not manifestation_matrix.active:
            manifestation_matrix.activate()
        
        # Initialize calibration variables
        best_coherence = 0.0
        best_settings = {}
        
        # Calibration steps
        steps = []
        for i in range(self.calibration_steps):
            # Create phi-based calibration points
            phi_step = ((i / (self.calibration_steps - 1)) * (PHI - 1)) + LAMBDA
            steps.append({
                "phi_alignment": phi_step,
                "coherence_threshold": phi_step * 0.9,
                "frequency": SACRED_FREQUENCIES["unity"]  # Start with unity frequency
            })
        
        # Run the calibration sequence
        for i, step in enumerate(steps):
            # Apply calibration settings
            manifestation_matrix.phi_alignment = step["phi_alignment"]
            manifestation_matrix.coherence_threshold = step["coherence_threshold"]
            
            # Simulate a calibration measurement
            # In a real system, this would involve actual field measurements
            coherence = self._simulate_calibration_measurement(
                phi_alignment=step["phi_alignment"],
                frequency=step["frequency"]
            )
            
            # Check if this is the best result so far
            if coherence > best_coherence:
                best_coherence = coherence
                best_settings = step.copy()
            
            # Report progress if callback provided
            if callback:
                progress = (i + 1) / len(steps)
                callback(progress, {
                    "step": i + 1,
                    "total_steps": len(steps),
                    "phi_alignment": step["phi_alignment"],
                    "coherence": coherence,
                    "best_coherence": best_coherence
                })
        
        # Apply the best settings to the profile
        profile.phi_alignment = best_settings["phi_alignment"]
        profile.coherence_threshold = best_settings["coherence_threshold"]
        
        # Update frequency weights
        best_freq_name = next((name for name, freq in SACRED_FREQUENCIES.items()
                             if freq == best_settings["frequency"]), "unity")
        
        profile.frequency_weights[best_freq_name] = 1.0
        
        # Update profile timestamp
        profile.last_update_timestamp = time.time()
        
        return profile
    
    def optimize_for_frequency(self, frequency_name: str) -> Dict:
        """
        Optimize calibration for a specific sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency
            
        Returns:
            Dictionary of optimized parameters
        """
        profile = self.get_active_profile()
        if not profile:
            # Create a default profile if none active
            profile = self.create_profile("default")
        
        if frequency_name not in SACRED_FREQUENCIES:
            raise ValueError(f"Unknown sacred frequency: {frequency_name}")
        
        # Update frequency weights to prefer this frequency
        for name in profile.frequency_weights:
            if name == frequency_name:
                profile.frequency_weights[name] = 1.0
            else:
                # Reduce other frequencies with phi scaling
                profile.frequency_weights[name] *= LAMBDA
        
        # Adjust phi-alignment for this frequency
        if frequency_name == "unity":
            # Unity frequency works best with higher phi-alignment
            profile.phi_alignment = min(profile.phi_alignment * 1.1, 0.95)
        elif frequency_name == "love":
            # Love frequency (creation) works with golden ratio itself
            profile.phi_alignment = (profile.phi_alignment + PHI / 3) / 2
        else:
            # Other frequencies with moderate adjustment
            profile.phi_alignment = (profile.phi_alignment + 0.75) / 2
        
        # Update profile timestamp
        profile.last_update_timestamp = time.time()
        
        # Return optimized parameters
        return profile.optimize_for_manifestation()
    
    def get_optimal_manifestation_settings(self) -> Dict:
        """
        Get the optimal settings for manifestation based on current profile.
        
        Returns:
            Dictionary of optimal manifestation settings
        """
        profile = self.get_active_profile()
        if not profile:
            # Create a default profile if none active
            profile = self.create_profile("default")
        
        return profile.optimize_for_manifestation()
    
    def _simulate_calibration_measurement(self, phi_alignment: float, frequency: float) -> float:
        """
        Simulate a calibration measurement for testing.
        
        In a real system, this would be replaced with actual field measurements.
        
        Args:
            phi_alignment: The phi alignment setting
            frequency: The frequency setting
            
        Returns:
            Simulated coherence measurement
        """
        # This is a simplified model - a real system would have more complex behavior
        
        # Base coherence dependent on phi alignment
        # Best coherence at specific phi-related values
        base_coherence = 0.5
        
        # Add phi-resonant peaks
        phi_factor = 1.0 - min(abs(phi_alignment - LAMBDA) * 2, 1.0)
        phi2_factor = 1.0 - min(abs(phi_alignment - (1 - 1/PHI)) * 3, 1.0)
        phi3_factor = 1.0 - min(abs(phi_alignment - (PHI-1)) * 4, 1.0)
        
        # Combine with weights
        phi_combined = (phi_factor * 0.5 + phi2_factor * 0.3 + phi3_factor * 0.2)
        
        # Add frequency-dependent factors
        # 432Hz (unity) and 528Hz (love) have special alignment
        freq_factor = 0.0
        if abs(frequency - 432) < 1:
            freq_factor = 0.9
        elif abs(frequency - 528) < 1:
            freq_factor = 0.85
        else:
            # Other frequencies with moderate alignment
            freq_factor = 0.7
        
        # Add some randomness for realism
        noise = np.random.random() * 0.1
        
        # Combine all factors
        coherence = base_coherence * 0.3 + phi_combined * 0.4 + freq_factor * 0.3 + noise
        
        # Ensure valid range
        coherence = min(max(coherence, 0.0), 1.0)
        
        return coherence