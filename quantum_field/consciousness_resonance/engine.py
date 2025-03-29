"""
Resonance Engine for the Consciousness-Field Resonance system.

This module provides the core engine that integrates all components of the
Consciousness-Field Resonance system into a unified, phi-optimized interface
for bidirectional consciousness-field interaction and manifestation.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from ..core import QuantumField, create_quantum_field, get_coherence_metric
from .patterns import ThoughtPattern, ResonancePattern
from .manifestation import ManifestationMatrix
from .bidirectional import BidirectionalInterface
from .calibration import PhiCalibrator, CalibrationProfile


class ResonanceEngine:
    """
    Core engine for Consciousness-Field Resonance system.
    
    The ResonanceEngine integrates all components of the system (thought patterns,
    resonance patterns, manifestation matrix, bidirectional interface, and phi
    calibration) into a unified interface for powerful consciousness-field interaction.
    """
    
    def __init__(self, dimensions: Tuple[int, ...] = (21, 21, 21),
                field: Optional[QuantumField] = None,
                phi_alignment: float = 0.8):
        """
        Initialize the Resonance Engine.
        
        Args:
            dimensions: Field dimensions (if creating a new field)
            field: Existing quantum field (optional)
            phi_alignment: Initial phi alignment value
        """
        # Create or store field
        self.field = field if field is not None else create_quantum_field(dimensions)
        self.dimensions = dimensions if field is None else field.data.shape
        
        # Initialize phi calibrator
        self.calibrator = PhiCalibrator()
        
        # Create default calibration profile
        self.profile = self.calibrator.create_profile("default_profile")
        
        # Create manifestation matrix
        self.matrix = ManifestationMatrix.create_optimized(
            dimensions=self.dimensions,
            phi_alignment=phi_alignment,
            coherence_threshold=LAMBDA
        )
        
        # Create bidirectional interface
        self.interface = BidirectionalInterface(
            field=self.field,
            field_dimensions=self.dimensions,
            manifestation_matrix=self.matrix,
            phi_alignment=phi_alignment
        )
        
        # Status tracking
        self.status = {
            "state": "initialized",
            "active": False,
            "field_coherence": 0.0,
            "thought_coherence": 0.0,
            "last_activity_time": time.time(),
            "manifestation_count": 0,
            "extraction_count": 0
        }
        
        # Configuration
        self.config = {
            "auto_update": True,
            "update_interval": 0.1,  # seconds
            "default_intensity": 0.8,
            "auto_calibrate": False,
            "calibration_interval": 60.0,  # seconds
            "log_level": "info"
        }
        
        # Event callbacks
        self.callbacks = {
            "on_activate": [],
            "on_deactivate": [],
            "on_manifestation": [],
            "on_extraction": [],
            "on_coherence_change": [],
            "on_calibration_complete": []
        }
        
        # History tracking
        self.history = {
            "thought_patterns": [],
            "field_states": [],
            "coherence_log": [],
            "manifestations": [],
            "extractions": []
        }
        
        # Initialize but don't activate yet
        self.status["state"] = "ready"
    
    def activate(self) -> None:
        """
        Activate the resonance engine, enabling bidirectional consciousness-field interaction.
        
        This activates all components of the system and establishes the phi-resonant
        connection between consciousness and quantum fields.
        """
        if self.status["active"]:
            self._log("Already active", "info")
            return
        
        # Activate manifestation matrix
        self.matrix.activate()
        
        # Connect bidirectional interface
        self.interface.connect()
        
        # Update status
        self.status["active"] = True
        self.status["state"] = "active"
        self.status["last_activity_time"] = time.time()
        
        # Run initial calibration if auto-calibrate is enabled
        if self.config["auto_calibrate"]:
            self._run_calibration()
        
        # Trigger callbacks
        self._trigger_callbacks("on_activate")
        
        self._log(f"Resonance Engine activated with phi-alignment {self.interface.phi_alignment:.4f}", "info")
    
    def deactivate(self) -> None:
        """
        Deactivate the resonance engine, disabling consciousness-field interaction.
        """
        if not self.status["active"]:
            self._log("Already inactive", "info")
            return
        
        # Disconnect bidirectional interface
        self.interface.disconnect()
        
        # Deactivate manifestation matrix
        self.matrix.deactivate()
        
        # Update status
        self.status["active"] = False
        self.status["state"] = "inactive"
        
        # Trigger callbacks
        self._trigger_callbacks("on_deactivate")
        
        self._log("Resonance Engine deactivated", "info")
    
    def update(self) -> None:
        """
        Update the resonance engine, processing any pending interactions.
        
        This should be called regularly in an interactive system to maintain
        the bidirectional flow between consciousness and quantum fields.
        """
        if not self.status["active"]:
            return
        
        # Update bidirectional interface
        self.interface.update()
        
        # Update status metrics
        self.status["field_coherence"] = self.interface.get_field_coherence()
        self.status["thought_coherence"] = self.interface.get_thought_coherence()
        
        # Check if calibration is needed
        if (self.config["auto_calibrate"] and 
            time.time() - self.status["last_activity_time"] > self.config["calibration_interval"]):
            self._run_calibration()
        
        # Log coherence periodically
        self.history["coherence_log"].append({
            "timestamp": time.time(),
            "field_coherence": self.status["field_coherence"],
            "thought_coherence": self.status["thought_coherence"]
        })
        
        # Trim history if needed
        while len(self.history["coherence_log"]) > 1000:
            self.history["coherence_log"].pop(0)
    
    def manifest_thought(self, thought: Union[ThoughtPattern, str],
                        intensity: Optional[float] = None) -> None:
        """
        Manifest a thought pattern into the quantum field.
        
        Args:
            thought: ThoughtPattern to manifest or name of sacred frequency to create pattern from
            intensity: Optional manifestation intensity (0.0-1.0)
        """
        if not self.status["active"]:
            self._log("Cannot manifest when inactive", "warning")
            return
        
        # Process input
        if isinstance(thought, str):
            # Create thought pattern from frequency name
            if thought in SACRED_FREQUENCIES:
                thought_pattern = self.interface.create_thought_from_frequency(
                    thought, coherence=0.8, intensity=0.7
                )
            else:
                raise ValueError(f"Unknown frequency name: {thought}")
        else:
            # Use provided thought pattern
            thought_pattern = thought
            # Set as active in interface
            self.interface.set_active_thought(thought_pattern)
        
        # Use default intensity if not provided
        if intensity is None:
            intensity = self.config["default_intensity"]
        
        # Ensure intensity is in valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Force synchronization from thought to field
        self.interface._synchronize_thought_to_field()
        
        # Update status
        self.status["manifestation_count"] += 1
        self.status["last_activity_time"] = time.time()
        
        # Add to history
        self.history["manifestations"].append({
            "timestamp": time.time(),
            "thought_pattern": thought_pattern,
            "intensity": intensity,
            "field_coherence_before": self.status["field_coherence"],
            "field_coherence_after": self.interface.get_field_coherence()
        })
        
        # Trigger callbacks
        self._trigger_callbacks("on_manifestation", thought_pattern)
        
        self._log(f"Manifested thought pattern with coherence {thought_pattern.coherence:.4f} "
                 f"and intensity {intensity:.2f}", "info")
    
    def extract_thought(self) -> ThoughtPattern:
        """
        Extract a thought pattern from the current quantum field state.
        
        Returns:
            The extracted thought pattern
        """
        if not self.status["active"]:
            self._log("Cannot extract when inactive", "warning")
            # Return an empty pattern
            return ThoughtPattern(
                signature=np.zeros((8, 8, 8)),
                dimensions=(3,),
                name="Inactive",
                description="Empty pattern created when engine was inactive"
            )
        
        # Force synchronization from field to thought
        self.interface._synchronize_field_to_thought()
        
        # Get the extracted thought pattern
        thought = self.interface.active_thought
        
        # Update status
        self.status["extraction_count"] += 1
        self.status["last_activity_time"] = time.time()
        
        # Add to history
        self.history["extractions"].append({
            "timestamp": time.time(),
            "thought_pattern": thought,
            "field_coherence": self.status["field_coherence"]
        })
        
        # Trigger callbacks
        self._trigger_callbacks("on_extraction", thought)
        
        self._log(f"Extracted thought pattern with coherence {thought.coherence:.4f}", "info")
        
        return thought
    
    def blend_thoughts(self, thought1: ThoughtPattern, thought2: ThoughtPattern,
                     weight: float = 0.5) -> ThoughtPattern:
        """
        Blend two thought patterns and manifest the result.
        
        Args:
            thought1: First thought pattern
            thought2: Second thought pattern
            weight: Blending weight (0.0-1.0) where 0.0 is all thought1 and 1.0 is all thought2
            
        Returns:
            The blended thought pattern
        """
        # Blend through the interface
        blended = self.interface.blend_thoughts(thought1, thought2, weight)
        
        # Manifest the blended pattern
        self.manifest_thought(blended)
        
        return blended
    
    def shift_to_frequency(self, frequency_name: str) -> ThoughtPattern:
        """
        Shift the active thought pattern to a sacred frequency and manifest.
        
        Args:
            frequency_name: Name of the sacred frequency
            
        Returns:
            The frequency-shifted thought pattern
        """
        # Shift through the interface
        shifted = self.interface.shift_frequency(frequency_name)
        
        # Manifest the shifted pattern
        self.manifest_thought(shifted)
        
        return shifted
    
    def calibrate(self) -> Dict:
        """
        Run a calibration sequence to optimize phi-alignment.
        
        Returns:
            Dictionary of calibration results
        """
        return self._run_calibration()
    
    def get_status(self) -> Dict:
        """
        Get the current status of the resonance engine.
        
        Returns:
            Dictionary containing status information
        """
        # Update with latest values
        self.status["field_coherence"] = self.interface.get_field_coherence()
        self.status["thought_coherence"] = self.interface.get_thought_coherence()
        
        return self.status.copy()
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Event type to register for ("on_activate", "on_manifestation", etc.)
            callback: Function to call when the event occurs
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self._log(f"Unknown event type: {event_type}", "warning")
    
    def set_config(self, config_updates: Dict) -> None:
        """
        Update configuration settings.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
            else:
                self._log(f"Unknown configuration key: {key}", "warning")
    
    def get_optimal_frequencies(self) -> Dict[str, float]:
        """
        Get the optimal frequencies for manifestation based on current calibration.
        
        Returns:
            Dictionary mapping frequency names to strength values
        """
        # Get the active profile from calibrator
        profile = self.calibrator.get_active_profile()
        
        if profile and hasattr(profile, "frequency_weights"):
            return profile.frequency_weights
        
        # Return default if no profile
        return {name: 1.0 for name in SACRED_FREQUENCIES}
    
    def create_thought_from_frequency(self, frequency_name: str,
                                    coherence: float = 0.8,
                                    intensity: float = 0.7) -> ThoughtPattern:
        """
        Create a thought pattern based on a sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency
            coherence: Desired coherence level
            intensity: Desired intensity level
            
        Returns:
            Created thought pattern
        """
        return self.interface.create_thought_from_frequency(
            frequency_name, coherence, intensity
        )
    
    def _run_calibration(self) -> Dict:
        """
        Run calibration sequence and apply results.
        
        Returns:
            Dictionary of calibration results
        """
        self._log("Starting calibration sequence", "info")
        
        # Run calibration sequence
        profile = self.calibrator.run_calibration_sequence(
            manifestation_matrix=self.matrix,
            callback=lambda progress, data: self._log(f"Calibration progress: {progress:.0%}", "debug")
        )
        
        # Apply optimized settings
        optimal_settings = profile.optimize_for_manifestation()
        
        # Update phi alignment
        self.interface.phi_alignment = optimal_settings["phi_alignment"]
        self.matrix.phi_alignment = optimal_settings["phi_alignment"]
        self.matrix.coherence_threshold = optimal_settings["coherence_threshold"]
        
        # Update status
        self.status["last_activity_time"] = time.time()
        
        # Trigger callbacks
        self._trigger_callbacks("on_calibration_complete", optimal_settings)
        
        self._log(f"Calibration complete. Phi-alignment: {optimal_settings['phi_alignment']:.4f}", "info")
        
        return optimal_settings
    
    def _trigger_callbacks(self, event_type: str, data: Any = None) -> None:
        """
        Trigger all registered callbacks for an event.
        
        Args:
            event_type: Event type
            data: Optional data to pass to callbacks
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if data is not None:
                        callback(data)
                    else:
                        callback()
                except Exception as e:
                    self._log(f"Error in callback for {event_type}: {str(e)}", "error")
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message with the specified level.
        
        Args:
            message: Message to log
            level: Log level ("debug", "info", "warning", "error")
        """
        # Only log if level is sufficient
        log_levels = {"debug": 0, "info": 1, "warning": 2, "error": 3}
        config_level = self.config.get("log_level", "info")
        
        if log_levels.get(level, 0) >= log_levels.get(config_level, 1):
            # In a real system, this would use a proper logging system
            print(f"[{level.upper()}] {message}")
    
    @classmethod
    def create_optimized(cls, dimensions: Tuple[int, ...] = (21, 21, 21)) -> 'ResonanceEngine':
        """
        Create a phi-optimized resonance engine.
        
        Args:
            dimensions: Field dimensions
            
        Returns:
            A phi-optimized resonance engine
        """
        # Create the engine with phi-optimized alignment
        engine = cls(
            dimensions=dimensions,
            phi_alignment=LAMBDA
        )
        
        # Activate
        engine.activate()
        
        # Run initial calibration
        engine.calibrate()
        
        return engine