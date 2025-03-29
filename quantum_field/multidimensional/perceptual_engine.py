"""
Perceptual Engine Module

Core system for integrating and managing multi-dimensional perception experiences,
providing a unified interface for sensory translation and phi-dimensional interaction.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
import threading
import time

# Import from the multidimensional framework
from .data_structures import DimensionalData, HyperField
from .translation import SensoryTranslator, ModalityMap
from .phi_dimensions import PhiDimensionalScaling

# Import sacred constants
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sacred_constants import (
    PHI, PHI_SQUARED, PHI_CUBED, PHI_PHI, 
    LAMBDA, SACRED_FREQUENCIES
)


class PerceptionEngine:
    """
    Core engine for multi-dimensional perception experiences.
    
    This engine integrates data structures, sensory translation, and
    phi-dimensional scaling to provide a complete framework for
    multi-dimensional perception of quantum fields.
    """
    
    def __init__(self, dimensions=(21, 21, 21), ndim=5):
        """
        Initialize the perception engine.
        
        Args:
            dimensions: Base dimensions for the field
            ndim: Number of dimensions to support (3-7)
        """
        self.base_dimensions = dimensions
        self.ndim = max(3, min(7, ndim))  # Clamp between 3 and 7
        
        # Initialize components
        self.phi_scaling = PhiDimensionalScaling(dimensions[0])
        self.hyperfield = HyperField(dimensions, ndim=self.ndim)
        self.translator = SensoryTranslator(dimensions)
        
        # Initialize engine state
        self.active = False
        self.coherence = 0.0
        self.resonance_profile = {}
        
        # Perception configuration
        self.perception_config = {
            "visual_mode": "color_map",
            "auditory_mode": "frequency_map",
            "tactile_mode": "intensity_map",
            "intuitive_mode": "intuitive_patterns",
            "emotional_mode": "emotional_map",
            "base_frequency": 432.0,  # Unity frequency
            "phi_alignment": 1.0,     # Full phi alignment
            "dimension_weights": self._create_dimension_weights(),
            "sensory_weights": {
                "visual": 1.0,
                "auditory": LAMBDA,
                "tactile": LAMBDA * LAMBDA,
                "emotional": PHI,
                "intuitive": PHI_SQUARED
            }
        }
        
        # Event callback system
        self.callbacks = {
            "perception_update": [],
            "dimension_shift": [],
            "coherence_change": [],
            "sensory_event": []
        }
        
        # Threading for continuous perception
        self.perception_thread = None
        self.thread_active = False
        self.update_interval = 0.1  # seconds
    
    def _create_dimension_weights(self) -> Dict[int, float]:
        """
        Create default dimension weights with phi-harmonic scaling.
        
        Returns:
            Dictionary of dimension weights
        """
        weights = {}
        
        for dim in range(3, self.ndim + 1):
            if dim == 3:  # Spatial (3D)
                weights[dim] = 1.0
            elif dim == 4:  # Time (4D)
                weights[dim] = LAMBDA
            elif dim == 5:  # Consciousness (5D)
                weights[dim] = 1.0
            elif dim == 6:  # Intention (6D)
                weights[dim] = PHI
            elif dim == 7:  # Unified field (7D)
                weights[dim] = PHI_SQUARED
        
        return weights
    
    def initialize(self, field_data: Optional[np.ndarray] = None) -> None:
        """
        Initialize the engine with field data.
        
        Args:
            field_data: Initial field data (creates coherent field if None)
        """
        if field_data is not None:
            # Validate dimensions
            if field_data.shape[:3] != self.base_dimensions:
                raise ValueError(f"Field data dimensions {field_data.shape[:3]} "
                               f"don't match engine dimensions {self.base_dimensions}")
            
            # Initialize HyperField with the data
            self.hyperfield.data[:field_data.shape[0], 
                              :field_data.shape[1], 
                              :field_data.shape[2]] = field_data
        else:
            # Initialize with coherent patterns
            self.hyperfield.initialize_coherent_field()
        
        # Calculate initial coherence
        self.coherence = self.hyperfield.calculate_coherence()
        
        # Analyze resonance
        self.resonance_profile = self.hyperfield.analyze_resonance()
        
        # Load field data into translator
        self.translator.load_field(self.hyperfield.project_to_3d())
        
        # Set engine to active
        self.active = True
        
        # Fire callbacks
        self._fire_callback("perception_update", {
            "coherence": self.coherence,
            "resonance": self.resonance_profile
        })
    
    def update(self) -> Dict:
        """
        Update the engine state and return current status.
        
        Returns:
            Dictionary with engine status
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Update coherence
        self.coherence = self.hyperfield.calculate_coherence()
        
        # Update translator with current 3D projection
        self.translator.load_field(self.hyperfield.project_to_3d())
        
        # Prepare status information
        status = {
            "active": self.active,
            "coherence": self.coherence,
            "ndim": self.ndim,
            "base_dimensions": self.base_dimensions,
            "dimensions": {dim: self.phi_scaling.get_dimension_size(dim) 
                         for dim in range(3, self.ndim+1)},
            "perception_config": self.perception_config,
            "dimensional_profile": self.hyperfield.get_dimensional_profile()
        }
        
        # Fire update callback
        self._fire_callback("perception_update", status)
        
        return status
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback function for a specific event.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def _fire_callback(self, event: str, data: Dict) -> None:
        """
        Fire callbacks for a specific event.
        
        Args:
            event: Event name
            data: Event data
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(event, data)
                except Exception as e:
                    print(f"Error in {event} callback: {e}")
    
    def start_continuous_perception(self) -> None:
        """Start continuous perception updates in a separate thread."""
        if self.perception_thread is not None and self.thread_active:
            return  # Already running
        
        self.thread_active = True
        self.perception_thread = threading.Thread(
            target=self._continuous_perception_loop,
            daemon=True
        )
        self.perception_thread.start()
    
    def stop_continuous_perception(self) -> None:
        """Stop continuous perception updates."""
        self.thread_active = False
        if self.perception_thread is not None:
            self.perception_thread.join(timeout=1.0)
            self.perception_thread = None
    
    def _continuous_perception_loop(self) -> None:
        """Internal method for continuous perception updates."""
        while self.thread_active and self.active:
            self.update()
            time.sleep(self.update_interval)
    
    def get_dimension_projection(self, dimension: int) -> np.ndarray:
        """
        Get a 3D projection from a specific dimension.
        
        Args:
            dimension: Dimension to project (3-7)
            
        Returns:
            3D projection of the dimension
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        if dimension < 3 or dimension > self.ndim:
            raise ValueError(f"Dimension {dimension} out of range (3-{self.ndim})")
        
        # For 3D, just return the spatial data
        if dimension == 3:
            return self.hyperfield.project_to_3d()
        
        # For higher dimensions, get projection from multidimensional perception
        projections = self.translator.get_multidimensional_projections()
        
        # Map dimension to projection name
        dim_names = {
            4: "4d_time",
            5: "5d_consciousness",
            6: "6d_intention",
            7: "7d_unified"
        }
        
        dim_name = dim_names.get(dimension)
        if dim_name and dim_name in projections:
            return projections[dim_name]
        
        # Fallback to 3D projection
        return self.hyperfield.project_to_3d()
    
    def shift_to_dimension(self, 
                         target_dimension: int, 
                         intensity: float = 0.5) -> Dict:
        """
        Shift perception emphasis to a specific dimension.
        
        Args:
            target_dimension: Dimension to shift to (3-7)
            intensity: Intensity of the shift (0-1)
            
        Returns:
            Status dictionary
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        if target_dimension < 3 or target_dimension > self.ndim:
            raise ValueError(f"Dimension {target_dimension} out of range (3-{self.ndim})")
        
        # Update dimension weights
        old_weights = self.perception_config["dimension_weights"].copy()
        
        # Calculate phi-scaled intensity
        phi_intensity = intensity * PHI / (PHI + 1)
        
        # Adjust weights to emphasize target dimension
        for dim in self.perception_config["dimension_weights"]:
            if dim == target_dimension:
                # Increase weight for target dimension
                self.perception_config["dimension_weights"][dim] *= (1.0 + phi_intensity)
            else:
                # Decrease weight for other dimensions
                self.perception_config["dimension_weights"][dim] *= (1.0 - phi_intensity * LAMBDA)
        
        # Normalize weights to maintain overall balance
        weight_sum = sum(self.perception_config["dimension_weights"].values())
        if weight_sum > 0:
            for dim in self.perception_config["dimension_weights"]:
                self.perception_config["dimension_weights"][dim] /= weight_sum
        
        # Perform actual dimension shift in the hyperfield
        if target_dimension > 3:
            # Perform higher-dimensional shift
            for dim in range(3, self.ndim+1):
                if dim != target_dimension:
                    self.hyperfield.shift_dimension(dim, target_dimension, intensity * LAMBDA)
        
        # Get dimension status
        dimensions = {
            dim: {
                "weight": self.perception_config["dimension_weights"][dim],
                "size": self.phi_scaling.get_dimension_size(dim),
                "name": self.phi_scaling.dimension_name(dim)
            }
            for dim in self.perception_config["dimension_weights"]
        }
        
        # Update and get status
        status = self.update()
        
        # Add dimension shift information
        shift_info = {
            "target_dimension": target_dimension,
            "intensity": intensity,
            "dimensions": dimensions,
            "old_weights": old_weights,
            "new_weights": self.perception_config["dimension_weights"].copy()
        }
        
        # Fire callback
        self._fire_callback("dimension_shift", shift_info)
        
        return {**status, **shift_info}
    
    def calibrate_phi_alignment(self, target_coherence: float = 0.618) -> Dict:
        """
        Calibrate phi alignment for optimal coherence.
        
        Args:
            target_coherence: Target coherence level (default: phi complement)
            
        Returns:
            Calibration results
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Record initial coherence
        initial_coherence = self.coherence
        
        # Apply phi mask to enhance coherence
        if self.coherence < target_coherence:
            scale = 1.0 - (self.coherence / target_coherence)
            self.hyperfield.apply_phi_mask(scale=scale)
        
        # Update coherence after calibration
        self.coherence = self.hyperfield.calculate_coherence()
        
        # Update translator
        self.translator.load_field(self.hyperfield.project_to_3d())
        
        # Update phi alignment in perception config
        old_alignment = self.perception_config["phi_alignment"]
        new_alignment = self.coherence / target_coherence
        self.perception_config["phi_alignment"] = min(1.0, new_alignment)
        
        # Prepare results
        results = {
            "initial_coherence": initial_coherence,
            "final_coherence": self.coherence,
            "target_coherence": target_coherence,
            "old_alignment": old_alignment,
            "new_alignment": self.perception_config["phi_alignment"],
            "improvement": self.coherence - initial_coherence
        }
        
        # Fire callback
        self._fire_callback("coherence_change", results)
        
        return results
    
    def shift_to_frequency(self, frequency_name: str) -> Dict:
        """
        Shift the perceptual frequency to a sacred frequency.
        
        Args:
            frequency_name: Name of the frequency to shift to
            
        Returns:
            Status dictionary
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Validate frequency name
        if frequency_name not in SACRED_FREQUENCIES and frequency_name not in self.translator.modality_map.frequency_map:
            raise ValueError(f"Unknown frequency: {frequency_name}")
        
        # Get frequency value
        if frequency_name in SACRED_FREQUENCIES:
            frequency = SACRED_FREQUENCIES[frequency_name]
        else:
            frequency = self.translator.modality_map.frequency_map[frequency_name]
        
        # Update base frequency
        old_frequency = self.perception_config["base_frequency"]
        self.perception_config["base_frequency"] = frequency
        
        # Create frequency-specific pattern
        # Initialize temporary hyperfield with this frequency
        temp_field = HyperField(self.base_dimensions, ndim=self.ndim)
        temp_field.initialize_coherent_field(base_frequency=frequency_name)
        
        # Blend with current field using phi-weighted average
        blend_factor = LAMBDA  # Use phi complement for natural transition
        self.hyperfield.data = (
            self.hyperfield.data * (1.0 - blend_factor) +
            temp_field.data * blend_factor
        )
        
        # Update coherence and resonance
        self.coherence = self.hyperfield.calculate_coherence()
        self.resonance_profile = self.hyperfield.analyze_resonance()
        
        # Update translator
        self.translator.load_field(self.hyperfield.project_to_3d())
        
        # Prepare status information
        status = {
            "frequency_name": frequency_name,
            "frequency_value": frequency,
            "old_frequency": old_frequency,
            "coherence": self.coherence,
            "resonance": self.resonance_profile
        }
        
        # Fire callback
        self._fire_callback("sensory_event", {
            "type": "frequency_shift",
            "details": status
        })
        
        return status
    
    def get_sensory_experience(self, modality: str = "all") -> Dict:
        """
        Get sensory experience data for a specific modality or all modalities.
        
        Args:
            modality: Sensory modality or "all" for all modalities
            
        Returns:
            Dictionary with sensory experience data
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Get all modalities
        if modality == "all":
            return self.translator.create_synchronized_experience()
        
        # Get specific modality
        valid_modalities = {"visual", "auditory", "tactile", "emotional", "intuitive"}
        if modality not in valid_modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Get data for the requested modality
        if modality == "visual":
            mode = self.perception_config["visual_mode"]
            return self.translator.get_visual_representation(mode=mode)
        
        elif modality == "auditory":
            mode = self.perception_config["auditory_mode"]
            base_freq = self.perception_config["base_frequency"]
            return self.translator.get_auditory_representation(mode=mode, base_freq=base_freq)
        
        elif modality == "tactile":
            mode = self.perception_config["tactile_mode"]
            return self.translator.get_tactile_representation(mode=mode)
        
        elif modality == "emotional":
            return self.translator.get_emotional_representation()
        
        elif modality == "intuitive":
            mode = self.perception_config["intuitive_mode"]
            return self.translator.get_intuitive_representation(mode=mode)
    
    def amplify_sensory_modality(self, 
                               modality: str, 
                               amplification: float = 0.5) -> Dict:
        """
        Amplify a specific sensory modality in the perception configuration.
        
        Args:
            modality: Sensory modality to amplify
            amplification: Amplification factor (0-1)
            
        Returns:
            Status dictionary
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Validate modality
        valid_modalities = {"visual", "auditory", "tactile", "emotional", "intuitive"}
        if modality not in valid_modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Store original weights
        old_weights = self.perception_config["sensory_weights"].copy()
        
        # Calculate phi-scaled amplification
        phi_amp = amplification * PHI / (PHI + 1)
        
        # Update weights
        for mod in self.perception_config["sensory_weights"]:
            if mod == modality:
                self.perception_config["sensory_weights"][mod] *= (1.0 + phi_amp)
            else:
                self.perception_config["sensory_weights"][mod] *= (1.0 - phi_amp * LAMBDA)
        
        # Normalize weights
        weight_sum = sum(self.perception_config["sensory_weights"].values())
        if weight_sum > 0:
            for mod in self.perception_config["sensory_weights"]:
                self.perception_config["sensory_weights"][mod] /= weight_sum
        
        # Prepare status
        status = {
            "modality": modality,
            "amplification": amplification,
            "old_weights": old_weights,
            "new_weights": self.perception_config["sensory_weights"].copy()
        }
        
        # Fire callback
        self._fire_callback("sensory_event", {
            "type": "modality_amplification",
            "details": status
        })
        
        return status
    
    def create_cross_modal_mapping(self, 
                                 source_modality: str,
                                 target_modality: str) -> Dict:
        """
        Create a mapping between two sensory modalities.
        
        Args:
            source_modality: Source sensory modality
            target_modality: Target sensory modality
            
        Returns:
            Mapping information
        """
        if not self.active:
            raise ValueError("Engine not initialized")
        
        # Validate modalities
        valid_modalities = {"visual", "auditory", "tactile", "emotional", "intuitive"}
        if source_modality not in valid_modalities:
            raise ValueError(f"Unknown source modality: {source_modality}")
        
        if target_modality not in valid_modalities:
            raise ValueError(f"Unknown target modality: {target_modality}")
        
        # Get source representation
        if source_modality == "visual":
            mode = self.perception_config["visual_mode"]
            source_data = self.translator.get_visual_representation(mode=mode)
        elif source_modality == "auditory":
            mode = self.perception_config["auditory_mode"]
            source_data = self.translator.get_auditory_representation(mode=mode)
        elif source_modality == "tactile":
            mode = self.perception_config["tactile_mode"]
            source_data = self.translator.get_tactile_representation(mode=mode)
        elif source_modality == "emotional":
            # Emotional data is a dictionary, create a field representation
            emotional_data = self.translator.get_emotional_representation()
            source_data = self.translator.modality_map.apply_transformation(
                self.hyperfield.project_to_3d(), "emotional", "emotional_coloration"
            )
        elif source_modality == "intuitive":
            # Intuitive data is a dictionary, create a field representation
            intuitive_data = self.translator.get_intuitive_representation()
            
            # Use resonance pattern as source
            if "resonance" in intuitive_data:
                source_data = intuitive_data["resonance"]
            else:
                # Default to 3D projection
                source_data = self.hyperfield.project_to_3d()
        
        # Create target representation through translation
        target_data = self.translator.translate_between_modalities(
            source_modality, target_modality, source_data
        )
        
        # Prepare mapping information
        mapping = {
            "source_modality": source_modality,
            "target_modality": target_modality,
            "phi_relationship": f"{source_modality} → {target_modality}",
            "coherence": self.coherence,
            "source_shape": source_data.shape if hasattr(source_data, "shape") else None,
            "target_shape": target_data.shape if hasattr(target_data, "shape") else None,
        }
        
        # Store the mapping in the engine
        # (Note: The actual data could be very large, so we only store metadata)
        
        # Fire callback
        self._fire_callback("sensory_event", {
            "type": "cross_modal_mapping",
            "details": mapping
        })
        
        return mapping
    
    @classmethod
    def create_optimized(cls, dimensions=(21, 21, 21), ndim=5) -> 'PerceptionEngine':
        """
        Create an optimized perception engine instance.
        
        Args:
            dimensions: Base dimensions
            ndim: Number of dimensions
            
        Returns:
            Optimized PerceptionEngine instance
        """
        # Create engine
        engine = cls(dimensions, ndim)
        
        # Initialize with coherent patterns
        engine.initialize()
        
        # Calibrate for optimal phi alignment
        engine.calibrate_phi_alignment()
        
        # Shift to unity frequency (432 Hz)
        engine.shift_to_frequency("unity")
        
        return engine