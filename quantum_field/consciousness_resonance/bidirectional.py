"""
Bidirectional Interface for the Consciousness-Field Resonance Engine.

This module provides the core interface that enables a perfect two-way flow
between consciousness states and quantum fields, creating a phi-harmonic 
alignment that enables direct manifestation.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

from ..constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from ..core import QuantumField, create_quantum_field, get_coherence_metric
from .patterns import ThoughtPattern, ResonancePattern
from .manifestation import ManifestationMatrix


@dataclass
class BidirectionalInterface:
    """
    Bidirectional Interface between consciousness and quantum fields.
    
    The BidirectionalInterface creates a two-way bridge between consciousness states
    and quantum fields, enabling intuitive, phi-resonant manifestation of thought
    into field states and extraction of field patterns into consciousness.
    """
    # Field component
    field: Optional[QuantumField] = None
    field_dimensions: Tuple[int, ...] = (21, 21, 21)
    
    # Manifestation system
    manifestation_matrix: Optional[ManifestationMatrix] = None
    
    # Thought patterns
    active_thought: Optional[ThoughtPattern] = None
    thought_history: List[ThoughtPattern] = field(default_factory=list)
    
    # Operational settings
    auto_sync: bool = True
    sync_interval: float = 0.1  # Seconds
    phi_alignment: float = 0.8
    
    # Performance metrics
    field_to_thought_latency: float = 0.0
    thought_to_field_latency: float = 0.0
    
    # State tracking
    last_sync_time: float = field(default_factory=time.time)
    is_connected: bool = False
    
    # Callback functions
    on_field_change: Optional[Callable[[np.ndarray], None]] = None
    on_thought_change: Optional[Callable[[ThoughtPattern], None]] = None
    
    def __post_init__(self):
        """Initialize the bidirectional interface."""
        # Create field if not provided
        if self.field is None:
            self.field = create_quantum_field(self.field_dimensions)
        
        # Create manifestation matrix if not provided
        if self.manifestation_matrix is None:
            self.manifestation_matrix = ManifestationMatrix.create_optimized(
                dimensions=self.field_dimensions,
                phi_alignment=self.phi_alignment,
                coherence_threshold=LAMBDA
            )
        
        # Create initial thought pattern
        if self.active_thought is None:
            self.active_thought = ThoughtPattern.from_sacred_frequency(
                "unity",  # Start with unity frequency
                dimensions=(3,),
                coherence=0.7,
                intensity=0.5
            )
            self.thought_history.append(self.active_thought)
    
    def connect(self) -> None:
        """
        Establish the bidirectional connection between consciousness and field.
        
        This activates the phi-resonant bridge allowing thought patterns to manifest
        in the quantum field and field patterns to be extracted into consciousness.
        """
        if self.is_connected:
            print("Already connected")
            return
        
        # Activate the manifestation matrix
        self.manifestation_matrix.activate()
        
        # Perform initial synchronization
        self._synchronize_field_to_thought()
        
        # Mark as connected
        self.is_connected = True
        self.last_sync_time = time.time()
        
        print(f"Bidirectional interface connected with phi-alignment {self.phi_alignment:.4f}")
    
    def disconnect(self) -> None:
        """Disconnect the bidirectional interface."""
        if not self.is_connected:
            print("Already disconnected")
            return
        
        # Deactivate the manifestation matrix
        self.manifestation_matrix.deactivate()
        
        # Mark as disconnected
        self.is_connected = False
        
        print("Bidirectional interface disconnected")
    
    def update(self) -> None:
        """
        Update the bidirectional interface, synchronizing field and thought if needed.
        
        This should be called regularly in an interactive system to maintain
        bidirectional flow between consciousness and field.
        """
        if not self.is_connected:
            return
        
        # Check if it's time to auto-sync
        current_time = time.time()
        elapsed = current_time - self.last_sync_time
        
        if self.auto_sync and elapsed >= self.sync_interval:
            # Synchronize both ways
            self._synchronize_thought_to_field()
            self._synchronize_field_to_thought()
            
            # Update last sync time
            self.last_sync_time = current_time
    
    def set_active_thought(self, thought: ThoughtPattern) -> None:
        """
        Set the active thought pattern and manifest it to the field.
        
        Args:
            thought: The thought pattern to set as active
        """
        # Store the new thought pattern
        self.active_thought = thought
        self.thought_history.append(thought)
        
        # Manifest to field if connected
        if self.is_connected:
            self._synchronize_thought_to_field()
        
        # Notify callback if provided
        if self.on_thought_change is not None:
            self.on_thought_change(thought)
    
    def update_field(self, field_data: np.ndarray) -> None:
        """
        Update the quantum field data and extract the updated thought pattern.
        
        Args:
            field_data: The new field data
        """
        # Update the field
        self.field.data = field_data.copy()
        
        # Extract thought if connected
        if self.is_connected:
            self._synchronize_field_to_thought()
        
        # Notify callback if provided
        if self.on_field_change is not None:
            self.on_field_change(field_data)
    
    def create_thought_from_frequency(self, frequency_name: str,
                                     coherence: float = 0.8, 
                                     intensity: float = 0.7) -> ThoughtPattern:
        """
        Create a thought pattern from a sacred frequency and set it as active.
        
        Args:
            frequency_name: Name of the sacred frequency
            coherence: Thought pattern coherence
            intensity: Thought pattern intensity
            
        Returns:
            The created thought pattern
        """
        # Create the thought pattern
        thought = ThoughtPattern.from_sacred_frequency(
            frequency_name,
            dimensions=(3,),
            coherence=coherence,
            intensity=intensity
        )
        
        # Set as active
        self.set_active_thought(thought)
        
        return thought
    
    def blend_thoughts(self, thought1: ThoughtPattern, thought2: ThoughtPattern,
                     weight: float = 0.5) -> ThoughtPattern:
        """
        Blend two thought patterns and set the result as active.
        
        Args:
            thought1: First thought pattern
            thought2: Second thought pattern
            weight: Blending weight (0.0-1.0) where 0.0 is all thought1 and 1.0 is all thought2
            
        Returns:
            The blended thought pattern
        """
        # Blend the thought patterns
        blended = thought1.blend(thought2, weight)
        
        # Set as active
        self.set_active_thought(blended)
        
        return blended
    
    def shift_frequency(self, target_frequency: Union[float, str]) -> ThoughtPattern:
        """
        Shift the active thought pattern to a new frequency.
        
        Args:
            target_frequency: Target frequency as float (Hz) or sacred frequency name
            
        Returns:
            The frequency-shifted thought pattern
        """
        if not self.active_thought:
            raise ValueError("No active thought pattern to shift")
        
        # Resolve frequency if string
        if isinstance(target_frequency, str):
            if target_frequency in SACRED_FREQUENCIES:
                freq = SACRED_FREQUENCIES[target_frequency]
            else:
                raise ValueError(f"Unknown sacred frequency: {target_frequency}")
        else:
            freq = target_frequency
        
        # Shift frequency
        shifted = self.active_thought.shift_frequency(freq)
        
        # Set as active
        self.set_active_thought(shifted)
        
        return shifted
    
    def get_field_coherence(self) -> float:
        """Get the current field coherence level."""
        if self.field is None:
            return 0.0
        
        return self.manifestation_matrix._calculate_field_coherence(self.field.data)
    
    def get_thought_coherence(self) -> float:
        """Get the current thought pattern coherence level."""
        if self.active_thought is None:
            return 0.0
        
        return self.active_thought.coherence
    
    def _synchronize_thought_to_field(self) -> None:
        """
        Synchronize from thought to field by manifesting the active thought pattern.
        
        Updates the field based on the current active thought pattern.
        """
        if not self.is_connected or self.active_thought is None or self.field is None:
            return
        
        # Start timing for latency measurement
        start_time = time.time()
        
        # Manifest the thought pattern to the field
        self.field.data = self.manifestation_matrix.manifest_thought_to_field(
            self.active_thought,
            self.field.data
        )
        
        # Calculate latency
        self.thought_to_field_latency = time.time() - start_time
    
    def _synchronize_field_to_thought(self) -> None:
        """
        Synchronize from field to thought by extracting a thought pattern.
        
        Updates the active thought pattern based on the current field state.
        """
        if not self.is_connected or self.field is None:
            return
        
        # Start timing for latency measurement
        start_time = time.time()
        
        # Extract thought pattern from field
        extracted = self.manifestation_matrix.extract_thought_from_field(self.field.data)
        
        # Set as active if sufficiently coherent
        # For field-to-thought, we use auto-update with more discretion
        # Only update if the extracted pattern is significantly more coherent
        if (self.active_thought is None or
            extracted.coherence > self.active_thought.coherence * 1.2):
            self.active_thought = extracted
            self.thought_history.append(extracted)
            
            # Notify callback if provided
            if self.on_thought_change is not None:
                self.on_thought_change(extracted)
        
        # Calculate latency
        self.field_to_thought_latency = time.time() - start_time
    
    @classmethod
    def create_phi_optimized(cls, dimensions: Tuple[int, ...] = (21, 21, 21),
                           field: Optional[QuantumField] = None) -> 'BidirectionalInterface':
        """
        Create a phi-optimized bidirectional interface.
        
        Args:
            dimensions: Field dimensions (if creating a new field)
            field: Existing quantum field (optional)
            
        Returns:
            A phi-optimized bidirectional interface
        """
        # Create the interface
        interface = cls(
            field=field,
            field_dimensions=dimensions,
            phi_alignment=PHI_PHI / 3.0,  # Optimized alignment
            auto_sync=True,
            sync_interval=LAMBDA / 10.0  # Phi-based interval
        )
        
        # Connect the interface
        interface.connect()
        
        return interface