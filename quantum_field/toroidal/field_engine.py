"""
Toroidal Field Engine

This module implements the main Toroidal Field Engine that integrates all
toroidal field components into a unified system for quantum field operations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import math

from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from .toroidal_field import ToroidalField
from .resonance_chamber import ResonanceEqualizer, ResonanceChamber
from .compression_cycle import CompressionCycle, ExpansionCycle
from .counter_rotation import CounterRotatingField

class ToroidalFieldEngine:
    """
    The main engine that integrates all toroidal field components.
    
    Attributes:
        field: The primary toroidal field
        resonance_chamber: Resonance chamber for frequency harmonization
        compression_cycle: Compression cycle for coherence amplification
        expansion_cycle: Expansion cycle for phi-harmonic energy patterns
        counter_rotation: Counter-rotating field system for dimensional stability
        active_frequency: Currently active sacred frequency
        coherence_level: Current coherence level of the field
        balance_factor: Balance between input and output energy flows
        stability_factor: Dimensional stability factor
        energy_conservation: Energy conservation factor during operations
    """
    
    def __init__(self,
                 dimensions: Tuple[int, int, int] = (32, 32, 32),
                 frequency_name: str = 'unity',
                 initialize: bool = True):
        """
        Initialize the toroidal field engine.
        
        Args:
            dimensions: Field dimensions (height, width, depth)
            frequency_name: Initial sacred frequency
            initialize: Whether to initialize the field immediately
        """
        # Create the main toroidal field
        self.field = ToroidalField(
            dimensions=dimensions,
            frequency_name=frequency_name,
            initialize=initialize
        )
        
        # Get the sacred frequency
        self.active_frequency = SACRED_FREQUENCIES.get(frequency_name, SACRED_FREQUENCIES['unity'])
        
        # Initialize components if requested
        if initialize:
            # Create resonance chamber
            self.resonance_chamber = ResonanceChamber(toroidal_field=self.field)
            
            # Create compression cycle
            self.compression_cycle = CompressionCycle(toroidal_field=self.field)
            
            # Create expansion cycle
            self.expansion_cycle = ExpansionCycle(toroidal_field=self.field)
            
            # Create counter-rotating field system
            self.counter_rotation = CounterRotatingField(toroidal_field=self.field)
        else:
            self.resonance_chamber = None
            self.compression_cycle = None
            self.expansion_cycle = None
            self.counter_rotation = None
        
        # Initialize metrics
        self.coherence_level = 0.0
        self.balance_factor = 1.0
        self.stability_factor = 0.0
        
        # Settings
        self.energy_conservation = 0.95
        
        # Initialize engine
        if initialize:
            self.initialize_engine()
    
    def initialize_engine(self) -> None:
        """Initialize all engine components and establish baseline metrics."""
        # Make sure field is initialized
        if not hasattr(self.field, 'data') or self.field.data is None:
            self.field.initialize_toroidal_field()
        
        # Initialize resonance chamber if not already
        if self.resonance_chamber is None:
            self.resonance_chamber = ResonanceChamber(toroidal_field=self.field)
        
        # Initialize compression cycle if not already
        if self.compression_cycle is None:
            self.compression_cycle = CompressionCycle(toroidal_field=self.field)
        
        # Initialize expansion cycle if not already
        if self.expansion_cycle is None:
            self.expansion_cycle = ExpansionCycle(toroidal_field=self.field)
        
        # Initialize counter-rotation if not already
        if self.counter_rotation is None:
            self.counter_rotation = CounterRotatingField(toroidal_field=self.field)
        
        # Synchronize all components to the active frequency
        self.synchronize_frequency(self.active_frequency)
        
        # Initialize balanced input/output cycle
        self.field.create_input_output_cycle(io_ratio=1.0)
        
        # Create initial resonance pattern
        self.resonance_chamber.create_phi_harmonic_standing_waves()
        
        # Initialize counter-rotating fields
        self.counter_rotation.split_field(separation=0.5)
        
        # Calculate initial metrics
        self.update_metrics()
        
        print("Toroidal Field Engine initialized")
        print(f"Coherence: {self.coherence_level:.4f}")
        print(f"Balance: {self.balance_factor:.4f}")
        print(f"Stability: {self.stability_factor:.4f}")
    
    def update_metrics(self) -> Dict[str, float]:
        """
        Update all engine metrics.
        
        Returns:
            Dictionary of current metrics
        """
        # Calculate field metrics
        field_metrics = self.field.calculate_flow_metrics()
        
        # Store key metrics
        self.coherence_level = self.field.coherence
        self.balance_factor = self.field.balance_factor
        
        # Calculate stability if available
        if self.counter_rotation is not None:
            stability_metrics = self.counter_rotation.calculate_stability_metrics()
            self.stability_factor = stability_metrics.get("stability_factor", 0.0)
        else:
            self.stability_factor = 0.0
        
        # Return combined metrics
        metrics = {
            "coherence": self.coherence_level,
            "balance": self.balance_factor,
            "stability": self.stability_factor,
            "energy_level": self.field.energy_level,
            "flow_rate": self.field.flow_rate,
            "phi_alignment": self.field.phi_alignment
        }
        
        return metrics
    
    def synchronize_frequency(self, frequency_name: str) -> None:
        """
        Synchronize all components to a specific sacred frequency.
        
        Args:
            frequency_name: Name of the sacred frequency
        """
        # Get frequency value
        if frequency_name in SACRED_FREQUENCIES:
            freq_value = SACRED_FREQUENCIES[frequency_name]
        else:
            freq_value = SACRED_FREQUENCIES['unity']  # Default to 432 Hz
        
        self.active_frequency = freq_value
        
        # Synchronize field
        self.field.sync_to_frequency(frequency_name)
        
        # Synchronize resonance chamber
        if self.resonance_chamber is not None:
            self.resonance_chamber.tune_to_frequency(frequency_name)
        
        print(f"Engine synchronized to {frequency_name} frequency ({freq_value} Hz)")
    
    def optimize_field_coherence(self, target_coherence: float = 0.9) -> float:
        """
        Optimize field coherence using all available systems.
        
        Args:
            target_coherence: Target coherence level (0.0-1.0)
            
        Returns:
            Final coherence level
        """
        # Current coherence
        current_coherence = self.coherence_level
        
        # Only optimize if needed
        if current_coherence >= target_coherence:
            return current_coherence
        
        # Start with resonance chamber
        if self.resonance_chamber is not None:
            print("Optimizing with resonance chamber...")
            self.resonance_chamber.amplify_coherence(target_coherence)
            self.update_metrics()
        
        # If still needed, use compression cycle
        if self.coherence_level < target_coherence and self.compression_cycle is not None:
            print("Optimizing with compression cycle...")
            self.compression_cycle.optimize_field(target_coherence)
            self.update_metrics()
        
        # If still needed, use expansion cycle
        if self.coherence_level < target_coherence and self.expansion_cycle is not None:
            print("Optimizing with expansion cycle...")
            self.expansion_cycle.optimize_with_phi_cycles(target_coherence)
            self.update_metrics()
        
        # Final direct optimization if needed
        if self.coherence_level < target_coherence:
            print("Performing direct field optimization...")
            self.field.optimize_coherence(target_coherence)
            self.update_metrics()
        
        return self.coherence_level
    
    def stabilize_dimensions(self, target_stability: float = 0.9) -> float:
        """
        Stabilize field dimensions using counter-rotating fields.
        
        Args:
            target_stability: Target stability factor (0.0-1.0)
            
        Returns:
            Final stability factor
        """
        if self.counter_rotation is None:
            return 0.0
        
        # Current stability
        current_stability = self.stability_factor
        
        # Only stabilize if needed
        if current_stability >= target_stability:
            return current_stability
        
        print("Creating dimensional stability...")
        self.counter_rotation.create_dimensional_stability(target_stability)
        
        # Update metrics
        self.update_metrics()
        
        return self.stability_factor
    
    def balance_energy_flow(self, target_balance: float = 0.95) -> float:
        """
        Balance energy flow through the toroidal field.
        
        Args:
            target_balance: Target balance factor (0.0-1.0)
            
        Returns:
            Final balance factor
        """
        # Current balance
        current_balance = self.balance_factor
        
        # Only balance if needed
        if current_balance >= target_balance:
            return current_balance
        
        print("Balancing toroidal energy flow...")
        
        # Create balanced input/output cycle
        self.field.create_input_output_cycle(io_ratio=1.0)
        
        # Apply resonance to harmonize
        if self.resonance_chamber is not None:
            self.resonance_chamber.apply_resonance(strength=0.5)
        
        # Update metrics
        self.update_metrics()
        
        return self.balance_factor
    
    def create_self_sustaining_cycle(self) -> bool:
        """
        Create a self-sustaining energy cycle in the toroidal field.
        
        Returns:
            True if successful, False otherwise
        """
        # First, ensure high coherence
        sufficient_coherence = self.optimize_field_coherence(target_coherence=0.85)
        
        # Then, ensure dimensional stability
        sufficient_stability = self.stabilize_dimensions(target_stability=0.85)
        
        # Finally, balance energy flow
        sufficient_balance = self.balance_energy_flow(target_balance=0.9)
        
        # Check if all conditions are met
        if sufficient_coherence and sufficient_stability and sufficient_balance:
            print("Creating self-sustaining toroidal cycle...")
            
            # Create phi-harmonic standing waves
            if self.resonance_chamber is not None:
                self.resonance_chamber.create_phi_harmonic_standing_waves()
            
            # Perform one complete compression-expansion cycle
            if self.compression_cycle is not None:
                self.compression_cycle.perform_complete_cycle()
            
            # Create counter-rotating stability
            if self.counter_rotation is not None:
                self.counter_rotation.perform_stability_cycle()
            
            # Update metrics
            self.update_metrics()
            
            # Check final state
            is_self_sustaining = (
                self.coherence_level >= 0.85 and
                self.stability_factor >= 0.85 and
                self.balance_factor >= 0.9
            )
            
            if is_self_sustaining:
                print("Self-sustaining toroidal cycle created successfully")
            else:
                print("Failed to create fully self-sustaining cycle")
            
            return is_self_sustaining
        else:
            print("Unable to create self-sustaining cycle - prerequisites not met")
            return False
    
    def run_toroidal_cycle(self, 
                           steps: int = 10, 
                           time_factor: float = 0.1,
                           maintain_coherence: bool = True) -> List[Dict[str, float]]:
        """
        Run a complete toroidal energy cycle.
        
        Args:
            steps: Number of steps in the cycle
            time_factor: Time scaling factor for each step
            maintain_coherence: Whether to maintain coherence during the cycle
            
        Returns:
            List of metrics for each step
        """
        metrics_history = []
        
        print(f"Running toroidal cycle with {steps} steps...")
        
        # Initial metrics
        metrics_history.append(self.update_metrics())
        
        for step in range(steps):
            # Apply flow evolution
            self.field.apply_flow(time_factor)
            
            # Apply counter-rotation if available
            if self.counter_rotation is not None:
                self.counter_rotation.rotate_components(time_factor * 0.5)
            
            # Perform compression cycle step if available
            if self.compression_cycle is not None and step % 3 == 0:
                self.compression_cycle.perform_cycle_step(0.1)
            
            # Apply resonance if available
            if self.resonance_chamber is not None and step % 2 == 0:
                self.resonance_chamber.apply_resonance(0.3)
            
            # Maintain coherence if requested
            if maintain_coherence and step % 4 == 0:
                min_coherence = 0.7
                if self.field.coherence < min_coherence:
                    self.optimize_field_coherence(min_coherence)
            
            # Update metrics
            current_metrics = self.update_metrics()
            metrics_history.append(current_metrics.copy())
            
            # Print progress
            if step % 5 == 0 or step == steps - 1:
                print(f"Step {step+1}/{steps} - "
                      f"Coherence: {current_metrics['coherence']:.4f}, "
                      f"Balance: {current_metrics['balance']:.4f}, "
                      f"Stability: {current_metrics['stability']:.4f}")
        
        # Final optimization after cycle
        if maintain_coherence:
            self.optimize_field_coherence(0.85)
            final_metrics = self.update_metrics()
            metrics_history.append(final_metrics.copy())
        
        return metrics_history
    
    def visualize_toroidal_field(self, axis: int = 2) -> List[np.ndarray]:
        """
        Create visualization slices of the toroidal field.
        
        Args:
            axis: Axis to slice along (0, 1, or 2)
            
        Returns:
            List of 2D arrays representing field slices
        """
        # Get central slices
        return self.field.to_slices(axis=axis, center_slice=True)
    
    def evolve_field(self, steps: int = 1, time_factor: float = 0.1) -> None:
        """
        Evolve the toroidal field for a number of steps.
        
        Args:
            steps: Number of evolution steps
            time_factor: Time scaling factor for each step
        """
        for step in range(steps):
            # Apply flow evolution
            self.field.apply_flow(time_factor)
            
            # Apply counter-rotation if available
            if self.counter_rotation is not None:
                self.counter_rotation.rotate_components(time_factor * 0.5)
            
            # Update metrics every few steps
            if step % 5 == 0 or step == steps - 1:
                self.update_metrics()
    
    def save_field_state(self, file_path: str) -> None:
        """
        Save the current field state to a file.
        
        Args:
            file_path: Path to save the field state to
        """
        # Create state dictionary with all relevant data
        state = {
            'field_data': self.field.data,
            'dimensions': self.field.dimensions,
            'major_radius': self.field.major_radius,
            'minor_radius': self.field.minor_radius,
            'frequency': self.active_frequency,
            'coherence': self.coherence_level,
            'balance': self.balance_factor,
            'stability': self.stability_factor,
        }
        
        # Save clockwise and counterclockwise components if available
        if self.counter_rotation is not None:
            state['clockwise_field'] = self.counter_rotation.clockwise_field
            state['counterclockwise_field'] = self.counter_rotation.counterclockwise_field
        
        # Save to file using numpy
        try:
            np.save(file_path, state)
            print(f"Field state saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving field state: {e}")
            return False
    
    def load_field_state(self, file_path: str) -> bool:
        """
        Load field state from a file.
        
        Args:
            file_path: Path to load the field state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state from file
            state = np.load(file_path, allow_pickle=True).item()
            
            # Update field properties
            self.field.data = state['field_data']
            self.field.dimensions = state['dimensions']
            self.field.major_radius = state['major_radius']
            self.field.minor_radius = state['minor_radius']
            self.active_frequency = state['frequency']
            
            # Update component fields if available
            if 'clockwise_field' in state and 'counterclockwise_field' in state:
                if self.counter_rotation is not None:
                    self.counter_rotation.clockwise_field = state['clockwise_field']
                    self.counter_rotation.counterclockwise_field = state['counterclockwise_field']
            
            # Update metrics
            self.update_metrics()
            
            print(f"Field state loaded from {file_path}")
            print(f"Coherence: {self.coherence_level:.4f}")
            print(f"Balance: {self.balance_factor:.4f}")
            print(f"Stability: {self.stability_factor:.4f}")
            
            return True
        except Exception as e:
            print(f"Error loading field state: {e}")
            return False


def demo_toroidal_field_engine():
    """
    Demonstrate the functionality of the toroidal field engine.
    
    Returns:
        The created toroidal field engine
    """
    # Create the engine
    print("Creating Toroidal Field Engine...")
    engine = ToroidalFieldEngine(dimensions=(32, 32, 32), frequency_name='unity')
    
    # Print initial metrics
    metrics = engine.update_metrics()
    print("\nInitial engine metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Synchronize to different frequency
    print("\nSynchronizing to 'love' frequency...")
    engine.synchronize_frequency('love')
    
    # Optimize field coherence
    print("\nOptimizing field coherence...")
    optimized_coherence = engine.optimize_field_coherence(target_coherence=0.9)
    print(f"Optimized coherence: {optimized_coherence:.4f}")
    
    # Stabilize dimensions
    print("\nStabilizing field dimensions...")
    stabilized = engine.stabilize_dimensions(target_stability=0.85)
    print(f"Dimension stability: {stabilized:.4f}")
    
    # Balance energy flow
    print("\nBalancing energy flow...")
    balanced = engine.balance_energy_flow(target_balance=0.9)
    print(f"Energy balance: {balanced:.4f}")
    
    # Create self-sustaining cycle
    print("\nCreating self-sustaining cycle...")
    is_self_sustaining = engine.create_self_sustaining_cycle()
    print(f"Self-sustaining: {is_self_sustaining}")
    
    # Run toroidal cycle
    print("\nRunning toroidal cycle...")
    cycle_metrics = engine.run_toroidal_cycle(steps=10, time_factor=0.1)
    
    # Print final metrics
    final_metrics = engine.update_metrics()
    print("\nFinal engine metrics:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return engine

if __name__ == "__main__":
    engine = demo_toroidal_field_engine()