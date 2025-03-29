"""
CASCADE⚡𓂧φ∞ Phi-Harmonic Python Demonstration

This example showcases the phi-harmonic extensions to Python's execution model,
providing a practical demonstration of toroidal execution flows, phi-resonant
function composition, and timeline navigation.
"""

import sys
import time
import math
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import argparse

# Import phi-harmonic core
try:
    from cascade.phi_python_core import (
        phi_function,
        cascade_system,
        create_toroidal_memory,
        phi_timeline_context,
        get_phi_coherence,
        set_phi_coherence,
        PhiTimer,
        PhiConversion,
        ToroidalMemory,
        TimelineNavigator
    )
except ImportError:
    print("Error: Phi-Harmonic Python Core not found.")
    print("Make sure cascade/phi_python_core.py exists in your project.")
    sys.exit(1)

# Constants
PHI = 1.618033988749895
LAMBDA = 0.618033988749895
PHI_PHI = PHI ** PHI

# Example phi-harmonic data structures
class PhiHarmonicMatrix:
    """Matrix with phi-harmonic operations."""
    
    def __init__(self, rows: int, cols: int):
        """Initialize matrix with phi-harmonic dimensions."""
        # Adjust dimensions to nearest phi-harmonic values
        self.rows = self._nearest_phi_harmonic(rows)
        self.cols = self._nearest_phi_harmonic(cols)
        
        # Initialize data with phi-based pattern
        self.data = np.zeros((self.rows, self.cols))
        self._initialize_phi_pattern()
        
        # Coherence tracking
        self.coherence = 0.8
        self.operation_count = 0
    
    def _nearest_phi_harmonic(self, n: int) -> int:
        """Find nearest phi-harmonic number (Fibonacci-adjacent)."""
        if n <= 0:
            return 1
            
        # Find Fibonacci number closest to n
        fib = [1, 1]
        while fib[-1] < n:
            fib.append(fib[-1] + fib[-2])
            
        # Return closest
        if abs(fib[-1] - n) < abs(fib[-2] - n):
            return fib[-1]
        return fib[-2]
    
    def _initialize_phi_pattern(self) -> None:
        """Initialize matrix with phi-harmonic pattern."""
        for i in range(self.rows):
            for j in range(self.cols):
                # Create phi-based wave pattern
                x = i / self.rows * 2 - 1  # [-1, 1]
                y = j / self.cols * 2 - 1  # [-1, 1]
                r = math.sqrt(x*x + y*y)
                
                # Apply phi-wave pattern
                self.data[i, j] = math.sin(r * PHI * 5) * math.exp(-r * LAMBDA)
    
    @phi_function
    def apply_phi_transform(self) -> None:
        """Apply phi-harmonic transformation to matrix."""
        # Create phi-harmonic gradient
        x = np.linspace(-1, 1, self.cols)
        y = np.linspace(-1, 1, self.rows)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X*X + Y*Y)
        
        # Apply phi-wave transformation
        phi_wave = np.sin(R * PHI * (1 + 0.2 * math.sin(self.operation_count / 10)))
        
        # Apply transformation with phi-attenuation
        self.data = self.data * LAMBDA + phi_wave * (1 - LAMBDA)
        
        # Update tracking
        self.operation_count += 1
        self._update_coherence()
    
    def _update_coherence(self) -> None:
        """Update matrix coherence based on values."""
        # Calculate spectral coherence using FFT
        fft = np.fft.fft2(self.data)
        fft_mag = np.abs(fft)
        
        # Measure energy concentration in phi-harmonic frequencies
        total_energy = np.sum(fft_mag)
        
        if total_energy > 0:
            # Find peak frequencies
            peaks = np.sort(fft_mag.flatten())[-10:]
            peak_energy = np.sum(peaks)
            
            # Calculate coherence ratio
            energy_ratio = peak_energy / total_energy
            
            # Update coherence with phi-weighted averaging
            self.coherence = self.coherence * LAMBDA + energy_ratio * (1 - LAMBDA)
        
        # Ensure coherence is in [0, 1]
        self.coherence = max(0, min(1, self.coherence))
    
    @phi_function
    def get_coherence(self) -> float:
        """Get current matrix coherence."""
        return self.coherence
    
    @cascade_system.cascade('memoize', 'validate')
    def phi_eigenvalues(self) -> List[float]:
        """Calculate phi-harmonic eigenvalues of matrix."""
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(self.data)
        
        # Sort by proximity to PHI
        phi_distance = np.abs(eigenvalues - PHI)
        sorted_indices = np.argsort(phi_distance)
        
        # Return as list
        return eigenvalues[sorted_indices].tolist()
    
    def to_string(self) -> str:
        """Get string representation suitable for display."""
        # Create compact representation
        lines = []
        for row in self.data:
            line = " ".join(f"{x:.2f}" for x in row)
            lines.append(line)
            
        # Add coherence info
        lines.append(f"Coherence: {self.coherence:.4f}")
        
        return "\n".join(lines)


class PhiHarmonicSystem:
    """
    Phi-harmonic computational system with toroidal execution flow.
    
    This system demonstrates:
    - Phi-based task scheduling
    - Toroidal execution patterns
    - Phi-resonant data transformations
    - Timeline navigation capabilities
    """
    
    def __init__(self):
        """Initialize phi-harmonic system."""
        self.creation_time = time.time()
        
        # Create toroidal memory
        self.memory = create_toroidal_memory("phi_system", 21)
        
        # Phi-harmonic timer
        self.timer = PhiTimer()
        
        # Create phi-harmonic matrix
        self.matrix = PhiHarmonicMatrix(8, 13)
        
        # Task queues with phi-harmonic scheduling
        self.task_queue = []
        self.completed_tasks = []
        
        # System coherence
        self.coherence = 0.8
        
        # Initialize timeline storage
        self.timeline_positions = []
    
    @phi_function(coherence_check=True, timeline_capture=True, phi_synchronize=True)
    def run_phi_task(self, task_id: int, iterations: int = 5) -> Dict[str, Any]:
        """Execute a phi-harmonic task."""
        # Record starting state
        start_coherence = self.coherence
        start_time = time.time()
        
        # Initialize result storage
        results = []
        coherence_log = []
        
        # Execute iterations
        for i in range(iterations):
            # Calculate phi-weighted iteration factor
            iteration_factor = PHI ** (i % 7)
            
            # Execute iteration with phi-weighting
            result = math.sin(task_id * PHI + i * LAMBDA) * iteration_factor
            results.append(result)
            
            # Update matrix
            self.matrix.apply_phi_transform()
            
            # Store value in toroidal memory
            self.memory.put(f"task_{task_id}_iter_{i}", result)
            
            # Update coherence
            self.coherence = PhiConversion.phi_clamp(
                self.coherence * 0.8 + self.matrix.coherence * 0.2
            )
            coherence_log.append(self.coherence)
            
            # Wait for next phi-harmonic pulse
            self.timer.wait_for_next_pulse()
        
        # Gather task data
        eigenvalues = self.matrix.phi_eigenvalues()
        top_eigenvalues = eigenvalues[:3] if len(eigenvalues) > 3 else eigenvalues
        
        # Create task result
        task_result = {
            "task_id": task_id,
            "start_time": start_time,
            "end_time": time.time(),
            "results": results,
            "coherence_start": start_coherence,
            "coherence_end": self.coherence,
            "coherence_log": coherence_log,
            "top_eigenvalues": top_eigenvalues,
            "memory_size": self.memory.size
        }
        
        # Record task as completed
        self.completed_tasks.append(task_result)
        
        # Record current timeline position
        self.timeline_positions.append({
            "time": time.time(),
            "task_id": task_id,
            "coherence": self.coherence,
            "matrix_state": self.matrix.data.copy()
        })
        
        return task_result
    
    @cascade_system.cascade('log', 'memoize')
    def phi_resonance_analysis(self) -> Dict[str, Any]:
        """Analyze phi-resonance patterns in completed tasks."""
        if not self.completed_tasks:
            return {"error": "No completed tasks"}
            
        # Collect task results
        coherence_values = [task["coherence_end"] for task in self.completed_tasks]
        result_values = []
        for task in self.completed_tasks:
            result_values.extend(task["results"])
            
        # Calculate phi-harmonic metrics
        phi_distances = [abs(val - PHI) for val in result_values]
        avg_phi_distance = sum(phi_distances) / len(phi_distances) if phi_distances else 0
        
        # Calculate phi-resonance score
        resonance = 1.0 - (avg_phi_distance / PHI)
        
        # Analyze coherence progression
        coherence_diff = coherence_values[-1] - coherence_values[0] if len(coherence_values) > 1 else 0
        
        # Calculate overall harmony
        harmony = (resonance * 0.6 + self.coherence * 0.4) * (1 + 0.2 * coherence_diff)
        
        return {
            "tasks_analyzed": len(self.completed_tasks),
            "current_coherence": self.coherence,
            "phi_resonance": resonance,
            "harmony": harmony,
            "coherence_progression": coherence_diff,
            "matrix_coherence": self.matrix.coherence
        }
    
    @phi_function
    def navigate_timeline(self, position: int) -> Optional[Dict[str, Any]]:
        """Navigate to a specific timeline position."""
        if not self.timeline_positions:
            return None
            
        if 0 <= position < len(self.timeline_positions):
            snapshot = self.timeline_positions[position]
            
            # Restore matrix state
            self.matrix.data = snapshot["matrix_state"].copy()
            self.matrix._update_coherence()
            
            # Restore coherence
            self.coherence = snapshot["coherence"]
            
            return {
                "position": position,
                "time": snapshot["time"],
                "task_id": snapshot["task_id"],
                "coherence": snapshot["coherence"],
                "matrix_coherence": self.matrix.coherence
            }
            
        return None
    
    @phi_function
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        # Calculate overall phi-resonance
        tasks_completed = len(self.completed_tasks)
        uptime = time.time() - self.creation_time
        
        # Get memory metrics
        memory_size = self.memory.size
        memory_coherence = self.memory.coherence
        
        # Calculate system-wide coherence
        system_coherence = (
            self.coherence * 0.5 + 
            self.matrix.coherence * 0.3 + 
            memory_coherence * 0.2
        )
        
        return {
            "uptime": uptime,
            "tasks_completed": tasks_completed,
            "system_coherence": system_coherence,
            "matrix_coherence": self.matrix.coherence,
            "memory_coherence": memory_coherence,
            "memory_size": memory_size,
            "timeline_positions": len(self.timeline_positions)
        }


def phi_demo_sequence(system: PhiHarmonicSystem, tasks: int = 5, iterations: int = 7) -> None:
    """
    Run phi-harmonic demonstration sequence.
    
    Args:
        system: The phi-harmonic system to use
        tasks: Number of tasks to run
        iterations: Iterations per task
    """
    print("\n=== CASCADE⚡𓂧φ∞ Phi-Harmonic Python Demonstration ===\n")
    
    # Show initial state
    print("Initial system status:")
    status = system.get_system_status()
    print(f"  System coherence: {status['system_coherence']:.4f}")
    print(f"  Matrix coherence: {status['matrix_coherence']:.4f}")
    print(f"  Memory coherence: {status['memory_coherence']:.4f}")
    
    # Run tasks
    print(f"\nExecuting {tasks} phi-harmonic tasks with {iterations} iterations each...")
    for i in range(tasks):
        print(f"\nRunning task {i+1}/{tasks}...")
        result = system.run_phi_task(i+1, iterations)
        
        # Show task results
        print(f"  Task completed in {result['end_time'] - result['start_time']:.4f}s")
        print(f"  Final coherence: {result['coherence_end']:.4f}")
        
        # Pause between tasks
        if i < tasks - 1:
            time.sleep(0.5)
    
    # Show final analysis
    print("\nPhi-Resonance Analysis:")
    analysis = system.phi_resonance_analysis()
    print(f"  Tasks analyzed: {analysis['tasks_analyzed']}")
    print(f"  Phi-resonance: {analysis['phi_resonance']:.4f}")
    print(f"  System harmony: {analysis['harmony']:.4f}")
    print(f"  Coherence progression: {analysis['coherence_progression']:+.4f}")
    
    # Demonstrate timeline navigation
    print("\nTimeline Navigation Demonstration:")
    if system.timeline_positions:
        # Navigate to middle position
        mid_pos = len(system.timeline_positions) // 2
        result = system.navigate_timeline(mid_pos)
        
        print(f"  Navigated to timeline position {mid_pos}")
        print(f"  Task ID: {result['task_id']}")
        print(f"  Coherence: {result['coherence']:.4f}")
        
        # Return to latest position
        latest_pos = len(system.timeline_positions) - 1
        system.navigate_timeline(latest_pos)
        print(f"  Returned to latest timeline position {latest_pos}")
    else:
        print("  No timeline positions available")
    
    # Show final state
    print("\nFinal System Status:")
    status = system.get_system_status()
    print(f"  Tasks completed: {status['tasks_completed']}")
    print(f"  Uptime: {status['uptime']:.2f}s")
    print(f"  System coherence: {status['system_coherence']:.4f}")
    print(f"  Matrix coherence: {status['matrix_coherence']:.4f}")
    print(f"  Memory coherence: {status['memory_coherence']:.4f}")
    print(f"  Memory size: {status['memory_size']} items")
    print(f"  Timeline positions: {status['timeline_positions']}")
    
    print("\n=== Demonstration Complete ===")


def timeline_navigation_demo(system: PhiHarmonicSystem) -> None:
    """
    Run an interactive timeline navigation demonstration.
    
    Args:
        system: The phi-harmonic system to use
    """
    print("\n=== CASCADE⚡𓂧φ∞ Timeline Navigation Demonstration ===\n")
    
    # Check if timeline positions are available
    if not system.timeline_positions:
        print("No timeline positions available. Run phi_demo_sequence first.")
        return
        
    # Show timeline positions
    print(f"Timeline has {len(system.timeline_positions)} positions:")
    for i, pos in enumerate(system.timeline_positions):
        print(f"  Position {i}: Task {pos['task_id']}, Coherence: {pos['coherence']:.4f}")
    
    # Use timeline navigator for advanced features
    with phi_timeline_context() as timeline:
        print(f"\nTimeline Navigator active with {len(timeline.snapshots)} snapshots.")
        
        # Navigate to snapshots
        if timeline.snapshots:
            mid_idx = len(timeline.snapshots) // 2
            snapshot = timeline.navigate_to(mid_idx)
            
            if snapshot:
                print(f"\nNavigated to snapshot {mid_idx}:")
                print(f"  Function: {snapshot['func_name']}")
                print(f"  Coherence: {snapshot['coherence']:.4f}")
                print(f"  Frame depth: {snapshot['frame_depth']}")
            
            # Create branch
            print("\nCreating timeline branch 'alternate'...")
            timeline.create_branch("alternate")
            print(f"Current branch: {timeline.current_branch}")
            
            # Switch back to main
            print("Switching back to main branch...")
            timeline.switch_branch("main")
            print(f"Current branch: {timeline.current_branch}")
            
            # Get latest snapshot
            latest = timeline.get_snapshot()
            if latest:
                print(f"\nLatest snapshot:")
                print(f"  Function: {latest['func_name']}")
                print(f"  Coherence: {latest['coherence']:.4f}")
        
        print("\nTimeline navigation complete.")
    
    print("\n=== Navigation Demonstration Complete ===")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Phi-Harmonic Python Demo")
    parser.add_argument('--tasks', type=int, default=5, help="Number of tasks to run")
    parser.add_argument('--iterations', type=int, default=7, help="Iterations per task")
    parser.add_argument('--timeline', action='store_true', help="Run timeline navigation demo")
    args = parser.parse_args()
    
    # Create phi-harmonic system
    system = PhiHarmonicSystem()
    
    # Run main demo sequence
    phi_demo_sequence(system, args.tasks, args.iterations)
    
    # Run timeline navigation demo if requested
    if args.timeline:
        timeline_navigation_demo(system)