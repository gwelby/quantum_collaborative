"""
CASCADE⚡𓂧φ∞ Network Field Visualization System

This module provides visualization tools for distributed quantum fields
using phi-harmonic principles, showing entanglement relationships and 
coherence patterns across network nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# Try to import existing visualization tools
try:
    from cascade.visualization.enhanced_visualizer import EnhancedFieldVisualizer
except ImportError:
    pass

# Try to import phi-quantum network
try:
    from cascade.phi_quantum_network import (
        PhiQuantumField,
        PhiQuantumNetwork,
        PHI, LAMBDA, PHI_PHI,
        PHI_QUANTUM_PORT,
        PHI_FREQUENCIES
    )
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895
    LAMBDA = 0.618033988749895
    PHI_PHI = PHI ** PHI
    PHI_FREQUENCIES = [432, 528, 594, 672, 720, 768, 888]
    logging.warning("CASCADE phi-quantum-network not found, using fallback implementations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("network_field_visualizer")


class NetworkFieldVisualizer:
    """
    Visualization system for distributed quantum fields across network nodes.
    
    This visualizer provides real-time visualization of:
    1. Quantum field states across multiple nodes
    2. Coherence patterns and entanglement relationships
    3. Timeline synchronization points
    4. Consciousness bridge synchronization
    """
    
    def __init__(self, 
               quantum_field: Optional['PhiQuantumField'] = None,
               node_limit: int = 6):
        """
        Initialize the network field visualizer.
        
        Args:
            quantum_field: PhiQuantumField instance to visualize
            node_limit: Maximum number of nodes to visualize
        """
        self.quantum_field = quantum_field
        self.node_limit = node_limit
        
        # Visualization components
        self.fig = None
        self.axes = {}
        self.node_plots = {}
        self.animation = None
        self.entanglement_lines = []
        self.coherence_plots = {}
        
        # Network data
        self.node_data = {}
        self.entanglement_data = {}
        self.coherence_history = {}
        self.consciousness_levels = {}
        
        # Status flags
        self.running = False
        self.update_thread = None
        self.update_queue = queue.Queue()
        
        # Create phi-harmonic color maps
        self.color_maps = self._create_phi_colormaps()
        
        # Set up field probe points (for field sampling)
        self.probe_points = self._create_probe_points()
        
        logger.info("Network field visualizer initialized")
    
    def start_visualization(self, 
                         mode: str = "3d", 
                         update_interval: int = 50) -> None:
        """
        Start visualization of network quantum fields.
        
        Args:
            mode: Visualization mode ("3d", "grid", "coherence", or "combined")
            update_interval: Animation update interval in milliseconds
        """
        if self.running:
            return
            
        self.running = True
        
        # Create figure and axes based on mode
        if mode == "3d":
            self._create_3d_visualization()
        elif mode == "grid":
            self._create_grid_visualization()
        elif mode == "coherence":
            self._create_coherence_visualization()
        elif mode == "combined":
            self._create_combined_visualization()
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            interval=update_interval,
            blit=False
        )
        
        logger.info(f"Started network visualization in {mode} mode")
        
        # Show the figure (blocks until closed)
        plt.show()
        
        # Clean up when window is closed
        self.stop_visualization()
    
    def stop_visualization(self) -> None:
        """Stop visualization and clean up."""
        if not self.running:
            return
            
        logger.info("Stopping network visualization")
        
        # Stop animation
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        # Set running flag to false to stop thread
        self.running = False
        
        # Wait for thread to terminate
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            
        logger.info("Network visualization stopped")
    
    def save_visualization(self, filename: str, dpi: int = 300) -> None:
        """
        Save current visualization to file.
        
        Args:
            filename: Output filename
            dpi: Image resolution
        """
        if not self.fig:
            logger.warning("No visualization to save")
            return
            
        try:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='black')
            logger.info(f"Visualization saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def save_animation(self, filename: str, 
                    duration: int = 10, 
                    fps: int = 30,
                    dpi: int = 150) -> None:
        """
        Save animation to file.
        
        Args:
            filename: Output filename (mp4 or gif)
            duration: Animation duration in seconds
            fps: Frames per second
            dpi: Video resolution
        """
        if not self.fig:
            logger.warning("No visualization to save")
            return
            
        try:
            # Create writer based on file extension
            if filename.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps)
            elif filename.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
            else:
                raise ValueError("Unsupported file format. Use .mp4 or .gif")
            
            # Create new animation for saving
            frames = duration * fps
            temp_anim = animation.FuncAnimation(
                self.fig,
                self._update_animation,
                frames=frames,
                interval=1000 // fps,
                blit=False
            )
            
            logger.info(f"Saving animation to {filename} ({duration}s at {fps} fps)...")
            temp_anim.save(filename, writer=writer, dpi=dpi)
            logger.info(f"Animation saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving animation: {e}")
    
    def _create_phi_colormaps(self) -> Dict[str, LinearSegmentedColormap]:
        """Create phi-harmonic color maps."""
        colormaps = {}
        
        # Phi-harmonic colors
        phi_colors = {
            'unity': [(0.0, (0.1, 0.2, 0.6)),
                     (LAMBDA, (0.2, 0.5, 0.8)),
                     (0.5, (0.3, 0.6, 0.9)),
                     (PHI - 1, (0.4, 0.7, 1.0)),
                     (1.0, (0.5, 0.8, 1.0))],
            
            'creation': [(0.0, (0.1, 0.4, 0.1)),
                        (LAMBDA, (0.2, 0.6, 0.3)),
                        (0.5, (0.3, 0.8, 0.4)),
                        (PHI - 1, (0.4, 0.9, 0.5)),
                        (1.0, (0.5, 1.0, 0.6))],
            
            'consciousness': [(0.0, (0.4, 0.1, 0.4)),
                           (LAMBDA, (0.6, 0.2, 0.6)),
                           (0.5, (0.8, 0.3, 0.8)),
                           (PHI - 1, (0.9, 0.4, 0.9)),
                           (1.0, (1.0, 0.5, 1.0))],
            
            'entanglement': [(0.0, (0.4, 0.1, 0.1)),
                           (LAMBDA, (0.6, 0.2, 0.2)),
                           (0.5, (0.8, 0.3, 0.3)),
                           (PHI - 1, (0.9, 0.4, 0.4)),
                           (1.0, (1.0, 0.5, 0.5))],
            
            'timeline': [(0.0, (0.1, 0.1, 0.1)),
                        (LAMBDA, (0.2, 0.2, 0.2)),
                        (0.5, (0.5, 0.5, 0.5)),
                        (PHI - 1, (0.8, 0.8, 0.8)),
                        (1.0, (1.0, 1.0, 1.0))]
        }
        
        # Create colormaps
        for name, colors in phi_colors.items():
            cdict = {
                'red': [(x, c[0], c[0]) for x, c in colors],
                'green': [(x, c[1], c[1]) for x, c in colors],
                'blue': [(x, c[2], c[2]) for x, c in colors]
            }
            colormaps[name] = LinearSegmentedColormap(name, cdict)
        
        return colormaps
    
    def _create_probe_points(self, count: int = 8) -> np.ndarray:
        """
        Create field probe points for sampling across nodes.
        
        Args:
            count: Number of probe points
            
        Returns:
            Array of probe point coordinates
        """
        # Create phi-harmonically distributed probe points on a sphere
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        
        # Use Fibonacci sphere distribution
        indices = np.arange(0, count, dtype=float) + 0.5
        phi_angle = 2.0 * np.pi * (indices / phi)
        theta = np.arccos(1.0 - 2.0 * indices / count)
        
        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi_angle)
        y = np.sin(theta) * np.sin(phi_angle)
        z = np.cos(theta)
        
        # Scale to field coordinates (assuming field is normalized to [-1,1])
        probe_points = np.column_stack([x, y, z])
        
        return probe_points
    
    def _update_loop(self) -> None:
        """Background thread for updating network data."""
        logger.debug("Update loop starting")
        
        update_count = 0
        
        while self.running:
            try:
                # Skip if no quantum field
                if not self.quantum_field:
                    time.sleep(0.5)
                    continue
                
                # Update network data
                self._update_network_data()
                
                # Add update to queue
                self.update_queue.put(update_count)
                update_count += 1
                
                # Sleep briefly
                time.sleep(0.2)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in update loop: {e}")
                    time.sleep(1.0)  # Avoid tight error loop
        
        logger.debug("Update loop terminated")
    
    def _update_network_data(self) -> None:
        """Update network data from quantum field."""
        try:
            # Get connected nodes
            nodes = self.quantum_field.get_connected_nodes()
            
            # Limit number of nodes
            if len(nodes) > self.node_limit:
                # Sort by coherence to prioritize high-coherence nodes
                nodes.sort(key=lambda n: n.get('coherence', 0), reverse=True)
                nodes = nodes[:self.node_limit]
            
            # Update node data
            for node in nodes:
                node_id = node['id']
                
                # Initialize if new node
                if node_id not in self.node_data:
                    self.node_data[node_id] = {
                        'id': node_id,
                        'sample_times': [],
                        'coherence_values': [],
                        'consciousness_level': node.get('consciousness_level', 1),
                        'field_samples': [],
                        'entangled': node.get('entangled', False)
                    }
                
                # Update node data
                self.node_data[node_id]['coherence'] = node.get('coherence', 0.0)
                self.node_data[node_id]['consciousness_level'] = node.get('consciousness_level', 1)
                self.node_data[node_id]['entangled'] = node.get('entangled', False)
                self.node_data[node_id]['last_seen'] = node.get('last_seen', 0)
                
                # Add to coherence history
                current_time = time.time()
                if not self.node_data[node_id]['sample_times'] or \
                   current_time - self.node_data[node_id]['sample_times'][-1] >= 0.5:
                    self.node_data[node_id]['sample_times'].append(current_time)
                    self.node_data[node_id]['coherence_values'].append(node.get('coherence', 0.0))
                    
                    # Limit history length
                    if len(self.node_data[node_id]['sample_times']) > 100:
                        self.node_data[node_id]['sample_times'].pop(0)
                        self.node_data[node_id]['coherence_values'].pop(0)
                
                # Sample field values (if possible)
                self._sample_node_field(node_id)
            
            # Update entanglement data
            entangled_nodes = self.quantum_field.get_entangled_nodes()
            
            # Reset all entanglement flags
            for node_id in self.node_data:
                self.node_data[node_id]['entangled'] = False
            
            # Update entanglement pairs
            self.entanglement_data = {}
            for node_id in entangled_nodes:
                if node_id in self.node_data:
                    self.node_data[node_id]['entangled'] = True
                    
                    # Create entanglement pairs
                    for other_id in entangled_nodes:
                        if other_id != node_id and other_id in self.node_data:
                            pair_key = tuple(sorted([node_id, other_id]))
                            self.entanglement_data[pair_key] = {
                                'strength': min(self.node_data[node_id]['coherence'],
                                              self.node_data[other_id]['coherence'])
                            }
            
            # Update own node data
            own_coherence = self.quantum_field.get_field_coherence()
            own_consciousness = self.quantum_field.get_consciousness_level()
            
            # Store in local coherence history
            if 'local' not in self.coherence_history:
                self.coherence_history['local'] = {
                    'times': [],
                    'values': [],
                    'consciousness': []
                }
            
            current_time = time.time()
            self.coherence_history['local']['times'].append(current_time)
            self.coherence_history['local']['values'].append(own_coherence)
            self.coherence_history['local']['consciousness'].append(own_consciousness)
            
            # Limit history length
            if len(self.coherence_history['local']['times']) > 100:
                self.coherence_history['local']['times'].pop(0)
                self.coherence_history['local']['values'].pop(0)
                self.coherence_history['local']['consciousness'].pop(0)
        
        except Exception as e:
            logger.error(f"Error updating network data: {e}")
    
    def _sample_node_field(self, node_id: str) -> None:
        """
        Sample field values from a node.
        
        Args:
            node_id: Node ID to sample from
        """
        if node_id not in self.node_data:
            return
            
        # Request field snapshot if it's an entangled node
        if self.node_data[node_id]['entangled']:
            try:
                # Request field snapshot
                query_id = self.quantum_field.request_field_info(node_id, "field_snapshot")
                
                if query_id:
                    # Wait for result
                    result = self.quantum_field.get_query_result(query_id, timeout=0.5)
                    
                    if result and 'result' in result:
                        # Get snapshot data
                        snapshot = result['result'].get('snapshot')
                        downsample = result['result'].get('downsample', 1)
                        
                        if snapshot:
                            # Convert to array
                            field = np.array(snapshot)
                            
                            # Sample at probe points
                            samples = []
                            for p in self.probe_points:
                                # Convert from [-1,1] to field indices
                                x = int((p[0] + 1) / 2 * (field.shape[0] - 1))
                                y = int((p[1] + 1) / 2 * (field.shape[1] - 1))
                                z = int((p[2] + 1) / 2 * (field.shape[2] - 1))
                                
                                # Get value (with bounds checking)
                                if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and 0 <= z < field.shape[2]:
                                    value = field[x, y, z]
                                else:
                                    value = 0.0
                                    
                                samples.append(value)
                            
                            # Store samples
                            self.node_data[node_id]['field_samples'] = samples
            
            except Exception as e:
                logger.debug(f"Error sampling field for node {node_id}: {e}")
    
    def _create_3d_visualization(self) -> None:
        """Create 3D visualization of network quantum fields."""
        # Create figure
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')
        self.fig.canvas.manager.set_window_title("CASCADE⚡𓂧φ∞ Network Field Visualization")
        
        # Create main 3D axis for network topology
        self.axes['network'] = self.fig.add_subplot(121, projection='3d', facecolor='black')
        
        # Create 3D axis for local field
        self.axes['field'] = self.fig.add_subplot(122, projection='3d', facecolor='black')
        
        # Set up axes
        for ax_name, ax in self.axes.items():
            ax.set_facecolor('black')
            ax.w_xaxis.set_pane_color((0, 0, 0, 0.2))
            ax.w_yaxis.set_pane_color((0, 0, 0, 0.2))
            ax.w_zaxis.set_pane_color((0, 0, 0, 0.2))
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')
        
        # Network topology axis
        self.axes['network'].set_title("Quantum Network Topology", color='white', fontsize=14)
        self.axes['network'].set_xlim(-1.2, 1.2)
        self.axes['network'].set_ylim(-1.2, 1.2)
        self.axes['network'].set_zlim(-1.2, 1.2)
        
        # Local field axis
        self.axes['field'].set_title("Local Quantum Field", color='white', fontsize=14)
        
        # Add placeholder for field visualization
        self._create_placeholder_field()
        
        # Add main title
        self.fig.suptitle("CASCADE⚡𓂧φ∞ Network Field Visualization", 
                        color='white', fontsize=16, y=0.98)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    def _create_grid_visualization(self) -> None:
        """Create grid visualization of multiple quantum fields."""
        # Create figure
        self.fig = plt.figure(figsize=(15, 10), facecolor='black')
        self.fig.canvas.manager.set_window_title("CASCADE⚡𓂧φ∞ Network Grid Visualization")
        
        # Create grid of axes for node fields
        max_nodes = min(self.node_limit, 6)  # Max display 6 nodes in grid
        cols = min(3, max_nodes)
        rows = (max_nodes + cols - 1) // cols  # Ceiling division
        
        # Create main grid
        grid = gridspec.GridSpec(rows, cols, figure=self.fig)
        
        # Create axes for each node
        for i in range(max_nodes):
            ax = self.fig.add_subplot(grid[i // cols, i % cols], projection='3d', facecolor='black')
            
            # Set up axis
            ax.set_facecolor('black')
            ax.w_xaxis.set_pane_color((0, 0, 0, 0.2))
            ax.w_yaxis.set_pane_color((0, 0, 0, 0.2))
            ax.w_zaxis.set_pane_color((0, 0, 0, 0.2))
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')
            
            # Set default title
            ax.set_title(f"Node {i+1}", color='white', fontsize=12)
            
            # Store axis
            self.axes[f"node_{i}"] = ax
        
        # Add main title
        self.fig.suptitle("CASCADE⚡𓂧φ∞ Network Grid Visualization", 
                        color='white', fontsize=16, y=0.98)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    def _create_coherence_visualization(self) -> None:
        """Create coherence visualization for network nodes."""
        # Create figure
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')
        self.fig.canvas.manager.set_window_title("CASCADE⚡𓂧φ∞ Network Coherence Visualization")
        
        # Create grid
        grid = gridspec.GridSpec(2, 2, figure=self.fig)
        
        # Create coherence plot
        self.axes['coherence'] = self.fig.add_subplot(grid[0, :], facecolor='black')
        self.axes['coherence'].set_title("Network Coherence", color='white', fontsize=14)
        self.axes['coherence'].set_facecolor('black')
        self.axes['coherence'].spines['bottom'].set_color('white')
        self.axes['coherence'].spines['top'].set_color('white')
        self.axes['coherence'].spines['right'].set_color('white')
        self.axes['coherence'].spines['left'].set_color('white')
        self.axes['coherence'].tick_params(axis='x', colors='white')
        self.axes['coherence'].tick_params(axis='y', colors='white')
        self.axes['coherence'].set_ylim(0, 1.0)
        self.axes['coherence'].set_xlabel("Time (s)", color='white')
        self.axes['coherence'].set_ylabel("Coherence", color='white')
        self.axes['coherence'].grid(True, color='white', alpha=0.2)
        
        # Mark phi-harmonic levels
        self.axes['coherence'].axhline(y=LAMBDA, color='gold', linestyle='--', alpha=0.5)
        self.axes['coherence'].axhline(y=1/PHI, color='teal', linestyle='--', alpha=0.5)
        
        # Create consciousness level plot
        self.axes['consciousness'] = self.fig.add_subplot(grid[1, 0], facecolor='black')
        self.axes['consciousness'].set_title("Consciousness Levels", color='white', fontsize=14)
        self.axes['consciousness'].set_facecolor('black')
        self.axes['consciousness'].spines['bottom'].set_color('white')
        self.axes['consciousness'].spines['top'].set_color('white')
        self.axes['consciousness'].spines['right'].set_color('white')
        self.axes['consciousness'].spines['left'].set_color('white')
        self.axes['consciousness'].tick_params(axis='x', colors='white')
        self.axes['consciousness'].tick_params(axis='y', colors='white')
        self.axes['consciousness'].set_ylim(0.5, 7.5)
        self.axes['consciousness'].set_yticks(range(1, 8))
        self.axes['consciousness'].set_xlabel("Node ID", color='white')
        self.axes['consciousness'].set_ylabel("Level", color='white')
        
        # Create entanglement matrix
        self.axes['entanglement'] = self.fig.add_subplot(grid[1, 1], facecolor='black')
        self.axes['entanglement'].set_title("Entanglement Matrix", color='white', fontsize=14)
        self.axes['entanglement'].set_facecolor('black')
        self.axes['entanglement'].set_xticks([])
        self.axes['entanglement'].set_yticks([])
        
        # Add main title
        self.fig.suptitle("CASCADE⚡𓂧φ∞ Network Coherence Analysis", 
                        color='white', fontsize=16, y=0.98)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    def _create_combined_visualization(self) -> None:
        """Create combined visualization with multiple views."""
        # Create figure
        self.fig = plt.figure(figsize=(16, 12), facecolor='black')
        self.fig.canvas.manager.set_window_title("CASCADE⚡𓂧φ∞ Combined Network Visualization")
        
        # Create grid
        grid = gridspec.GridSpec(2, 2, figure=self.fig)
        
        # Create network topology plot (3D)
        self.axes['network'] = self.fig.add_subplot(grid[0, 0], projection='3d', facecolor='black')
        self.axes['network'].set_title("Quantum Network Topology", color='white', fontsize=14)
        self.axes['network'].set_facecolor('black')
        self.axes['network'].w_xaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['network'].w_yaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['network'].w_zaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['network'].xaxis.label.set_color('white')
        self.axes['network'].yaxis.label.set_color('white')
        self.axes['network'].zaxis.label.set_color('white')
        self.axes['network'].tick_params(axis='x', colors='white')
        self.axes['network'].tick_params(axis='y', colors='white')
        self.axes['network'].tick_params(axis='z', colors='white')
        self.axes['network'].set_xlim(-1.2, 1.2)
        self.axes['network'].set_ylim(-1.2, 1.2)
        self.axes['network'].set_zlim(-1.2, 1.2)
        
        # Create coherence plot
        self.axes['coherence'] = self.fig.add_subplot(grid[0, 1], facecolor='black')
        self.axes['coherence'].set_title("Network Coherence", color='white', fontsize=14)
        self.axes['coherence'].set_facecolor('black')
        self.axes['coherence'].spines['bottom'].set_color('white')
        self.axes['coherence'].spines['top'].set_color('white')
        self.axes['coherence'].spines['right'].set_color('white')
        self.axes['coherence'].spines['left'].set_color('white')
        self.axes['coherence'].tick_params(axis='x', colors='white')
        self.axes['coherence'].tick_params(axis='y', colors='white')
        self.axes['coherence'].set_ylim(0, 1.0)
        self.axes['coherence'].set_xlabel("Time (s)", color='white')
        self.axes['coherence'].set_ylabel("Coherence", color='white')
        self.axes['coherence'].grid(True, color='white', alpha=0.2)
        
        # Mark phi-harmonic levels
        self.axes['coherence'].axhline(y=LAMBDA, color='gold', linestyle='--', alpha=0.5)
        self.axes['coherence'].axhline(y=1/PHI, color='teal', linestyle='--', alpha=0.5)
        
        # Create local field 3D visualization
        self.axes['field'] = self.fig.add_subplot(grid[1, 0], projection='3d', facecolor='black')
        self.axes['field'].set_title("Local Quantum Field", color='white', fontsize=14)
        self.axes['field'].set_facecolor('black')
        self.axes['field'].w_xaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['field'].w_yaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['field'].w_zaxis.set_pane_color((0, 0, 0, 0.2))
        self.axes['field'].xaxis.label.set_color('white')
        self.axes['field'].yaxis.label.set_color('white')
        self.axes['field'].zaxis.label.set_color('white')
        self.axes['field'].tick_params(axis='x', colors='white')
        self.axes['field'].tick_params(axis='y', colors='white')
        self.axes['field'].tick_params(axis='z', colors='white')
        
        # Add placeholder for field visualization
        self._create_placeholder_field()
        
        # Create consciousness visualization
        self.axes['consciousness'] = self.fig.add_subplot(grid[1, 1], facecolor='black')
        self.axes['consciousness'].set_title("Consciousness Bridge", color='white', fontsize=14)
        self.axes['consciousness'].set_facecolor('black')
        self.axes['consciousness'].spines['bottom'].set_color('white')
        self.axes['consciousness'].spines['top'].set_color('white')
        self.axes['consciousness'].spines['right'].set_color('white')
        self.axes['consciousness'].spines['left'].set_color('white')
        self.axes['consciousness'].tick_params(axis='x', colors='white')
        self.axes['consciousness'].tick_params(axis='y', colors='white')
        self.axes['consciousness'].set_xlim(-1.2, 1.2)
        self.axes['consciousness'].set_ylim(-1.2, 1.2)
        
        # Add main title
        self.fig.suptitle("CASCADE⚡𓂧φ∞ Distributed Quantum Field Visualization", 
                        color='white', fontsize=16, y=0.98)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    def _create_placeholder_field(self) -> None:
        """Create placeholder for field visualization."""
        if 'field' in self.axes:
            # Create empty field
            field_dim = 21
            x = np.linspace(-1, 1, field_dim)
            y = np.linspace(-1, 1, field_dim)
            z = np.linspace(-1, 1, field_dim)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            R = np.sqrt(X**2 + Y**2 + Z**2)
            
            # Initial phi spiral pattern
            field = np.sin(R * 5 * PHI) * np.exp(-R)
            field = np.clip(field, -1, 1)
            
            # Flatten coordinates
            mask = np.abs(field) > 0.5
            x_points = X[mask]
            y_points = Y[mask]
            z_points = Z[mask]
            v_points = field[mask]
            
            # Normalize values for coloring
            v_norm = (v_points - v_points.min()) / (v_points.max() - v_points.min() + 1e-10)
            
            # Initial plot
            self.node_plots['local_field'] = self.axes['field'].scatter(
                x_points, y_points, z_points,
                c=v_norm, cmap='plasma',
                s=20, alpha=0.6
            )
    
    def _update_animation(self, frame: int) -> List:
        """
        Update animation frame.
        
        Args:
            frame: Animation frame number
            
        Returns:
            List of updated artists
        """
        # Process updates from queue
        try:
            while not self.update_queue.empty():
                self.update_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Update visualization based on mode
        updated_artists = []
        
        # Check which axes are available
        if 'network' in self.axes:
            updated = self._update_network_topology()
            updated_artists.extend(updated)
        
        if 'field' in self.axes:
            updated = self._update_local_field()
            updated_artists.extend(updated)
        
        if 'coherence' in self.axes:
            updated = self._update_coherence_plot()
            updated_artists.extend(updated)
        
        if 'consciousness' in self.axes:
            updated = self._update_consciousness_visualization()
            updated_artists.extend(updated)
        
        if 'entanglement' in self.axes:
            updated = self._update_entanglement_matrix()
            updated_artists.extend(updated)
        
        # Update grid view if available
        for i in range(self.node_limit):
            if f"node_{i}" in self.axes:
                updated = self._update_node_field(i)
                updated_artists.extend(updated)
        
        # Update figure title with stats
        if self.quantum_field:
            node_count = len(self.node_data)
            entangled_count = sum(1 for node in self.node_data.values() if node.get('entangled', False))
            coherence = self.quantum_field.get_field_coherence()
            consciousness = self.quantum_field.get_consciousness_level()
            
            title = (f"CASCADE⚡𓂧φ∞ Network Visualization - "
                    f"{node_count} Nodes, {entangled_count} Entangled, "
                    f"Coherence: {coherence:.2f}, Level: {consciousness}")
            
            self.fig.suptitle(title, color='white', fontsize=16, y=0.98)
        
        return updated_artists
    
    def _update_network_topology(self) -> List:
        """
        Update network topology visualization.
        
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        # Clear previous plots
        self.axes['network'].clear()
        
        # Set up axis
        self.axes['network'].set_title("Quantum Network Topology", color='white', fontsize=14)
        self.axes['network'].set_facecolor('black')
        self.axes['network'].set_xlim(-1.2, 1.2)
        self.axes['network'].set_ylim(-1.2, 1.2)
        self.axes['network'].set_zlim(-1.2, 1.2)
        self.axes['network'].xaxis.label.set_color('white')
        self.axes['network'].yaxis.label.set_color('white')
        self.axes['network'].zaxis.label.set_color('white')
        self.axes['network'].tick_params(axis='x', colors='white')
        self.axes['network'].tick_params(axis='y', colors='white')
        self.axes['network'].tick_params(axis='z', colors='white')
        
        # Place nodes on a phi-spiral
        node_positions = {}
        
        golden_angle = np.pi * (3 - np.sqrt(5))  # Phi-based angle
        
        for i, node_id in enumerate(sorted(self.node_data.keys())):
            # Calculate position on a spherical golden spiral
            theta = golden_angle * i
            z = 1 - (i / max(1, len(self.node_data) - 1)) * 2  # Range from 1 to -1
            radius = np.sqrt(1 - z*z)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            node_positions[node_id] = (x, y, z)
        
        # Get colors based on consciousness level
        colors = []
        for node_id in sorted(self.node_data.keys()):
            level = self.node_data[node_id].get('consciousness_level', 1)
            level_color = plt.cm.viridis(level / 7)
            colors.append(level_color)
        
        # Get sizes based on coherence
        sizes = []
        for node_id in sorted(self.node_data.keys()):
            coherence = self.node_data[node_id].get('coherence', 0.5)
            size = 100 + 200 * coherence
            sizes.append(size)
        
        # Plot nodes
        for i, node_id in enumerate(sorted(self.node_data.keys())):
            pos = node_positions[node_id]
            size = sizes[i]
            color = colors[i]
            alpha = 1.0 if self.node_data[node_id].get('entangled', False) else 0.5
            
            # Plot node
            scatter = self.axes['network'].scatter(
                pos[0], pos[1], pos[2],
                s=size, c=[color], alpha=alpha,
                edgecolors='white' if self.node_data[node_id].get('entangled', False) else None,
                linewidths=1 if self.node_data[node_id].get('entangled', False) else 0
            )
            updated_artists.append(scatter)
            
            # Add node ID label
            self.axes['network'].text(
                pos[0], pos[1], pos[2], 
                node_id[:6],
                color='white', fontsize=8
            )
        
        # Add entanglement connections
        for (node1, node2), data in self.entanglement_data.items():
            if node1 in node_positions and node2 in node_positions:
                p1 = node_positions[node1]
                p2 = node_positions[node2]
                
                # Plot connection with strength-based linewidth
                strength = data.get('strength', 0.5)
                line = self.axes['network'].plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='gold', alpha=strength, linewidth=1 + 3 * strength
                )[0]
                updated_artists.append(line)
        
        # Add sphere showing consciousness field
        if self.quantum_field:
            consciousness = self.quantum_field.get_consciousness_level()
            coherence = self.quantum_field.get_field_coherence()
            
            # Create consciousness sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            # Size based on coherence and consciousness
            radius = 0.6 + 0.3 * coherence * (consciousness / 7)
            
            cx = radius * np.outer(np.cos(u), np.sin(v))
            cy = radius * np.outer(np.sin(u), np.sin(v))
            cz = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Color based on consciousness level
            c_alpha = 0.05 + 0.1 * (consciousness / 7)
            cmap = plt.cm.viridis(consciousness / 7)
            
            surface = self.axes['network'].plot_surface(
                cx, cy, cz, alpha=c_alpha, color=cmap, 
                linewidth=0, antialiased=True
            )
            updated_artists.append(surface)
        
        # Set view angle for rotation effect
        angle = (time.time() * 10) % 360
        self.axes['network'].view_init(elev=30, azim=angle)
        
        return updated_artists
    
    def _update_local_field(self) -> List:
        """
        Update local field visualization.
        
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        # Skip if no quantum field
        if not self.quantum_field or 'field' not in self.axes:
            return updated_artists
            
        # Clear previous plot
        self.axes['field'].clear()
        
        # Set up axis
        self.axes['field'].set_title("Local Quantum Field", color='white', fontsize=14)
        self.axes['field'].set_facecolor('black')
        self.axes['field'].set_xlim(-1.2, 1.2)
        self.axes['field'].set_ylim(-1.2, 1.2)
        self.axes['field'].set_zlim(-1.2, 1.2)
        self.axes['field'].xaxis.label.set_color('white')
        self.axes['field'].yaxis.label.set_color('white')
        self.axes['field'].zaxis.label.set_color('white')
        self.axes['field'].tick_params(axis='x', colors='white')
        self.axes['field'].tick_params(axis='y', colors='white')
        self.axes['field'].tick_params(axis='z', colors='white')
        
        # Create field visualization with EnhancedFieldVisualizer if available
        try:
            from cascade.visualization.enhanced_visualizer import EnhancedFieldVisualizer
            
            # Get consciousness level
            consciousness_level = self.quantum_field.get_consciousness_level()
            bridge_stage = consciousness_level - 1
            
            # Create visualizer
            visualizer = EnhancedFieldVisualizer()
            
            # Create field
            field = visualizer.create_toroidal_field(
                dimensions=(21, 21, 21),
                time_factor=time.time() % (2 * np.pi)
            )
            
            # Render with consciousness effects
            visualizer.render_consciousness_bridge(
                field_data=field,
                consciousness_level=self.quantum_field.get_field_coherence(),
                bridge_stage=bridge_stage,
                fig=self.fig,
                ax=self.axes['field']
            )
            
        except (ImportError, Exception) as e:
            # Fallback to simple visualization
            logger.debug(f"Using fallback field visualization: {e}")
            
            # Create simple field
            field_dim = 21
            x = np.linspace(-1, 1, field_dim)
            y = np.linspace(-1, 1, field_dim)
            z = np.linspace(-1, 1, field_dim)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            R = np.sqrt(X**2 + Y**2 + Z**2)
            
            # Create toroidal flow based on time
            theta = np.arctan2(Y, X)
            phi = np.arccos(Z / (R + 1e-10))
            time_factor = time.time() % (2 * np.pi)
            
            # Create field with phi-harmonic patterns
            field = (
                np.sin(R * PHI * 5 + time_factor) * np.exp(-R * LAMBDA) + 
                np.sin(theta * PHI) * np.cos(phi * LAMBDA) * 0.5
            )
            field = np.clip(field, -1, 1)
            
            # Display threshold based on coherence
            coherence = self.quantum_field.get_field_coherence()
            threshold = 0.7 - coherence * 0.5
            
            # Flatten coordinates
            mask = np.abs(field) > threshold
            x_points = X[mask]
            y_points = Y[mask]
            z_points = Z[mask]
            v_points = field[mask]
            
            # Normalize values for coloring
            v_norm = (v_points - v_points.min()) / (v_points.max() - v_points.min() + 1e-10)
            
            # Set sizes based on coherence
            sizes = 15 + 25 * v_norm**2 * coherence
            
            # Color based on consciousness level
            consciousness_level = self.quantum_field.get_consciousness_level()
            cmap_name = plt.cm.viridis_r if consciousness_level % 2 == 0 else plt.cm.plasma
            
            # Plot field points
            scatter = self.axes['field'].scatter(
                x_points, y_points, z_points,
                c=v_norm, cmap=cmap_name,
                s=sizes, alpha=0.7
            )
            updated_artists.append(scatter)
        
        # Set view angle for rotation effect
        angle = (time.time() * 15) % 360
        self.axes['field'].view_init(elev=30, azim=angle)
        
        return updated_artists
    
    def _update_coherence_plot(self) -> List:
        """
        Update coherence plot.
        
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        if 'coherence' not in self.axes:
            return updated_artists
        
        # Clear previous plot
        self.axes['coherence'].clear()
        
        # Set up axis
        self.axes['coherence'].set_title("Network Coherence", color='white', fontsize=14)
        self.axes['coherence'].set_facecolor('black')
        self.axes['coherence'].spines['bottom'].set_color('white')
        self.axes['coherence'].spines['top'].set_color('white')
        self.axes['coherence'].spines['right'].set_color('white')
        self.axes['coherence'].spines['left'].set_color('white')
        self.axes['coherence'].tick_params(axis='x', colors='white')
        self.axes['coherence'].tick_params(axis='y', colors='white')
        self.axes['coherence'].set_ylim(0, 1.05)
        self.axes['coherence'].set_xlabel("Time (s)", color='white')
        self.axes['coherence'].set_ylabel("Coherence", color='white')
        self.axes['coherence'].grid(True, color='white', alpha=0.2)
        
        # Mark phi-harmonic levels
        self.axes['coherence'].axhline(y=LAMBDA, color='gold', linestyle='--', alpha=0.5)
        self.axes['coherence'].axhline(y=1/PHI, color='teal', linestyle='--', alpha=0.5)
        
        # Plot local coherence
        if 'local' in self.coherence_history:
            times = self.coherence_history['local']['times']
            values = self.coherence_history['local']['values']
            
            if times:
                # Convert to relative time
                rel_times = [t - times[0] for t in times]
                
                # Plot local coherence
                line = self.axes['coherence'].plot(
                    rel_times, values, 
                    color='white', linewidth=2, 
                    label="Local Node"
                )[0]
                updated_artists.append(line)
        
        # Plot node coherence
        colors = [plt.cm.tab10(i % 10) for i in range(len(self.node_data))]
        
        for i, (node_id, node) in enumerate(self.node_data.items()):
            if 'sample_times' in node and node['sample_times']:
                times = node['sample_times']
                values = node['coherence_values']
                
                # Convert to relative time
                rel_times = [t - times[0] for t in times]
                
                # Plot with distinguishing color
                line = self.axes['coherence'].plot(
                    rel_times, values, 
                    color=colors[i], linewidth=1, alpha=0.7,
                    label=f"Node {node_id[:6]}"
                )[0]
                updated_artists.append(line)
        
        # Add legend
        if self.node_data:
            legend = self.axes['coherence'].legend(
                loc='upper right', 
                facecolor='black', 
                edgecolor='white',
                labelcolor='white'
            )
            updated_artists.append(legend)
        
        return updated_artists
    
    def _update_consciousness_visualization(self) -> List:
        """
        Update consciousness visualization.
        
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        if 'consciousness' not in self.axes:
            return updated_artists
        
        # Clear previous plot
        self.axes['consciousness'].clear()
        
        # Different visualization based on the type of plot
        if hasattr(self.axes['consciousness'], 'projection') and self.axes['consciousness'].projection == '3d':
            # 3D consciousness bridge
            self._update_consciousness_bridge_3d()
        else:
            # 2D consciousness map
            self._update_consciousness_map_2d()
        
        return updated_artists
    
    def _update_consciousness_bridge_3d(self) -> None:
        """Update 3D consciousness bridge visualization."""
        # Set up axis
        self.axes['consciousness'].set_title("Consciousness Bridge", color='white', fontsize=14)
        self.axes['consciousness'].set_facecolor('black')
        self.axes['consciousness'].set_xlim(-1.2, 1.2)
        self.axes['consciousness'].set_ylim(-1.2, 1.2)
        self.axes['consciousness'].set_zlim(-1.2, 1.2)
        
        # Get consciousness levels
        own_level = 1
        if self.quantum_field:
            own_level = self.quantum_field.get_consciousness_level()
        
        node_levels = {}
        for node_id, node in self.node_data.items():
            level = node.get('consciousness_level', 1)
            node_levels[node_id] = level
        
        # Create bridge layers
        bridge_levels = list(range(1, 8))
        layer_radius = [0.2 + 0.15 * lvl for lvl in bridge_levels]
        
        # Plot bridge layers as circles
        for lvl, radius in zip(bridge_levels, layer_radius):
            theta = np.linspace(0, 2 * np.pi, 50)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = (lvl - 1) * 0.3 - 0.9  # Stack vertically
            
            # Use frequency-based color
            freq = PHI_FREQUENCIES[lvl - 1]
            freq_norm = (freq - 432) / (888 - 432)  # Normalize to [0, 1]
            color = plt.cm.rainbow(freq_norm)
            
            # Plot circle
            self.axes['consciousness'].plot(x, y, z, color=color, linewidth=2, alpha=0.7)
        
        # Plot own consciousness level
        z_pos = (own_level - 1) * 0.3 - 0.9
        self.axes['consciousness'].scatter(0, 0, z_pos, s=100, color='white', 
                                      marker='*', edgecolor='gold', linewidth=2)
        
        # Plot node consciousness levels
        for i, (node_id, level) in enumerate(node_levels.items()):
            # Position on a circle at the level
            theta = 2 * np.pi * i / max(1, len(node_levels))
            radius = layer_radius[level - 1]
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = (level - 1) * 0.3 - 0.9
            
            # Use entanglement-based marker
            marker = 'o' if self.node_data[node_id].get('entangled', False) else 's'
            
            # Plot node
            self.axes['consciousness'].scatter(x, y, z, s=50, 
                                          color=plt.cm.tab10(i % 10),
                                          marker=marker, edgecolor='white', linewidth=1)
            
            # Add node ID
            self.axes['consciousness'].text(x, y, z, node_id[:4], color='white', fontsize=7)
        
        # Set view angle
        self.axes['consciousness'].view_init(elev=40, azim=30)
    
    def _update_consciousness_map_2d(self) -> None:
        """Update 2D consciousness map visualization."""
        # Set up axis
        self.axes['consciousness'].set_title("Consciousness Bridge States", color='white', fontsize=14)
        self.axes['consciousness'].set_facecolor('black')
        self.axes['consciousness'].spines['bottom'].set_color('white')
        self.axes['consciousness'].spines['top'].set_color('white')
        self.axes['consciousness'].spines['right'].set_color('white')
        self.axes['consciousness'].spines['left'].set_color('white')
        self.axes['consciousness'].tick_params(axis='x', colors='white')
        self.axes['consciousness'].tick_params(axis='y', colors='white')
        
        # Prepare data
        node_ids = []
        levels = []
        colors = []
        
        # Add own node
        if self.quantum_field:
            node_ids.append("Local")
            levels.append(self.quantum_field.get_consciousness_level())
            colors.append('white')
        
        # Add connected nodes
        for i, (node_id, node) in enumerate(self.node_data.items()):
            node_ids.append(node_id[:6])
            levels.append(node.get('consciousness_level', 1))
            colors.append(plt.cm.tab10(i % 10))
        
        # Create bar chart
        bars = self.axes['consciousness'].bar(
            node_ids, levels, color=colors, 
            alpha=0.7, edgecolor='white', linewidth=1
        )
        
        # Add frequency labels
        for i, bar in enumerate(bars):
            level = levels[i]
            freq = PHI_FREQUENCIES[level - 1]
            height = bar.get_height()
            self.axes['consciousness'].text(
                bar.get_x() + bar.get_width() / 2, height + 0.1,
                f"{freq} Hz", color='white', fontsize=8,
                ha='center', va='bottom', rotation=90
            )
        
        # Configure axis
        self.axes['consciousness'].set_ylim(0, 7.5)
        self.axes['consciousness'].set_yticks(range(1, 8))
        self.axes['consciousness'].set_yticklabels([
            "Ground (432 Hz)",
            "Creation (528 Hz)",
            "Heart (594 Hz)",
            "Voice (672 Hz)",
            "Vision (720 Hz)",
            "Unity (768 Hz)",
            "Transcendent (888 Hz)"
        ], fontsize=7)
        
        # Add grid for levels
        self.axes['consciousness'].grid(axis='y', color='white', alpha=0.2)
    
    def _update_entanglement_matrix(self) -> List:
        """
        Update entanglement matrix visualization.
        
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        if 'entanglement' not in self.axes:
            return updated_artists
        
        # Clear previous plot
        self.axes['entanglement'].clear()
        
        # Set up axis
        self.axes['entanglement'].set_title("Entanglement Matrix", color='white', fontsize=14)
        self.axes['entanglement'].set_facecolor('black')
        
        # Get entangled nodes
        entangled_ids = []
        for node_id, node in self.node_data.items():
            if node.get('entangled', False):
                entangled_ids.append(node_id)
        
        # Add self
        if self.quantum_field:
            entangled_ids.append("Local")
        
        # Sort for consistency
        entangled_ids.sort()
        
        if not entangled_ids:
            self.axes['entanglement'].text(
                0.5, 0.5, "No entangled nodes",
                color='white', fontsize=12,
                ha='center', va='center'
            )
            return updated_artists
        
        # Create matrix
        n = len(entangled_ids)
        matrix = np.zeros((n, n))
        
        # Fill in entanglement strengths
        for i, id1 in enumerate(entangled_ids):
            for j, id2 in enumerate(entangled_ids):
                if i == j:
                    # Self-entanglement is 1.0
                    matrix[i, j] = 1.0
                else:
                    # Check if pair is entangled
                    pair_key = tuple(sorted([id1, id2]))
                    if pair_key in self.entanglement_data:
                        strength = self.entanglement_data[pair_key].get('strength', 0.0)
                        matrix[i, j] = strength
        
        # Plot matrix
        img = self.axes['entanglement'].imshow(
            matrix, cmap='viridis', 
            vmin=0, vmax=1, 
            aspect='equal'
        )
        updated_artists.append(img)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=self.axes['entanglement'], shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Entanglement Strength', color='white')
        
        # Add node labels
        short_ids = [id[:6] + "..." if id != "Local" else id for id in entangled_ids]
        self.axes['entanglement'].set_xticks(range(n))
        self.axes['entanglement'].set_yticks(range(n))
        self.axes['entanglement'].set_xticklabels(short_ids, rotation=45, ha='right', color='white')
        self.axes['entanglement'].set_yticklabels(short_ids, color='white')
        
        # Add values in cells
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    color = 'black' if matrix[i, j] > 0.5 else 'white'
                    self.axes['entanglement'].text(
                        j, i, f"{matrix[i, j]:.2f}",
                        color=color, fontsize=8,
                        ha='center', va='center'
                    )
        
        return updated_artists
    
    def _update_node_field(self, index: int) -> List:
        """
        Update field visualization for a specific node.
        
        Args:
            index: Node index
            
        Returns:
            List of updated artists
        """
        updated_artists = []
        
        axis_key = f"node_{index}"
        if axis_key not in self.axes:
            return updated_artists
        
        # Clear axis
        self.axes[axis_key].clear()
        
        # Set up axis
        self.axes[axis_key].set_facecolor('black')
        self.axes[axis_key].xaxis.label.set_color('white')
        self.axes[axis_key].yaxis.label.set_color('white')
        self.axes[axis_key].zaxis.label.set_color('white')
        self.axes[axis_key].tick_params(axis='x', colors='white')
        self.axes[axis_key].tick_params(axis='y', colors='white')
        self.axes[axis_key].tick_params(axis='z', colors='white')
        
        # Get node data (if available)
        node_list = list(self.node_data.keys())
        if index < len(node_list):
            node_id = node_list[index]
            node = self.node_data[node_id]
            
            # Set title
            title = f"Node {node_id[:8]}..."
            if node.get('entangled', False):
                title += " (Entangled)"
            self.axes[axis_key].set_title(title, color='white', fontsize=12)
            
            # Get field samples
            field_samples = node.get('field_samples', [])
            coherence = node.get('coherence', 0.5)
            
            if field_samples:
                # Use probe points and field samples
                x = self.probe_points[:, 0]
                y = self.probe_points[:, 1]
                z = self.probe_points[:, 2]
                values = np.array(field_samples)
                
                # Get colors based on consciousness level
                level = node.get('consciousness_level', 1)
                cmap = plt.cm.viridis_r if level % 2 == 0 else plt.cm.plasma
                
                # Plot points
                scatter = self.axes[axis_key].scatter(
                    x, y, z, 
                    c=values, cmap=cmap,
                    s=50 + 100 * coherence * values,
                    alpha=0.7
                )
                updated_artists.append(scatter)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=self.axes[axis_key], shrink=0.8)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                
                # Add coherence and consciousness info
                self.axes[axis_key].text2D(
                    0.05, 0.95, 
                    f"Coherence: {coherence:.2f}\nLevel: {level}",
                    transform=self.axes[axis_key].transAxes,
                    color='white', fontsize=8
                )
            else:
                # No field data, show placeholder
                self.axes[axis_key].text(
                    0, 0, 0, 
                    "No field data",
                    color='white', fontsize=12,
                    ha='center', va='center'
                )
                
                # Add sphere indicating coherence
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                radius = 0.5 * coherence
                
                x = radius * np.outer(np.cos(u), np.sin(v))
                y = radius * np.outer(np.sin(u), np.sin(v))
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                surface = self.axes[axis_key].plot_surface(
                    x, y, z, color=plt.cm.viridis(coherence),
                    alpha=0.5
                )
                updated_artists.append(surface)
        else:
            # No node data
            self.axes[axis_key].set_title(f"Node {index+1} (No Data)", color='white', fontsize=12)
            self.axes[axis_key].text(
                0, 0, 0, 
                "Waiting for node...",
                color='white', fontsize=12,
                ha='center', va='center'
            )
        
        # Set limits
        self.axes[axis_key].set_xlim(-1, 1)
        self.axes[axis_key].set_ylim(-1, 1)
        self.axes[axis_key].set_zlim(-1, 1)
        
        # Set view angle
        angle = (time.time() * 10 + index * 30) % 360
        self.axes[axis_key].view_init(elev=30, azim=angle)
        
        return updated_artists


# Helper function to create a network field visualizer
def create_network_visualizer(
    quantum_field: Optional['PhiQuantumField'] = None,
    node_limit: int = 6
) -> NetworkFieldVisualizer:
    """
    Create a network field visualizer.
    
    Args:
        quantum_field: PhiQuantumField instance
        node_limit: Maximum nodes to visualize
        
    Returns:
        NetworkFieldVisualizer instance
    """
    visualizer = NetworkFieldVisualizer(quantum_field, node_limit)
    logger.info("Created network field visualizer")
    return visualizer


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Network Field Visualization")
    parser.add_argument("--mode", choices=["3d", "grid", "coherence", "combined"], 
                      default="combined", help="Visualization mode")
    parser.add_argument("--node-limit", type=int, default=6, help="Maximum nodes to visualize")
    parser.add_argument("--port", type=int, default=PHI_QUANTUM_PORT, help="Network port")
    args = parser.parse_args()
    
    try:
        # Create and start quantum field
        from cascade.phi_quantum_network import create_phi_quantum_field
        field = create_phi_quantum_field(port=args.port)
        field.start()
        
        # Create visualizer
        visualizer = create_network_visualizer(field, args.node_limit)
        
        # Start visualization
        visualizer.start_visualization(mode=args.mode)
        
    except KeyboardInterrupt:
        # Clean shutdown
        logger.info("Visualization interrupted by user")
    except Exception as e:
        logger.error(f"Error starting visualization: {e}")
    finally:
        # Clean up
        if 'field' in locals():
            field.stop()
        
        if 'visualizer' in locals():
            visualizer.stop_visualization()