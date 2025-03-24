"""
CASCADE⚡𓂧φ∞ Optimized Network Renderer

This module provides optimized rendering techniques for network field visualization,
improving performance for large networks and high-dimensional quantum fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable, Generator

# Try to import numba for JIT compilation
try:
    import numba
    has_numba = True
except ImportError:
    has_numba = False
    logging.warning("Numba not available, falling back to standard implementations")

# Try to import networkx for network algorithms
try:
    import networkx as nx
    has_networkx = True
except ImportError:
    has_networkx = False
    logging.warning("NetworkX not available, some network optimizations disabled")

# Import local modules
try:
    from cascade.phi_quantum_network import PHI, LAMBDA, PHI_PHI
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895
    LAMBDA = 0.618033988749895
    PHI_PHI = PHI ** PHI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("optimized_renderer")


# Optimize field sampling with numba if available
if has_numba:
    @numba.jit(nopython=True)
    def sample_field_at_points(field: np.ndarray, 
                              points: np.ndarray) -> np.ndarray:
        """
        Sample field values at specified points using fast compiled code.
        
        Args:
            field: 3D NumPy array of field values
            points: Nx3 array of sampling coordinates in [-1,1] range
            
        Returns:
            Array of sampled values
        """
        # Get field dimensions
        nx, ny, nz = field.shape
        
        # Create output array
        num_points = points.shape[0]
        samples = np.zeros(num_points)
        
        # Sample each point
        for i in range(num_points):
            # Convert from [-1,1] to field indices
            x = int((points[i, 0] + 1) / 2 * (nx - 1))
            y = int((points[i, 1] + 1) / 2 * (ny - 1))
            z = int((points[i, 2] + 1) / 2 * (nz - 1))
            
            # Get value (with bounds checking)
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                samples[i] = field[x, y, z]
        
        return samples
else:
    # Fallback implementation without numba
    def sample_field_at_points(field: np.ndarray, 
                              points: np.ndarray) -> np.ndarray:
        """
        Sample field values at specified points (standard implementation).
        
        Args:
            field: 3D NumPy array of field values
            points: Nx3 array of sampling coordinates in [-1,1] range
            
        Returns:
            Array of sampled values
        """
        # Get field dimensions
        nx, ny, nz = field.shape
        
        # Create output array
        num_points = points.shape[0]
        samples = np.zeros(num_points)
        
        # Sample each point
        for i in range(num_points):
            # Convert from [-1,1] to field indices
            x = int((points[i, 0] + 1) / 2 * (nx - 1))
            y = int((points[i, 1] + 1) / 2 * (ny - 1))
            z = int((points[i, 2] + 1) / 2 * (nz - 1))
            
            # Get value (with bounds checking)
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                samples[i] = field[x, y, z]
        
        return samples


def optimize_node_positions(node_ids: List[str], 
                           entangled_pairs: List[Tuple[str, str]],
                           use_graphviz: bool = False) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimize node positions for visualization using network algorithms.
    
    Args:
        node_ids: List of node IDs
        entangled_pairs: List of (node1, node2) pairs that are entangled
        use_graphviz: Whether to use GraphViz for layout (requires pygraphviz)
        
    Returns:
        Dictionary mapping node IDs to (x,y,z) positions
    """
    if not has_networkx:
        # Fallback to phi-spiral layout
        return create_phi_spiral_layout(node_ids)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for node_id in node_ids:
        G.add_node(node_id)
    
    # Add edges for entangled pairs
    for node1, node2 in entangled_pairs:
        if node1 in node_ids and node2 in node_ids:
            G.add_edge(node1, node2)
    
    # Choose layout algorithm based on graph size and structure
    if len(node_ids) <= 10:
        # For small graphs, use force-directed layout
        if use_graphviz and nx.nx_agraph:
            pos = nx.nx_agraph.graphviz_layout(G)
        else:
            pos = nx.spring_layout(G, dim=3, seed=PHI * 100)
    else:
        # For larger graphs, use spectral layout for better structure
        if G.number_of_edges() > 0:
            pos = nx.spectral_layout(G, dim=3)
        else:
            pos = nx.circular_layout(G, dim=3)
    
    # Normalize positions to [-1,1] range
    max_pos = max(max(abs(x), abs(y), abs(z)) for node_id, (x, y, z) in pos.items())
    if max_pos > 0:
        positions = {}
        for node_id, (x, y, z) in pos.items():
            positions[node_id] = (x/max_pos, y/max_pos, z/max_pos)
    else:
        # Fallback to phi-spiral layout
        positions = create_phi_spiral_layout(node_ids)
    
    return positions


def create_phi_spiral_layout(node_ids: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Create a layout with nodes arranged on a phi-harmonic spiral.
    
    Args:
        node_ids: List of node IDs
        
    Returns:
        Dictionary mapping node IDs to (x,y,z) positions
    """
    positions = {}
    golden_angle = np.pi * (3 - np.sqrt(5))  # Phi-based angle
    
    for i, node_id in enumerate(sorted(node_ids)):
        # Calculate position on a spherical golden spiral
        theta = golden_angle * i
        z = 1 - (i / max(1, len(node_ids) - 1)) * 2  # Range from 1 to -1
        radius = np.sqrt(1 - z*z)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        positions[node_id] = (x, y, z)
    
    return positions


def get_adaptive_sampling_grid(field_dimensions: Tuple[int, int, int], 
                              target_points: int = 1000,
                              focus_regions: Optional[List[Tuple[float, float, float, float]]] = None) -> np.ndarray:
    """
    Create an adaptive sampling grid for large field visualization.
    
    Args:
        field_dimensions: Dimensions of the quantum field
        target_points: Target number of sampling points
        focus_regions: Optional list of (x,y,z,radius) focal regions with higher sampling density
        
    Returns:
        Nx3 array of sampling coordinates in [-1,1] range
    """
    # Base sampling using phi-based spacing
    w, h, d = field_dimensions
    max_dim = max(w, h, d)
    
    # Determine sampling density based on field size
    base_density = int(np.power(target_points, 1/3))
    
    # Create adaptive grid with phi-based spacing
    x = np.linspace(-1, 1, base_density)
    y = np.linspace(-1, 1, base_density)
    z = np.linspace(-1, 1, base_density)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten to points
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # If focus regions are specified, add more points there
    if focus_regions:
        focus_points = []
        for x, y, z, radius in focus_regions:
            # Sample more densely in focus region
            density = int(base_density * PHI)  # More points in focus region
            r = np.linspace(0, radius, density)
            phi = np.linspace(0, 2*np.pi, density)
            theta = np.linspace(0, np.pi, density)
            
            # Create spherical coordinates
            R, PHI, THETA = np.meshgrid(r, phi, theta, indexing='ij')
            
            # Convert to Cartesian
            X_focus = x + R * np.sin(THETA) * np.cos(PHI)
            Y_focus = y + R * np.sin(THETA) * np.sin(PHI)
            Z_focus = z + R * np.cos(THETA)
            
            # Filter to points within [-1,1] range
            mask = ((X_focus >= -1) & (X_focus <= 1) &
                   (Y_focus >= -1) & (Y_focus <= 1) &
                   (Z_focus >= -1) & (Z_focus <= 1))
            
            # Add to focus points
            points_focus = np.column_stack([
                X_focus[mask].flatten(), 
                Y_focus[mask].flatten(), 
                Z_focus[mask].flatten()
            ])
            
            focus_points.append(points_focus)
        
        # Combine all points
        if focus_points:
            points = np.vstack([points] + focus_points)
    
    # If we have too many points, subsample
    if len(points) > target_points:
        # Use phi-based subsampling
        indices = np.linspace(0, len(points)-1, target_points, dtype=int)
        points = points[indices]
    
    return points


class CachingFieldSampler:
    """Caches field samples to reduce redundant computation."""
    
    def __init__(self, cache_size: int = 10):
        """
        Initialize the caching field sampler.
        
        Args:
            cache_size: Number of fields to cache
        """
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
    
    def sample_field(self, 
                    field: np.ndarray, 
                    points: np.ndarray,
                    field_id: str) -> np.ndarray:
        """
        Sample field values with caching.
        
        Args:
            field: 3D NumPy array of field values
            points: Nx3 array of sampling coordinates
            field_id: Unique identifier for this field
            
        Returns:
            Array of sampled values
        """
        # Check if field is in cache
        cache_key = (field_id, hash(points.tobytes()))
        
        if cache_key in self.cache:
            # Move to front of cache order
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            
            return self.cache[cache_key]
        
        # Sample field
        samples = sample_field_at_points(field, points)
        
        # Add to cache
        self.cache[cache_key] = samples
        self.cache_order.append(cache_key)
        
        # Remove oldest entry if cache is full
        if len(self.cache_order) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        return samples


def generate_node_positions_incremental(existing_nodes: Dict[str, Tuple[float, float, float]],
                                      new_nodes: List[str],
                                      entangled_pairs: List[Tuple[str, str]]) -> Dict[str, Tuple[float, float, float]]:
    """
    Generate positions for new nodes while preserving existing node positions.
    
    Args:
        existing_nodes: Dictionary of existing node positions
        new_nodes: List of new node IDs
        entangled_pairs: List of (node1, node2) pairs that are entangled
        
    Returns:
        Updated dictionary with all node positions
    """
    if not new_nodes:
        return existing_nodes
    
    if not has_networkx:
        # Fallback to phi-spiral for new nodes
        golden_angle = np.pi * (3 - np.sqrt(5))
        positions = existing_nodes.copy()
        
        # Start index based on existing nodes
        start_idx = len(existing_nodes)
        
        for i, node_id in enumerate(new_nodes):
            idx = start_idx + i
            theta = golden_angle * idx
            z = 1 - (idx / (start_idx + len(new_nodes) - 1)) * 2
            radius = np.sqrt(1 - z*z)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            positions[node_id] = (x, y, z)
        
        return positions
    
    # Create graph with existing nodes fixed
    G = nx.Graph()
    
    # Add all nodes
    for node_id, pos in existing_nodes.items():
        G.add_node(node_id, pos=pos)
    
    for node_id in new_nodes:
        G.add_node(node_id)
    
    # Add edges for entangled pairs
    for node1, node2 in entangled_pairs:
        if node1 in G.nodes and node2 in G.nodes:
            G.add_edge(node1, node2)
    
    # Get existing positions
    pos = {n: existing_nodes[n] for n in G.nodes if n in existing_nodes}
    
    # Set fixed parameter for existing nodes
    fixed = [n for n in G.nodes if n in existing_nodes]
    
    # Use spring layout with fixed positions
    if not new_nodes:
        return pos
    
    # Run spring layout for a few iterations to place new nodes
    new_pos = nx.spring_layout(
        G, dim=3, pos=pos, fixed=fixed, 
        k=1/np.sqrt(len(G.nodes)), 
        iterations=50,
        seed=int(time.time() * PHI)
    )
    
    # Check if any positions are outside [-1,1] range
    max_pos = max(max(abs(x), abs(y), abs(z)) 
                 for node_id, (x, y, z) in new_pos.items())
    
    if max_pos > 1:
        # Scale back to [-1,1] range
        for node_id in new_pos:
            if node_id not in fixed:
                x, y, z = new_pos[node_id]
                new_pos[node_id] = (x/max_pos, y/max_pos, z/max_pos)
    
    return new_pos


class ParallelNodeSampler:
    """Samples node field data in parallel for better performance."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel node sampler.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.queue = queue.Queue()
        self.results = {}
        self.running = False
        self.workers = []
    
    def start(self):
        """Start worker threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for _ in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
    
    def stop(self):
        """Stop worker threads."""
        self.running = False
        
        # Wait for workers to terminate
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
    
    def _worker_loop(self):
        """Worker thread function."""
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task, args = self.queue.get(timeout=0.5)
                    
                    # Execute task
                    result = task(*args)
                    
                    # Store result
                    task_id = id(task) ^ hash(args)
                    self.results[task_id] = result
                    
                    # Mark as done
                    self.queue.task_done()
                    
                except queue.Empty:
                    pass
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in worker thread: {e}")
    
    def sample_node_field(self, node_id: str, 
                         request_func: Callable, 
                         process_func: Callable) -> int:
        """
        Queue a node field sampling task.
        
        Args:
            node_id: Node ID to sample
            request_func: Function to request field data
            process_func: Function to process field data
            
        Returns:
            Task ID for retrieving result
        """
        task_id = id(request_func) ^ hash(node_id)
        
        # Add task to queue
        self.queue.put((self._sample_task, (node_id, request_func, process_func)))
        
        return task_id
    
    def _sample_task(self, node_id: str, request_func: Callable, process_func: Callable) -> Any:
        """
        Sample node field task implementation.
        
        Args:
            node_id: Node ID to sample
            request_func: Function to request field data
            process_func: Function to process field data
            
        Returns:
            Processed field data
        """
        # Request field data
        field_data = request_func(node_id)
        
        if field_data:
            # Process field data
            return process_func(field_data)
        
        return None
    
    def get_result(self, task_id: int, timeout: float = 0.0) -> Optional[Any]:
        """
        Get result of a sampling task.
        
        Args:
            task_id: Task ID from sample_node_field
            timeout: Timeout in seconds (0 for no wait)
            
        Returns:
            Task result or None if not available
        """
        if timeout > 0:
            # Wait for result
            start = time.time()
            while time.time() - start < timeout:
                if task_id in self.results:
                    return self.results.pop(task_id)
                time.sleep(0.01)
        
        # Check for immediate result
        if task_id in self.results:
            return self.results.pop(task_id)
        
        return None


def create_phi_weighted_animation(frames: int) -> List[Dict[str, Any]]:
    """
    Create phi-weighted animation parameters.
    
    Args:
        frames: Number of animation frames
        
    Returns:
        List of animation parameter dictionaries
    """
    # Create frame parameters with phi-weighted timing
    frame_params = []
    
    # Use phi-weighted time distribution
    for i in range(frames):
        t = i / (frames - 1)
        
        # Apply phi-weighted time acceleration
        if t < LAMBDA:
            # First part: slower
            t_adjusted = t * LAMBDA
        else:
            # Second part: faster
            t_adjusted = LAMBDA + (t - LAMBDA) * (1/LAMBDA)
        
        # Create rotation parameters
        rotation = 360 * t_adjusted
        view_angle = 30 + 10 * np.sin(t_adjusted * np.pi * 2)
        
        # Pulse effect parameters
        pulse = 0.5 + 0.5 * np.sin(t_adjusted * np.pi * PHI)
        
        frame_params.append({
            'rotation': rotation,
            'view_angle': view_angle,
            'pulse': pulse,
            'time': t_adjusted
        })
    
    return frame_params


def get_entanglement_matrix(node_ids: List[str],
                           entangled_pairs: List[Tuple[str, str]],
                           coherence_values: Dict[str, float]) -> np.ndarray:
    """
    Create optimized entanglement matrix for visualization.
    
    Args:
        node_ids: List of node IDs
        entangled_pairs: List of (node1, node2) pairs that are entangled
        coherence_values: Dictionary mapping node IDs to coherence values
        
    Returns:
        2D NumPy array of entanglement strengths
    """
    n = len(node_ids)
    matrix = np.zeros((n, n))
    
    # Create lookup for node indices
    node_lookup = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Fill diagonal with own coherence
    for i, node_id in enumerate(node_ids):
        matrix[i, i] = coherence_values.get(node_id, 1.0)
    
    # Fill entanglement strengths
    for node1, node2 in entangled_pairs:
        if node1 in node_lookup and node2 in node_lookup:
            # Get indices
            i = node_lookup[node1]
            j = node_lookup[node2]
            
            # Calculate entanglement strength as minimum coherence
            strength = min(
                coherence_values.get(node1, 0.0),
                coherence_values.get(node2, 0.0)
            )
            
            # Set symmetric values
            matrix[i, j] = strength
            matrix[j, i] = strength
    
    return matrix