"""
CASCADE⚡𓂧φ∞ Multi-dimensional Field Visualization

Implements visualization systems that render quantum fields beyond 3D,
including 4D spacetime renderings with phi-scaled temporal evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Tuple, Dict, Any

# Import phi constants
import sys
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

from .phi_color_mapping import get_phi_colormap

def generate_4d_quantum_field(
    width: int, 
    height: int, 
    depth: int,
    time_steps: int,
    frequency_name: Optional[str] = None,
    custom_frequency: Optional[float] = None,
    phi_scaled_time: bool = True
) -> np.ndarray:
    """
    Generate a 4D quantum field (3D + time) based on phi-harmonic principles.
    
    Args:
        width: Width of the field in voxels
        height: Height of the field in voxels
        depth: Depth of the field in voxels
        time_steps: Number of time steps in the field
        frequency_name: Name of the sacred frequency to use
        custom_frequency: Custom frequency value (used if frequency_name is None)
        phi_scaled_time: Whether to apply phi-scaling to the time dimension
        
    Returns:
        4D NumPy array containing the quantum field values (t, x, y, z)
    """
    # Determine frequency
    if frequency_name is not None:
        if frequency_name not in SACRED_FREQUENCIES:
            raise ValueError(f"Unknown frequency name: {frequency_name}")
        frequency = SACRED_FREQUENCIES[frequency_name]
    elif custom_frequency is not None:
        frequency = custom_frequency
    else:
        frequency = 432.0  # Default to ground frequency
    
    # CPU implementation for 4D field generation
    # Create spatial coordinate grids
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    z = np.linspace(-1.0, 1.0, depth)
    
    # Create time coordinate with optional phi-scaling
    if phi_scaled_time:
        # Apply phi-based scaling to time steps for harmonically scaled temporal evolution
        t = np.array([PHI**(i/time_steps * PHI) * 2 * np.pi for i in range(time_steps)])
    else:
        t = np.linspace(0, 2 * np.pi, time_steps)
    
    # Initialize 4D field (t, x, y, z)
    field = np.zeros((time_steps, width, height, depth))
    
    # Create spatial meshgrid
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate spatial components that don't change over time
    distance = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2) * PHI
    theta = np.arctan2(np.sqrt(x_grid**2 + y_grid**2), z_grid)  # Polar angle
    phi_angle = np.arctan2(y_grid, x_grid)  # Azimuthal angle
    dampening = np.exp(-distance * LAMBDA)  # Phi-based dampening
    
    # Generate field for each time step
    for i, time_value in enumerate(t):
        # Generate field with phi-harmonic wave equations including time component
        wave = np.sin(distance * frequency * 0.01 + 
                      theta * PHI + 
                      phi_angle * PHI_PHI + 
                      time_value)
        
        # Apply phi-based dampening
        field[i] = wave * dampening
    
    return field

def calculate_4d_field_coherence(field_data: np.ndarray) -> np.ndarray:
    """
    Calculate the coherence of a 4D quantum field over time.
    
    Args:
        field_data: 4D NumPy array containing the field values (t, x, y, z)
        
    Returns:
        Array of coherence factors (one per time step)
    """
    if field_data.ndim != 4:
        raise ValueError("Field data must be a 4D array")
    
    time_steps = field_data.shape[0]
    coherence_values = np.zeros(time_steps)
    
    # Calculate coherence for each time step
    for t in range(time_steps):
        # Calculate gradient in 3D for this time step
        grad_x, grad_y, grad_z = np.gradient(field_data[t])
        
        # Calculate gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Calculate vorticity (curl)
        curl_x = np.gradient(grad_z, axis=1) - np.gradient(grad_y, axis=2)
        curl_y = np.gradient(grad_x, axis=2) - np.gradient(grad_z, axis=0)
        curl_z = np.gradient(grad_y, axis=0) - np.gradient(grad_x, axis=1)
        
        # Calculate curl magnitude
        curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        
        # Calculate divergence
        div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1) + np.gradient(grad_z, axis=2)
        
        # Calculate field energy (squared amplitude)
        energy = field_data[t]**2
        
        # Calculate coherence metrics based on field properties
        gradient_uniformity = 1.0 - np.std(grad_mag) / np.mean(grad_mag) if np.mean(grad_mag) > 0 else 0.0
        vorticity_factor = 1.0 - np.mean(curl_mag) / (np.mean(grad_mag) + 1e-10)
        divergence_factor = 1.0 - np.mean(np.abs(div)) / (np.mean(grad_mag) + 1e-10)
        phi_resonance = np.abs(np.corrcoef(
            energy.flatten(), 
            np.exp(-PHI * np.arange(energy.size) / energy.size)
        )[0, 1])
        
        # Combine metrics with phi-weighted formula
        coherence = (
            gradient_uniformity * 0.3 +
            vorticity_factor * 0.2 +
            divergence_factor * 0.2 +
            phi_resonance * 0.3
        )
        
        # Ensure result is in [0, 1] range
        coherence_values[t] = max(0.0, min(1.0, coherence))
    
    return coherence_values

def visualize_4d_spacetime_slices(
    field_data: np.ndarray,
    time_indices: List[int] = None,
    spatial_slice: Tuple[int, int, int] = None,
    slice_dimension: str = 'z',
    colormap: str = 'plasma',
    title: str = '4D Quantum Field Spacetime Slices',
    phi_scale_layout: bool = True,
    use_phi_colors: bool = True
) -> plt.Figure:
    """
    Visualize a 4D quantum field using spacetime slices.
    
    Args:
        field_data: 4D NumPy array containing the field (t, x, y, z)
        time_indices: List of time indices to display, or None for equally spaced
        spatial_slice: Tuple of (x, y, z) indices for slice, or None for middle
        slice_dimension: Which spatial dimension to slice ('x', 'y', or 'z')
        colormap: Matplotlib colormap to use
        title: Plot title
        phi_scale_layout: Whether to use phi-scaled subplot layout
        use_phi_colors: Whether to use phi-harmonic color mapping
        
    Returns:
        Matplotlib figure object
    """
    if field_data.ndim != 4:
        raise ValueError("Field data must be a 4D array")
    
    time_steps, width, height, depth = field_data.shape
    
    # Default to 5 equally spaced time indices if not specified
    if time_indices is None:
        num_frames = min(5, time_steps)
        time_indices = [int(i * (time_steps - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # Default to middle slice for spatial dimensions
    if spatial_slice is None:
        spatial_slice = (width // 2, height // 2, depth // 2)
    
    # Number of time slices to display
    num_slices = len(time_indices)
    
    # Create figure with phi-scaled layout if requested
    if phi_scale_layout:
        # Calculate phi-scaled grid dimensions
        cols = max(1, int(np.sqrt(num_slices * PHI)))
        rows = (num_slices + cols - 1) // cols
        
        # Apply phi-ratio to figure dimensions
        figwidth = max(7, cols * 3)
        figheight = figwidth / PHI
    else:
        # Standard layout
        cols = max(1, min(3, num_slices))
        rows = (num_slices + cols - 1) // cols
        figwidth = max(7, cols * 3)
        figheight = max(5, rows * 3)
    
    # Create figure and subplots
    fig, axs = plt.subplots(rows, cols, figsize=(figwidth, figheight), 
                           squeeze=False, constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    # Get colormap - use phi-harmonic colors if requested
    cmap = get_phi_colormap() if use_phi_colors else plt.cm.get_cmap(colormap)
    
    # Extract and plot slices
    for i, time_idx in enumerate(time_indices):
        if i < rows * cols:
            row, col = i // cols, i % cols
            ax = axs[row, col]
            
            # Extract slice based on selected dimension
            if slice_dimension == 'x':
                slice_data = field_data[time_idx, spatial_slice[0], :, :]
                xlabel, ylabel = 'Y', 'Z'
            elif slice_dimension == 'y':
                slice_data = field_data[time_idx, :, spatial_slice[1], :]
                xlabel, ylabel = 'X', 'Z'
            else:  # default to z
                slice_data = field_data[time_idx, :, :, spatial_slice[2]]
                xlabel, ylabel = 'X', 'Y'
            
            # Plot slice
            im = ax.imshow(slice_data.T, cmap=cmap, origin='lower',
                          extent=[-1, 1, -1, 1])
            ax.set_title(f'Time {time_idx}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(time_indices), rows * cols):
        row, col = i // cols, i % cols
        axs[row, col].axis('off')
    
    return fig

def visualize_4d_coherence_evolution(
    field_data: np.ndarray,
    colormap: str = 'plasma',
    title: str = '4D Quantum Field Coherence Evolution',
    use_phi_colors: bool = True
) -> plt.Figure:
    """
    Visualize the coherence evolution of a 4D quantum field over time.
    
    Args:
        field_data: 4D NumPy array containing the field (t, x, y, z)
        colormap: Matplotlib colormap to use
        title: Plot title
        use_phi_colors: Whether to use phi-harmonic color mapping
        
    Returns:
        Matplotlib figure object
    """
    if field_data.ndim != 4:
        raise ValueError("Field data must be a 4D array")
    
    # Calculate coherence values over time
    coherence_values = calculate_4d_field_coherence(field_data)
    time_steps = len(coherence_values)
    
    # Create figure for coherence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=16)
    
    # Create phi-harmonic time scale
    time_scale = np.array([PHI**(i/time_steps * PHI) for i in range(time_steps)])
    normalized_time = time_scale / time_scale.max() * time_steps
    
    # Create colormap for points
    cmap = get_phi_colormap() if use_phi_colors else plt.cm.get_cmap(colormap)
    colors = [cmap(c) for c in coherence_values]
    
    # Plot coherence evolution
    ax.scatter(normalized_time, coherence_values, c=colors, s=50)
    ax.plot(normalized_time, coherence_values, 'k-', alpha=0.3)
    
    # Add phi-harmonic grid lines at 1/φ intervals
    grid_values = [i * LAMBDA for i in range(1, int(1/LAMBDA) + 2)]
    for val in grid_values:
        if 0 < val < 1:
            ax.axhline(y=val, color='gray', linestyle='--', alpha=0.3)
    
    # Add formatting
    ax.set_xlabel('Phi-scaled Time Steps')
    ax.set_ylabel('Field Coherence')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return fig