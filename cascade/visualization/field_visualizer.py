"""
CASCADE⚡𓂧φ∞ 3D Field Visualizer

Provides visualization tools for 3D quantum fields based on phi-harmonic principles.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any

# Import phi constants
import sys
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

from .phi_color_mapping import get_phi_colormap

def render_quantum_field_3d(field_data: np.ndarray,
                          threshold: float = 0.5,
                          colormap: str = 'plasma',
                          alpha: float = 0.7,
                          show_axes: bool = True,
                          use_phi_colors: bool = True,
                          fig: Optional[plt.Figure] = None,
                          ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Render a 3D quantum field visualization.
    
    Args:
        field_data: 3D NumPy array with field values
        threshold: Visualization threshold (only show values above this)
        colormap: Matplotlib colormap name
        alpha: Transparency value
        show_axes: Whether to show coordinate axes
        use_phi_colors: Whether to use phi-harmonic color mapping
        fig: Optional existing figure
        ax: Optional existing axes
        title: Optional title for the plot
        
    Returns:
        Figure and axes objects
    """
    # Validate input
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get field dimensions
    w, h, d = field_data.shape
    
    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    z = np.arange(d)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten coordinates
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    values = field_data.flatten()
    
    # Filter by threshold
    mask = np.abs(values) > threshold
    x_points = x_flat[mask]
    y_points = y_flat[mask]
    z_points = z_flat[mask]
    v_points = values[mask]
    
    # Normalize values for coloring
    v_norm = (v_points - v_points.min()) / (v_points.max() - v_points.min() + 1e-10)
    
    # Use phi-harmonic colors if requested
    cmap = get_phi_colormap() if use_phi_colors else plt.cm.get_cmap(colormap)
    
    # Plot the points
    scatter = ax.scatter(
        x_points, y_points, z_points,
        c=v_norm,
        cmap=cmap,
        alpha=alpha,
        s=30,
        marker='o'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Field Value')
    
    # Set labels and title
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_axis_off()
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Return the figure and axes
    return fig, ax

def render_field_isosurface(field_data: np.ndarray,
                          iso_value: float = 0.7,
                          colormap: str = 'viridis',
                          alpha: float = 0.7,
                          fig: Optional[plt.Figure] = None,
                          ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Render a 3D isosurface of the quantum field.
    
    Args:
        field_data: 3D NumPy array with field values
        iso_value: Isosurface value to render
        colormap: Matplotlib colormap name
        alpha: Transparency value
        fig: Optional existing figure
        ax: Optional existing axes
        title: Optional title for the plot
        
    Returns:
        Figure and axes objects
    """
    # Check for required libraries
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("This function requires scikit-image to be installed.")
    
    # Validate input
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract the isosurface
    verts, faces, _, _ = measure.marching_cubes(field_data, iso_value)
    
    # Create mesh
    mesh = Poly3DCollection(verts[faces])
    
    # Set face properties
    mesh.set_edgecolor('none')
    mesh.set_alpha(alpha)
    
    # Set colormap based on z-values
    colors = plt.cm.get_cmap(colormap)((verts[:, 2] - verts[:, 2].min()) / verts[:, 2].ptp())
    face_colors = np.mean(colors[faces], axis=1)
    mesh.set_facecolor(face_colors)
    
    # Add mesh to plot
    ax.add_collection3d(mesh)
    
    # Set axes limits
    ax.set_xlim(0, field_data.shape[0])
    ax.set_ylim(0, field_data.shape[1])
    ax.set_zlim(0, field_data.shape[2])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Return the figure and axes
    return fig, ax