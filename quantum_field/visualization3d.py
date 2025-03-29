"""
3D visualization module for quantum fields.

This module provides tools for generating, manipulating, and visualizing
3D quantum fields based on phi-harmonic principles.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from .constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

# Try to import optional 3D visualization libraries
try:
    import plotly.graph_objs as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def generate_3d_quantum_field(
    width: int, 
    height: int, 
    depth: int, 
    frequency_name: Optional[str] = None,
    time_factor: float = 0.0,
    custom_frequency: Optional[float] = None,
    backend = None
) -> np.ndarray:
    """
    Generate a 3D quantum field based on phi-harmonic principles.
    
    Args:
        width: Width of the field in voxels
        height: Height of the field in voxels
        depth: Depth of the field in voxels
        frequency_name: Name of the sacred frequency to use
        time_factor: Time evolution factor (0.0 to 2π)
        custom_frequency: Custom frequency value (used if frequency_name is None)
        backend: Optional backend to use for acceleration
        
    Returns:
        3D NumPy array containing the quantum field values
    """
    # Determine frequency
    if frequency_name is not None:
        if frequency_name not in SACRED_FREQUENCIES:
            raise ValueError(f"Unknown frequency name: {frequency_name}")
        frequency = SACRED_FREQUENCIES[frequency_name]
    elif custom_frequency is not None:
        frequency = custom_frequency
    else:
        raise ValueError("Either frequency_name or custom_frequency must be provided")
    
    # Check if backend is provided and has 3D capabilities
    if backend is not None:
        capabilities = backend.get_capabilities()
        if capabilities.get("3d_fields", False):
            # Use backend's 3D field generation if available
            try:
                return backend.generate_3d_quantum_field(
                    width, height, depth, frequency_name, time_factor, custom_frequency
                )
            except (AttributeError, NotImplementedError):
                # Fall back to CPU implementation
                pass
    
    # CPU implementation for 3D field generation
    # Create coordinate grids
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    z = np.linspace(-1.0, 1.0, depth)
    
    # Create meshgrid
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate field using phi-harmonic principles
    # 3D distance from center
    distance = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2) * PHI
    
    # 3D angular components
    theta = np.arctan2(np.sqrt(x_grid**2 + y_grid**2), z_grid)  # Polar angle
    phi = np.arctan2(y_grid, x_grid)  # Azimuthal angle
    
    # Generate field with phi-harmonic wave equations
    field = np.sin(distance * frequency * 0.01 + 
                   theta * PHI + 
                   phi * PHI_PHI + 
                   time_factor * PHI_PHI)
    
    # Apply phi-based dampening
    dampening = np.exp(-distance * LAMBDA)
    
    # Combine wave and dampening
    field = field * dampening
    
    return field


def calculate_3d_field_coherence(field_data: np.ndarray) -> float:
    """
    Calculate the coherence of a 3D quantum field.
    
    Args:
        field_data: 3D NumPy array containing the field values
        
    Returns:
        Coherence factor between 0.0 and 1.0
    """
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Calculate gradient in 3D
    grad_x, grad_y, grad_z = np.gradient(field_data)
    
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
    energy = field_data**2
    
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
    coherence = max(0.0, min(1.0, coherence))
    
    return coherence


def extract_isosurface(
    field_data: np.ndarray, 
    iso_value: float = 0.0,
    smooth_iterations: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an isosurface from a 3D quantum field.
    
    Args:
        field_data: 3D NumPy array containing the field
        iso_value: Value at which to extract the isosurface (-1.0 to 1.0)
        smooth_iterations: Number of smoothing iterations to apply
        
    Returns:
        Tuple of (vertices, triangles) defining the isosurface mesh
    """
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista is required for isosurface extraction. "
            "Install with: pip install pyvista"
        )
    
    # Convert field to PyVista uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = np.array(field_data.shape) + 1
    grid.spacing = (2.0/field_data.shape[0], 2.0/field_data.shape[1], 2.0/field_data.shape[2])
    grid.origin = (-1.0, -1.0, -1.0)
    
    # Add field data
    grid.cell_data["field"] = field_data.flatten(order='F')
    
    # Extract isosurface
    surface = grid.contour([iso_value], scalars="field")
    
    # Apply smoothing if requested
    if smooth_iterations > 0:
        surface = surface.smooth(n_iter=smooth_iterations)
    
    # Extract vertices and faces
    vertices = np.array(surface.points)
    triangles = np.array(surface.faces.reshape(-1, 4))[:, 1:4]
    
    return vertices, triangles


def visualize_3d_slices(
    field_data: np.ndarray,
    slices: Tuple[int, int, int] = None,
    colormap: str = 'viridis',
    title: str = '3D Quantum Field Slices'
) -> plt.Figure:
    """
    Visualize a 3D quantum field using orthogonal slices.
    
    Args:
        field_data: 3D NumPy array containing the field
        slices: Tuple of (x, y, z) indices for slices, or None for middle slices
        colormap: Matplotlib colormap to use
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Default to middle slices
    if slices is None:
        slices = (
            field_data.shape[0] // 2,
            field_data.shape[1] // 2,
            field_data.shape[2] // 2
        )
    
    # Create slices
    x_slice = field_data[slices[0], :, :]
    y_slice = field_data[:, slices[1], :]
    z_slice = field_data[:, :, slices[2]]
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Plot slices
    im0 = axs[0].imshow(x_slice.T, cmap=colormap, origin='lower',
                       extent=[-1, 1, -1, 1])
    axs[0].set_title(f'X Slice (x={slices[0]})')
    axs[0].set_xlabel('Y')
    axs[0].set_ylabel('Z')
    
    im1 = axs[1].imshow(y_slice.T, cmap=colormap, origin='lower',
                       extent=[-1, 1, -1, 1])
    axs[1].set_title(f'Y Slice (y={slices[1]})')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Z')
    
    im2 = axs[2].imshow(z_slice.T, cmap=colormap, origin='lower',
                       extent=[-1, 1, -1, 1])
    axs[2].set_title(f'Z Slice (z={slices[2]})')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    
    # Add colorbar
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def visualize_3d_volume(
    field_data: np.ndarray,
    colormap: str = 'viridis',
    opacity: Optional[Union[float, np.ndarray]] = None,
    title: str = '3D Quantum Field Volume Rendering'
) -> Any:
    """
    Visualize a 3D quantum field using volume rendering with Plotly.
    
    Args:
        field_data: 3D NumPy array containing the field
        colormap: Colormap name 
        opacity: Opacity value or array for volume rendering
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for 3D volume rendering. "
            "Install with: pip install plotly"
        )
    
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Default opacity scale - use a non-linear scale to highlight features
    if opacity is None:
        # Scale field to [0, 1] range for opacity mapping
        field_range = field_data.max() - field_data.min()
        if field_range > 0:
            opacity = np.clip((field_data - field_data.min()) / field_range, 0, 1)**2
        else:
            opacity = 0.5
    
    # Create plotly volume
    fig = go.Figure(data=go.Volume(
        x=np.linspace(-1, 1, field_data.shape[0]),
        y=np.linspace(-1, 1, field_data.shape[1]),
        z=np.linspace(-1, 1, field_data.shape[2]),
        value=field_data.flatten(),
        opacity=opacity.flatten() if isinstance(opacity, np.ndarray) else opacity,
        colorscale=colormap,
        surface_count=25,  # Isosurface count for performance/quality tradeoff
        colorbar=dict(
            title='Field Value',
            thickness=20,
        ),
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X'),
            yaxis=dict(range=[-1, 1], title='Y'),
            zaxis=dict(range=[-1, 1], title='Z'),
            aspectratio=dict(x=1, y=1, z=1),
        ),
    )
    
    return fig


def visualize_3d_isosurface(
    field_data: np.ndarray,
    iso_values: List[float] = None,
    colormap: str = 'viridis',
    opacity: float = 0.7,
    title: str = '3D Quantum Field Isosurfaces'
) -> Any:
    """
    Visualize a 3D quantum field using isosurfaces with Plotly.
    
    Args:
        field_data: 3D NumPy array containing the field
        iso_values: List of isovalues to visualize
        colormap: Colormap name
        opacity: Opacity value for isosurfaces
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for 3D isosurface visualization. "
            "Install with: pip install plotly"
        )
    
    if field_data.ndim != 3:
        raise ValueError("Field data must be a 3D array")
    
    # Default iso-values: distribute between min and max
    if iso_values is None:
        field_min, field_max = field_data.min(), field_data.max()
        iso_values = np.linspace(field_min + 0.2 * (field_max - field_min),
                                 field_max - 0.2 * (field_max - field_min),
                                 3)
    
    # Create figure
    fig = go.Figure()
    
    # Get colormap from matplotlib
    cmap = plt.cm.get_cmap(colormap)
    
    # Add isosurfaces
    for i, iso_value in enumerate(iso_values):
        # Normalize isovalue for color mapping
        norm_value = (iso_value - field_data.min()) / (field_data.max() - field_data.min())
        color = 'rgb' + str(tuple(int(c * 255) for c in cmap(norm_value)[:3]))
        
        # Extract isosurface vertices and triangles
        if HAS_PYVISTA:
            try:
                vertices, triangles = extract_isosurface(field_data, iso_value, smooth_iterations=1)
                
                # Add isosurface as mesh
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color=color,
                    opacity=opacity,
                    name=f'Iso {iso_value:.2f}'
                ))
                continue  # Skip fallback if isosurface was extracted successfully
            except Exception:
                # Fall back to plotly's built-in isosurface
                pass
        
        # Fallback: Use plotly's built-in isosurface
        fig.add_trace(go.Isosurface(
            x=np.linspace(-1, 1, field_data.shape[0]),
            y=np.linspace(-1, 1, field_data.shape[1]),
            z=np.linspace(-1, 1, field_data.shape[2]),
            value=field_data.flatten(),
            isomin=iso_value,
            isomax=iso_value,
            color=color,
            opacity=opacity,
            name=f'Iso {iso_value:.2f}'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X'),
            yaxis=dict(range=[-1, 1], title='Y'),
            zaxis=dict(range=[-1, 1], title='Z'),
            aspectratio=dict(x=1, y=1, z=1),
        ),
    )
    
    return fig


def animate_3d_field(
    width: int,
    height: int,
    depth: int,
    frequency_name: str,
    frames: int = 30,
    colormap: str = 'viridis',
    mode: str = 'isosurface',
    output_path: Optional[str] = None
) -> None:
    """
    Create an animation of a 3D quantum field evolving over time.
    
    Args:
        width: Width of the field in voxels
        height: Height of the field in voxels
        depth: Depth of the field in voxels
        frequency_name: Name of the sacred frequency to use
        frames: Number of frames in the animation
        colormap: Colormap name
        mode: Visualization mode ('isosurface', 'volume', or 'slices')
        output_path: Path to save the animation (default: None, display only)
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for 3D field animation. "
            "Install with: pip install plotly"
        )
    
    # Create figures for each frame
    figures = []
    for i in range(frames):
        # Calculate time factor (0 to 2π over the animation)
        time_factor = i * 2 * np.pi / frames
        
        # Generate field for this time step
        field = generate_3d_quantum_field(
            width, height, depth, 
            frequency_name=frequency_name,
            time_factor=time_factor
        )
        
        # Visualize based on selected mode
        if mode == 'isosurface':
            fig = visualize_3d_isosurface(
                field,
                title=f'3D Quantum Field Isosurface (t={time_factor:.2f})',
                colormap=colormap
            )
        elif mode == 'volume':
            fig = visualize_3d_volume(
                field,
                title=f'3D Quantum Field Volume (t={time_factor:.2f})',
                colormap=colormap
            )
        elif mode == 'slices':
            fig = visualize_3d_slices(
                field,
                title=f'3D Quantum Field Slices (t={time_factor:.2f})',
                colormap=colormap
            )
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        figures.append(fig)
    
    # Create animation
    if output_path is not None:
        if mode == 'slices':
            # For matplotlib animations
            import matplotlib.animation as animation
            ani = animation.ArtistAnimation(figures[0], 
                                           [fig.get_children() for fig in figures], 
                                           interval=100)
            ani.save(output_path)
        else:
            # For plotly animations
            import plotly.io as pio
            
            # Configure frames
            frames = [go.Frame(
                data=fig.data,
                layout=fig.layout,
                name=f"frame{i}"
            ) for i, fig in enumerate(figures)]
            
            # Create figure with frames
            fig = figures[0]
            fig.frames = frames
            
            # Add animation controls
            fig.update_layout(
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 100, "redraw": True}}]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}}]
                        }
                    ]
                }]
            )
            
            # Save animation
            pio.write_html(fig, output_path)
    
    return figures[0]