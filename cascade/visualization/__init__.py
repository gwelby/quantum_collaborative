"""
CASCADE⚡𓂧φ∞ Visualization Module

This module contains visualization tools for the CASCADE quantum field system,
including multi-dimensional field rendering, network field visualization,
and phi-harmonic color mapping.
"""

from .field_visualizer import render_quantum_field_3d, render_field_isosurface
from .multidimensional import visualize_4d_spacetime_slices, visualize_4d_coherence_evolution
from .phi_color_mapping import create_phi_harmonic_colors, get_phi_colormap

# Import enhanced visualizer if available
try:
    from .enhanced_visualizer import (
        EnhancedFieldVisualizer, 
        render_phi_harmonic_mandala,
        render_sacred_geometry_grid
    )
except ImportError:
    pass

# Import network visualizer
try:
    from .network_field_visualizer import (
        NetworkFieldVisualizer,
        create_network_visualizer
    )
except ImportError:
    pass

__all__ = [
    # Basic field visualization
    'render_quantum_field_3d',
    'render_field_isosurface',
    
    # Multidimensional visualization
    'visualize_4d_spacetime_slices',
    'visualize_4d_coherence_evolution',
    
    # Phi-harmonic colors
    'create_phi_harmonic_colors',
    'get_phi_colormap',
    
    # Enhanced visualization
    'EnhancedFieldVisualizer',
    'render_phi_harmonic_mandala',
    'render_sacred_geometry_grid',
    
    # Network visualization
    'NetworkFieldVisualizer',
    'create_network_visualizer',
]