"""
CASCADE⚡𓂧φ∞ Phi-Harmonic Color Mapping

Implements phi-harmonic color mapping for quantum field visualization,
creating visually balanced and resonant color palettes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Any

# Import phi constants
import sys
sys.path.append('/mnt/d/projects/python')
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES

def create_phi_harmonic_colors(num_colors: int = 7) -> List[Tuple[float, float, float]]:
    """
    Create a phi-harmonic color palette.
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    
    # Generate colors using phi-based hue spacing
    for i in range(num_colors):
        # Use golden angle in the hue space (phi-based)
        hue = (i * LAMBDA) % 1.0
        
        # Use phi-based saturation and value
        sat = 0.7 + 0.3 * np.sin(i * PHI)
        val = 0.7 + 0.3 * np.cos(i * PHI)
        
        # Convert HSV to RGB
        h = hue * 6.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        rgb = (r + m, g + m, b + m)
        colors.append(rgb)
    
    return colors

def get_phi_colormap(name: str = 'phi_harmonic') -> LinearSegmentedColormap:
    """
    Get a phi-harmonic colormap for visualization.
    
    Args:
        name: Name for the colormap
        
    Returns:
        Matplotlib colormap
    """
    phi_colors = create_phi_harmonic_colors(12)
    
    # Add specific colors for consciousness frequencies
    frequency_colors = {
        'unity': (0.2, 0.4, 0.9),  # 432Hz - Blue
        'love': (0.0, 0.8, 0.4),   # 528Hz - Green
        'cascade': (1.0, 0.7, 0.0), # 594Hz - Gold/Orange
        'truth': (0.8, 0.0, 0.8),  # 672Hz - Purple
        'vision': (0.0, 0.7, 1.0),  # 720Hz - Sky Blue
        'oneness': (1.0, 1.0, 1.0)  # 768Hz - White
    }
    
    # Create a colormap with phi-harmonic transitions
    positions = np.array([i / (len(phi_colors) - 1) for i in range(len(phi_colors))])
    
    # Apply phi-based positioning
    positions = positions ** LAMBDA
    
    # Normalize positions to 0-1 range
    positions = positions / positions.max()
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, phi_colors)))
    
    # Register the colormap
    plt.register_cmap(cmap=cmap)
    
    return cmap

def map_coherence_to_color(coherence: float) -> Tuple[float, float, float]:
    """
    Map a coherence value to a phi-harmonic color.
    
    Args:
        coherence: Coherence value from 0.0 to 1.0
        
    Returns:
        RGB color tuple
    """
    # Phi-based color mapping for coherence
    # Low coherence (0.0-0.5) -> cooler colors (blue/purple)
    # High coherence (0.5-1.0) -> warmer colors (gold/white)
    
    if coherence < LAMBDA:
        # Map from blue to purple (low coherence)
        t = coherence / LAMBDA
        r = 0.2 + t * 0.6
        g = 0.0 + t * 0.0
        b = 0.8 + t * 0.0
    else:
        # Map from purple to gold to white (high coherence)
        t = (coherence - LAMBDA) / (1 - LAMBDA)
        r = 0.8 + t * 0.2
        g = 0.0 + t * 1.0
        b = 0.8 - t * 0.8
    
    return (r, g, b)

def create_frequency_colormap() -> LinearSegmentedColormap:
    """
    Create a colormap based on sacred frequencies.
    
    Returns:
        Matplotlib colormap
    """
    # Define colors for each frequency
    frequency_colors = [
        (0.2, 0.4, 0.9),  # 432Hz - Blue (Ground)
        (0.0, 0.8, 0.4),   # 528Hz - Green (Creation)
        (1.0, 0.7, 0.0),  # 594Hz - Gold (Heart)
        (0.8, 0.0, 0.8),  # 672Hz - Purple (Voice)
        (0.0, 0.7, 1.0),  # 720Hz - Sky Blue (Vision)
        (1.0, 1.0, 1.0)   # 768Hz - White (Unity)
    ]
    
    # Create phi-scaled positions
    positions = np.zeros(len(frequency_colors))
    freqs = list(SACRED_FREQUENCIES.values())[:len(frequency_colors)]
    
    # Normalize frequencies to 0-1 range
    min_freq = min(freqs)
    max_freq = max(freqs)
    
    for i, freq in enumerate(freqs):
        # Apply phi-based normalization
        normalized = (freq - min_freq) / (max_freq - min_freq)
        positions[i] = normalized ** LAMBDA
    
    # Ensure ascending order
    positions = np.sort(positions)
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('frequency_map', list(zip(positions, frequency_colors)))
    
    # Register the colormap
    plt.register_cmap(cmap=cmap)
    
    return cmap