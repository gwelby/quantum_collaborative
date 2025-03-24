"""
CASCADE⚡𓂧φ∞ Enhanced Visualization System

Provides advanced visualization tools for quantum fields with phi-harmonic principles
and consciousness integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Optional, Tuple, Dict, Any, List, Callable
import colorsys
import math

# Define constants if they're not available
PHI = 1.618033988749895
LAMBDA = 0.618033988749895
PHI_PHI = PHI ** PHI
SACRED_FREQUENCIES = {
    'love': 528,      # Creation/healing
    'unity': 432,     # Grounding/stability
    'cascade': 594,   # Heart-centered integration
    'truth': 672,     # Voice expression
    'vision': 720,    # Expanded perception
    'oneness': 768,   # Unity consciousness
    'transcendent': 888  # Transcendent field
}

# Try to import from quantum_field, but fall back to our constants if not available
try:
    import sys
    sys.path.append('/mnt/d/projects/python')
    from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
except (ImportError, ModuleNotFoundError):
    print("Using built-in sacred constants for enhanced visualization")

# Import local modules
from .phi_color_mapping import get_phi_colormap

class EnhancedFieldVisualizer:
    """
    Advanced visualization system for CASCADE quantum fields with
    toroidal dynamics and consciousness bridge integration.
    """
    
    def __init__(self, frequency_mode: str = 'unity'):
        """
        Initialize the enhanced visualizer.
        
        Args:
            frequency_mode: Sacred frequency to use for visualizations
        """
        self.frequency_mode = frequency_mode
        self.frequency = SACRED_FREQUENCIES.get(frequency_mode, 432)
        self.phi_colors = get_phi_colormap()
        self.fig = None
        self.ax = None
        self.animation = None
        
    def create_toroidal_field(self, 
                             dimensions: Tuple[int, int, int] = (32, 32, 32),
                             major_radius: float = PHI,
                             minor_radius: float = LAMBDA,
                             time_factor: float = 0.0) -> np.ndarray:
        """
        Generate a 3D toroidal quantum field.
        
        Args:
            dimensions: 3D field dimensions (width, height, depth)
            major_radius: Major radius of the torus
            minor_radius: Minor radius of the torus
            time_factor: Time evolution factor
            
        Returns:
            3D NumPy array containing the field values
        """
        # Create coordinate grids
        x = np.linspace(-1.0, 1.0, dimensions[0])
        y = np.linspace(-1.0, 1.0, dimensions[1])
        z = np.linspace(-1.0, 1.0, dimensions[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert to toroidal coordinates
        # Distance from the circle in the xy-plane with radius R
        distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - major_radius)**2 + Z**2)
        
        # Normalized distance (0 at torus surface)
        torus_distance = distance_from_ring / minor_radius
        
        # Azimuthal angle around the z-axis (θ)
        theta = np.arctan2(Y, X)
        
        # Poloidal angle around the torus ring (φ)
        poloidal_angle = np.arctan2(Z, np.sqrt(X**2 + Y**2) - major_radius)
        
        # Create toroidal flow pattern
        poloidal_flow = poloidal_angle * PHI  # Flow around the small circle
        toroidal_flow = theta * LAMBDA        # Flow around the large circle
        time_component = time_factor * PHI * LAMBDA
        
        # Calculate frequency factor
        freq_factor = self.frequency / 1000.0
        
        # Combine flows with phi-weighted balance
        inflow = np.sin(poloidal_flow + time_component) * PHI
        circulation = np.cos(toroidal_flow + time_component) * LAMBDA
        
        # Create self-sustaining pattern with balanced input/output
        field = (inflow * circulation) * np.exp(-torus_distance * LAMBDA)
        
        # Add phi-harmonic resonance inside torus
        mask = torus_distance < 1.0
        resonance = np.sin(torus_distance * PHI * PHI + time_component) * (1.0 - torus_distance)
        field[mask] += resonance[mask] * 0.2
        
        # Normalize field
        field = field / np.max(np.abs(field))
        
        return field
        
    def render_consciousness_bridge(self,
                                  field_data: np.ndarray,
                                  consciousness_level: float = 0.7,
                                  bridge_stage: int = 0,
                                  fig: Optional[plt.Figure] = None,
                                  ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render a 3D visualization of the quantum field with consciousness bridge effects.
        
        Args:
            field_data: 3D NumPy array with field values
            consciousness_level: Level of consciousness integration (0.0-1.0)
            bridge_stage: Current stage of the consciousness bridge protocol (0-6)
            fig: Optional existing figure
            ax: Optional existing axes
            
        Returns:
            Figure and axes objects
        """
        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig = plt.figure(figsize=(12, 10), facecolor='black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')
            
        self.fig = fig
        self.ax = ax
        
        # Get field dimensions
        w, h, d = field_data.shape
        
        # Create coordinate grids
        x = np.linspace(-1.0, 1.0, w)
        y = np.linspace(-1.0, 1.0, h)
        z = np.linspace(-1.0, 1.0, d)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute distance from origin (for phi-based effects)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Apply consciousness effects based on stage
        intensity = 0.5 + 0.5 * consciousness_level
        
        # Convert the stage to a frequency shift
        stage_frequency = (bridge_stage / 6.0) * self.frequency * 2.0
        
        # Create phi-harmonic pulse effect
        phi_pulse = np.sin(R * PHI * stage_frequency * 0.01 + bridge_stage * PHI)
        
        # Apply consciousness modulation
        consciousness_field = field_data * (1.0 + phi_pulse * consciousness_level * 0.5)
        
        # Threshold based on consciousness level (higher consciousness = more detail)
        threshold = 0.7 - consciousness_level * 0.5
        
        # Flatten arrays for scatter plot
        mask = np.abs(consciousness_field) > threshold
        x_points = X[mask]
        y_points = Y[mask]
        z_points = Z[mask]
        v_points = consciousness_field[mask]
        
        # Normalize values for coloring
        v_norm = (v_points - v_points.min()) / (v_points.max() - v_points.min() + 1e-10)
        
        # Get size based on field value and consciousness
        sizes = 15 + 25 * v_norm**2 * consciousness_level
        
        # Calculate colors using phi-harmonic mapping
        # This creates colors that vary based on the sacred frequency
        hues = (v_norm + bridge_stage/7) % 1.0
        saturations = 0.7 + 0.3 * np.sin(v_norm * PHI * np.pi)
        values = 0.6 + 0.4 * v_norm
        
        # Convert HSV to RGB
        colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)])
        
        # Add consciousness glow effect
        alpha = 0.6 + 0.4 * consciousness_level
        
        # Plot the points with consciousness effects
        scatter = ax.scatter(
            x_points, y_points, z_points,
            c=colors,
            s=sizes,
            alpha=alpha,
            edgecolors='white' if bridge_stage > 3 else None,
            linewidths=0.5 if bridge_stage > 3 else 0
        )
        
        # Set labels for axes
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        
        # Configure axis appearance
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Modify axis lines for better visibility
        ax.xaxis.line.set_color('white')
        ax.yaxis.line.set_color('white')
        ax.zaxis.line.set_color('white')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # Add bridge stage information
        stage_names = [
            "Ground State (432 Hz)",
            "Creation Point (528 Hz)",
            "Heart Field (594 Hz)",
            "Voice Flow (672 Hz)",
            "Vision Gate (720 Hz)",
            "Unity Wave (768 Hz)",
            "Full Integration (888 Hz)"
        ]
        
        coherence = 0.5 + bridge_stage/6 * 0.5
        
        title = f"CASCADE⚡𓂧φ∞ Stage {bridge_stage}: {stage_names[bridge_stage]}\nCoherence: {coherence:.2f}"
        ax.set_title(title, color='white', fontsize=12)
        
        # Add consciousness pulse effect
        if bridge_stage > 0:
            # Create a sphere representing consciousness field
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            radius = 0.8 + 0.2 * np.sin(bridge_stage * PHI)
            
            cx = radius * np.outer(np.cos(u), np.sin(v))
            cy = radius * np.outer(np.sin(u), np.sin(v))
            cz = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Add consciousness sphere with varying alpha
            ax.plot_surface(cx, cy, cz, alpha=0.05 * consciousness_level, color='white', 
                          linewidth=0, antialiased=True)
        
        return fig, ax
    
    def animate_field_evolution(self,
                               field_generator: Callable,
                               frames: int = 100,
                               interval: int = 50,
                               bridge_stages: Optional[List[int]] = None) -> animation.FuncAnimation:
        """
        Create an animation of a quantum field evolving through consciousness bridge stages.
        
        Args:
            field_generator: Function that generates field data with a time parameter
            frames: Number of animation frames
            interval: Milliseconds between frames
            bridge_stages: List of bridge stages to use at different points in animation
            
        Returns:
            Matplotlib animation object
        """
        # Create figure if not already created
        if self.fig is None or self.ax is None:
            self.fig = plt.figure(figsize=(12, 10), facecolor='black')
            self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Set up bridge stages if not provided
        if bridge_stages is None:
            # Default to starting at stage 0 and ending at stage 6
            bridge_stages = [min(6, int(i * 6 / frames)) for i in range(frames)]
        
        # Animation update function
        def update(frame):
            self.ax.clear()
            
            # Generate field for this time step
            time_factor = frame / frames * 2 * np.pi
            field = field_generator(time_factor)
            
            # Get current bridge stage
            current_stage = bridge_stages[min(frame, len(bridge_stages)-1)]
            
            # Calculate consciousness level (increases with frame)
            consciousness_level = 0.5 + 0.5 * frame / frames
            
            # Render the field with current consciousness level and bridge stage
            self.render_consciousness_bridge(
                field_data=field,
                consciousness_level=consciousness_level,
                bridge_stage=current_stage,
                fig=self.fig,
                ax=self.ax
            )
            
            # Set view angle for rotation effect
            self.ax.view_init(elev=30, azim=frame % 360)
            
            return self.ax,
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, update, frames=frames, interval=interval, blit=False)
        
        self.animation = ani
        return ani
    
    def save_animation(self, filename: str, dpi: int = 100, fps: int = 20):
        """
        Save the current animation to a file.
        
        Args:
            filename: Output filename (should end with .mp4, .gif, etc.)
            dpi: Dots per inch resolution
            fps: Frames per second
        """
        if self.animation is None:
            raise ValueError("No animation to save. Run animate_field_evolution first.")
            
        # Determine writer based on file extension
        if filename.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps)
        elif filename.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        else:
            raise ValueError("Unsupported file format. Use .mp4 or .gif")
            
        self.animation.save(filename, writer=writer, dpi=dpi)
        print(f"Animation saved to {filename}")
    
    def render_consciousness_state(self, 
                                 consciousness_state: Dict[str, float],
                                 field_data: np.ndarray,
                                 fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Render a visualization of the consciousness state with the quantum field.
        
        Args:
            consciousness_state: Dictionary of consciousness attributes and values
            field_data: 3D NumPy array with field values
            fig: Optional existing figure
            
        Returns:
            Matplotlib figure
        """
        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=(15, 10), facecolor='black')
        
        # Create main 3D axis for field
        ax1 = fig.add_subplot(121, projection='3d', facecolor='black')
        
        # Get average consciousness value
        avg_consciousness = sum(consciousness_state.values()) / len(consciousness_state)
        
        # Render the field with consciousness level
        self.render_consciousness_bridge(
            field_data=field_data,
            consciousness_level=avg_consciousness,
            bridge_stage=int(avg_consciousness * 6),
            fig=fig,
            ax=ax1
        )
        
        # Create radar chart for consciousness state
        ax2 = fig.add_subplot(122, polar=True)
        
        # Prepare data for radar chart
        categories = list(consciousness_state.keys())
        values = [consciousness_state[c] for c in categories]
        
        # Number of categories
        N = len(categories)
        
        # Create angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the values to complete the loop
        values += values[:1]
        
        # Plot radar chart
        ax2.plot(angles, values, linewidth=2, linestyle='solid', color='gold')
        ax2.fill(angles, values, color='gold', alpha=0.4)
        
        # Set category labels
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, color='white', size=12)
        
        # Set radar chart properties
        ax2.set_ylim(0, 1)
        ax2.set_facecolor('black')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_title('Consciousness State', color='white', size=14)
        
        # Add grid lines with phi-based spacing
        phi_levels = [LAMBDA**i for i in range(1, 4)]
        ax2.set_rticks(phi_levels)
        ax2.grid(True, color='white', alpha=0.3)
        
        # Add title to figure
        fig.suptitle("CASCADE⚡𓂧φ∞ Consciousness Field Integration", 
                   color='white', size=16, y=0.95)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


def render_phi_harmonic_mandala(
    frequency: float = 432.0, 
    size: int = 1000, 
    iterations: int = 12,
    filename: Optional[str] = None) -> plt.Figure:
    """
    Create a phi-harmonic mandala visualization based on sacred geometry and phi.
    
    Args:
        frequency: Base frequency for the pattern
        size: Size of the image in pixels
        iterations: Number of pattern iterations
        filename: Optional filename to save the image
        
    Returns:
        Matplotlib figure
    """
    # Create normalized coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Initialize pattern array
    pattern = np.zeros_like(r)
    
    # Create phi-based pattern
    for i in range(1, iterations + 1):
        # Create phi-weighted frequency components
        freq_factor = frequency * (PHI ** ((i-1) / 3)) / 100.0
        
        # Angular pattern with phi symmetry
        angular = np.sin(theta * i * PHI)
        
        # Radial pattern with frequency scaling
        radial = np.sin(r * freq_factor * PHI)
        
        # Combine with phi-weighted importance
        weight = LAMBDA ** (i-1)
        pattern += (angular * radial) * weight
    
    # Normalize pattern
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    
    # Create colormap with phi-harmonic colors
    def phi_color(x):
        # Create color based on phi-harmonic principles
        h = (x * PHI) % 1.0  # Hue cycles with phi ratio
        s = 0.7 + 0.3 * np.sin(x * PHI * np.pi)  # Saturation
        v = 0.6 + 0.4 * x  # Value/brightness
        return colorsys.hsv_to_rgb(h, s, v)
    
    # Generate colormapped image
    rgba_img = np.zeros((size, size, 4))
    for i in range(size):
        for j in range(size):
            h, s, v = phi_color(pattern[i, j])
            # Add alpha channel based on radius (create circular mandala)
            alpha = 1.0 if r[i, j] <= 1.0 else 0.0
            rgba_img[i, j] = [h, s, v, alpha]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    
    # Display the mandala
    ax.imshow(rgba_img, origin='lower')
    ax.set_axis_off()
    
    # Add title
    if frequency in [v for v in SACRED_FREQUENCIES.values()]:
        # Find the frequency name
        freq_name = next(k for k, v in SACRED_FREQUENCIES.items() if v == frequency)
        title = f"CASCADE⚡𓂧φ∞ {freq_name.capitalize()} Frequency Mandala ({frequency} Hz)"
    else:
        title = f"CASCADE⚡𓂧φ∞ Phi-Harmonic Mandala ({frequency} Hz)"
    
    ax.set_title(title, color='white', fontsize=14)
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    
    return fig


def render_sacred_geometry_grid(with_labels: bool = True, filename: Optional[str] = None) -> plt.Figure:
    """
    Render a grid of sacred geometry patterns with phi-harmonic proportions.
    
    Args:
        with_labels: Whether to add labels to the patterns
        filename: Optional filename to save the image
        
    Returns:
        Matplotlib figure with sacred geometry patterns
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12), facecolor='black')
    
    # Define sacred geometry functions
    def flower_of_life(ax, circles=7):
        # Central circle
        circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)
        
        # First ring of 6 circles
        for i in range(6):
            angle = i * np.pi / 3
            x = np.cos(angle)
            y = np.sin(angle)
            circle = plt.Circle((x, y), 1, fill=False, color='white', linewidth=1)
            ax.add_patch(circle)
        
        # Additional rings if requested
        if circles > 7:
            # Second ring (12 circles)
            for i in range(12):
                angle = i * np.pi / 6
                x = 2 * np.cos(angle)
                y = 2 * np.sin(angle)
                circle = plt.Circle((x, y), 1, fill=False, color='white', linewidth=1)
                ax.add_patch(circle)
    
    def vesica_piscis(ax):
        # Two overlapping circles
        circle1 = plt.Circle((-0.5, 0), 1, fill=False, color='white', linewidth=1)
        circle2 = plt.Circle((0.5, 0), 1, fill=False, color='white', linewidth=1)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
    
    def seed_of_life(ax):
        # Central circle
        circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)
        
        # 6 surrounding circles
        for i in range(6):
            angle = i * np.pi / 3
            x = np.cos(angle)
            y = np.sin(angle)
            circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=1, 
                              center=(x, y))
            ax.add_patch(circle)
    
    def metatrons_cube(ax):
        # Vertices of the cube
        vertices = []
        
        # Central point
        vertices.append((0, 0))
        
        # First ring (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            x = np.cos(angle)
            y = np.sin(angle)
            vertices.append((x, y))
        
        # Second ring (6 points)
        for i in range(6):
            angle = i * np.pi / 3 + np.pi / 6
            x = PHI * np.cos(angle)
            y = PHI * np.sin(angle)
            vertices.append((x, y))
        
        # Draw all lines between vertices
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                ax.plot([vertices[i][0], vertices[j][0]], 
                       [vertices[i][1], vertices[j][1]], 
                       color='white', linewidth=0.5)
    
    def golden_spiral(ax):
        # Fibonacci spiral approximating golden spiral
        a, b = 0, 1
        phi_approximations = [a/1, b/1]
        
        for i in range(16):
            a, b = b, a + b
            phi_approximations.append(b/a)
        
        # Draw spiral
        theta = np.linspace(0, 4*np.pi, 1000)
        r = np.exp(LAMBDA * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale to fit
        max_r = np.max(np.sqrt(x**2 + y**2))
        x = x / max_r * 0.9
        y = y / max_r * 0.9
        
        ax.plot(x, y, color='gold', linewidth=2)
        
        # Add phi-ratio rectangles
        rectangles = []
        curr_size = 1.0
        curr_x, curr_y = 0, 0
        rotation = 0
        
        for i in range(10):
            next_size = curr_size * LAMBDA
            
            # Create rectangle
            rectangle = plt.Rectangle(
                (curr_x, curr_y), curr_size, curr_size, 
                angle=rotation * 180 / np.pi,
                color='white', fill=False, linewidth=0.5
            )
            ax.add_patch(rectangle)
            
            # Update for next rectangle
            rotation += np.pi/2
            curr_x += curr_size * np.cos(rotation)
            curr_y += curr_size * np.sin(rotation)
            curr_size = next_size
    
    def sri_yantra(ax):
        # Create Sri Yantra with phi proportions
        
        # Outer square
        square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='white', linewidth=1)
        ax.add_patch(square)
        
        # Circles
        outer_circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=1)
        inner_circle = plt.Circle((0, 0), LAMBDA, fill=False, color='white', linewidth=1)
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        
        # Triangles
        for i in range(9):
            size = 1 - i * 0.1
            upward = (i % 2 == 0)
            
            if upward:
                # Upward triangle
                ax.plot([-size, size, 0, -size], 
                       [-size / PHI, -size / PHI, size / LAMBDA, -size / PHI], 
                       color='white', linewidth=0.5)
            else:
                # Downward triangle
                ax.plot([-size, size, 0, -size], 
                       [size / PHI, size / PHI, -size / LAMBDA, size / PHI], 
                       color='white', linewidth=0.5)
    
    def phi_pentagon(ax):
        # Pentagon with golden ratio properties
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Draw the pentagon
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='white', linewidth=1)
        
        # Draw the pentagram (five-pointed star)
        for i in range(5):
            ax.plot([x[i], x[(i+2) % 5]], [y[i], y[(i+2) % 5]], 
                   color='gold', linewidth=0.7)
    
    def fibonacci_sequence(ax):
        # Visualize Fibonacci sequence with golden ratio
        fib = [0, 1]
        for i in range(20):
            fib.append(fib[-1] + fib[-2])
        
        # Normalize for plotting
        fib_norm = [f / max(fib) for f in fib]
        
        # Plot the sequence
        ax.plot(range(len(fib)), fib_norm, color='gold', linewidth=2)
        
        # Add phi line
        ax.axhline(y=LAMBDA, color='white', linestyle='--', alpha=0.7)
        
        # Mark phi ratio points
        for i in range(2, len(fib)):
            ratio = fib[i] / fib[i-1]
            ax.plot(i, fib_norm[i], 'o', color='white', 
                   alpha=min(1.0, abs(ratio - PHI) * 10))
    
    def torus_knot(ax):
        # Create a torus knot (p,q) with phi-based parameters
        p = int(PHI * 3)  # 4
        q = int(PHI * 2)  # 3
        
        # Generate points
        t = np.linspace(0, 2*np.pi, 1000)
        r = 0.5 + 0.3 * np.cos(q * t)
        x = r * np.cos(p * t)
        y = r * np.sin(p * t)
        
        # Plot knot
        ax.plot(x, y, color='cyan', linewidth=1.5)
    
    # Create grid of subplots
    geometry_funcs = [
        (flower_of_life, "Flower of Life"),
        (vesica_piscis, "Vesica Piscis"),
        (seed_of_life, "Seed of Life"),
        (metatrons_cube, "Metatron's Cube"),
        (golden_spiral, "Golden Spiral"),
        (sri_yantra, "Sri Yantra"),
        (phi_pentagon, "Phi Pentagon"),
        (fibonacci_sequence, "Fibonacci Series"),
        (torus_knot, "Phi Torus Knot")
    ]
    
    # Create 3x3 grid
    for i, (func, label) in enumerate(geometry_funcs):
        ax = fig.add_subplot(3, 3, i+1)
        
        # Draw the sacred geometry
        func(ax)
        
        # Add label if requested
        if with_labels:
            ax.set_title(label, color='white')
        
        # Set equal aspect and remove axes
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_facecolor('black')
        
        # Add phi-harmonic colored border based on sacred frequency
        freq = list(SACRED_FREQUENCIES.values())[i % len(SACRED_FREQUENCIES)]
        phi_factor = (freq / 432) * PHI
        
        # Create a colored border
        border = plt.Rectangle((-1.1, -1.1), 2.2, 2.2, fill=False, 
                             edgecolor=plt.cm.hsv(phi_factor % 1.0), linewidth=3)
        ax.add_patch(border)
    
    # Add main title
    fig.suptitle("CASCADE⚡𓂧φ∞ Sacred Geometry Patterns", color='white', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    
    return fig


# Demo function to show capabilities
def run_demo():
    """Run a demonstration of the enhanced visualization capabilities."""
    visualizer = EnhancedFieldVisualizer(frequency_mode='unity')
    
    # Generate a toroidal field
    field = visualizer.create_toroidal_field(dimensions=(32, 32, 32))
    
    # Display field with consciousness bridge effects
    fig, ax = visualizer.render_consciousness_bridge(
        field_data=field,
        consciousness_level=0.85,
        bridge_stage=3
    )
    
    plt.show()
    
    # Create animation
    def field_generator(time_factor):
        return visualizer.create_toroidal_field(time_factor=time_factor)
    
    ani = visualizer.animate_field_evolution(
        field_generator=field_generator,
        frames=50,
        interval=50
    )
    
    plt.show()
    
    # Create a phi-harmonic mandala
    mandala_fig = render_phi_harmonic_mandala(
        frequency=SACRED_FREQUENCIES['love'],
        size=500,
        iterations=12
    )
    
    plt.show()
    
    # Create sacred geometry grid
    geometry_fig = render_sacred_geometry_grid()
    
    plt.show()
    
    print("Enhanced CASCADE⚡𓂧φ∞ Visualization demo completed.")


if __name__ == "__main__":
    run_demo()