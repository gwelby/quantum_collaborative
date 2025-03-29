"""
CASCADE⚡𓂧φ∞ Phi Pattern Matcher

Pattern matching system specifically designed for phi-harmonic patterns within quantum fields.
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import os

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
}

# Try to import from quantum_field, but fall back to our constants if not available
try:
    sys.path.append('/mnt/d/projects/python')
    from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
except (ImportError, ModuleNotFoundError):
    print("Using built-in sacred constants")


class PhiPatternMatcher:
    """
    Specialized pattern matcher focusing on phi-harmonic patterns and sacred geometry
    principles within quantum fields.
    """
    
    def __init__(self, use_sacred_geometry: bool = True):
        """
        Initialize the phi pattern matcher.
        
        Args:
            use_sacred_geometry: Whether to include sacred geometry patterns
        """
        self.use_sacred_geometry = use_sacred_geometry
        self.phi_patterns = self._initialize_phi_patterns()
        self.sacred_geometry_patterns = self._initialize_sacred_geometry() if use_sacred_geometry else {}
        
    def _initialize_phi_patterns(self) -> Dict[str, Any]:
        """Initialize built-in phi-harmonic patterns."""
        patterns = {}
        
        # 1D phi-harmonic patterns
        patterns['phi_wave'] = {
            'generator': lambda size: np.sin(np.linspace(0, PHI_PHI * 2 * np.pi, size)),
            'description': "Pure phi-harmonic sine wave"
        }
        
        patterns['phi_exponential'] = {
            'generator': lambda size: np.exp(np.linspace(-2, 2, size) * LAMBDA) / np.e**2,
            'description': "Phi-based exponential growth/decay pattern"
        }
        
        patterns['phi_fibonacci'] = {
            'generator': self._generate_fibonacci_pattern,
            'description': "Pattern based on Fibonacci sequence approximating phi"
        }
        
        # 2D phi-harmonic patterns
        patterns['phi_spiral'] = {
            'generator': self._generate_phi_spiral,
            'description': "Golden spiral based on phi ratio"
        }
        
        patterns['phi_grid'] = {
            'generator': self._generate_phi_grid,
            'description': "Grid pattern with phi-harmonic spacing"
        }
        
        # 3D phi-harmonic patterns
        patterns['phi_torus'] = {
            'generator': self._generate_phi_torus,
            'description': "Toroidal field with phi proportions"
        }
        
        patterns['phi_sphere'] = {
            'generator': self._generate_phi_sphere,
            'description': "Spherical field with phi-harmonic resonance"
        }
        
        return patterns
    
    def _initialize_sacred_geometry(self) -> Dict[str, Any]:
        """Initialize sacred geometry patterns."""
        patterns = {}
        
        # 2D sacred geometry patterns
        patterns['flower_of_life'] = {
            'generator': self._generate_flower_of_life,
            'description': "Flower of Life sacred geometry pattern"
        }
        
        patterns['sri_yantra'] = {
            'generator': self._generate_sri_yantra,
            'description': "Sri Yantra sacred geometry pattern"
        }
        
        patterns['metatrons_cube'] = {
            'generator': self._generate_metatrons_cube,
            'description': "Metatron's Cube sacred geometry pattern"
        }
        
        # 3D sacred geometry patterns
        patterns['merkaba'] = {
            'generator': self._generate_merkaba,
            'description': "Merkaba (Star Tetrahedron) pattern"
        }
        
        patterns['platonic_solids'] = {
            'generator': self._generate_platonic_solids,
            'description': "Platonic solids field resonance pattern"
        }
        
        return patterns
    
    def get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get list of all available phi and sacred geometry patterns.
        
        Returns:
            List of pattern information dictionaries
        """
        patterns = []
        
        # Add phi patterns
        for name, info in self.phi_patterns.items():
            patterns.append({
                'name': name,
                'type': 'phi',
                'description': info['description']
            })
            
        # Add sacred geometry patterns
        for name, info in self.sacred_geometry_patterns.items():
            patterns.append({
                'name': name,
                'type': 'sacred_geometry',
                'description': info['description']
            })
            
        return patterns
    
    def generate_pattern(self, 
                        pattern_name: str, 
                        dimensions: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Generate a specific phi-harmonic or sacred geometry pattern.
        
        Args:
            pattern_name: Name of the pattern to generate
            dimensions: Dimensions of the pattern (e.g., (64, 64) for 2D)
            
        Returns:
            NumPy array containing the pattern, or None if pattern not found
        """
        # Check phi patterns
        if pattern_name in self.phi_patterns:
            generator = self.phi_patterns[pattern_name]['generator']
            return self._generate_pattern(generator, dimensions)
            
        # Check sacred geometry patterns
        if pattern_name in self.sacred_geometry_patterns:
            generator = self.sacred_geometry_patterns[pattern_name]['generator']
            return self._generate_pattern(generator, dimensions)
            
        return None
    
    def _generate_pattern(self, 
                        generator: callable, 
                        dimensions: Tuple[int, ...]) -> np.ndarray:
        """Generate a pattern using the provided generator function."""
        try:
            # Handle different dimensions
            if len(dimensions) == 1:
                return generator(dimensions[0])
            elif len(dimensions) == 2:
                return generator(dimensions)
            elif len(dimensions) == 3:
                return generator(dimensions)
            else:
                raise ValueError(f"Unsupported dimensions: {dimensions}")
        except Exception as e:
            print(f"Error generating pattern: {e}")
            # Return empty array of correct shape
            return np.zeros(dimensions)
    
    def _generate_fibonacci_pattern(self, size: int) -> np.ndarray:
        """Generate a pattern based on Fibonacci sequence."""
        # Generate Fibonacci sequence
        fib = [0, 1]
        while len(fib) < size:
            fib.append(fib[-1] + fib[-2])
            
        # Normalize to [0, 1] range
        fib_arr = np.array(fib[:size], dtype=float)
        if np.max(fib_arr) > 0:
            fib_arr = fib_arr / np.max(fib_arr)
            
        # Create a wave pattern modulated by Fibonacci ratios
        x = np.linspace(0, PHI_PHI * 2 * np.pi, size)
        pattern = np.sin(x) * (0.5 + 0.5 * fib_arr)
        
        return pattern
    
    def _generate_phi_spiral(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate a golden spiral pattern based on phi."""
        width, height = dimensions
        X, Y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='ij'
        )
        
        # Convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Create phi spiral
        spiral = np.exp(-R / PHI) * np.sin(Theta * PHI + R * PHI_PHI * 5)
        
        return spiral
    
    def _generate_phi_grid(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate a grid pattern with phi-harmonic spacing."""
        width, height = dimensions
        X, Y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='ij'
        )
        
        # Create grid lines with phi-harmonic spacing
        grid_x = np.zeros_like(X)
        grid_y = np.zeros_like(Y)
        
        # Generate phi-spaced grid lines
        for i in range(-3, 4):
            pos = i * LAMBDA
            
            # Add horizontal and vertical lines
            mask_x = np.abs(X - pos) < 0.02
            mask_y = np.abs(Y - pos) < 0.02
            
            grid_x[mask_x] = 1.0
            grid_y[mask_y] = 1.0
            
        # Combine grids
        grid = grid_x + grid_y
        grid = np.minimum(grid, 1.0)
        
        return grid
    
    def _generate_phi_torus(self, dimensions: Tuple[int, int, int]) -> np.ndarray:
        """Generate a toroidal field with phi proportions."""
        width, height, depth = dimensions
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, depth),
            indexing='ij'
        )
        
        # Create toroidal shape with phi-based proportions
        major_radius = PHI * 0.5  # Scaled to fit in [-1,1] range
        minor_radius = LAMBDA * 0.5
        
        # Distance from a ring in the xy-plane
        ring_distance = np.sqrt((np.sqrt(X**2 + Y**2) - major_radius)**2 + Z**2)
        
        # Create torus
        torus = np.exp(-(ring_distance - minor_radius)**2 / (0.05))
        
        # Add phi-harmonic resonance
        theta = np.arctan2(Y, X)  # Azimuthal angle
        poloidal = np.arctan2(Z, np.sqrt(X**2 + Y**2) - major_radius)  # Poloidal angle
        
        # Add toroidal flow pattern
        resonance = np.sin(theta * PHI + poloidal * PHI_PHI)
        
        # Combine with torus shape
        field = torus * (0.5 + 0.5 * resonance)
        
        return field
    
    def _generate_phi_sphere(self, dimensions: Tuple[int, int, int]) -> np.ndarray:
        """Generate a spherical field with phi-harmonic resonance."""
        width, height, depth = dimensions
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, depth),
            indexing='ij'
        )
        
        # Create spherical shape
        R = np.sqrt(X**2 + Y**2 + Z**2)
        sphere = np.exp(-(R - 0.5)**2 / 0.05)
        
        # Add phi-harmonic resonance using spherical harmonics
        theta = np.arccos(Z / (R + 1e-10))  # Polar angle
        phi = np.arctan2(Y, X)  # Azimuthal angle
        
        # Create phi-based spherical harmonic
        harmonic = np.sin(theta * PHI) * np.sin(phi * PHI_PHI)
        
        # Combine sphere and harmonic
        field = sphere * (0.5 + 0.5 * harmonic)
        
        return field
    
    def _generate_flower_of_life(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate the Flower of Life sacred geometry pattern."""
        width, height = dimensions
        X, Y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='ij'
        )
        
        flower = np.zeros((width, height))
        r = 1/3  # Circle radius
        
        # Center circle
        center_circle = np.exp(-((X)**2 + (Y)**2) / (r/2)**2)
        flower = np.maximum(flower, center_circle)
        
        # First ring of 6 circles
        for i in range(6):
            angle = i * np.pi / 3
            cx = r * np.cos(angle)
            cy = r * np.sin(angle)
            
            circle = np.exp(-((X - cx)**2 + (Y - cy)**2) / (r/2)**2)
            flower = np.maximum(flower, circle)
            
        # Second ring of 12 circles (partial)
        for i in range(6):
            angle = i * np.pi / 3 + np.pi / 6
            cx = r * 2 * np.cos(angle)
            cy = r * 2 * np.sin(angle)
            
            if cx**2 + cy**2 <= 1:  # Keep within bounds
                circle = np.exp(-((X - cx)**2 + (Y - cy)**2) / (r/2)**2)
                flower = np.maximum(flower, circle)
        
        return flower
    
    def _generate_sri_yantra(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate a simplified Sri Yantra pattern."""
        width, height = dimensions
        X, Y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='ij'
        )
        
        yantra = np.zeros((width, height))
        
        # Central dot (bindu)
        bindu = np.exp(-((X)**2 + (Y)**2) / 0.01)
        yantra = np.maximum(yantra, bindu)
        
        # Concentric triangles with phi-based scaling
        for i in range(4):
            # Scale factor based on phi
            scale = LAMBDA**(i+1)
            
            # Upward triangle
            y_up = -scale + 2 * scale * (0.5 + 0.5 * np.sign(X*PHI))
            tri_up = 1.0 - np.minimum(1.0, np.abs(Y - y_up) / 0.02)
            yantra = np.maximum(yantra, tri_up)
            
            # Downward triangle
            y_down = scale - 2 * scale * (0.5 + 0.5 * np.sign(X*PHI))
            tri_down = 1.0 - np.minimum(1.0, np.abs(Y - y_down) / 0.02)
            yantra = np.maximum(yantra, tri_down)
        
        # Circles
        for r in [0.3, 0.5, 0.7, 0.9]:
            circle = 1.0 - np.minimum(1.0, np.abs(np.sqrt(X**2 + Y**2) - r) / 0.02)
            yantra = np.maximum(yantra, circle)
            
        return yantra
    
    def _generate_metatrons_cube(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate a simplified Metatron's Cube pattern."""
        width, height = dimensions
        X, Y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='ij'
        )
        
        cube = np.zeros((width, height))
        
        # Create vertices of cube (13 total points)
        points = [
            (0, 0),  # Center
        ]
        
        # First ring of 6 points
        for i in range(6):
            angle = i * np.pi / 3
            points.append((0.5 * np.cos(angle), 0.5 * np.sin(angle)))
            
        # Second ring of 6 points
        for i in range(6):
            angle = i * np.pi / 3 + np.pi / 6
            points.append((0.9 * np.cos(angle), 0.9 * np.sin(angle)))
            
        # Add points to field
        for point in points:
            px, py = point
            marker = np.exp(-((X - px)**2 + (Y - py)**2) / 0.01)
            cube = np.maximum(cube, marker)
            
        # Add lines connecting points
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i < j:  # Only connect each pair once
                    px1, py1 = p1
                    px2, py2 = p2
                    
                    # Vector from p1 to p2
                    dx, dy = px2 - px1, py2 - py1
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length < 1.5:  # Only connect nearby points
                        # Parameterize line from 0 to 1
                        t = ((X - px1) * dx + (Y - py1) * dy) / (length**2 + 1e-10)
                        t = np.clip(t, 0, 1)
                        
                        # Find closest point on line
                        closest_x = px1 + t * dx
                        closest_y = py1 + t * dy
                        
                        # Distance to line
                        dist = np.sqrt((X - closest_x)**2 + (Y - closest_y)**2)
                        
                        # Add line to field
                        line = np.exp(-(dist**2) / 0.001)
                        cube = np.maximum(cube, line)
        
        return cube
    
    def _generate_merkaba(self, dimensions: Tuple[int, int, int]) -> np.ndarray:
        """Generate a Merkaba (Star Tetrahedron) pattern."""
        width, height, depth = dimensions
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, depth),
            indexing='ij'
        )
        
        merkaba = np.zeros((width, height, depth))
        
        # Create upward tetrahedron
        up_tetra = np.ones((width, height, depth))
        
        # Four planes defining the tetrahedron
        planes_up = [
            X + Y + Z - 0.6,       # Base
            X - Y - Z - 0.6,       # Face 1
            -X + Y - Z - 0.6,      # Face 2
            -X - Y + Z - 0.6       # Face 3
        ]
        
        # Intersection of half-spaces
        for plane in planes_up:
            up_tetra = np.minimum(up_tetra, np.maximum(0, np.sign(plane)))
            
        # Create downward tetrahedron
        down_tetra = np.ones((width, height, depth))
        
        # Four planes defining the downward tetrahedron
        planes_down = [
            -(X + Y + Z) - 0.6,    # Base
            -(X - Y - Z) - 0.6,    # Face 1
            -(-X + Y - Z) - 0.6,   # Face 2
            -(-X - Y + Z) - 0.6    # Face 3
        ]
        
        # Intersection of half-spaces
        for plane in planes_down:
            down_tetra = np.minimum(down_tetra, np.maximum(0, np.sign(plane)))
            
        # Combine tetrahedra with phi-weighted scaling
        up_tetra = up_tetra * LAMBDA
        down_tetra = down_tetra * LAMBDA
        merkaba = up_tetra + down_tetra
        
        # Add resonance pattern
        R = np.sqrt(X**2 + Y**2 + Z**2)
        spin = np.sin(R * PHI_PHI * 5)
        
        # Final field with resonance
        field = merkaba * (0.5 + 0.5 * spin)
        
        return field
    
    def _generate_platonic_solids(self, dimensions: Tuple[int, int, int]) -> np.ndarray:
        """Generate a field with resonant Platonic solid patterns."""
        width, height, depth = dimensions
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, depth),
            indexing='ij'
        )
        
        field = np.zeros((width, height, depth))
        
        # Create spherical base
        R = np.sqrt(X**2 + Y**2 + Z**2)
        sphere = np.exp(-(R - 0.7)**2 / 0.1)
        
        # Add tetrahedron pattern
        tetra = abs(X) + abs(Y) + abs(Z)
        tetra_pattern = np.exp(-(tetra - 0.8)**2 / 0.05)
        
        # Add cube pattern
        cube = np.maximum(np.maximum(abs(X), abs(Y)), abs(Z))
        cube_pattern = np.exp(-(cube - 0.6)**2 / 0.05)
        
        # Add octahedron pattern
        octa = abs(X) + abs(Y) + abs(Z)
        octa_pattern = np.exp(-(octa - 1.0)**2 / 0.05)
        
        # Add dodecahedron approximation
        dodeca = (abs(X) + abs(Y) + abs(Z))**2 / (X**2 + Y**2 + Z**2 + 1e-10)
        dodeca_pattern = np.exp(-(dodeca - 1.5)**2 / 0.1)
        
        # Add icosahedron approximation
        icosa = (X**2 + Y**2 + Z**2) + PHI * abs(X * Y * Z)
        icosa_pattern = np.exp(-(icosa - 1.5)**2 / 0.1)
        
        # Combine all patterns with phi-weighted averaging
        combined = (
            sphere * 0.2 +
            tetra_pattern * LAMBDA**4 +
            cube_pattern * LAMBDA**3 +
            octa_pattern * LAMBDA**2 +
            dodeca_pattern * LAMBDA +
            icosa_pattern * 0.1
        )
        
        # Normalize
        combined = combined / np.max(combined)
        
        # Add phi-harmonic resonance
        resonance = np.sin(R * PHI_PHI * 5)
        field = combined * (0.7 + 0.3 * resonance)
        
        return field
    
    def detect_phi_patterns(self, field_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect phi-harmonic patterns present in a quantum field.
        
        Args:
            field_data: NumPy array containing the field
            
        Returns:
            List of detected patterns with confidence scores
        """
        # Get field dimensions
        dimensions = len(field_data.shape)
        
        # Select patterns matching field dimensions
        relevant_patterns = {}
        for name, info in self.phi_patterns.items():
            pattern = self.generate_pattern(name, field_data.shape)
            if pattern is not None and pattern.shape == field_data.shape:
                relevant_patterns[name] = pattern
                
        if self.use_sacred_geometry:
            for name, info in self.sacred_geometry_patterns.items():
                pattern = self.generate_pattern(name, field_data.shape)
                if pattern is not None and pattern.shape == field_data.shape:
                    relevant_patterns[name] = pattern
        
        # Detect patterns
        results = []
        
        # Normalize field for comparison
        field_norm = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data) + 1e-10)
        
        for name, pattern in relevant_patterns.items():
            # Normalize pattern
            pattern_norm = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
            
            # Calculate correlation
            try:
                from scipy.signal import correlate
                correlation = correlate(field_norm, pattern_norm, mode='valid')
                max_corr = np.max(correlation) / np.prod(pattern.shape)
            except ImportError:
                # Manual calculation (simplified)
                diff = np.abs(field_norm - pattern_norm)
                similarity = 1.0 - np.mean(diff)
                max_corr = similarity
                
            # Calculate phase coherence
            try:
                # Frequency domain comparison
                field_fft = np.fft.fftn(field_norm)
                pattern_fft = np.fft.fftn(pattern_norm)
                
                # Phase difference
                field_phase = np.angle(field_fft)
                pattern_phase = np.angle(pattern_fft)
                
                # Phase coherence (simplified)
                phase_diff = np.abs(field_phase - pattern_phase)
                phase_coherence = np.mean(np.cos(phase_diff))
            except:
                phase_coherence = 0.5  # Default if calculation fails
                
            # Overall phi-pattern score with phi-weighted components
            pattern_score = (
                max_corr * LAMBDA + 
                phase_coherence * LAMBDA**2
            ) / (LAMBDA + LAMBDA**2)
            
            # Add to results if score is significant
            if pattern_score > 0.6:
                results.append({
                    'pattern_name': name,
                    'confidence': pattern_score,
                    'correlation': max_corr,
                    'phase_coherence': phase_coherence
                })
                
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def get_phi_metrics(self, field_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate phi-harmonic metrics for a quantum field.
        
        Args:
            field_data: NumPy array containing the field
            
        Returns:
            Dictionary of phi-harmonic metrics
        """
        metrics = {}
        
        # Calculate phi resonance
        flat_data = field_data.flatten()
        
        # Sample a subset of points for large fields
        if flat_data.size > 1000:
            indices = np.random.choice(flat_data.size, 1000, replace=False)
            flat_data = flat_data[indices]
            
        # Normalize data
        data_range = np.max(flat_data) - np.min(flat_data)
        if data_range > 0:
            flat_data = (flat_data - np.min(flat_data)) / data_range
            
        # Check phi alignment
        phi_powers = np.array([PHI ** i for i in range(-3, 4)])
        distances = np.min(np.abs(flat_data[:, np.newaxis] - phi_powers), axis=1)
        phi_alignment = 1.0 - np.mean(distances) / PHI
        metrics['phi_alignment'] = phi_alignment
        
        # Calculate phi ratio presence in the field structure
        if len(field_data.shape) > 1:
            # Find local maxima
            try:
                maxima = []
                if len(field_data.shape) == 2:
                    from scipy.ndimage import maximum_filter
                    local_max = maximum_filter(field_data, size=3)
                    maxima = np.where(local_max == field_data)
                elif len(field_data.shape) == 3:
                    from scipy.ndimage import maximum_filter
                    local_max = maximum_filter(field_data, size=3)
                    maxima = np.where(local_max == field_data)
                    
                if len(maxima[0]) > 1:
                    # Calculate distances between adjacent maxima
                    coords = np.column_stack(maxima)
                    
                    # Calculate pairwise distances
                    dists = []
                    for i in range(len(coords) - 1):
                        dist = np.sqrt(np.sum((coords[i] - coords[i+1])**2))
                        dists.append(dist)
                        
                    if dists:
                        # Calculate ratios between adjacent distances
                        ratios = np.array([dists[i]/dists[i+1] if dists[i+1] > 0 else 1.0 
                                          for i in range(len(dists)-1)])
                        
                        # Calculate how close these ratios are to phi
                        phi_ratio_alignment = 1.0 - np.mean(np.abs(ratios - PHI)) / PHI
                        metrics['phi_ratio_alignment'] = phi_ratio_alignment
            except ImportError:
                # Skip if scipy not available
                pass
                
        # Calculate field entropy and phi-entropy ratio
        try:
            # Calculate entropy of field values
            hist, _ = np.histogram(flat_data, bins=20)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Calculate phi-optimal entropy (entropy value that would occur if perfectly phi-distributed)
            phi_optimal = -np.log2(LAMBDA)
            
            # Ratio of actual entropy to phi-optimal entropy
            phi_entropy_ratio = 1.0 - abs(entropy - phi_optimal) / phi_optimal
            metrics['phi_entropy_ratio'] = phi_entropy_ratio
        except:
            # Skip if calculation fails
            pass
            
        # Calculate phi-harmonic frequency components if field is large enough
        if min(field_data.shape) >= 8:
            try:
                # Compute FFT
                fft = np.fft.fftn(field_data)
                fft_mag = np.abs(fft)
                
                # Look for peaks at phi-related frequencies
                phi_freqs = [1/PHI, 1.0, PHI, PHI**2]
                
                # Normalize frequencies to [0,1] range
                fft_freqs = []
                for dim in range(len(field_data.shape)):
                    fft_freqs.append(np.fft.fftfreq(field_data.shape[dim]))
                    
                # Find phi frequency resonance for 1D case
                if len(field_data.shape) == 1:
                    freqs = fft_freqs[0]
                    resonance_values = []
                    
                    for phi_f in phi_freqs:
                        # Find closest frequency bin
                        closest_idx = np.argmin(np.abs(freqs - phi_f/2))
                        resonance_values.append(fft_mag[closest_idx])
                        
                    # Normalize and average
                    if np.max(fft_mag) > 0:
                        phi_freq_resonance = np.mean(resonance_values) / np.max(fft_mag)
                        metrics['phi_frequency_resonance'] = phi_freq_resonance
            except:
                # Skip if calculation fails
                pass
        
        return metrics