"""
CASCADE⚡𓂧φ∞ Quantum Pattern Engine

Core pattern recognition system for quantum fields based on phi-harmonic principles.
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import importlib.util
from pathlib import Path
import pickle
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


class QuantumPatternEngine:
    """
    Core pattern recognition engine for quantum fields that identifies
    and categorizes phi-harmonic patterns using advanced recognition techniques.
    """
    
    def __init__(self, 
                 pattern_library_path: Optional[str] = None,
                 use_phi_scaling: bool = True,
                 sensitivity: float = PHI):
        """
        Initialize the quantum pattern recognition engine.
        
        Args:
            pattern_library_path: Path to stored pattern library (optional)
            use_phi_scaling: Whether to use phi-based scaling for pattern matching
            sensitivity: Pattern matching sensitivity (default: PHI)
        """
        self.patterns = {}
        self.field_states = {}
        self.use_phi_scaling = use_phi_scaling
        self.sensitivity = sensitivity
        self.field_metrics = {}
        
        # Try to load pattern library if provided
        if pattern_library_path:
            self.load_pattern_library(pattern_library_path)
            
        # Initialize phi-harmonic transformations
        self.phi_powers = [PHI ** i for i in range(-3, 4)]
        self.phi_kernels = self._create_phi_kernels()
            
    def _create_phi_kernels(self) -> Dict[str, np.ndarray]:
        """Create convolution kernels based on phi for pattern detection."""
        kernels = {}
        
        # 1D kernel (phi-based)
        kernels['1d'] = np.array([LAMBDA**2, LAMBDA, 1.0, LAMBDA, LAMBDA**2])
        kernels['1d'] = kernels['1d'] / np.sum(kernels['1d'])
        
        # 2D kernel (phi-based)
        kernel_2d = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                distance = np.sqrt((i - 2)**2 + (j - 2)**2)
                kernel_2d[i, j] = np.exp(-distance * LAMBDA)
        kernels['2d'] = kernel_2d / np.sum(kernel_2d)
        
        # 3D kernel (phi-based)
        kernel_3d = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance = np.sqrt((i - 1)**2 + (j - 1)**2 + (k - 1)**2)
                    kernel_3d[i, j, k] = np.exp(-distance * LAMBDA)
        kernels['3d'] = kernel_3d / np.sum(kernel_3d)
        
        return kernels
    
    def register_pattern(self, 
                        name: str, 
                        pattern_data: np.ndarray,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Register a new pattern in the pattern library.
        
        Args:
            name: Unique identifier for the pattern
            pattern_data: NumPy array containing the pattern
            metadata: Additional information about the pattern
            
        Returns:
            Success status
        """
        if name in self.patterns:
            print(f"Pattern '{name}' already exists. Use update_pattern instead.")
            return False
            
        # Extract pattern features
        features = self._extract_pattern_features(pattern_data)
        
        # Create pattern entry
        self.patterns[name] = {
            'data': pattern_data,
            'features': features,
            'metadata': metadata or {},
            'creation_time': np.datetime64('now')
        }
        
        return True
    
    def update_pattern(self, 
                     name: str, 
                     pattern_data: np.ndarray = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing pattern in the library.
        
        Args:
            name: Pattern identifier to update
            pattern_data: New pattern data (optional)
            metadata: New metadata (optional)
            
        Returns:
            Success status
        """
        if name not in self.patterns:
            print(f"Pattern '{name}' does not exist. Use register_pattern instead.")
            return False
            
        # Update pattern data if provided
        if pattern_data is not None:
            features = self._extract_pattern_features(pattern_data)
            self.patterns[name]['data'] = pattern_data
            self.patterns[name]['features'] = features
            
        # Update metadata if provided
        if metadata is not None:
            if self.patterns[name].get('metadata') is None:
                self.patterns[name]['metadata'] = {}
            self.patterns[name]['metadata'].update(metadata)
            
        # Update modification time
        self.patterns[name]['modification_time'] = np.datetime64('now')
        
        return True
    
    def _extract_pattern_features(self, pattern_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract key features from a pattern for efficient matching.
        
        Args:
            pattern_data: NumPy array containing the pattern
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(pattern_data)
        features['std'] = np.std(pattern_data)
        features['min'] = np.min(pattern_data)
        features['max'] = np.max(pattern_data)
        
        # Shape and dimensionality
        features['shape'] = pattern_data.shape
        features['dimensions'] = len(pattern_data.shape)
        
        # Frequency domain features (if large enough)
        if min(pattern_data.shape) >= 4:
            # Compute FFT
            fft = np.fft.fftn(pattern_data)
            fft_mag = np.abs(fft)
            
            # Get dominant frequencies
            flat_indices = np.argsort(fft_mag.flatten())[-10:]  # Top 10 frequencies
            features['dominant_frequencies'] = []
            
            # Convert flat indices to multi-dimensional indices
            for idx in flat_indices:
                idx_tuple = np.unravel_index(idx, fft_mag.shape)
                features['dominant_frequencies'].append((idx_tuple, fft_mag[idx_tuple]))
        
        # Phi-harmonic features
        # Calculate alignment with phi in multiple ways
        phi_alignment_scores = []
        flat_data = pattern_data.flatten()
        
        # 1. Distance to nearest phi multiple
        phi_multiples = np.array([PHI * i for i in range(-3, 4)])
        distances1 = np.min(np.abs(flat_data[:, np.newaxis] - phi_multiples), axis=1)
        phi_alignment_scores.append(1.0 - np.mean(distances1) / PHI)
        
        # 2. Distance to nearest phi power
        phi_powers = np.array([PHI ** i for i in range(-2, 3)])
        distances2 = np.min(np.abs(flat_data[:, np.newaxis] - phi_powers), axis=1)
        phi_alignment_scores.append(1.0 - np.mean(distances2) / PHI)
        
        # Overall phi alignment
        features['phi_alignment'] = np.mean(phi_alignment_scores)
        
        # Get histogram and gradient distribution
        features['histogram'] = np.histogram(pattern_data, bins=10, range=(features['min'], features['max']))[0]
        
        # Calculate gradients if multidimensional
        if features['dimensions'] > 1:
            gradients = np.gradient(pattern_data)
            features['gradient_magnitude'] = np.sqrt(sum(np.square(g) for g in gradients))
            features['gradient_mean'] = [np.mean(g) for g in gradients]
            features['gradient_std'] = [np.std(g) for g in gradients]
        
        return features
    
    def save_pattern_library(self, path: str) -> bool:
        """
        Save the current pattern library to disk.
        
        Args:
            path: File path to save the pattern library
            
        Returns:
            Success status
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'field_states': self.field_states,
                    'phi_parameters': {
                        'phi': PHI,
                        'lambda': LAMBDA,
                        'phi_phi': PHI_PHI
                    }
                }, f)
            return True
        except Exception as e:
            print(f"Failed to save pattern library: {e}")
            return False
    
    def load_pattern_library(self, path: str) -> bool:
        """
        Load a pattern library from disk.
        
        Args:
            path: File path to load the pattern library from
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(path):
                print(f"Pattern library file not found: {path}")
                return False
                
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Validate library format
            if not all(key in data for key in ['patterns', 'field_states', 'phi_parameters']):
                print("Invalid pattern library format.")
                return False
                
            # Check phi parameters match
            phi_params = data['phi_parameters']
            if abs(phi_params['phi'] - PHI) > 1e-10:
                print(f"Warning: Pattern library uses different PHI value: {phi_params['phi']}")
                
            # Load data
            self.patterns = data['patterns']
            self.field_states = data['field_states']
            
            return True
        
        except Exception as e:
            print(f"Failed to load pattern library: {e}")
            return False
    
    def match_pattern(self, 
                     field_data: np.ndarray, 
                     threshold: float = 0.75,
                     max_matches: int = 5) -> List[Dict[str, Any]]:
        """
        Find matches for known patterns in a quantum field.
        
        Args:
            field_data: NumPy array containing the field to analyze
            threshold: Minimum similarity score to be considered a match (0.0-1.0)
            max_matches: Maximum number of matches to return
            
        Returns:
            List of matches with similarity scores and positions
        """
        if not self.patterns:
            return []
            
        # Extract features from the field
        field_features = self._extract_pattern_features(field_data)
        self.field_metrics = field_features
        
        matches = []
        
        # Perform multiscale pattern matching
        scales = [1.0, PHI, 1/PHI] if self.use_phi_scaling else [1.0]
        
        for pattern_name, pattern_info in self.patterns.items():
            pattern_data = pattern_info['data']
            pattern_features = pattern_info['features']
            
            best_match = {
                'name': pattern_name,
                'similarity': 0.0,
                'position': None,
                'scale': 1.0
            }
            
            # Skip if dimensions don't match
            if len(pattern_data.shape) != len(field_data.shape):
                continue
                
            # Try different scales
            for scale in scales:
                # Skip if pattern is too large at this scale
                if any(int(d * scale) > fd for d, fd in zip(pattern_data.shape, field_data.shape)):
                    continue
                
                # Resize pattern for matching if scale != 1.0
                if scale != 1.0:
                    # Simple resize using scipy if available
                    try:
                        from scipy.ndimage import zoom
                        scaled_pattern = zoom(pattern_data, scale)
                    except ImportError:
                        # Fallback for 1D and 2D only
                        if len(pattern_data.shape) <= 2:
                            # Very simple scaling - repeating/skipping values
                            scaled_pattern = self._simple_rescale(pattern_data, scale)
                        else:
                            continue
                else:
                    scaled_pattern = pattern_data
                
                # Perform pattern matching using convolution-based approach
                similarity_map = self._compute_similarity_map(field_data, scaled_pattern)
                
                # Find best match position
                max_sim = np.max(similarity_map)
                if max_sim > best_match['similarity']:
                    max_pos = np.unravel_index(np.argmax(similarity_map), similarity_map.shape)
                    best_match['similarity'] = max_sim
                    best_match['position'] = max_pos
                    best_match['scale'] = scale
            
            # Add to matches if above threshold
            if best_match['similarity'] >= threshold:
                matches.append({
                    'name': pattern_name,
                    'similarity': best_match['similarity'],
                    'position': best_match['position'],
                    'scale': best_match['scale'],
                    'pattern_info': {
                        'shape': pattern_info['data'].shape,
                        'metadata': pattern_info['metadata']
                    }
                })
        
        # Sort by similarity and limit to max_matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:max_matches]
    
    def _compute_similarity_map(self, field: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """
        Compute similarity between pattern and field using convolution.
        
        Args:
            field: Field data to search in
            pattern: Pattern to match
            
        Returns:
            Map of similarity values at each position
        """
        # Normalize both arrays to [0,1] range for comparison
        field_norm = (field - np.min(field)) / (np.max(field) - np.min(field) + 1e-10)
        pattern_norm = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
        
        # Get dimensions and create result map
        dims = len(field.shape)
        result_shape = tuple(max(0, fs - ps) + 1 for fs, ps in zip(field.shape, pattern.shape))
        similarity_map = np.zeros(result_shape)
        
        # Perform convolution-based matching
        try:
            # Try with scipy for speed
            from scipy.signal import correlate
            similarity_map = correlate(field_norm, pattern_norm, mode='valid')
            similarity_map = similarity_map / np.prod(pattern.shape)  # Normalize
        except ImportError:
            # Manual implementation
            if dims == 1:
                # 1D case
                for i in range(result_shape[0]):
                    window = field_norm[i:i+pattern.shape[0]]
                    similarity_map[i] = 1.0 - np.mean(np.abs(window - pattern_norm))
            
            elif dims == 2:
                # 2D case
                for i in range(result_shape[0]):
                    for j in range(result_shape[1]):
                        window = field_norm[i:i+pattern.shape[0], j:j+pattern.shape[1]]
                        similarity_map[i, j] = 1.0 - np.mean(np.abs(window - pattern_norm))
            
            elif dims == 3:
                # 3D case
                for i in range(result_shape[0]):
                    for j in range(result_shape[1]):
                        for k in range(result_shape[2]):
                            window = field_norm[i:i+pattern.shape[0], 
                                               j:j+pattern.shape[1], 
                                               k:k+pattern.shape[2]]
                            similarity_map[i, j, k] = 1.0 - np.mean(np.abs(window - pattern_norm))
        
        return similarity_map
    
    def _simple_rescale(self, arr: np.ndarray, scale: float) -> np.ndarray:
        """Simple array rescaling when scipy is not available."""
        if len(arr.shape) == 1:
            # 1D case
            new_size = int(arr.shape[0] * scale)
            result = np.zeros(new_size)
            for i in range(new_size):
                orig_idx = min(int(i / scale), arr.shape[0] - 1)
                result[i] = arr[orig_idx]
            return result
            
        elif len(arr.shape) == 2:
            # 2D case
            new_h = int(arr.shape[0] * scale)
            new_w = int(arr.shape[1] * scale)
            result = np.zeros((new_h, new_w))
            
            for i in range(new_h):
                for j in range(new_w):
                    orig_i = min(int(i / scale), arr.shape[0] - 1)
                    orig_j = min(int(j / scale), arr.shape[1] - 1)
                    result[i, j] = arr[orig_i, orig_j]
                    
            return result
        
        # Not implemented for higher dimensions
        return arr
    
    def recognize_field_state(self, field_data: np.ndarray) -> Dict[str, float]:
        """
        Recognize the state of a quantum field using trained recognizers.
        
        Args:
            field_data: The quantum field data to analyze
            
        Returns:
            Dictionary of field state probabilities
        """
        if not self.field_states:
            return {"unknown": 1.0}
            
        # Extract field features
        field_features = self._extract_pattern_features(field_data)
        
        # Calculate similarity to each known field state
        similarities = {}
        for state_name, state_info in self.field_states.items():
            similarity = self._calculate_state_similarity(field_features, state_info['features'])
            similarities[state_name] = similarity
            
        # Convert to probabilities (normalize)
        total = sum(similarities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in similarities.items()}
        else:
            probabilities = {"unknown": 1.0}
            
        return probabilities
    
    def _calculate_state_similarity(self, field_features: Dict, state_features: Dict) -> float:
        """Calculate similarity between field features and state features."""
        if 'phi_alignment' in field_features and 'phi_alignment' in state_features:
            phi_align_diff = abs(field_features['phi_alignment'] - state_features['phi_alignment'])
            
        # Start with basic metrics
        sim_mean = 1.0 - min(1.0, abs(field_features['mean'] - state_features['mean']) / 
                          max(abs(field_features['mean']), abs(state_features['mean']) + 1e-10))
                          
        sim_std = 1.0 - min(1.0, abs(field_features['std'] - state_features['std']) / 
                         max(abs(field_features['std']), abs(state_features['std']) + 1e-10))
        
        # Compare histograms if available
        if 'histogram' in field_features and 'histogram' in state_features:
            # Normalize histograms
            hist1 = field_features['histogram'] / np.sum(field_features['histogram'])
            hist2 = state_features['histogram'] / np.sum(state_features['histogram'])
            
            # Histogram intersection similarity
            sim_hist = np.sum(np.minimum(hist1, hist2))
        else:
            sim_hist = 0.5  # Neutral if not available
            
        # Compare phi alignment
        if 'phi_alignment' in field_features and 'phi_alignment' in state_features:
            sim_phi = 1.0 - min(1.0, abs(field_features['phi_alignment'] - state_features['phi_alignment']))
        else:
            sim_phi = 0.5  # Neutral if not available
            
        # Weighted similarity (phi-weighted)
        similarity = (
            sim_mean * LAMBDA +
            sim_std * LAMBDA**2 +
            sim_hist * LAMBDA**3 +
            sim_phi * LAMBDA**4
        ) / (LAMBDA + LAMBDA**2 + LAMBDA**3 + LAMBDA**4)
        
        return similarity
    
    def register_field_state(self, 
                           name: str, 
                           exemplar_fields: List[np.ndarray],
                           description: str = None) -> bool:
        """
        Register a new field state in the recognition system.
        
        Args:
            name: Unique identifier for the field state
            exemplar_fields: List of example fields in this state
            description: Optional description of the state
            
        Returns:
            Success status
        """
        if name in self.field_states:
            print(f"Field state '{name}' already exists.")
            return False
            
        if not exemplar_fields:
            print("No exemplar fields provided.")
            return False
            
        # Extract features from all exemplars
        all_features = [self._extract_pattern_features(field) for field in exemplar_fields]
        
        # Combine features to create representative state
        combined_features = self._combine_exemplar_features(all_features)
        
        # Store the field state
        self.field_states[name] = {
            'features': combined_features,
            'exemplar_count': len(exemplar_fields),
            'description': description,
            'creation_time': np.datetime64('now')
        }
        
        return True
    
    def _combine_exemplar_features(self, features_list: List[Dict]) -> Dict:
        """Combine multiple exemplar features into a representative state."""
        if not features_list:
            return {}
            
        combined = {}
        n = len(features_list)
        
        # Average numerical features
        numerical_keys = ['mean', 'std', 'min', 'max', 'phi_alignment']
        for key in numerical_keys:
            if all(key in f for f in features_list):
                combined[key] = sum(f[key] for f in features_list) / n
                
        # For shape, use the most common one
        if 'shape' in features_list[0]:
            shape_counts = {}
            for f in features_list:
                shape = f['shape']
                shape_str = str(shape)
                shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            
            # Find most common shape
            most_common = max(shape_counts.items(), key=lambda x: x[1])
            combined['shape'] = eval(most_common[0])  # Convert back to tuple
            combined['dimensions'] = len(combined['shape'])
            
        # Combine histograms if present
        if all('histogram' in f for f in features_list):
            # Ensure all histograms have the same length
            hist_len = len(features_list[0]['histogram'])
            if all(len(f['histogram']) == hist_len for f in features_list):
                # Sum histograms
                combined_hist = sum(f['histogram'] for f in features_list)
                # Normalize
                combined['histogram'] = combined_hist / np.sum(combined_hist)
            
        # Gradient information if present
        grad_keys = ['gradient_mean', 'gradient_std']
        for key in grad_keys:
            if all(key in f for f in features_list):
                # Average each dimension separately
                dims = len(features_list[0][key])
                combined[key] = []
                
                for d in range(dims):
                    avg_val = sum(f[key][d] for f in features_list) / n
                    combined[key].append(avg_val)
                    
        return combined
    
    def analyze_field(self, field_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete analysis of a quantum field, identifying patterns and states.
        
        Args:
            field_data: The quantum field data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Extract field features
        field_features = self._extract_pattern_features(field_data)
        self.field_metrics = field_features
        
        # Find pattern matches
        pattern_matches = self.match_pattern(field_data)
        
        # Recognize field state
        state_probabilities = self.recognize_field_state(field_data)
        
        # Calculate phi-harmonic metrics
        phi_metrics = self._calculate_phi_metrics(field_data)
        
        # Assemble complete analysis
        analysis = {
            'field_shape': field_data.shape,
            'field_features': field_features,
            'pattern_matches': pattern_matches,
            'state_probabilities': state_probabilities,
            'phi_metrics': phi_metrics,
            'analysis_time': np.datetime64('now')
        }
        
        return analysis
    
    def _calculate_phi_metrics(self, field_data: np.ndarray) -> Dict[str, float]:
        """Calculate phi-harmonic metrics for a field."""
        metrics = {}
        
        # Basic phi alignment
        phi_alignment_scores = []
        flat_data = field_data.flatten()
        
        # Sample a subset of points for large fields
        if flat_data.size > 1000:
            indices = np.random.choice(flat_data.size, 1000, replace=False)
            flat_data = flat_data[indices]
        
        # 1. Distance to nearest phi multiple
        phi_multiples = np.array([PHI * i for i in range(-3, 4)])
        distances1 = np.min(np.abs(flat_data[:, np.newaxis] - phi_multiples), axis=1)
        phi_alignment_scores.append(1.0 - np.mean(distances1) / PHI)
        
        # 2. Distance to nearest phi power
        phi_powers = np.array([PHI ** i for i in range(-2, 3)])
        distances2 = np.min(np.abs(flat_data[:, np.newaxis] - phi_powers), axis=1)
        phi_alignment_scores.append(1.0 - np.mean(distances2) / PHI)
        
        metrics['phi_alignment'] = np.mean(phi_alignment_scores)
        
        # Measure field resonance with sacred frequencies
        if len(field_data.shape) >= 2:
            # Get frequency domain representation
            fft = np.fft.fftn(field_data)
            fft_mag = np.abs(fft)
            fft_mag_norm = fft_mag / np.max(fft_mag)
            
            # Calculate resonance at each sacred frequency
            freq_resonance = {}
            for name, freq in SACRED_FREQUENCIES.items():
                # Normalize frequency to [0, 1] range for the FFT
                norm_freq = freq / 1000.0
                
                # Find closest bin in FFT
                resonance_val = 0
                for dim in range(len(field_data.shape)):
                    size = field_data.shape[dim]
                    indices = np.fft.fftfreq(size)
                    closest_idx = np.argmin(np.abs(indices - norm_freq))
                    
                    # Get resonance from this dimension
                    if dim == 0:
                        slice_obj = tuple([closest_idx] + [slice(None)] * (len(field_data.shape) - 1))
                    elif dim == 1:
                        slice_obj = tuple([slice(None), closest_idx] + [slice(None)] * (len(field_data.shape) - 2))
                    else:  # dim == 2
                        slice_obj = tuple([slice(None), slice(None), closest_idx])
                        
                    dim_resonance = np.mean(fft_mag_norm[slice_obj])
                    resonance_val += dim_resonance
                
                # Average across dimensions
                resonance_val /= len(field_data.shape)
                freq_resonance[name] = resonance_val
                
            metrics['frequency_resonance'] = freq_resonance
        
        # Additional metrics for 3D fields
        if len(field_data.shape) == 3:
            # Calculate toroidal metrics if field is 3D
            try:
                metrics['toroidal_metrics'] = self._calculate_toroidal_metrics(field_data)
            except Exception:
                # Skip if calculation fails
                pass
        
        return metrics
    
    def _calculate_toroidal_metrics(self, field_data: np.ndarray) -> Dict[str, float]:
        """Calculate metrics specific to toroidal fields."""
        width, height, depth = field_data.shape
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            np.linspace(-1, 1, depth),
            indexing='ij'
        )
        
        # Calculate field gradients
        grad_x, grad_y, grad_z = np.gradient(field_data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Calculate vorticity (curl)
        curl_x = np.gradient(grad_z, axis=1) - np.gradient(grad_y, axis=2)
        curl_y = np.gradient(grad_x, axis=2) - np.gradient(grad_z, axis=0)
        curl_z = np.gradient(grad_y, axis=0) - np.gradient(grad_x, axis=1)
        curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        
        # Calculate divergence
        div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1) + np.gradient(grad_z, axis=2)
        
        # Toroidal flow balance (input/output cycle)
        flow_balance = 1.0 - np.mean(np.abs(div)) / (np.mean(grad_mag) + 1e-10)
        
        # Circulation metric
        circulation = np.mean(curl_mag) / (np.mean(grad_mag) + 1e-10)
        
        # Torus detection - look for characteristic toroidal pattern
        # Distance from a ring with radius PHI in the xy-plane
        ring_distance = np.sqrt((np.sqrt(X**2 + Y**2) - PHI)**2 + Z**2)
        
        # Check if field values peak along the torus shape with radius PHI
        torus_mask = np.abs(ring_distance - LAMBDA) < 0.2
        field_values_on_torus = field_data[torus_mask]
        field_values_elsewhere = field_data[~torus_mask]
        
        if len(field_values_on_torus) > 0 and len(field_values_elsewhere) > 0:
            torus_intensity = np.mean(field_values_on_torus) / (np.mean(field_values_elsewhere) + 1e-10)
        else:
            torus_intensity = 0.0
            
        # Calculate toroidal coherence
        toroidal_coherence = (flow_balance * 0.4 + 
                             circulation * 0.3 + 
                             min(1.0, torus_intensity / PHI) * 0.3)
        
        return {
            'flow_balance': flow_balance,
            'circulation': circulation,
            'torus_intensity': torus_intensity,
            'toroidal_coherence': toroidal_coherence
        }