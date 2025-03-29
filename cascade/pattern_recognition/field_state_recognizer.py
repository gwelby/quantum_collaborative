"""
CASCADE⚡𓂧φ∞ Field State Recognizer

System for recognizing and categorizing quantum field states using
trained models and phi-harmonic principles.
"""

import numpy as np
import sys
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

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


class FieldStateRecognizer:
    """
    System for recognizing quantum field states using a combination of
    feature extraction, statistical analysis, and phi-harmonic principles.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the field state recognizer.
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.state_definitions = self._initialize_state_definitions()
        self.trained_models = {}
        self.state_templates = {}
        
        # Try to load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _initialize_state_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize built-in state definitions."""
        states = {}
        
        # Ground state (432 Hz)
        states['ground'] = {
            'description': "Ground State - Earth resonance (432 Hz)",
            'frequencies': [432],
            'coherence_range': [0.6, 0.8],
            'features': {
                'phi_alignment': 0.65,
                'gradient_strength': 'low',
                'symmetry': 'vertical',
                'field_distribution': 'concentrated_bottom'
            }
        }
        
        # Creation state (528 Hz)
        states['creation'] = {
            'description': "Creation Point - Phi-harmonic generation (528 Hz)",
            'frequencies': [528],
            'coherence_range': [0.7, 0.85],
            'features': {
                'phi_alignment': 0.75,
                'gradient_strength': 'medium',
                'symmetry': 'spiral',
                'field_distribution': 'fractal'
            }
        }
        
        # Heart field state (594 Hz)
        states['heart'] = {
            'description': "Heart Field - Emotional coherence (594 Hz)",
            'frequencies': [594],
            'coherence_range': [0.75, 0.9],
            'features': {
                'phi_alignment': 0.85,
                'gradient_strength': 'medium',
                'symmetry': 'toroidal',
                'field_distribution': 'centered'
            }
        }
        
        # Voice flow state (672 Hz)
        states['voice'] = {
            'description': "Voice Flow - Harmonic expression (672 Hz)",
            'frequencies': [672],
            'coherence_range': [0.8, 0.9],
            'features': {
                'phi_alignment': 0.88,
                'gradient_strength': 'high',
                'symmetry': 'wave',
                'field_distribution': 'harmonic'
            }
        }
        
        # Vision gate state (720 Hz)
        states['vision'] = {
            'description': "Vision Gate - Multiple timelines (720 Hz)",
            'frequencies': [720],
            'coherence_range': [0.85, 0.95],
            'features': {
                'phi_alignment': 0.92,
                'gradient_strength': 'high',
                'symmetry': 'radial',
                'field_distribution': 'multidimensional'
            }
        }
        
        # Unity wave state (768 Hz)
        states['unity'] = {
            'description': "Unity Wave - Field unification (768 Hz)",
            'frequencies': [768],
            'coherence_range': [0.9, 1.0],
            'features': {
                'phi_alignment': 0.95,
                'gradient_strength': 'very_high',
                'symmetry': 'complete',
                'field_distribution': 'unified'
            }
        }
        
        # Transcendent state (888 Hz)
        states['transcendent'] = {
            'description': "Transcendent Field - Full integration (888 Hz)",
            'frequencies': [888],
            'coherence_range': [0.95, 1.0],
            'features': {
                'phi_alignment': 0.99,
                'gradient_strength': 'highest',
                'symmetry': 'hyperdimensional',
                'field_distribution': 'phi_fractal'
            }
        }
        
        return states
    
    def extract_field_state_features(self, field_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from field data relevant for state recognition.
        
        Args:
            field_data: NumPy array containing the quantum field
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(field_data)
        features['std'] = np.std(field_data)
        features['min'] = np.min(field_data)
        features['max'] = np.max(field_data)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(field_data, bins=10, range=(features['min'], features['max']))
        features['histogram'] = hist / np.sum(hist)  # Normalize
        
        # Entropy
        entropy = -np.sum(features['histogram'] * np.log2(features['histogram'] + 1e-10))
        features['entropy'] = entropy
        
        # Distribution symmetry
        if len(field_data.shape) >= 2:
            # For 2D or 3D fields, check symmetry
            symm_scores = {}
            
            # Vertical symmetry
            if len(field_data.shape) == 2:
                mid_idx = field_data.shape[0] // 2
                upper_half = field_data[:mid_idx, :]
                lower_half = field_data[mid_idx:, :]
                
                if lower_half.shape[0] > upper_half.shape[0]:
                    lower_half = lower_half[:upper_half.shape[0], :]
                elif upper_half.shape[0] > lower_half.shape[0]:
                    upper_half = upper_half[:lower_half.shape[0], :]
                
                # Flip lower half and calculate similarity
                lower_flipped = np.flip(lower_half, axis=0)
                diff = np.abs(upper_half - lower_flipped)
                symm_scores['vertical'] = 1.0 - np.mean(diff) / (np.max(field_data) - np.min(field_data) + 1e-10)
                
                # Horizontal symmetry
                mid_idx = field_data.shape[1] // 2
                left_half = field_data[:, :mid_idx]
                right_half = field_data[:, mid_idx:]
                
                if right_half.shape[1] > left_half.shape[1]:
                    right_half = right_half[:, :left_half.shape[1]]
                elif left_half.shape[1] > right_half.shape[1]:
                    left_half = left_half[:, :right_half.shape[1]]
                
                # Flip right half and calculate similarity
                right_flipped = np.flip(right_half, axis=1)
                diff = np.abs(left_half - right_flipped)
                symm_scores['horizontal'] = 1.0 - np.mean(diff) / (np.max(field_data) - np.min(field_data) + 1e-10)
                
                # Radial symmetry (approximation)
                center_i, center_j = field_data.shape[0] // 2, field_data.shape[1] // 2
                i_coords, j_coords = np.indices(field_data.shape)
                r = np.sqrt((i_coords - center_i)**2 + (j_coords - center_j)**2)
                
                # Group by radius
                max_r = int(np.max(r))
                radial_symm = 0
                for radius in range(1, max_r):
                    mask = np.logical_and(r > radius - 0.5, r < radius + 0.5)
                    if np.sum(mask) > 0:
                        values = field_data[mask]
                        radial_symm += 1.0 - np.std(values) / (np.max(field_data) - np.min(field_data) + 1e-10)
                
                if max_r > 0:
                    symm_scores['radial'] = radial_symm / max_r
                else:
                    symm_scores['radial'] = 0
            
            elif len(field_data.shape) == 3:
                # For 3D, calculate simpler metrics
                center = [s // 2 for s in field_data.shape]
                distances = np.sqrt(
                    np.sum(
                        [(np.indices(field_data.shape)[i] - center[i])**2 for i in range(3)],
                        axis=0
                    )
                )
                
                # Spherical correlation
                values_by_dist = {}
                max_dist = int(np.max(distances))
                for d in range(1, max_dist + 1):
                    mask = np.logical_and(distances >= d-0.5, distances < d+0.5)
                    if np.sum(mask) > 0:
                        values_by_dist[d] = np.mean(field_data[mask])
                
                # Calculate smoothness of radial distribution (higher = more radially symmetric)
                if len(values_by_dist) > 2:
                    dists = sorted(values_by_dist.keys())
                    values = [values_by_dist[d] for d in dists]
                    diffs = np.abs(np.diff(values))
                    radial_smoothness = 1.0 - np.mean(diffs) / (np.max(field_data) - np.min(field_data) + 1e-10)
                    symm_scores['radial_3d'] = radial_smoothness
            
            features['symmetry_scores'] = symm_scores
        
        # Gradient strength and distribution
        if len(field_data.shape) > 1:
            # Calculate gradients
            grads = np.gradient(field_data)
            
            # Calculate gradient magnitude
            if len(grads) == 2:
                grad_mag = np.sqrt(grads[0]**2 + grads[1]**2)
            elif len(grads) == 3:
                grad_mag = np.sqrt(grads[0]**2 + grads[1]**2 + grads[2]**2)
            else:
                grad_mag = np.abs(grads[0])
            
            features['gradient_mean'] = np.mean(grad_mag)
            features['gradient_std'] = np.std(grad_mag)
            features['gradient_max'] = np.max(grad_mag)
            
            # Gradient distribution
            hist, _ = np.histogram(grad_mag, bins=10)
            features['gradient_histogram'] = hist / np.sum(hist)
            
            # Directional gradient information
            grad_directions = {}
            if len(grads) >= 2:
                # For 2D or 3D
                grad_directions['x_mean'] = np.mean(grads[0])
                grad_directions['y_mean'] = np.mean(grads[1])
                
                if len(grads) == 3:
                    grad_directions['z_mean'] = np.mean(grads[2])
                    
            features['gradient_directions'] = grad_directions
        
        # Phi-harmonic features
        phi_features = self._extract_phi_features(field_data)
        features.update(phi_features)
        
        # Field distribution characteristics
        if len(field_data.shape) == 2:
            # 2D field distribution
            height, width = field_data.shape
            
            # Create normalized position maps
            y_norm = np.linspace(0, 1, height)[:, np.newaxis]
            x_norm = np.linspace(0, 1, width)[np.newaxis, :]
            
            # Calculate weighted center of field energy
            total_energy = np.sum(field_data)
            if total_energy > 0:
                center_y = np.sum(y_norm * field_data) / total_energy
                center_x = np.sum(x_norm * field_data) / total_energy
                features['energy_center'] = (center_y, center_x)
                
                # Check if energy is concentrated at bottom
                bottom_ratio = np.sum(field_data[:height//2, :]) / (total_energy + 1e-10)
                features['bottom_energy_ratio'] = bottom_ratio
                
                # Check if energy is concentrated in center
                center_mask = np.zeros((height, width), dtype=bool)
                cy, cx = height // 2, width // 2
                radius = min(height, width) // 4
                
                for i in range(height):
                    for j in range(width):
                        if (i - cy)**2 + (j - cx)**2 < radius**2:
                            center_mask[i, j] = True
                            
                center_energy = np.sum(field_data[center_mask])
                center_ratio = center_energy / (total_energy + 1e-10)
                features['center_energy_ratio'] = center_ratio
        
        elif len(field_data.shape) == 3:
            # 3D field distribution
            depth, height, width = field_data.shape
            
            # Create normalized position maps
            z_norm = np.linspace(0, 1, depth)[:, np.newaxis, np.newaxis]
            y_norm = np.linspace(0, 1, height)[np.newaxis, :, np.newaxis]
            x_norm = np.linspace(0, 1, width)[np.newaxis, np.newaxis, :]
            
            # Calculate weighted center of field energy
            total_energy = np.sum(field_data)
            if total_energy > 0:
                center_z = np.sum(z_norm * field_data) / total_energy
                center_y = np.sum(y_norm * field_data) / total_energy
                center_x = np.sum(x_norm * field_data) / total_energy
                features['energy_center'] = (center_z, center_y, center_x)
                
                # Check if energy is concentrated at bottom
                bottom_ratio = np.sum(field_data[:depth//2, :, :]) / (total_energy + 1e-10)
                features['bottom_energy_ratio'] = bottom_ratio
                
                # Check if energy is concentrated in center
                center_mask = np.zeros((depth, height, width), dtype=bool)
                cz, cy, cx = depth // 2, height // 2, width // 2
                radius = min(depth, height, width) // 4
                
                for i in range(depth):
                    for j in range(height):
                        for k in range(width):
                            if (i - cz)**2 + (j - cy)**2 + (k - cx)**2 < radius**2:
                                center_mask[i, j, k] = True
                                
                center_energy = np.sum(field_data[center_mask])
                center_ratio = center_energy / (total_energy + 1e-10)
                features['center_energy_ratio'] = center_ratio
        
        return features
    
    def _extract_phi_features(self, field_data: np.ndarray) -> Dict[str, float]:
        """Extract phi-harmonic features from field data."""
        phi_features = {}
        
        # Flatten and normalize field for phi analysis
        flat_data = field_data.flatten()
        
        # Sample subset for large fields
        if flat_data.size > 1000:
            indices = np.random.choice(flat_data.size, 1000, replace=False)
            flat_data = flat_data[indices]
        
        # Normalize to [0,1] range for consistency
        data_range = np.max(flat_data) - np.min(flat_data)
        if data_range > 0:
            flat_data_norm = (flat_data - np.min(flat_data)) / data_range
        else:
            flat_data_norm = np.zeros_like(flat_data)
        
        # Calculate phi alignment
        phi_powers = np.array([PHI ** i for i in range(-3, 4)])
        distances = np.min(np.abs(flat_data_norm[:, np.newaxis] - phi_powers), axis=1)
        phi_alignment = 1.0 - np.mean(distances) / PHI
        phi_features['phi_alignment'] = phi_alignment
        
        # Calculate phi ratio in value distribution
        hist, bin_edges = np.histogram(flat_data_norm, bins=20, range=(0, 1))
        hist_norm = hist / np.sum(hist)
        
        # Check golden ratio in histogram distribution
        ratios = []
        for i in range(len(hist_norm) - 1):
            if hist_norm[i+1] > 0:
                ratios.append(hist_norm[i] / hist_norm[i+1])
                
        if ratios:
            # Calculate how close ratios are to phi
            phi_ratio_score = 1.0 - np.mean(np.abs(np.array(ratios) - PHI)) / PHI
            phi_features['phi_ratio_score'] = phi_ratio_score
        
        # Detect phi-harmonic frequencies in field
        if min(field_data.shape) >= 8:
            try:
                # Compute FFT
                fft = np.fft.fftn(field_data)
                fft_mag = np.abs(fft)
                
                # Normalize
                fft_mag_norm = fft_mag / np.max(fft_mag)
                
                # Calculate resonance at phi-harmonic frequencies
                phi_harmonic_resonance = 0
                
                # Get frequency bins
                freq_bins = []
                for dim in range(len(field_data.shape)):
                    freq_bins.append(np.fft.fftfreq(field_data.shape[dim]))
                
                # Check resonance at phi-related frequency ratios
                phi_freqs = [1/PHI**2, 1/PHI, 1.0, PHI, PHI**2]
                
                for phi_f in phi_freqs:
                    resonance_val = 0
                    
                    for dim in range(len(field_data.shape)):
                        freqs = freq_bins[dim]
                        # Find bin closest to phi-based frequency
                        bin_idx = np.argmin(np.abs(freqs - phi_f/5))  # Scaled to appropriate range
                        
                        # Create slice to get this frequency component
                        slice_obj = [slice(None)] * len(field_data.shape)
                        slice_obj[dim] = bin_idx
                        
                        # Get mean magnitude at this frequency across other dimensions
                        freq_slice = fft_mag_norm[tuple(slice_obj)]
                        resonance_val += np.mean(freq_slice)
                    
                    # Average across dimensions
                    resonance_val /= len(field_data.shape)
                    phi_harmonic_resonance += resonance_val
                
                # Average resonance across all phi frequencies
                phi_harmonic_resonance /= len(phi_freqs)
                phi_features['phi_harmonic_resonance'] = phi_harmonic_resonance
                
            except Exception as e:
                # Skip if frequency analysis fails
                pass
                
        # Check for toroidal structure in 3D fields
        if len(field_data.shape) == 3:
            try:
                depth, height, width = field_data.shape
                
                # Create coordinate grids
                z, y, x = np.meshgrid(
                    np.linspace(-1, 1, depth),
                    np.linspace(-1, 1, height),
                    np.linspace(-1, 1, width),
                    indexing='ij'
                )
                
                # Calculate distance from torus with phi-based proportions
                major_radius = PHI * 0.5  # Scaled
                minor_radius = LAMBDA * 0.5  # Scaled
                
                # Distance from ring in xy-plane
                r_xy = np.sqrt(x**2 + y**2)
                torus_dist = np.sqrt((r_xy - major_radius)**2 + z**2) - minor_radius
                
                # Check if field values are concentrated near the torus surface
                torus_mask = np.abs(torus_dist) < 0.1
                
                if np.sum(torus_mask) > 0:
                    # Compare field values on torus vs. elsewhere
                    torus_value = np.mean(field_data[torus_mask])
                    non_torus_value = np.mean(field_data[~torus_mask])
                    
                    if non_torus_value > 0:
                        torus_ratio = torus_value / non_torus_value
                    else:
                        torus_ratio = 1.0
                        
                    # Convert to a score
                    torus_score = 1.0 - 1.0 / (1.0 + torus_ratio / PHI)
                    phi_features['torus_alignment'] = torus_score
            except Exception as e:
                # Skip if torus analysis fails
                pass
        
        return phi_features
    
    def recognize_state(self, field_data: np.ndarray) -> Dict[str, float]:
        """
        Recognize the state of a quantum field.
        
        Args:
            field_data: NumPy array containing the quantum field
            
        Returns:
            Dictionary mapping state names to probability scores
        """
        # Extract features from the field
        features = self.extract_field_state_features(field_data)
        
        # If we have trained ML models, use them
        if self.trained_models:
            return self._recognize_with_models(features)
        
        # Otherwise use rule-based recognition with built-in states
        return self._recognize_with_rules(features)
    
    def _recognize_with_rules(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Rule-based state recognition using built-in state definitions."""
        scores = {}
        
        for state_name, state_def in self.state_definitions.items():
            state_score = 0.0
            score_count = 0
            
            # Check phi alignment
            if 'phi_alignment' in features and 'phi_alignment' in state_def['features']:
                target_alignment = state_def['features']['phi_alignment']
                actual_alignment = features['phi_alignment']
                
                # Calculate score based on proximity to target
                alignment_score = 1.0 - min(abs(actual_alignment - target_alignment), 0.5) / 0.5
                state_score += alignment_score
                score_count += 1
            
            # Check gradient strength
            if 'gradient_mean' in features and 'gradient_strength' in state_def['features']:
                grad_strength = state_def['features']['gradient_strength']
                actual_grad = features['gradient_mean']
                
                # Convert qualitative strength to numeric range
                strength_map = {
                    'lowest': 0.1,
                    'very_low': 0.2,
                    'low': 0.3,
                    'medium': 0.5,
                    'high': 0.7,
                    'very_high': 0.8,
                    'highest': 0.9
                }
                
                if grad_strength in strength_map:
                    target_grad = strength_map[grad_strength]
                    
                    # Normalize actual gradient to [0,1] range (assuming max gradient ~= 1.0)
                    actual_grad_norm = min(1.0, actual_grad)
                    
                    # Calculate score based on proximity to target
                    grad_score = 1.0 - min(abs(actual_grad_norm - target_grad), 0.5) / 0.5
                    state_score += grad_score
                    score_count += 1
            
            # Check symmetry
            if 'symmetry_scores' in features and 'symmetry' in state_def['features']:
                target_symmetry = state_def['features']['symmetry']
                
                # Map qualitative symmetry type to actual symmetry scores
                if target_symmetry == 'vertical' and 'vertical' in features['symmetry_scores']:
                    symm_score = features['symmetry_scores']['vertical']
                    state_score += symm_score
                    score_count += 1
                    
                elif target_symmetry == 'horizontal' and 'horizontal' in features['symmetry_scores']:
                    symm_score = features['symmetry_scores']['horizontal']
                    state_score += symm_score
                    score_count += 1
                    
                elif target_symmetry == 'radial' and 'radial' in features['symmetry_scores']:
                    symm_score = features['symmetry_scores']['radial']
                    state_score += symm_score
                    score_count += 1
                    
                elif target_symmetry == 'toroidal' and 'torus_alignment' in features:
                    symm_score = features['torus_alignment']
                    state_score += symm_score
                    score_count += 1
                    
                elif target_symmetry in ['complete', 'hyperdimensional']:
                    # Check all symmetry types and take average
                    all_symm = 0.0
                    symm_count = 0
                    
                    for symm_type, score in features['symmetry_scores'].items():
                        all_symm += score
                        symm_count += 1
                        
                    if symm_count > 0:
                        symm_score = all_symm / symm_count
                        state_score += symm_score
                        score_count += 1
            
            # Check field distribution
            if 'field_distribution' in state_def['features']:
                target_dist = state_def['features']['field_distribution']
                
                # Check various distribution types
                if target_dist == 'concentrated_bottom' and 'bottom_energy_ratio' in features:
                    # Higher ratio = more energy at bottom
                    dist_score = features['bottom_energy_ratio']
                    state_score += dist_score
                    score_count += 1
                    
                elif target_dist == 'centered' and 'center_energy_ratio' in features:
                    # Higher ratio = more energy in center
                    dist_score = features['center_energy_ratio']
                    state_score += dist_score
                    score_count += 1
                    
                elif target_dist in ['harmonic', 'phi_fractal'] and 'phi_harmonic_resonance' in features:
                    # Higher phi harmonic resonance matches these states
                    dist_score = features['phi_harmonic_resonance']
                    state_score += dist_score
                    score_count += 1
                    
                elif target_dist == 'unified' and 'entropy' in features:
                    # Lower entropy indicates more unified field
                    max_entropy = np.log2(10)  # Based on 10 histogram bins
                    entropy_ratio = 1.0 - features['entropy'] / max_entropy
                    state_score += entropy_ratio
                    score_count += 1
            
            # Only include score if we have at least some matching features
            if score_count > 0:
                scores[state_name] = state_score / score_count
            else:
                scores[state_name] = 0.1  # Minimal baseline probability
        
        # Normalize scores to probabilities
        total = sum(scores.values())
        if total > 0:
            probabilities = {k: v / total for k, v in scores.items()}
        else:
            # Equal probability if no scores
            prob_value = 1.0 / len(scores)
            probabilities = {k: prob_value for k in scores}
            
        return probabilities
    
    def _recognize_with_models(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Use trained models for state recognition."""
        # Implement various model types depending on what's available
        
        # Simple template matching
        if self.state_templates:
            return self._template_matching(features)
            
        # If no valid model is available, fall back to rule-based
        return self._recognize_with_rules(features)
    
    def _template_matching(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Simple template-based recognition using stored templates."""
        scores = {}
        
        # Features to use for comparison
        compare_features = [
            'phi_alignment',
            'entropy',
            'gradient_mean',
            'mean',
            'std'
        ]
        
        # Check each template
        for state_name, template in self.state_templates.items():
            feature_scores = []
            
            # Compare each feature
            for feature in compare_features:
                if feature in features and feature in template:
                    # Calculate similarity based on feature type
                    if isinstance(features[feature], (int, float)) and isinstance(template[feature], (int, float)):
                        # For numeric values, use relative difference
                        max_val = max(abs(features[feature]), abs(template[feature]))
                        if max_val > 0:
                            diff = abs(features[feature] - template[feature]) / max_val
                            sim = 1.0 - min(1.0, diff)
                        else:
                            sim = 1.0
                            
                        feature_scores.append(sim)
                        
                    elif isinstance(features[feature], np.ndarray) and isinstance(template[feature], np.ndarray):
                        # For arrays (like histograms), use array similarity
                        if features[feature].shape == template[feature].shape:
                            diff = np.mean(np.abs(features[feature] - template[feature]))
                            sim = 1.0 - min(1.0, diff)
                            feature_scores.append(sim)
            
            # Calculate overall score using phi-weighted average
            if feature_scores:
                # Phi-weighted scoring gives more weight to higher similarity features
                feature_scores.sort(reverse=True)
                
                weights = np.array([LAMBDA**(i+1) for i in range(len(feature_scores))])
                weights = weights / np.sum(weights)
                
                weighted_score = np.sum(np.array(feature_scores) * weights)
                scores[state_name] = weighted_score
            else:
                scores[state_name] = 0.1  # Minimal probability if no features match
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            probabilities = {k: v / total for k, v in scores.items()}
        else:
            # Equal probability if no scores
            prob_value = 1.0 / len(scores)
            probabilities = {k: prob_value for k in scores}
            
        return probabilities
    
    def train_on_examples(self, 
                         examples: Dict[str, List[np.ndarray]],
                         method: str = 'template') -> bool:
        """
        Train the recognizer on example fields for each state.
        
        Args:
            examples: Dictionary mapping state names to lists of example fields
            method: Training method ('template', 'statistical', etc.)
            
        Returns:
            Success status
        """
        if method == 'template':
            return self._train_templates(examples)
        else:
            print(f"Unsupported training method: {method}")
            return False
    
    def _train_templates(self, examples: Dict[str, List[np.ndarray]]) -> bool:
        """Train template-based recognizer using example fields."""
        self.state_templates = {}
        
        for state_name, fields in examples.items():
            if not fields:
                continue
                
            # Extract features from all examples
            all_features = [self.extract_field_state_features(field) for field in fields]
            
            # Combine features into a template
            template = {}
            
            # For numeric features, take average
            numeric_features = [
                'mean', 'std', 'min', 'max', 'phi_alignment', 
                'entropy', 'gradient_mean', 'gradient_std'
            ]
            
            for feature in numeric_features:
                if all(feature in f for f in all_features):
                    template[feature] = np.mean([f[feature] for f in all_features])
            
            # For histogram features, take average of histograms
            histogram_features = ['histogram', 'gradient_histogram']
            
            for feature in histogram_features:
                if all(feature in f for f in all_features):
                    hist_shape = all_features[0][feature].shape
                    if all(f[feature].shape == hist_shape for f in all_features):
                        template[feature] = np.mean([f[feature] for f in all_features], axis=0)
            
            # Store template
            self.state_templates[state_name] = template
            
        return True
    
    def save_model(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare data to save
            model_data = {
                'state_templates': self.state_templates,
                'trained_models': self.trained_models,
                'state_definitions': self.state_definitions,
                'phi_constants': {
                    'phi': PHI,
                    'lambda': LAMBDA,
                    'phi_phi': PHI_PHI
                },
                'version': '1.0.0'
            }
            
            # Save to file
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
                
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(path):
                print(f"Model file not found: {path}")
                return False
                
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model format
            required_keys = ['state_templates', 'state_definitions', 'phi_constants']
            if not all(key in model_data for key in required_keys):
                print("Invalid model format")
                return False
            
            # Check phi constants match
            phi_constants = model_data['phi_constants']
            if abs(phi_constants['phi'] - PHI) > 1e-10:
                print(f"Warning: Model uses different PHI value: {phi_constants['phi']}")
            
            # Load model components
            self.state_templates = model_data['state_templates']
            self.trained_models = model_data.get('trained_models', {})
            
            # Update state definitions if available
            if 'state_definitions' in model_data:
                self.state_definitions.update(model_data['state_definitions'])
                
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False