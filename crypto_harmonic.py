#!/usr/bin/env python3
"""
Crypto Harmonic System - Phi-Based Cryptographic Analysis Tool

This module provides phi-harmonic cryptocurrency analysis tools, leveraging
sacred constants and quantum field principles for pattern recognition and
price movement prediction.
"""

import os
import time
import json
import math
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

try:
    import sacred_constants as sc
except ImportError:
    print("Warning: sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 2.1784575679375995
        
        SACRED_FREQUENCIES = {
            'love': 528,      # Creation/healing
            'unity': 432,     # Grounding/stability
            'cascade': 594,   # Heart-centered integration
            'truth': 672,     # Voice expression
            'vision': 720,    # Expanded perception
            'oneness': 768,   # Unity consciousness
        }

# CUDA support if available
try:
    from cuda.core.experimental import Device, Stream, Program, Linker, Module
    from cuda.core.experimental import Context, Memory
    CUDA_AVAILABLE = True
    print("CUDA acceleration available. Using GPU for pattern recognition.")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA modules not available. Falling back to CPU computation.")

# Constants for crypto analysis
CRYPTO_TIME_PERIODS = {
    'minute': 60,
    'hour': 60 * 60,
    'day': 24 * 60 * 60,
    'week': 7 * 24 * 60 * 60,
    'month': 30 * 24 * 60 * 60,
    'year': 365 * 24 * 60 * 60
}

# Phi-harmonic Fibonacci levels for crypto analysis
PHI_FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 4.236]

@dataclass
class CryptoConfig:
    """Configuration for the Crypto Harmonic Analyzer"""
    use_cuda: bool = CUDA_AVAILABLE
    phi_precision: int = 12  # Decimal places for phi calculations
    coherence_threshold: float = 0.65
    harmonic_precision: int = 5
    confidence_level: float = 0.85
    pattern_recognition_iterations: int = 1000
    quantum_field_size: int = 128  # Size of quantum field for pattern mapping
    use_sacred_frequencies: bool = True
    pattern_search_depth: int = 21
    price_cycle_length: int = int(sc.PHI * 100)  # Phi-based cycle length
    mayer_multiple_threshold: float = sc.PHI
    
@dataclass
class CryptoPrice:
    """Single cryptocurrency price data point"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(
            timestamp=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        )

@dataclass
class CryptoPriceHistory:
    """Price history for a cryptocurrency"""
    symbol: str
    prices: List[CryptoPrice] = field(default_factory=list)
    timeframe: str = 'day'
    
    def add_price(self, price: CryptoPrice):
        """Add a price data point"""
        self.prices.append(price)
        # Sort by timestamp to ensure chronological order
        self.prices.sort(key=lambda x: x.timestamp)
    
    def get_close_prices(self) -> np.ndarray:
        """Get array of close prices"""
        return np.array([price.close for price in self.prices])
    
    def get_high_prices(self) -> np.ndarray:
        """Get array of high prices"""
        return np.array([price.high for price in self.prices])
    
    def get_low_prices(self) -> np.ndarray:
        """Get array of low prices"""
        return np.array([price.low for price in self.prices])
    
    def get_volumes(self) -> np.ndarray:
        """Get array of volumes"""
        return np.array([price.volume for price in self.prices])
    
    def get_timestamps(self) -> np.ndarray:
        """Get array of timestamps"""
        return np.array([price.timestamp for price in self.prices])
    
    def get_dates(self) -> List[datetime]:
        """Get list of datetime objects"""
        return [datetime.fromtimestamp(price.timestamp) for price in self.prices]
    
    def save_to_file(self, filepath: str):
        """Save price history to file"""
        data = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'prices': [price.to_dict() for price in self.prices]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load price history from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history = cls(
            symbol=data['symbol'],
            timeframe=data.get('timeframe', 'day')
        )
        
        for price_data in data['prices']:
            history.add_price(CryptoPrice.from_dict(price_data))
        
        return history
    
    def generate_sample_data(self, days: int = 365, start_price: float = 30000.0, volatility: float = 0.03):
        """Generate sample price data for testing"""
        self.prices = []
        price = start_price
        
        # Start timestamp at current time minus days
        start_time = int(time.time()) - (days * 24 * 60 * 60)
        
        # Add phi-harmonic cycles
        phi_cycle_length = int(sc.PHI * 10)  # Phi-based cycle length in days
        
        for day in range(days):
            timestamp = start_time + (day * 24 * 60 * 60)
            
            # Add phi-harmonic pattern 
            phi_cycle_position = day % phi_cycle_length
            phi_cycle_factor = math.sin(phi_cycle_position / phi_cycle_length * 2 * math.pi)
            
            # Add secondary phi cycle
            secondary_cycle = math.sin(day / (phi_cycle_length * sc.PHI) * 2 * math.pi)
            
            # Day's random factor
            random_change = random.normalvariate(0, volatility)
            
            # Combine factors with phi-weighting
            daily_change = (
                random_change * (1 - sc.LAMBDA) + 
                phi_cycle_factor * sc.LAMBDA * 0.6 + 
                secondary_cycle * sc.LAMBDA * 0.4
            )
            
            # Update price
            price = price * (1 + daily_change)
            
            # Calculate high and low with phi-based ranges
            price_range = price * volatility * sc.PHI
            high = price + price_range * random.uniform(0.2, 1.0)
            low = price - price_range * random.uniform(0.2, 1.0)
            
            # Ensure low < open < high
            open_price = low + (high - low) * random.uniform(0.2, 0.8)
            
            # Volume also follows phi pattern
            base_volume = price * 1000
            volume_factor = 1 + 0.5 * phi_cycle_factor
            volume = base_volume * volume_factor * random.uniform(0.8, 1.2)
            
            # Add price point
            self.add_price(CryptoPrice(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume
            ))

class HarmonicPatternFinder:
    """
    Finds harmonic patterns in price data based on phi ratios
    """
    
    def __init__(self, config: CryptoConfig = None):
        self.config = config or CryptoConfig()
        self.patterns = self._define_harmonic_patterns()
        self.cuda_module = None
        
        # Initialize CUDA if available
        if self.config.use_cuda and CUDA_AVAILABLE:
            self._init_cuda()
    
    def _init_cuda(self):
        """Initialize CUDA resources"""
        try:
            # Get the CUDA device
            self.device = Device(0)  # Use the first GPU
            print(f"Using GPU: {self.device.name}")
            
            # Compile the pattern recognition kernels
            self.cuda_module = self._compile_cuda_kernels()
            if self.cuda_module:
                print("CUDA kernels compiled successfully")
            
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            self.config.use_cuda = False
    
    def _compile_cuda_kernels(self):
        """Compile CUDA kernels for pattern recognition"""
        if not CUDA_AVAILABLE:
            return None
        
        kernel_source = """
        extern "C" __global__ void find_potential_patterns(
            float *prices, int length, float *patterns, int num_patterns, int *results
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int pattern_idx = blockIdx.y;
            
            if (tid < length - 5 && pattern_idx < num_patterns) {
                float p0 = prices[tid];
                float p1 = prices[tid + 1];
                float p2 = prices[tid + 2];
                float p3 = prices[tid + 3];
                float p4 = prices[tid + 4];
                
                // Pattern definition: [XA, AB, BC, CD] ratios
                float XA_target = patterns[pattern_idx * 4 + 0];
                float AB_target = patterns[pattern_idx * 4 + 1];
                float BC_target = patterns[pattern_idx * 4 + 2];
                float CD_target = patterns[pattern_idx * 4 + 3];
                
                // Calculate actual ratios
                float XA = (p1 - p0) / p0;
                float AB = (p2 - p1) / p1;
                float BC = (p3 - p2) / p2;
                float CD = (p4 - p3) / p3;
                
                // Check if ratios match pattern with tolerance
                float tolerance = 0.05f; // 5% tolerance
                
                if (fabsf(XA - XA_target) < tolerance &&
                    fabsf(AB - AB_target) < tolerance &&
                    fabsf(BC - BC_target) < tolerance &&
                    fabsf(CD - CD_target) < tolerance) {
                    
                    // Store result: found pattern at index
                    atomicAdd(&results[pattern_idx], 1);
                }
            }
        }
        
        extern "C" __global__ void calculate_phi_levels(
            float *prices, int length, float *levels, int num_levels, int *hits
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (tid < length - 1) {
                float price0 = prices[tid];
                float price1 = prices[tid + 1];
                float move = price1 - price0;
                
                // Check each retracement level
                for (int i = 0; i < num_levels; i++) {
                    float level = levels[i];
                    float target = price0 + move * level;
                    
                    // Check if any price came close to this level
                    for (int j = tid + 2; j < min(tid + 10, length); j++) {
                        float price = prices[j];
                        if (fabsf(price - target) / target < 0.01f) { // 1% tolerance
                            atomicAdd(&hits[i], 1);
                            break;
                        }
                    }
                }
            }
        }
        """
        
        try:
            # Compile and link the kernel
            program = Program(kernel_source, compile_options=["-use_fast_math"])
            linker = Linker()
            linker.add_program(program)
            module = linker.link()
            
            return module
        except Exception as e:
            print(f"Error compiling CUDA kernels: {e}")
            return None
    
    def _define_harmonic_patterns(self):
        """Define harmonic patterns based on phi ratios"""
        patterns = {
            "Gartley": {
                "XA": sc.LAMBDA,  # 0.618
                "AB": 0.382,  # Approx 1 - 0.618
                "BC": sc.LAMBDA,  # 0.618
                "CD": 1.272  # Approx SQRT(PHI)
            },
            "Butterfly": {
                "XA": sc.LAMBDA,  # 0.618
                "AB": 0.786,  # Approx PHI - 0.832
                "BC": sc.LAMBDA,  # 0.618
                "CD": 1.618  # PHI
            },
            "Bat": {
                "XA": sc.LAMBDA,  # 0.618
                "AB": 0.382,  # Approx 1 - 0.618
                "BC": sc.LAMBDA,  # 0.618
                "CD": 2.618  # PHI^2
            },
            "Crab": {
                "XA": sc.LAMBDA,  # 0.618
                "AB": 0.382,  # Approx 1 - 0.618
                "BC": sc.LAMBDA,  # 0.618
                "CD": 3.618  # PHI^2 + 1
            },
            "Phi-Harmonic": {  # Custom pattern based on PHI
                "XA": sc.LAMBDA,  # 0.618
                "AB": sc.LAMBDA,  # 0.618
                "BC": sc.LAMBDA,  # 0.618
                "CD": sc.PHI  # 1.618
            }
        }
        return patterns
    
    def find_patterns(self, price_history: CryptoPriceHistory) -> Dict[str, List[Dict]]:
        """Find harmonic patterns in price data"""
        # Use GPU if available, otherwise CPU
        if self.config.use_cuda and self.cuda_module and len(price_history.prices) > 1000:
            return self._find_patterns_cuda(price_history)
        else:
            return self._find_patterns_cpu(price_history)
    
    def _find_patterns_cpu(self, price_history: CryptoPriceHistory) -> Dict[str, List[Dict]]:
        """Find harmonic patterns using CPU"""
        prices = price_history.get_close_prices()
        dates = price_history.get_dates()
        results = {}
        
        for pattern_name, pattern_ratios in self.patterns.items():
            pattern_instances = []
            
            # Find patterns in price data
            for i in range(len(prices) - 5):
                # Get 5 consecutive prices for potential XABCD pattern
                p0, p1, p2, p3, p4 = prices[i:i+5]
                
                # Calculate actual ratios
                XA = (p1 - p0) / p0 if p0 != 0 else 0
                AB = (p2 - p1) / p1 if p1 != 0 else 0
                BC = (p3 - p2) / p2 if p2 != 0 else 0
                CD = (p4 - p3) / p3 if p3 != 0 else 0
                
                # Check if ratios match pattern with tolerance
                tolerance = 0.05  # 5% tolerance
                
                if (abs(XA - pattern_ratios["XA"]) < tolerance and
                    abs(AB - pattern_ratios["AB"]) < tolerance and
                    abs(BC - pattern_ratios["BC"]) < tolerance and
                    abs(CD - pattern_ratios["CD"]) < tolerance):
                    
                    # Pattern found
                    pattern_instances.append({
                        "start_idx": i,
                        "end_idx": i + 4,
                        "start_date": dates[i],
                        "end_date": dates[i + 4],
                        "prices": [p0, p1, p2, p3, p4],
                        "confidence": self._calculate_pattern_confidence(
                            XA, AB, BC, CD, pattern_ratios
                        )
                    })
            
            results[pattern_name] = pattern_instances
        
        return results
    
    def _find_patterns_cuda(self, price_history: CryptoPriceHistory) -> Dict[str, List[Dict]]:
        """Find harmonic patterns using CUDA acceleration"""
        if not CUDA_AVAILABLE or self.cuda_module is None:
            return self._find_patterns_cpu(price_history)
        
        try:
            prices = price_history.get_close_prices()
            dates = price_history.get_dates()
            results = {}
            
            # Prepare pattern data for GPU
            pattern_data = []
            pattern_names = []
            
            for pattern_name, pattern_ratios in self.patterns.items():
                pattern_names.append(pattern_name)
                pattern_data.extend([
                    pattern_ratios["XA"],
                    pattern_ratios["AB"],
                    pattern_ratios["BC"],
                    pattern_ratios["CD"]
                ])
            
            # Create device memory
            d_prices = Memory.alocate(len(prices) * np.dtype(np.float32).itemsize)
            d_patterns = Memory.alocate(len(pattern_data) * np.dtype(np.float32).itemsize)
            d_results = Memory.alocate(len(self.patterns) * np.dtype(np.int32).itemsize)
            
            # Copy data to device
            h_prices = np.array(prices, dtype=np.float32)
            h_patterns = np.array(pattern_data, dtype=np.float32)
            h_results = np.zeros(len(self.patterns), dtype=np.int32)
            
            Memory.copy_from_host(h_prices.ctypes.data, d_prices, len(prices) * np.dtype(np.float32).itemsize)
            Memory.copy_from_host(h_patterns.ctypes.data, d_patterns, len(pattern_data) * np.dtype(np.float32).itemsize)
            Memory.copy_from_host(h_results.ctypes.data, d_results, len(self.patterns) * np.dtype(np.int32).itemsize)
            
            # Launch kernel for pattern finding
            block_dim = (256, 1, 1)
            grid_dim = ((len(prices) + block_dim[0] - 1) // block_dim[0], len(self.patterns), 1)
            
            stream = Stream()
            kernel_func = self.cuda_module.get_function("find_potential_patterns")
            kernel_func.launch(grid_dim, block_dim, stream=stream, args=[
                d_prices.handle, len(prices), d_patterns.handle, 
                len(self.patterns), d_results.handle
            ])
            
            # Copy results back
            Memory.copy_to_host(d_results, h_results.ctypes.data, len(self.patterns) * np.dtype(np.int32).itemsize)
            
            # CPU-side validation of potential patterns
            for i, pattern_name in enumerate(pattern_names):
                pattern_count = h_results[i]
                pattern_ratios = self.patterns[pattern_name]
                pattern_instances = []
                
                if pattern_count > 0:
                    # Validate patterns on CPU
                    for j in range(len(prices) - 5):
                        p0, p1, p2, p3, p4 = prices[j:j+5]
                        
                        # Calculate actual ratios
                        XA = (p1 - p0) / p0 if p0 != 0 else 0
                        AB = (p2 - p1) / p1 if p1 != 0 else 0
                        BC = (p3 - p2) / p2 if p2 != 0 else 0
                        CD = (p4 - p3) / p3 if p3 != 0 else 0
                        
                        # Check if ratios match pattern
                        tolerance = 0.05
                        
                        if (abs(XA - pattern_ratios["XA"]) < tolerance and
                            abs(AB - pattern_ratios["AB"]) < tolerance and
                            abs(BC - pattern_ratios["BC"]) < tolerance and
                            abs(CD - pattern_ratios["CD"]) < tolerance):
                            
                            pattern_instances.append({
                                "start_idx": j,
                                "end_idx": j + 4,
                                "start_date": dates[j],
                                "end_date": dates[j + 4],
                                "prices": [p0, p1, p2, p3, p4],
                                "confidence": self._calculate_pattern_confidence(
                                    XA, AB, BC, CD, pattern_ratios
                                )
                            })
                
                results[pattern_name] = pattern_instances
            
            # Clean up memory
            d_prices.free()
            d_patterns.free()
            d_results.free()
            
            return results
            
        except Exception as e:
            print(f"Error in CUDA pattern detection: {e}")
            return self._find_patterns_cpu(price_history)
    
    def _calculate_pattern_confidence(self, XA, AB, BC, CD, pattern_ratios):
        """Calculate confidence level for a pattern match"""
        # Calculate how close each ratio is to the ideal
        XA_match = 1.0 - min(1.0, abs(XA - pattern_ratios["XA"]) / 0.1)
        AB_match = 1.0 - min(1.0, abs(AB - pattern_ratios["AB"]) / 0.1)
        BC_match = 1.0 - min(1.0, abs(BC - pattern_ratios["BC"]) / 0.1)
        CD_match = 1.0 - min(1.0, abs(CD - pattern_ratios["CD"]) / 0.1)
        
        # Weight the ratios by golden ratio
        weights = [sc.LAMBDA, sc.LAMBDA * sc.LAMBDA, sc.LAMBDA, sc.PHI * sc.LAMBDA]
        weight_sum = sum(weights)
        
        # Calculate weighted confidence
        confidence = (
            XA_match * weights[0] + 
            AB_match * weights[1] + 
            BC_match * weights[2] + 
            CD_match * weights[3]
        ) / weight_sum
        
        return confidence
    
    def analyze_phi_retracements(self, price_history: CryptoPriceHistory) -> Dict:
        """Analyze price data for phi-based retracement levels"""
        prices = price_history.get_close_prices()
        dates = price_history.get_dates()
        
        # Define phi-based retracement levels
        levels = PHI_FIBONACCI_LEVELS
        
        if self.config.use_cuda and self.cuda_module:
            return self._analyze_phi_retracements_cuda(prices, dates, levels)
        else:
            return self._analyze_phi_retracements_cpu(prices, dates, levels)
    
    def _analyze_phi_retracements_cpu(self, prices, dates, levels):
        """Analyze phi retracements using CPU"""
        results = {level: [] for level in levels}
        
        # Find significant price moves
        for i in range(len(prices) - 1):
            price0 = prices[i]
            price1 = prices[i + 1]
            move = price1 - price0
            
            # Only consider significant moves
            if abs(move) / price0 < 0.01:  # 1% threshold
                continue
            
            # Check each retracement level
            for level in levels:
                target = price0 + move * level
                
                # Look ahead for prices hitting this level
                for j in range(i + 2, min(i + 10, len(prices))):
                    if abs(prices[j] - target) / target < 0.01:  # 1% tolerance
                        results[level].append({
                            "start_idx": i,
                            "target_idx": j,
                            "start_date": dates[i],
                            "target_date": dates[j],
                            "start_price": price0,
                            "move_price": price1,
                            "target_price": target,
                            "actual_price": prices[j]
                        })
                        break
        
        # Calculate statistics
        stats = {
            "total_moves": sum(len(hits) for hits in results.values()),
            "level_hits": {level: len(hits) for level, hits in results.items()},
            "phi_dominance": 0.0
        }
        
        # Check if 0.618 (phi) level is dominant
        if stats["total_moves"] > 0:
            phi_hits = stats["level_hits"].get(0.618, 0)
            stats["phi_dominance"] = phi_hits / stats["total_moves"]
        
        return {"retracements": results, "statistics": stats}
    
    def _analyze_phi_retracements_cuda(self, prices, dates, levels):
        """Analyze phi retracements using CUDA"""
        if not CUDA_AVAILABLE or self.cuda_module is None:
            return self._analyze_phi_retracements_cpu(prices, dates, levels)
        
        try:
            # Create device memory
            d_prices = Memory.alocate(len(prices) * np.dtype(np.float32).itemsize)
            d_levels = Memory.alocate(len(levels) * np.dtype(np.float32).itemsize)
            d_hits = Memory.alocate(len(levels) * np.dtype(np.int32).itemsize)
            
            # Copy data to device
            h_prices = np.array(prices, dtype=np.float32)
            h_levels = np.array(levels, dtype=np.float32)
            h_hits = np.zeros(len(levels), dtype=np.int32)
            
            Memory.copy_from_host(h_prices.ctypes.data, d_prices, len(prices) * np.dtype(np.float32).itemsize)
            Memory.copy_from_host(h_levels.ctypes.data, d_levels, len(levels) * np.dtype(np.float32).itemsize)
            Memory.copy_from_host(h_hits.ctypes.data, d_hits, len(levels) * np.dtype(np.int32).itemsize)
            
            # Launch kernel
            block_dim = (256, 1, 1)
            grid_dim = ((len(prices) + block_dim[0] - 1) // block_dim[0], 1, 1)
            
            stream = Stream()
            kernel_func = self.cuda_module.get_function("calculate_phi_levels")
            kernel_func.launch(grid_dim, block_dim, stream=stream, args=[
                d_prices.handle, len(prices), d_levels.handle, 
                len(levels), d_hits.handle
            ])
            
            # Copy results back
            Memory.copy_to_host(d_hits, h_hits.ctypes.data, len(levels) * np.dtype(np.int32).itemsize)
            
            # Clean up memory
            d_prices.free()
            d_levels.free()
            d_hits.free()
            
            # CPU-side detailed analysis (only for patterns actually found)
            results = {level: [] for level in levels}
            
            for i, level in enumerate(levels):
                if h_hits[i] > 0:
                    # CPU-side detailed analysis
                    for j in range(len(prices) - 1):
                        price0 = prices[j]
                        price1 = prices[j + 1]
                        move = price1 - price0
                        
                        # Only consider significant moves
                        if abs(move) / price0 < 0.01:
                            continue
                        
                        target = price0 + move * level
                        
                        # Look ahead for prices hitting this level
                        for k in range(j + 2, min(j + 10, len(prices))):
                            if abs(prices[k] - target) / target < 0.01:
                                results[level].append({
                                    "start_idx": j,
                                    "target_idx": k,
                                    "start_date": dates[j],
                                    "target_date": dates[k],
                                    "start_price": price0,
                                    "move_price": price1,
                                    "target_price": target,
                                    "actual_price": prices[k]
                                })
                                break
            
            # Calculate statistics
            total_hits = sum(h_hits)
            level_hits = {level: h_hits[i] for i, level in enumerate(levels)}
            
            stats = {
                "total_moves": total_hits,
                "level_hits": level_hits,
                "phi_dominance": level_hits.get(sc.LAMBDA, 0) / total_hits if total_hits > 0 else 0.0
            }
            
            return {"retracements": results, "statistics": stats}
            
        except Exception as e:
            print(f"Error in CUDA retracement analysis: {e}")
            return self._analyze_phi_retracements_cpu(prices, dates, levels)

class PhiHarmonicCryptoAnalyzer:
    """
    Phi-harmonic cryptocurrency analysis tool
    
    Integrates sacred constants, quantum field principles, and pattern 
    recognition to analyze cryptocurrency market movements.
    """
    
    def __init__(self, config: CryptoConfig = None):
        self.config = config or CryptoConfig()
        self.pattern_finder = HarmonicPatternFinder(self.config)
        self.price_histories = {}
        self.analysis_results = {}
    
    def load_price_history(self, symbol: str, filepath: str = None) -> CryptoPriceHistory:
        """Load price history from file or generate sample data"""
        if filepath and os.path.exists(filepath):
            price_history = CryptoPriceHistory.load_from_file(filepath)
        else:
            # Generate sample data for testing
            price_history = CryptoPriceHistory(symbol=symbol)
            price_history.generate_sample_data()
        
        self.price_histories[symbol] = price_history
        return price_history
    
    def save_price_history(self, symbol: str, filepath: str):
        """Save price history to file"""
        if symbol not in self.price_histories:
            raise ValueError(f"Price history for {symbol} not loaded")
        
        self.price_histories[symbol].save_to_file(filepath)
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a cryptocurrency symbol"""
        if symbol not in self.price_histories:
            raise ValueError(f"Price history for {symbol} not loaded")
        
        price_history = self.price_histories[symbol]
        
        # Perform multiple analyses
        harmonic_patterns = self.pattern_finder.find_patterns(price_history)
        phi_retracements = self.pattern_finder.analyze_phi_retracements(price_history)
        moving_averages = self._analyze_moving_averages(price_history)
        cycle_analysis = self._analyze_price_cycles(price_history)
        mayer_multiple = self._calculate_mayer_multiple(price_history)
        
        # Combine results
        results = {
            "symbol": symbol,
            "timestamp": int(time.time()),
            "data_points": len(price_history.prices),
            "date_range": [
                price_history.get_dates()[0].strftime("%Y-%m-%d"),
                price_history.get_dates()[-1].strftime("%Y-%m-%d")
            ],
            "harmonic_patterns": harmonic_patterns,
            "phi_retracements": phi_retracements,
            "moving_averages": moving_averages,
            "cycle_analysis": cycle_analysis,
            "mayer_multiple": mayer_multiple,
            
            # Summary metrics
            "summary": self._generate_analysis_summary(
                harmonic_patterns, phi_retracements, 
                moving_averages, cycle_analysis, mayer_multiple
            )
        }
        
        self.analysis_results[symbol] = results
        return results
    
    def _analyze_moving_averages(self, price_history: CryptoPriceHistory) -> Dict:
        """Analyze moving averages with phi-harmonic periods"""
        prices = price_history.get_close_prices()
        
        # Define phi-harmonic periods
        phi_periods = [
            int(sc.PHI * 10),     # ~16 days
            int(sc.PHI * 20),     # ~32 days
            int(sc.PHI * 50),     # ~80 days
            int(sc.PHI * 100),    # ~162 days
            int(sc.PHI * 200)     # ~324 days
        ]
        
        mas = {}
        crossovers = []
        
        # Calculate MAs for each period
        for period in phi_periods:
            if period < len(prices):
                # Simple moving average
                ma = np.convolve(prices, np.ones(period)/period, mode='valid')
                mas[period] = ma.tolist()
        
        # Find crossovers between adjacent MA pairs
        sorted_periods = sorted(phi_periods)
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            if short_period in mas and long_period in mas:
                short_ma = np.array(mas[short_period])
                long_ma = np.array(mas[long_period][-len(short_ma):])
                
                # Find crossovers
                for j in range(1, len(short_ma)):
                    if short_ma[j-1] < long_ma[j-1] and short_ma[j] > long_ma[j]:
                        # Bullish crossover
                        crossovers.append({
                            "type": "bullish",
                            "index": j + (len(prices) - len(short_ma)),
                            "date": price_history.get_dates()[j + (len(prices) - len(short_ma))].strftime("%Y-%m-%d"),
                            "short_period": short_period,
                            "long_period": long_period
                        })
                    elif short_ma[j-1] > long_ma[j-1] and short_ma[j] < long_ma[j]:
                        # Bearish crossover
                        crossovers.append({
                            "type": "bearish",
                            "index": j + (len(prices) - len(short_ma)),
                            "date": price_history.get_dates()[j + (len(prices) - len(short_ma))].strftime("%Y-%m-%d"),
                            "short_period": short_period,
                            "long_period": long_period
                        })
        
        # Current MA relationship (bullish/bearish bias)
        current_bias = "neutral"
        if crossovers:
            current_bias = crossovers[-1]["type"]
        
        return {
            "periods": phi_periods,
            "mas": mas,
            "crossovers": crossovers,
            "current_bias": current_bias
        }
    
    def _analyze_price_cycles(self, price_history: CryptoPriceHistory) -> Dict:
        """Analyze price cycles based on phi-harmonic periods"""
        prices = price_history.get_close_prices()
        dates = price_history.get_dates()
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Use FFT to find dominant cycles
        if len(returns) >= 64:
            fft = np.fft.fft(returns)
            frequencies = np.fft.fftfreq(len(returns))
            
            # Take positive frequencies only
            pos_mask = frequencies > 0
            pos_frequencies = frequencies[pos_mask]
            pos_amplitudes = np.abs(fft)[pos_mask]
            
            # Find dominant frequencies
            sorted_indices = np.argsort(pos_amplitudes)[::-1][:5]  # Top 5
            dominant_cycles = []
            
            for idx in sorted_indices:
                period = 1 / pos_frequencies[idx]
                if period < len(returns) / 2:  # Only include reasonable periods
                    dominant_cycles.append({
                        "period": round(period),
                        "amplitude": float(pos_amplitudes[idx] / sum(pos_amplitudes)),
                        "phi_resonance": self._calculate_phi_resonance(period)
                    })
        else:
            dominant_cycles = []
        
        # Find phi-based cycle candidates
        phi_cycle = self.config.price_cycle_length
        cycle_candidates = []
        
        for multiplier in [0.5, 1.0, sc.PHI, 2.0, sc.PHI_PHI]:
            candidate_period = int(phi_cycle * multiplier)
            if candidate_period <= len(prices) / 2:
                # Check if this cycle has explanatory power
                score = self._test_cycle_explanation(prices, candidate_period)
                cycle_candidates.append({
                    "period": candidate_period,
                    "phi_factor": multiplier,
                    "score": score
                })
        
        # Sort by score
        cycle_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Determine current cycle position
        best_cycle = cycle_candidates[0] if cycle_candidates else {"period": phi_cycle}
        current_position = len(prices) % best_cycle["period"]
        cycle_phase = current_position / best_cycle["period"]
        
        # Predict next turning point
        days_to_next_peak = (best_cycle["period"] - current_position) % best_cycle["period"]
        next_peak_date = (dates[-1] + timedelta(days=days_to_next_peak)).strftime("%Y-%m-%d")
        
        return {
            "dominant_cycles": dominant_cycles,
            "phi_cycles": cycle_candidates,
            "best_cycle": best_cycle,
            "current_position": current_position,
            "cycle_phase": cycle_phase,
            "next_peak_date": next_peak_date
        }
    
    def _calculate_phi_resonance(self, period: float) -> float:
        """Calculate how closely a period resonates with phi-harmonics"""
        phi_periods = [
            sc.PHI * 10,
            sc.PHI * sc.PHI * 10,
            sc.PHI * 50,
            sc.PHI * sc.PHI * 50,
            sc.PHI * 100,
            sc.PHI * sc.PHI * 100
        ]
        
        # Find closest phi period
        closest_diff = float('inf')
        for phi_period in phi_periods:
            diff = abs(period - phi_period) / phi_period
            if diff < closest_diff:
                closest_diff = diff
        
        # Convert to resonance score (0 to 1)
        resonance = max(0, 1 - closest_diff * 5)
        return resonance
    
    def _test_cycle_explanation(self, prices: np.ndarray, period: int) -> float:
        """Test how well a cycle period explains price movements"""
        # Short-circuit for too long periods
        if period > len(prices) / 2:
            return 0.0
            
        # Create a sine wave model for this cycle
        x = np.arange(len(prices))
        cycle_model = np.sin(2 * np.pi * x / period)
        
        # Normalize prices
        norm_prices = (prices - np.mean(prices)) / np.std(prices)
        
        # Calculate correlation
        correlation = np.corrcoef(cycle_model, norm_prices)[0, 1]
        
        # Return absolute correlation as score
        return abs(correlation)
    
    def _calculate_mayer_multiple(self, price_history: CryptoPriceHistory) -> Dict:
        """Calculate Mayer Multiple (current price / 200-day MA)"""
        prices = price_history.get_close_prices()
        
        # Calculate 200-day MA if we have enough data
        ma_period = 200
        if len(prices) >= ma_period:
            ma_200 = np.convolve(prices, np.ones(ma_period)/ma_period, mode='valid')[-1]
            mayer = prices[-1] / ma_200
        else:
            ma_200 = None
            mayer = None
        
        # Calculate phi-based MA for comparison
        phi_period = int(sc.PHI * 100)  # ~162 days
        if len(prices) >= phi_period:
            phi_ma = np.convolve(prices, np.ones(phi_period)/phi_period, mode='valid')[-1]
            phi_mayer = prices[-1] / phi_ma
        else:
            phi_ma = None
            phi_mayer = None
        
        # Interpret the multiple
        interpretation = "neutral"
        if mayer is not None:
            if mayer > self.config.mayer_multiple_threshold:
                interpretation = "overvalued"
            elif mayer < 1 / self.config.mayer_multiple_threshold:
                interpretation = "undervalued"
        
        return {
            "mayer_multiple": mayer,
            "phi_mayer_multiple": phi_mayer,
            "ma_200": ma_200,
            "phi_ma": phi_ma,
            "interpretation": interpretation
        }
    
    def _generate_analysis_summary(self, patterns, retracements, mas, cycles, mayer) -> Dict:
        """Generate a summary of all analysis components"""
        # Count active patterns
        pattern_count = sum(len(instances) for instances in patterns.values())
        
        # Phi-retracement dominance
        phi_dominance = retracements["statistics"].get("phi_dominance", 0)
        
        # Current market bias from MAs
        market_bias = mas.get("current_bias", "neutral")
        
        # Current cycle phase
        cycle_phase = cycles.get("cycle_phase", 0)
        
        # Mayer multiple interpretation
        valuation = mayer.get("interpretation", "neutral")
        
        # Calculate overall phi-harmonic confidence
        pattern_confidence = 0.2 if pattern_count > 0 else 0
        phi_confidence = phi_dominance * 0.3
        cycle_confidence = 0.0
        
        if cycles.get("phi_cycles"):
            best_cycle = cycles["phi_cycles"][0]
            cycle_confidence = best_cycle.get("score", 0) * 0.3
        
        ma_confidence = 0.2 if market_bias == "bullish" else 0.0
        
        overall_confidence = pattern_confidence + phi_confidence + cycle_confidence + ma_confidence
        
        # Market direction prediction
        if overall_confidence > 0.6:
            if market_bias == "bullish" and cycle_phase < 0.5:
                prediction = "strongly bullish"
            elif market_bias == "bearish" and cycle_phase > 0.5:
                prediction = "strongly bearish"
            else:
                prediction = "mixed signals"
        else:
            prediction = "insufficient data"
        
        return {
            "pattern_count": pattern_count,
            "phi_dominance": phi_dominance,
            "market_bias": market_bias,
            "cycle_phase": cycle_phase,
            "valuation": valuation,
            "phi_harmonic_confidence": overall_confidence,
            "prediction": prediction
        }
    
    def visualize_patterns(self, symbol: str, pattern_type: str = None, save_path: str = None):
        """Visualize detected harmonic patterns"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Analysis for {symbol} not found. Run analyze_symbol first.")
        
        price_history = self.price_histories[symbol]
        analysis = self.analysis_results[symbol]
        
        # Get pattern data
        patterns = analysis["harmonic_patterns"]
        
        if pattern_type:
            # Filter to specific pattern type
            if pattern_type not in patterns:
                raise ValueError(f"Pattern type {pattern_type} not found")
            pattern_data = {pattern_type: patterns[pattern_type]}
        else:
            pattern_data = patterns
        
        # Check if we have any patterns to visualize
        pattern_count = sum(len(instances) for instances in pattern_data.values())
        if pattern_count == 0:
            print(f"No patterns found for {symbol}")
            return
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Get price data
        dates = price_history.get_dates()
        prices = price_history.get_close_prices()
        
        # Plot price data
        plt.plot(dates, prices, label=symbol, color='black', alpha=0.5)
        
        # Color map for different patterns
        colors = {
            "Gartley": "blue",
            "Butterfly": "green",
            "Bat": "purple",
            "Crab": "red",
            "Phi-Harmonic": "orange"
        }
        
        # Plot each pattern
        for pattern_name, instances in pattern_data.items():
            color = colors.get(pattern_name, "gray")
            
            for instance in instances:
                # Get pattern points
                start_idx = instance["start_idx"]
                end_idx = instance["end_idx"]
                pattern_dates = dates[start_idx:end_idx+1]
                pattern_prices = prices[start_idx:end_idx+1]
                
                # Plot pattern
                plt.plot(pattern_dates, pattern_prices, 'o-', 
                         color=color, alpha=0.7, linewidth=2,
                         label=f"{pattern_name} ({instance['confidence']:.2f})")
                
                # Mark XABCD points
                plt.scatter(pattern_dates, pattern_prices, color=color, s=100, zorder=5)
                
                # Add labels
                for i, label in enumerate(['X', 'A', 'B', 'C', 'D']):
                    plt.annotate(label, 
                                (pattern_dates[i], pattern_prices[i]),
                                textcoords="offset points",
                                xytext=(0,10), 
                                ha='center')
        
        # Add phi ratio grid lines
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        phi_levels = []
        for level in [0, 0.382, 0.5, 0.618, 0.786, 1, 1.618]:
            level_price = min_price + price_range * level
            phi_levels.append(level_price)
            plt.axhline(y=level_price, linestyle='--', alpha=0.3, 
                      color='gray', label=f"Phi {level}" if level in [0, 0.618, 1, 1.618] else "")
        
        # Add MA lines if available
        try:
            ma_data = analysis["moving_averages"]["mas"]
            for period, ma_vals in ma_data.items():
                if len(ma_vals) > 0:
                    # Calculate offset for the MA (it's shorter than the price series)
                    offset = len(prices) - len(ma_vals)
                    ma_dates = dates[offset:]
                    plt.plot(ma_dates, ma_vals, label=f"{period}-MA", alpha=0.7)
        except (KeyError, IndexError):
            pass
        
        # Set up the title and labels
        plt.title(f"Harmonic Patterns for {symbol}", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        
        # Add legend with unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add phi watermark
        plt.figtext(0.5, 0.03, f"Phi (φ): {sc.PHI}", ha='center', alpha=0.7)
        
        # Add sacred geometry background
        plt.gca().set_facecolor('#f9f9f9')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def visualize_phi_cycles(self, symbol: str, save_path: str = None):
        """Visualize phi-harmonic price cycles"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Analysis for {symbol} not found. Run analyze_symbol first.")
        
        price_history = self.price_histories[symbol]
        analysis = self.analysis_results[symbol]
        
        # Get cycle data
        cycle_data = analysis["cycle_analysis"]
        if not cycle_data.get("phi_cycles"):
            print(f"No phi cycles found for {symbol}")
            return
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Get price data
        dates = price_history.get_dates()
        prices = price_history.get_close_prices()
        
        # Plot price data
        plt.subplot(2, 1, 1)
        plt.plot(dates, prices, label=symbol, color='black')
        
        # Add best cycle overlay
        if cycle_data.get("best_cycle"):
            best_cycle = cycle_data["best_cycle"]
            period = best_cycle["period"]
            
            # Generate idealized cycle
            x = np.arange(len(prices))
            cycle = np.sin(2 * np.pi * x / period)
            
            # Scale to price range
            price_range = max(prices) - min(prices)
            cycle = cycle * price_range * 0.2 + np.mean(prices)
            
            plt.plot(dates, cycle, '--', color='blue', 
                   alpha=0.7, linewidth=2,
                   label=f"Phi Cycle ({period} days)")
            
            # Mark next projected peak
            try:
                next_peak_date = datetime.strptime(cycle_data["next_peak_date"], "%Y-%m-%d")
                plt.axvline(x=next_peak_date, color='green', linestyle='--', 
                          alpha=0.7, label=f"Next Peak: {cycle_data['next_peak_date']}")
            except (ValueError, KeyError):
                pass
        
        # Add phi ratio grid lines
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        for level in [0, 0.382, 0.5, 0.618, 0.786, 1, 1.618]:
            level_price = min_price + price_range * level
            plt.axhline(y=level_price, linestyle='--', alpha=0.3, 
                      color='gray', label=f"Phi {level}" if level in [0, 0.618, 1, 1.618] else "")
        
        # Set up the title and labels
        plt.title(f"Phi-Harmonic Cycles for {symbol}", fontsize=16)
        plt.ylabel("Price", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot cycle breakdown
        plt.subplot(2, 1, 2)
        
        # Bar chart of phi cycles
        cycles = cycle_data.get("phi_cycles", [])
        periods = [c["period"] for c in cycles]
        scores = [c["score"] for c in cycles]
        phi_factors = [c.get("phi_factor", 0) for c in cycles]
        
        bars = plt.bar(range(len(periods)), scores, alpha=0.7)
        
        # Color bars by phi factor
        for i, bar in enumerate(bars):
            if phi_factors[i] == sc.PHI:
                bar.set_color('gold')
            elif phi_factors[i] == sc.PHI_PHI:
                bar.set_color('purple')
            else:
                bar.set_color('blue')
        
        plt.xticks(range(len(periods)), [f"{p} days" for p in periods])
        plt.ylabel("Cycle Strength", fontsize=12)
        plt.xlabel("Cycle Period", fontsize=12)
        plt.title("Phi-Harmonic Cycle Strength", fontsize=14)
        
        # Add phi cycle annotations
        for i, (period, score, factor) in enumerate(zip(periods, scores, phi_factors)):
            if factor in [sc.PHI, sc.PHI_PHI]:
                plt.annotate(f"φ×{factor/sc.PHI:.1f}",
                            (i, score),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            color='darkred')
        
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Add phi watermark
        plt.figtext(0.5, 0.01, f"Phi (φ): {sc.PHI} - Phi^Phi: {sc.PHI_PHI}", ha='center', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, symbol: str, filepath: str = None):
        """Generate a comprehensive analysis report"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Analysis for {symbol} not found. Run analyze_symbol first.")
        
        analysis = self.analysis_results[symbol]
        
        # Build report text
        report = f"""
        PHI-HARMONIC CRYPTOCURRENCY ANALYSIS REPORT
        ==========================================
        
        Symbol: {symbol}
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Data Range: {analysis["date_range"][0]} to {analysis["date_range"][1]}
        
        SUMMARY
        -------
        Prediction: {analysis["summary"]["prediction"]}
        Phi-Harmonic Confidence: {analysis["summary"]["phi_harmonic_confidence"]:.2f}
        Current Market Bias: {analysis["summary"]["market_bias"]}
        Cycle Phase: {analysis["summary"]["cycle_phase"]:.2f}
        Valuation: {analysis["summary"]["valuation"]}
        
        HARMONIC PATTERNS
        ----------------
        """
        
        # Add pattern information
        patterns = analysis["harmonic_patterns"]
        for pattern_name, instances in patterns.items():
            if instances:
                report += f"\n{pattern_name}: {len(instances)} instances\n"
                for i, instance in enumerate(instances):
                    report += f"  - Instance {i+1}: {instance['start_date']} to {instance['end_date']}, Confidence: {instance['confidence']:.2f}\n"
        
        if not any(instances for instances in patterns.values()):
            report += "\nNo harmonic patterns detected.\n"
        
        # Add phi retracement information
        report += """
        PHI RETRACEMENTS
        ---------------
        """
        
        retracements = analysis["phi_retracements"]
        report += f"Phi (0.618) Level Dominance: {retracements['statistics']['phi_dominance']:.2f}\n"
        report += f"Total Retracement Hits: {retracements['statistics']['total_moves']}\n\n"
        
        for level, hits in retracements["statistics"]["level_hits"].items():
            report += f"  - Level {level}: {hits} hits\n"
        
        # Add moving average information
        report += """
        MOVING AVERAGES
        --------------
        """
        
        mas = analysis["moving_averages"]
        report += f"Current Bias: {mas['current_bias']}\n\n"
        
        if mas.get("crossovers"):
            report += "Recent Crossovers:\n"
            for crossover in mas["crossovers"][-3:]:  # Show last 3 crossovers
                report += f"  - {crossover['date']}: {crossover['type'].capitalize()} crossover of {crossover['short_period']} and {crossover['long_period']} day MAs\n"
        
        # Add cycle analysis
        report += """
        CYCLE ANALYSIS
        -------------
        """
        
        cycles = analysis["cycle_analysis"]
        if cycles.get("phi_cycles"):
            best_cycle = cycles["phi_cycles"][0]
            report += f"Best Phi-Harmonic Cycle: {best_cycle['period']} days (Phi Factor: {best_cycle['phi_factor']:.2f})\n"
            report += f"Current Cycle Position: {cycles['current_position']} days into cycle ({cycles['cycle_phase']:.2f} phase)\n"
            report += f"Projected Next Peak: {cycles['next_peak_date']}\n\n"
            
            if cycles.get("dominant_cycles"):
                report += "Dominant Statistical Cycles:\n"
                for cycle in cycles["dominant_cycles"]:
                    report += f"  - {cycle['period']} days (Amplitude: {cycle['amplitude']:.2f}, Phi Resonance: {cycle['phi_resonance']:.2f})\n"
        else:
            report += "Insufficient data for cycle analysis.\n"
        
        # Add Mayer multiple information
        report += """
        VALUATION METRICS
        ---------------
        """
        
        mayer = analysis["mayer_multiple"]
        if mayer.get("mayer_multiple") is not None:
            report += f"Mayer Multiple: {mayer['mayer_multiple']:.2f} (Price / 200-day MA)\n"
            report += f"Phi-Mayer Multiple: {mayer['phi_mayer_multiple']:.2f} (Price / {int(sc.PHI * 100)}-day MA)\n"
            report += f"Interpretation: {mayer['interpretation']}\n"
        else:
            report += "Insufficient data for Mayer multiple calculation.\n"
        
        # Add phi constants references
        report += f"""
        PHI CONSTANTS REFERENCE
        ---------------------
        Phi (φ): {sc.PHI}
        Lambda (λ): {sc.LAMBDA}
        Phi^Phi: {sc.PHI_PHI}
        
        Generated with Phi-Harmonic Cryptocurrency Analyzer
        """
        
        # Save to file if requested
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"Report saved to {filepath}")
        
        return report

def main():
    """Main function to demonstrate the crypto harmonic analyzer"""
    # Create analyzer with default configuration
    config = CryptoConfig(use_cuda=CUDA_AVAILABLE)
    analyzer = PhiHarmonicCryptoAnalyzer(config)
    
    print("\nCRYPTO HARMONIC ANALYZER")
    print("=======================")
    print(f"PHI: {sc.PHI}")
    print(f"LAMBDA: {sc.LAMBDA}")
    print(f"PHI^PHI: {sc.PHI_PHI}")
    print(f"CUDA Enabled: {config.use_cuda}")
    print()
    
    # Menu system
    while True:
        print("\nCrypto Harmonic Analyzer Operations:")
        print("1. Load/Generate Price History")
        print("2. Analyze Symbol")
        print("3. Visualize Harmonic Patterns")
        print("4. Visualize Phi Cycles")
        print("5. Generate Analysis Report")
        print("6. Save Price History")
        print("7. Exit")
        
        choice = input("\nSelect an operation (1-7): ")
        
        if choice == '1':
            symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").upper()
            
            use_sample = input("Use sample data? (y/n): ").lower() == 'y'
            
            if not use_sample:
                filepath = input("Enter filepath to load price history (or leave blank to generate): ")
                if not filepath:
                    use_sample = True
            
            if use_sample:
                days = int(input("Enter number of days to generate (default 365): ") or "365")
                start_price = float(input("Enter starting price (default 30000): ") or "30000")
                volatility = float(input("Enter volatility factor 0.01-0.1 (default 0.03): ") or "0.03")
                
                price_history = CryptoPriceHistory(symbol=symbol)
                price_history.generate_sample_data(days, start_price, volatility)
                analyzer.price_histories[symbol] = price_history
                print(f"\nGenerated sample data for {symbol} with {days} days of history")
            else:
                try:
                    price_history = analyzer.load_price_history(symbol, filepath)
                    print(f"\nLoaded price history for {symbol} with {len(price_history.prices)} data points")
                except Exception as e:
                    print(f"Error loading price history: {e}")
        
        elif choice == '2':
            if not analyzer.price_histories:
                print("No price histories loaded. Load a price history first.")
                continue
                
            print("\nAvailable symbols:")
            for symbol in analyzer.price_histories.keys():
                print(f"- {symbol}")
                
            symbol = input("\nEnter symbol to analyze: ").upper()
            
            if symbol not in analyzer.price_histories:
                print(f"Symbol {symbol} not loaded")
                continue
                
            try:
                results = analyzer.analyze_symbol(symbol)
                
                print(f"\nAnalysis complete for {symbol}")
                print(f"Harmonic patterns found: {results['summary']['pattern_count']}")
                print(f"Phi dominance: {results['summary']['phi_dominance']:.2f}")
                print(f"Market bias: {results['summary']['market_bias']}")
                print(f"Prediction: {results['summary']['prediction']}")
                
            except Exception as e:
                print(f"Error analyzing symbol: {e}")
        
        elif choice == '3':
            if not analyzer.analysis_results:
                print("No analysis results available. Analyze a symbol first.")
                continue
                
            print("\nAvailable symbols:")
            for symbol in analyzer.analysis_results.keys():
                print(f"- {symbol}")
                
            symbol = input("\nEnter symbol to visualize: ").upper()
            
            if symbol not in analyzer.analysis_results:
                print(f"Analysis for {symbol} not found")
                continue
                
            # List available patterns
            patterns = analyzer.analysis_results[symbol]["harmonic_patterns"]
            pattern_counts = {name: len(instances) for name, instances in patterns.items() if instances}
            
            if not pattern_counts:
                print(f"No harmonic patterns found for {symbol}")
                continue
                
            print("\nAvailable patterns:")
            for name, count in pattern_counts.items():
                print(f"- {name}: {count} instances")
                
            pattern_type = input("\nEnter pattern type to visualize (or leave blank for all): ")
            
            if pattern_type and pattern_type not in patterns:
                print(f"Pattern type {pattern_type} not found")
                continue
                
            save_path = input("Enter filepath to save visualization (or leave blank): ")
            
            try:
                analyzer.visualize_patterns(symbol, pattern_type or None, save_path or None)
            except Exception as e:
                print(f"Error visualizing patterns: {e}")
        
        elif choice == '4':
            if not analyzer.analysis_results:
                print("No analysis results available. Analyze a symbol first.")
                continue
                
            print("\nAvailable symbols:")
            for symbol in analyzer.analysis_results.keys():
                print(f"- {symbol}")
                
            symbol = input("\nEnter symbol to visualize: ").upper()
            
            if symbol not in analyzer.analysis_results:
                print(f"Analysis for {symbol} not found")
                continue
                
            save_path = input("Enter filepath to save visualization (or leave blank): ")
            
            try:
                analyzer.visualize_phi_cycles(symbol, save_path or None)
            except Exception as e:
                print(f"Error visualizing cycles: {e}")
        
        elif choice == '5':
            if not analyzer.analysis_results:
                print("No analysis results available. Analyze a symbol first.")
                continue
                
            print("\nAvailable symbols:")
            for symbol in analyzer.analysis_results.keys():
                print(f"- {symbol}")
                
            symbol = input("\nEnter symbol to generate report for: ").upper()
            
            if symbol not in analyzer.analysis_results:
                print(f"Analysis for {symbol} not found")
                continue
                
            filepath = input("Enter filepath to save report (or leave blank to display): ")
            
            try:
                report = analyzer.generate_report(symbol, filepath or None)
                
                if not filepath:
                    print("\n" + report)
            except Exception as e:
                print(f"Error generating report: {e}")
        
        elif choice == '6':
            if not analyzer.price_histories:
                print("No price histories loaded. Load a price history first.")
                continue
                
            print("\nAvailable symbols:")
            for symbol in analyzer.price_histories.keys():
                print(f"- {symbol}")
                
            symbol = input("\nEnter symbol to save: ").upper()
            
            if symbol not in analyzer.price_histories:
                print(f"Symbol {symbol} not loaded")
                continue
                
            filepath = input("Enter filepath to save price history: ")
            
            try:
                analyzer.save_price_history(symbol, filepath)
                print(f"Price history for {symbol} saved to {filepath}")
            except Exception as e:
                print(f"Error saving price history: {e}")
        
        elif choice == '7':
            print("\nExiting Crypto Harmonic Analyzer.")
            print(f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}")
            break
            
        else:
            print("Invalid choice. Please select a number between 1 and 7.")

if __name__ == "__main__":
    main()