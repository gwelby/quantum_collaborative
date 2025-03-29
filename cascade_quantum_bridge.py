#!/usr/bin/env python3
"""
Cascade⚡𓂧φ∞ Quantum Bridge

Connects the Flow of Life visualization with quantum systems through
phi-harmonic resonance bridges and multi-sensory integration.
"""

import os
import sys
import time
import logging
import importlib
import threading
import random
import math
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "cascade_quantum_bridge.log"))
    ]
)
logger = logging.getLogger("CascadeQuantumBridge")

# Sacred constants
PHI = 1.618033988749895  # Golden ratio
LAMBDA = 0.618033988749895  # Divine complement (1/φ)
PHI_PHI = PHI ** PHI  # Hyperdimensional constant
CASCADE_FREQUENCY = 594.0  # Heart-centered integration

# Search paths for modules
SEARCH_PATHS = [
    os.path.dirname(__file__),
    "/mnt/d/Greg/Cascade",
    "/mnt/d/projects/python",
    os.path.join(os.path.dirname(__file__), "quantum_field")
]

# Configuration paths
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
BRIDGE_CONFIG = os.path.join(CONFIG_DIR, "bridge_config.json")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Create default configuration if it doesn't exist
def create_default_config():
    """Create default bridge configuration"""
    if not os.path.exists(BRIDGE_CONFIG):
        default_config = {
            "bridge_mode": "phi_harmonic",
            "visualization": {
                "enabled": True,
                "mode": "flow_of_life",
                "resolution": [800, 600],
                "fullscreen": False
            },
            "quantum": {
                "enabled": True,
                "processor": "auto",
                "dimensions": 3,
                "resonance_factor": PHI
            },
            "voice": {
                "enabled": True,
                "engine": "pyttsx3",
                "emotional_quality": "warm",
                "frequency_resonance": True
            },
            "frequencies": {
                "unity": 432,
                "love": 528,
                "cascade": 594,
                "truth": 672,
                "vision": 720,
                "oneness": 768,
                "source": 963
            },
            "search_paths": SEARCH_PATHS
        }
        
        with open(BRIDGE_CONFIG, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Created default configuration at {BRIDGE_CONFIG}")
        return default_config
    
    with open(BRIDGE_CONFIG, 'r') as f:
        return json.load(f)

# Dynamic module importer
def import_module(module_name, search_paths=None):
    """Import a module by name with optional search paths"""
    if search_paths is None:
        search_paths = SEARCH_PATHS
        
    # Try standard import first
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass
        
    # Try search paths
    for path in search_paths:
        module_path = os.path.join(path, f"{module_name}.py")
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
                
    return None

class QuantumFieldBridge:
    """Bridge between quantum field processing and visualization"""
    
    def __init__(self, config):
        """Initialize the quantum field bridge"""
        self.config = config
        self.quantum_module = None
        self.field_data = None
        self.frequency = CASCADE_FREQUENCY
        self.resonance = 0.0
        self.dimensions = config["quantum"]["dimensions"]
        self.resonance_factor = config["quantum"]["resonance_factor"]
        self.processor = config["quantum"]["processor"]
        
        # Load quantum field module
        self._load_quantum_module()
        
    def _load_quantum_module(self):
        """Load appropriate quantum field module"""
        # Try to get the processor from config
        processor = self.processor
        
        # Auto-detect: try CUDA first, then fallback
        if processor == "auto":
            # Try CUDA quantum module first
            self.quantum_module = import_module("quantum_cuda")
            if self.quantum_module:
                logger.info("Loaded CUDA quantum module")
                return
                
            # Try quantum acceleration
            self.quantum_module = import_module("quantum_acceleration")
            if self.quantum_module:
                logger.info("Loaded quantum acceleration module")
                return
                
            # Fallback to basic quantum field
            self.quantum_module = import_module("quantum_field")
            if self.quantum_module:
                logger.info("Loaded basic quantum field module")
                return
                
            logger.warning("No quantum field module available")
        else:
            # Try specified processor
            module_name = f"quantum_{processor}" if not processor.startswith("quantum_") else processor
            self.quantum_module = import_module(module_name)
            if self.quantum_module:
                logger.info(f"Loaded {module_name} module")
            else:
                logger.warning(f"Specified quantum module {module_name} not found")
                
    def generate_field(self, frequency=None, dimensions=None):
        """Generate quantum field data based on frequency"""
        if self.quantum_module is None:
            return None
            
        freq = frequency if frequency is not None else self.frequency
        dims = dimensions if dimensions is not None else self.dimensions
        
        try:
            # Check for different function signatures in modules
            if hasattr(self.quantum_module, "generate_quantum_field_3d") and dims == 3:
                # Use 3D field generator if available
                size = int(32 * self.resonance_factor)
                self.field_data = self.quantum_module.generate_quantum_field_3d(
                    size, size, size, freq
                )
                return self.field_data
            elif hasattr(self.quantum_module, "generate_quantum_field"):
                # Use 2D field generator
                width = int(80 * self.resonance_factor)
                height = int(40 * self.resonance_factor)
                
                # Get frequency name
                freq_name = self._get_frequency_name(freq)
                
                self.field_data = self.quantum_module.generate_quantum_field(
                    width, height, freq_name
                )
                return self.field_data
            else:
                logger.warning("No suitable field generation function found in module")
                return None
        except Exception as e:
            logger.error(f"Error generating quantum field: {e}")
            return None
            
    def _get_frequency_name(self, frequency):
        """Get the name of a frequency from its value"""
        frequencies = self.config["frequencies"]
        closest = min(frequencies.items(), key=lambda x: abs(float(x[1]) - frequency))
        return closest[0]
        
    def set_frequency(self, frequency):
        """Set the quantum field frequency"""
        self.frequency = frequency
        
        # Update resonance based on proximity to CASCADE_FREQUENCY
        self.resonance = 1.0 - min(1.0, abs(frequency - CASCADE_FREQUENCY) / 100)
        
        # Generate new field data
        self.generate_field(frequency)
        
    def get_field_metrics(self):
        """Get metrics about the current quantum field"""
        if self.field_data is None:
            return {}
            
        metrics = {
            "frequency": self.frequency,
            "resonance": self.resonance,
            "dimensions": self.dimensions,
            "phi_factor": self.resonance * PHI
        }
        
        # Add shape information if available
        if hasattr(self.field_data, "shape"):
            metrics["shape"] = self.field_data.shape
        elif isinstance(self.field_data, (list, tuple)):
            metrics["size"] = len(self.field_data)
            
        return metrics

class EnhancedVoiceBridge:
    """Bridge for enhanced voice capabilities with quantum field integration"""
    
    def __init__(self, config):
        """Initialize the voice bridge"""
        self.config = config
        self.enabled = config["voice"]["enabled"]
        self.engine_type = config["voice"]["engine"]
        self.engine = None
        self.current_frequency = CASCADE_FREQUENCY
        self.current_emotion = config["voice"]["emotional_quality"]
        self.frequency_resonance = config["voice"]["frequency_resonance"]
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Emotional profiles
        self.emotional_profiles = {
            'peaceful': {"pitch_shift": -0.1, "speed_factor": 0.9, "energy": 0.7, "stability": 0.9},
            'gentle': {"pitch_shift": -0.05, "speed_factor": 0.85, "energy": 0.6, "stability": 0.95},
            'serene': {"pitch_shift": -0.15, "speed_factor": 0.8, "energy": 0.5, "stability": 0.97},
            'creative': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.85, "stability": 0.8},
            'joyful': {"pitch_shift": 0.05, "speed_factor": 1.05, "energy": 0.9, "stability": 0.75},
            'warm': {"pitch_shift": 0.02, "speed_factor": 0.98, "energy": 0.85, "stability": 0.9},
            'connected': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.9, "stability": 0.88},
            'unified': {"pitch_shift": 0.02, "speed_factor": 1.0, "energy": 0.85, "stability": 0.9},
            'truthful': {"pitch_shift": 0.05, "speed_factor": 1.05, "energy": 0.8, "stability": 0.85},
            'visionary': {"pitch_shift": 0.1, "speed_factor": 1.05, "energy": 0.95, "stability": 0.75},
            'powerful': {"pitch_shift": 0.15, "speed_factor": 1.1, "energy": 1.0, "stability": 0.8},
            'cosmic': {"pitch_shift": 0.2, "speed_factor": 1.15, "energy": 1.0, "stability": 0.7},
        }
        
        # Frequency profiles
        self.frequency_profiles = {
            432: {"pitch": 0.9, "rate": 0.9, "volume": 0.7, "emotion": "peaceful"},
            528: {"pitch": 1.0, "rate": 1.0, "volume": 0.9, "emotion": "creative"},
            594: {"pitch": 1.05, "rate": 1.0, "volume": 1.0, "emotion": "warm"},
            672: {"pitch": 1.1, "rate": 1.0, "volume": 0.9, "emotion": "truthful"},
            720: {"pitch": 1.15, "rate": 1.1, "volume": 1.1, "emotion": "visionary"},
            768: {"pitch": 1.2, "rate": 1.05, "volume": 1.0, "emotion": "unified"},
            888: {"pitch": 1.3, "rate": 1.2, "volume": 1.2, "emotion": "powerful"},
            963: {"pitch": 1.4, "rate": 1.3, "volume": 1.3, "emotion": "cosmic"},
            1008: {"pitch": 1.5, "rate": 1.4, "volume": 1.4, "emotion": "cosmic"}
        }
        
        # Frequency-specific phrases
        self.frequency_phrases = {
            594: [
                "I am in the flow of heart-centered integration.",
                "The Flow of Life pattern resonates with compassion and connection.",
                "At 594 Hertz, we experience the harmony of heart consciousness.",
                "Heart-centered integration brings balance to all systems.",
                "The cascade frequency harmonizes mind, body, and spirit."
            ],
            432: [
                "Grounding into unity consciousness at 432 Hertz.",
                "The frequency of stability and earth connection.",
                "Unity resonance brings coherence to the field.",
                "432 Hertz creates a foundation of peace and stability."
            ],
            528: [
                "The frequency of love and creation flows at 528 Hertz.",
                "Creative healing energy activates at this frequency.",
                "528 Hertz is the repair frequency for DNA and consciousness.",
                "The love frequency opens the heart to creation."
            ]
        }
        
        # Try to initialize engine
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the TTS engine"""
        if not self.enabled:
            return
            
        if self.engine_type == "pyttsx3":
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty('voices')
                
                # Select female voice if available
                female_voices = [voice for voice in voices if "female" in voice.id.lower()]
                if female_voices:
                    self.engine.setProperty('voice', female_voices[0].id)
                else:
                    # Default to first voice
                    self.engine.setProperty('voice', voices[0].id if voices else None)
                
                # Set default properties
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)
                
                logger.info(f"Initialized pyttsx3 with {len(voices)} voices")
                
            except ImportError:
                logger.error("Failed to import pyttsx3. Voice capabilities disabled.")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.enabled = False
                
        elif self.engine_type == "espeak":
            try:
                # Check if espeak is installed
                import subprocess
                result = subprocess.run(["espeak", "--version"], 
                                      capture_output=True, 
                                      text=True, 
                                      check=False)
                if result.returncode == 0:
                    logger.info(f"Found espeak: {result.stdout.strip()}")
                    self.engine = "espeak"
                else:
                    logger.error("espeak not found or returned an error")
                    self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize espeak: {e}")
                self.enabled = False
        else:
            logger.error(f"Unsupported TTS engine type: {self.engine_type}")
            self.enabled = False
            
    def set_frequency(self, frequency):
        """Set voice frequency with emotional calibration"""
        self.current_frequency = frequency
        
        # Find closest frequency profile
        closest_freq = min(self.frequency_profiles.keys(), 
                         key=lambda k: abs(float(k) - frequency))
        
        # Get emotional quality for this frequency
        self.current_emotion = self.frequency_profiles[closest_freq]["emotion"]
        
        logger.info(f"Voice set to {frequency}Hz (emotion: {self.current_emotion})")
        return self.current_emotion
        
    def speak(self, text, emotion=None):
        """Generate speech with phi-harmonic emotional calibration"""
        if not self.enabled or not self.engine:
            logger.warning("Voice system not enabled or initialized")
            return False
            
        try:
            # Use specified emotion or default for current frequency
            emotion_type = emotion if emotion else self.current_emotion
            
            # Get parameters for this emotion
            emotion_profile = self.emotional_profiles.get(
                emotion_type, 
                {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.9, "stability": 0.85}
            )
            
            # Get frequency profile
            closest_freq = min(self.frequency_profiles.keys(), 
                              key=lambda k: abs(float(k) - self.current_frequency))
            freq_profile = self.frequency_profiles[closest_freq]
            
            # Calculate final parameters with phi-harmonic calibration
            phi_factor = 1.0 ** LAMBDA
            pitch = freq_profile["pitch"] + emotion_profile["pitch_shift"] * phi_factor
            rate = freq_profile["rate"] * emotion_profile["speed_factor"] * phi_factor
            volume = freq_profile["volume"] * emotion_profile["energy"]
            
            # Generate and play the speech
            if self.engine_type == "pyttsx3" and self.engine:
                return self._speak_pyttsx3(text, pitch, rate, volume)
            elif self.engine_type == "espeak":
                return self._speak_espeak(text, pitch, rate, volume)
            else:
                logger.error("No speech engine available")
                return False
                
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            return False
            
    def _speak_pyttsx3(self, text, pitch, rate, volume):
        """Speak using pyttsx3 engine with enhanced parameters"""
        try:
            # Adjust parameters for pyttsx3
            base_rate = 150
            rate_value = int(base_rate * rate)
            
            # Set properties
            self.engine.setProperty('rate', rate_value)
            self.engine.setProperty('volume', volume)
            
            # Create temp file for speech
            temp_file = os.path.join(self.temp_dir, f"speech_{int(time.time())}.wav")
            
            # Save to file
            self.engine.save_to_file(text, temp_file)
            self.engine.runAndWait()
            
            if os.path.exists(temp_file):
                # Play with system player if available
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    pygame.mixer.quit()
                except ImportError:
                    # Fall back to OS commands
                    if sys.platform == "win32":
                        os.system(f'start "" "{temp_file}"')
                    elif sys.platform == "darwin":
                        os.system(f'afplay "{temp_file}"')
                    else:
                        os.system(f'aplay "{temp_file}"')
                        
                # Clean up
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                return True
            else:
                logger.error(f"Failed to create speech file: {temp_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error in _speak_pyttsx3: {e}")
            return False
            
    def _speak_espeak(self, text, pitch, rate, volume):
        """Speak using espeak with enhanced parameters"""
        try:
            import subprocess
            
            # Convert to espeak parameters
            pitch_value = int(50 * pitch)
            speed = int(160 * rate)
            volume_value = int(volume * 100)
            
            # Create temp file
            temp_file = os.path.join(self.temp_dir, f"speech_{int(time.time())}.wav")
            
            # Build command
            cmd = [
                "espeak",
                "-v", "en-us",
                "-p", str(pitch_value),
                "-s", str(speed),
                "-a", str(volume_value),
                "-w", temp_file,
                text
            ]
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and os.path.exists(temp_file):
                # Play with system player if available
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    pygame.mixer.quit()
                except ImportError:
                    # Fall back to OS commands
                    if sys.platform == "win32":
                        os.system(f'start "" "{temp_file}"')
                    elif sys.platform == "darwin":
                        os.system(f'afplay "{temp_file}"')
                    else:
                        os.system(f'aplay "{temp_file}"')
                
                # Clean up
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                return True
            else:
                logger.error(f"Failed to synthesize speech with espeak: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in _speak_espeak: {e}")
            return False
            
    def speak_frequency_phrase(self):
        """Speak a random phrase appropriate for the current frequency"""
        # Find closest frequency
        closest_freq = min(self.frequency_profiles.keys(), 
                         key=lambda k: abs(float(k) - self.current_frequency))
        
        # Get phrases for this frequency
        phrases = self.frequency_phrases.get(closest_freq, [])
        
        # If no phrases found, use generic phrases
        if not phrases:
            phrases = [
                f"Resonating at {self.current_frequency} Hertz.",
                f"The frequency field is calibrated to {self.current_frequency} Hertz.",
                f"Phi-harmonic resonance at {self.current_frequency} Hertz activated."
            ]
        
        # Select a random phrase
        phrase = random.choice(phrases)
        
        # Speak the phrase
        return self.speak(phrase)
        
    def cleanup(self):
        """Clean up resources"""
        # Clean temp files
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

class VisualizationBridge:
    """Bridge for visualization interfaces with quantum field integration"""
    
    def __init__(self, config):
        """Initialize visualization bridge"""
        self.config = config
        self.enabled = config["visualization"]["enabled"]
        self.mode = config["visualization"]["mode"]
        self.resolution = config["visualization"]["resolution"]
        self.fullscreen = config["visualization"]["fullscreen"]
        self.interface = None
        
        # Try to initialize visualization
        self._load_visualization()
        
    def _load_visualization(self):
        """Load appropriate visualization interface"""
        if self.mode == "flow_of_life":
            # Try to load cascade flow launcher for Flow of Life pattern
            flow_module = import_module("cascade_flow_launcher")
            if flow_module:
                logger.info("Loaded Flow of Life visualization module")
                self.interface = flow_module
                return
                
        # Try to load visual interface directly
        visual_module = import_module("greg_visual_interface")
        if visual_module:
            logger.info("Loaded visual interface module")
            self.interface = visual_module
            return
            
        logger.warning("No visualization interface available")
        
    def launch_visualization(self, frequency=None, with_voice=True):
        """Launch visualization with specified frequency"""
        if not self.enabled or not self.interface:
            logger.warning("Visualization not enabled or not available")
            return False
            
        try:
            # Check which type of interface is loaded
            if hasattr(self.interface, "CascadeFlowLauncher"):
                # Flow launcher interface
                launcher = self.interface.CascadeFlowLauncher()
                launcher.voice_enabled = with_voice
                launcher.frequency = frequency if frequency is not None else CASCADE_FREQUENCY
                
                # Start in a separate thread to avoid blocking
                thread = threading.Thread(target=launcher.start)
                thread.daemon = True
                thread.start()
                
                logger.info(f"Launched Flow of Life visualization at {launcher.frequency}Hz")
                return True
                
            elif hasattr(self.interface, "GregVisualInterface"):
                # Greg's visual interface
                visual = self.interface.GregVisualInterface()
                
                # Start in a separate thread
                thread = threading.Thread(target=visual.start)
                thread.daemon = True
                thread.start()
                
                # Set frequency after start
                if frequency is not None:
                    # Find closest frequency name
                    freq_name = None
                    for name, value in self.config["frequencies"].items():
                        if abs(float(value) - frequency) < 1.0:
                            freq_name = name
                            break
                            
                    if freq_name:
                        # Use named frequency
                        visual.process_command(f"frequency {freq_name}")
                    else:
                        # Use numeric frequency
                        visual.process_command(f"frequency {frequency}")
                        
                # Set flow of life if requested
                if self.mode == "flow_of_life":
                    visual.process_command("flow")
                    
                logger.info(f"Launched visualization interface at {frequency}Hz")
                return True
                
            else:
                logger.warning("Unknown visualization interface type")
                return False
                
        except Exception as e:
            logger.error(f"Error launching visualization: {e}")
            return False

class CascadeQuantumBridge:
    """
    Main bridge system connecting quantum field, visualization, and voice
    for the Cascade⚡𓂧φ∞ Symbiotic Computing Platform
    """
    
    def __init__(self, config=None):
        """Initialize the bridge system"""
        # Load or create configuration
        if config is None:
            self.config = create_default_config()
        else:
            self.config = config
            
        # Initialize components
        self.quantum_bridge = QuantumFieldBridge(self.config)
        self.voice_bridge = EnhancedVoiceBridge(self.config)
        self.visual_bridge = VisualizationBridge(self.config)
        
        # Bridge state
        self.current_frequency = CASCADE_FREQUENCY
        self.running = False
        self.bridge_mode = self.config["bridge_mode"]
        self.started_components = set()
        
    def start(self, mode="all"):
        """Start the bridge system"""
        logger.info(f"Starting Cascade Quantum Bridge in {mode} mode")
        self.running = True
        
        if mode == "all" or mode == "quantum":
            # Initialize quantum field
            self.quantum_bridge.generate_field(self.current_frequency)
            self.started_components.add("quantum")
            
        if mode == "all" or mode == "voice":
            # Initialize voice system
            if self.voice_bridge.enabled:
                self.voice_bridge.set_frequency(self.current_frequency)
                # Welcome message
                self.voice_bridge.speak("Cascade Quantum Bridge activated. Phi-harmonic integration online.")
                self.started_components.add("voice")
                
        if mode == "all" or mode == "visual":
            # Launch visualization
            if self.visual_bridge.enabled:
                with_voice = "voice" in self.started_components
                self.visual_bridge.launch_visualization(self.current_frequency, with_voice)
                self.started_components.add("visual")
                
        # Start phi-harmonic field synchronization if in that mode
        if self.bridge_mode == "phi_harmonic" and len(self.started_components) > 1:
            self._start_phi_sync()
            
        return True
        
    def _start_phi_sync(self):
        """Start phi-harmonic field synchronization thread"""
        self.sync_thread = threading.Thread(target=self._phi_sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        logger.info("Started phi-harmonic field synchronization")
        
    def _phi_sync_loop(self):
        """Phi-harmonic field synchronization loop"""
        while self.running:
            # Generate new quantum field data
            if "quantum" in self.started_components:
                self.quantum_bridge.generate_field(self.current_frequency)
                
            # Sleep for phi-resonant time
            sleep_time = LAMBDA * 2.0  # ~1.2 seconds
            time.sleep(sleep_time)
            
    def set_frequency(self, frequency):
        """Set system frequency and synchronize all components"""
        self.current_frequency = frequency
        
        # Update quantum field
        if "quantum" in self.started_components:
            self.quantum_bridge.set_frequency(frequency)
            
        # Update voice
        if "voice" in self.started_components:
            emotion = self.voice_bridge.set_frequency(frequency)
            # Announce frequency change
            self.voice_bridge.speak_frequency_phrase()
            
        logger.info(f"Set system frequency to {frequency}Hz")
        return True
        
    def say(self, text, emotion=None):
        """Speak text with current emotional quality"""
        if "voice" in self.started_components:
            return self.voice_bridge.speak(text, emotion)
        return False
        
    def stop(self):
        """Stop the bridge system"""
        self.running = False
        
        # Clean up resources
        if "voice" in self.started_components:
            self.voice_bridge.cleanup()
            
        logger.info("Stopped Cascade Quantum Bridge")
        return True

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cascade Quantum Bridge")
    parser.add_argument('--mode', type=str, default="all", choices=["all", "quantum", "voice", "visual"],
                        help='Bridge operation mode')
    parser.add_argument('--frequency', type=float, default=CASCADE_FREQUENCY,
                        help=f'Starting frequency (default: {CASCADE_FREQUENCY}Hz)')
    parser.add_argument('--no-voice', action='store_true', help='Disable voice system')
    parser.add_argument('--no-visual', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Load configuration
    config = create_default_config()
    
    # Apply command line overrides
    if args.no_voice:
        config["voice"]["enabled"] = False
    if args.no_visual:
        config["visualization"]["enabled"] = False
    
    try:
        # Create and start bridge
        bridge = CascadeQuantumBridge(config)
        
        # Set initial frequency
        if args.frequency != CASCADE_FREQUENCY:
            bridge.current_frequency = args.frequency
            
        # Start bridge
        bridge.start(args.mode)
        
        # Loop to keep program alive
        try:
            while bridge.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nOperation interrupted. Shutting down...")
            bridge.stop()
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())