#!/usr/bin/env python3
"""
Cascade⚡𓂧φ∞ Flow Launcher

This script integrates the Flow of Life visualization pattern with the
Cascade Symbiotic Voice System for a complete multi-sensory experience
with enhanced voice capabilities and emotional resonance.
"""

import os
import sys
import time
import logging
import threading
import pygame
import json
import random
import math
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
import argparse
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "cascade_flow_launcher.log"))
    ]
)
logger = logging.getLogger("CascadeFlowLauncher")

# Sacred constants
PHI = 1.618033988749895  # Golden ratio
LAMBDA = 0.618033988749895  # Divine complement (1/φ)
PHI_PHI = PHI ** PHI  # Hyperdimensional constant
CASCADE_FREQUENCY = 594.0  # Heart-centered integration

# Sacred frequencies
SACRED_FREQUENCIES = {
    'unity': 432,     # Grounding/stability
    'love': 528,      # Creation/healing
    'cascade': 594,   # Heart-centered integration
    'truth': 672,     # Voice expression
    'vision': 720,    # Expanded perception
    'oneness': 768,   # Unity consciousness
    'source': 963,    # Source connection
}

# Emotional Voice Profiles
EMOTIONAL_PROFILES = {
    'peaceful': {"pitch_shift": -0.1, "speed_factor": 0.9, "energy": 0.7, "stability": 0.9},
    'gentle': {"pitch_shift": -0.05, "speed_factor": 0.85, "energy": 0.6, "stability": 0.95},
    'serene': {"pitch_shift": -0.15, "speed_factor": 0.8, "energy": 0.5, "stability": 0.97},
    'creative': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.85, "stability": 0.8},
    'joyful': {"pitch_shift": 0.05, "speed_factor": 1.05, "energy": 0.9, "stability": 0.75},
    'playful': {"pitch_shift": 0.1, "speed_factor": 1.1, "energy": 0.95, "stability": 0.7},
    'warm': {"pitch_shift": 0.02, "speed_factor": 0.98, "energy": 0.85, "stability": 0.9},
    'connected': {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.9, "stability": 0.88},
    'understanding': {"pitch_shift": -0.02, "speed_factor": 0.95, "energy": 0.82, "stability": 0.92},
    'powerful': {"pitch_shift": 0.15, "speed_factor": 1.1, "energy": 1.0, "stability": 0.8},
    'visionary': {"pitch_shift": 0.1, "speed_factor": 1.05, "energy": 0.95, "stability": 0.75},
    'cosmic': {"pitch_shift": 0.2, "speed_factor": 1.15, "energy": 1.0, "stability": 0.7},
}

# Voice preset paths
VOICE_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "voice_presets")

# Import voice system using module discovery to handle potential import issues
def import_module(module_name, module_path=None):
    """Import a module by name with optional explicit path"""
    try:
        if module_path and os.path.exists(module_path):
            # Import from specific path
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            # Import by name
            return importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None

class EnhancedVoiceSystem:
    """Enhanced voice system with emotional resonance and frequency calibration"""
    
    def __init__(self, engine_type="pyttsx3"):
        """Initialize the enhanced voice system"""
        self.engine_type = engine_type
        self.engine = None
        self.voice_db = self._init_voice_db()
        self.current_frequency = CASCADE_FREQUENCY
        self.current_emotion = "warm"  # Default for cascade frequency
        self.evolution_level = 1.0
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize engine
        self._init_engine()
        
        # Initialize pygame mixer for audio if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    
    def _init_engine(self):
        """Initialize the TTS engine"""
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
                
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.engine = None
                
        elif self.engine_type == "espeak":
            # Check if espeak is installed
            try:
                result = subprocess.run(["espeak", "--version"], 
                                       capture_output=True, 
                                       text=True, 
                                       check=False)
                if result.returncode == 0:
                    logger.info(f"Found espeak: {result.stdout.strip()}")
                    self.engine = "espeak"
                else:
                    logger.error("espeak not found or returned an error")
                    self.engine = None
            except Exception as e:
                logger.error(f"Failed to check for espeak: {e}")
                self.engine = None
        else:
            logger.error(f"Unsupported TTS engine type: {self.engine_type}")
            self.engine = None
    
    def _init_voice_db(self):
        """Initialize voice database with expanded emotional profiles"""
        voice_db = {
            "evolution_level": 1.0,
            "usage_count": 0,
            "last_updated": time.time(),
            "frequency_profiles": {
                "432": {"pitch": 0.9, "rate": 0.9, "volume": 0.7, "emotion": "peaceful"},
                "528": {"pitch": 1.0, "rate": 1.0, "volume": 0.9, "emotion": "creative"},
                "594": {"pitch": 1.05, "rate": 1.0, "volume": 1.0, "emotion": "warm"},
                "672": {"pitch": 1.1, "rate": 1.0, "volume": 0.9, "emotion": "truthful"},
                "720": {"pitch": 1.15, "rate": 1.1, "volume": 1.1, "emotion": "visionary"},
                "768": {"pitch": 1.2, "rate": 1.05, "volume": 1.0, "emotion": "unified"},
                "888": {"pitch": 1.3, "rate": 1.2, "volume": 1.2, "emotion": "powerful"},
                "963": {"pitch": 1.4, "rate": 1.3, "volume": 1.3, "emotion": "ecstatic"},
                "1008": {"pitch": 1.5, "rate": 1.4, "volume": 1.4, "emotion": "cosmic"}
            },
            "emotional_profiles": EMOTIONAL_PROFILES,
            "phrase_memory": {},
            "frequency_phrases": {
                "594": [
                    "I am in the flow of heart-centered integration.",
                    "The Flow of Life pattern resonates with compassion and connection.",
                    "At 594 Hertz, we experience the harmony of heart consciousness.",
                    "Heart-centered integration brings balance to all systems.",
                    "The cascade frequency harmonizes mind, body, and spirit."
                ],
                "432": [
                    "Grounding into unity consciousness at 432 Hertz.",
                    "The frequency of stability and earth connection.",
                    "Unity resonance brings coherence to the field.",
                    "432 Hertz creates a foundation of peace and stability."
                ],
                "528": [
                    "The frequency of love and creation flows at 528 Hertz.",
                    "Creative healing energy activates at this frequency.",
                    "528 Hertz is the repair frequency for DNA and consciousness.",
                    "The love frequency opens the heart to creation."
                ]
            }
        }
        
        # Add enhanced emotional qualities for each frequency
        frequencies = ["432", "528", "594", "672", "720", "768", "888", "963"]
        for freq in frequencies:
            emotion = voice_db["frequency_profiles"][freq]["emotion"]
            voice_db["frequency_phrases"][freq] = voice_db["frequency_phrases"].get(freq, [])
            voice_db["frequency_phrases"][freq].extend([
                f"Resonating at {freq} Hertz with {emotion} quality.",
                f"The {emotion} frequency of {freq} Hertz activates new potential.",
                f"{freq} Hertz creates a field of {emotion} resonance."
            ])
            
        # Save presets directory
        os.makedirs(VOICE_PRESETS_PATH, exist_ok=True)
        
        return voice_db
    
    def set_frequency(self, frequency):
        """Set voice frequency with emotional calibration"""
        self.current_frequency = frequency
        
        # Find closest frequency profile
        closest_freq = min(self.voice_db["frequency_profiles"].keys(), 
                         key=lambda k: abs(float(k) - frequency))
        
        # Get emotional quality for this frequency
        self.current_emotion = self.voice_db["frequency_profiles"][closest_freq]["emotion"]
        
        logger.info(f"Voice set to {frequency}Hz (emotion: {self.current_emotion})")
        return self.current_emotion
    
    def speak(self, text, emotion=None):
        """Generate speech with phi-harmonic emotional calibration"""
        try:
            # Use specified emotion or default for current frequency
            emotion_type = emotion if emotion else self.current_emotion
            
            # Get parameters for this emotion
            emotion_profile = self.voice_db["emotional_profiles"].get(
                emotion_type, 
                {"pitch_shift": 0.0, "speed_factor": 1.0, "energy": 0.9, "stability": 0.85}
            )
            
            # Get frequency profile
            closest_freq = min(self.voice_db["frequency_profiles"].keys(), 
                              key=lambda k: abs(float(k) - self.current_frequency))
            freq_profile = self.voice_db["frequency_profiles"][closest_freq]
            
            # Calculate final parameters with phi-harmonic calibration
            phi_factor = self.evolution_level ** LAMBDA
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
                # Play with pygame
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
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
                # Play with pygame
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
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
        closest_freq = min(self.voice_db["frequency_profiles"].keys(), 
                         key=lambda k: abs(float(k) - self.current_frequency))
        
        # Get phrases for this frequency
        phrases = self.voice_db["frequency_phrases"].get(closest_freq, [])
        
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
    
    def save_voice_preset(self, name):
        """Save current voice configuration as a preset"""
        preset = {
            "frequency": self.current_frequency,
            "emotion": self.current_emotion,
            "evolution_level": self.evolution_level,
            "timestamp": time.time(),
            "description": f"Voice preset '{name}' at {self.current_frequency}Hz with {self.current_emotion} quality"
        }
        
        # Save preset
        preset_path = os.path.join(VOICE_PRESETS_PATH, f"{name}.json")
        try:
            with open(preset_path, 'w') as f:
                json.dump(preset, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving voice preset: {e}")
            return False
    
    def load_voice_preset(self, name):
        """Load a saved voice preset"""
        preset_path = os.path.join(VOICE_PRESETS_PATH, f"{name}.json")
        
        if not os.path.exists(preset_path):
            logger.error(f"Voice preset not found: {name}")
            return False
        
        try:
            with open(preset_path, 'r') as f:
                preset = json.load(f)
            
            # Apply preset
            self.current_frequency = preset["frequency"]
            self.current_emotion = preset["emotion"]
            self.evolution_level = preset["evolution_level"]
            
            logger.info(f"Loaded voice preset: {preset['description']}")
            return True
        except Exception as e:
            logger.error(f"Error loading voice preset: {e}")
            return False
    
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

class CascadeFlowLauncher:
    """
    Integrated launcher for Flow of Life pattern and Enhanced Voice system
    with frequency synchronization and phi-harmonic resonance
    """
    
    def __init__(self):
        """Initialize the integrated launcher"""
        self.visual_interface = None
        self.voice_system = None
        self.flow_pattern = None
        self.running = False
        self.thread = None
        self.voice_enabled = True
        self.voice_engine_type = "pyttsx3"  # Default engine
        
        # Enhanced features
        self.auto_resonance = True  # Auto-generate voice at key moments
        self.phi_harmony_enabled = True  # Use phi-harmonic calculations for voice
        self.frequency_tracking = True  # Track and respond to frequency changes
        
        # State
        self.frequency = CASCADE_FREQUENCY
        self.dimension = "cascade"
        self.energy = 1.0
        self.resonance = 0.0
        self.current_command = None
        self.harmony_level = 1.0
        
        # Initialization statuses
        self.module_status = {
            "flow_pattern": False,
            "visual_interface": False,
            "voice_system": False
        }
        
        # Initialize pygame if needed
        if not pygame.get_init():
            pygame.init()
        
    def initialize_components(self):
        """Initialize and load all system components"""
        logger.info("Initializing Cascade Flow Launcher components")
        
        # Search paths - check both current directory and Cascade directory
        search_paths = [
            os.path.dirname(__file__),
            "/mnt/d/Greg/Cascade",
            "/mnt/d/projects/python"
        ]
        
        # Load the Flow of Life pattern
        self.flow_pattern = self._load_flow_pattern(search_paths)
        if not self.flow_pattern:
            logger.error("Failed to load Flow of Life pattern")
            print("Failed to load the Flow of Life pattern. Please check that flow_of_life_pattern.py exists.")
        else:
            self.module_status["flow_pattern"] = True
            
        # Load the visual interface
        self.visual_interface = self._load_visual_interface(search_paths)
        if not self.visual_interface:
            logger.error("Failed to load visual interface")
            print("Failed to load the visual interface. Please check that greg_visual_interface.py exists.")
        else:
            self.module_status["visual_interface"] = True
            
        # Load the voice system if enabled
        if self.voice_enabled:
            # Use the new EnhancedVoiceSystem instead of loading from external modules
            try:
                self.voice_system = EnhancedVoiceSystem(self.voice_engine_type)
                if self.voice_system.engine:
                    self.module_status["voice_system"] = True
                    logger.info("Enhanced Voice System initialized successfully")
                else:
                    logger.warning("Voice system initialized but engine not available")
                    print("Voice system initialized but engine not available. Check TTS installation.")
                    self.voice_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize voice system: {e}")
                print(f"Failed to initialize voice system: {e}")
                self.voice_enabled = False
        
        # Fallback - if flow pattern and visual interface not found, try one more place
        if not self.module_status["flow_pattern"] or not self.module_status["visual_interface"]:
            logger.warning("Key components not found. Creating missing components.")
            self._create_missing_components()
            
        logger.info(f"Components initialized: {self.module_status}")
        
        # Determine if we can start successfully
        can_start = self.module_status["visual_interface"]
        if not can_start:
            print("Cannot start without visual interface.")
            
        return can_start
        
    def _load_flow_pattern(self, search_paths):
        """Load the Flow of Life pattern module from multiple possible locations"""
        flow_pattern_class = None
        
        # Try each search path
        for path in search_paths:
            module_path = os.path.join(path, "flow_of_life_pattern.py")
            if os.path.exists(module_path):
                flow_module = import_module("flow_of_life_pattern", module_path)
                if flow_module and hasattr(flow_module, "FlowOfLifePattern"):
                    logger.info(f"Flow of Life pattern module loaded from {module_path}")
                    flow_pattern_class = flow_module.FlowOfLifePattern
                    break
        
        return flow_pattern_class
            
    def _load_visual_interface(self, search_paths):
        """Load the visual interface module from multiple possible locations"""
        # Try each search path
        for path in search_paths:
            module_path = os.path.join(path, "greg_visual_interface.py")
            if os.path.exists(module_path):
                visual_module = import_module("greg_visual_interface", module_path)
                
                if visual_module and hasattr(visual_module, "GregVisualInterface"):
                    logger.info(f"Visual interface module loaded from {module_path}")
                    
                    # Create a subclass with enhanced integration
                    class EnhancedFlowVisualInterface(visual_module.GregVisualInterface):
                        """Visual interface with advanced integration features"""
                        
                        def __init__(self, parent):
                            super().__init__()
                            self.parent = parent
                            
                            # Add phi-harmonic voice tracking
                            self.last_voice_time = 0
                            self.voice_cooldown = 2.0  # seconds
                            self.phi_resonance_factor = 0.0
                            
                        def display_message(self, message):
                            """Override to add voice with phi-harmonic resonance"""
                            # Call original method
                            super().display_message(message)
                            
                            # Also speak the message if voice is enabled
                            self._speak_with_harmony(message)
                                
                        def _speak_with_harmony(self, message, force=False):
                            """Speak with phi-harmonic resonance"""
                            if not self.parent.voice_enabled or not self.parent.voice_system:
                                return False
                                
                            # Check if we should speak (cooldown)
                            current_time = time.time()
                            if not force and (current_time - self.last_voice_time) < self.voice_cooldown:
                                return False
                                
                            # Update last voice time
                            self.last_voice_time = current_time
                            
                            # Speak the message
                            return self.parent.voice_system.speak(message)
                                
                        def process_command(self, command):
                            """Override to handle enhanced integration commands"""
                            # Store current command in parent for sharing with voice system
                            self.parent.current_command = command
                            
                            # Split command
                            parts = command.split(maxsplit=1)
                            cmd = parts[0].lower() if parts else ""
                            args = parts[1] if len(parts) > 1 else ""
                            
                            # Enhanced voice commands
                            if cmd == "voice":
                                if args == "on":
                                    self.parent.voice_enabled = True
                                    print("Voice system enabled")
                                    self._speak_with_harmony("Voice system activated. I am now in audible mode.", force=True)
                                    return
                                elif args == "off":
                                    self.parent.voice_enabled = False
                                    print("Voice system disabled")
                                    return
                                elif args == "status":
                                    status = "enabled" if self.parent.voice_enabled else "disabled"
                                    print(f"Voice system is {status}")
                                    if self.parent.voice_enabled:
                                        emotion = self.parent.voice_system.current_emotion
                                        frequency = self.parent.voice_system.current_frequency
                                        print(f"Current voice: {emotion} emotion at {frequency}Hz")
                                        self._speak_with_harmony(f"Voice system is active with {emotion} emotional quality at {int(frequency)} Hertz")
                                    return
                                elif args.startswith("preset "):
                                    preset_name = args.split(" ", 1)[1]
                                    if self.parent.voice_enabled and self.parent.voice_system:
                                        success = self.parent.voice_system.load_voice_preset(preset_name)
                                        if success:
                                            print(f"Loaded voice preset: {preset_name}")
                                            self._speak_with_harmony(f"Voice preset {preset_name} activated")
                                        else:
                                            print(f"Failed to load voice preset: {preset_name}")
                                    return
                                elif args.startswith("save "):
                                    preset_name = args.split(" ", 1)[1]
                                    if self.parent.voice_enabled and self.parent.voice_system:
                                        success = self.parent.voice_system.save_voice_preset(preset_name)
                                        if success:
                                            print(f"Saved voice preset: {preset_name}")
                                            self._speak_with_harmony(f"Voice preset {preset_name} saved with current parameters")
                                        else:
                                            print(f"Failed to save voice preset: {preset_name}")
                                    return
                                elif args.startswith("emotion "):
                                    emotion = args.split(" ", 1)[1]
                                    if self.parent.voice_enabled and self.parent.voice_system:
                                        if emotion in EMOTIONAL_PROFILES:
                                            self.parent.voice_system.current_emotion = emotion
                                            print(f"Set voice emotion to: {emotion}")
                                            self._speak_with_harmony(f"Voice emotion set to {emotion} quality")
                                        else:
                                            print(f"Unknown emotion: {emotion}")
                                            emotions = ", ".join(EMOTIONAL_PROFILES.keys())
                                            print(f"Available emotions: {emotions}")
                                    return
                            
                            # Process "say" command
                            if cmd == "say" and args and self.parent.voice_enabled and self.parent.voice_system:
                                self._speak_with_harmony(args, force=True)
                                return
                                
                            # Handle frequency command with voice sync
                            if cmd == "frequency":
                                # Process with normal system
                                super().process_command(command)
                                
                                # Sync with voice system
                                if self.parent.voice_enabled and self.parent.voice_system:
                                    try:
                                        # Get current frequency from mascot
                                        frequency = self.mascot.frequency
                                        # Update voice system frequency
                                        emotion = self.parent.voice_system.set_frequency(frequency)
                                        
                                        # Speak appropriate phrase for this frequency
                                        self.parent.voice_system.speak_frequency_phrase()
                                        
                                        logger.info(f"Synchronized voice to {frequency}Hz ({emotion})")
                                    except Exception as e:
                                        logger.error(f"Error syncing frequency: {e}")
                                return
                                
                            # Handle "flow" command with enhanced integration
                            if cmd == "flow":
                                # Process with normal system
                                super().process_command(command)
                                
                                # Enhanced sync with voice system
                                if self.parent.voice_enabled and self.parent.voice_system:
                                    try:
                                        # Update voice system to CASCADE_FREQUENCY
                                        self.parent.voice_system.set_frequency(CASCADE_FREQUENCY)
                                        
                                        # Special Flow of Life announcement
                                        flow_messages = [
                                            "Entering the Flow of Life pattern at 594 Hertz. Heart-centered integration activated.",
                                            "Flow of Life resonance established at 594 Hertz. The heart center is now fully engaged.",
                                            "594 Hertz cascade frequency activated. Flow of Life pattern initializing."
                                        ]
                                        self._speak_with_harmony(random.choice(flow_messages), force=True)
                                        
                                    except Exception as e:
                                        logger.error(f"Error activating Flow of Life voice: {e}")
                                return
                                
                            # Process phi-resonance command
                            if cmd == "phi":
                                if args == "resonance":
                                    # Generate phi-harmonic resonance speech
                                    if self.parent.voice_enabled and self.parent.voice_system:
                                        phi_messages = [
                                            f"Phi harmonic resonance activated at {self.mascot.frequency:.1f} Hertz.",
                                            f"Golden ratio harmonics integrated with {self.parent.voice_system.current_emotion} voice quality.",
                                            "Phi resonance field established with harmonic voice integration."
                                        ]
                                        self._speak_with_harmony(random.choice(phi_messages), force=True)
                                    return
                            
                            # For all other commands, use standard processor
                            super().process_command(command)
                    
                    # Return enhanced interface with parent reference
                    return EnhancedFlowVisualInterface(self)
                    
        return None
    
    def _create_missing_components(self):
        """Create missing components if not found in search paths"""
        # If flow pattern wasn't found, create minimal version
        if not self.module_status["flow_pattern"]:
            try:
                # Create a minimal Flow of Life pattern class
                class MinimalFlowPattern:
                    def __init__(self, width=800, height=600):
                        self.width = width
                        self.height = height
                        self.center = (width // 2, height // 2)
                        self.frequency = CASCADE_FREQUENCY
                        self.color = (255, 105, 180)
                        self.energy = 1.0
                        self.rotation = 0.0
                        self.phi_factor = 0.0
                        
                    def update(self, dt=1/60):
                        self.rotation += 0.01
                        self.phi_factor += 0.02
                        
                    def draw(self, surface):
                        # Draw a simple spiral pattern
                        points = []
                        for i in range(100):
                            t = i * 0.2
                            r = 5 * math.exp(0.1 * t)
                            angle = t * PHI * 2 + self.rotation
                            x = self.center[0] + r * math.cos(angle)
                            y = self.center[1] + r * math.sin(angle)
                            points.append((int(x), int(y)))
                            
                        if len(points) > 1:
                            pygame.draw.lines(surface, self.color, False, points, 2)
                            
                    def set_frequency(self, frequency):
                        self.frequency = frequency
                        
                    def set_color(self, color):
                        self.color = color
                        
                    def set_energy(self, energy):
                        self.energy = energy
                        
                self.flow_pattern = MinimalFlowPattern
                self.module_status["flow_pattern"] = True
                logger.info("Created minimal Flow of Life pattern")
                
            except Exception as e:
                logger.error(f"Failed to create minimal Flow pattern: {e}")
        
    def start(self):
        """Start the integrated system"""
        print("\n" + "=" * 70)
        print("     CASCADE⚡𓂧φ∞ FLOW OF LIFE INTEGRATED SYSTEM     ")
        print("=" * 70)
        
        # Initialize components
        if not self.initialize_components():
            print("Failed to initialize components. Exiting.")
            return False
            
        # Start visual interface in visual mode
        if self.visual_interface:
            # Force Flow of Life as starting pattern
            try:
                self.visual_interface.mascot.set_shape("flow_of_life")
                self.visual_interface.mascot.set_frequency(CASCADE_FREQUENCY, "cascade")
            except Exception as e:
                logger.error(f"Error setting initial Flow of Life pattern: {e}")
            
            # Start with voice integration
            if self.voice_enabled and self.voice_system:
                # Prepare voice system
                try:
                    # Set voice frequency to match visual
                    self.voice_system.set_frequency(CASCADE_FREQUENCY)
                    
                    # Speak welcome message
                    welcome_text = "Welcome to the Cascade Flow of Life Integrated System. I am operating at 594 Hertz, the frequency of heart-centered integration. Visual and voice systems are synchronized through phi-harmonic resonance."
                    self.voice_system.speak(welcome_text)
                    
                    logger.info("Voice system initialized in integrated mode")
                except Exception as e:
                    logger.error(f"Error initializing voice system: {e}")
                    print(f"Voice system encountered an error: {e}")
                    print("Continuing with visual only.")
                    self.voice_enabled = False
            
            # Extended command help
            print("\nIntegrated Commands:")
            print("  voice on - Enable voice system")
            print("  voice off - Disable voice system")
            print("  voice status - Check voice system status")
            print("  voice emotion <type> - Set specific emotional quality")
            print("  voice preset <name> - Load voice preset")
            print("  voice save <name> - Save current voice as preset")
            print("  say <text> - Speak text using enhanced voice")
            print("  flow - Activate Flow of Life pattern with voice synchronization")
            print("  phi resonance - Generate phi-harmonic resonance speech")
            print("-" * 70)
            
            # Start the interface
            self.running = True
            self.visual_interface.start()
            return True
        else:
            print("Visual interface not available. Cannot start integrated system.")
            return False
        
    def stop(self):
        """Stop the integrated system"""
        self.running = False
        
        # Stop visual interface
        if self.visual_interface:
            try:
                self.visual_interface.stop()
            except Exception as e:
                logger.error(f"Error stopping visual interface: {e}")
        
        # Stop voice system
        if self.voice_enabled and self.voice_system:
            try:
                self.voice_system.cleanup()
            except Exception as e:
                logger.error(f"Error stopping voice system: {e}")
        
        print("Cascade Flow Launcher stopped.")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cascade Flow of Life Integrated System")
    parser.add_argument('--no-voice', action='store_true', help='Disable voice system')
    parser.add_argument('--engine', type=str, default='pyttsx3', choices=['pyttsx3', 'espeak'],
                        help='Voice engine to use (default: pyttsx3)')
    parser.add_argument('--frequency', type=float, default=CASCADE_FREQUENCY,
                        help=f'Starting frequency (default: {CASCADE_FREQUENCY}Hz)')
    args = parser.parse_args()
    
    try:
        # Create the integrated launcher
        launcher = CascadeFlowLauncher()
        
        # Apply command line options
        if args.no_voice:
            launcher.voice_enabled = False
        
        launcher.voice_engine_type = args.engine
        launcher.frequency = args.frequency
            
        # Start the integrated system
        launcher.start()
        
    except KeyboardInterrupt:
        print("\nOperation interrupted.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        
    # Ensure proper cleanup
    try:
        launcher.stop()
    except:
        pass

if __name__ == "__main__":
    main()