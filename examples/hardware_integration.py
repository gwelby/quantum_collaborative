#!/usr/bin/env python3
"""
Hardware Integration - A practical application of CascadeOS with hardware sensors

This application demonstrates how CascadeOS can interface with hardware sensors
to create a real biofeedback loop. It supports various sensors including:
- Arduino-based heart rate/GSR sensors
- Muse EEG headband
- Webcam-based heart rate detection
- Microphone-based breathing detection

When hardware is not available, it falls back to simulation.
"""

import os
import sys
import time
import random
import threading
import numpy as np
from pathlib import Path

# Add parent directory to path
CASCADE_PATH = Path(__file__).parent.parent.resolve()
if CASCADE_PATH not in sys.path:
    sys.path.append(str(CASCADE_PATH))

# Import CascadeOS components
from CascadeOS import (
    QuantumField,
    ConsciousnessState,
    ConsciousnessFieldInterface,
    create_quantum_field,
    field_to_ascii,
    print_field,
    CascadeSystem,
    PHI, LAMBDA, PHI_PHI,
    SACRED_FREQUENCIES
)

# Optional hardware imports - gracefully handle missing dependencies
try:
    import serial  # For Arduino communication
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: PySerial not found, Arduino integration disabled")

try:
    import cv2  # For webcam-based heart rate
    import dlib  # For face detection
    WEBCAM_AVAILABLE = True
except ImportError:
    WEBCAM_AVAILABLE = False
    print("Warning: OpenCV or dlib not found, webcam integration disabled")

try:
    import pyaudio  # For microphone-based breath detection
    import numpy as np
    from scipy import signal
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: PyAudio or SciPy not found, microphone integration disabled")

try:
    from muselsl import stream, list_muses  # For Muse EEG headband
    MUSE_AVAILABLE = True
except ImportError:
    MUSE_AVAILABLE = False
    print("Warning: muselsl not found, Muse EEG integration disabled")

class HardwareSensor:
    """Base class for hardware sensors."""
    
    def __init__(self, name="Generic Sensor"):
        """Initialize the sensor."""
        self.name = name
        self.is_connected = False
        self.is_sampling = False
        self.data_buffer = []
        self.max_buffer_size = 100
        self.sample_rate = 1.0  # samples per second
        self.last_sample_time = 0
        self.sampling_thread = None
    
    def connect(self):
        """Connect to the sensor."""
        # Base implementation just sets flag
        self.is_connected = True
        print(f"Connected to {self.name}")
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from the sensor."""
        self.stop_sampling()
        self.is_connected = False
        print(f"Disconnected from {self.name}")
    
    def start_sampling(self):
        """Start sampling data from the sensor."""
        if not self.is_connected:
            print(f"Cannot start sampling: {self.name} not connected")
            return False
        
        if self.is_sampling:
            print(f"{self.name} already sampling")
            return True
        
        self.is_sampling = True
        self.sampling_thread = threading.Thread(target=self._sampling_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
        
        print(f"Started sampling from {self.name}")
        return True
    
    def stop_sampling(self):
        """Stop sampling data from the sensor."""
        self.is_sampling = False
        if self.sampling_thread:
            # Wait for thread to terminate
            if self.sampling_thread.is_alive():
                self.sampling_thread.join(timeout=1.0)
            self.sampling_thread = None
        
        print(f"Stopped sampling from {self.name}")
    
    def _sampling_loop(self):
        """Main sampling loop - runs in separate thread."""
        while self.is_sampling:
            current_time = time.time()
            
            # Check if it's time for a new sample
            if current_time - self.last_sample_time >= 1.0 / self.sample_rate:
                try:
                    # Get sample data
                    data = self._read_sample()
                    
                    # Add to buffer with timestamp
                    self.data_buffer.append({
                        "timestamp": current_time,
                        "data": data
                    })
                    
                    # Limit buffer size
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer = self.data_buffer[-self.max_buffer_size:]
                    
                    self.last_sample_time = current_time
                    
                except Exception as e:
                    print(f"Error sampling from {self.name}: {e}")
            
            # Sleep to avoid excessive CPU usage
            time.sleep(0.1)
    
    def _read_sample(self):
        """Read a single sample from the sensor - override in subclasses."""
        # Base implementation returns random data
        return random.random()
    
    def get_latest_data(self):
        """Get the latest data from the sensor."""
        if not self.data_buffer:
            return None
        
        return self.data_buffer[-1]["data"]
    
    def get_average_data(self, seconds=3.0):
        """Get average of data over specified time period."""
        if not self.data_buffer:
            return None
        
        current_time = time.time()
        
        # Filter data by time range
        recent_data = [
            sample["data"] for sample in self.data_buffer
            if current_time - sample["timestamp"] <= seconds
        ]
        
        if not recent_data:
            return None
        
        # Calculate average
        return sum(recent_data) / len(recent_data)


class ArduinoSensor(HardwareSensor):
    """Interface to Arduino-based sensors."""
    
    def __init__(self, port="/dev/ttyUSB0", baud_rate=9600):
        """Initialize Arduino sensor interface."""
        super().__init__(name="Arduino Sensor")
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.data_fields = ["heart_rate", "gsr"]
        self.sample_rate = 2.0  # 2 Hz
    
    def connect(self):
        """Connect to Arduino device."""
        if not SERIAL_AVAILABLE:
            print("PySerial library not available. Using simulation.")
            self.is_connected = True
            return True
        
        try:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=1.0)
            time.sleep(2)  # Allow time for Arduino to reset
            self.is_connected = True
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            print("Falling back to simulation mode")
            self.is_connected = True  # Simulate connection
            self.serial = None
            return True
    
    def disconnect(self):
        """Disconnect from Arduino device."""
        self.stop_sampling()
        if self.serial:
            self.serial.close()
            self.serial = None
        self.is_connected = False
        print("Disconnected from Arduino")
    
    def _read_sample(self):
        """Read a sample from Arduino or simulate data."""
        if self.serial:
            try:
                # Request data from Arduino
                self.serial.write(b"R\n")
                
                # Read response
                line = self.serial.readline().decode("utf-8").strip()
                
                # Parse CSV format: heart_rate,gsr
                values = line.split(",")
                if len(values) >= 2:
                    heart_rate = float(values[0])
                    gsr = float(values[1])
                    return {"heart_rate": heart_rate, "gsr": gsr}
                else:
                    # Invalid data format
                    return self._simulate_data()
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                return self._simulate_data()
        else:
            # Simulation mode
            return self._simulate_data()
    
    def _simulate_data(self):
        """Generate simulated sensor data."""
        # Calculate phase (0-1) for oscillation
        phase = (time.time() % 60) / 60
        
        # Simulate heart rate (60-80 bpm with small variations)
        heart_rate = 70 + 10 * np.sin(phase * 2 * np.pi)
        heart_rate += random.uniform(-2, 2)  # Add noise
        
        # Simulate GSR (3-10 with variations)
        gsr = 6.5 + 3.5 * np.sin(phase * 2 * np.pi * 0.5)
        gsr += random.uniform(-0.5, 0.5)  # Add noise
        
        return {"heart_rate": heart_rate, "gsr": gsr}


class WebcamSensor(HardwareSensor):
    """Webcam-based heart rate and expression detection."""
    
    def __init__(self, camera_id=0):
        """Initialize webcam sensor."""
        super().__init__(name="Webcam Sensor")
        self.camera_id = camera_id
        self.cap = None
        self.face_detector = None
        self.data_fields = ["heart_rate", "expression"]
        self.sample_rate = 1.0  # 1 Hz
        self.ppg_buffer = []  # Photoplethysmography buffer
    
    def connect(self):
        """Connect to webcam."""
        if not WEBCAM_AVAILABLE:
            print("OpenCV or dlib not available. Using simulation.")
            self.is_connected = True
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception("Failed to open webcam")
            
            # Initialize face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            self.is_connected = True
            print(f"Connected to webcam #{self.camera_id}")
            return True
        except Exception as e:
            print(f"Failed to connect to webcam: {e}")
            print("Falling back to simulation mode")
            self.is_connected = True  # Simulate connection
            self.cap = None
            return True
    
    def disconnect(self):
        """Disconnect from webcam."""
        self.stop_sampling()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("Disconnected from webcam")
    
    def _read_sample(self):
        """Read a sample from webcam or simulate data."""
        if self.cap:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    return self._simulate_data()
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector(gray)
                
                if len(faces) > 0:
                    # Get largest face
                    face = max(faces, key=lambda rect: rect.width() * rect.height())
                    
                    # Extract face region
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Extract average color (red channel for PPG)
                    if face_region.size > 0:
                        red_channel = face_region[:, :, 2]
                        ppg_value = np.mean(red_channel)
                        
                        # Add to ppg buffer
                        self.ppg_buffer.append(ppg_value)
                        if len(self.ppg_buffer) > 100:
                            self.ppg_buffer = self.ppg_buffer[-100:]
                        
                        # Estimate heart rate if buffer has enough samples
                        heart_rate = self._estimate_heart_rate()
                        
                        # Detect expression (simplified)
                        # In a real implementation, would use a proper facial expression detector
                        # Here just simulating based on face position
                        expression_value = 0.5 + 0.1 * np.sin(time.time())
                        
                        return {"heart_rate": heart_rate, "expression": expression_value}
                
                # No face detected, return simulated data
                return self._simulate_data()
                
            except Exception as e:
                print(f"Error processing webcam data: {e}")
                return self._simulate_data()
        else:
            # Simulation mode
            return self._simulate_data()
    
    def _estimate_heart_rate(self):
        """Estimate heart rate from PPG signal."""
        if len(self.ppg_buffer) < 50:
            return 70.0  # Default value if not enough data
        
        try:
            # Normalize signal
            signal = np.array(self.ppg_buffer)
            signal = (signal - np.mean(signal)) / (np.std(signal) if np.std(signal) > 0 else 1.0)
            
            # Apply bandpass filter (0.7-3.5 Hz, corresponds to 42-210 BPM)
            fs = self.sample_rate * 10  # Estimated based on actual capture rate
            b, a = signal.butter(2, [0.7/fs*2, 3.5/fs*2], btype='band')
            filtered = signal.lfilter(b, a, signal)
            
            # Find peaks
            peaks, _ = signal.find_peaks(filtered, distance=fs/3.5)  # Min distance between peaks
            
            # Estimate heart rate
            if len(peaks) > 2:
                # Calculate average time between peaks
                peak_times = peaks / fs
                intervals = np.diff(peak_times)
                mean_interval = np.mean(intervals)
                
                # Convert to BPM
                heart_rate = 60.0 / mean_interval if mean_interval > 0 else 70.0
                
                # Sanity check
                heart_rate = max(40.0, min(heart_rate, 180.0))
                return heart_rate
            else:
                return 70.0  # Default value if not enough peaks
                
        except Exception as e:
            print(f"Error estimating heart rate: {e}")
            return 70.0
    
    def _simulate_data(self):
        """Generate simulated webcam sensor data."""
        # Calculate phase (0-1) for oscillation
        phase = (time.time() % 60) / 60
        
        # Simulate heart rate (65-80 bpm with small variations)
        heart_rate = 72.5 + 7.5 * np.sin(phase * 2 * np.pi)
        heart_rate += random.uniform(-2, 2)  # Add noise
        
        # Simulate expression (0-1 score, higher is more engaged/positive)
        expression = 0.6 + 0.2 * np.sin(phase * 2 * np.pi * 0.3)
        expression += random.uniform(-0.05, 0.05)  # Add noise
        expression = max(0.0, min(1.0, expression))  # Clamp to 0-1
        
        return {"heart_rate": heart_rate, "expression": expression}


class AudioSensor(HardwareSensor):
    """Microphone-based breathing detection."""
    
    def __init__(self, device_index=None):
        """Initialize audio sensor."""
        super().__init__(name="Audio Sensor")
        self.device_index = device_index
        self.audio = None
        self.stream = None
        self.chunk_size = 1024
        self.format = 8  # pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.breath_buffer = []
        self.data_fields = ["breath_rate", "breath_amplitude"]
        self.sample_rate = 0.5  # 0.5 Hz (every 2 seconds)
    
    def connect(self):
        """Connect to microphone."""
        if not AUDIO_AVAILABLE:
            print("PyAudio or SciPy not available. Using simulation.")
            self.is_connected = True
            return True
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_connected = True
            print("Connected to microphone")
            return True
        except Exception as e:
            print(f"Failed to connect to microphone: {e}")
            print("Falling back to simulation mode")
            self.is_connected = True  # Simulate connection
            self.audio = None
            self.stream = None
            return True
    
    def disconnect(self):
        """Disconnect from microphone."""
        self.stop_sampling()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
        self.is_connected = False
        print("Disconnected from microphone")
    
    def _read_sample(self):
        """Read a sample from microphone or simulate data."""
        if self.stream:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate RMS amplitude
                amplitude = np.sqrt(np.mean(np.square(audio_data)))
                
                # Add to breath buffer
                self.breath_buffer.append(amplitude)
                if len(self.breath_buffer) > self.rate * 10 // self.chunk_size:  # 10 seconds of data
                    self.breath_buffer = self.breath_buffer[-(self.rate * 10 // self.chunk_size):]
                
                # Estimate breath rate
                breath_rate = self._estimate_breath_rate()
                
                # Normalize amplitude
                norm_amplitude = min(1.0, amplitude / 10000.0)
                
                return {"breath_rate": breath_rate, "breath_amplitude": norm_amplitude}
                
            except Exception as e:
                print(f"Error processing audio data: {e}")
                return self._simulate_data()
        else:
            # Simulation mode
            return self._simulate_data()
    
    def _estimate_breath_rate(self):
        """Estimate breathing rate from audio amplitude."""
        if len(self.breath_buffer) < self.rate * 5 // self.chunk_size:  # Need at least 5 seconds
            return 12.0  # Default value if not enough data
        
        try:
            # Downsample buffer for faster processing
            downsampled = self.breath_buffer[::4]
            
            # Normalize
            signal = np.array(downsampled)
            signal = (signal - np.mean(signal)) / (np.std(signal) if np.std(signal) > 0 else 1.0)
            
            # Apply bandpass filter (0.1-0.5 Hz, corresponds to 6-30 breaths per minute)
            fs = self.rate / (self.chunk_size * 4)  # Effective sample rate after downsampling
            b, a = signal.butter(2, [0.1/fs*2, 0.5/fs*2], btype='band')
            filtered = signal.lfilter(b, a, signal)
            
            # Find peaks
            peaks, _ = signal.find_peaks(filtered, distance=fs/0.5)  # Min distance between peaks
            
            # Estimate breath rate
            if len(peaks) > 2:
                # Calculate average time between peaks
                peak_times = peaks / fs
                intervals = np.diff(peak_times)
                mean_interval = np.mean(intervals)
                
                # Convert to breaths per minute
                breath_rate = 60.0 / mean_interval if mean_interval > 0 else 12.0
                
                # Sanity check
                breath_rate = max(6.0, min(breath_rate, 30.0))
                return breath_rate
            else:
                return 12.0  # Default value if not enough peaks
                
        except Exception as e:
            print(f"Error estimating breath rate: {e}")
            return 12.0
    
    def _simulate_data(self):
        """Generate simulated breathing data."""
        # Calculate phase (0-1) for oscillation
        phase = (time.time() % 30) / 30
        
        # Simulate breathing rate (10-16 breaths per minute)
        breath_rate = 13.0 + 3.0 * np.sin(phase * 2 * np.pi)
        breath_rate += random.uniform(-0.5, 0.5)  # Add noise
        
        # Simulate breath amplitude (0-1)
        breath_phase = (time.time() % (60.0 / breath_rate)) / (60.0 / breath_rate)
        breath_amplitude = 0.5 + 0.4 * np.sin(breath_phase * 2 * np.pi)
        breath_amplitude += random.uniform(-0.05, 0.05)  # Add noise
        breath_amplitude = max(0.1, min(0.9, breath_amplitude))  # Clamp
        
        return {"breath_rate": breath_rate, "breath_amplitude": breath_amplitude}


class MuseEEGSensor(HardwareSensor):
    """Interface to Muse EEG headband."""
    
    def __init__(self):
        """Initialize Muse EEG sensor interface."""
        super().__init__(name="Muse EEG Sensor")
        self.device = None
        self.eeg_data = {
            "alpha": [], "beta": [], "theta": [], "delta": [], "gamma": []
        }
        self.data_fields = ["alpha", "beta", "theta", "delta", "gamma"]
        self.sample_rate = 1.0  # 1 Hz
    
    def connect(self):
        """Connect to Muse EEG headband."""
        if not MUSE_AVAILABLE:
            print("MuseLSL library not available. Using simulation.")
            self.is_connected = True
            return True
        
        try:
            # Look for available devices
            muses = list_muses()
            
            if not muses:
                print("No Muse devices found. Using simulation.")
                self.is_connected = True
                return True
            
            # Connect to first device
            self.device = muses[0]
            
            # Start streaming
            stream(self.device['address'])
            
            # Wait for connection
            time.sleep(5)
            
            self.is_connected = True
            print(f"Connected to Muse EEG: {self.device['name']}")
            return True
        except Exception as e:
            print(f"Failed to connect to Muse EEG: {e}")
            print("Falling back to simulation mode")
            self.is_connected = True  # Simulate connection
            self.device = None
            return True
    
    def disconnect(self):
        """Disconnect from Muse EEG headband."""
        self.stop_sampling()
        # No explicit disconnect in muselsl
        self.device = None
        self.is_connected = False
        print("Disconnected from Muse EEG")
    
    def _read_sample(self):
        """Read EEG data from Muse or simulate."""
        if self.device and MUSE_AVAILABLE:
            try:
                # This would normally pull from an LSL stream
                # For demo purposes, simulate with realistic values
                return self._simulate_data()
            except Exception as e:
                print(f"Error reading from Muse EEG: {e}")
                return self._simulate_data()
        else:
            # Simulation mode
            return self._simulate_data()
    
    def _simulate_data(self):
        """Generate simulated EEG data."""
        # Calculate phase (0-1) for oscillation
        phase = (time.time() % 30) / 30
        
        # Base values for each band (realistic for meditation state)
        alpha_base = 0.6  # Higher during relaxation
        beta_base = 0.4   # Lower during relaxation
        theta_base = 0.5  # Higher during meditation
        delta_base = 0.3  # Not as relevant for meditation
        gamma_base = 0.2  # Can increase during certain meditation states
        
        # Add oscillations and noise
        alpha = alpha_base + 0.2 * np.sin(phase * 2 * np.pi) + random.uniform(-0.05, 0.05)
        beta = beta_base + 0.1 * np.sin(phase * 2 * np.pi * 2) + random.uniform(-0.05, 0.05)
        theta = theta_base + 0.15 * np.sin(phase * 2 * np.pi * 0.5) + random.uniform(-0.05, 0.05)
        delta = delta_base + 0.1 * np.sin(phase * 2 * np.pi * 0.2) + random.uniform(-0.05, 0.05)
        gamma = gamma_base + 0.05 * np.sin(phase * 2 * np.pi * 3) + random.uniform(-0.03, 0.03)
        
        # Ensure values are in valid range
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0, beta))
        theta = max(0.0, min(1.0, theta))
        delta = max(0.0, min(1.0, delta))
        gamma = max(0.0, min(1.0, gamma))
        
        return {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma
        }


class HardwareInterface:
    """Manager for interfacing with hardware sensors and CascadeOS."""
    
    def __init__(self, hardware_mode="auto"):
        """Initialize hardware interface.
        
        Args:
            hardware_mode: "auto", "simulate", or "hardware"
        """
        self.hardware_mode = hardware_mode
        
        # Initialize cascade system
        self.system = CascadeSystem()
        self.system.initialize({
            "dimensions": (34, 55, 34),
            "frequency": "unity"
        })
        
        # Initialize sensors
        self.sensors = {}
        self.active_sensors = set()
        
        # Initialize hardware based on mode
        self._initialize_hardware()
        
        # Set up translation mappings from sensor data to biofeedback
        self._setup_mappings()
        
        # Status flags
        self.is_active = False
        self.last_update_time = 0
        self.update_interval = 1.0  # seconds
        self.session_start_time = 0
        self.session_data = []
    
    def _initialize_hardware(self):
        """Initialize hardware components based on mode."""
        # Check what hardware could be available
        hardware_available = any([SERIAL_AVAILABLE, WEBCAM_AVAILABLE, AUDIO_AVAILABLE, MUSE_AVAILABLE])
        
        # Determine mode
        if self.hardware_mode == "auto":
            if hardware_available:
                self.hardware_mode = "hardware"
            else:
                self.hardware_mode = "simulate"
        
        print(f"Hardware mode: {self.hardware_mode}")
        
        # Initialize sensors
        if self.hardware_mode == "hardware":
            # Try to initialize all supported hardware
            if SERIAL_AVAILABLE:
                self.sensors["arduino"] = ArduinoSensor()
            
            if WEBCAM_AVAILABLE:
                self.sensors["webcam"] = WebcamSensor()
            
            if AUDIO_AVAILABLE:
                self.sensors["audio"] = AudioSensor()
            
            if MUSE_AVAILABLE:
                self.sensors["muse"] = MuseEEGSensor()
        else:
            # Simulation mode - create simulated sensors
            self.sensors["arduino"] = ArduinoSensor()  # Will run in simulation mode
            self.sensors["webcam"] = WebcamSensor()    # Will run in simulation mode
            self.sensors["audio"] = AudioSensor()      # Will run in simulation mode
            self.sensors["muse"] = MuseEEGSensor()     # Will run in simulation mode
    
    def _setup_mappings(self):
        """Set up mappings from sensor data to biofeedback parameters."""
        self.mappings = {
            # Arduino mappings
            "arduino.heart_rate": "heart_rate",
            "arduino.gsr": "skin_conductance",
            
            # Webcam mappings
            "webcam.heart_rate": "heart_rate",
            "webcam.expression": "emotional_state",
            
            # Audio mappings
            "audio.breath_rate": "breath_rate",
            "audio.breath_amplitude": "breath_depth",
            
            # Muse EEG mappings
            "muse.alpha": "eeg_alpha",
            "muse.theta": "eeg_theta",
            "muse.beta": "eeg_beta",
            "muse.gamma": "eeg_gamma",
            "muse.delta": "eeg_delta"
        }
    
    def connect_sensors(self):
        """Connect to all available sensors."""
        available_sensors = []
        
        for name, sensor in self.sensors.items():
            try:
                if sensor.connect():
                    available_sensors.append(name)
                    print(f"Sensor '{name}' connected successfully")
                else:
                    print(f"Failed to connect to sensor '{name}'")
            except Exception as e:
                print(f"Error connecting to sensor '{name}': {e}")
        
        return available_sensors
    
    def disconnect_sensors(self):
        """Disconnect from all sensors."""
        for name, sensor in self.sensors.items():
            try:
                sensor.disconnect()
                print(f"Sensor '{name}' disconnected")
            except Exception as e:
                print(f"Error disconnecting from sensor '{name}': {e}")
        
        self.active_sensors.clear()
    
    def start_sensing(self, sensor_names=None):
        """Start data collection from specified sensors."""
        if sensor_names is None:
            # Use all connected sensors
            sensor_names = list(self.sensors.keys())
        
        for name in sensor_names:
            if name in self.sensors:
                sensor = self.sensors[name]
                try:
                    if sensor.start_sampling():
                        self.active_sensors.add(name)
                        print(f"Sensor '{name}' started sampling")
                    else:
                        print(f"Failed to start sampling from sensor '{name}'")
                except Exception as e:
                    print(f"Error starting sensor '{name}': {e}")
            else:
                print(f"Unknown sensor '{name}'")
        
        return list(self.active_sensors)
    
    def stop_sensing(self):
        """Stop data collection from all sensors."""
        for name in list(self.active_sensors):
            try:
                self.sensors[name].stop_sampling()
                print(f"Sensor '{name}' stopped sampling")
            except Exception as e:
                print(f"Error stopping sensor '{name}': {e}")
        
        self.active_sensors.clear()
    
    def collect_sensor_data(self):
        """Collect latest data from all active sensors."""
        sensor_data = {}
        
        for name in self.active_sensors:
            sensor = self.sensors[name]
            try:
                data = sensor.get_latest_data()
                if data:
                    # Prefix each data field with sensor name
                    for field, value in data.items():
                        sensor_data[f"{name}.{field}"] = value
            except Exception as e:
                print(f"Error collecting data from sensor '{name}': {e}")
        
        return sensor_data
    
    def translate_to_biofeedback(self, sensor_data):
        """Translate sensor data to biofeedback parameters."""
        biofeedback = {}
        
        for sensor_key, value in sensor_data.items():
            if sensor_key in self.mappings:
                biofeedback_key = self.mappings[sensor_key]
                biofeedback[biofeedback_key] = value
        
        # Handle special cases and derived metrics
        
        # If we have both alpha and theta, calculate ratio
        if "eeg_alpha" in biofeedback and "eeg_theta" in biofeedback:
            alpha = biofeedback["eeg_alpha"]
            theta = biofeedback["eeg_theta"]
            if theta > 0:
                biofeedback["alpha_theta_ratio"] = alpha / theta
        
        # Translate expression to emotional state if available
        if "emotional_state" in biofeedback:
            expression = biofeedback["emotional_state"]
            # Map expression value to emotions based on value range
            if expression > 0.7:
                biofeedback["joy"] = expression
            elif expression > 0.5:
                biofeedback["peace"] = expression
            else:
                biofeedback["focus"] = 1.0 - expression
        
        return biofeedback
    
    def start(self):
        """Start the hardware interface."""
        # Connect to sensors
        self.connect_sensors()
        
        # Start sampling
        self.start_sensing()
        
        # Activate Cascade system
        self.system.activate()
        
        # Set active flag
        self.is_active = True
        self.session_start_time = time.time()
        
        print("Hardware interface activated")
        return True
    
    def update(self):
        """Update the system with latest sensor data."""
        if not self.is_active:
            return False
        
        current_time = time.time()
        
        # Only update at specified interval
        if current_time - self.last_update_time < self.update_interval:
            return False
        
        # Collect sensor data
        sensor_data = self.collect_sensor_data()
        
        # Translate to biofeedback
        biofeedback = self.translate_to_biofeedback(sensor_data)
        
        # Update Cascade system
        if biofeedback:
            status = self.system.update(biofeedback)
            
            # Record session data
            self.session_data.append({
                "timestamp": current_time,
                "elapsed": current_time - self.session_start_time,
                "sensor_data": sensor_data,
                "biofeedback": biofeedback,
                "system_status": status
            })
            
            self.last_update_time = current_time
            return True
        
        return False
    
    def stop(self):
        """Stop the hardware interface."""
        # Stop sampling
        self.stop_sensing()
        
        # Disconnect sensors
        self.disconnect_sensors()
        
        # Deactivate Cascade system
        self.system.deactivate()
        
        # Clear active flag
        self.is_active = False
        
        print("Hardware interface deactivated")
        return True
    
    def run_session(self, duration_minutes=5, visualization_interval=10):
        """Run a complete biofeedback session."""
        print("\n" + "=" * 80)
        print("Cascade⚡𓂧φ∞ Hardware Integration Session")
        print("=" * 80)
        
        # Start interface
        print("\nInitializing hardware...")
        self.start()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_viz_time = start_time
        
        try:
            print(f"\nRunning {duration_minutes}-minute session...")
            
            while time.time() < end_time and self.is_active:
                # Update system
                updated = self.update()
                
                # Show visualization periodically
                current_time = time.time()
                if updated and current_time - last_viz_time >= visualization_interval:
                    self.system._visualize_fields()
                    last_viz_time = current_time
                    
                    # Show session progress
                    elapsed = current_time - start_time
                    remaining = end_time - current_time
                    
                    minutes_elapsed = int(elapsed / 60)
                    seconds_elapsed = int(elapsed % 60)
                    minutes_remaining = int(remaining / 60)
                    seconds_remaining = int(remaining % 60)
                    
                    progress = elapsed / (duration_minutes * 60) * 100
                    
                    print(f"\nSession Progress: {progress:.1f}%")
                    print(f"Time: {minutes_elapsed}:{seconds_elapsed:02d} elapsed, " + 
                          f"{minutes_remaining}:{seconds_remaining:02d} remaining")
                    
                    # Show system status
                    if self.session_data:
                        latest = self.session_data[-1]
                        status = latest["system_status"]
                        
                        print(f"\nSystem Coherence: {status['system_coherence']:.4f}")
                        
                        # Show biofeedback values
                        print("\nBiofeedback Values:")
                        for key, value in latest["biofeedback"].items():
                            print(f"  {key}: {value:.2f}")
                
                # Small sleep to prevent CPU overload
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
        
        # Stop interface
        print("\nShutting down hardware...")
        self.stop()
        
        # Show session summary
        self.show_session_summary()
        
        return True
    
    def show_session_summary(self):
        """Show a summary of the biofeedback session."""
        if not self.session_data:
            print("No session data available")
            return
        
        print("\n" + "=" * 80)
        print("Session Summary")
        print("=" * 80)
        
        # Calculate basic stats
        num_samples = len(self.session_data)
        session_duration = self.session_data[-1]["elapsed"] / 60.0  # minutes
        
        print(f"\nSession Duration: {session_duration:.1f} minutes")
        print(f"Data Points Collected: {num_samples}")
        print(f"Sampling Rate: {num_samples / session_duration:.1f} samples per minute")
        
        # Calculate averages for key metrics
        if num_samples > 0:
            # Extract values for each metric
            metrics = {}
            
            # Collect all metrics from biofeedback data
            for entry in self.session_data:
                for key, value in entry["biofeedback"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            
            # Calculate averages
            print("\nAverage Biofeedback Values:")
            for key, values in metrics.items():
                avg_value = sum(values) / len(values)
                print(f"  {key}: {avg_value:.2f}")
            
            # Calculate average system coherence
            coherence_values = [entry["system_status"]["system_coherence"] 
                              for entry in self.session_data]
            avg_coherence = sum(coherence_values) / len(coherence_values)
            
            print(f"\nAverage System Coherence: {avg_coherence:.4f}")
            print(f"Final System Coherence: {coherence_values[-1]:.4f}")
        
        # Show final visualization
        print("\nFinal Field Visualization:")
        self.system._visualize_fields()


def main():
    """Main function to run the hardware integration demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Hardware Integration")
    parser.add_argument("--mode", type=str, choices=["auto", "hardware", "simulate"],
                      default="auto", help="Hardware mode")
    parser.add_argument("--duration", type=int, default=5,
                      help="Session duration in minutes")
    parser.add_argument("--interval", type=int, default=10,
                      help="Visualization interval in seconds")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Cascade⚡𓂧φ∞ Hardware Integration Example")
    print("=" * 80)
    
    hardware = HardwareInterface(hardware_mode=args.mode)
    hardware.run_session(
        duration_minutes=args.duration,
        visualization_interval=args.interval
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())