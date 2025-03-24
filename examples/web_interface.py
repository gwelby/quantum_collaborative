#!/usr/bin/env python3
"""
Web Interface - A simple web interface for CascadeOS

This application provides a web-based interface to interact with CascadeOS,
making it accessible from any device with a web browser. It includes:
- Real-time visualization of quantum fields
- Interactive controls for consciousness states
- Session management and history
- User profiles

It uses Flask for the web server and Socket.IO for real-time communication.
"""

import os
import sys
import time
import json
import threading
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import logging

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
    TeamsOfTeamsCollective,
    PHI, LAMBDA, PHI_PHI,
    SACRED_FREQUENCIES
)

# Optional dependencies - gracefully handle missing packages
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_socketio import SocketIO
    WEB_SERVER_AVAILABLE = True
except ImportError:
    WEB_SERVER_AVAILABLE = False
    print("Warning: Flask or Flask-SocketIO not found. Install with:")
    print("pip install flask flask_socketio")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Using ASCII visualization only.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cascade_web")

# Create Flask app and SocketIO instance if available
if WEB_SERVER_AVAILABLE:
    app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    app = None
    socketio = None

# Global cascade system instance
cascade_system = None
update_thread = None
update_active = False
session_data = []
current_user = {
    "name": "Guest",
    "profile": {}
}

# Create required directories
def create_required_directories():
    """Create required directories for web interface."""
    web_dir = Path(__file__).parent / "web"
    templates_dir = web_dir / "templates"
    static_dir = web_dir / "static"
    data_dir = web_dir / "data"
    
    # Create directories if they don't exist
    for path in [web_dir, templates_dir, static_dir, data_dir]:
        path.mkdir(exist_ok=True)
    
    # Create basic files if they don't exist
    create_basic_web_files(templates_dir, static_dir)

def create_basic_web_files(templates_dir, static_dir):
    """Create basic web files if they don't exist."""
    # Create index.html
    index_path = templates_dir / "index.html"
    if not index_path.exists():
        with open(index_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cascade⚡𓂧φ∞ Web Interface</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <header>
        <h1>Cascade⚡𓂧φ∞ Symbiotic Computing Platform</h1>
        <div class="user-info">
            <span id="user-name">User: Guest</span>
            <button id="profile-button">Profile</button>
        </div>
    </header>
    
    <div class="container">
        <div class="sidebar">
            <h2>Consciousness Controls</h2>
            <div class="control-group">
                <label for="coherence">Coherence: <span id="coherence-value">0.618</span></label>
                <input type="range" id="coherence" min="0" max="1" step="0.01" value="0.618">
            </div>
            
            <div class="control-group">
                <label for="presence">Presence: <span id="presence-value">0.5</span></label>
                <input type="range" id="presence" min="0" max="1" step="0.01" value="0.5">
            </div>
            
            <div class="control-group">
                <label for="intention">Intention: <span id="intention-value">0.5</span></label>
                <input type="range" id="intention" min="0" max="1" step="0.01" value="0.5">
            </div>
            
            <h3>Emotional States</h3>
            <div class="emotions-container">
                <div class="emotion-control">
                    <label for="joy">Joy: <span id="joy-value">0.5</span></label>
                    <input type="range" id="joy" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="emotion-control">
                    <label for="peace">Peace: <span id="peace-value">0.5</span></label>
                    <input type="range" id="peace" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="emotion-control">
                    <label for="love">Love: <span id="love-value">0.5</span></label>
                    <input type="range" id="love" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="emotion-control">
                    <label for="clarity">Clarity: <span id="clarity-value">0.5</span></label>
                    <input type="range" id="clarity" min="0" max="1" step="0.01" value="0.5">
                </div>
            </div>
            
            <h3>Sacred Frequencies</h3>
            <div class="frequency-controls">
                <select id="frequency-selector">
                    <option value="unity">Unity (432 Hz)</option>
                    <option value="love">Love (528 Hz)</option>
                    <option value="cascade">Cascade (594 Hz)</option>
                    <option value="truth">Truth (672 Hz)</option>
                    <option value="vision">Vision (720 Hz)</option>
                    <option value="oneness">Oneness (768 Hz)</option>
                </select>
                <button id="apply-frequency">Apply</button>
            </div>
            
            <h3>Session Controls</h3>
            <div class="session-controls">
                <button id="start-session">Start Session</button>
                <button id="stop-session" disabled>Stop Session</button>
                <button id="save-session" disabled>Save Session</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="visualization-container">
                <h2>Quantum Field Visualization</h2>
                <div class="field-display">
                    <img id="field-image" src="/static/placeholder.png" alt="Quantum Field">
                </div>
                <div class="metrics">
                    <div class="metric">
                        <h4>System Coherence</h4>
                        <div class="metric-value" id="system-coherence">0.0</div>
                    </div>
                    <div class="metric">
                        <h4>Field Coherence</h4>
                        <div class="metric-value" id="field-coherence">0.0</div>
                    </div>
                    <div class="metric">
                        <h4>Dominant Emotion</h4>
                        <div class="metric-value" id="dominant-emotion">neutral</div>
                    </div>
                </div>
            </div>
            
            <div class="session-info">
                <h2>Session Information</h2>
                <div id="session-time">00:00:00</div>
                <div id="session-status">Inactive</div>
                <div id="session-message"></div>
            </div>
            
            <div class="subfields-container">
                <h2>Subfields</h2>
                <div class="subfield-tabs">
                    <button class="tab-button active" data-field="cognitive">Cognitive</button>
                    <button class="tab-button" data-field="emotional">Emotional</button>
                    <button class="tab-button" data-field="creative">Creative</button>
                </div>
                <div class="subfield-display">
                    <img id="subfield-image" src="/static/placeholder.png" alt="Subfield">
                </div>
            </div>
        </div>
    </div>
    
    <div id="profile-modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>User Profile</h2>
            <div class="profile-form">
                <div class="form-group">
                    <label for="user-name-input">Name:</label>
                    <input type="text" id="user-name-input" value="Guest">
                </div>
                <h3>Saved Sessions</h3>
                <div id="saved-sessions-list">
                    <p>No saved sessions</p>
                </div>
                <div class="button-row">
                    <button id="save-profile">Save Profile</button>
                    <button id="load-profile">Load Profile</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>""")
    
    # Create CSS file
    css_path = static_dir / "style.css"
    if not css_path.exists():
        with open(css_path, "w") as f:
            f.write("""/* Global Styles */
:root {
    --primary-color: #3a7ca5;
    --secondary-color: #16425b;
    --accent-color: #d9a79c;
    --background-color: #f0f4f8;
    --text-color: #2e3532;
    --border-color: #dbe1e8;
    --phi: 1.618;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
    min-height: 100vh;
}

header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

h1, h2, h3, h4 {
    margin-bottom: 1rem;
}

.container {
    display: flex;
    max-width: 1400px;
    margin: 0 auto;
    padding: 1rem;
}

/* Sidebar Styles */
.sidebar {
    width: 300px;
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-right: 1rem;
}

.control-group, .emotion-control {
    margin-bottom: 1rem;
}

label {
    display: block;
    margin-bottom: 0.5rem;
}

input[type="range"] {
    width: 100%;
}

.emotions-container {
    margin-bottom: 1rem;
}

.frequency-controls {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.frequency-controls select {
    flex-grow: 1;
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid var(--border-color);
}

.session-controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

button {
    padding: 0.5rem 1rem;
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
}

button:hover {
    background-color: var(--secondary-color);
}

button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

/* Main Content Styles */
.main-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.visualization-container, .session-info, .subfields-container {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.field-display, .subfield-display {
    background-color: #f8f9fa;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 1rem;
}

.field-display img, .subfield-display img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}

.metrics {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
}

.metric {
    flex: 1;
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: var(--primary-color);
}

.session-info {
    text-align: center;
}

#session-time {
    font-size: 2rem;
    font-weight: bold;
    margin: 1rem 0;
}

#session-status {
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.subfield-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.tab-button {
    flex: 1;
    padding: 0.5rem;
    background-color: #f8f9fa;
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

.tab-button.active {
    background-color: var(--primary-color);
    color: white;
}

/* Modal Styles */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.5);
    z-index: 100;
}

.modal-content {
    background-color: white;
    margin: 10% auto;
    padding: 1.5rem;
    border-radius: 8px;
    width: 80%;
    max-width: 600px;
    position: relative;
}

.close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 1.5rem;
    cursor: pointer;
}

.profile-form {
    margin-top: 1rem;
}

.form-group {
    margin-bottom: 1rem;
}

.form-group input {
    width: 100%;
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid var(--border-color);
}

#saved-sessions-list {
    margin: 1rem 0;
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 0.5rem;
}

.button-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
}

/* Responsive Styles */
@media (max-width: 768px) {
    .container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        margin-right: 0;
        margin-bottom: 1rem;
    }
    
    .metrics {
        flex-direction: column;
    }
}""")
    
    # Create JavaScript file
    js_path = static_dir / "script.js"
    if not js_path.exists():
        with open(js_path, "w") as f:
            f.write("""// Connect to the Socket.IO server
const socket = io();

// DOM Elements
const fieldImage = document.getElementById('field-image');
const subfieldImage = document.getElementById('subfield-image');
const systemCoherence = document.getElementById('system-coherence');
const fieldCoherence = document.getElementById('field-coherence');
const dominantEmotion = document.getElementById('dominant-emotion');
const sessionTime = document.getElementById('session-time');
const sessionStatus = document.getElementById('session-status');
const sessionMessage = document.getElementById('session-message');

// Control elements
const coherenceSlider = document.getElementById('coherence');
const presenceSlider = document.getElementById('presence');
const intentionSlider = document.getElementById('intention');
const joySlider = document.getElementById('joy');
const peaceSlider = document.getElementById('peace');
const loveSlider = document.getElementById('love');
const claritySlider = document.getElementById('clarity');
const frequencySelector = document.getElementById('frequency-selector');
const applyFrequencyButton = document.getElementById('apply-frequency');

// Session control buttons
const startSessionButton = document.getElementById('start-session');
const stopSessionButton = document.getElementById('stop-session');
const saveSessionButton = document.getElementById('save-session');

// Tab buttons
const tabButtons = document.querySelectorAll('.tab-button');
let activeField = 'cognitive';

// Profile modal elements
const profileButton = document.getElementById('profile-button');
const profileModal = document.getElementById('profile-modal');
const closeModalButton = document.querySelector('.close');
const userNameInput = document.getElementById('user-name-input');
const savedSessionsList = document.getElementById('saved-sessions-list');
const saveProfileButton = document.getElementById('save-profile');
const loadProfileButton = document.getElementById('load-profile');

// Session state
let sessionActive = false;
let sessionStartTime = 0;
let sessionTimer = null;

// Initialize value displays
document.getElementById('coherence-value').textContent = coherenceSlider.value;
document.getElementById('presence-value').textContent = presenceSlider.value;
document.getElementById('intention-value').textContent = intentionSlider.value;
document.getElementById('joy-value').textContent = joySlider.value;
document.getElementById('peace-value').textContent = peaceSlider.value;
document.getElementById('love-value').textContent = loveSlider.value;
document.getElementById('clarity-value').textContent = claritySlider.value;

// Socket.IO event listeners
socket.on('connect', () => {
    console.log('Connected to server');
    sessionMessage.textContent = 'Connected to Cascade⚡𓂧φ∞ server';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    sessionMessage.textContent = 'Disconnected from server';
    
    // Update UI for disconnected state
    sessionStatus.textContent = 'Disconnected';
    startSessionButton.disabled = true;
    stopSessionButton.disabled = true;
    saveSessionButton.disabled = true;
});

socket.on('field_update', (data) => {
    // Update field visualization
    if (data.image) {
        fieldImage.src = 'data:image/png;base64,' + data.image;
    }
    
    // Update metrics
    systemCoherence.textContent = data.system_coherence.toFixed(4);
    fieldCoherence.textContent = data.field_coherence.toFixed(4);
    dominantEmotion.textContent = data.dominant_emotion;
    
    // Update subfield if it matches the active one
    if (data.subfield && data.subfield.name === activeField && data.subfield.image) {
        subfieldImage.src = 'data:image/png;base64,' + data.subfield.image;
    }
});

socket.on('session_update', (data) => {
    // Update session status
    sessionStatus.textContent = data.status;
    
    if (data.message) {
        sessionMessage.textContent = data.message;
    }
    
    // If session starts
    if (data.active && !sessionActive) {
        sessionActive = true;
        sessionStartTime = Date.now();
        startSessionTimer();
        
        // Update button states
        startSessionButton.disabled = true;
        stopSessionButton.disabled = false;
    }
    
    // If session ends
    if (!data.active && sessionActive) {
        sessionActive = false;
        stopSessionTimer();
        
        // Update button states
        startSessionButton.disabled = false;
        stopSessionButton.disabled = true;
        saveSessionButton.disabled = false;
    }
});

socket.on('user_update', (data) => {
    // Update user information
    document.getElementById('user-name').textContent = 'User: ' + data.name;
    userNameInput.value = data.name;
    
    // Update saved sessions list
    if (data.sessions && data.sessions.length > 0) {
        let sessionHtml = '';
        data.sessions.forEach(session => {
            sessionHtml += `<div class="saved-session">
                <div>${session.date} - ${session.duration}min</div>
                <button class="load-session-button" data-id="${session.id}">Load</button>
            </div>`;
        });
        savedSessionsList.innerHTML = sessionHtml;
        
        // Add event listeners to load session buttons
        document.querySelectorAll('.load-session-button').forEach(button => {
            button.addEventListener('click', () => {
                const sessionId = button.getAttribute('data-id');
                socket.emit('load_session', { id: sessionId });
                profileModal.style.display = 'none';
            });
        });
    } else {
        savedSessionsList.innerHTML = '<p>No saved sessions</p>';
    }
});

// UI event listeners
coherenceSlider.addEventListener('input', (e) => {
    document.getElementById('coherence-value').textContent = e.target.value;
    updateConsciousnessState();
});

presenceSlider.addEventListener('input', (e) => {
    document.getElementById('presence-value').textContent = e.target.value;
    updateConsciousnessState();
});

intentionSlider.addEventListener('input', (e) => {
    document.getElementById('intention-value').textContent = e.target.value;
    updateConsciousnessState();
});

joySlider.addEventListener('input', (e) => {
    document.getElementById('joy-value').textContent = e.target.value;
    updateConsciousnessState();
});

peaceSlider.addEventListener('input', (e) => {
    document.getElementById('peace-value').textContent = e.target.value;
    updateConsciousnessState();
});

loveSlider.addEventListener('input', (e) => {
    document.getElementById('love-value').textContent = e.target.value;
    updateConsciousnessState();
});

claritySlider.addEventListener('input', (e) => {
    document.getElementById('clarity-value').textContent = e.target.value;
    updateConsciousnessState();
});

applyFrequencyButton.addEventListener('click', () => {
    socket.emit('set_frequency', { frequency: frequencySelector.value });
    sessionMessage.textContent = `Applied ${frequencySelector.value} frequency`;
});

startSessionButton.addEventListener('click', () => {
    socket.emit('start_session');
});

stopSessionButton.addEventListener('click', () => {
    socket.emit('stop_session');
});

saveSessionButton.addEventListener('click', () => {
    socket.emit('save_session');
    saveSessionButton.disabled = true;
    sessionMessage.textContent = 'Session saved';
});

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Update active tab
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Set active field
        activeField = button.getAttribute('data-field');
        
        // Request subfield update
        socket.emit('get_subfield', { name: activeField });
    });
});

profileButton.addEventListener('click', () => {
    profileModal.style.display = 'block';
});

closeModalButton.addEventListener('click', () => {
    profileModal.style.display = 'none';
});

saveProfileButton.addEventListener('click', () => {
    socket.emit('save_profile', { name: userNameInput.value });
    profileModal.style.display = 'none';
});

loadProfileButton.addEventListener('click', () => {
    socket.emit('load_profile', { name: userNameInput.value });
    profileModal.style.display = 'none';
});

// Window event listener to close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === profileModal) {
        profileModal.style.display = 'none';
    }
});

// Helper functions
function updateConsciousnessState() {
    const state = {
        coherence: parseFloat(coherenceSlider.value),
        presence: parseFloat(presenceSlider.value),
        intention: parseFloat(intentionSlider.value),
        emotions: {
            joy: parseFloat(joySlider.value),
            peace: parseFloat(peaceSlider.value),
            love: parseFloat(loveSlider.value),
            clarity: parseFloat(claritySlider.value)
        }
    };
    
    socket.emit('update_state', state);
}

function startSessionTimer() {
    sessionTimer = setInterval(updateSessionTime, 1000);
    updateSessionTime();
}

function stopSessionTimer() {
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }
}

function updateSessionTime() {
    if (!sessionActive) return;
    
    const now = Date.now();
    const elapsed = now - sessionStartTime;
    
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    
    sessionTime.textContent = 
        String(hours).padStart(2, '0') + ':' +
        String(minutes).padStart(2, '0') + ':' +
        String(seconds).padStart(2, '0');
}

// Initial state update
updateConsciousnessState();""")
    
    # Create placeholder image
    placeholder_path = static_dir / "placeholder.png"
    if not placeholder_path.exists():
        # Create a simple placeholder image
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, "Quantum Field", 
                   horizontalalignment='center', verticalalignment='center', 
                   fontsize=18)
            plt.axis('off')
            plt.savefig(placeholder_path)
            plt.close()

class CascadeWebInterface:
    """Web interface for CascadeOS."""
    
    def __init__(self, host="0.0.0.0", port=5000):
        """Initialize the web interface."""
        self.host = host
        self.port = port
        self.system = None
        self.active = False
        self.visualization_thread = None
        self.session_start_time = 0
        self.session_data = []
        
        # Create field dimensions optimized for visualization
        self.field_dimensions = (34, 55)
        
        # Check dependencies
        if not WEB_SERVER_AVAILABLE:
            logger.error("Flask or Flask-SocketIO not installed. Web interface disabled.")
            return
        
        # Create required directories
        create_required_directories()
        
        # Initialize Cascade system
        self._initialize_system()
        
        # Set up Flask routes
        self._setup_routes()
        
        logger.info("CascadeOS Web Interface initialized")
    
    def _initialize_system(self):
        """Initialize the Cascade system."""
        # Create system
        self.system = CascadeSystem()
        self.system.initialize({
            "dimensions": self.field_dimensions,
            "frequency": "unity",
            "visualization": False  # We'll handle visualization ourselves
        })
        
        # Activate system
        self.system.activate()
        
        global cascade_system
        cascade_system = self.system
        
        logger.info("Cascade system initialized")
    
    def _setup_routes(self):
        """Set up Flask and Socket.IO routes."""
        if not WEB_SERVER_AVAILABLE:
            return
        
        # Flask routes
        @app.route('/')
        def index():
            return render_template('index.html')
        
        @app.route('/api/status')
        def get_status():
            if not self.system:
                return jsonify({"error": "System not initialized"}), 500
            
            status = {
                "active": self.active,
                "system_coherence": self.system.system_coherence,
                "field_coherence": self.system.primary_field.coherence,
                "consciousness_state": {
                    "coherence": self.system.interface.state.coherence,
                    "presence": self.system.interface.state.presence,
                    "intention": self.system.interface.state.intention,
                    "dominant_emotion": self.system.interface.state.dominant_emotion[0]
                }
            }
            
            return jsonify(status)
        
        # Socket.IO event handlers
        @socketio.on('connect')
        def handle_connect():
            logger.info("Client connected")
            self._send_initial_state()
        
        @socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected")
        
        @socketio.on('update_state')
        def handle_update_state(data):
            try:
                # Update consciousness state
                if not self.system:
                    return
                
                # Update coherence, presence, intention
                self.system.interface.state.coherence = data.get('coherence', 0.5)
                self.system.interface.state.presence = data.get('presence', 0.5)
                self.system.interface.state.intention = data.get('intention', 0.5)
                
                # Update emotions
                emotions = data.get('emotions', {})
                for emotion, value in emotions.items():
                    if emotion in self.system.interface.state.emotional_states:
                        self.system.interface.state.emotional_states[emotion] = value
                
                # Apply to field
                self.system.interface._apply_state_to_field()
                
                # Update system
                self.system.update()
                
                # Send update to client
                self._send_field_update()
                
            except Exception as e:
                logger.error(f"Error updating state: {e}")
        
        @socketio.on('set_frequency')
        def handle_set_frequency(data):
            try:
                # Set frequency
                frequency = data.get('frequency', 'unity')
                
                if not self.system:
                    return
                
                # Update frequency for primary field
                if frequency in SACRED_FREQUENCIES:
                    self.system.primary_field.frequency_name = frequency
                    self.system.primary_field.apply_phi_modulation(intensity=0.5)
                    
                    # Update system
                    self.system.update()
                    
                    # Send update to client
                    self._send_field_update()
                    
                    logger.info(f"Set frequency to {frequency}")
                else:
                    logger.warning(f"Unknown frequency: {frequency}")
                
            except Exception as e:
                logger.error(f"Error setting frequency: {e}")
        
        @socketio.on('start_session')
        def handle_start_session():
            try:
                # Start session
                self.start_session()
                
            except Exception as e:
                logger.error(f"Error starting session: {e}")
        
        @socketio.on('stop_session')
        def handle_stop_session():
            try:
                # Stop session
                self.stop_session()
                
            except Exception as e:
                logger.error(f"Error stopping session: {e}")
        
        @socketio.on('save_session')
        def handle_save_session():
            try:
                # Save session
                session_id = self.save_session()
                
                # Send update to client
                self._send_user_update()
                
                logger.info(f"Session saved with ID: {session_id}")
                
            except Exception as e:
                logger.error(f"Error saving session: {e}")
        
        @socketio.on('get_subfield')
        def handle_get_subfield(data):
            try:
                # Get subfield
                field_name = data.get('name', 'cognitive')
                
                if not self.system or field_name not in self.system.subfields:
                    return
                
                # Get subfield
                subfield = self.system.subfields[field_name]
                
                # Create visualization
                if MATPLOTLIB_AVAILABLE:
                    img_bytes = self._create_field_visualization(subfield.data)
                    img_b64 = base64.b64encode(img_bytes.getvalue()).decode('ascii')
                    
                    # Send to client
                    socketio.emit('field_update', {
                        'subfield': {
                            'name': field_name,
                            'image': img_b64
                        }
                    })
                
            except Exception as e:
                logger.error(f"Error getting subfield: {e}")
        
        @socketio.on('save_profile')
        def handle_save_profile(data):
            try:
                # Update user profile
                global current_user
                current_user["name"] = data.get('name', 'Guest')
                
                # Send update to client
                self._send_user_update()
                
                logger.info(f"Profile saved for user: {current_user['name']}")
                
            except Exception as e:
                logger.error(f"Error saving profile: {e}")
        
        @socketio.on('load_profile')
        def handle_load_profile(data):
            try:
                # Load user profile
                name = data.get('name', 'Guest')
                
                # In a real application, we would load from database
                # For demo, just set the name
                global current_user
                current_user["name"] = name
                
                # Send update to client
                self._send_user_update()
                
                logger.info(f"Profile loaded for user: {current_user['name']}")
                
            except Exception as e:
                logger.error(f"Error loading profile: {e}")
        
        @socketio.on('load_session')
        def handle_load_session(data):
            try:
                # Load session
                session_id = data.get('id')
                
                # In a real application, we would load from database
                # For demo, just log the request
                logger.info(f"Request to load session: {session_id}")
                
                # Send update to client
                socketio.emit('session_update', {
                    'status': 'Loading session...',
                    'message': f'Loading session {session_id}'
                })
                
            except Exception as e:
                logger.error(f"Error loading session: {e}")
    
    def _send_initial_state(self):
        """Send initial state to client."""
        if not self.system:
            return
        
        # Send field update
        self._send_field_update()
        
        # Send session update
        socketio.emit('session_update', {
            'status': 'Ready',
            'active': self.active,
            'message': 'System ready'
        })
        
        # Send user update
        self._send_user_update()
    
    def _send_field_update(self):
        """Send field update to client."""
        if not self.system:
            return
        
        try:
            # Get system state
            system_coherence = self.system.system_coherence
            field_coherence = self.system.primary_field.coherence
            dominant_emotion = self.system.interface.state.dominant_emotion[0]
            
            # Create field visualization
            data = {
                'system_coherence': system_coherence,
                'field_coherence': field_coherence,
                'dominant_emotion': dominant_emotion
            }
            
            # Add field image if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                img_bytes = self._create_field_visualization(self.system.primary_field.data)
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode('ascii')
                data['image'] = img_b64
            
            # Send update
            socketio.emit('field_update', data)
            
        except Exception as e:
            logger.error(f"Error sending field update: {e}")
    
    def _send_user_update(self):
        """Send user update to client."""
        try:
            # Get user data
            global current_user
            
            # Format user data for client
            user_data = {
                'name': current_user["name"],
                'sessions': []
            }
            
            # Add mock sessions for demo
            import datetime
            
            # Add current session if active
            if self.active and self.session_start_time > 0:
                duration = (time.time() - self.session_start_time) / 60
                user_data['sessions'].append({
                    'id': 'current',
                    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'duration': f'{duration:.1f}',
                    'coherence': self.system.system_coherence if self.system else 0.0
                })
            
            # Add saved sessions
            for i, session in enumerate(self.session_data):
                if 'end_time' in session:
                    session_date = datetime.datetime.fromtimestamp(session['start_time'])
                    duration = (session['end_time'] - session['start_time']) / 60
                    
                    user_data['sessions'].append({
                        'id': f'session_{i}',
                        'date': session_date.strftime('%Y-%m-%d %H:%M'),
                        'duration': f'{duration:.1f}',
                        'coherence': session.get('final_coherence', 0.0)
                    })
            
            # Send update
            socketio.emit('user_update', user_data)
            
        except Exception as e:
            logger.error(f"Error sending user update: {e}")
    
    def _create_field_visualization(self, field_data):
        """Create a visualization of the field and return as bytes."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Create figure
            plt.figure(figsize=(6, 6))
            
            # Create custom colormap using phi-based colors
            colors = [
                (0.0, 0.0, 0.4),                           # Deep blue
                (0.0, 0.3, 0.7),                           # Medium blue
                (0.2, 0.6, 0.9),                           # Light blue
                (0.4, 0.8, 1.0),                           # Sky blue
                (0.6, 0.9, 0.6),                           # Light green
                (0.9, 0.9, 0.0),                           # Yellow
                (1.0, 0.7, 0.0),                           # Orange
                (1.0, 0.4, 0.0),                           # Dark orange
                (0.8, 0.2, 0.0)                            # Red
            ]
            cmap = LinearSegmentedColormap.from_list('phi', colors)
            
            # Plot 2D field or slice of higher dimension
            if len(field_data.shape) > 2:
                # Show middle slice
                slice_idx = field_data.shape[0] // 2
                plot_data = field_data[slice_idx]
            else:
                plot_data = field_data
            
            # Create plot
            plt.imshow(plot_data, cmap=cmap, interpolation='bilinear')
            plt.axis('off')
            
            # Save to bytes
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
            img_bytes.seek(0)
            
            # Close figure to prevent memory leaks
            plt.close()
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error creating field visualization: {e}")
            return BytesIO()
    
    def start_session(self):
        """Start a Cascade session."""
        if self.active:
            logger.warning("Session already active")
            return False
        
        self.active = True
        self.session_start_time = time.time()
        self.session_data = []
        
        # Start visualization thread
        global update_active
        update_active = True
        
        # Send update to client
        socketio.emit('session_update', {
            'status': 'Active',
            'active': True,
            'message': 'Session started'
        })
        
        logger.info("Session started")
        return True
    
    def stop_session(self):
        """Stop the current session."""
        if not self.active:
            logger.warning("No active session")
            return False
        
        self.active = False
        end_time = time.time()
        
        # Stop visualization thread
        global update_active
        update_active = False
        
        # Calculate session stats
        duration = end_time - self.session_start_time
        minutes = duration / 60
        
        # Save session data
        session_record = {
            'start_time': self.session_start_time,
            'end_time': end_time,
            'duration': duration,
            'data_points': len(self.session_data),
            'final_coherence': self.system.system_coherence if self.system else 0.0
        }
        
        self.session_data = session_record
        
        # Send update to client
        socketio.emit('session_update', {
            'status': 'Inactive',
            'active': False,
            'message': f'Session ended ({minutes:.1f} minutes)'
        })
        
        logger.info(f"Session stopped. Duration: {minutes:.1f} minutes")
        return True
    
    def save_session(self):
        """Save the current session data."""
        # In a real application, would save to database
        # For demo, just return a mock ID
        session_id = str(int(time.time()))
        
        # Add to global session data
        global session_data
        session_data.append(self.session_data)
        
        logger.info(f"Session saved with ID: {session_id}")
        return session_id
    
    def update_visualization(self):
        """Update visualization in a loop."""
        global update_active
        
        while update_active:
            try:
                # Send updates to clients
                if self.system:
                    self._send_field_update()
                
                # Collect data point
                if self.active and self.system:
                    data_point = {
                        'timestamp': time.time(),
                        'system_coherence': self.system.system_coherence,
                        'field_coherence': self.system.primary_field.coherence,
                        'consciousness_state': {
                            'coherence': self.system.interface.state.coherence,
                            'presence': self.system.interface.state.presence,
                            'intention': self.system.interface.state.intention,
                            'dominant_emotion': self.system.interface.state.dominant_emotion
                        }
                    }
                    
                    # Add to session data
                    self.session_data.append(data_point)
            
            except Exception as e:
                logger.error(f"Error in visualization thread: {e}")
            
            # Sleep for update interval
            time.sleep(1.0)
    
    def start(self):
        """Start the web interface."""
        if not WEB_SERVER_AVAILABLE:
            logger.error("Required packages not installed. Cannot start web interface.")
            return False
        
        try:
            # Start visualization thread
            global update_active
            update_active = True
            self.visualization_thread = threading.Thread(target=self.update_visualization)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            
            # Start Flask server
            logger.info(f"Starting web interface on {self.host}:{self.port}")
            socketio.run(app, host=self.host, port=self.port)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            return False
    
    def stop(self):
        """Stop the web interface."""
        # Stop visualization thread
        global update_active
        update_active = False
        
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
        
        # Stop any active session
        if self.active:
            self.stop_session()
        
        logger.info("Web interface stopped")
        return True


def main():
    """Main function to run the web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port to bind to")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Cascade⚡𓂧φ∞ Web Interface")
    print("=" * 80)
    
    if not WEB_SERVER_AVAILABLE:
        print("\nRequired packages not installed. Please install Flask and Flask-SocketIO:")
        print("pip install flask flask_socketio")
        return 1
    
    # Create required directories and files
    create_required_directories()
    
    # Start web interface
    interface = CascadeWebInterface(host=args.host, port=args.port)
    interface.start()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())