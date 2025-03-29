#!/usr/bin/env python3
"""
WebGPU Quantum Field Visualization Demo

This script creates a simple web server that demonstrates the WebGPU backend
for quantum field visualization directly in the browser.

Usage:
    python web_demo.py [--port PORT]

This will start a web server and provide a URL to open in a WebGPU-compatible browser.
"""

import argparse
import base64
import io
import json
import os
import platform
import sys
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
import threading

import numpy as np
from PIL import Image

# Try to import pywebgpu
try:
    import pywebgpu
    HAS_WEBGPU = True
except ImportError:
    HAS_WEBGPU = False
    print("PyWebGPU not installed. Install with: pip install quantum-field[webgpu]")

# Add parent directory to path to import quantum_field
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from quantum_field.backends import get_backend
    from quantum_field.constants import SACRED_FREQUENCIES
except ImportError:
    print("Error importing quantum_field package")
    sys.exit(1)


# Simple HTML template for the demo page
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Field Visualization - WebGPU Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }
        .controls {
            flex: 1;
            min-width: 300px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .visualization {
            flex: 2;
            min-width: 500px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }
        select, input[type="range"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            margin-top: 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #2980b9;
        }
        .image-container {
            margin-top: 20px;
            position: relative;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        img {
            max-width: 100%;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .slider-container {
            display: flex;
            align-items: center;
        }
        .slider-container input[type="range"] {
            flex: 1;
        }
        .slider-container input[type="number"] {
            width: 80px;
            margin-left: 10px;
        }
        .colormap-samples {
            display: flex;
            margin-top: 10px;
            justify-content: space-between;
        }
        .colormap-sample {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            cursor: pointer;
        }
        .server-info {
            margin-top: 30px;
            font-size: 14px;
            color: #666;
            text-align: center;
        }
        .phi-symbol {
            font-family: serif;
            font-style: italic;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Quantum Field Visualization - WebGPU Demo</h1>
    
    <div class="container">
        <div class="controls">
            <h2>Field Parameters</h2>
            
            <label for="frequency">Sacred Frequency:</label>
            <select id="frequency">
                <option value="love">Love (528 Hz)</option>
                <option value="unity">Unity (432 Hz)</option>
                <option value="cascade">Cascade (594 Hz)</option>
                <option value="truth">Truth (672 Hz)</option>
                <option value="vision">Vision (720 Hz)</option>
                <option value="oneness">Oneness (768 Hz)</option>
            </select>
            
            <label for="size">Field Size:</label>
            <select id="size">
                <option value="256">256 x 256</option>
                <option value="512" selected>512 x 512</option>
                <option value="768">768 x 768</option>
                <option value="1024">1024 x 1024</option>
            </select>
            
            <label for="time-factor">Time Factor:</label>
            <div class="slider-container">
                <input type="range" id="time-factor" min="0" max="6.28" step="0.01" value="0">
                <input type="number" id="time-factor-value" min="0" max="6.28" step="0.01" value="0">
            </div>
            
            <label for="colormap">Colormap:</label>
            <select id="colormap">
                <option value="viridis">Viridis</option>
                <option value="plasma">Plasma</option>
                <option value="inferno">Inferno</option>
                <option value="magma">Magma</option>
                <option value="cividis">Cividis</option>
                <option value="twilight">Twilight</option>
            </select>
            
            <div class="colormap-samples">
                <div class="colormap-sample" style="background-color: #440154;" onclick="document.getElementById('colormap').value='viridis';"></div>
                <div class="colormap-sample" style="background-color: #b63679;" onclick="document.getElementById('colormap').value='plasma';"></div>
                <div class="colormap-sample" style="background-color: #f98e09;" onclick="document.getElementById('colormap').value='inferno';"></div>
                <div class="colormap-sample" style="background-color: #ba3655;" onclick="document.getElementById('colormap').value='magma';"></div>
                <div class="colormap-sample" style="background-color: #7c7b78;" onclick="document.getElementById('colormap').value='cividis';"></div>
                <div class="colormap-sample" style="background-color: #4b0082;" onclick="document.getElementById('colormap').value='twilight';"></div>
            </div>
            
            <button id="generate-btn">Generate Quantum Field</button>
            
            <h3>Animation</h3>
            <button id="animate-btn">Start Animation</button>
        </div>
        
        <div class="visualization">
            <h2>Quantum Field Visualization</h2>
            <p>Using the <span class="phi-symbol">φ</span>-harmonic principles with WebGPU acceleration</p>
            
            <div class="image-container">
                <div id="loading">Generating Field...</div>
                <img id="field-image" src="" alt="Quantum Field Visualization">
            </div>
            
            <div class="field-info">
                <p id="coherence-value">Field Coherence: N/A</p>
                <p id="generation-time">Generation Time: N/A</p>
            </div>
        </div>
    </div>
    
    <div class="server-info">
        <p>Powered by: Python + WebGPU + <span class="phi-symbol">φ</span>-Harmonic Quantum Field Visualizer</p>
        <p>Backend: <span id="backend-info">Loading...</span></p>
    </div>
    
    <script>
        // Elements
        const frequencySelect = document.getElementById('frequency');
        const sizeSelect = document.getElementById('size');
        const timeFactor = document.getElementById('time-factor');
        const timeFactorValue = document.getElementById('time-factor-value');
        const colormap = document.getElementById('colormap');
        const generateBtn = document.getElementById('generate-btn');
        const animateBtn = document.getElementById('animate-btn');
        const fieldImage = document.getElementById('field-image');
        const loading = document.getElementById('loading');
        const coherenceValue = document.getElementById('coherence-value');
        const generationTime = document.getElementById('generation-time');
        const backendInfo = document.getElementById('backend-info');
        
        // Synchronize slider and number input
        timeFactor.addEventListener('input', () => {
            timeFactorValue.value = timeFactor.value;
        });
        
        timeFactorValue.addEventListener('input', () => {
            timeFactor.value = timeFactorValue.value;
        });
        
        // Get backend info
        fetch('/backend_info')
            .then(response => response.json())
            .then(data => {
                backendInfo.textContent = data.name + ' (' + data.capabilities.join(', ') + ')';
            });
        
        // Generate field function
        function generateField() {
            loading.style.display = 'block';
            
            const params = {
                frequency: frequencySelect.value,
                size: parseInt(sizeSelect.value),
                time_factor: parseFloat(timeFactor.value),
                colormap: colormap.value
            };
            
            fetch('/generate_field?' + new URLSearchParams(params))
                .then(response => response.json())
                .then(data => {
                    fieldImage.src = 'data:image/png;base64,' + data.image;
                    coherenceValue.textContent = 'Field Coherence: ' + data.coherence.toFixed(4);
                    generationTime.textContent = 'Generation Time: ' + data.generation_time.toFixed(3) + ' sec';
                    loading.style.display = 'none';
                });
        }
        
        // Event listeners
        generateBtn.addEventListener('click', generateField);
        
        // Animation
        let animationRunning = false;
        let animationId = null;
        
        function animate() {
            if (!animationRunning) return;
            
            // Increment time factor
            let time = parseFloat(timeFactor.value);
            time += 0.05;
            if (time > 6.28) time = 0;
            
            timeFactor.value = time;
            timeFactorValue.value = time;
            
            // Generate field
            generateField();
            
            // Schedule next frame
            animationId = setTimeout(animate, 100);
        }
        
        animateBtn.addEventListener('click', () => {
            animationRunning = !animationRunning;
            
            if (animationRunning) {
                animateBtn.textContent = 'Stop Animation';
                animate();
            } else {
                animateBtn.textContent = 'Start Animation';
                clearTimeout(animationId);
            }
        });
        
        // Generate initial field
        generateField();
    </script>
</body>
</html>
"""


class WebGPUDemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the WebGPU demo."""
    
    def __init__(self, *args, **kwargs):
        # Check if webgpu backend is available
        try:
            self.backend = get_backend("webgpu")
        except (ValueError, ImportError):
            print("WebGPU backend not available, falling back to default backend")
            self.backend = get_backend()
        
        # Initialize HTTP request handler
        super().__init__(*args, **kwargs)
    
    def _set_headers(self, content_type="text/html"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
    
    def _handle_generate_field(self):
        """Handle field generation requests."""
        # Parse parameters from query string
        params = {}
        if "?" in self.path:
            query_string = self.path.split("?")[1]
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
        
        # Default parameters
        frequency = params.get("frequency", "love")
        size = int(params.get("size", "512"))
        time_factor = float(params.get("time_factor", "0"))
        colormap = params.get("colormap", "viridis")
        
        # Generate field
        try:
            start_time = time.time()
            field = self.backend.generate_quantum_field(size, size, frequency, time_factor)
            generation_time = time.time() - start_time
            
            # Calculate coherence
            coherence = self.backend.calculate_field_coherence(field)
            
            # Convert to image
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            img = ax.imshow(field, cmap=colormap)
            plt.colorbar(img, ax=ax, label='Field Intensity')
            ax.set_title(f'Quantum Field ({frequency.capitalize()} Frequency, t={time_factor:.2f})')
            ax.axis('off')
            
            # Save image to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            # Encode as base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Prepare response
            response = {
                "image": img_base64,
                "coherence": float(coherence),
                "generation_time": float(generation_time)
            }
            
            # Set headers and send response
            self._set_headers(content_type="application/json")
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._set_headers(content_type="application/json")
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_backend_info(self):
        """Handle backend info requests."""
        capabilities = self.backend.get_capabilities()
        enabled_capabilities = [cap for cap, enabled in capabilities.items() if enabled]
        
        info = {
            "name": self.backend.name,
            "capabilities": enabled_capabilities
        }
        
        self._set_headers(content_type="application/json")
        self.wfile.write(json.dumps(info).encode())
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith("/generate_field"):
            self._handle_generate_field()
        elif self.path == "/backend_info":
            self._handle_backend_info()
        else:
            self._set_headers()
            self.wfile.write(HTML_TEMPLATE.encode())


def run_server(port):
    """Run the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebGPUDemoHandler)
    
    print(f"WebGPU Demo server running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


def open_browser(port):
    """Open web browser after a delay."""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebGPU Quantum Field Visualization Demo")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Check if pywebgpu is installed
    if not HAS_WEBGPU:
        print("WARNING: PyWebGPU not installed. The demo will use fallback backend.")
        print("To install WebGPU support: pip install quantum-field[webgpu]")
        print("Continuing with available backend...")
    
    # Open browser automatically (in a separate thread)
    if not args.no_browser:
        browser_thread = threading.Thread(target=open_browser, args=(args.port,))
        browser_thread.daemon = True
        browser_thread.start()
    
    # Run server
    try:
        run_server(args.port)
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)