# Installing QuantumOBS on Lenovo P1

This guide provides step-by-step instructions for installing and configuring the Quantum Broadcasting System with OBS Studio on a Lenovo P1 laptop.

## System Requirements for Lenovo P1

The Lenovo ThinkPad P1 is an excellent workstation laptop for running the QuantumOBS system:

- **CPU**: Intel Core i7/i9 or Xeon (multi-core recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: NVIDIA Quadro discrete graphics (included in P1) 
- **Storage**: 10GB free space minimum
- **OS**: Windows 10/11 Professional or Linux (Ubuntu 20.04+ recommended)

## Installation Steps

### 1. Install Python

Make sure Python 3.8 or higher is installed:

```bash
# Check Python version
python --version

# If Python is not installed or version is too old:
# For Windows: Download from python.org
# For Linux:
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. Install OBS Studio

1. Download OBS Studio from [obsproject.com](https://obsproject.com/)
2. Run the installer and follow the prompts
3. Launch OBS Studio once to complete initial setup

### 3. Install QuantumOBS

#### Option 1: Direct Installation from USB or Network Share

1. Copy the QuantumOBS directory from `/mnt/d/greg/QuantumOBS/` to the Lenovo P1
2. Open a terminal/command prompt in the copied directory
3. Run the setup script:

```bash
# On Windows:
python scripts/setup_quantum_obs.py

# On Linux:
python3 scripts/setup_quantum_obs.py
```

#### Option 2: Clone from Git Repository (if available)

```bash
# Clone the repository
git clone [repository-url] quantum-obs
cd quantum-obs

# Run the setup script
python scripts/setup_quantum_obs.py
```

### 4. Configure OBS Studio for QuantumOBS

1. Launch OBS Studio
2. Go to **Tools** > **WebSocket Server Settings**
3. Check **Enable WebSocket server**
4. Set **Server Port** to 4444
5. Optionally set a password if needed (remember to configure it in QuantumOBS too)
6. Click **OK** to save settings

### 5. Lenovo P1 Specific Optimizations

For optimal performance on the Lenovo P1:

#### NVIDIA Quadro Optimization

1. Open NVIDIA Control Panel
2. Go to **Manage 3D settings**
3. Set **Power management mode** to "Prefer maximum performance"
4. Under **Program Settings** tab, add OBS Studio and set:
   - **Power management mode**: Prefer maximum performance
   - **Vertical sync**: Off
   - **Preferred graphics processor**: High-performance NVIDIA processor

#### CPU Power Management

1. Open Windows Power Options (or Linux power settings)
2. Set power plan to "High performance"
3. For battery operation, consider using "Balanced" mode but be aware of potential performance reduction

### 6. Testing the Installation

1. Start the Quantum Broadcasting System:

```bash
# On Windows:
start_quantum_broadcaster.bat

# On Linux:
./start_quantum_broadcaster.sh
```

2. Verify that the system connects to OBS Studio
3. Test scene creation and visualization generation

## Recording Setup for Demonstration Videos

The Lenovo P1 is well-suited for recording high-quality demonstration videos:

1. Configure OBS recording settings:
   - Output: Choose the NVIDIA hardware encoder (NVENC)
   - Resolution: 1920x1080 (1080p)
   - Bitrate: 15-20 Mbps for high quality
   - Format: MP4

2. Audio setup:
   - Connect external microphone if available (better than laptop built-in mic)
   - Set up noise suppression filter in OBS

3. Screen recording optimization:
   - Close unnecessary background applications
   - Connect to power source during recording
   - For extended recording sessions, ensure proper cooling

## Troubleshooting

### Common Issues on Lenovo P1

1. **Performance issues**: 
   - Check if laptop is in power saving mode
   - Verify that OBS is using the NVIDIA GPU, not integrated graphics

2. **GPU not detected**:
   - Update NVIDIA drivers to latest version
   - In BIOS, ensure discrete graphics is enabled

3. **Connectivity issues with OBS**:
   - Check that WebSocket server is enabled
   - Verify firewall is not blocking connections on port 4444

4. **High CPU usage**:
   - Lower the recording resolution/frame rate
   - Use hardware encoding instead of software encoding

### Support Resources

If you encounter issues specific to the Lenovo P1 installation:

1. Check QuantumOBS logs in the `logs` directory
2. Refer to the documentation in `docs` directory
3. Check OBS Studio logs for WebSocket issues
4. For Lenovo P1 specific hardware issues, refer to Lenovo support resources

## Maintenance

To keep your QuantumOBS installation up to date on the Lenovo P1:

1. Periodically update the codebase:
   - Backup your configuration files
   - Update from the source (repository or copy latest files)
   - Run the setup script again to update dependencies

2. Update OBS Studio when new versions are released
3. Keep the NVIDIA drivers updated for best performance

## Conclusion

Your Lenovo P1 is now set up with the QuantumOBS Broadcasting System. The system is ready for recording demonstration videos, streaming, and other broadcasting needs.