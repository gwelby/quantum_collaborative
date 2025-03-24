# OBS Studio Setup Guide for Quantum Field Demo

This guide will help you set up OBS Studio for recording a professional demonstration video for the Quantum Field Visualization library.

## Installation

1. Download OBS Studio from the official website: [https://obsproject.com/](https://obsproject.com/)
2. Follow the installation instructions for your operating system (Windows, macOS, or Linux)
3. Run OBS Studio and go through the initial setup wizard

## System Configuration

### Video Settings

1. Go to **Settings** → **Video**
2. Set **Base (Canvas) Resolution** to 1920x1080
3. Set **Output (Scaled) Resolution** to 1920x1080
4. Set **FPS** to 30 (or 60 if your system can handle it)

### Output Settings

1. Go to **Settings** → **Output**
2. Set **Output Mode** to "Advanced"
3. Under the **Recording** tab:
   - Set **Recording Path** to your desired location
   - Set **Recording Format** to "mp4"
   - Set **Encoder** to hardware encoding if available (NVIDIA NVENC, AMD AMF, or Apple VT H264)
   - Set **Rate Control** to "CQP"
   - Set **CQ Level** to 18 (lower is better quality)
   - Set **Keyframe Interval** to 2 seconds
   - Set **Profile** to "high"
   - Set **Preset** to "Quality"

### Audio Settings

1. Go to **Settings** → **Audio**
2. Set **Sample Rate** to 48kHz
3. Set **Channels** to Stereo
4. Configure your microphone and desktop audio devices

## Scene Collection Setup

Create a new Scene Collection by going to **Scene Collection** → **New**.

### 1. Introduction Scene

1. Create a new scene called "Introduction"
2. Add a **Text** source with the title "Quantum Field Visualization"
3. Add a **Text** source with subtitle "A Python package for quantum field generation and visualization"
4. Optionally add an image source with the project logo

### 2. Terminal/Code Scene

1. Create a new scene called "Code Demo"
2. Add a **Window Capture** or **Display Capture** source that captures your terminal or code editor
3. Add a **Text** source for section titles
4. Optionally add a facecam using **Video Capture Device** if you want to appear in the video

### 3. Visualization Scene

1. Create a new scene called "Visualization"
2. Add a **Window Capture** source that captures matplotlib or other visualization windows
3. Add a **Text** source for explaining what's being shown

### 4. Side-by-Side Scene

1. Create a new scene called "Side by Side"
2. Add both code and visualization sources
3. Resize and position them to show both side-by-side
4. Add text sources for context

### 5. Command Line Scene

1. Create a new scene called "CLI Demo"
2. Add a **Window Capture** source for your terminal
3. Add a **Text** source for CLI command explanations

### 6. Conclusion Scene

1. Create a new scene called "Conclusion"
2. Add a **Text** source thanking viewers
3. Add a **Text** source with links to GitHub, documentation, etc.

## Scene Transitions

1. Go to the **Scene Transitions** section
2. Set **Transition** to "Fade" with 300ms duration
3. For more professional look, you can use "Stinger" transitions with custom animation

## Recording Workflow

1. Prepare your demo environment according to the demo script
2. Start with the Introduction scene
3. Switch to appropriate scenes as you progress through the script
4. Use hotkeys to switch between scenes (configure in **Settings** → **Hotkeys**)
5. Record the entire session in one take if possible
6. For mistakes, either start over or plan to edit in post-production

## Audio Setup

1. Consider using a quality microphone with pop filter
2. Record in a quiet environment to minimize background noise
3. Do a test recording to check audio levels
4. Add a **Noise Suppression** filter to your microphone source:
   - Right-click on your microphone source
   - Select **Filters**
   - Click the "+" icon under "Audio Filters"
   - Select "Noise Suppression"
   - Set method to "RNNoise" with default settings

## Additional Tips

1. **Practice**: Run through the entire demo a few times before recording
2. **Lighting**: Ensure good lighting if using a webcam
3. **Script**: Keep the demo script visible on another monitor or printed
4. **Timing**: Aim for 6-8 minutes total length
5. **Breaks**: If needed, plan natural break points for editing later
6. **Captions**: Consider adding captions in post-production
7. **Music**: Low background music can enhance professionalism (ensure it's royalty-free)

## Post-Production

1. Use video editing software (DaVinci Resolve, Adobe Premiere, or similar)
2. Trim any mistakes or long pauses
3. Add intro/outro music
4. Add lower-third texts for important information
5. Add captions if needed
6. Export at 1080p (or 4K if available) with high bitrate

## Integration with QuantumOBS

If using the QuantumOBS integration:

1. Install the Quantum Broadcasting System as described in `/mnt/d/greg/QuantumOBS/README.md`
2. Run the setup script: `python scripts/setup_quantum_obs.py`
3. Enable the WebSocket server in OBS:
   - Go to **Tools** → **WebSocket Server Settings**
   - Check "Enable WebSocket server"
   - Set Port to 4444
4. Start the quantum broadcasting system:
   - Run `python run_quantum_broadcaster.py`
   - The system will automatically create scenes for each dimension

## Final Checklist

- [ ] OBS installed and configured
- [ ] Scenes created and transitions set
- [ ] Audio tested and working properly
- [ ] Demo script prepared and practiced
- [ ] Recording storage has enough space
- [ ] System resources sufficient for smooth recording
- [ ] Background applications closed to prevent interruptions
- [ ] Notifications disabled during recording