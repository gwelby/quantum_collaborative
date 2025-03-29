#!/usr/bin/env python
"""
Quantum Field 3D Demo with CUDA Graphs Optimization

This script demonstrates the generation of 3D quantum fields using CUDA Graphs
for optimized performance, particularly for animations and time evolutions.

Usage:
    python quantum_field_3d_demo_with_graphs.py [options]

Options:
    --dimensions DEPTH HEIGHT WIDTH   Field dimensions (default: 64 64 64)
    --frequency FREQ_NAME             Sacred frequency name (default: love)
    --frames N                        Number of animation frames (default: 60)
    --output-dir DIR                  Directory to save output files (default: None)
    --benchmark                       Run benchmark comparing different methods
    --device ID                       CUDA device to use (default: 0)
    --all-devices                     Use all available CUDA devices
    --visualize                       Generate visualization (requires matplotlib & mayavi)
"""

import os
import sys
import time
import argparse
import numpy as np
from quantum_field.backends.cuda import CUDABackend
from quantum_field.constants import SACRED_FREQUENCIES

# Visualization imports - only imported when --visualize is used
visualization_available = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="3D Quantum Field Demo with CUDA Graphs")
    parser.add_argument("--dimensions", nargs=3, type=int, default=[64, 64, 64],
                      help="Field dimensions (depth, height, width)")
    parser.add_argument("--frequency", type=str, default="love",
                      help=f"Sacred frequency name: {', '.join(SACRED_FREQUENCIES.keys())}")
    parser.add_argument("--frames", type=int, default=60,
                      help="Number of animation frames")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save output files")
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark comparing different methods")
    parser.add_argument("--device", type=int, default=0,
                      help="CUDA device to use")
    parser.add_argument("--all-devices", action="store_true",
                      help="Use all available CUDA devices")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualization (requires matplotlib & mayavi)")
    
    return parser.parse_args()


def setup_visualization():
    """Import visualization libraries and setup environment."""
    global visualization_available
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        try:
            import mayavi.mlab as mlab
            has_mayavi = True
        except ImportError:
            has_mayavi = False
            print("Mayavi not found, will use matplotlib for visualization")
        
        visualization_available = True
        return has_mayavi
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")
        print("Install matplotlib and mayavi for visualization support")
        visualization_available = False
        return False


def visualize_field_slice(field, frame_num, output_dir, has_mayavi=False):
    """Visualize a 3D field slice or volume rendering."""
    if not visualization_available:
        return
    
    depth, height, width = field.shape
    mid_z = depth // 2
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if has_mayavi:
        import mayavi.mlab as mlab
        
        # Clear previous figure
        mlab.clf()
        
        # Volume rendering
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(field))
        
        # Add isosurfaces
        for iso_val in [-0.5, 0, 0.5]:
            mlab.contour3d(field, contours=[iso_val], opacity=0.3)
        
        # Save figure
        if output_dir:
            mlab.savefig(os.path.join(output_dir, f"frame_{frame_num:04d}.png"))
        
        # Close figure
        mlab.close()
    else:
        import matplotlib.pyplot as plt
        
        # Create a new figure
        fig = plt.figure(figsize=(10, 8))
        
        # Plot XY slice at middle Z
        plt.subplot(221)
        plt.imshow(field[mid_z, :, :], cmap='viridis')
        plt.title(f"XY Slice (Z={mid_z})")
        plt.colorbar()
        
        # Plot XZ slice at middle Y
        plt.subplot(222)
        plt.imshow(field[:, height//2, :], cmap='viridis')
        plt.title(f"XZ Slice (Y={height//2})")
        plt.colorbar()
        
        # Plot YZ slice at middle X
        plt.subplot(223)
        plt.imshow(field[:, :, width//2], cmap='viridis')
        plt.title(f"YZ Slice (X={width//2})")
        plt.colorbar()
        
        # 3D scatter plot of high-value points
        ax = fig.add_subplot(224, projection='3d')
        threshold = 0.7
        points = np.argwhere(np.abs(field) > threshold)
        if len(points) > 1000:  # Limit number of points for performance
            points = points[np.random.choice(len(points), 1000, replace=False)]
        
        if len(points) > 0:
            ax.scatter(points[:, 2], points[:, 1], points[:, 0], 
                      c=field[points[:, 0], points[:, 1], points[:, 2]], 
                      cmap='viridis', s=5)
        
        ax.set_title(f"High Value Points (|value| > {threshold})")
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"frame_{frame_num:04d}.png"))
        
        plt.close(fig)


def generate_animation_with_graph(backend, dimensions, frequency_name, num_frames, output_dir=None, visualize=False):
    """Generate an animation of a 3D quantum field using CUDA Graphs."""
    depth, height, width = dimensions
    
    print(f"Generating {num_frames} frames of a {depth}x{height}x{width} 3D quantum field using CUDA Graphs")
    
    # Create a CUDA Graph for the 3D field
    graph_name = "animation_graph"
    backend.create_cuda_graph(
        graph_name=graph_name,
        width=width,
        height=height,
        depth=depth,
        frequency_name=frequency_name
    )
    
    # Setup visualization if requested
    has_mayavi = False
    if visualize:
        has_mayavi = setup_visualization()
    
    frames = []
    start_time = time.time()
    
    # Generate each frame
    for i in range(num_frames):
        # Calculate time factor (0 to 2π)
        time_factor = i * (2 * np.pi / num_frames)
        
        # Execute the graph with the current time factor
        field = backend.execute_cuda_graph(graph_name, time_factor=time_factor)
        
        # Store the frame if requested
        if output_dir:
            frames.append(field)
        
        # Visualize if requested
        if visualize:
            visualize_field_slice(field, i, output_dir, has_mayavi)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0 or i == num_frames - 1:
            print(f"Generated frame {i + 1}/{num_frames}")
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    print(f"Animation generation complete in {total_time:.2f} seconds ({fps:.2f} FPS)")
    
    # Clean up
    backend.destroy_cuda_graph(graph_name)
    
    return frames, total_time


def generate_animation_standard(backend, dimensions, frequency_name, num_frames, output_dir=None, visualize=False):
    """Generate an animation of a 3D quantum field using standard method."""
    depth, height, width = dimensions
    
    print(f"Generating {num_frames} frames of a {depth}x{height}x{width} 3D quantum field using standard method")
    
    # Setup visualization if requested
    has_mayavi = False
    if visualize:
        has_mayavi = setup_visualization()
    
    frames = []
    start_time = time.time()
    
    # Generate each frame
    for i in range(num_frames):
        # Calculate time factor (0 to 2π)
        time_factor = i * (2 * np.pi / num_frames)
        
        # Generate the field
        field = backend.generate_3d_quantum_field(
            width=width,
            height=height,
            depth=depth,
            frequency_name=frequency_name,
            time_factor=time_factor
        )
        
        # Store the frame if requested
        if output_dir:
            frames.append(field)
        
        # Visualize if requested
        if visualize:
            visualize_field_slice(field, i, output_dir, has_mayavi)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0 or i == num_frames - 1:
            print(f"Generated frame {i + 1}/{num_frames}")
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    print(f"Animation generation complete in {total_time:.2f} seconds ({fps:.2f} FPS)")
    
    return frames, total_time


def run_benchmark(backend, dimensions, frequency_name, num_frames=30):
    """Run a benchmark comparing standard and CUDA Graph methods."""
    depth, height, width = dimensions
    
    print("\n=== BENCHMARK ===")
    print(f"Field dimensions: {depth}x{height}x{width}")
    print(f"Frequency: {frequency_name}")
    print(f"Frames: {num_frames}")
    
    # Run standard method
    print("\nRunning standard method...")
    _, standard_time = generate_animation_standard(
        backend, dimensions, frequency_name, num_frames
    )
    
    # Run CUDA Graph method
    print("\nRunning CUDA Graph method...")
    _, graph_time = generate_animation_with_graph(
        backend, dimensions, frequency_name, num_frames
    )
    
    # Calculate performance metrics
    speedup = standard_time / graph_time
    standard_fps = num_frames / standard_time
    graph_fps = num_frames / graph_time
    
    # Print results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Standard method: {standard_time:.2f} seconds ({standard_fps:.2f} FPS)")
    print(f"CUDA Graph method: {graph_time:.2f} seconds ({graph_fps:.2f} FPS)")
    print(f"Speedup: {speedup:.2f}x")
    
    # Return results
    return {
        "standard_time": standard_time,
        "graph_time": graph_time,
        "standard_fps": standard_fps,
        "graph_fps": graph_fps,
        "speedup": speedup
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    # Extract parameters
    depth, height, width = args.dimensions
    frequency_name = args.frequency
    num_frames = args.frames
    output_dir = args.output_dir
    benchmark = args.benchmark
    device_id = args.device
    use_all_devices = args.all_devices
    visualize = args.visualize
    
    # Initialize CUDA backend
    backend = CUDABackend()
    if not backend.initialize():
        print("Failed to initialize CUDA backend")
        return 1
    
    # Print GPU information
    devices = backend.devices
    if not devices:
        print("No CUDA devices available")
        return 1
    
    print("Available CUDA devices:")
    for i, device in enumerate(devices):
        print(f"[{i}] {device['name']} ({device['total_memory'] / 1024**3:.1f} GB)")
    
    # Set device if not using all devices
    if not use_all_devices:
        if device_id >= len(devices):
            print(f"Invalid device ID: {device_id}")
            return 1
        
        backend._set_device(device_id)
        print(f"Using device: [{device_id}] {devices[device_id]['name']}")
    else:
        print("Using all available devices")
    
    # Check frequency
    if frequency_name not in SACRED_FREQUENCIES:
        print(f"Invalid frequency name: {frequency_name}")
        print(f"Available frequencies: {', '.join(SACRED_FREQUENCIES.keys())}")
        return 1
    
    # Run benchmark if requested
    if benchmark:
        results = run_benchmark(backend, args.dimensions, frequency_name, min(num_frames, 30))
    
    # Generate animation
    print("\nGenerating animation with CUDA Graphs...")
    frames, _ = generate_animation_with_graph(
        backend, args.dimensions, frequency_name, num_frames, output_dir, visualize
    )
    
    # Calculate coherence for the first frame
    print("\nCalculating field coherence for the first frame...")
    coherence = backend.calculate_3d_field_coherence(frames[0] if frames else None)
    print(f"Field coherence: {coherence:.6f}")
    
    # Clean up
    backend.shutdown()
    print("\nDemo completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())