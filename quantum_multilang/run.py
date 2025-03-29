#!/usr/bin/env python3
"""
Quantum Field Multi-Language Architecture Runner

This script serves as the entry point for the Quantum Field Multi-Language Architecture,
starting the Python controller and demonstrating the multi-language capabilities.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantum_multilang.log")
    ]
)

logger = logging.getLogger("quantum_multilang")

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

# Import the controller
from controller.src.main import QuantumMultiLangController
from controller.src.bridges import is_bridge_available

# Import DSL bridges if available
phiflow_available = is_bridge_available("phiflow")
gregscript_available = is_bridge_available("gregscript")

if phiflow_available:
    from controller.src.bridges.phiflow_bridge import PhiFlowBridge
    logger.info("φFlow DSL bridge available")

if gregscript_available:
    from controller.src.bridges.gregscript_bridge import GregScriptBridge
    logger.info("GregScript DSL bridge available")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantum Field Multi-Language Architecture Runner"
    )
    
    parser.add_argument(
        "--width", type=int, default=80,
        help="Width of the quantum field"
    )
    
    parser.add_argument(
        "--height", type=int, default=20,
        help="Height of the quantum field"
    )
    
    parser.add_argument(
        "--frequency", type=str, default="love",
        choices=["love", "unity", "cascade", "truth", "vision", "oneness"],
        help="Sacred frequency to use"
    )
    
    parser.add_argument(
        "--animate", action="store_true",
        help="Animate the quantum field"
    )
    
    parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of animation frames"
    )
    
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Delay between animation frames in seconds"
    )
    
    parser.add_argument(
        "--consciousness", type=float, default=1.618033988749895,
        help="Consciousness level (default is PHI)"
    )
    
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--phiflow", help="Use φFlow DSL with the specified file",
        metavar="FILE"
    )
    
    parser.add_argument(
        "--gregscript", help="Use GregScript DSL with the specified file",
        metavar="FILE"
    )
    
    parser.add_argument(
        "--state-transition", action="store_true",
        help="Run φFlow state transitions on the field"
    )
    
    parser.add_argument(
        "--pattern-recognition", action="store_true",
        help="Run GregScript pattern recognition on the field"
    )
    
    return parser.parse_args()

def animate_field(controller, width, height, frames=10, delay=0.2, frequency_name="love"):
    """Animate a quantum field visualization."""
    for i in range(frames):
        # Generate a new field with a time factor
        field = controller.generate_quantum_field(width, height, frequency_name, i * 0.2)
        
        # Visualize the field
        visualization = controller.visualize_field(field)
        
        # Clear screen (might not work in all terminals/environments)
        print("\033c", end="")
        
        # Print the field
        print("\n" + "=" * 80)
        print(f"QUANTUM FIELD VISUALIZATION (Multi-Language)")
        print(f"Frequency: {frequency_name}, Frame: {i+1}/{frames}")
        print("=" * 80)
        
        for row in visualization:
            print(row)
        
        print("=" * 80)
        
        # Calculate and print coherence
        coherence = controller.calculate_field_coherence(field)
        print(f"Field coherence: {coherence:.6f}")
        
        # Wait before the next frame
        time.sleep(delay)

def run_benchmark(controller, iterations=5):
    """Run a performance benchmark across languages."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    field_sizes = [
        (50, 20, "Small"),    # Small field
        (200, 100, "Medium"), # Medium field
        (500, 300, "Large")   # Large field
    ]
    
    for width, height, size_name in field_sizes:
        print(f"\nBenchmarking {size_name} Field ({width}x{height}):")
        print("-" * 40)
        
        # Check language status to see what's available
        status = controller.check_languages_status()
        available_languages = [lang for lang, state in status.items() if state == "active"]
        
        timings = {}
        
        for language in available_languages:
            # Skip languages that don't have generate_quantum_field
            if language not in ["python", "rust", "cpp", "julia"]:
                continue
            
            # Run benchmark
            start_time = time.time()
            
            for i in range(iterations):
                if language == "python":
                    field = controller._generate_quantum_field_python(width, height, 'love', i*0.1)
                elif language == "rust" and hasattr(controller.rust_bridge, "generate_quantum_field"):
                    field = controller.rust_bridge.generate_quantum_field(width, height, 'love', i*0.1)
                elif language == "cpp" and hasattr(controller.cpp_bridge, "generate_quantum_field"):
                    field = controller.cpp_bridge.generate_quantum_field(width, height, 'love', i*0.1)
                elif language == "julia" and hasattr(controller.julia_bridge, "generate_quantum_field"):
                    field = controller.julia_bridge.generate_quantum_field(width, height, 'love', i*0.1)
                else:
                    continue
                
                # Calculate coherence
                if language == "python":
                    coherence = controller._calculate_field_coherence_python(field)
                elif language == "rust" and hasattr(controller.rust_bridge, "calculate_field_coherence"):
                    coherence = controller.rust_bridge.calculate_field_coherence(field)
                elif language == "cpp" and hasattr(controller.cpp_bridge, "calculate_field_coherence"):
                    coherence = controller.cpp_bridge.calculate_field_coherence(field)
                elif language == "julia" and hasattr(controller.julia_bridge, "calculate_field_coherence"):
                    coherence = controller.julia_bridge.calculate_field_coherence(field)
                else:
                    coherence = controller._calculate_field_coherence_python(field)
            
            # Calculate performance
            elapsed = time.time() - start_time
            timings[language] = elapsed
            
            print(f"{language}: {elapsed:.4f} seconds ({iterations} iterations)")
            print(f"  Average: {elapsed/iterations:.4f} seconds per iteration")
        
        # Calculate speedups relative to Python
        if "python" in timings and timings["python"] > 0:
            python_time = timings["python"]
            
            print("\nSpeedups relative to Python:")
            for language, elapsed in timings.items():
                if language != "python":
                    speedup = python_time / elapsed
                    print(f"{language}: {speedup:.2f}x")
        
        print()

def run_phiflow_transitions(field, phiflow_code=None):
    """Run φFlow state transitions on a field."""
    if not phiflow_available:
        print("φFlow DSL not available. Please install it first.")
        return field, 0.0, "none"
    
    # Create φFlow bridge
    bridge = PhiFlowBridge()
    
    # Load φFlow code if provided
    if phiflow_code:
        bridge.load_phiflow_code(phiflow_code)
    else:
        # Use default state machine
        default_code = """
        # Default φFlow State Machine
        
        state initial
            frequency = love
            coherence >= 0.0
            compression = 1.0
        
        state harmonized
            frequency = unity
            coherence >= 0.7
            compression = 1.2
        
        state expanded
            frequency = truth
            coherence >= 0.8
            compression = 0.8
        
        # Transitions
        transition initial -> harmonized
            when coherence >= 0.5 then
            harmonize by φ
            blend by 2.0
        
        transition harmonized -> expanded
            when coherence >= 0.75 then
            expand by 1.5
            amplify by 1.2
        """
        bridge.load_phiflow_code(default_code)
    
    # Run auto transition
    new_field, coherence, transition = bridge.run_auto_transition(field)
    
    return new_field, coherence, transition

def run_gregscript_recognition(field, gregscript_code=None):
    """Run GregScript pattern recognition on a field."""
    if not gregscript_available:
        print("GregScript DSL not available. Please install it first.")
        return {}, 0.0, None
    
    # Create GregScript bridge
    bridge = GregScriptBridge()
    
    # Load GregScript code if provided
    if gregscript_code:
        bridge.load_gregscript_code(gregscript_code)
    else:
        # Use default patterns
        default_code = """
        // Default GregScript Patterns
        
        rhythm phi_pulse sequence=[1.0, 0.618, 0.382, 0.618] tempo=1.0
        rhythm golden_wave sequence=[1.0, 0.809, 0.618, 0.382, 0.236, 0.382, 0.618, 0.809] tempo=φ
        
        harmony love_harmony frequency=love overtones=[1.0, 0.618, 0.382, 0.236] phase=0.0
        harmony unity_harmony frequency=unity overtones=[1.0, 0.618, 0.382, 0.236] phase=0.5
        
        pattern coherent_field {
            use phi_pulse weight=0.618
            use love_harmony weight=1.0
        }
        
        pattern expanding_field {
            use golden_wave weight=1.0
            use unity_harmony weight=0.618
        }
        """
        bridge.load_gregscript_code(default_code)
    
    # Process field message
    pattern_results = bridge.process_field_message(field)
    
    # Extract best match
    best_pattern = None
    best_score = 0.0
    
    if "best_known_pattern" in pattern_results:
        best_pattern = pattern_results["best_known_pattern"]
        best_score = pattern_results["best_known_score"]
    elif "top_pattern" in pattern_results:
        best_pattern = pattern_results["top_pattern"]["name"]
        best_score = pattern_results["top_pattern"]["score"]
    
    return pattern_results, best_score, best_pattern

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting Quantum Multi-Language Architecture Runner")
    
    # Initialize the controller
    controller = QuantumMultiLangController()
    
    # Set consciousness level
    controller.set_consciousness_level(args.consciousness)
    
    # Check language status
    status = controller.check_languages_status()
    active_languages = ", ".join(lang for lang, state in status.items() if state == "active")
    logger.info(f"Active languages: {active_languages}")
    
    # Load DSL code if specified
    phiflow_code = None
    gregscript_code = None
    
    if args.phiflow:
        try:
            with open(args.phiflow, 'r') as f:
                phiflow_code = f.read()
            logger.info(f"Loaded φFlow code from {args.phiflow}")
        except Exception as e:
            logger.error(f"Error loading φFlow code: {e}")
    
    if args.gregscript:
        try:
            with open(args.gregscript, 'r') as f:
                gregscript_code = f.read()
            logger.info(f"Loaded GregScript code from {args.gregscript}")
        except Exception as e:
            logger.error(f"Error loading GregScript code: {e}")
    
    if args.benchmark:
        # Run benchmarks
        run_benchmark(controller)
    elif args.animate:
        # Animate field
        animate_field(controller, args.width, args.height, args.frames, args.delay, args.frequency)
    else:
        # Generate a single field
        field = controller.generate_quantum_field(args.width, args.height, args.frequency)
        
        # Calculate initial coherence
        coherence = controller.calculate_field_coherence(field)
        
        # Apply φFlow state transitions if requested
        if args.state_transition or args.phiflow:
            print("\n" + "=" * 80)
            print("φFLOW STATE TRANSITIONS")
            print("=" * 80)
            
            field, coherence, transition = run_phiflow_transitions(field, phiflow_code)
            print(f"Applied transition: {transition}")
            print(f"New coherence: {coherence:.6f}")
        
        # Run GregScript pattern recognition if requested
        if args.pattern_recognition or args.gregscript:
            print("\n" + "=" * 80)
            print("GREGSCRIPT PATTERN RECOGNITION")
            print("=" * 80)
            
            pattern_results, best_score, best_pattern = run_gregscript_recognition(field, gregscript_code)
            
            if best_pattern:
                print(f"Best matching pattern: {best_pattern} (score: {best_score:.4f})")
            else:
                print("No matching patterns found")
            
            if "rhythms" in pattern_results and pattern_results["rhythms"]:
                top_rhythm = pattern_results["rhythms"][0]
                print(f"Top rhythm: {top_rhythm['name']} (score: {top_rhythm['score']:.4f})")
            
            if "harmonies" in pattern_results and pattern_results["harmonies"]:
                top_harmony = pattern_results["harmonies"][0]
                print(f"Top harmony: {top_harmony['name']} (score: {top_harmony['score']:.4f})")
        
        # Visualize the field
        visualization = controller.visualize_field(field)
        
        # Print the visualization
        print("\n" + "=" * 80)
        print(f"QUANTUM FIELD VISUALIZATION (Multi-Language)")
        print(f"Frequency: {args.frequency}")
        print("=" * 80)
        
        for row in visualization:
            print(row)
        
        print("=" * 80)
        print(f"Field coherence: {coherence:.6f}")
        print(f"System coherence: {status['system_coherence']:.6f}")
        print(f"Active languages: {active_languages}")
        print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())