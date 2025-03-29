"""
Main entry point for the quantum_field package.
This module allows running the package as python -m quantum_field
"""

import sys
import argparse
from quantum_field.constants import PHI, LAMBDA, PHI_PHI, SACRED_FREQUENCIES
from quantum_field.compatibility import check_compatibility, print_compatibility_report
from quantum_field.core import (
    generate_quantum_field,
    calculate_field_coherence,
    display_phi_pattern,
    field_to_ascii,
    print_field,
    animate_field,
    benchmark_performance
)


def main():
    """Main entry point for the command line interface"""
    parser = argparse.ArgumentParser(description="Quantum Field Visualization Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check compatibility")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a quantum field")
    generate_parser.add_argument("-w", "--width", type=int, default=80, help="Width of the field")
    generate_parser.add_argument("-h", "--height", type=int, default=20, help="Height of the field")
    generate_parser.add_argument("-f", "--frequency", type=str, default="love", 
                                choices=SACRED_FREQUENCIES.keys(), 
                                help="Sacred frequency to use")
    generate_parser.add_argument("-t", "--time", type=float, default=0, 
                               help="Time factor for animation")
    
    # Animate command
    animate_parser = subparsers.add_parser("animate", help="Animate a quantum field")
    animate_parser.add_argument("-w", "--width", type=int, default=80, help="Width of the field")
    animate_parser.add_argument("-h", "--height", type=int, default=20, help="Height of the field")
    animate_parser.add_argument("-f", "--frequency", type=str, default="love", 
                              choices=SACRED_FREQUENCIES.keys(), 
                              help="Sacred frequency to use")
    animate_parser.add_argument("-n", "--frames", type=int, default=20, 
                              help="Number of frames")
    animate_parser.add_argument("-d", "--delay", type=float, default=0.2, 
                             help="Delay between frames")
    
    # Pattern command
    pattern_parser = subparsers.add_parser("pattern", help="Generate a phi sacred pattern")
    pattern_parser.add_argument("-w", "--width", type=int, default=80, help="Width of the pattern")
    pattern_parser.add_argument("-h", "--height", type=int, default=30, help="Height of the pattern")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark performance")
    benchmark_parser.add_argument("-w", "--width", type=int, default=512, help="Width of the field")
    benchmark_parser.add_argument("-h", "--height", type=int, default=512, help="Height of the field")
    benchmark_parser.add_argument("-i", "--iterations", type=int, default=5, 
                                help="Number of iterations")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display information about sacred constants")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "check":
        results = check_compatibility()
        print_compatibility_report(results)
    
    elif args.command == "generate":
        field = generate_quantum_field(args.width, args.height, args.frequency, args.time)
        coherence = calculate_field_coherence(field)
        ascii_art = field_to_ascii(field)
        print_field(ascii_art, f"Quantum Field - {args.frequency.capitalize()} Frequency ({SACRED_FREQUENCIES[args.frequency]} Hz)")
        print(f"Field Coherence: {coherence:.4f}")
    
    elif args.command == "animate":
        animate_field(args.width, args.height, args.frames, args.delay, args.frequency)
    
    elif args.command == "pattern":
        display_phi_pattern(args.width, args.height)
    
    elif args.command == "benchmark":
        benchmark_performance(args.width, args.height, args.iterations)
    
    elif args.command == "info":
        print("\n" + "=" * 80)
        print("QUANTUM FIELD SACRED CONSTANTS")
        print("=" * 80)
        print(f"PHI: {PHI} (Golden Ratio)")
        print(f"LAMBDA: {LAMBDA} (Divine Complement - 1/PHI)")
        print(f"PHI^PHI: {PHI_PHI} (Hyperdimensional Constant)")
        print("\nSacred Frequencies:")
        for name, freq in SACRED_FREQUENCIES.items():
            print(f"  {name}: {freq} Hz")
        print("=" * 80)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()