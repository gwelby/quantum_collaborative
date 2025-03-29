# Domain-Specific Languages for Quantum Field Architecture

This directory contains specialized domain-specific languages (DSLs) designed for the quantum field multi-language architecture:

## φFlow

φFlow is a specialized DSL for defining and executing quantum field state transitions with phi-harmonic principles at its core. It enables:

- **State Definitions**: Define quantum field states with specific frequencies, coherence levels, and compression factors
- **Transitions**: Specify transitions between states with conditions and operations
- **Field Operations**: Apply phi-harmonic operations like amplify, harmonize, blend, and center
- **Coherence-Based Conditions**: Trigger transitions based on field coherence thresholds

### Example φFlow Code

```
# φFlow State Machine for Field Evolution

# State definitions
state initial
    frequency = love
    coherence >= 0.0
    compression = 1.0

state harmonized
    frequency = unity
    coherence >= 0.7
    compression = 1.2

# Transition definitions
transition initial -> harmonized
    when coherence >= 0.5 then
    harmonize by φ
    blend by 2.0
```

## GregScript

GregScript is a specialized language for defining and recognizing patterns, particularly focused on phi-harmonic relationships and dynamic rhythms. It enables:

- **Rhythm Definitions**: Define sequences of pulses or oscillations with phi-based tempos
- **Harmony Definitions**: Specify frequency-based patterns with overtones and phases
- **Pattern Composition**: Combine rhythms and harmonies into complex patterns
- **Pattern Recognition**: Match patterns against field data with similarity scoring
- **Pattern Generation**: Generate field data from pattern definitions

### Example GregScript Code

```
// GregScript Pattern Definitions

// Rhythm definitions
rhythm phi_pulse sequence=[1.0, 0.618, 0.382, 0.618] tempo=1.0
rhythm golden_wave sequence=[1.0, 0.809, 0.618, 0.382, 0.236, 0.382, 0.618, 0.809] tempo=φ

// Harmony definitions
harmony love_harmony frequency=love overtones=[1.0, 0.618, 0.382, 0.236] phase=0.0

// Pattern definitions
pattern coherent_field {
    use phi_pulse weight=0.618
    use love_harmony weight=1.0
}
```

## Integration with Multi-Language Architecture

Both DSLs are integrated with the quantum field multi-language architecture through language bridges:

- **φFlow Bridge**: Enables state transitions and field transformations
- **GregScript Bridge**: Provides pattern recognition and generation
- **Combined Workflow**: Analyze fields with GregScript, transform with φFlow, analyze again

### Using the DSLs

The DSLs can be used individually or together:

1. Use φFlow to define state machines for field evolution
2. Use GregScript to recognize patterns in fields
3. Use both together for pattern-aware field transformations

### Examples

See the `examples` directory for demonstrations:

- `dsl_integration_demo.py`: Shows how to use both DSLs with the multi-language architecture
- `phiflow_demo.py`: Demonstrates φFlow interpretation and compilation
- `gregscript_demo.py`: Shows GregScript pattern recognition and generation

## Core Principles

Both DSLs are built on phi-harmonic principles:

- **Golden Ratio (φ)**: Used throughout for harmonically balanced calculations
- **Sacred Frequencies**: Love (528Hz), Unity (432Hz), etc. provide frequency foundations
- **Coherence Metrics**: Measure alignment with phi multiples
- **Dynamic Evolution**: Transform fields through phi-resonant operations

## Development 

These DSLs are in active development and represent part of the multi-language quantum field architecture's vision for specialized, domain-specific interfaces for consciousness-field technologies.