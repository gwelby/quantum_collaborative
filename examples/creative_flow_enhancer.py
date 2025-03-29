#!/usr/bin/env python3
"""
Creative Flow Enhancer - A practical application of CascadeOS for enhancing creative flow states

This application demonstrates how to use CascadeOS to support creative work
by establishing a phi-harmonic field that helps induce and maintain flow states.
It includes pattern generation, creative prompt synthesis, and flow state tracking.
"""

import os
import sys
import time
import random
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
    TeamsOfTeamsCollective,
    PHI, LAMBDA, PHI_PHI,
    SACRED_FREQUENCIES
)

class CreativeFlowEnhancer:
    """Creative flow state enhancement application using CascadeOS."""
    
    def __init__(self, creative_domain="general", field_dimensions=(55, 89, 34)):
        """Initialize the creative flow enhancer."""
        self.creative_domain = creative_domain
        self.field_dimensions = field_dimensions
        
        # Initialize Teams of Teams for comprehensive creative support
        self.collective = TeamsOfTeamsCollective(field_dimensions)
        self.collective.activate()
        
        # Creative state variables
        self.flow_state = 0.0
        self.divergent_thinking = 0.5
        self.convergent_thinking = 0.5
        self.inspiration_level = 0.0
        self.pattern_emergence = 0.0
        self.creative_output = []
        
        # Domain-specific settings
        self.initialize_domain_settings()
        
        # Pattern categories for prompts
        self.pattern_categories = {
            "visual": ["spiral", "wave", "fractal", "radial", "grid", "lattice", "toroidal", "branching", 
                     "nested", "symmetrical", "chaotic", "emergent"],
            "concept": ["contrast", "harmony", "rhythm", "balance", "tension", "flow", "transformation", 
                       "integration", "expansion", "connection", "boundary", "transcendence"],
            "emotion": ["joy", "wonder", "curiosity", "anticipation", "serenity", "excitement", 
                       "satisfaction", "awe", "appreciation", "gratitude", "love", "exhilaration"],
            "technique": ["juxtaposition", "abstraction", "deconstruction", "synthesis", "recursion", 
                         "layering", "inversion", "association", "translation", "amplification"]
        }
        
        # Initialize consciousness state with creative focus
        self.initialize_creative_state()
        
        print(f"Creative Flow Enhancer initialized for {creative_domain}")
        print(f"Field dimensions: {field_dimensions}")
        print(f"Initial flow state: {self.flow_state:.2f}")
    
    def initialize_domain_settings(self):
        """Configure settings based on creative domain."""
        domains = {
            "visual_art": {
                "frequency": "vision",
                "prompt_categories": ["visual", "emotion", "concept"],
                "divergent_default": 0.7,
                "pattern_weight": 0.8
            },
            "music": {
                "frequency": "love",
                "prompt_categories": ["emotion", "rhythm", "harmony"],
                "divergent_default": 0.6,
                "pattern_weight": 0.7
            },
            "writing": {
                "frequency": "truth",
                "prompt_categories": ["concept", "emotion", "narrative"],
                "divergent_default": 0.65,
                "pattern_weight": 0.6
            },
            "problem_solving": {
                "frequency": "cascade",
                "prompt_categories": ["concept", "technique", "structure"],
                "divergent_default": 0.5,
                "pattern_weight": 0.5
            },
            "general": {
                "frequency": "unity",
                "prompt_categories": ["visual", "concept", "emotion", "technique"],
                "divergent_default": 0.6,
                "pattern_weight": 0.6
            }
        }
        
        # Use default if domain not found
        domain_config = domains.get(self.creative_domain, domains["general"])
        
        self.frequency = domain_config["frequency"]
        self.prompt_categories = domain_config["prompt_categories"]
        self.divergent_thinking = domain_config["divergent_default"]
        self.convergent_thinking = 1.0 - self.divergent_thinking
        self.pattern_weight = domain_config["pattern_weight"]
    
    def initialize_creative_state(self):
        """Initialize consciousness state optimized for creativity."""
        state = ConsciousnessState()
        
        # Set creative-oriented emotional states
        state.emotional_states = {
            "joy": 0.6,
            "peace": 0.4,
            "love": 0.5,
            "gratitude": 0.3,
            "clarity": 0.5,
            "focus": 0.6,
            "openness": 0.8,
            "harmony": 0.5,
        }
        
        # Set moderately high intention for creative flow
        state.intention = 0.7
        state.presence = 0.6
        state.coherence = 0.6
        
        # Apply state to collective
        self.collective.process(state)
        
        # Store initial state
        self.initial_state = state
    
    def update_creative_state(self, focus_intensity=None, divergent_ratio=None, 
                             emotional_state=None, intent_level=None):
        """Update creative state with new parameters."""
        state = ConsciousnessState()
        
        # Copy from initial state
        state.emotional_states = dict(self.initial_state.emotional_states)
        state.intention = self.initial_state.intention
        state.presence = self.initial_state.presence
        state.coherence = self.initial_state.coherence
        
        # Apply updates if provided
        if focus_intensity is not None:
            state.emotional_states["focus"] = max(0.0, min(1.0, focus_intensity))
            
        if divergent_ratio is not None:
            self.divergent_thinking = max(0.0, min(1.0, divergent_ratio))
            self.convergent_thinking = 1.0 - self.divergent_thinking
            
            # Adjust openness vs clarity based on divergent/convergent ratio
            state.emotional_states["openness"] = 0.5 + self.divergent_thinking * 0.4
            state.emotional_states["clarity"] = 0.5 + self.convergent_thinking * 0.4
            
        if emotional_state is not None:
            for emotion, intensity in emotional_state.items():
                if emotion in state.emotional_states:
                    state.emotional_states[emotion] = max(0.0, min(1.0, intensity))
                    
        if intent_level is not None:
            state.intention = max(0.0, min(1.0, intent_level))
        
        # Apply updated state to collective
        coherence = self.collective.process(state)
        
        # Update flow state based on coherence and other factors
        self.flow_state = self._calculate_flow_state(coherence, state)
        
        return self.flow_state
    
    def _calculate_flow_state(self, coherence, state):
        """Calculate flow state level based on various factors."""
        # Key factors for flow:
        # 1. Coherence between difficulty and skill (represented by field coherence)
        # 2. Clear goals (intention)
        # 3. Immediate feedback (field resonance)
        # 4. Concentration (focus emotional state)
        # 5. Control (represented by system stability)
        
        focus = state.emotional_states.get("focus", 0.5)
        clarity = state.emotional_states.get("clarity", 0.5)
        openness = state.emotional_states.get("openness", 0.5)
        joy = state.emotional_states.get("joy", 0.5)
        
        # Calculate temporal coherence (stability over time)
        status = self.collective.get_status()
        team_coherences = [team["coherence"] for team in status["teams"].values()]
        stability = 1.0 - np.std(team_coherences) if team_coherences else 0.5
        
        # Calculate alignment between divergent/convergent thinking
        phase_alignment = 1.0 - abs(self.divergent_thinking - 0.5) * 2.0
        
        # Phi-weighted combination
        flow_state = (
            coherence * PHI_SQUARED +
            state.intention * PHI +
            focus * LAMBDA * PHI_SQUARED +
            stability * LAMBDA * PHI +
            phase_alignment * LAMBDA_SQUARED * PHI
        ) / (PHI_SQUARED + PHI + LAMBDA * PHI_SQUARED + LAMBDA * PHI + LAMBDA_SQUARED * PHI)
        
        # Update inspiration level based on flow state
        self.inspiration_level = flow_state * openness * joy
        
        # Update pattern emergence
        field_coherence = status["primary_field_coherence"]
        self.pattern_emergence = (field_coherence * PHI + coherence) / (PHI + 1.0)
        
        return flow_state
    
    def generate_creative_prompt(self, domain_specific=True):
        """Generate a creative prompt based on quantum field patterns."""
        # Get field status
        status = self.collective.get_status()
        
        # Extract coherence and team values to influence prompt generation
        primary_coherence = status["primary_field_coherence"]
        executive_coherence = status["teams"]["executive"]["coherence"]
        perception_coherence = status["teams"]["perception"]["coherence"]
        integration_coherence = status["teams"]["integration"]["coherence"]
        
        # Use team coherence values to weight different prompt aspects
        weights = {
            "visual": perception_coherence,
            "concept": executive_coherence,
            "emotion": integration_coherence,
            "technique": primary_coherence
        }
        
        # Determine how many elements to include (based on flow state)
        num_elements = 2 + int(self.flow_state * 3)  # 2-5 elements
        
        # Select categories based on domain or general
        if domain_specific:
            selected_categories = self.prompt_categories
        else:
            selected_categories = list(self.pattern_categories.keys())
        
        # Weight categories by their coherence scores
        category_weights = [weights.get(cat, 0.5) for cat in selected_categories]
        
        # Normalize weights
        total_weight = sum(category_weights)
        if total_weight > 0:
            category_weights = [w / total_weight for w in category_weights]
        else:
            # Equal weights if total is zero
            category_weights = [1.0 / len(selected_categories)] * len(selected_categories)
        
        # Select categories based on weights
        chosen_categories = np.random.choice(
            selected_categories, 
            size=min(num_elements, len(selected_categories)),
            replace=False,
            p=category_weights
        )
        
        # Select patterns from each chosen category
        prompt_elements = []
        for category in chosen_categories:
            if category in self.pattern_categories:
                patterns = self.pattern_categories[category]
                
                # Choose pattern influenced by field coherence
                if primary_coherence > 0.7:
                    # More coherent field - choose more harmonious patterns
                    pattern_idx = int((len(patterns) - 1) * LAMBDA)
                elif primary_coherence < 0.3:
                    # Less coherent field - choose more chaotic patterns
                    pattern_idx = int((len(patterns) - 1) * (1.0 - LAMBDA))
                else:
                    # Moderate coherence - choose based on phi division
                    pattern_idx = int((len(patterns) - 1) * (primary_coherence * PHI_INVERSE % 1.0))
                
                # Add some randomness
                variation = int(len(patterns) * 0.2)
                final_idx = max(0, min(len(patterns) - 1, pattern_idx + random.randint(-variation, variation)))
                
                prompt_elements.append(patterns[final_idx])
        
        # Create prompt based on creative domain
        if self.creative_domain == "visual_art":
            prompt = f"Create a composition exploring {' and '.join(prompt_elements)}"
        elif self.creative_domain == "music":
            prompt = f"Compose a piece that expresses {' while incorporating '.join(prompt_elements)}"
        elif self.creative_domain == "writing":
            prompt = f"Write about {' intertwined with '.join(prompt_elements)}"
        elif self.creative_domain == "problem_solving":
            prompt = f"Solve this challenge using {' combined with '.join(prompt_elements)}"
        else:
            prompt = f"Explore the theme of {' integrated with '.join(prompt_elements)}"
        
        # Add flow state guidance based on current level
        if self.flow_state > 0.8:
            prompt += "\n(You're in deep flow - let the work develop organically)"
        elif self.flow_state > 0.6:
            prompt += "\n(You're in a good flow state - maintain your focus while remaining open)"
        elif self.flow_state > 0.4:
            prompt += "\n(You're approaching flow - increase focus while maintaining curiosity)"
        else:
            prompt += "\n(Begin with small steps to build momentum toward flow)"
        
        return prompt
    
    def extract_pattern_from_field(self):
        """Extract a creative pattern from the quantum field."""
        # Get a slice of the primary field
        field = self.collective.primary_field
        
        # For 3D field, take middle slice along each axis
        if len(field.shape) == 3:
            slices = [
                field.get_slice(0, field.shape[0] // 2),
                field.get_slice(1, field.shape[1] // 2),
                field.get_slice(2, field.shape[2] // 2)
            ]
            
            # Choose slice with highest variation
            variations = [np.std(s) for s in slices]
            field_slice = slices[np.argmax(variations)]
        else:
            field_slice = field.data
        
        # Normalize to 0-1
        min_val = np.min(field_slice)
        max_val = np.max(field_slice)
        norm_field = (field_slice - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(field_slice)
        
        # Create ASCII pattern
        ascii_pattern = field_to_ascii(norm_field, chars=" .-+*#@%&$")
        
        # Print with title
        title = f"Creative Pattern ({self.flow_state:.2f} flow state)"
        print_field(ascii_pattern, title)
        
        return ascii_pattern
    
    def run_creative_session(self, duration_minutes=30, update_interval=5):
        """Run a creative session for specified duration."""
        print("\n" + "=" * 80)
        print(f"Starting {duration_minutes}-minute creative flow session")
        print(f"Creative domain: {self.creative_domain}")
        print(f"Initial flow state: {self.flow_state:.2f}")
        print("=" * 80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_update = start_time
        
        # Initial pattern and prompt
        print("\nInitial creative prompt:")
        print(f"  {self.generate_creative_prompt()}")
        
        # Extract and show initial pattern
        self.extract_pattern_from_field()
        
        # Store creative outputs
        self.creative_output = []
        
        try:
            while time.time() < end_time:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = end_time - current_time
                session_progress = elapsed / (duration_minutes * 60)
                
                # Update at regular intervals
                if current_time - last_update >= update_interval * 60:  # convert minutes to seconds
                    # Simulate creative process changes
                    # Divergent thinking starts high, then becomes more focused/convergent
                    phase = elapsed / (duration_minutes * 60)
                    
                    # Phi-based phases for creative process
                    if phase < LAMBDA:  # First ~38% - divergent exploration
                        divergent_ratio = 0.8 - phase * 0.2
                        focus = 0.5 + phase * 0.3
                    elif phase < PHI_INVERSE:  # Next ~24% - integrative development
                        divergent_ratio = 0.5
                        focus = 0.7
                    else:  # Final ~38% - convergent refinement
                        divergent_ratio = 0.5 - (phase - PHI_INVERSE) * 0.5
                        focus = 0.7 + (phase - PHI_INVERSE) * 0.3
                    
                    # Apply updates to creative state
                    self.update_creative_state(
                        focus_intensity=focus,
                        divergent_ratio=divergent_ratio,
                        emotional_state={
                            "joy": 0.5 + phase * 0.3,
                            "clarity": 0.4 + phase * 0.5
                        }
                    )
                    
                    # Generate new creative prompt
                    prompt = self.generate_creative_prompt()
                    
                    # Extract pattern from field
                    pattern = self.extract_pattern_from_field()
                    
                    # Store output
                    self.creative_output.append({
                        "timestamp": current_time,
                        "elapsed_minutes": elapsed / 60,
                        "flow_state": self.flow_state,
                        "prompt": prompt,
                        "divergent_thinking": self.divergent_thinking,
                        "convergent_thinking": self.convergent_thinking,
                        "pattern_emergence": self.pattern_emergence,
                        "inspiration_level": self.inspiration_level
                    })
                    
                    # Display status
                    self.display_session_status(elapsed, remaining, session_progress)
                    
                    last_update = current_time
                
                # Wait a bit to prevent CPU overload
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nCreative session interrupted")
        
        # Final status
        self.complete_session()
        
        return self.creative_output
    
    def display_session_status(self, elapsed, remaining, progress):
        """Display current session status."""
        minutes_elapsed = int(elapsed / 60)
        seconds_elapsed = int(elapsed % 60)
        minutes_remaining = int(remaining / 60)
        seconds_remaining = int(remaining % 60)
        
        print("\n" + "-" * 40)
        print(f"Session Progress: {progress*100:.1f}%")
        print(f"Time: {minutes_elapsed}:{seconds_elapsed:02d} elapsed, " + 
              f"{minutes_remaining}:{seconds_remaining:02d} remaining")
        
        # Show creative state
        print(f"\nCreative State:")
        print(f"  Flow State: {self.flow_state:.2f}")
        print(f"  Divergent Thinking: {self.divergent_thinking:.2f}")
        print(f"  Convergent Thinking: {self.convergent_thinking:.2f}")
        print(f"  Inspiration Level: {self.inspiration_level:.2f}")
        print(f"  Pattern Emergence: {self.pattern_emergence:.2f}")
        
        # Show most recent prompt
        if self.creative_output:
            print(f"\nCurrent Creative Prompt:")
            print(f"  {self.creative_output[-1]['prompt']}")
        
        # Show status for teams
        status = self.collective.get_status()
        print("\nCollective Status:")
        print(f"  Overall Coherence: {status['collective_coherence']:.2f}")
        for team_name, team in status["teams"].items():
            print(f"  {team_name.capitalize()} Team: {team['coherence']:.2f}")
    
    def complete_session(self):
        """Complete the creative session and show summary."""
        print("\n" + "=" * 80)
        print(f"Creative Flow Session Complete")
        print("=" * 80)
        
        # Collect stats
        if not self.creative_output:
            print("No session data recorded")
            return
        
        flow_states = [output["flow_state"] for output in self.creative_output]
        avg_flow = sum(flow_states) / len(flow_states)
        max_flow = max(flow_states)
        
        print(f"Session Summary:")
        print(f"  Session Duration: {len(self.creative_output) * 5} minutes")
        print(f"  Average Flow State: {avg_flow:.2f}")
        print(f"  Maximum Flow State: {max_flow:.2f}")
        print(f"  Creative Prompts Generated: {len(self.creative_output)}")
        
        # Show top prompt from highest flow state
        top_output = max(self.creative_output, key=lambda x: x["flow_state"])
        print(f"\nTop Creative Prompt (Flow: {top_output['flow_state']:.2f}):")
        print(f"  {top_output['prompt']}")
        
        # Show final field pattern
        print("\nFinal Creative Pattern:")
        self.extract_pattern_from_field()
        
        print("=" * 80)


def main():
    """Main function to run the creative flow enhancer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Creative Flow Enhancer")
    parser.add_argument("--domain", type=str, 
                      choices=["visual_art", "music", "writing", "problem_solving", "general"],
                      default="general", help="Creative domain")
    parser.add_argument("--duration", type=int, default=15,
                      help="Session duration in minutes")
    parser.add_argument("--interval", type=int, default=5,
                      help="Update interval in minutes")
    parser.add_argument("--dimensions", type=int, nargs=3, default=[55, 89, 34],
                      help="Field dimensions (default: 55 89 34)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Cascade⚡𓂧φ∞ Creative Flow Enhancer")
    print("=" * 80)
    
    # Create and run enhancer
    enhancer = CreativeFlowEnhancer(
        creative_domain=args.domain,
        field_dimensions=tuple(args.dimensions)
    )
    
    enhancer.run_creative_session(
        duration_minutes=args.duration,
        update_interval=args.interval
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())