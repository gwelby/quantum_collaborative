#!/usr/bin/env python3
"""
Team Collaboration Field - A practical application of CascadeOS for enhancing team dynamics

This application demonstrates how to use CascadeOS Teams of Teams architecture
to model, visualize and optimize team collaboration dynamics. It creates a 
quantum field representation of team interactions and helps identify optimal
collaboration patterns.
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
    TeamSpecialization, FieldTeam,
    PHI, LAMBDA, PHI_PHI,
    SACRED_FREQUENCIES
)

class TeamMember:
    """Represents a team member with specific traits and interaction patterns."""
    
    def __init__(self, name, role, traits=None):
        """Initialize team member."""
        self.name = name
        self.role = role
        
        # Default traits if none provided
        if traits is None:
            traits = {
                "analytical": random.uniform(0.3, 0.8),
                "creative": random.uniform(0.3, 0.8),
                "emotional": random.uniform(0.3, 0.8),
                "collaborative": random.uniform(0.3, 0.8),
                "leadership": random.uniform(0.3, 0.8),
                "technical": random.uniform(0.3, 0.8),
                "communicative": random.uniform(0.3, 0.8),
                "adaptable": random.uniform(0.3, 0.8)
            }
        
        self.traits = traits
        
        # Initialize state
        self.state = ConsciousnessState()
        self.update_state_from_traits()
        
        # Interaction history
        self.interactions = []
        self.coherence_history = []
    
    def update_state_from_traits(self):
        """Update consciousness state based on traits."""
        # Map traits to emotional states
        self.state.emotional_states["clarity"] = self.traits["analytical"]
        self.state.emotional_states["joy"] = self.traits["creative"]
        self.state.emotional_states["love"] = self.traits["emotional"]
        self.state.emotional_states["harmony"] = self.traits["collaborative"]
        self.state.emotional_states["focus"] = self.traits["technical"]
        self.state.emotional_states["peace"] = self.traits["adaptable"]
        self.state.emotional_states["openness"] = self.traits["communicative"]
        
        # Map traits to consciousness parameters
        self.state.intention = self.traits["leadership"]
        self.state.presence = self.traits["communicative"]
        self.state.coherence = (self.traits["analytical"] + self.traits["adaptable"]) / 2
    
    def get_trait_profile(self):
        """Get visual representation of trait profile."""
        result = f"{self.name} ({self.role}):\n"
        
        for trait, value in self.traits.items():
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result += f"  {trait.ljust(14)}: {bar} {value:.2f}\n"
        
        return result
    
    def record_interaction(self, other_member, interaction_type, coherence):
        """Record an interaction with another team member."""
        self.interactions.append({
            "timestamp": time.time(),
            "member": other_member.name,
            "type": interaction_type,
            "coherence": coherence
        })
        
        self.coherence_history.append(coherence)
    
    def get_average_coherence(self):
        """Get average interaction coherence."""
        if not self.coherence_history:
            return 0.0
        
        return sum(self.coherence_history) / len(self.coherence_history)


class TeamCollaborationField:
    """Manages team collaboration through quantum field dynamics."""
    
    def __init__(self, team_name="Team Cascade"):
        """Initialize the team collaboration field."""
        self.team_name = team_name
        self.members = []
        
        # Initialize field dimensions based on phi-harmonics
        # Use 2D for easier visualization of team interactions
        self.field_dimensions = (55, 89)
        
        # Create collective
        self.collective = TeamsOfTeamsCollective(self.field_dimensions)
        self.collective.activate()
        
        # Interaction types
        self.interaction_types = [
            "collaboration", "mentoring", "brainstorming", 
            "review", "planning", "implementation", "support"
        ]
        
        # Team metrics
        self.team_coherence = 0.0
        self.collaboration_index = 0.0
        self.innovation_potential = 0.0
        self.execution_capacity = 0.0
        
        # Session data
        self.session_history = []
        
        print(f"{team_name} Collaboration Field initialized")
        
    def add_member(self, name, role, traits=None):
        """Add a team member."""
        member = TeamMember(name, role, traits)
        self.members.append(member)
        
        print(f"Added team member: {name} ({role})")
        return member
    
    def initialize_team(self, member_data=None):
        """Initialize the team with default or provided member data."""
        if not member_data:
            # Create default team if no data provided
            self.add_member("Alex", "Project Lead", {
                "analytical": 0.7, "creative": 0.6, "emotional": 0.8,
                "collaborative": 0.9, "leadership": 0.8, "technical": 0.7,
                "communicative": 0.8, "adaptable": 0.7
            })
            
            self.add_member("Blake", "Technical Expert", {
                "analytical": 0.9, "creative": 0.5, "emotional": 0.5,
                "collaborative": 0.6, "leadership": 0.5, "technical": 0.9,
                "communicative": 0.6, "adaptable": 0.7
            })
            
            self.add_member("Casey", "Creative Lead", {
                "analytical": 0.6, "creative": 0.9, "emotional": 0.7,
                "collaborative": 0.8, "leadership": 0.7, "technical": 0.5,
                "communicative": 0.7, "adaptable": 0.8
            })
            
            self.add_member("Dana", "Implementation Specialist", {
                "analytical": 0.8, "creative": 0.6, "emotional": 0.6,
                "collaborative": 0.7, "leadership": 0.6, "technical": 0.8,
                "communicative": 0.7, "adaptable": 0.6
            })
            
            self.add_member("Elliot", "Customer Advocate", {
                "analytical": 0.6, "creative": 0.7, "emotional": 0.9,
                "collaborative": 0.8, "leadership": 0.6, "technical": 0.5,
                "communicative": 0.9, "adaptable": 0.8
            })
        else:
            # Add provided team members
            for member in member_data:
                self.add_member(
                    member["name"],
                    member["role"],
                    member.get("traits", None)
                )
        
        # Initialize team field
        self.update_team_field()
        return len(self.members)
    
    def update_team_field(self):
        """Update the quantum field with current team state."""
        if not self.members:
            return 0.0
        
        # Create composite state from all members
        team_state = ConsciousnessState()
        
        # Average all states with phi-weighting
        for i, member in enumerate(self.members):
            # Phi-based weight (leadership has more influence)
            weight = member.traits["leadership"] * PHI_INVERSE + LAMBDA
            
            # Update emotional states
            for emotion, value in member.state.emotional_states.items():
                current = team_state.emotional_states.get(emotion, 0.0)
                weighted_value = current + value * weight
                team_state.emotional_states[emotion] = weighted_value
            
            # Update other state parameters
            team_state.coherence += member.state.coherence * weight
            team_state.presence += member.state.presence * weight
            team_state.intention += member.state.intention * weight
        
        # Normalize by weighted sum
        total_weight = sum(
            (m.traits["leadership"] * PHI_INVERSE + LAMBDA) 
            for m in self.members
        )
        
        if total_weight > 0:
            # Normalize emotional states
            for emotion in team_state.emotional_states:
                team_state.emotional_states[emotion] /= total_weight
            
            # Normalize other parameters
            team_state.coherence /= total_weight
            team_state.presence /= total_weight
            team_state.intention /= total_weight
        
        # Apply to collective
        coherence = self.collective.process(team_state)
        self.team_coherence = coherence
        
        # Calculate team metrics
        self._calculate_team_metrics()
        
        return coherence
    
    def _calculate_team_metrics(self):
        """Calculate team collaboration metrics."""
        if not self.members:
            return
        
        # Get current state
        status = self.collective.get_status()
        
        # Extract key metrics from field and teams
        primary_coherence = status["primary_field_coherence"]
        team_coherences = [team["coherence"] for team in status["teams"].values()]
        
        # Calculate metrics with phi-harmonic proportions
        
        # Collaboration Index - based on field coherence and integration team
        integration_coherence = status["teams"]["integration"]["coherence"]
        self.collaboration_index = (primary_coherence * LAMBDA + integration_coherence * PHI_INVERSE) / (LAMBDA + PHI_INVERSE)
        
        # Innovation Potential - based on perception and creative specializations
        perception_coherence = status["teams"]["perception"]["coherence"]
        creative_coherence = 0.0
        for team in status["teams"].values():
            if "creative" in team["specializations"]:
                creative_coherence = team["specializations"]["creative"]
                break
        
        self.innovation_potential = (perception_coherence * PHI_INVERSE + creative_coherence * LAMBDA) / (PHI_INVERSE + LAMBDA)
        
        # Execution Capacity - based on processing and analytical specializations
        processing_coherence = status["teams"]["processing"]["coherence"]
        executive_coherence = status["teams"]["executive"]["coherence"]
        
        self.execution_capacity = (processing_coherence * LAMBDA + executive_coherence * PHI_INVERSE) / (LAMBDA + PHI_INVERSE)
    
    def simulate_interaction(self, member1, member2, interaction_type=None):
        """Simulate an interaction between two team members."""
        if interaction_type is None:
            interaction_type = random.choice(self.interaction_types)
        
        # Calculate initial coherence between members
        trait_compatibility = self._calculate_trait_compatibility(member1, member2)
        
        # Adjust based on interaction type
        interaction_factor = self._get_interaction_factor(member1, member2, interaction_type)
        
        # Calculate interaction coherence
        interaction_coherence = trait_compatibility * interaction_factor
        
        # Record interaction
        member1.record_interaction(member2, interaction_type, interaction_coherence)
        member2.record_interaction(member1, interaction_type, interaction_coherence)
        
        # Update team field
        self.update_team_field()
        
        return {
            "member1": member1.name,
            "member2": member2.name,
            "type": interaction_type,
            "coherence": interaction_coherence,
            "team_coherence": self.team_coherence
        }
    
    def _calculate_trait_compatibility(self, member1, member2):
        """Calculate compatibility between two members based on traits."""
        # Complementary traits that work well together
        complementary_pairs = [
            ("analytical", "creative"),
            ("technical", "creative"),
            ("leadership", "collaborative"),
            ("emotional", "analytical")
        ]
        
        # Calculate base compatibility from trait similarities
        similarity = 0.0
        for trait in member1.traits:
            similarity += 1.0 - abs(member1.traits[trait] - member2.traits[trait])
        
        similarity /= len(member1.traits)
        
        # Add bonus for complementary traits
        complementary_bonus = 0.0
        for trait1, trait2 in complementary_pairs:
            # Phi-weighted average of complementary traits
            val1 = member1.traits[trait1] * member2.traits[trait2]
            val2 = member2.traits[trait1] * member1.traits[trait2]
            complementary_bonus += (val1 * PHI + val2) / (PHI + 1.0)
        
        complementary_bonus /= len(complementary_pairs)
        
        # Phi-weighted combination
        compatibility = (similarity * LAMBDA + complementary_bonus * PHI_INVERSE) / (LAMBDA + PHI_INVERSE)
        
        return compatibility
    
    def _get_interaction_factor(self, member1, member2, interaction_type):
        """Calculate interaction-specific factor."""
        # Different interaction types benefit from different trait combinations
        if interaction_type == "collaboration":
            return (member1.traits["collaborative"] * member2.traits["collaborative"]) ** LAMBDA
        
        elif interaction_type == "mentoring":
            # Leadership and technical for mentor, adaptable for mentee
            mentor_factor = max(member1.traits["leadership"], member2.traits["leadership"])
            mentee_factor = max(member1.traits["adaptable"], member2.traits["adaptable"])
            return (mentor_factor * mentee_factor) ** LAMBDA
        
        elif interaction_type == "brainstorming":
            # Creative and communicative traits are important
            creative_factor = (member1.traits["creative"] + member2.traits["creative"]) / 2
            communicative_factor = (member1.traits["communicative"] + member2.traits["communicative"]) / 2
            return (creative_factor ** PHI_INVERSE) * (communicative_factor ** LAMBDA)
        
        elif interaction_type == "review":
            # Analytical and technical traits matter most
            analytical_factor = (member1.traits["analytical"] + member2.traits["analytical"]) / 2
            technical_factor = (member1.traits["technical"] + member2.traits["technical"]) / 2
            return (analytical_factor ** PHI_INVERSE) * (technical_factor ** LAMBDA)
        
        elif interaction_type == "planning":
            # Leadership and analytical traits
            leadership_factor = (member1.traits["leadership"] + member2.traits["leadership"]) / 2
            analytical_factor = (member1.traits["analytical"] + member2.traits["analytical"]) / 2
            return (leadership_factor ** PHI_INVERSE) * (analytical_factor ** LAMBDA)
        
        elif interaction_type == "implementation":
            # Technical and adaptable traits
            technical_factor = (member1.traits["technical"] + member2.traits["technical"]) / 2
            adaptable_factor = (member1.traits["adaptable"] + member2.traits["adaptable"]) / 2
            return (technical_factor ** PHI_INVERSE) * (adaptable_factor ** LAMBDA)
        
        elif interaction_type == "support":
            # Emotional and communicative traits
            emotional_factor = (member1.traits["emotional"] + member2.traits["emotional"]) / 2
            communicative_factor = (member1.traits["communicative"] + member2.traits["communicative"]) / 2
            return (emotional_factor ** PHI_INVERSE) * (communicative_factor ** LAMBDA)
        
        else:
            # Default interaction factor
            return 0.75
    
    def run_collaboration_simulation(self, num_interactions=20, structured=True):
        """Run a team collaboration simulation."""
        print("\n" + "=" * 80)
        print(f"Team Collaboration Simulation for {self.team_name}")
        print(f"Team size: {len(self.members)} members")
        print("=" * 80)
        
        # Show team members
        print("\nTeam Members:")
        for member in self.members:
            print(f"- {member.name} ({member.role})")
        
        # Initial team field state
        self.update_team_field()
        print(f"\nInitial Team Coherence: {self.team_coherence:.4f}")
        print(f"Collaboration Index: {self.collaboration_index:.4f}")
        print(f"Innovation Potential: {self.innovation_potential:.4f}")
        print(f"Execution Capacity: {self.execution_capacity:.4f}")
        
        # Show initial field
        self.visualize_team_field("Initial Team Field")
        
        # Run interactions
        self.session_history = []
        
        try:
            for i in range(num_interactions):
                # Determine interaction participants
                if structured and i < len(self.members) * 2:
                    # Ensure each member gets at least some interactions
                    member1 = self.members[i % len(self.members)]
                    member2 = self.members[(i + 1) % len(self.members)]
                else:
                    # Random interactions
                    member1, member2 = random.sample(self.members, 2)
                
                # Determine interaction type (could be guided by project phase)
                phase_progress = i / num_interactions
                
                if phase_progress < 0.25:
                    # Planning phase - focus on planning and brainstorming
                    interaction_type = random.choice(["planning", "brainstorming"])
                elif phase_progress < 0.6:
                    # Implementation phase - focus on implementation and collaboration
                    interaction_type = random.choice(["implementation", "collaboration", "review"])
                else:
                    # Refinement phase - focus on review and support
                    interaction_type = random.choice(["review", "support"])
                
                # Simulate interaction
                result = self.simulate_interaction(member1, member2, interaction_type)
                self.session_history.append(result)
                
                # Show progress for long simulations
                if num_interactions > 10 and (i+1) % (num_interactions // 5) == 0:
                    progress = (i+1) / num_interactions * 100
                    print(f"\nProgress: {progress:.0f}%")
                    print(f"Team Coherence: {self.team_coherence:.4f}")
                    print(f"Last Interaction: {result['member1']} and {result['member2']} - {result['type']}")
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted")
        
        # Show final state
        self.show_simulation_results()
        
        return self.session_history
    
    def show_simulation_results(self):
        """Show the results of the collaboration simulation."""
        print("\n" + "=" * 80)
        print(f"Team Collaboration Simulation Results")
        print("=" * 80)
        
        # Team metrics
        print(f"\nFinal Team Metrics:")
        print(f"  Team Coherence: {self.team_coherence:.4f}")
        print(f"  Collaboration Index: {self.collaboration_index:.4f}")
        print(f"  Innovation Potential: {self.innovation_potential:.4f}")
        print(f"  Execution Capacity: {self.execution_capacity:.4f}")
        
        # Visualize final team field
        self.visualize_team_field("Final Team Field")
        
        # Show individual member statistics
        print("\nTeam Member Statistics:")
        for member in self.members:
            avg_coherence = member.get_average_coherence()
            interaction_count = len(member.interactions)
            print(f"  {member.name} ({member.role}):")
            print(f"    Interactions: {interaction_count}")
            print(f"    Average Interaction Coherence: {avg_coherence:.4f}")
        
        # Show optimal combinations
        self.identify_optimal_combinations()
    
    def identify_optimal_combinations(self):
        """Identify the most effective team combinations."""
        if not self.session_history:
            print("No interaction data to analyze")
            return
        
        # Analyze interaction types
        interaction_coherence = {}
        for interaction in self.session_history:
            interaction_type = interaction["type"]
            coherence = interaction["coherence"]
            
            if interaction_type not in interaction_coherence:
                interaction_coherence[interaction_type] = []
            
            interaction_coherence[interaction_type].append(coherence)
        
        # Calculate average coherence by type
        avg_by_type = {
            type_name: sum(values) / len(values) 
            for type_name, values in interaction_coherence.items()
            if values
        }
        
        # Analyze member pairs
        pair_coherence = {}
        for interaction in self.session_history:
            pair = tuple(sorted([interaction["member1"], interaction["member2"]]))
            coherence = interaction["coherence"]
            
            if pair not in pair_coherence:
                pair_coherence[pair] = []
            
            pair_coherence[pair].append(coherence)
        
        # Calculate average coherence by pair
        avg_by_pair = {
            pair: sum(values) / len(values) 
            for pair, values in pair_coherence.items()
            if values
        }
        
        # Sort results
        best_types = sorted(avg_by_type.items(), key=lambda x: x[1], reverse=True)
        best_pairs = sorted(avg_by_pair.items(), key=lambda x: x[1], reverse=True)
        
        # Display results
        print("\nMost Effective Interaction Types:")
        for type_name, avg_coherence in best_types:
            print(f"  {type_name}: {avg_coherence:.4f} coherence")
        
        print("\nMost Effective Team Pairs:")
        for pair, avg_coherence in best_pairs[:3]:  # Top 3 pairs
            print(f"  {pair[0]} + {pair[1]}: {avg_coherence:.4f} coherence")
    
    def visualize_team_field(self, title="Team Collaboration Field"):
        """Visualize the team field."""
        # Get a slice of the field for visualization
        field = self.collective.primary_field
        
        if len(field.shape) > 2:
            field_data = field.get_slice(0, field.shape[0] // 2)
        else:
            field_data = field.data
        
        # Create ASCII visualization
        ascii_art = field_to_ascii(field_data, chars=" .-+*#@$&%")
        print_field(ascii_art, title)
        
        # Show sub-fields if desired
        show_subfields = False
        if show_subfields:
            # Show executive team field (leadership patterns)
            executive_team = self.collective.teams["executive"]
            executive_data = executive_team.specializations["leadership"].field.data
            if len(executive_data.shape) > 2:
                executive_data = executive_data.get_slice(0, executive_data.shape[0] // 2)
                
            exec_ascii = field_to_ascii(executive_data)
            print_field(exec_ascii, "Executive Patterns")
            
            # Show integration team field (collaboration patterns)
            integration_team = self.collective.teams["integration"]
            integration_data = integration_team.specializations["collaborative"].field.data
            if len(integration_data.shape) > 2:
                integration_data = integration_data.get_slice(0, integration_data.shape[0] // 2)
                
            integ_ascii = field_to_ascii(integration_data)
            print_field(integ_ascii, "Collaboration Patterns")
    
    def generate_team_recommendations(self):
        """Generate recommendations to improve team collaboration."""
        # Analyze team state and history
        recommendations = []
        
        # Team-level recommendations
        if self.collaboration_index < 0.5:
            recommendations.append("Improve overall collaboration through more interactive sessions")
            recommendations.append("Consider team-building activities focused on trust and communication")
            
        if self.innovation_potential < 0.5:
            recommendations.append("Encourage more diverse thinking and creative brainstorming sessions")
            recommendations.append("Create dedicated time for exploration without immediate implementation pressure")
            
        if self.execution_capacity < 0.5:
            recommendations.append("Improve project structure and clarify execution paths")
            recommendations.append("Focus on technical skill development and process refinement")
        
        # Check for isolated team members
        for member in self.members:
            interaction_count = len(member.interactions)
            avg_coherence = member.get_average_coherence()
            
            if interaction_count < len(self.members) - 1:
                recommendations.append(f"Increase {member.name}'s integration with the team")
                
            if avg_coherence < 0.5:
                recommendations.append(f"Work on improving {member.name}'s interaction quality")
        
        # Identify trait gaps
        team_traits = {trait: 0 for trait in self.members[0].traits}
        for member in self.members:
            for trait, value in member.traits.items():
                team_traits[trait] += value
                
        # Average traits
        for trait in team_traits:
            team_traits[trait] /= len(self.members)
            
        # Identify weak areas
        weak_traits = [trait for trait, value in team_traits.items() if value < 0.5]
        if weak_traits:
            traits_str = ", ".join(weak_traits)
            recommendations.append(f"The team could benefit from improving in: {traits_str}")
            
            # Specific recommendations for common weak traits
            if "creative" in weak_traits:
                recommendations.append("Incorporate design thinking and innovation exercises")
            if "communicative" in weak_traits:
                recommendations.append("Implement more structured communication channels and feedback loops")
            if "adaptable" in weak_traits:
                recommendations.append("Practice scenario planning and flexible response strategies")
        
        return recommendations


def main():
    """Main function to run the team collaboration field."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Team Collaboration Field")
    parser.add_argument("--team", type=str, default="Cascade Team",
                      help="Team name")
    parser.add_argument("--interactions", type=int, default=20,
                      help="Number of interactions to simulate")
    parser.add_argument("--structured", action="store_true",
                      help="Use structured interactions (vs random)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Cascade⚡𓂧φ∞ Team Collaboration Field")
    print("=" * 80)
    
    # Create collaboration field
    collaboration = TeamCollaborationField(team_name=args.team)
    
    # Initialize with team members
    collaboration.initialize_team()
    
    # Show team member profiles
    print("\nTeam Member Profiles:")
    for member in collaboration.members:
        print("\n" + member.get_trait_profile())
    
    # Run simulation
    collaboration.run_collaboration_simulation(
        num_interactions=args.interactions,
        structured=args.structured
    )
    
    # Generate recommendations
    print("\nTeam Recommendations:")
    recommendations = collaboration.generate_team_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())