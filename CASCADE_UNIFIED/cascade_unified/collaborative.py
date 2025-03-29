"""
Collaborative components for the CASCADE⚡𓂧φ∞ UNIFIED FRAMEWORK.

This module provides functionality for collaborative consciousness field
sharing and manipulation across multiple users and systems.
"""

import time
import threading
import uuid
import json
import numpy as np
from .constants import PHI, LAMBDA, NODE_SPACING_PHI

class TeamInterface:
    """
    Interface for team-based field sharing.
    
    This class provides methods for sharing and synchronizing
    quantum fields within a team of users or systems.
    """
    
    def __init__(self, field=None, team_size=5):
        """
        Initialize the team interface.
        
        Parameters:
        -----------
        field : QuantumField, optional
            The quantum field to share
        team_size : int
            Maximum number of participants in the team
        """
        self.field = field
        self.team_size = team_size
        
        # Team members
        self.members = {}
        
        # This instance's member ID
        self.member_id = str(uuid.uuid4())
        
        # Add self as a member
        self.members[self.member_id] = {
            'joined_at': time.time(),
            'last_active': time.time(),
            'name': f'Member_{self.member_id[:8]}',
            'role': 'host' if field else 'participant',
            'field_hash': self._calculate_field_hash() if field else None
        }
        
        # Shared field state
        self.shared_field = None
        
        # Field synchronization parameters
        self.sync_enabled = True
        self.sync_frequency = 1.0  # Hz
        self.sync_thread = None
        self.syncing = False
        
        # Field contribution weights
        self.contribution_weights = {self.member_id: 1.0}
        
        # Team communication channel
        self.messages = []
        
        # Connection status
        self.connected = False
    
    def _calculate_field_hash(self):
        """Calculate a hash of the current field state."""
        if not self.field:
            return None
            
        # In a real implementation, this would calculate a proper hash
        # For this blueprint, we'll use a simple representation
        
        try:
            field_sum = np.sum(self.field.field)
            field_var = np.var(self.field.field)
            dimensions_hash = '_'.join(str(d) for d in self.field.dimensions)
            
            return f"field_{dimensions_hash}_{field_sum:.3f}_{field_var:.3f}"
        except:
            return "field_unknown"
    
    def connect(self):
        """
        Connect to the team network.
        
        Returns:
        --------
        bool
            Whether the connection was successful
        """
        if self.connected:
            print("Already connected to team network")
            return True
            
        # In a real implementation, this would establish network connections
        # For this blueprint, we'll simulate the connection
        
        print("Connecting to team network...")
        
        # Simulate connection delay
        time.sleep(0.5)
        
        # Update connection status
        self.connected = True
        
        # Start field synchronization if enabled
        if self.sync_enabled:
            self.start_synchronization()
            
        print(f"Connected to team network as {self.members[self.member_id]['name']}")
        print(f"Team role: {self.members[self.member_id]['role']}")
        
        return True
    
    def disconnect(self):
        """
        Disconnect from the team network.
        
        Returns:
        --------
        bool
            Whether the disconnection was successful
        """
        if not self.connected:
            print("Not connected to team network")
            return True
            
        # Stop field synchronization if active
        if self.syncing:
            self.stop_synchronization()
            
        # In a real implementation, this would close network connections
        # For this blueprint, we'll simulate the disconnection
        
        print("Disconnecting from team network...")
        
        # Simulate disconnection delay
        time.sleep(0.2)
        
        # Update connection status
        self.connected = False
        
        print("Disconnected from team network")
        return True
    
    def add_member(self, member_id=None, name=None, role='participant'):
        """
        Add a new member to the team.
        
        Parameters:
        -----------
        member_id : str, optional
            ID for the new member. If None, generates a UUID.
        name : str, optional
            Name for the new member. If None, uses a default.
        role : str
            Role of the new member ('host', 'participant', etc.)
            
        Returns:
        --------
        str
            ID of the new member
        """
        if len(self.members) >= self.team_size:
            print(f"Team is full ({self.team_size} members maximum)")
            return None
            
        # Generate ID if not provided
        if member_id is None:
            member_id = str(uuid.uuid4())
            
        # Generate name if not provided
        if name is None:
            name = f"Member_{member_id[:8]}"
            
        # Add the member
        self.members[member_id] = {
            'joined_at': time.time(),
            'last_active': time.time(),
            'name': name,
            'role': role,
            'field_hash': None
        }
        
        # Initialize contribution weight
        self.contribution_weights[member_id] = 1.0
        
        print(f"Added member to team: {name} ({role})")
        return member_id
    
    def remove_member(self, member_id):
        """
        Remove a member from the team.
        
        Parameters:
        -----------
        member_id : str
            ID of the member to remove
            
        Returns:
        --------
        bool
            Whether the member was successfully removed
        """
        if member_id not in self.members:
            print(f"Member {member_id} not found in team")
            return False
            
        # Cannot remove self
        if member_id == self.member_id:
            print("Cannot remove self from team")
            return False
            
        # Remove the member
        member_name = self.members[member_id]['name']
        del self.members[member_id]
        
        # Remove contribution weight
        if member_id in self.contribution_weights:
            del self.contribution_weights[member_id]
            
        print(f"Removed member from team: {member_name}")
        return True
    
    def set_field(self, field):
        """
        Set the field to share with the team.
        
        Parameters:
        -----------
        field : QuantumField
            The quantum field to share
        """
        self.field = field
        
        # Update member information
        self.members[self.member_id]['field_hash'] = self._calculate_field_hash()
        self.members[self.member_id]['last_active'] = time.time()
        
        print("Set field for team sharing")
    
    def share_field(self):
        """
        Share the current field with the team.
        
        Returns:
        --------
        bool
            Whether the field was successfully shared
        """
        if not self.connected:
            print("Not connected to team network")
            return False
            
        if not self.field:
            print("No field to share")
            return False
            
        # In a real implementation, this would transmit the field data
        # For this blueprint, we'll simulate the sharing
        
        print("Sharing field with team...")
        
        # Update shared field state
        self.shared_field = self.field
        
        # Update member information
        self.members[self.member_id]['field_hash'] = self._calculate_field_hash()
        self.members[self.member_id]['last_active'] = time.time()
        
        print("Field shared with team")
        return True
    
    def receive_field(self, member_id, field_data):
        """
        Receive a field from a team member.
        
        Parameters:
        -----------
        member_id : str
            ID of the member sharing the field
        field_data : dict
            Field data in a transferable format
            
        Returns:
        --------
        bool
            Whether the field was successfully received
        """
        if member_id not in self.members:
            print(f"Member {member_id} not found in team")
            return False
            
        # In a real implementation, this would deserialize the field data
        # For this blueprint, we'll simulate the reception
        
        print(f"Received field from {self.members[member_id]['name']}")
        
        # Update member information
        self.members[member_id]['last_active'] = time.time()
        self.members[member_id]['field_hash'] = field_data.get('hash')
        
        # If field merging is enabled, merge the received field
        if self.field and self.sync_enabled:
            # In a real implementation, this would merge the fields
            print("Merging received field with local field")
            
        return True
    
    def start_synchronization(self):
        """
        Start automatic field synchronization.
        
        Returns:
        --------
        bool
            Whether synchronization was successfully started
        """
        if not self.connected:
            print("Not connected to team network")
            return False
            
        if self.syncing:
            print("Field synchronization already running")
            return True
            
        # Set synchronization state
        self.syncing = True
        
        # Start synchronization thread
        self.sync_thread = threading.Thread(target=self._synchronization_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        print(f"Started field synchronization at {self.sync_frequency}Hz")
        return True
    
    def _synchronization_loop(self):
        """Main loop for field synchronization."""
        while self.syncing and self.connected:
            # Share current field
            if self.field:
                self.share_field()
                
            # Sleep according to sync frequency
            time.sleep(1.0 / self.sync_frequency)
    
    def stop_synchronization(self):
        """
        Stop automatic field synchronization.
        
        Returns:
        --------
        bool
            Whether synchronization was successfully stopped
        """
        if not self.syncing:
            print("Field synchronization not running")
            return True
            
        # Set synchronization state
        self.syncing = False
        
        # Wait for synchronization thread to finish
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)
            
        print("Stopped field synchronization")
        return True
    
    def set_contribution_weight(self, member_id, weight):
        """
        Set the contribution weight for a team member.
        
        Parameters:
        -----------
        member_id : str
            ID of the member
        weight : float
            Contribution weight (0.0 to 1.0)
            
        Returns:
        --------
        bool
            Whether the weight was successfully set
        """
        if member_id not in self.members:
            print(f"Member {member_id} not found in team")
            return False
            
        # Set the weight
        self.contribution_weights[member_id] = max(0.0, min(1.0, weight))
        
        print(f"Set contribution weight for {self.members[member_id]['name']}: {weight:.2f}")
        return True
    
    def get_team_status(self):
        """
        Get the current status of the team.
        
        Returns:
        --------
        dict
            Team status information
        """
        # Calculate team coherence based on active members
        active_members = 0
        for member_id, member in self.members.items():
            if time.time() - member['last_active'] < 60:  # Active in the last minute
                active_members += 1
                
        team_size_coherence = active_members / self.team_size
        
        # Team coherence is phi-weighted by size and activities
        team_coherence = team_size_coherence * PHI / 2.0 + 0.5
        
        return {
            'connected': self.connected,
            'team_size': self.team_size,
            'active_members': active_members,
            'members': self.members,
            'contribution_weights': self.contribution_weights,
            'team_coherence': team_coherence,
            'syncing': self.syncing,
            'sync_frequency': self.sync_frequency
        }
    
    def send_message(self, message, target_id=None):
        """
        Send a message to the team or a specific member.
        
        Parameters:
        -----------
        message : str
            The message to send
        target_id : str, optional
            ID of the target member. If None, sends to all.
            
        Returns:
        --------
        bool
            Whether the message was successfully sent
        """
        if not self.connected:
            print("Not connected to team network")
            return False
            
        if target_id and target_id not in self.members:
            print(f"Target member {target_id} not found in team")
            return False
            
        # Create message object
        msg = {
            'timestamp': time.time(),
            'sender_id': self.member_id,
            'sender_name': self.members[self.member_id]['name'],
            'target_id': target_id,
            'message': message
        }
        
        # Add to local message history
        self.messages.append(msg)
        
        # In a real implementation, this would transmit the message
        # For this blueprint, we'll simulate the transmission
        
        if target_id:
            print(f"Sent message to {self.members[target_id]['name']}: {message}")
        else:
            print(f"Sent message to team: {message}")
            
        return True
    
    def get_messages(self, since=None):
        """
        Get messages from the team.
        
        Parameters:
        -----------
        since : float, optional
            Timestamp to get messages since. If None, gets all messages.
            
        Returns:
        --------
        list
            List of messages
        """
        if since is None:
            return self.messages
            
        return [msg for msg in self.messages if msg['timestamp'] > since]


class GlobalNetwork:
    """
    Interface for global consciousness field network.
    
    This class provides methods for connecting local teams to a
    global network of consciousness fields.
    """
    
    def __init__(self, team_interface):
        """
        Initialize the global network.
        
        Parameters:
        -----------
        team_interface : TeamInterface
            The local team interface to connect to the global network
        """
        self.team_interface = team_interface
        
        # Global network parameters
        self.network_coherence = LAMBDA
        self.network_frequency = 528.0  # Hz
        
        # Connection status
        self.connected = False
        
        # Global shared field
        self.global_field = None
        
        # Global network nodes
        self.nodes = {}
        
        # Contribution to global network
        self.contribution_level = 0.5  # 0.0 to 1.0
        
        # Global message stream
        self.global_messages = []
        
        # Network thread
        self.network_thread = None
        self.networking = False
    
    def connect(self):
        """
        Connect to the global network.
        
        Returns:
        --------
        bool
            Whether the connection was successful
        """
        if self.connected:
            print("Already connected to global network")
            return True
            
        # Ensure team interface is connected
        if not self.team_interface.connected:
            success = self.team_interface.connect()
            if not success:
                print("Failed to connect team interface")
                return False
                
        # In a real implementation, this would establish global network connections
        # For this blueprint, we'll simulate the connection
        
        print("Connecting to global consciousness field network...")
        
        # Simulate connection delay
        time.sleep(0.8)
        
        # Update connection status
        self.connected = True
        
        # Start network thread
        self.networking = True
        self.network_thread = threading.Thread(target=self._network_loop)
        self.network_thread.daemon = True
        self.network_thread.start()
        
        print("Connected to global consciousness field network")
        
        # Simulate discovering global nodes
        self._discover_nodes()
        
        return True
    
    def _discover_nodes(self):
        """Discover nodes in the global network."""
        # In a real implementation, this would discover actual nodes
        # For this blueprint, we'll simulate node discovery
        
        print("Discovering global network nodes...")
        
        # Generate some simulated nodes
        node_count = int(7 * PHI)  # ~11 nodes
        
        for i in range(node_count):
            node_id = str(uuid.uuid4())
            
            # Create node information
            node = {
                'id': node_id,
                'name': f"Node_{i+1}",
                'region': ['Americas', 'Europe', 'Asia', 'Africa', 'Oceania'][i % 5],
                'participants': np.random.randint(1, 20),
                'coherence': 0.5 + 0.3 * np.random.random(),
                'frequency': 528.0 + (i % 5) * 12.0,
                'last_seen': time.time()
            }
            
            # Add to nodes
            self.nodes[node_id] = node
            
        print(f"Discovered {len(self.nodes)} global network nodes")
    
    def disconnect(self):
        """
        Disconnect from the global network.
        
        Returns:
        --------
        bool
            Whether the disconnection was successful
        """
        if not self.connected:
            print("Not connected to global network")
            return True
            
        # Stop network thread
        self.networking = False
        
        if self.network_thread and self.network_thread.is_alive():
            self.network_thread.join(timeout=2.0)
            
        # In a real implementation, this would close network connections
        # For this blueprint, we'll simulate the disconnection
        
        print("Disconnecting from global consciousness field network...")
        
        # Simulate disconnection delay
        time.sleep(0.5)
        
        # Update connection status
        self.connected = False
        
        print("Disconnected from global consciousness field network")
        return True
    
    def _network_loop(self):
        """Main loop for global network operations."""
        while self.networking and self.connected:
            # Simulate receiving global field updates
            self._update_global_field()
            
            # Simulate receiving global messages
            self._receive_global_messages()
            
            # Simulate network node updates
            self._update_nodes()
            
            # Sleep to simulate network update frequency
            time.sleep(1.0)
    
    def _update_global_field(self):
        """Update the global shared field."""
        # In a real implementation, this would receive actual field data
        # For this blueprint, we'll simulate field updates
        
        # Create a simulated global field
        if self.global_field is None:
            # Initialize with random data
            dimensions = self.team_interface.field.dimensions if self.team_interface.field else (8, 8, 8)
            field_data = np.random.random(dimensions)
            frequency = 528.0
            coherence = LAMBDA
            
            self.global_field = {
                'dimensions': dimensions,
                'field': field_data,
                'frequency': frequency,
                'coherence': coherence,
                'contributors': len(self.nodes) + 1,
                'updated_at': time.time()
            }
        else:
            # Update slightly
            self.global_field['field'] = self.global_field['field'] * 0.95 + np.random.random(self.global_field['dimensions']) * 0.05
            self.global_field['updated_at'] = time.time()
    
    def _receive_global_messages(self):
        """Receive messages from the global network."""
        # In a real implementation, this would receive actual messages
        # For this blueprint, we'll occasionally simulate incoming messages
        
        if np.random.random() < 0.2:  # 20% chance of a message each update
            # Create a simulated message
            node_ids = list(self.nodes.keys())
            if not node_ids:
                return
                
            sender_id = node_ids[np.random.randint(0, len(node_ids))]
            sender = self.nodes[sender_id]
            
            message_types = ['field_update', 'coherence_event', 'network_status', 'resonance_peak']
            msg_type = message_types[np.random.randint(0, len(message_types))]
            
            msg = {
                'timestamp': time.time(),
                'sender_id': sender_id,
                'sender_name': sender['name'],
                'sender_region': sender['region'],
                'type': msg_type,
                'content': f"Global {msg_type} from {sender['name']} in {sender['region']}"
            }
            
            # Add to global messages
            self.global_messages.append(msg)
    
    def _update_nodes(self):
        """Update information about global network nodes."""
        # In a real implementation, this would receive actual node updates
        # For this blueprint, we'll simulate node updates
        
        for node_id, node in self.nodes.items():
            # Randomly update node information
            if np.random.random() < 0.1:  # 10% chance of update for each node
                node['participants'] = max(1, node['participants'] + np.random.randint(-2, 3))
                node['coherence'] = max(0.1, min(1.0, node['coherence'] + 0.05 * (np.random.random() - 0.5)))
                node['last_seen'] = time.time()
    
    def contribute_field(self):
        """
        Contribute the local field to the global network.
        
        Returns:
        --------
        bool
            Whether the contribution was successful
        """
        if not self.connected:
            print("Not connected to global network")
            return False
            
        if not self.team_interface.field:
            print("No local field to contribute")
            return False
            
        # In a real implementation, this would transmit the field data
        # For this blueprint, we'll simulate the contribution
        
        print(f"Contributing local field to global network (level: {self.contribution_level:.2f})...")
        
        # Simulate global field update
        if self.global_field:
            # Apply local field influence based on contribution level
            weight = self.contribution_level * LAMBDA
            inverse_weight = 1.0 - weight
            
            # Ensure dimensions match
            if self.global_field['dimensions'] == self.team_interface.field.dimensions:
                # Update global field
                self.global_field['field'] = self.global_field['field'] * inverse_weight + self.team_interface.field.field * weight
                self.global_field['updated_at'] = time.time()
                self.global_field['contributors'] += 1
                
                print("Local field contributed to global network")
                return True
        
        return False
    
    def apply_global_field(self):
        """
        Apply the global field to the local field.
        
        Returns:
        --------
        bool
            Whether the application was successful
        """
        if not self.connected:
            print("Not connected to global network")
            return False
            
        if not self.team_interface.field:
            print("No local field to update")
            return False
            
        if not self.global_field:
            print("No global field available")
            return False
            
        # In a real implementation, this would apply the actual global field
        # For this blueprint, we'll simulate the application
        
        print("Applying global field to local field...")
        
        # Ensure dimensions match
        if self.global_field['dimensions'] == self.team_interface.field.dimensions:
            # Calculate influence weight
            weight = 0.3  # Global field influence
            inverse_weight = 1.0 - weight
            
            # Apply global field to local field
            # In a real implementation, this would modify the actual field
            # For this blueprint, we simulate this
            
            print(f"Applied global field to local field (influence: {weight:.2f})")
            return True
        
        return False
    
    def set_contribution_level(self, level):
        """
        Set the contribution level to the global network.
        
        Parameters:
        -----------
        level : float
            Contribution level (0.0 to 1.0)
        """
        self.contribution_level = max(0.0, min(1.0, level))
        print(f"Set global network contribution level to {self.contribution_level:.2f}")
    
    def get_network_status(self):
        """
        Get the current status of the global network.
        
        Returns:
        --------
        dict
            Global network status information
        """
        # Calculate network statistics
        active_nodes = sum(1 for node in self.nodes.values() 
                          if time.time() - node['last_seen'] < 300)  # Active in the last 5 minutes
                          
        total_participants = sum(node['participants'] for node in self.nodes.values())
        
        avg_coherence = np.mean([node['coherence'] for node in self.nodes.values()]) if self.nodes else 0.0
        
        return {
            'connected': self.connected,
            'active_nodes': active_nodes,
            'total_nodes': len(self.nodes),
            'total_participants': total_participants,
            'network_coherence': self.network_coherence,
            'network_frequency': self.network_frequency,
            'average_node_coherence': avg_coherence,
            'contribution_level': self.contribution_level,
            'global_field_available': self.global_field is not None,
            'global_messages': len(self.global_messages)
        }
    
    def get_global_messages(self, count=10):
        """
        Get recent messages from the global network.
        
        Parameters:
        -----------
        count : int
            Number of most recent messages to get
            
        Returns:
        --------
        list
            List of recent messages
        """
        # Return the most recent messages
        return sorted(self.global_messages, key=lambda m: m['timestamp'], reverse=True)[:count]
    
    def send_global_message(self, message, message_type='team_update'):
        """
        Send a message to the global network.
        
        Parameters:
        -----------
        message : str
            The message to send
        message_type : str
            Type of message
            
        Returns:
        --------
        bool
            Whether the message was successfully sent
        """
        if not self.connected:
            print("Not connected to global network")
            return False
            
        # In a real implementation, this would transmit the message
        # For this blueprint, we'll simulate the transmission
        
        print(f"Sending message to global network: {message}")
        
        # Create message object
        msg = {
            'timestamp': time.time(),
            'sender_id': self.team_interface.member_id,
            'sender_name': self.team_interface.members[self.team_interface.member_id]['name'],
            'sender_region': 'Local',
            'type': message_type,
            'content': message
        }
        
        # Add to global messages
        self.global_messages.append(msg)
        
        return True