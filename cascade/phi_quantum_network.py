"""
CASCADE⚡𓂧φ∞ Phi-Quantum Network Protocol

This module implements a distributed quantum field synchronization system 
that maintains phi-harmonic coherence across network boundaries.

The system creates a virtual quantum entanglement between nodes, allowing
for synchronized field evolution while preserving local quantum state autonomy.
"""

import socket
import threading
import time
import json
import hashlib
import math
import random
import struct
import queue
import uuid
import logging
import asyncio
import contextlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

# Try to import cascade core
try:
    from cascade.phi_python_core import (
        PHI, LAMBDA, PHI_PHI, phi_function, PhiTimer, PhiConversion,
        create_toroidal_memory, get_phi_coherence, set_phi_coherence
    )
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895
    LAMBDA = 0.618033988749895
    PHI_PHI = PHI ** PHI
    
    # Simple fallback decorators
    def phi_function(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else decorator(func)
    
    # Logging
    logging.warning("CASCADE phi-python-core not found, using fallback implementations")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("phi_quantum_network")


# Protocol constants
PHI_QUANTUM_PORT = 4321  # Default port (1000 * PHI^2 rounded)
PHI_SYNC_INTERVAL = 1.0 / LAMBDA  # ~1.618 seconds between syncs
PHI_ENTANGLEMENT_TIMEOUT = 60.0 * PHI  # ~97 seconds
PHI_PACKET_MAGIC = 0x504831  # Magic number for packet identification (hex for "PHI1")

# Special phi frequencies for consciousness bridge 
PHI_FREQUENCIES = [432, 528, 594, 672, 720, 768, 888]


class PacketType(Enum):
    """Phi-quantum packet types."""
    HEARTBEAT = 1
    FIELD_SYNC = 2
    ENTANGLEMENT_REQUEST = 3
    ENTANGLEMENT_ACCEPT = 4
    CONSCIOUSNESS_SYNC = 5
    TIMELINE_MARKER = 6
    COHERENCE_UPDATE = 7
    FIELD_QUERY = 8
    FIELD_RESPONSE = 9
    NODE_INFO = 10


@dataclass
class NodeInfo:
    """Information about a network node."""
    id: str
    address: Tuple[str, int]
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    coherence: float = 0.8
    entangled: bool = False
    entanglement_key: Optional[bytes] = None
    consciousness_level: int = 1  # 1-7 corresponding to bridge stages
    timeline_position: int = 0
    capabilities: Set[str] = field(default_factory=set)
    field_dimensions: Tuple[int, int, int] = (21, 21, 21)
    

@dataclass
class QuantumPacket:
    """Phi-quantum network packet."""
    packet_type: PacketType
    node_id: str
    timestamp: float
    payload: Any
    coherence: float = 0.8
    sequence: int = 0
    entanglement_id: Optional[str] = None
    signature: Optional[bytes] = None


class PhiQuantumNetwork:
    """
    Distributed quantum field synchronization system.
    
    This system creates a virtual quantum entanglement between nodes,
    allowing for synchronized phi-harmonic field evolution while
    preserving local quantum coherence.
    """
    
    def __init__(self, port: int = PHI_QUANTUM_PORT, node_id: Optional[str] = None):
        """
        Initialize the phi-quantum network.
        
        Args:
            port: Network port to use
            node_id: Optional node identifier (auto-generated if None)
        """
        # Node identification
        self.node_id = node_id or self._generate_node_id()
        self.port = port
        
        # Node registry
        self.nodes: Dict[str, NodeInfo] = {}
        self.entangled_nodes: Set[str] = set()
        
        # Network components
        self.socket = None
        self.running = False
        self.receive_thread = None
        self.sync_thread = None
        self.packet_handlers = {}
        self.packet_queue = queue.Queue()
        self.sequence_counter = 0
        
        # Phi-harmonic components
        self.coherence = 0.8
        self.consciousness_level = 1
        self.phi_timer = PhiTimer()
        self.timeline_position = 0
        self.timeline_markers = []
        
        # Field state and synchronization
        self.field_dimensions = (21, 21, 21)
        self.quantum_field = np.zeros(self.field_dimensions)
        self.field_lock = threading.RLock()
        self.last_sync_time = 0
        self.sync_interval = PHI_SYNC_INTERVAL
        
        # Create toroidal memory for network state
        try:
            self.memory = create_toroidal_memory("network_state", 34)
        except Exception:
            # Fallback simple dict if toroidal memory not available
            self.memory = {}
        
        # Register packet handlers
        self._register_packet_handlers()
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID with phi-harmonic properties."""
        # Use a combination of machine info and phi values
        host_info = socket.gethostname()
        timestamp = time.time()
        phi_factor = PHI * math.floor(timestamp % 1000)
        
        # Create hash with phi-weighting
        hash_input = f"{host_info}:{timestamp}:{phi_factor}"
        node_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Take a phi-weighted slice of the hash
        start = int(len(node_hash) * LAMBDA)
        length = int(len(node_hash) * LAMBDA * LAMBDA)
        slice_end = min(start + length, len(node_hash))
        
        return node_hash[start:slice_end]
    
    def _register_packet_handlers(self) -> None:
        """Register handlers for different packet types."""
        self.packet_handlers = {
            PacketType.HEARTBEAT: self._handle_heartbeat,
            PacketType.FIELD_SYNC: self._handle_field_sync,
            PacketType.ENTANGLEMENT_REQUEST: self._handle_entanglement_request,
            PacketType.ENTANGLEMENT_ACCEPT: self._handle_entanglement_accept,
            PacketType.CONSCIOUSNESS_SYNC: self._handle_consciousness_sync,
            PacketType.TIMELINE_MARKER: self._handle_timeline_marker,
            PacketType.COHERENCE_UPDATE: self._handle_coherence_update,
            PacketType.FIELD_QUERY: self._handle_field_query,
            PacketType.FIELD_RESPONSE: self._handle_field_response,
            PacketType.NODE_INFO: self._handle_node_info
        }
    
    def start(self, bind_address: str = '', discover: bool = True) -> None:
        """
        Start the phi-quantum network.
        
        Args:
            bind_address: Address to bind to (empty for all interfaces)
            discover: Whether to discover other nodes
        """
        if self.running:
            return
            
        try:
            # Create and bind socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind((bind_address, self.port))
            
            logger.info(f"Phi-quantum node {self.node_id} starting on port {self.port}")
            
            # Mark as running
            self.running = True
            
            # Start receiver thread
            self.receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            self.receive_thread.start()
            
            # Start packet processor thread
            self.processor_thread = threading.Thread(
                target=self._process_packets,
                daemon=True
            )
            self.processor_thread.start()
            
            # Start sync thread
            self.sync_thread = threading.Thread(
                target=self._sync_loop,
                daemon=True
            )
            self.sync_thread.start()
            
            # Initialize our quantum field with phi-harmonic pattern
            self._initialize_quantum_field()
            
            # Announce our presence
            if discover:
                self._send_node_info(broadcast=True)
                self._discover_nodes()
        
        except Exception as e:
            logger.error(f"Failed to start phi-quantum network: {e}")
            self.running = False
            if self.socket:
                self.socket.close()
                self.socket = None
    
    def stop(self) -> None:
        """Stop the phi-quantum network."""
        if not self.running:
            return
            
        logger.info(f"Stopping phi-quantum node {self.node_id}")
        
        # Set running flag to false to stop threads
        self.running = False
        
        # Close socket
        if self.socket:
            self.socket.close()
            self.socket = None
        
        # Wait for threads to terminate
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)
            
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
            
        logger.info(f"Phi-quantum node {self.node_id} stopped")
    
    def _receive_loop(self) -> None:
        """Background thread for receiving packets."""
        logger.debug("Receive loop starting")
        
        while self.running:
            try:
                # Set socket timeout to allow checking running flag
                self.socket.settimeout(0.5)
                
                try:
                    # Receive data
                    data, addr = self.socket.recvfrom(16384)
                    
                    # Parse packet
                    packet = self._parse_packet(data)
                    if packet:
                        # Add to processing queue
                        self.packet_queue.put((packet, addr))
                        
                except socket.timeout:
                    pass
            
            except Exception as e:
                if self.running:
                    logger.error(f"Error in receive loop: {e}")
                    time.sleep(1.0)  # Avoid tight error loop
        
        logger.debug("Receive loop terminated")
    
    def _process_packets(self) -> None:
        """Background thread for processing received packets."""
        logger.debug("Packet processor starting")
        
        while self.running:
            try:
                # Get packet from queue with timeout
                try:
                    packet, addr = self.packet_queue.get(timeout=0.5)
                    
                    # Skip our own packets
                    if packet.node_id == self.node_id:
                        continue
                    
                    # Update node info
                    self._update_node_info(packet, addr)
                    
                    # Process packet
                    handler = self.packet_handlers.get(packet.packet_type)
                    if handler:
                        handler(packet, addr)
                    
                    # Mark as processed
                    self.packet_queue.task_done()
                    
                except queue.Empty:
                    pass
            
            except Exception as e:
                if self.running:
                    logger.error(f"Error processing packet: {e}")
        
        logger.debug("Packet processor terminated")
    
    def _sync_loop(self) -> None:
        """Background thread for periodic synchronization."""
        logger.debug("Sync loop starting")
        
        while self.running:
            try:
                # Wait for next phi-harmonic pulse
                self.phi_timer.wait_for_next_pulse()
                
                # Check if it's time to sync
                current_time = time.time()
                if current_time - self.last_sync_time >= self.sync_interval:
                    # Perform synchronization
                    self._sync_quantum_field()
                    self._send_heartbeat()
                    
                    # Update sync time
                    self.last_sync_time = current_time
                
                # Purge stale nodes
                self._purge_stale_nodes()
                
                # Update our coherence
                self._update_local_coherence()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in sync loop: {e}")
                    time.sleep(1.0)  # Avoid tight error loop
        
        logger.debug("Sync loop terminated")
    
    def _parse_packet(self, data: bytes) -> Optional[QuantumPacket]:
        """Parse received packet data."""
        try:
            # Check minimum length
            if len(data) < 24:
                return None
                
            # Check magic number
            magic = int.from_bytes(data[0:4], byteorder='big')
            if magic != PHI_PACKET_MAGIC:
                return None
                
            # Parse header
            header_size = 24
            header = data[4:header_size]
            
            packet_type_val, sequence, timestamp_int = struct.unpack('!BxHd', header)
            
            # Get packet type
            try:
                packet_type = PacketType(packet_type_val)
            except ValueError:
                logger.warning(f"Unknown packet type: {packet_type_val}")
                return None
            
            # Parse payload
            try:
                payload_data = data[header_size:]
                payload_dict = json.loads(payload_data.decode('utf-8'))
                
                # Extract required fields
                node_id = payload_dict.get('node_id', '')
                coherence = payload_dict.get('coherence', 0.8)
                entanglement_id = payload_dict.get('entanglement_id', None)
                payload = payload_dict.get('payload', {})
                signature = payload_dict.get('signature', None)
                
                if signature:
                    # Convert signature from hex to bytes
                    signature = bytes.fromhex(signature)
                
                return QuantumPacket(
                    packet_type=packet_type,
                    node_id=node_id,
                    timestamp=timestamp_int,
                    sequence=sequence,
                    coherence=coherence,
                    entanglement_id=entanglement_id,
                    payload=payload,
                    signature=signature
                )
            
            except json.JSONDecodeError:
                logger.warning("Failed to decode packet payload")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to parse packet: {e}")
            return None
    
    def _create_packet(self, packet_type: PacketType, payload: Any) -> bytes:
        """Create a packet for transmission."""
        # Increment sequence number
        self.sequence_counter = (self.sequence_counter + 1) % 65536
        
        # Create header with magic number and packet type
        header = struct.pack(
            '!IBHQ', 
            PHI_PACKET_MAGIC,
            packet_type.value,
            self.sequence_counter,
            int(time.time() * 1000)  # Timestamp in milliseconds
        )
        
        # Create payload dictionary
        payload_dict = {
            'node_id': self.node_id,
            'coherence': self.coherence,
            'payload': payload
        }
        
        # Add entanglement ID if we're entangled
        if self.entangled_nodes:
            # Use first entanglement for now
            entangled_node = next(iter(self.entangled_nodes))
            node_info = self.nodes.get(entangled_node)
            if node_info and node_info.entanglement_key:
                # Create entanglement ID
                entanglement_id = hashlib.sha256(
                    f"{self.node_id}:{entangled_node}:{time.time()}".encode()
                ).hexdigest()
                payload_dict['entanglement_id'] = entanglement_id
                
                # Create signature with entanglement key
                signature_data = f"{self.node_id}:{entanglement_id}:{self.sequence_counter}".encode()
                signature = hmac.new(
                    node_info.entanglement_key, 
                    signature_data, 
                    hashlib.sha256
                ).hexdigest()
                payload_dict['signature'] = signature
        
        # Encode payload
        payload_bytes = json.dumps(payload_dict).encode('utf-8')
        
        # Combine header and payload
        return header + payload_bytes
    
    def _send_packet(self, packet_type: PacketType, payload: Any, 
                   addr: Optional[Tuple[str, int]] = None) -> bool:
        """
        Send a packet to a specific address or broadcast.
        
        Args:
            packet_type: Type of packet to send
            payload: Packet payload
            addr: Destination address (None for broadcast)
            
        Returns:
            True if packet was sent successfully
        """
        if not self.running or not self.socket:
            return False
            
        try:
            # Create packet
            packet_data = self._create_packet(packet_type, payload)
            
            if addr:
                # Send to specific address
                self.socket.sendto(packet_data, addr)
            else:
                # Broadcast
                broadcast_addr = ('255.255.255.255', self.port)
                self.socket.sendto(packet_data, broadcast_addr)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send packet: {e}")
            return False
    
    def _update_node_info(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Update node information from received packet."""
        node_id = packet.node_id
        
        if node_id == self.node_id:
            # Skip our own packets
            return
        
        # Check if we know this node
        if node_id in self.nodes:
            # Update existing node
            node = self.nodes[node_id]
            node.last_seen = time.time()
            node.coherence = packet.coherence
            
            # Update address if it changed
            if node.address != addr:
                node.address = addr
        else:
            # Create new node entry
            node = NodeInfo(
                id=node_id,
                address=addr,
                coherence=packet.coherence
            )
            self.nodes[node_id] = node
            
            logger.info(f"New node discovered: {node_id} at {addr}")
    
    def _purge_stale_nodes(self) -> None:
        """Remove nodes that haven't been seen recently."""
        current_time = time.time()
        stale_nodes = []
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_seen > PHI_ENTANGLEMENT_TIMEOUT:
                stale_nodes.append(node_id)
                
                # If entangled, remove entanglement
                if node_id in self.entangled_nodes:
                    self.entangled_nodes.remove(node_id)
                    
                    # Reduce coherence due to lost entanglement
                    self.coherence = max(0.5, self.coherence * LAMBDA)
                    
                    logger.info(f"Lost entanglement with node {node_id}")
        
        # Remove stale nodes
        for node_id in stale_nodes:
            del self.nodes[node_id]
            logger.info(f"Removed stale node {node_id}")
    
    def _discover_nodes(self) -> None:
        """Send discovery broadcast to find other nodes."""
        logger.info("Broadcasting node discovery")
        
        # Create node info payload
        node_info = {
            'id': self.node_id,
            'coherence': self.coherence,
            'consciousness_level': self.consciousness_level,
            'capabilities': list(self._get_capabilities()),
            'field_dimensions': self.field_dimensions,
            'timeline_position': self.timeline_position
        }
        
        # Send broadcast
        self._send_packet(PacketType.NODE_INFO, node_info)
    
    def _get_capabilities(self) -> Set[str]:
        """Get this node's capabilities."""
        capabilities = {"basic", "field_sync"}
        
        # Add phi-harmonic capabilities if available
        try:
            from cascade.phi_python_core import phi_function
            capabilities.add("phi_harmonic")
        except ImportError:
            pass
            
        # Add entanglement capability
        capabilities.add("entanglement")
        
        # Add consciousness bridge if available
        try:
            from cascade.core.consciousness_bridge import ConsciousnessBridge
            capabilities.add("consciousness_bridge")
        except ImportError:
            pass
            
        # Add timeline capability
        capabilities.add("timeline")
        
        return capabilities
    
    def _initialize_quantum_field(self) -> None:
        """Initialize quantum field with phi-harmonic pattern."""
        with self.field_lock:
            # Create coordinates
            x = np.linspace(-1, 1, self.field_dimensions[0])
            y = np.linspace(-1, 1, self.field_dimensions[1])
            z = np.linspace(-1, 1, self.field_dimensions[2])
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Calculate radius from center
            R = np.sqrt(X**2 + Y**2 + Z**2)
            
            # Create phi-harmonic field
            theta = np.arctan2(Y, X)
            phi = np.arccos(Z / (R + 1e-10))
            
            # Apply phi-harmonic patterns
            self.quantum_field = (
                np.sin(R * PHI * 5) * np.exp(-R * LAMBDA) + 
                np.sin(theta * PHI) * np.cos(phi * LAMBDA) * 0.5
            )
            
            # Normalize
            self.quantum_field = (
                self.quantum_field - np.min(self.quantum_field)
            ) / (
                np.max(self.quantum_field) - np.min(self.quantum_field)
            )
            
            logger.info(f"Quantum field initialized with dimensions {self.field_dimensions}")
    
    def _update_local_coherence(self) -> None:
        """Update local field coherence."""
        # Calculate field coherence metrics
        with self.field_lock:
            # Calculate gradients
            dx, dy, dz = np.gradient(self.quantum_field)
            grad_mag = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Calculate smoothness (higher is better)
            smoothness = 1.0 - np.mean(grad_mag) / np.max(self.quantum_field)
            
            # Calculate phi-resonance
            flat_field = self.quantum_field.flatten()
            phi_dists = np.abs(flat_field - PHI)
            lambda_dists = np.abs(flat_field - LAMBDA)
            min_dists = np.minimum(phi_dists, lambda_dists)
            phi_resonance = 1.0 - np.mean(min_dists) / PHI
            
            # Calculate energy balance
            energy = np.mean(self.quantum_field**2)
            energy_balance = 1.0 - abs(energy - LAMBDA)
            
            # Combine metrics with phi-weighted formula
            field_coherence = (
                smoothness * 0.5 +
                phi_resonance * 0.3 +
                energy_balance * 0.2
            )
            
            # Update coherence with smooth transition
            target_coherence = (
                0.7 * self.coherence + 
                0.3 * field_coherence
            )
            
            # Apply consciousness level modifier
            consciousness_factor = LAMBDA + (1 - LAMBDA) * (self.consciousness_level / 7)
            target_coherence *= consciousness_factor
            
            # Factor in entangled nodes
            if self.entangled_nodes:
                entangled_coherence = 0.0
                count = 0
                
                for node_id in self.entangled_nodes:
                    if node_id in self.nodes:
                        entangled_coherence += self.nodes[node_id].coherence
                        count += 1
                
                if count > 0:
                    avg_entangled = entangled_coherence / count
                    
                    # Phi-weighted blend with entangled coherence
                    target_coherence = (
                        target_coherence * 0.7 +
                        avg_entangled * 0.3
                    )
            
            # Update coherence
            self.coherence = target_coherence
            
            # Also update global phi-coherence if available
            try:
                set_phi_coherence(self.coherence)
            except:
                pass
    
    def _sync_quantum_field(self) -> None:
        """Synchronize quantum field with entangled nodes."""
        # Only sync if we have entangled nodes
        if not self.entangled_nodes:
            return
            
        with self.field_lock:
            # Prepare field data
            # We don't send the whole field to reduce bandwidth,
            # instead we send a downsampled version
            
            # Determine downsampling factor based on dimensions
            downsample = max(1, min(self.field_dimensions) // 8)
            
            # Downsample field
            downsampled = self.quantum_field[::downsample, ::downsample, ::downsample]
            
            # Convert to list for JSON serialization
            field_data = downsampled.tolist()
            
            # Create field sync payload
            payload = {
                'field_data': field_data,
                'downsample_factor': downsample,
                'dimensions': self.field_dimensions,
                'timestamp': time.time(),
                'coherence': self.coherence,
                'consciousness_level': self.consciousness_level
            }
            
            # Send to all entangled nodes
            for node_id in self.entangled_nodes:
                if node_id in self.nodes:
                    self._send_packet(
                        PacketType.FIELD_SYNC,
                        payload,
                        self.nodes[node_id].address
                    )
    
    def _handle_heartbeat(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle heartbeat packet."""
        # Update node's last seen time
        if packet.node_id in self.nodes:
            self.nodes[packet.node_id].last_seen = time.time()
            self.nodes[packet.node_id].coherence = packet.coherence
    
    def _handle_field_sync(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle field synchronization packet."""
        # Only process if from entangled node
        if packet.node_id not in self.entangled_nodes:
            return
            
        payload = packet.payload
        
        # Extract field data
        field_data = payload.get('field_data')
        downsample_factor = payload.get('downsample_factor', 1)
        dimensions = payload.get('dimensions')
        node_coherence = payload.get('coherence', 0.8)
        
        if not field_data or not dimensions:
            return
            
        try:
            # Convert to numpy array
            remote_field = np.array(field_data)
            
            # Validate dimensions
            if remote_field.shape[0] * downsample_factor > self.field_dimensions[0] or \
               remote_field.shape[1] * downsample_factor > self.field_dimensions[1] or \
               remote_field.shape[2] * downsample_factor > self.field_dimensions[2]:
                logger.warning(f"Received field too large from {packet.node_id}")
                return
                
            # Upsample field to match dimensions
            if downsample_factor > 1:
                # Create upsampled field
                from scipy.ndimage import zoom
                upsampled = zoom(remote_field, (downsample_factor, downsample_factor, downsample_factor))
                
                # Ensure upsampled field isn't larger than our field
                slice_x = slice(0, min(upsampled.shape[0], self.field_dimensions[0]))
                slice_y = slice(0, min(upsampled.shape[1], self.field_dimensions[1]))
                slice_z = slice(0, min(upsampled.shape[2], self.field_dimensions[2]))
                
                remote_field = upsampled[slice_x, slice_y, slice_z]
            
            # Pad to match dimensions if needed
            if remote_field.shape != self.field_dimensions:
                padded = np.zeros(self.field_dimensions)
                slice_x = slice(0, min(remote_field.shape[0], self.field_dimensions[0]))
                slice_y = slice(0, min(remote_field.shape[1], self.field_dimensions[1]))
                slice_z = slice(0, min(remote_field.shape[2], self.field_dimensions[2]))
                
                padded[slice_x, slice_y, slice_z] = remote_field[slice_x, slice_y, slice_z]
                remote_field = padded
            
            # Blend fields using phi-harmonic mixing
            with self.field_lock:
                # Calculate weighting factors
                self_weight = max(0.5, (self.coherence / (self.coherence + node_coherence))) * LAMBDA
                remote_weight = 1.0 - self_weight
                
                # Apply phi-harmonic blending
                self.quantum_field = (
                    self.quantum_field * self_weight +
                    remote_field * remote_weight
                )
                
                # Apply phi-resonance enhancement to maintain coherence
                self._enhance_phi_resonance()
                
                logger.debug(f"Synchronized field with node {packet.node_id}, " + 
                           f"weights: local={self_weight:.2f}, remote={remote_weight:.2f}")
        
        except Exception as e:
            logger.error(f"Error processing field sync: {e}")
    
    def _enhance_phi_resonance(self) -> None:
        """Enhance phi-harmonic resonance in the quantum field."""
        # Find points close to PHI values
        mask_phi = np.abs(self.quantum_field - PHI) < 0.05
        mask_lambda = np.abs(self.quantum_field - LAMBDA) < 0.05
        
        # Enhance these points
        if np.any(mask_phi):
            self.quantum_field[mask_phi] = PHI
            
        if np.any(mask_lambda):
            self.quantum_field[mask_lambda] = LAMBDA
    
    def _handle_entanglement_request(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle entanglement request packet."""
        node_id = packet.node_id
        
        # Auto-accept for now
        if node_id in self.nodes:
            logger.info(f"Accepting entanglement request from {node_id}")
            
            # Generate entanglement key
            key = self._generate_entanglement_key(node_id)
            
            # Store in node info
            self.nodes[node_id].entanglement_key = key
            self.nodes[node_id].entangled = True
            
            # Add to entangled nodes
            self.entangled_nodes.add(node_id)
            
            # Send acceptance
            payload = {
                'accepted': True,
                'key_confirmation': self._get_key_confirmation(key)
            }
            
            self._send_packet(
                PacketType.ENTANGLEMENT_ACCEPT,
                payload,
                self.nodes[node_id].address
            )
            
            logger.info(f"Established quantum entanglement with node {node_id}")
    
    def _handle_entanglement_accept(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle entanglement acceptance packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        if node_id in self.nodes and payload.get('accepted', False):
            # Retrieve confirmation
            key_confirmation = payload.get('key_confirmation')
            
            if not key_confirmation:
                logger.warning(f"Missing key confirmation from {node_id}")
                return
                
            # Generate our key
            key = self._generate_entanglement_key(node_id)
            
            # Verify confirmation
            if self._get_key_confirmation(key) == key_confirmation:
                # Store key and mark as entangled
                self.nodes[node_id].entanglement_key = key
                self.nodes[node_id].entangled = True
                
                # Add to entangled nodes
                self.entangled_nodes.add(node_id)
                
                logger.info(f"Confirmed quantum entanglement with node {node_id}")
            else:
                logger.warning(f"Key confirmation mismatch with {node_id}")
    
    def _generate_entanglement_key(self, node_id: str) -> bytes:
        """Generate a shared entanglement key with another node."""
        # Generate key based on node IDs and phi values
        key_input = "".join(sorted([self.node_id, node_id]))
        key_input += f":{PHI}:{LAMBDA}:{PHI_PHI}"
        
        # Create key using SHA-256
        return hashlib.sha256(key_input.encode()).digest()
    
    def _get_key_confirmation(self, key: bytes) -> str:
        """Generate a confirmation code for an entanglement key."""
        # Take only part of the key for confirmation
        partial_key = key[:16]
        return hashlib.md5(partial_key).hexdigest()
    
    def _handle_consciousness_sync(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle consciousness synchronization packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        # Only process from entangled nodes
        if node_id not in self.entangled_nodes:
            return
            
        # Extract consciousness level
        consciousness_level = payload.get('consciousness_level', 1)
        frequency = payload.get('frequency', 432)
        
        if node_id in self.nodes:
            # Update node's consciousness level
            self.nodes[node_id].consciousness_level = consciousness_level
            
            # If node is at a higher level, consider advancing
            if consciousness_level > self.consciousness_level:
                # Probability of advancing increases with coherence
                advance_prob = self.coherence * LAMBDA
                
                if random.random() < advance_prob:
                    self.advance_consciousness()
                    logger.info(f"Advanced to consciousness level {self.consciousness_level} " +
                              f"in resonance with node {node_id}")
    
    def _handle_timeline_marker(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle timeline marker packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        # Only process from entangled nodes
        if node_id not in self.entangled_nodes:
            return
            
        # Extract timeline position
        timeline_position = payload.get('position', 0)
        marker_id = payload.get('marker_id')
        marker_data = payload.get('marker_data', {})
        
        if not marker_id:
            return
            
        # Store in our timeline markers
        self.timeline_markers.append({
            'id': marker_id,
            'node_id': node_id,
            'position': timeline_position,
            'timestamp': time.time(),
            'data': marker_data
        })
        
        # Limit timeline markers
        if len(self.timeline_markers) > 42:  # Phi-significant number
            self.timeline_markers = self.timeline_markers[-42:]
            
        # Update node's timeline position
        if node_id in self.nodes:
            self.nodes[node_id].timeline_position = timeline_position
    
    def _handle_coherence_update(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle coherence update packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        # Only process from entangled nodes
        if node_id not in self.entangled_nodes:
            return
            
        # Extract coherence
        node_coherence = payload.get('coherence', 0.8)
        
        if node_id in self.nodes:
            # Update node's coherence
            self.nodes[node_id].coherence = node_coherence
    
    def _handle_field_query(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle field query packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        # Extract query details
        query_id = payload.get('query_id')
        query_type = payload.get('query_type')
        
        if not query_id or not query_type:
            return
            
        # Process query
        result = None
        
        if query_type == 'dimensions':
            result = {'dimensions': self.field_dimensions}
        elif query_type == 'coherence':
            result = {'coherence': self.coherence}
        elif query_type == 'consciousness':
            result = {'level': self.consciousness_level}
        elif query_type == 'field_snapshot':
            # Return a low-resolution snapshot
            downsample = max(2, min(self.field_dimensions) // 5)
            with self.field_lock:
                snapshot = self.quantum_field[::downsample, ::downsample, ::downsample].tolist()
            result = {
                'snapshot': snapshot,
                'downsample': downsample
            }
        
        # Send response if we have a result
        if result and node_id in self.nodes:
            response = {
                'query_id': query_id,
                'query_type': query_type,
                'result': result
            }
            
            self._send_packet(
                PacketType.FIELD_RESPONSE,
                response,
                self.nodes[node_id].address
            )
    
    def _handle_field_response(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle field response packet."""
        # Store response in memory
        try:
            self.memory.put(f"response_{packet.payload.get('query_id')}", packet.payload)
        except:
            # Fallback if toroidal memory not available
            if isinstance(self.memory, dict):
                self.memory[f"response_{packet.payload.get('query_id')}"] = packet.payload
    
    def _handle_node_info(self, packet: QuantumPacket, addr: Tuple[str, int]) -> None:
        """Handle node info packet."""
        node_id = packet.node_id
        payload = packet.payload
        
        if node_id == self.node_id:
            # Skip our own packets
            return
            
        # Extract node info
        consciousness_level = payload.get('consciousness_level', 1)
        capabilities = set(payload.get('capabilities', []))
        field_dimensions = tuple(payload.get('dimensions', (0, 0, 0)))
        timeline_position = payload.get('timeline_position', 0)
        
        # Update or create node info
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.consciousness_level = consciousness_level
            node.capabilities = capabilities
            node.field_dimensions = field_dimensions
            node.timeline_position = timeline_position
        else:
            node = NodeInfo(
                id=node_id,
                address=addr,
                coherence=packet.coherence,
                consciousness_level=consciousness_level,
                capabilities=capabilities,
                field_dimensions=field_dimensions,
                timeline_position=timeline_position
            )
            self.nodes[node_id] = node
            
            logger.info(f"Discovered node {node_id} at {addr} with capabilities: {capabilities}")
        
        # Respond with our info
        self._send_node_info(addr=addr)
        
        # Consider requesting entanglement
        if "entanglement" in capabilities and node_id not in self.entangled_nodes:
            # Wait a short time to avoid flooding
            time.sleep(1.0)
            
            self._request_entanglement(node_id)
    
    def _send_node_info(self, addr: Optional[Tuple[str, int]] = None, broadcast: bool = False) -> None:
        """Send node info to a specific address or broadcast."""
        # Create node info payload
        node_info = {
            'id': self.node_id,
            'coherence': self.coherence,
            'consciousness_level': self.consciousness_level,
            'capabilities': list(self._get_capabilities()),
            'dimensions': self.field_dimensions,
            'timeline_position': self.timeline_position
        }
        
        if broadcast:
            # Send broadcast
            self._send_packet(PacketType.NODE_INFO, node_info)
        elif addr:
            # Send to specific address
            self._send_packet(PacketType.NODE_INFO, node_info, addr)
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to entangled nodes."""
        # Only send if we have entangled nodes
        if not self.entangled_nodes:
            return
        
        # Create heartbeat payload
        heartbeat = {
            'coherence': self.coherence,
            'timestamp': time.time(),
            'consciousness_level': self.consciousness_level
        }
        
        # Send to all entangled nodes
        for node_id in self.entangled_nodes:
            if node_id in self.nodes:
                self._send_packet(
                    PacketType.HEARTBEAT,
                    heartbeat,
                    self.nodes[node_id].address
                )
    
    def _request_entanglement(self, node_id: str) -> bool:
        """
        Request quantum entanglement with another node.
        
        Args:
            node_id: ID of node to entangle with
            
        Returns:
            True if request was sent
        """
        if node_id not in self.nodes:
            return False
            
        logger.info(f"Requesting quantum entanglement with node {node_id}")
        
        # Create request payload
        payload = {
            'requester': self.node_id,
            'coherence': self.coherence,
            'capabilities': list(self._get_capabilities()),
            'timestamp': time.time()
        }
        
        # Send request
        return self._send_packet(
            PacketType.ENTANGLEMENT_REQUEST,
            payload,
            self.nodes[node_id].address
        )
    
    def request_field_info(self, node_id: str, query_type: str) -> str:
        """
        Request field information from another node.
        
        Args:
            node_id: Node to query
            query_type: Type of query (dimensions, coherence, consciousness, field_snapshot)
            
        Returns:
            Query ID for retrieving results
        """
        if node_id not in self.nodes:
            return ""
            
        # Generate query ID
        query_id = f"{int(time.time())}_{self.node_id}_{node_id}"
        
        # Create query payload
        payload = {
            'query_id': query_id,
            'query_type': query_type,
            'timestamp': time.time()
        }
        
        # Send query
        success = self._send_packet(
            PacketType.FIELD_QUERY,
            payload,
            self.nodes[node_id].address
        )
        
        if success:
            return query_id
        return ""
    
    def get_query_result(self, query_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Get result of a previous query.
        
        Args:
            query_id: Query ID from request_field_info
            timeout: Timeout in seconds
            
        Returns:
            Query result or None if not available
        """
        start_time = time.time()
        
        # Wait for response
        while time.time() - start_time < timeout:
            # Check memory for response
            try:
                result = self.memory.get(f"response_{query_id}")
                if result:
                    return result
            except:
                # Fallback if toroidal memory not available
                if isinstance(self.memory, dict):
                    result = self.memory.get(f"response_{query_id}")
                    if result:
                        return result
            
            # Sleep briefly
            time.sleep(0.1)
        
        return None
    
    def advance_consciousness(self) -> int:
        """
        Advance to next consciousness level.
        
        Returns:
            New consciousness level
        """
        if self.consciousness_level < 7:
            self.consciousness_level += 1
            
            # Broadcast to entangled nodes
            self._send_consciousness_sync()
            
            logger.info(f"Advanced to consciousness level {self.consciousness_level}")
        
        return self.consciousness_level
    
    def set_consciousness_level(self, level: int) -> int:
        """
        Set consciousness level.
        
        Args:
            level: New level (1-7)
            
        Returns:
            Updated level
        """
        if 1 <= level <= 7:
            self.consciousness_level = level
            
            # Broadcast to entangled nodes
            self._send_consciousness_sync()
            
            logger.info(f"Set consciousness level to {self.consciousness_level}")
        
        return self.consciousness_level
    
    def _send_consciousness_sync(self) -> None:
        """Send consciousness sync to entangled nodes."""
        # Only send if we have entangled nodes
        if not self.entangled_nodes:
            return
        
        # Get frequency for current level
        frequency = PHI_FREQUENCIES[self.consciousness_level - 1]
        
        # Create sync payload
        payload = {
            'consciousness_level': self.consciousness_level,
            'frequency': frequency,
            'timestamp': time.time()
        }
        
        # Send to all entangled nodes
        for node_id in self.entangled_nodes:
            if node_id in self.nodes:
                self._send_packet(
                    PacketType.CONSCIOUSNESS_SYNC,
                    payload,
                    self.nodes[node_id].address
                )
    
    def create_timeline_marker(self, marker_data: Dict[str, Any]) -> str:
        """
        Create a timeline marker and broadcast to entangled nodes.
        
        Args:
            marker_data: Marker data
            
        Returns:
            Marker ID
        """
        # Generate marker ID
        marker_id = f"marker_{int(time.time())}_{self.node_id}"
        
        # Increment timeline position
        self.timeline_position += 1
        
        # Create marker
        marker = {
            'id': marker_id,
            'position': self.timeline_position,
            'timestamp': time.time(),
            'data': marker_data
        }
        
        # Add to our timeline
        self.timeline_markers.append(marker)
        
        # Broadcast to entangled nodes
        payload = {
            'marker_id': marker_id,
            'position': self.timeline_position,
            'marker_data': marker_data
        }
        
        # Send to all entangled nodes
        for node_id in self.entangled_nodes:
            if node_id in self.nodes:
                self._send_packet(
                    PacketType.TIMELINE_MARKER,
                    payload,
                    self.nodes[node_id].address
                )
        
        return marker_id
    
    def get_field_value(self, x: int, y: int, z: int) -> float:
        """
        Get value from quantum field at specified coordinates.
        
        Args:
            x, y, z: Coordinates
            
        Returns:
            Field value
        """
        with self.field_lock:
            if (0 <= x < self.field_dimensions[0] and
                0 <= y < self.field_dimensions[1] and
                0 <= z < self.field_dimensions[2]):
                return float(self.quantum_field[x, y, z])
        
        return 0.0
    
    def set_field_value(self, x: int, y: int, z: int, value: float) -> None:
        """
        Set value in quantum field at specified coordinates.
        
        Args:
            x, y, z: Coordinates
            value: New value
        """
        with self.field_lock:
            if (0 <= x < self.field_dimensions[0] and
                0 <= y < self.field_dimensions[1] and
                0 <= z < self.field_dimensions[2]):
                self.quantum_field[x, y, z] = value
    
    def apply_field_transformation(self, transformation_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Apply a transformation function to the quantum field.
        
        Args:
            transformation_func: Function that takes field and returns transformed field
        """
        with self.field_lock:
            # Apply transformation
            self.quantum_field = transformation_func(self.quantum_field)
            
            # Ensure field remains in valid range
            self.quantum_field = np.clip(self.quantum_field, 0.0, 1.0)
    
    def phi_harmonic_transform(self) -> None:
        """Apply a phi-harmonic transformation to the field."""
        with self.field_lock:
            # Create coordinates
            x = np.linspace(-1, 1, self.field_dimensions[0])
            y = np.linspace(-1, 1, self.field_dimensions[1])
            z = np.linspace(-1, 1, self.field_dimensions[2])
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Calculate radius from center
            R = np.sqrt(X**2 + Y**2 + Z**2)
            
            # Create phi-harmonic transformation
            transform = np.sin(R * PHI * 3 + time.time() * LAMBDA) * np.exp(-R)
            
            # Apply transformation with phi-weighted blending
            self.quantum_field = self.quantum_field * LAMBDA + transform * (1.0 - LAMBDA)
            
            # Ensure field remains in valid range
            self.quantum_field = np.clip(self.quantum_field, 0.0, 1.0)
            
            logger.debug("Applied phi-harmonic transformation to quantum field")
    
    def get_node_info(self, node_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get information about connected nodes.
        
        Args:
            node_id: Specific node ID or None for all nodes
            
        Returns:
            Node information
        """
        if node_id:
            # Return specific node
            if node_id in self.nodes:
                node = self.nodes[node_id]
                return {
                    'id': node.id,
                    'address': node.address,
                    'coherence': node.coherence,
                    'entangled': node.entangled,
                    'consciousness_level': node.consciousness_level,
                    'capabilities': list(node.capabilities),
                    'timeline_position': node.timeline_position,
                    'field_dimensions': node.field_dimensions,
                    'first_seen': node.first_seen,
                    'last_seen': node.last_seen
                }
            return {}
        else:
            # Return all nodes
            return [
                {
                    'id': node.id,
                    'address': node.address,
                    'coherence': node.coherence,
                    'entangled': node.entangled,
                    'consciousness_level': node.consciousness_level,
                    'capabilities': list(node.capabilities),
                    'timeline_position': node.timeline_position,
                    'field_dimensions': node.field_dimensions,
                    'first_seen': node.first_seen,
                    'last_seen': node.last_seen
                }
                for node in self.nodes.values()
            ]
    
    def get_entangled_nodes(self) -> List[str]:
        """Get list of entangled node IDs."""
        return list(self.entangled_nodes)
    
    def get_timeline_markers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get timeline markers.
        
        Args:
            limit: Maximum number of markers to return
            
        Returns:
            List of timeline markers
        """
        if limit:
            return self.timeline_markers[-limit:]
        return self.timeline_markers


class PhiQuantumField:
    """
    Higher-level abstraction for working with distributed quantum fields.
    
    This class provides a simplified interface for working with the phi-quantum
    network and performing field operations with proper coherence handling.
    """
    
    def __init__(self, dimensions: Tuple[int, int, int] = (21, 21, 21), port: int = PHI_QUANTUM_PORT):
        """
        Initialize phi-quantum field.
        
        Args:
            dimensions: Field dimensions
            port: Network port
        """
        # Create network
        self.network = PhiQuantumNetwork(port=port)
        self.network.field_dimensions = dimensions
        
        # Timeline tracking
        self.timeline_positions = {}
        
        # Event callbacks
        self.on_entanglement = None
        self.on_field_sync = None
        self.on_consciousness_change = None
    
    def start(self, bind_address: str = '', discover: bool = True) -> None:
        """
        Start the quantum field network.
        
        Args:
            bind_address: Network address to bind to
            discover: Whether to discover other nodes
        """
        self.network.start(bind_address, discover)
    
    def stop(self) -> None:
        """Stop the quantum field network."""
        self.network.stop()
    
    @phi_function
    def apply_transformation(self, operation: str, params: Optional[Dict[str, Any]] = None) -> float:
        """
        Apply a transformation to the quantum field.
        
        Args:
            operation: Type of transformation
            params: Optional parameters
            
        Returns:
            Field coherence after transformation
        """
        if not params:
            params = {}
            
        # Apply different transformations based on operation type
        if operation == "phi_wave":
            # Apply phi-harmonic wave
            frequency = params.get("frequency", 5.0)
            amplitude = params.get("amplitude", 0.5)
            
            def transform(field):
                # Create coordinates
                x = np.linspace(-1, 1, field.shape[0])
                y = np.linspace(-1, 1, field.shape[1])
                z = np.linspace(-1, 1, field.shape[2])
                
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                R = np.sqrt(X**2 + Y**2 + Z**2)
                
                # Create phi-harmonic wave
                wave = np.sin(R * PHI * frequency) * amplitude * np.exp(-R * LAMBDA)
                
                # Apply with phi-weighted blending
                return field * (1.0 - LAMBDA) + wave * LAMBDA
                
            self.network.apply_field_transformation(transform)
            
        elif operation == "toroidal_flow":
            # Apply toroidal flow pattern
            major_radius = params.get("major_radius", PHI * 0.4)
            minor_radius = params.get("minor_radius", LAMBDA * 0.3)
            
            def transform(field):
                # Create coordinates
                x = np.linspace(-1, 1, field.shape[0])
                y = np.linspace(-1, 1, field.shape[1])
                z = np.linspace(-1, 1, field.shape[2])
                
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                
                # Create toroidal shape
                distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - major_radius)**2 + Z**2)
                torus_distance = distance_from_ring / minor_radius
                
                # Create torus pattern
                theta = np.arctan2(Y, X)
                phi = np.arctan2(Z, np.sqrt(X**2 + Y**2) - major_radius)
                
                torus_pattern = (
                    np.sin(theta * PHI + phi * LAMBDA) * 
                    np.exp(-torus_distance**2)
                )
                
                # Apply with phi-weighted blending
                return field * 0.7 + torus_pattern * 0.3
                
            self.network.apply_field_transformation(transform)
            
        elif operation == "consciousness_resonance":
            # Apply consciousness frequency resonance
            level = params.get("level", self.network.consciousness_level)
            frequency = PHI_FREQUENCIES[level - 1] if 1 <= level <= 7 else 432
            
            def transform(field):
                # Create coordinates
                x = np.linspace(-1, 1, field.shape[0])
                y = np.linspace(-1, 1, field.shape[1])
                z = np.linspace(-1, 1, field.shape[2])
                
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                R = np.sqrt(X**2 + Y**2 + Z**2)
                
                # Create frequency pattern
                freq_factor = frequency / 1000.0
                pattern = np.sin(R * freq_factor * PHI * 10) * np.exp(-R)
                
                # Apply with level-dependent blending
                blend_factor = level / 7.0
                return field * (1.0 - blend_factor * 0.3) + pattern * (blend_factor * 0.3)
                
            self.network.apply_field_transformation(transform)
            
        elif operation == "timeline_shift":
            # Apply timeline shift
            position = params.get("position", self.network.timeline_position)
            amplitude = params.get("amplitude", 0.3)
            
            if position in self.timeline_positions:
                # Blend with stored timeline position
                stored_field = self.timeline_positions[position]
                
                def transform(field):
                    return field * (1.0 - amplitude) + stored_field * amplitude
                    
                self.network.apply_field_transformation(transform)
            else:
                # Store current position
                with self.network.field_lock:
                    self.timeline_positions[self.network.timeline_position] = \
                        self.network.quantum_field.copy()
                        
                # Create timeline marker
                self.network.create_timeline_marker({
                    "operation": operation,
                    "params": params
                })
        
        else:
            # Unknown operation, apply generic phi-harmonic transform
            self.network.phi_harmonic_transform()
        
        # Return updated coherence
        return self.network.coherence
    
    @phi_function
    def advance_consciousness(self) -> int:
        """
        Advance consciousness level.
        
        Returns:
            New consciousness level
        """
        level = self.network.advance_consciousness()
        
        # Notify callback if set
        if self.on_consciousness_change:
            try:
                self.on_consciousness_change(level)
            except Exception as e:
                logger.error(f"Error in consciousness change callback: {e}")
        
        return level
    
    @phi_function
    def set_consciousness_level(self, level: int) -> int:
        """
        Set consciousness level.
        
        Args:
            level: New level (1-7)
            
        Returns:
            Updated level
        """
        level = self.network.set_consciousness_level(level)
        
        # Notify callback if set
        if self.on_consciousness_change:
            try:
                self.on_consciousness_change(level)
            except Exception as e:
                logger.error(f"Error in consciousness change callback: {e}")
        
        return level
    
    def get_field_coherence(self) -> float:
        """Get current field coherence."""
        return self.network.coherence
    
    def get_consciousness_level(self) -> int:
        """Get current consciousness level."""
        return self.network.consciousness_level
    
    def get_node_coherence(self, node_id: str) -> float:
        """
        Get coherence of a connected node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node coherence
        """
        if node_id in self.network.nodes:
            return self.network.nodes[node_id].coherence
        return 0.0
    
    def get_field_value(self, x: int, y: int, z: int) -> float:
        """
        Get value from quantum field at specified coordinates.
        
        Args:
            x, y, z: Coordinates
            
        Returns:
            Field value
        """
        return self.network.get_field_value(x, y, z)
    
    def set_field_value(self, x: int, y: int, z: int, value: float) -> None:
        """
        Set value in quantum field at specified coordinates.
        
        Args:
            x, y, z: Coordinates
            value: New value
        """
        self.network.set_field_value(x, y, z, value)
    
    def get_field_dimensions(self) -> Tuple[int, int, int]:
        """Get field dimensions."""
        return self.network.field_dimensions
    
    def get_connected_nodes(self) -> List[Dict[str, Any]]:
        """Get information about connected nodes."""
        return self.network.get_node_info()
    
    def get_entangled_nodes(self) -> List[str]:
        """Get list of entangled node IDs."""
        return self.network.get_entangled_nodes()
    
    def get_timeline_markers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get timeline markers.
        
        Args:
            limit: Maximum number of markers to return
            
        Returns:
            List of timeline markers
        """
        return self.network.get_timeline_markers(limit)
    
    def request_field_info(self, node_id: str, query_type: str) -> str:
        """
        Request field information from another node.
        
        Args:
            node_id: Node to query
            query_type: Type of query (dimensions, coherence, consciousness, field_snapshot)
            
        Returns:
            Query ID for retrieving results
        """
        return self.network.request_field_info(node_id, query_type)
    
    def get_query_result(self, query_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Get result of a previous query.
        
        Args:
            query_id: Query ID from request_field_info
            timeout: Timeout in seconds
            
        Returns:
            Query result or None if not available
        """
        return self.network.get_query_result(query_id, timeout)


async def async_phi_network(bind_address: str = '', port: int = PHI_QUANTUM_PORT) -> None:
    """
    Run phi-quantum network with asyncio event loop.
    
    Args:
        bind_address: Address to bind to
        port: Network port
    """
    # Create network
    network = PhiQuantumNetwork(port=port)
    
    try:
        # Start network
        network.start(bind_address)
        
        # Run until cancelled
        while True:
            await asyncio.sleep(1)
            
            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                node_count = len(network.nodes)
                entangled_count = len(network.entangled_nodes)
                logger.info(f"Phi-quantum network status: {node_count} nodes, " +
                          f"{entangled_count} entangled, coherence: {network.coherence:.4f}")
    
    except asyncio.CancelledError:
        # Stop network
        network.stop()
    
    except Exception as e:
        logger.error(f"Error in async phi network: {e}")
        network.stop()


def create_phi_quantum_field(dimensions: Tuple[int, int, int] = (21, 21, 21), 
                         port: int = PHI_QUANTUM_PORT) -> PhiQuantumField:
    """
    Create a phi-quantum field.
    
    Args:
        dimensions: Field dimensions
        port: Network port
        
    Returns:
        PhiQuantumField instance
    """
    field = PhiQuantumField(dimensions, port)
    
    logger.info(f"Created phi-quantum field with dimensions {dimensions}")
    
    return field


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Phi-Quantum Network")
    parser.add_argument("--port", type=int, default=PHI_QUANTUM_PORT, help="Network port")
    parser.add_argument("--bind", default="", help="Address to bind to")
    parser.add_argument("--dimensions", default="21,21,21", help="Field dimensions (comma-separated)")
    parser.add_argument("--mode", choices=["standalone", "client", "server"], default="standalone",
                       help="Operation mode")
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = tuple(map(int, args.dimensions.split(",")))
    
    if args.mode == "standalone":
        # Create and start phi-quantum field
        field = create_phi_quantum_field(dimensions, args.port)
        
        try:
            field.start(args.bind)
            
            print(f"Phi-quantum field running on port {args.port}")
            print("Press Ctrl+C to stop")
            
            # Main loop
            while True:
                time.sleep(1)
                
                # Periodically apply transformation
                if int(time.time()) % 30 == 0:
                    field.apply_transformation("phi_wave")
                    print(f"Applied phi wave, coherence: {field.get_field_coherence():.4f}")
                
                # Print status every 10 seconds
                if int(time.time()) % 10 == 0:
                    nodes = field.get_connected_nodes()
                    entangled = field.get_entangled_nodes()
                    print(f"Connected to {len(nodes)} nodes, {len(entangled)} entangled")
                    print(f"Consciousness level: {field.get_consciousness_level()}")
                    print(f"Field coherence: {field.get_field_coherence():.4f}")
                    
            
        except KeyboardInterrupt:
            print("Stopping phi-quantum field")
            field.stop()
            
    elif args.mode == "server":
        # Run asyncio server
        try:
            import asyncio
            
            print(f"Starting phi-quantum network server on port {args.port}")
            
            asyncio.run(async_phi_network(args.bind, args.port))
            
        except KeyboardInterrupt:
            print("Stopping phi-quantum network server")
            
    elif args.mode == "client":
        # Create and start phi-quantum field as client
        field = create_phi_quantum_field(dimensions, args.port)
        
        try:
            field.start(args.bind)
            
            print(f"Phi-quantum field client running on port {args.port}")
            print("Press Ctrl+C to stop")
            
            # Main loop
            while True:
                time.sleep(1)
                
                # Periodically apply transformation
                if int(time.time()) % 20 == 0:
                    field.apply_transformation("toroidal_flow")
                    print(f"Applied toroidal flow, coherence: {field.get_field_coherence():.4f}")
                
                # Advance consciousness every 60 seconds
                if int(time.time()) % 60 == 0:
                    level = field.get_consciousness_level()
                    if level < 7:
                        field.advance_consciousness()
                        print(f"Advanced to consciousness level {field.get_consciousness_level()}")
                
                # Print status every 10 seconds
                if int(time.time()) % 10 == 0:
                    nodes = field.get_connected_nodes()
                    entangled = field.get_entangled_nodes()
                    print(f"Connected to {len(nodes)} nodes, {len(entangled)} entangled")
                    print(f"Consciousness level: {field.get_consciousness_level()}")
                    print(f"Field coherence: {field.get_field_coherence():.4f}")
            
        except KeyboardInterrupt:
            print("Stopping phi-quantum field client")
            field.stop()