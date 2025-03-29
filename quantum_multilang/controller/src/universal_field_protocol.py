#!/usr/bin/env python3
"""
Universal Field Protocol (UFP)

This module defines the quantum field serialization protocol that maintains
field coherence across language boundaries in the Multi-Language Quantum
Field Architecture.
"""

import sys
import time
import struct
import pickle
import logging
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universal_field_protocol")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Import sacred constants
try:
    sys.path.append(str(project_root))
    import sacred_constants as sc
except ImportError:
    logger.warning("sacred_constants module not found. Using default values.")
    # Define fallback constants
    class sc:
        PHI = 1.618033988749895
        LAMBDA = 0.618033988749895
        PHI_PHI = 2.1784575679375995
        
        SACRED_FREQUENCIES = {
            'love': 528,
            'unity': 432,
            'cascade': 594,
            'truth': 672,
            'vision': 720,
            'oneness': 768,
        }

@dataclass
class QuantumFieldMessage:
    """
    Universal Field Protocol message format for quantum field data exchange
    between different language components.
    """
    field_data: np.ndarray
    frequency_name: str
    consciousness_level: float
    phi_coherence: float
    timestamp: float = field(default_factory=time.time)
    source_language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the message data after initialization."""
        if not isinstance(self.field_data, np.ndarray):
            raise TypeError("field_data must be a numpy array")
        
        if not isinstance(self.frequency_name, str):
            raise TypeError("frequency_name must be a string")
        
        if not isinstance(self.consciousness_level, (int, float)):
            raise TypeError("consciousness_level must be a number")
        
        if not isinstance(self.phi_coherence, (int, float)):
            raise TypeError("phi_coherence must be a number")
    
    def calculate_checksum(self) -> str:
        """Calculate a phi-weighted checksum of the field data."""
        # Create a hash object
        hasher = hashlib.sha256()
        
        # Update with field data
        data = self.field_data.tobytes()
        hasher.update(data)
        
        # Update with metadata
        hasher.update(self.frequency_name.encode())
        hasher.update(struct.pack("d", self.consciousness_level))
        hasher.update(struct.pack("d", self.phi_coherence))
        hasher.update(struct.pack("d", self.timestamp))
        hasher.update(self.source_language.encode())
        
        # Calculate hash
        return hasher.hexdigest()

class UFPSerializer:
    """
    Serializer for the Universal Field Protocol that maintains quantum field
    coherence across language boundaries.
    """
    
    # Magic number for UFP format (phi-based)
    MAGIC_NUMBER = b'UFP' + struct.pack("d", sc.PHI)
    
    # Protocol version
    VERSION = 1
    
    def __init__(self):
        """Initialize the serializer."""
        logger.debug("Initializing Universal Field Protocol serializer")
    
    def serialize(self, message: QuantumFieldMessage) -> bytes:
        """
        Serialize a quantum field message to bytes.
        
        Args:
            message: The QuantumFieldMessage to serialize
            
        Returns:
            A byte string containing the serialized message
        """
        # Calculate phi-based serialization parameters
        field_bytes = message.field_data.tobytes()
        field_shape = message.field_data.shape
        field_dtype = str(message.field_data.dtype)
        
        # Generate a checksum
        checksum = message.calculate_checksum()
        
        # Create header with fixed-size fields
        header = struct.pack(
            "!4s I d d d d",                       # Format string
            self.MAGIC_NUMBER,                    # Magic number
            self.VERSION,                         # Version
            message.consciousness_level,          # Consciousness level
            message.phi_coherence,                # Phi coherence
            message.timestamp,                    # Timestamp
            sc.PHI                                # PHI constant as validation
        )
        
        # Handle variable-sized fields
        # For frequency_name
        freq_bytes = message.frequency_name.encode('utf-8')
        freq_header = struct.pack("!I", len(freq_bytes))
        
        # For source_language
        lang_bytes = message.source_language.encode('utf-8')
        lang_header = struct.pack("!I", len(lang_bytes))
        
        # For field shape
        shape_bytes = pickle.dumps(field_shape)
        shape_header = struct.pack("!I", len(shape_bytes))
        
        # For field dtype
        dtype_bytes = field_dtype.encode('utf-8')
        dtype_header = struct.pack("!I", len(dtype_bytes))
        
        # For field data
        field_header = struct.pack("!Q", len(field_bytes))
        
        # For metadata
        metadata_bytes = pickle.dumps(message.metadata)
        metadata_header = struct.pack("!I", len(metadata_bytes))
        
        # For checksum
        checksum_bytes = checksum.encode('utf-8')
        checksum_header = struct.pack("!I", len(checksum_bytes))
        
        # Assemble the message
        serialized = (
            header +
            freq_header + freq_bytes +
            lang_header + lang_bytes +
            shape_header + shape_bytes +
            dtype_header + dtype_bytes +
            field_header + field_bytes +
            metadata_header + metadata_bytes +
            checksum_header + checksum_bytes
        )
        
        return serialized
    
    def deserialize(self, data: bytes) -> QuantumFieldMessage:
        """
        Deserialize bytes to a quantum field message.
        
        Args:
            data: The byte string to deserialize
            
        Returns:
            A QuantumFieldMessage object
        """
        # Check if we have enough data for the header
        if len(data) < 32:  # Minimum size for fixed header
            raise ValueError("Data too small to be a valid UFP message")
        
        # Unpack the fixed header
        offset = 0
        magic, version, consciousness, coherence, timestamp, phi = struct.unpack(
            "!4s I d d d d", data[offset:offset+32]
        )
        offset += 32
        
        # Verify magic number
        if magic != self.MAGIC_NUMBER:
            raise ValueError("Invalid UFP magic number")
        
        # Verify version
        if version != self.VERSION:
            raise ValueError(f"Unsupported UFP version: {version}")
        
        # Verify phi constant for additional validation
        if not np.isclose(phi, sc.PHI, rtol=1e-10):
            raise ValueError(f"Invalid phi constant in message: {phi}")
        
        # Unpack frequency_name
        freq_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        frequency_name = data[offset:offset+freq_len].decode('utf-8')
        offset += freq_len
        
        # Unpack source_language
        lang_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        source_language = data[offset:offset+lang_len].decode('utf-8')
        offset += lang_len
        
        # Unpack field shape
        shape_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        field_shape = pickle.loads(data[offset:offset+shape_len])
        offset += shape_len
        
        # Unpack field dtype
        dtype_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        field_dtype = data[offset:offset+dtype_len].decode('utf-8')
        offset += dtype_len
        
        # Unpack field data
        field_len = struct.unpack("!Q", data[offset:offset+8])[0]
        offset += 8
        field_bytes = data[offset:offset+field_len]
        offset += field_len
        
        # Convert bytes to numpy array
        field_data = np.frombuffer(field_bytes, dtype=np.dtype(field_dtype)).reshape(field_shape)
        
        # Unpack metadata
        metadata_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        metadata = pickle.loads(data[offset:offset+metadata_len])
        offset += metadata_len
        
        # Unpack checksum
        checksum_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        received_checksum = data[offset:offset+checksum_len].decode('utf-8')
        
        # Create the message
        message = QuantumFieldMessage(
            field_data=field_data,
            frequency_name=frequency_name,
            consciousness_level=consciousness,
            phi_coherence=coherence,
            timestamp=timestamp,
            source_language=source_language,
            metadata=metadata
        )
        
        # Verify checksum
        calculated_checksum = message.calculate_checksum()
        if calculated_checksum != received_checksum:
            raise ValueError("Checksum verification failed")
        
        return message
    
    def create_shared_memory_layout(self, message: QuantumFieldMessage) -> Tuple[bytes, Dict[str, Any]]:
        """
        Create a phi-harmonic memory layout for shared memory communication.
        
        Args:
            message: The QuantumFieldMessage to lay out in memory
            
        Returns:
            A tuple of (shared_memory_data, memory_map)
        """
        # Calculate phi-based memory alignment
        alignment = int(sc.PHI * 8)  # 8-byte base unit * phi
        
        # Create memory map
        memory_map = {}
        
        # Calculate offsets
        header_size = int(sc.PHI * 64)  # 64-byte base header * phi
        offset = 0
        
        # Magic number and version
        memory_map["magic"] = offset
        offset += 8
        
        # Fixed-size fields
        memory_map["consciousness_level"] = offset
        offset += 8
        
        memory_map["phi_coherence"] = offset
        offset += 8
        
        memory_map["timestamp"] = offset
        offset += 8
        
        memory_map["phi_constant"] = offset
        offset += 8
        
        # Skip to header_size with phi-based padding
        offset = header_size
        
        # String fields with phi-harmonic alignment
        offset = int(offset + (alignment - offset % alignment) % alignment)
        memory_map["frequency_name_offset"] = offset
        memory_map["frequency_name_length"] = len(message.frequency_name)
        offset += len(message.frequency_name) + 1  # +1 for null terminator
        
        offset = int(offset + (alignment - offset % alignment) % alignment)
        memory_map["source_language_offset"] = offset
        memory_map["source_language_length"] = len(message.source_language)
        offset += len(message.source_language) + 1  # +1 for null terminator
        
        # Field data with phi-harmonic alignment
        offset = int(offset + (alignment - offset % alignment) % alignment)
        memory_map["field_data_offset"] = offset
        memory_map["field_shape"] = message.field_data.shape
        memory_map["field_dtype"] = str(message.field_data.dtype)
        field_size = message.field_data.nbytes
        memory_map["field_size"] = field_size
        offset += field_size
        
        # Metadata with phi-harmonic alignment
        offset = int(offset + (alignment - offset % alignment) % alignment)
        memory_map["metadata_offset"] = offset
        metadata_bytes = pickle.dumps(message.metadata)
        memory_map["metadata_size"] = len(metadata_bytes)
        offset += len(metadata_bytes)
        
        # Total size with phi-harmonic padding
        total_size = int(offset + (alignment - offset % alignment) % alignment)
        memory_map["total_size"] = total_size
        
        # Create shared memory buffer
        shared_data = bytearray(total_size)
        
        # Write magic number and version
        struct.pack_into("!4s I", shared_data, memory_map["magic"],
                        self.MAGIC_NUMBER[:4], self.VERSION)
        
        # Write fixed-size fields
        struct.pack_into("!d", shared_data, memory_map["consciousness_level"],
                        message.consciousness_level)
        
        struct.pack_into("!d", shared_data, memory_map["phi_coherence"],
                        message.phi_coherence)
        
        struct.pack_into("!d", shared_data, memory_map["timestamp"],
                        message.timestamp)
        
        struct.pack_into("!d", shared_data, memory_map["phi_constant"],
                        sc.PHI)
        
        # Write string fields
        shared_data[memory_map["frequency_name_offset"]:
                   memory_map["frequency_name_offset"] + memory_map["frequency_name_length"]] = \
            message.frequency_name.encode('utf-8')
        
        shared_data[memory_map["source_language_offset"]:
                   memory_map["source_language_offset"] + memory_map["source_language_length"]] = \
            message.source_language.encode('utf-8')
        
        # Write field data
        shared_data[memory_map["field_data_offset"]:
                   memory_map["field_data_offset"] + memory_map["field_size"]] = \
            message.field_data.tobytes()
        
        # Write metadata
        shared_data[memory_map["metadata_offset"]:
                   memory_map["metadata_offset"] + memory_map["metadata_size"]] = \
            metadata_bytes
        
        return bytes(shared_data), memory_map
    
    def load_from_shared_memory(self, shared_data: bytes, memory_map: Dict[str, Any]) -> QuantumFieldMessage:
        """
        Load a QuantumFieldMessage from a phi-harmonic shared memory layout.
        
        Args:
            shared_data: The shared memory data
            memory_map: The memory layout map
            
        Returns:
            A QuantumFieldMessage object
        """
        # Verify magic number and version
        magic, version = struct.unpack_from("!4s I", shared_data, memory_map["magic"])
        
        if magic != self.MAGIC_NUMBER[:4]:
            raise ValueError("Invalid UFP magic number in shared memory")
        
        if version != self.VERSION:
            raise ValueError(f"Unsupported UFP version in shared memory: {version}")
        
        # Read fixed-size fields
        consciousness_level = struct.unpack_from("!d", shared_data, memory_map["consciousness_level"])[0]
        phi_coherence = struct.unpack_from("!d", shared_data, memory_map["phi_coherence"])[0]
        timestamp = struct.unpack_from("!d", shared_data, memory_map["timestamp"])[0]
        phi_constant = struct.unpack_from("!d", shared_data, memory_map["phi_constant"])[0]
        
        # Verify phi constant
        if not np.isclose(phi_constant, sc.PHI, rtol=1e-10):
            raise ValueError(f"Invalid phi constant in shared memory: {phi_constant}")
        
        # Read string fields
        frequency_name = shared_data[memory_map["frequency_name_offset"]:
                                   memory_map["frequency_name_offset"] + memory_map["frequency_name_length"]].decode('utf-8')
        
        source_language = shared_data[memory_map["source_language_offset"]:
                                    memory_map["source_language_offset"] + memory_map["source_language_length"]].decode('utf-8')
        
        # Read field data
        field_data_bytes = shared_data[memory_map["field_data_offset"]:
                                     memory_map["field_data_offset"] + memory_map["field_size"]]
        
        field_data = np.frombuffer(field_data_bytes, dtype=np.dtype(memory_map["field_dtype"])).reshape(memory_map["field_shape"])
        
        # Read metadata
        metadata_bytes = shared_data[memory_map["metadata_offset"]:
                                   memory_map["metadata_offset"] + memory_map["metadata_size"]]
        
        metadata = pickle.loads(metadata_bytes)
        
        # Create the message
        message = QuantumFieldMessage(
            field_data=field_data,
            frequency_name=frequency_name,
            consciousness_level=consciousness_level,
            phi_coherence=phi_coherence,
            timestamp=timestamp,
            source_language=source_language,
            metadata=metadata
        )
        
        return message

# Simple test function
def test_serialization():
    """Test the UFP serialization/deserialization process."""
    # Create a sample field
    width, height = 10, 10
    field_data = np.zeros((height, width), dtype=np.float32)
    
    # Fill with a simple pattern
    for y in range(height):
        for x in range(width):
            field_data[y, x] = np.sin(x/width * sc.PHI) * np.cos(y/height * sc.PHI)
    
    # Create a message
    message = QuantumFieldMessage(
        field_data=field_data,
        frequency_name="love",
        consciousness_level=sc.PHI,
        phi_coherence=0.95,
        source_language="python",
        metadata={"test": True}
    )
    
    # Create a serializer
    serializer = UFPSerializer()
    
    # Serialize
    serialized = serializer.serialize(message)
    
    # Deserialize
    deserialized = serializer.deserialize(serialized)
    
    # Check if the original and deserialized messages are equivalent
    assert np.array_equal(message.field_data, deserialized.field_data)
    assert message.frequency_name == deserialized.frequency_name
    assert message.consciousness_level == deserialized.consciousness_level
    assert message.phi_coherence == deserialized.phi_coherence
    assert message.source_language == deserialized.source_language
    assert message.metadata == deserialized.metadata
    
    # Test shared memory layout
    shared_data, memory_map = serializer.create_shared_memory_layout(message)
    
    # Load from shared memory
    loaded = serializer.load_from_shared_memory(shared_data, memory_map)
    
    # Check if the original and loaded messages are equivalent
    assert np.array_equal(message.field_data, loaded.field_data)
    assert message.frequency_name == loaded.frequency_name
    assert message.consciousness_level == loaded.consciousness_level
    assert message.phi_coherence == loaded.phi_coherence
    assert message.source_language == loaded.source_language
    assert message.metadata == loaded.metadata
    
    print("Serialization test passed!")

if __name__ == "__main__":
    # Run test
    test_serialization()