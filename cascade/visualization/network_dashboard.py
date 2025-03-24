"""
CASCADE⚡𓂧φ∞ Network Visualization Dashboard

This module provides a real-time monitoring dashboard for the distributed
quantum field network, showing node health, coherence metrics, and performance statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import time
import threading
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# Try to import dash for web-based dashboard
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    import plotly.express as px
    has_dash = True
except ImportError:
    has_dash = False
    logging.warning("Dash not available, web dashboard disabled")

# Import local modules
try:
    from cascade.phi_quantum_network import (
        PhiQuantumField,
        PhiQuantumNetwork,
        PHI, LAMBDA, PHI_PHI
    )
except ImportError:
    # Fallback constants
    PHI = 1.618033988749895
    LAMBDA = 0.618033988749895
    PHI_PHI = PHI ** PHI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("network_dashboard")


class NetworkMonitor:
    """Monitors network health, coherence, and performance metrics."""
    
    def __init__(self, quantum_field: 'PhiQuantumField'):
        """
        Initialize the network monitor.
        
        Args:
            quantum_field: PhiQuantumField instance to monitor
        """
        self.quantum_field = quantum_field
        
        # Monitoring data
        self.node_history = {}
        self.coherence_history = []
        self.performance_metrics = {
            'node_query_time': [],
            'visualization_time': [],
            'entanglement_requests': [],
            'network_latency': []
        }
        
        # Status flags
        self.running = False
        self.monitor_thread = None
        self.last_update_time = 0
        self.update_interval = 2.0  # seconds
        
        # Initialize history
        self._initialize_history()
    
    def _initialize_history(self) -> None:
        """Initialize monitoring history."""
        # Initialize coherence history
        self.coherence_history = [{
            'timestamp': time.time(),
            'local_coherence': self.quantum_field.get_field_coherence(),
            'consciousness_level': self.quantum_field.get_consciousness_level(),
            'node_count': 0,
            'entangled_count': 0,
            'avg_network_coherence': 0.0
        }]
    
    def start_monitoring(self) -> None:
        """Start monitoring the network."""
        if self.running:
            return
            
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Network monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring the network."""
        if not self.running:
            return
            
        # Set running flag to false to stop thread
        self.running = False
        
        # Wait for thread to terminate
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        logger.info("Network monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background thread for monitoring network status."""
        logger.debug("Monitoring loop starting")
        
        while self.running:
            try:
                # Check if it's time to update
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    # Update monitoring data
                    self._update_monitoring_data()
                    
                    # Update last update time
                    self.last_update_time = current_time
                
                # Sleep briefly
                time.sleep(0.5)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(1.0)  # Avoid tight error loop
        
        logger.debug("Monitoring loop terminated")
    
    def _update_monitoring_data(self) -> None:
        """Update network monitoring data."""
        try:
            # Measure performance
            start_time = time.time()
            
            # Get connected nodes
            nodes = self.quantum_field.get_connected_nodes()
            
            # Measure node query time
            node_query_time = time.time() - start_time
            self.performance_metrics['node_query_time'].append({
                'timestamp': start_time,
                'value': node_query_time
            })
            
            # Get entangled nodes
            entangled_nodes = self.quantum_field.get_entangled_nodes()
            
            # Get local coherence
            local_coherence = self.quantum_field.get_field_coherence()
            consciousness_level = self.quantum_field.get_consciousness_level()
            
            # Update node history
            for node in nodes:
                node_id = node['id']
                
                if node_id not in self.node_history:
                    # Initialize node history
                    self.node_history[node_id] = {
                        'first_seen': time.time(),
                        'coherence_history': [],
                        'consciousness_history': [],
                        'entanglement_history': []
                    }
                
                # Update node data
                self.node_history[node_id]['last_seen'] = time.time()
                self.node_history[node_id]['coherence_history'].append({
                    'timestamp': time.time(),
                    'value': node.get('coherence', 0.0)
                })
                self.node_history[node_id]['consciousness_history'].append({
                    'timestamp': time.time(),
                    'value': node.get('consciousness_level', 1)
                })
                self.node_history[node_id]['entanglement_history'].append({
                    'timestamp': time.time(),
                    'value': node_id in entangled_nodes
                })
                
                # Limit history length
                max_history = 100
                if len(self.node_history[node_id]['coherence_history']) > max_history:
                    self.node_history[node_id]['coherence_history'] = \
                        self.node_history[node_id]['coherence_history'][-max_history:]
                if len(self.node_history[node_id]['consciousness_history']) > max_history:
                    self.node_history[node_id]['consciousness_history'] = \
                        self.node_history[node_id]['consciousness_history'][-max_history:]
                if len(self.node_history[node_id]['entanglement_history']) > max_history:
                    self.node_history[node_id]['entanglement_history'] = \
                        self.node_history[node_id]['entanglement_history'][-max_history:]
            
            # Calculate average network coherence
            avg_coherence = 0.0
            if nodes:
                avg_coherence = sum(node.get('coherence', 0.0) for node in nodes) / len(nodes)
            
            # Update coherence history
            self.coherence_history.append({
                'timestamp': time.time(),
                'local_coherence': local_coherence,
                'consciousness_level': consciousness_level,
                'node_count': len(nodes),
                'entangled_count': len(entangled_nodes),
                'avg_network_coherence': avg_coherence
            })
            
            # Limit coherence history length
            max_coherence_history = 1000
            if len(self.coherence_history) > max_coherence_history:
                self.coherence_history = self.coherence_history[-max_coherence_history:]
            
            # Measure visualization time
            self.performance_metrics['visualization_time'].append({
                'timestamp': time.time(),
                'value': time.time() - start_time
            })
            
            # Limit performance metrics history
            max_metrics_history = 100
            for metric in self.performance_metrics:
                if len(self.performance_metrics[metric]) > max_metrics_history:
                    self.performance_metrics[metric] = \
                        self.performance_metrics[metric][-max_metrics_history:]
            
        except Exception as e:
            logger.error(f"Error updating monitoring data: {e}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get current network status.
        
        Returns:
            Network status data
        """
        # Get connected nodes
        nodes = self.quantum_field.get_connected_nodes()
        
        # Get entangled nodes
        entangled_nodes = self.quantum_field.get_entangled_nodes()
        
        # Get local coherence
        local_coherence = self.quantum_field.get_field_coherence()
        consciousness_level = self.quantum_field.get_consciousness_level()
        
        # Calculate network metrics
        avg_coherence = 0.0
        avg_consciousness = 0.0
        if nodes:
            avg_coherence = sum(node.get('coherence', 0.0) for node in nodes) / len(nodes)
            avg_consciousness = sum(node.get('consciousness_level', 1) for node in nodes) / len(nodes)
        
        # Calculate network health based on coherence and connectivity
        health_score = local_coherence * LAMBDA
        if nodes:
            health_score += avg_coherence * (1 - LAMBDA) * 0.5
            health_score += (len(entangled_nodes) / len(nodes)) * (1 - LAMBDA) * 0.5
        
        # Return status data
        return {
            'timestamp': time.time(),
            'local_coherence': local_coherence,
            'consciousness_level': consciousness_level,
            'node_count': len(nodes),
            'entangled_count': len(entangled_nodes),
            'avg_network_coherence': avg_coherence,
            'avg_consciousness_level': avg_consciousness,
            'network_health': health_score
        }
    
    def get_node_details(self, node_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get detailed information about nodes.
        
        Args:
            node_id: Specific node ID or None for all nodes
            
        Returns:
            Node details
        """
        if node_id:
            # Get specific node information
            if node_id in self.node_history:
                node_info = self.quantum_field.network.get_node_info(node_id)
                history = self.node_history[node_id]
                
                # Calculate uptime
                uptime = time.time() - history['first_seen']
                
                # Calculate average coherence
                coherence_values = [entry['value'] for entry in history['coherence_history']]
                avg_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
                
                # Calculate entanglement percentage
                entanglement_values = [entry['value'] for entry in history['entanglement_history']]
                entanglement_pct = sum(1 for v in entanglement_values if v) / len(entanglement_values) \
                                  if entanglement_values else 0.0
                
                return {
                    'id': node_id,
                    'address': node_info.get('address', ('', 0)),
                    'first_seen': history['first_seen'],
                    'last_seen': history['last_seen'],
                    'uptime': uptime,
                    'current_coherence': node_info.get('coherence', 0.0),
                    'avg_coherence': avg_coherence,
                    'consciousness_level': node_info.get('consciousness_level', 1),
                    'entangled': node_info.get('entangled', False),
                    'entanglement_percentage': entanglement_pct,
                    'capabilities': node_info.get('capabilities', []),
                    'field_dimensions': node_info.get('field_dimensions', (0, 0, 0)),
                    'timeline_position': node_info.get('timeline_position', 0),
                    'coherence_history': history['coherence_history'],
                    'consciousness_history': history['consciousness_history'],
                    'entanglement_history': history['entanglement_history']
                }
            
            return {}
        else:
            # Get all node details (summary)
            nodes = []
            
            for node_id in self.node_history:
                node_info = self.quantum_field.network.get_node_info(node_id)
                history = self.node_history[node_id]
                
                # Calculate uptime
                uptime = time.time() - history['first_seen']
                
                # Calculate average coherence
                coherence_values = [entry['value'] for entry in history['coherence_history']]
                avg_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
                
                nodes.append({
                    'id': node_id,
                    'address': node_info.get('address', ('', 0)),
                    'uptime': uptime,
                    'current_coherence': node_info.get('coherence', 0.0),
                    'avg_coherence': avg_coherence,
                    'consciousness_level': node_info.get('consciousness_level', 1),
                    'entangled': node_info.get('entangled', False),
                    'last_seen': history['last_seen']
                })
            
            return nodes
    
    def get_performance_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return self.performance_metrics
    
    def get_coherence_history(self) -> List[Dict[str, Any]]:
        """
        Get coherence history.
        
        Returns:
            Coherence history
        """
        return self.coherence_history


class NetworkDashboard:
    """Dashboard for visualizing network status and performance."""
    
    def __init__(self, monitor: NetworkMonitor):
        """
        Initialize the network dashboard.
        
        Args:
            monitor: NetworkMonitor instance
        """
        self.monitor = monitor
        self.fig = None
        self.axes = {}
        self.plots = {}
        self.update_interval = 2000  # ms
        self.web_dashboard = None
    
    def create_matplotlib_dashboard(self) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Create a matplotlib dashboard.
        
        Returns:
            Figure and axes dictionary
        """
        # Create figure
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        fig.canvas.manager.set_window_title("CASCADE⚡𓂧φ∞ Network Dashboard")
        
        # Create grid layout
        grid = gridspec.GridSpec(3, 3, figure=fig)
        
        # Create axes
        axes = {}
        
        # Network coherence plot
        axes['coherence'] = fig.add_subplot(grid[0, :2], facecolor='black')
        axes['coherence'].set_title("Network Coherence", color='white', fontsize=14)
        axes['coherence'].set_facecolor('black')
        axes['coherence'].spines['bottom'].set_color('white')
        axes['coherence'].spines['top'].set_color('white')
        axes['coherence'].spines['right'].set_color('white')
        axes['coherence'].spines['left'].set_color('white')
        axes['coherence'].tick_params(axis='x', colors='white')
        axes['coherence'].tick_params(axis='y', colors='white')
        axes['coherence'].set_ylim(0, 1.05)
        axes['coherence'].set_xlabel("Time", color='white')
        axes['coherence'].set_ylabel("Coherence", color='white')
        axes['coherence'].grid(True, color='white', alpha=0.2)
        
        # Mark phi-harmonic levels
        axes['coherence'].axhline(y=LAMBDA, color='gold', linestyle='--', alpha=0.5)
        axes['coherence'].axhline(y=1/PHI, color='teal', linestyle='--', alpha=0.5)
        
        # Network status plot (gauge)
        axes['status'] = fig.add_subplot(grid[0, 2], polar=True, facecolor='black')
        axes['status'].set_title("Network Health", color='white', fontsize=14)
        axes['status'].spines['polar'].set_color('white')
        axes['status'].tick_params(axis='x', colors='white')
        axes['status'].tick_params(axis='y', colors='white')
        
        # Node statistics plot
        axes['nodes'] = fig.add_subplot(grid[1, 0], facecolor='black')
        axes['nodes'].set_title("Connected Nodes", color='white', fontsize=14)
        axes['nodes'].set_facecolor('black')
        axes['nodes'].spines['bottom'].set_color('white')
        axes['nodes'].spines['top'].set_color('white')
        axes['nodes'].spines['right'].set_color('white')
        axes['nodes'].spines['left'].set_color('white')
        axes['nodes'].tick_params(axis='x', colors='white')
        axes['nodes'].tick_params(axis='y', colors='white')
        axes['nodes'].set_xlabel("Time", color='white')
        axes['nodes'].set_ylabel("Count", color='white')
        axes['nodes'].grid(True, color='white', alpha=0.2)
        
        # Consciousness level plot
        axes['consciousness'] = fig.add_subplot(grid[1, 1], facecolor='black')
        axes['consciousness'].set_title("Consciousness Levels", color='white', fontsize=14)
        axes['consciousness'].set_facecolor('black')
        axes['consciousness'].spines['bottom'].set_color('white')
        axes['consciousness'].spines['top'].set_color('white')
        axes['consciousness'].spines['right'].set_color('white')
        axes['consciousness'].spines['left'].set_color('white')
        axes['consciousness'].tick_params(axis='x', colors='white')
        axes['consciousness'].tick_params(axis='y', colors='white')
        axes['consciousness'].set_ylim(0.5, 7.5)
        axes['consciousness'].set_yticks(range(1, 8))
        axes['consciousness'].set_xlabel("Time", color='white')
        axes['consciousness'].set_ylabel("Level", color='white')
        axes['consciousness'].grid(True, color='white', alpha=0.2)
        
        # Performance metrics plot
        axes['performance'] = fig.add_subplot(grid[1, 2], facecolor='black')
        axes['performance'].set_title("Performance Metrics", color='white', fontsize=14)
        axes['performance'].set_facecolor('black')
        axes['performance'].spines['bottom'].set_color('white')
        axes['performance'].spines['top'].set_color('white')
        axes['performance'].spines['right'].set_color('white')
        axes['performance'].spines['left'].set_color('white')
        axes['performance'].tick_params(axis='x', colors='white')
        axes['performance'].tick_params(axis='y', colors='white')
        axes['performance'].set_xlabel("Time", color='white')
        axes['performance'].set_ylabel("Time (s)", color='white')
        axes['performance'].grid(True, color='white', alpha=0.2)
        
        # Node details table
        axes['details'] = fig.add_subplot(grid[2, :], facecolor='black')
        axes['details'].set_title("Node Details", color='white', fontsize=14)
        axes['details'].set_axis_off()
        
        # Add main title
        fig.suptitle("CASCADE⚡𓂧φ∞ Network Dashboard", 
                   color='white', fontsize=16, y=0.98)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        self.fig = fig
        self.axes = axes
        
        return fig, axes
    
    def update_matplotlib_dashboard(self) -> None:
        """Update matplotlib dashboard with current data."""
        if not self.fig or not self.axes:
            return
        
        # Get network status
        status = self.monitor.get_network_status()
        
        # Get coherence history
        coherence_history = self.monitor.get_coherence_history()
        
        # Get performance metrics
        performance_metrics = self.monitor.get_performance_metrics()
        
        # Get node details
        node_details = self.monitor.get_node_details()
        
        # Format timestamps
        timestamps = [datetime.datetime.fromtimestamp(entry['timestamp']) 
                     for entry in coherence_history]
        
        # Update coherence plot
        self.axes['coherence'].clear()
        self.axes['coherence'].set_title("Network Coherence", color='white', fontsize=14)
        self.axes['coherence'].set_facecolor('black')
        self.axes['coherence'].spines['bottom'].set_color('white')
        self.axes['coherence'].spines['top'].set_color('white')
        self.axes['coherence'].spines['right'].set_color('white')
        self.axes['coherence'].spines['left'].set_color('white')
        self.axes['coherence'].tick_params(axis='x', colors='white')
        self.axes['coherence'].tick_params(axis='y', colors='white')
        self.axes['coherence'].set_ylim(0, 1.05)
        self.axes['coherence'].set_xlabel("Time", color='white')
        self.axes['coherence'].set_ylabel("Coherence", color='white')
        self.axes['coherence'].grid(True, color='white', alpha=0.2)
        
        # Format x-axis for time
        self.axes['coherence'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Plot local coherence
        local_coherence = [entry['local_coherence'] for entry in coherence_history]
        self.axes['coherence'].plot(
            timestamps, local_coherence, 
            color='white', linewidth=2, 
            label="Local Coherence"
        )
        
        # Plot average network coherence
        avg_coherence = [entry['avg_network_coherence'] for entry in coherence_history]
        self.axes['coherence'].plot(
            timestamps, avg_coherence, 
            color='cyan', linewidth=1.5, 
            label="Network Average"
        )
        
        # Mark phi-harmonic levels
        self.axes['coherence'].axhline(y=LAMBDA, color='gold', linestyle='--', alpha=0.5)
        self.axes['coherence'].axhline(y=1/PHI, color='teal', linestyle='--', alpha=0.5)
        
        # Add legend
        self.axes['coherence'].legend(
            loc='upper right', 
            facecolor='black', 
            edgecolor='white', 
            labelcolor='white'
        )
        
        # Update network status gauge
        self.axes['status'].clear()
        self.axes['status'].set_title("Network Health", color='white', fontsize=14)
        self.axes['status'].set_facecolor('black')
        
        # Create gauge chart
        health = status['network_health']
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.ones_like(theta)
        
        # Background circle
        self.axes['status'].plot(theta, r, color='white', alpha=0.2, linewidth=3)
        
        # Health indicator
        health_angle = 2*np.pi * health
        health_theta = np.linspace(0, health_angle, 100)
        health_r = np.ones_like(health_theta)
        
        # Color based on health
        if health < 0.5:
            color = 'red'
        elif health < 0.8:
            color = 'yellow'
        else:
            color = 'green'
            
        self.axes['status'].plot(health_theta, health_r, color=color, linewidth=4)
        
        # Add health value text
        self.axes['status'].text(
            0, 0, f"{health:.1%}",
            color='white', fontsize=20,
            ha='center', va='center'
        )
        
        # Add node stats
        self.axes['status'].text(
            0, -0.5, f"Nodes: {status['node_count']}",
            color='white', fontsize=10,
            ha='center', va='center'
        )
        
        self.axes['status'].text(
            0, -0.7, f"Entangled: {status['entangled_count']}",
            color='white', fontsize=10,
            ha='center', va='center'
        )
        
        # Update node statistics plot
        self.axes['nodes'].clear()
        self.axes['nodes'].set_title("Connected Nodes", color='white', fontsize=14)
        self.axes['nodes'].set_facecolor('black')
        self.axes['nodes'].spines['bottom'].set_color('white')
        self.axes['nodes'].spines['top'].set_color('white')
        self.axes['nodes'].spines['right'].set_color('white')
        self.axes['nodes'].spines['left'].set_color('white')
        self.axes['nodes'].tick_params(axis='x', colors='white')
        self.axes['nodes'].tick_params(axis='y', colors='white')
        self.axes['nodes'].set_xlabel("Time", color='white')
        self.axes['nodes'].set_ylabel("Count", color='white')
        self.axes['nodes'].grid(True, color='white', alpha=0.2)
        
        # Format x-axis for time
        self.axes['nodes'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Plot node count
        node_counts = [entry['node_count'] for entry in coherence_history]
        self.axes['nodes'].plot(
            timestamps, node_counts, 
            color='white', linewidth=2,
            label="Total Nodes"
        )
        
        # Plot entangled count
        entangled_counts = [entry['entangled_count'] for entry in coherence_history]
        self.axes['nodes'].plot(
            timestamps, entangled_counts, 
            color='gold', linewidth=2,
            label="Entangled Nodes"
        )
        
        # Add legend
        self.axes['nodes'].legend(
            loc='upper right', 
            facecolor='black', 
            edgecolor='white', 
            labelcolor='white'
        )
        
        # Update consciousness level plot
        self.axes['consciousness'].clear()
        self.axes['consciousness'].set_title("Consciousness Levels", color='white', fontsize=14)
        self.axes['consciousness'].set_facecolor('black')
        self.axes['consciousness'].spines['bottom'].set_color('white')
        self.axes['consciousness'].spines['top'].set_color('white')
        self.axes['consciousness'].spines['right'].set_color('white')
        self.axes['consciousness'].spines['left'].set_color('white')
        self.axes['consciousness'].tick_params(axis='x', colors='white')
        self.axes['consciousness'].tick_params(axis='y', colors='white')
        self.axes['consciousness'].set_ylim(0.5, 7.5)
        self.axes['consciousness'].set_yticks(range(1, 8))
        self.axes['consciousness'].set_xlabel("Time", color='white')
        self.axes['consciousness'].set_ylabel("Level", color='white')
        self.axes['consciousness'].grid(True, color='white', alpha=0.2)
        
        # Format x-axis for time
        self.axes['consciousness'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Plot consciousness level
        consciousness_levels = [entry['consciousness_level'] for entry in coherence_history]
        self.axes['consciousness'].plot(
            timestamps, consciousness_levels, 
            color='magenta', linewidth=2,
            drawstyle='steps-post'
        )
        
        # Update performance metrics plot
        self.axes['performance'].clear()
        self.axes['performance'].set_title("Performance Metrics", color='white', fontsize=14)
        self.axes['performance'].set_facecolor('black')
        self.axes['performance'].spines['bottom'].set_color('white')
        self.axes['performance'].spines['top'].set_color('white')
        self.axes['performance'].spines['right'].set_color('white')
        self.axes['performance'].spines['left'].set_color('white')
        self.axes['performance'].tick_params(axis='x', colors='white')
        self.axes['performance'].tick_params(axis='y', colors='white')
        self.axes['performance'].set_xlabel("Time", color='white')
        self.axes['performance'].set_ylabel("Time (s)", color='white')
        self.axes['performance'].grid(True, color='white', alpha=0.2)
        
        # Format x-axis for time
        self.axes['performance'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Plot node query time
        if performance_metrics['node_query_time']:
            query_timestamps = [datetime.datetime.fromtimestamp(entry['timestamp']) 
                               for entry in performance_metrics['node_query_time']]
            query_times = [entry['value'] for entry in performance_metrics['node_query_time']]
            
            self.axes['performance'].plot(
                query_timestamps, query_times, 
                color='cyan', linewidth=1.5,
                label="Node Query"
            )
        
        # Plot visualization time
        if performance_metrics['visualization_time']:
            vis_timestamps = [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in performance_metrics['visualization_time']]
            vis_times = [entry['value'] for entry in performance_metrics['visualization_time']]
            
            self.axes['performance'].plot(
                vis_timestamps, vis_times, 
                color='green', linewidth=1.5,
                label="Visualization"
            )
        
        # Add legend
        self.axes['performance'].legend(
            loc='upper right', 
            facecolor='black', 
            edgecolor='white', 
            labelcolor='white'
        )
        
        # Update node details table
        self.axes['details'].clear()
        self.axes['details'].set_title("Node Details", color='white', fontsize=14)
        self.axes['details'].set_axis_off()
        
        # Sort nodes by coherence
        sorted_nodes = sorted(
            node_details, 
            key=lambda n: n['current_coherence'], 
            reverse=True
        )
        
        # Limit to top nodes
        display_nodes = sorted_nodes[:5]
        
        # Create table
        if display_nodes:
            headers = ['Node ID', 'Coherence', 'Level', 'Entangled', 'Uptime']
            data = []
            
            for node in display_nodes:
                # Format node ID
                node_id = node['id'][:8] + '...'
                
                # Format coherence
                coherence = f"{node['current_coherence']:.2f}"
                
                # Format level
                level = str(node['consciousness_level'])
                
                # Format entangled
                entangled = "Yes" if node['entangled'] else "No"
                
                # Format uptime
                uptime_sec = node['uptime']
                hours = int(uptime_sec // 3600)
                minutes = int((uptime_sec % 3600) // 60)
                seconds = int(uptime_sec % 60)
                uptime = f"{hours:02}:{minutes:02}:{seconds:02}"
                
                data.append([node_id, coherence, level, entangled, uptime])
            
            # Create table
            table = self.axes['details'].table(
                cellText=data,
                colLabels=headers,
                loc='center',
                cellLoc='center',
                colColours=['#333333'] * len(headers),
                colLoc='center'
            )
            
            # Style table
            table.scale(1, 1.5)
            table.set_fontsize(10)
            
            # Color cells based on coherence
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(color='white', fontweight='bold')
                    cell.set_facecolor('#333333')
                else:
                    if j == 0:  # Node ID column
                        cell.set_text_props(color='white')
                    elif j == 1:  # Coherence column
                        coherence = float(data[i-1][j])
                        if coherence < 0.5:
                            color = 'red'
                        elif coherence < 0.8:
                            color = 'yellow'
                        else:
                            color = 'green'
                        cell.set_text_props(color=color)
                    elif j == 2:  # Level column
                        level = int(data[i-1][j])
                        color = plt.cm.viridis(level / 7)
                        cell.set_text_props(color=color)
                    elif j == 3:  # Entangled column
                        entangled = data[i-1][j]
                        color = 'gold' if entangled == 'Yes' else 'white'
                        cell.set_text_props(color=color)
                    else:
                        cell.set_text_props(color='white')
                    
                    cell.set_facecolor('black')
        else:
            # No nodes to display
            self.axes['details'].text(
                0.5, 0.5, "No nodes connected",
                color='white', fontsize=14,
                ha='center', va='center'
            )
    
    def create_web_dashboard(self, host: str = '0.0.0.0', port: int = 8050) -> Optional[dash.Dash]:
        """
        Create a web-based dashboard using Dash.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            
        Returns:
            Dash app instance or None if Dash is not available
        """
        if not has_dash:
            logger.warning("Dash not available, web dashboard not created")
            return None
        
        # Create Dash app
        app = dash.Dash(__name__, title="CASCADE⚡𓂧φ∞ Network Dashboard")
        
        # Define layout
        app.layout = html.Div([
            html.H1("CASCADE⚡𓂧φ∞ Network Dashboard", style={'color': 'white', 'textAlign': 'center'}),
            
            # Refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Top row: Coherence and Status
            html.Div([
                # Coherence chart
                html.Div([
                    dcc.Graph(id='coherence-graph')
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                # Network health gauge
                html.Div([
                    dcc.Graph(id='health-gauge')
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            # Middle row: Node stats, Consciousness, Performance
            html.Div([
                # Node statistics
                html.Div([
                    dcc.Graph(id='node-stats')
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                # Consciousness levels
                html.Div([
                    dcc.Graph(id='consciousness-levels')
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                # Performance metrics
                html.Div([
                    dcc.Graph(id='performance-metrics')
                ], style={'width': '33%', 'display': 'inline-block'})
            ]),
            
            # Bottom row: Node details table
            html.Div([
                html.H3("Node Details", style={'color': 'white', 'textAlign': 'center'}),
                html.Div(id='node-details-table')
            ])
        ], style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px'})
        
        # Define callbacks
        @app.callback(
            [Output('coherence-graph', 'figure'),
             Output('health-gauge', 'figure'),
             Output('node-stats', 'figure'),
             Output('consciousness-levels', 'figure'),
             Output('performance-metrics', 'figure'),
             Output('node-details-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get network status
            status = self.monitor.get_network_status()
            
            # Get coherence history
            coherence_history = self.monitor.get_coherence_history()
            
            # Get performance metrics
            performance_metrics = self.monitor.get_performance_metrics()
            
            # Get node details
            node_details = self.monitor.get_node_details()
            
            # Coherence graph
            coherence_graph = {
                'data': [
                    {
                        'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in coherence_history],
                        'y': [entry['local_coherence'] for entry in coherence_history],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Local Coherence',
                        'line': {'color': 'white', 'width': 2}
                    },
                    {
                        'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in coherence_history],
                        'y': [entry['avg_network_coherence'] for entry in coherence_history],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Network Average',
                        'line': {'color': 'cyan', 'width': 1.5}
                    },
                    {
                        'x': [datetime.datetime.fromtimestamp(coherence_history[0]['timestamp']),
                             datetime.datetime.fromtimestamp(coherence_history[-1]['timestamp'])],
                        'y': [LAMBDA, LAMBDA],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Phi-Golden Ratio',
                        'line': {'color': 'gold', 'width': 1, 'dash': 'dash'}
                    }
                ],
                'layout': {
                    'title': 'Network Coherence',
                    'xaxis': {'title': 'Time', 'color': 'white'},
                    'yaxis': {'title': 'Coherence', 'range': [0, 1.05], 'color': 'white'},
                    'paper_bgcolor': 'black',
                    'plot_bgcolor': 'black',
                    'font': {'color': 'white'},
                    'showlegend': True,
                    'legend': {'orientation': 'h', 'y': 1.1}
                }
            }
            
            # Health gauge
            health = status['network_health']
            health_color = 'red'
            if health >= 0.8:
                health_color = 'green'
            elif health >= 0.5:
                health_color = 'yellow'
                
            health_gauge = {
                'data': [
                    {
                        'type': 'indicator',
                        'mode': 'gauge+number',
                        'value': health * 100,
                        'title': {'text': 'Network Health'},
                        'gauge': {
                            'axis': {'range': [0, 100], 'tickcolor': 'white'},
                            'bar': {'color': health_color},
                            'bgcolor': 'rgba(50, 50, 50, 0.5)',
                            'bordercolor': 'white',
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.5)'},
                                {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.5)'},
                                {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.5)'}
                            ],
                        },
                        'number': {'suffix': '%', 'font': {'color': 'white', 'size': 24}}
                    }
                ],
                'layout': {
                    'annotations': [
                        {
                            'x': 0.5,
                            'y': 0.25,
                            'text': f"Nodes: {status['node_count']}",
                            'showarrow': False,
                            'font': {'color': 'white', 'size': 14}
                        },
                        {
                            'x': 0.5,
                            'y': 0.15,
                            'text': f"Entangled: {status['entangled_count']}",
                            'showarrow': False,
                            'font': {'color': 'white', 'size': 14}
                        }
                    ],
                    'paper_bgcolor': 'black',
                    'font': {'color': 'white'},
                    'margin': {'t': 50, 'b': 0, 'l': 0, 'r': 0}
                }
            }
            
            # Node stats
            node_stats = {
                'data': [
                    {
                        'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in coherence_history],
                        'y': [entry['node_count'] for entry in coherence_history],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Total Nodes',
                        'line': {'color': 'white', 'width': 2}
                    },
                    {
                        'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in coherence_history],
                        'y': [entry['entangled_count'] for entry in coherence_history],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Entangled Nodes',
                        'line': {'color': 'gold', 'width': 2}
                    }
                ],
                'layout': {
                    'title': 'Connected Nodes',
                    'xaxis': {'title': 'Time', 'color': 'white'},
                    'yaxis': {'title': 'Count', 'color': 'white'},
                    'paper_bgcolor': 'black',
                    'plot_bgcolor': 'black',
                    'font': {'color': 'white'},
                    'showlegend': True,
                    'legend': {'orientation': 'h', 'y': 1.1}
                }
            }
            
            # Consciousness levels
            consciousness_levels = {
                'data': [
                    {
                        'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                             for entry in coherence_history],
                        'y': [entry['consciousness_level'] for entry in coherence_history],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Consciousness Level',
                        'line': {'color': 'magenta', 'width': 2, 'shape': 'hv'}
                    }
                ],
                'layout': {
                    'title': 'Consciousness Levels',
                    'xaxis': {'title': 'Time', 'color': 'white'},
                    'yaxis': {
                        'title': 'Level', 
                        'range': [0.5, 7.5], 
                        'tickvals': list(range(1, 8)),
                        'color': 'white'
                    },
                    'paper_bgcolor': 'black',
                    'plot_bgcolor': 'black',
                    'font': {'color': 'white'}
                }
            }
            
            # Performance metrics
            performance_data = []
            
            # Add node query time
            if performance_metrics['node_query_time']:
                performance_data.append({
                    'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                         for entry in performance_metrics['node_query_time']],
                    'y': [entry['value'] for entry in performance_metrics['node_query_time']],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Node Query Time',
                    'line': {'color': 'cyan', 'width': 1.5}
                })
            
            # Add visualization time
            if performance_metrics['visualization_time']:
                performance_data.append({
                    'x': [datetime.datetime.fromtimestamp(entry['timestamp']) 
                         for entry in performance_metrics['visualization_time']],
                    'y': [entry['value'] for entry in performance_metrics['visualization_time']],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Visualization Time',
                    'line': {'color': 'green', 'width': 1.5}
                })
            
            performance_metrics_graph = {
                'data': performance_data,
                'layout': {
                    'title': 'Performance Metrics',
                    'xaxis': {'title': 'Time', 'color': 'white'},
                    'yaxis': {'title': 'Time (s)', 'color': 'white'},
                    'paper_bgcolor': 'black',
                    'plot_bgcolor': 'black',
                    'font': {'color': 'white'},
                    'showlegend': True,
                    'legend': {'orientation': 'h', 'y': 1.1}
                }
            }
            
            # Node details table
            sorted_nodes = sorted(
                node_details, 
                key=lambda n: n['current_coherence'], 
                reverse=True
            )
            display_nodes = sorted_nodes[:5]
            
            if display_nodes:
                # Create table
                table = html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Node ID', style={'textAlign': 'center', 'color': 'white'}),
                            html.Th('Coherence', style={'textAlign': 'center', 'color': 'white'}),
                            html.Th('Level', style={'textAlign': 'center', 'color': 'white'}),
                            html.Th('Entangled', style={'textAlign': 'center', 'color': 'white'}),
                            html.Th('Uptime', style={'textAlign': 'center', 'color': 'white'})
                        ], style={'backgroundColor': '#333333'})
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(node['id'][:8] + '...', style={'textAlign': 'center', 'color': 'white'}),
                            html.Td(
                                f"{node['current_coherence']:.2f}", 
                                style={
                                    'textAlign': 'center', 
                                    'color': 'red' if node['current_coherence'] < 0.5 else
                                            'yellow' if node['current_coherence'] < 0.8 else 'green'
                                }
                            ),
                            html.Td(
                                str(node['consciousness_level']), 
                                style={'textAlign': 'center', 'color': 'magenta'}
                            ),
                            html.Td(
                                "Yes" if node['entangled'] else "No", 
                                style={
                                    'textAlign': 'center', 
                                    'color': 'gold' if node['entangled'] else 'white'
                                }
                            ),
                            html.Td(
                                f"{int(node['uptime'] // 3600):02}:{int((node['uptime'] % 3600) // 60):02}:{int(node['uptime'] % 60):02}",
                                style={'textAlign': 'center', 'color': 'white'}
                            )
                        ], style={'backgroundColor': 'black'})
                        for node in display_nodes
                    ])
                ], style={'width': '100%', 'border': '1px solid white'})
            else:
                table = html.Div("No nodes connected", style={'color': 'white', 'textAlign': 'center'})
            
            return coherence_graph, health_gauge, node_stats, consciousness_levels, performance_metrics_graph, table
        
        # Store app instance
        self.web_dashboard = app
        
        # Run web server
        logger.info(f"Web dashboard available at http://{host}:{port}")
        
        return app
    
    def run_web_dashboard(self, host: str = '0.0.0.0', port: int = 8050) -> None:
        """
        Run the web dashboard server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        if not has_dash:
            logger.warning("Dash not available, web dashboard not started")
            return
        
        if not self.web_dashboard:
            self.create_web_dashboard(host, port)
        
        if self.web_dashboard:
            # Run web server
            self.web_dashboard.run_server(host=host, port=port)
        else:
            logger.error("Failed to create web dashboard")
    
    def start_interactive_dashboard(self) -> None:
        """Start interactive matplotlib dashboard."""
        # Create dashboard
        self.create_matplotlib_dashboard()
        
        # Setup animation
        anim = animation.FuncAnimation(
            self.fig, 
            lambda _: self.update_matplotlib_dashboard(),
            interval=self.update_interval,
            blit=False
        )
        
        # Show dashboard
        plt.show()


# Helper function to create and run dashboard
def create_network_dashboard(quantum_field: 'PhiQuantumField', 
                           mode: str = 'matplotlib',
                           host: str = '0.0.0.0',
                           port: int = 8050) -> NetworkDashboard:
    """
    Create and run a network dashboard.
    
    Args:
        quantum_field: PhiQuantumField instance to monitor
        mode: Dashboard mode ('matplotlib' or 'web')
        host: Host for web dashboard
        port: Port for web dashboard
        
    Returns:
        NetworkDashboard instance
    """
    # Create monitor
    monitor = NetworkMonitor(quantum_field)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Create dashboard
    dashboard = NetworkDashboard(monitor)
    
    # Run dashboard in requested mode
    if mode == 'web':
        if has_dash:
            dashboard.create_web_dashboard(host, port)
            dashboard.run_web_dashboard(host, port)
        else:
            logger.warning("Web dashboard not available, falling back to matplotlib")
            dashboard.start_interactive_dashboard()
    else:
        dashboard.start_interactive_dashboard()
    
    return dashboard


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CASCADE⚡𓂧φ∞ Network Dashboard")
    parser.add_argument("--port", type=int, default=4321, help="Network port")
    parser.add_argument("--mode", choices=["matplotlib", "web"], default="matplotlib", help="Dashboard mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web dashboard")
    parser.add_argument("--web-port", type=int, default=8050, help="Port for web dashboard")
    args = parser.parse_args()
    
    try:
        # Create and start quantum field
        from cascade.phi_quantum_network import create_phi_quantum_field
        field = create_phi_quantum_field(port=args.port)
        field.start()
        
        # Create and run dashboard
        dashboard = create_network_dashboard(field, args.mode, args.host, args.web_port)
        
    except KeyboardInterrupt:
        # Clean shutdown
        print("\nShutting down dashboard...")
        if 'field' in locals():
            field.stop()
        
        if 'dashboard' in locals() and hasattr(dashboard, 'monitor'):
            dashboard.monitor.stop_monitoring()
            
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        if 'field' in locals():
            field.stop()