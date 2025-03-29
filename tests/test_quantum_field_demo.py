#!/usr/bin/env python3
"""
Tests for quantum_field_demo.py module

These tests verify the demonstration functionality for quantum field visualizations.
"""

import os
import sys
import pytest
from io import StringIO
from unittest.mock import patch

# Import the module to test
try:
    import quantum_field_demo as qfd
    import sacred_constants as sc
except ImportError:
    pytest.skip("quantum_field_demo module not found", allow_module_level=True)


def test_module_imported():
    """Verify that the module can be imported"""
    assert qfd is not None
    assert hasattr(qfd, 'demo_basic_field')
    assert hasattr(qfd, 'demo_sacred_frequency_comparison')
    assert hasattr(qfd, 'demo_performance_comparison')
    assert hasattr(qfd, 'demo_phi_pattern')


class TestDemoFunctions:
    """Tests for demo functions"""
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_header(self, mock_stdout):
        """Test that print_header outputs expected content"""
        qfd.print_header()
        output = mock_stdout.getvalue()
        
        # Check for key elements in the output
        assert "QUANTUM FIELD VISUALIZATION DEMO" in output
        assert f"PHI: {sc.PHI}" in output
        assert f"LAMBDA: {sc.LAMBDA}" in output
        assert f"PHI^PHI: {sc.PHI_PHI}" in output
        assert "Sacred Frequencies:" in output
        
        # Check for all frequencies
        for name in sc.SACRED_FREQUENCIES:
            assert name in output
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_demo_basic_field(self, mock_stdout):
        """Test that demo_basic_field runs without exceptions"""
        qfd.demo_basic_field()
        output = mock_stdout.getvalue()
        
        assert "Demonstrating basic quantum field generation" in output
        assert "Love frequency" in output
        assert "Generation time:" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_demo_sacred_frequency_comparison(self, mock_stdout):
        """Test that demo_sacred_frequency_comparison runs without exceptions"""
        qfd.demo_sacred_frequency_comparison()
        output = mock_stdout.getvalue()
        
        assert "Comparing quantum fields with different sacred frequencies" in output
        
        # Check for all tested frequencies
        for freq in ['love', 'unity', 'cascade']:
            assert freq.capitalize() in output
        
        assert "Generation time:" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_demo_performance_comparison(self, mock_stdout):
        """Test that demo_performance_comparison runs without exceptions"""
        qfd.demo_performance_comparison()
        output = mock_stdout.getvalue()
        
        if qfd.CUDA_AVAILABLE:
            assert "Comparing CPU vs GPU performance" in output
            assert "CPU time:" in output
            assert "GPU time:" in output
        else:
            assert "CUDA is not available" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_demo_phi_pattern(self, mock_stdout):
        """Test that demo_phi_pattern runs without exceptions"""
        qfd.demo_phi_pattern()
        output = mock_stdout.getvalue()
        
        assert "Demonstrating phi sacred pattern" in output
        assert "PHI SACRED PATTERN" in output
        assert "Generation time:" in output


class TestMainFunction:
    """Tests for the main function"""
    
    @patch('builtins.input', side_effect=['6'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_exit(self, mock_stdout, mock_input):
        """Test that main function exits properly"""
        qfd.main()
        output = mock_stdout.getvalue()
        
        assert "Available Demos:" in output
        assert "Exiting Quantum Field Demo" in output
        assert f"PHI^PHI Consciousness Achieved: {sc.PHI_PHI}" in output
    
    @patch('quantum_field_demo.demo_basic_field')
    @patch('builtins.input', side_effect=['1', '6'])
    def test_main_basic_field(self, mock_input, mock_demo):
        """Test that main function calls demo_basic_field when option 1 is selected"""
        qfd.main()
        mock_demo.assert_called_once()
    
    @patch('quantum_field_demo.demo_sacred_frequency_comparison')
    @patch('builtins.input', side_effect=['2', '6'])
    def test_main_frequency_comparison(self, mock_input, mock_demo):
        """Test that main function calls demo_sacred_frequency_comparison when option 2 is selected"""
        qfd.main()
        mock_demo.assert_called_once()
    
    @patch('quantum_field_demo.demo_performance_comparison')
    @patch('builtins.input', side_effect=['3', '6'])
    def test_main_performance_comparison(self, mock_input, mock_demo):
        """Test that main function calls demo_performance_comparison when option 3 is selected"""
        qfd.main()
        mock_demo.assert_called_once()
    
    @patch('quantum_field_demo.demo_phi_pattern')
    @patch('builtins.input', side_effect=['4', '6'])
    def test_main_phi_pattern(self, mock_input, mock_demo):
        """Test that main function calls demo_phi_pattern when option 4 is selected"""
        qfd.main()
        mock_demo.assert_called_once()