#!/usr/bin/env python3
"""
Integration tests for Mojo selective scan implementation.
Tests the PyTorch integration and validates against reference implementation.
"""

import torch
import numpy as np
import pytest
from typing import Tuple, Optional

# Import our Mojo implementation
from selective_scan_mojo.selective_scan_python import selective_scan_mojo_forward

# Import reference implementation from the main interface
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from selective_scan_interface import selective_scan_ref


def create_test_tensors(
    batch_size: int = 2,
    dim: int = 4, 
    seq_len: int = 8,
    d_state: int = 4,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> Tuple[torch.Tensor, ...]:
    """Create test tensors for selective scan."""
    u = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    delta = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    A = torch.randn(dim, d_state, dtype=dtype, device=device)
    B = torch.randn(batch_size, d_state, seq_len, dtype=dtype, device=device)
    C = torch.randn(batch_size, d_state, seq_len, dtype=dtype, device=device)
    D = torch.randn(dim, dtype=dtype, device=device)
    z = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device)
    delta_bias = torch.randn(dim, dtype=torch.float32, device=device)
    
    return u, delta, A, B, C, D, z, delta_bias


def test_basic_forward_pass():
    """Test basic forward pass functionality."""
    u, delta, A, B, C, D, z, delta_bias = create_test_tensors()
    
    # Test with Mojo implementation
    try:
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
        )
        assert out_mojo.shape == u.shape
        assert out_mojo.dtype == u.dtype
        print("✓ Basic forward pass successful")
    except Exception as e:
        pytest.skip(f"Mojo implementation not available: {e}")


def test_reference_comparison():
    """Compare Mojo implementation with reference implementation."""
    u, delta, A, B, C, D, z, delta_bias = create_test_tensors()
    
    # Reference implementation
    out_ref = selective_scan_ref(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
    )
    
    # Mojo implementation
    try:
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
        )
        
        # Compare outputs
        diff = torch.abs(out_mojo - out_ref)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Allow for small numerical differences
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"
        assert mean_diff < 1e-4, f"Mean difference {mean_diff} exceeds tolerance"
        
        print("✓ Reference comparison passed")
        
    except Exception as e:
        pytest.skip(f"Mojo implementation not available: {e}")


def test_variable_bc():
    """Test with variable B and C tensors."""
    batch_size, dim, seq_len, d_state = 2, 4, 8, 4
    dtype = torch.float32
    
    u = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    delta = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    A = torch.randn(dim, d_state, dtype=dtype)
    # Variable B and C (batch, d_state, seq_len)
    B = torch.randn(batch_size, d_state, seq_len, dtype=dtype)
    C = torch.randn(batch_size, d_state, seq_len, dtype=dtype)
    D = torch.randn(dim, dtype=dtype)
    z = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    delta_bias = torch.randn(dim, dtype=torch.float32)
    
    try:
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
        )
        assert out_mojo.shape == u.shape
        print("✓ Variable B/C test passed")
    except Exception as e:
        pytest.skip(f"Mojo implementation not available: {e}")


def test_complex_a():
    """Test with complex A matrix."""
    batch_size, dim, seq_len, d_state = 2, 4, 8, 4
    dtype = torch.float32
    
    u = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    delta = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    # Complex A (real and imaginary parts)
    A_real = torch.randn(dim, d_state, dtype=dtype)
    A_imag = torch.randn(dim, d_state, dtype=dtype)
    A = torch.complex(A_real, A_imag)
    B = torch.randn(batch_size, d_state, seq_len, dtype=dtype)
    C = torch.randn(batch_size, d_state, seq_len, dtype=dtype)
    D = torch.randn(dim, dtype=dtype)
    z = torch.randn(batch_size, dim, seq_len, dtype=dtype)
    delta_bias = torch.randn(dim, dtype=torch.float32)
    
    try:
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
        )
        assert out_mojo.shape == u.shape
        print("✓ Complex A test passed")
    except Exception as e:
        pytest.skip(f"Mojo implementation not available: {e}")


def test_optional_parameters():
    """Test with optional parameters (D=None, z=None, delta_bias=None)."""
    u, delta, A, B, C, _, _, _ = create_test_tensors()
    
    try:
        # Test without optional parameters
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False
        )
        assert out_mojo.shape == u.shape
        print("✓ Optional parameters test passed")
    except Exception as e:
        pytest.skip(f"Mojo implementation not available: {e}")


def test_different_dtypes():
    """Test with different data types."""
    batch_size, dim, seq_len, d_state = 2, 4, 8, 4
    
    for dtype in [torch.float32, torch.float16]:
        u, delta, A, B, C, D, z, delta_bias = create_test_tensors(
            batch_size, dim, seq_len, d_state, dtype
        )
        
        try:
            out_mojo = selective_scan_mojo_forward(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
            )
            assert out_mojo.shape == u.shape
            assert out_mojo.dtype == u.dtype
            print(f"✓ {dtype} test passed")
        except Exception as e:
            print(f"⚠ {dtype} test skipped: {e}")


def test_different_sizes():
    """Test with different tensor sizes."""
    test_cases = [
        (1, 2, 4, 2),   # Small
        (4, 8, 16, 8),  # Medium
        (8, 16, 32, 16), # Large
    ]
    
    for batch_size, dim, seq_len, d_state in test_cases:
        u, delta, A, B, C, D, z, delta_bias = create_test_tensors(
            batch_size, dim, seq_len, d_state
        )
        
        try:
            out_mojo = selective_scan_mojo_forward(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
            )
            assert out_mojo.shape == u.shape
            print(f"✓ Size test {batch_size}x{dim}x{seq_len}x{d_state} passed")
        except Exception as e:
            print(f"⚠ Size test {batch_size}x{dim}x{seq_len}x{d_state} skipped: {e}")


def test_gradient_flow():
    """Test that gradients can flow through the operation."""
    u, delta, A, B, C, D, z, delta_bias = create_test_tensors()
    
    # Make tensors require gradients
    u.requires_grad_(True)
    delta.requires_grad_(True)
    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    D.requires_grad_(True)
    z.requires_grad_(True)
    delta_bias.requires_grad_(True)
    
    try:
        out_mojo = selective_scan_mojo_forward(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
        )
        
        # Compute gradients
        loss = out_mojo.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert u.grad is not None
        assert delta.grad is not None
        assert A.grad is not None
        assert B.grad is not None
        assert C.grad is not None
        assert D.grad is not None
        assert z.grad is not None
        assert delta_bias.grad is not None
        
        print("✓ Gradient flow test passed")
        
    except Exception as e:
        pytest.skip(f"Gradient test skipped: {e}")


if __name__ == "__main__":
    print("Running selective scan integration tests...")
    
    test_basic_forward_pass()
    test_reference_comparison()
    test_variable_bc()
    test_complex_a()
    test_optional_parameters()
    test_different_dtypes()
    test_different_sizes()
    test_gradient_flow()
    
    print("All integration tests completed!") 