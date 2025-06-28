# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation

"""Python integration for Mojo selective scan forward pass."""

import torch
from typing import Optional, Tuple
from mamba_ssm.utils.torch import custom_fwd

# Try to import the Mojo implementation
try:
    from .selective_scan_forward import selective_scan_forward_mojo
    MOJO_AVAILABLE = True
except ImportError:
    MOJO_AVAILABLE = False
    print("Warning: Mojo selective scan implementation not available")

def selective_scan_forward_mojo_wrapper(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    use_mojo: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Wrapper function for selective scan forward pass with Mojo support.
    
    Args:
        u: Input tensor of shape (batch, dim, seqlen)
        delta: Delta tensor of shape (batch, dim, seqlen)
        A: A matrix of shape (dim, dstate)
        B: B tensor of shape (dim, dstate) or variable shape
        C: C tensor of shape (dim, dstate) or variable shape
        D: Optional D vector of shape (dim,)
        z: Optional z tensor of shape (batch, dim, seqlen)
        delta_bias: Optional delta bias vector of shape (dim,)
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return the last state
        use_mojo: Whether to use Mojo implementation if available
        
    Returns:
        Tuple of (output, last_state) where last_state is optional
    """
    
    if use_mojo and MOJO_AVAILABLE:
        # Use Mojo implementation
        # Note: This would require proper tensor conversion and device context setup
        # For now, this is a placeholder showing the intended interface
        raise NotImplementedError("Mojo implementation integration not yet complete")
    else:
        # Fall back to existing CUDA implementation
        from .selective_scan_interface import selective_scan_fn
        return selective_scan_fn(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
        )

class SelectiveScanMojoFn(torch.autograd.Function):
    """PyTorch autograd function for Mojo selective scan forward pass."""
    
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        delta_softplus: bool = False,
        return_last_state: bool = False,
        use_mojo: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for selective scan with Mojo support."""
        
        # Ensure tensors are contiguous
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        
        # Handle variable B/C reshaping
        if B.dim() == 3:
            B = B.unsqueeze(1)  # Add group dimension
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(1)  # Add group dimension
            ctx.squeeze_C = True
        
        # Try Mojo implementation first
        if use_mojo and MOJO_AVAILABLE:
            try:
                result = selective_scan_forward_mojo_wrapper(
                    u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, True
                )
                ctx.use_mojo = True
                return result
            except Exception as e:
                print(f"Mojo implementation failed, falling back to CUDA: {e}")
                use_mojo = False
        
        # Fall back to CUDA implementation
        from .selective_scan_interface import selective_scan_fn
        ctx.use_mojo = False
        result = selective_scan_fn(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
        )
        
        # Save context for backward pass
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.squeeze_B = getattr(ctx, 'squeeze_B', False)
        ctx.squeeze_C = getattr(ctx, 'squeeze_C', False)
        
        return result
    
    @staticmethod
    def backward(ctx, dout, *args):
        """Backward pass - currently delegates to CUDA implementation."""
        # For now, always use CUDA backward pass
        # TODO: Implement Mojo backward pass when needed
        from .selective_scan_interface import SelectiveScanFn
        return SelectiveScanFn.backward(ctx, dout, *args)

def selective_scan_mojo_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    use_mojo: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """High-level function for selective scan with Mojo support.
    
    Args:
        u: Input tensor of shape (batch, dim, seqlen)
        delta: Delta tensor of shape (batch, dim, seqlen)
        A: A matrix of shape (dim, dstate)
        B: B tensor of shape (dim, dstate) or variable shape
        C: C tensor of shape (dim, dstate) or variable shape
        D: Optional D vector of shape (dim,)
        z: Optional z tensor of shape (batch, dim, seqlen)
        delta_bias: Optional delta bias vector of shape (dim,)
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return the last state
        use_mojo: Whether to use Mojo implementation if available
        
    Returns:
        Tuple of (output, last_state) where last_state is optional
    """
    return SelectiveScanMojoFn.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, use_mojo
    )

# Example usage function
def example_usage():
    """Example of how to use the Mojo selective scan implementation."""
    
    # Create sample tensors
    batch, dim, seqlen, dstate = 2, 64, 128, 16
    
    u = torch.randn(batch, dim, seqlen, device='cuda')
    delta = torch.randn(batch, dim, seqlen, device='cuda')
    A = torch.randn(dim, dstate, device='cuda')
    B = torch.randn(dim, dstate, device='cuda')
    C = torch.randn(dim, dstate, device='cuda')
    D = torch.randn(dim, device='cuda')
    z = torch.randn(batch, dim, seqlen, device='cuda')
    delta_bias = torch.randn(dim, device='cuda')
    
    # Use Mojo implementation
    output, last_state = selective_scan_mojo_fn(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus=True, use_mojo=True
    )
    
    print(f"Output shape: {output.shape}")
    if last_state is not None:
        print(f"Last state shape: {last_state.shape}")
    
    return output, last_state

if __name__ == "__main__":
    example_usage() 