# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation
# ===----------------------------------------------------------------------=== #

"""Main forward pass implementation for selective scan."""

from memory import UnsafePointer
from collections.optional import OptionalReg
from builtin.dtype import DType, Scalar
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor, ManagedTensorSlice

from .selective_scan_params import SSMParamsForward, KernelConfig, get_kernel_config
from .selective_scan_utils import setup_ssm_params_forward, validate_inputs, calculate_scan_buffer_size
from .selective_scan_kernels import launch_selective_scan_kernel

# ===-----------------------------------------------------------------------===#
# Main Forward Pass Implementation
# ===-----------------------------------------------------------------------===#

struct SelectiveScanForward:
    """Main implementation of selective scan forward pass.
    
    This struct provides the core functionality for the selective scan forward
    pass operation, handling parameter validation, memory allocation, and
    GPU kernel launching.
    """
    
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        x_intermediates: OutputTensor[dtype=dtype, rank=4],
        u: InputTensor[dtype=dtype, rank=3],
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=2],
        C: InputTensor[dtype=dtype, rank=2],
        D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        z: OptionalReg[InputTensor[dtype=dtype, rank=3]],
        delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        delta_softplus: Bool,
        ctx: DeviceContextPtr
    ) raises:
        """Execute selective scan forward pass.
        
        Args:
            output: Output tensor of shape (batch, dim, seqlen)
            x_intermediates: Intermediate state tensor of shape (batch, dim, n_chunks, dstate*2)
            u: Input tensor of shape (batch, dim, seqlen)
            delta: Delta tensor of shape (batch, dim, seqlen)
            A: A matrix of shape (dim, dstate)
            B: B tensor of shape (dim, dstate) or (batch, n_groups, dstate, seqlen)
            C: C tensor of shape (dim, dstate) or (batch, n_groups, dstate, seqlen)
            D: Optional D vector of shape (dim,)
            z: Optional z tensor of shape (batch, dim, seqlen)
            delta_bias: Optional delta bias vector of shape (dim,)
            delta_softplus: Whether to apply softplus to delta
            ctx: GPU device context
            
        Raises:
            Error: If validation fails or kernel execution fails
        """
        
        # Get tensor dimensions
        let batch = u.shape()[0]
        let dim = u.shape()[1]
        let seqlen = u.shape()[2]
        let dstate = A.shape()[1]
        
        # Determine if B and C are variable
        let is_variable_B = B.rank() >= 3
        let is_variable_C = C.rank() >= 3
        let n_groups = if is_variable_B: B.shape()[1] else: 1
        
        # Validate inputs
        validate_inputs[dtype](batch, dim, seqlen, dstate, n_groups, is_variable_B, is_variable_C)
        
        # Get kernel configuration
        let config = get_kernel_config(seqlen)
        
        # Get device context
        let dev_ctx = ctx.get_device_context()
        
        # Setup parameter structure
        var params = SSMParamsForward()
        
        # Get tensor pointers and strides
        let u_slice = u.to_layout_tensor()
        let delta_slice = delta.to_layout_tensor()
        let A_slice = A.to_layout_tensor()
        let B_slice = B.to_layout_tensor()
        let C_slice = C.to_layout_tensor()
        let output_slice = output.to_layout_tensor()
        let x_slice = x_intermediates.to_layout_tensor()
        
        # Setup parameters
        setup_ssm_params_forward[dtype](
            params,
            batch, dim, seqlen, dstate, n_groups,
            is_variable_B, is_variable_C,
            # u tensor
            u_slice.unsafe_ptr(),
            u_slice.strides()[0].cast[UInt32](),
            u_slice.strides()[1].cast[UInt32](),
            # delta tensor
            delta_slice.unsafe_ptr(),
            delta_slice.strides()[0].cast[UInt32](),
            delta_slice.strides()[1].cast[UInt32](),
            # A matrix
            A_slice.unsafe_ptr(),
            A_slice.strides()[0].cast[UInt32](),
            A_slice.strides()[1].cast[UInt32](),
            # B tensor
            B_slice.unsafe_ptr(),
            if is_variable_B: B_slice.strides()[0].cast[UInt32]() else: 0,
            if not is_variable_B: B_slice.strides()[0].cast[UInt32]() else: 0,
            B_slice.strides()[if is_variable_B: 2 else: 1].cast[UInt32](),
            if is_variable_B: B_slice.strides()[1].cast[UInt32]() else: 0,
            # C tensor
            C_slice.unsafe_ptr(),
            if is_variable_C: C_slice.strides()[0].cast[UInt32]() else: 0,
            if not is_variable_C: C_slice.strides()[0].cast[UInt32]() else: 0,
            C_slice.strides()[if is_variable_C: 2 else: 1].cast[UInt32](),
            if is_variable_C: C_slice.strides()[1].cast[UInt32]() else: 0,
            # output tensor
            output_slice.unsafe_ptr(),
            output_slice.strides()[0].cast[UInt32](),
            output_slice.strides()[1].cast[UInt32](),
            # x intermediates tensor
            x_slice.unsafe_ptr(),
            # z tensor (optional)
            if z.has_value():
                let z_slice = z.value().to_layout_tensor()
                OptionalReg(z_slice.unsafe_ptr())
            else:
                OptionalReg[UnsafePointer[Scalar[dtype]]]()
            ,
            if z.has_value():
                let z_slice = z.value().to_layout_tensor()
                z_slice.strides()[0].cast[UInt32]()
            else:
                0
            ,
            if z.has_value():
                let z_slice = z.value().to_layout_tensor()
                z_slice.strides()[1].cast[UInt32]()
            else:
                0
            ,
            # out_z tensor (optional - same as output for now)
            OptionalReg[UnsafePointer[Scalar[dtype]]](),
            0, 0,
            # D vector (optional)
            if D.has_value():
                let D_slice = D.value().to_layout_tensor()
                OptionalReg(D_slice.unsafe_ptr())
            else:
                OptionalReg[UnsafePointer[Float32]]()
            ,
            # delta_bias vector (optional)
            if delta_bias.has_value():
                let delta_bias_slice = delta_bias.value().to_layout_tensor()
                OptionalReg(delta_bias_slice.unsafe_ptr())
            else:
                OptionalReg[UnsafePointer[Float32]]()
            ,
            z.has_value(),
            delta_softplus
        )
        
        # Launch kernel
        launch_selective_scan_kernel[dtype](params, config, dev_ctx)
        
        # Synchronize to ensure completion
        dev_ctx.synchronize()

# ===-----------------------------------------------------------------------===#
# Helper Functions
# ===-----------------------------------------------------------------------===#

fn selective_scan_forward_mojo[
    dtype: DType
](
    u: InputTensor[dtype=dtype, rank=3],
    delta: InputTensor[dtype=dtype, rank=3],
    A: InputTensor[dtype=dtype, rank=2],
    B: InputTensor[dtype=dtype, rank=2],
    C: InputTensor[dtype=dtype, rank=2],
    D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
    z: OptionalReg[InputTensor[dtype=dtype, rank=3]],
    delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
    delta_softplus: Bool,
    return_last_state: Bool,
    ctx: DeviceContextPtr
) raises -> (OutputTensor[dtype=dtype, rank=3], OptionalReg[OutputTensor[dtype=dtype, rank=3]]):
    """High-level interface for selective scan forward pass.
    
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
        ctx: GPU device context
        
    Returns:
        Tuple of (output, last_state) where last_state is optional
    """
    
    let batch = u.shape()[0]
    let dim = u.shape()[1]
    let seqlen = u.shape()[2]
    let dstate = A.shape()[1]
    let n_chunks = ceildiv(seqlen, 2048)  # CHUNK_SIZE
    
    # Create output tensor
    let output = OutputTensor[dtype=dtype, rank=3](batch, dim, seqlen)
    
    # Create intermediate state tensor
    let x_intermediates = OutputTensor[dtype=dtype, rank=4](batch, dim, n_chunks, dstate * 2)
    
    # Execute forward pass
    SelectiveScanForward.execute[dtype, "gpu"](
        output, x_intermediates, u, delta, A, B, C,
        D, z, delta_bias, delta_softplus, ctx
    )
    
    # Handle last state if requested
    if return_last_state:
        # Extract last state from intermediates
        let last_state = OutputTensor[dtype=dtype, rank=3](batch, dim, dstate)
        # TODO: Implement extraction of last state from x_intermediates
        return (output, OptionalReg(last_state))
    else:
        return (output, OptionalReg[OutputTensor[dtype=dtype, rank=3]]()) 