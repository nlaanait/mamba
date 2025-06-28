# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation
# ===----------------------------------------------------------------------=== #

"""Utility functions for selective scan forward pass implementation."""

from memory import UnsafePointer
from collections.optional import OptionalReg
from builtin.dtype import DType, Scalar
from math import ceildiv

from .selective_scan_params import SSMParamsForward, MAX_DSTATE, CHUNK_SIZE

# ===-----------------------------------------------------------------------===#
# Parameter Setup Functions
# ===-----------------------------------------------------------------------===#

fn setup_ssm_params_forward[
    dtype: DType
](
    params: SSMParamsForward,
    batch: Int,
    dim: Int,
    seqlen: Int,
    dstate: Int,
    n_groups: Int,
    is_variable_B: Bool,
    is_variable_C: Bool,
    # Tensor pointers and strides
    u_ptr: UnsafePointer[Scalar[dtype]],
    u_batch_stride: UInt32,
    u_d_stride: UInt32,
    delta_ptr: UnsafePointer[Scalar[dtype]],
    delta_batch_stride: UInt32,
    delta_d_stride: UInt32,
    A_ptr: UnsafePointer[Scalar[dtype]],
    A_d_stride: UInt32,
    A_dstate_stride: UInt32,
    B_ptr: UnsafePointer[Scalar[dtype]],
    B_batch_stride: UInt32,
    B_d_stride: UInt32,
    B_dstate_stride: UInt32,
    B_group_stride: UInt32,
    C_ptr: UnsafePointer[Scalar[dtype]],
    C_batch_stride: UInt32,
    C_d_stride: UInt32,
    C_dstate_stride: UInt32,
    C_group_stride: UInt32,
    out_ptr: UnsafePointer[Scalar[dtype]],
    out_batch_stride: UInt32,
    out_d_stride: UInt32,
    x_ptr: UnsafePointer[Scalar[dtype]],
    z_ptr: OptionalReg[UnsafePointer[Scalar[dtype]]],
    z_batch_stride: UInt32,
    z_d_stride: UInt32,
    out_z_ptr: OptionalReg[UnsafePointer[Scalar[dtype]]],
    out_z_batch_stride: UInt32,
    out_z_d_stride: UInt32,
    D_ptr: OptionalReg[UnsafePointer[Float32]],
    delta_bias_ptr: OptionalReg[UnsafePointer[Float32]],
    has_z: Bool,
    delta_softplus: Bool
):
    """Setup SSMParamsForward structure with all necessary parameters.
    
    Args:
        params: The parameter structure to populate
        batch: Batch size
        dim: Hidden dimension size
        seqlen: Sequence length
        dstate: State dimension size
        n_groups: Number of groups for variable B/C
        is_variable_B: Whether B varies across sequence
        is_variable_C: Whether C varies across sequence
        u_ptr: Pointer to input tensor u
        u_batch_stride: Stride for batch dimension in u
        u_d_stride: Stride for dimension in u
        delta_ptr: Pointer to delta tensor
        delta_batch_stride: Stride for batch dimension in delta
        delta_d_stride: Stride for dimension in delta
        A_ptr: Pointer to A matrix
        A_d_stride: Stride for dimension in A
        A_dstate_stride: Stride for state dimension in A
        B_ptr: Pointer to B tensor
        B_batch_stride: Stride for batch dimension in B
        B_d_stride: Stride for dimension in B
        B_dstate_stride: Stride for state dimension in B
        B_group_stride: Stride for group dimension in B
        C_ptr: Pointer to C tensor
        C_batch_stride: Stride for batch dimension in C
        C_d_stride: Stride for dimension in C
        C_dstate_stride: Stride for state dimension in C
        C_group_stride: Stride for group dimension in C
        out_ptr: Pointer to output tensor
        out_batch_stride: Stride for batch dimension in output
        out_d_stride: Stride for dimension in output
        x_ptr: Pointer to intermediate state tensor
        z_ptr: Optional pointer to z tensor
        z_batch_stride: Stride for batch dimension in z
        z_d_stride: Stride for dimension in z
        out_z_ptr: Optional pointer to output z tensor
        out_z_batch_stride: Stride for batch dimension in output z
        out_z_d_stride: Stride for dimension in output z
        D_ptr: Optional pointer to D vector
        delta_bias_ptr: Optional pointer to delta bias vector
        has_z: Whether z tensor is provided
        delta_softplus: Whether to apply softplus to delta
    """
    # Set core dimensions
    params.batch = batch
    params.dim = dim
    params.seqlen = seqlen
    params.dstate = dstate
    params.n_groups = n_groups
    params.n_chunks = ceildiv(seqlen, CHUNK_SIZE)
    params.dim_ngroups_ratio = dim // n_groups
    
    # Set flags
    params.is_variable_B = is_variable_B
    params.is_variable_C = is_variable_C
    params.delta_softplus = delta_softplus
    params.has_z = has_z
    
    # Set strides
    params.A_d_stride = A_d_stride
    params.A_dstate_stride = A_dstate_stride
    
    if not is_variable_B:
        params.B_d_stride = B_d_stride
    else:
        params.B_batch_stride = B_batch_stride
        params.B_group_stride = B_group_stride
    params.B_dstate_stride = B_dstate_stride
    
    if not is_variable_C:
        params.C_d_stride = C_d_stride
    else:
        params.C_batch_stride = C_batch_stride
        params.C_group_stride = C_group_stride
    params.C_dstate_stride = C_dstate_stride
    
    params.u_batch_stride = u_batch_stride
    params.u_d_stride = u_d_stride
    params.delta_batch_stride = delta_batch_stride
    params.delta_d_stride = delta_d_stride
    
    if has_z:
        params.z_batch_stride = z_batch_stride
        params.z_d_stride = z_d_stride
        params.out_z_batch_stride = out_z_batch_stride
        params.out_z_d_stride = out_z_d_stride
    
    params.out_batch_stride = out_batch_stride
    params.out_d_stride = out_d_stride
    
    # Set pointers
    params.u_ptr = u_ptr
    params.delta_ptr = delta_ptr
    params.A_ptr = A_ptr
    params.B_ptr = B_ptr
    params.C_ptr = C_ptr
    params.out_ptr = out_ptr
    params.x_ptr = x_ptr
    
    if has_z:
        params.z_ptr = z_ptr.value
        params.out_z_ptr = out_z_ptr.value
    else:
        params.z_ptr = UnsafePointer[Scalar].null()
        params.out_z_ptr = UnsafePointer[Scalar].null()
    
    params.D_ptr = D_ptr.value_or(UnsafePointer[Float32].null())
    params.delta_bias_ptr = delta_bias_ptr.value_or(UnsafePointer[Float32].null())

# ===-----------------------------------------------------------------------===#
# Validation Functions
# ===-----------------------------------------------------------------------===#

fn validate_inputs[
    dtype: DType
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    dstate: Int,
    n_groups: Int,
    is_variable_B: Bool,
    is_variable_C: Bool
) raises:
    """Validate input parameters for selective scan forward pass.
    
    Args:
        batch: Batch size
        dim: Hidden dimension size
        seqlen: Sequence length
        dstate: State dimension size
        n_groups: Number of groups for variable B/C
        is_variable_B: Whether B varies across sequence
        is_variable_C: Whether C varies across sequence
        
    Raises:
        Error: If any validation fails
    """
    if batch <= 0:
        raise Error("Batch size must be positive")
    if dim <= 0:
        raise Error("Hidden dimension must be positive")
    if seqlen <= 0:
        raise Error("Sequence length must be positive")
    if dstate <= 0:
        raise Error("State dimension must be positive")
    if dstate > MAX_DSTATE:
        raise Error("State dimension must be <= " + MAX_DSTATE.__str__())
    if n_groups <= 0:
        raise Error("Number of groups must be positive")
    if dim % n_groups != 0:
        raise Error("Hidden dimension must be divisible by number of groups")

# ===-----------------------------------------------------------------------===#
# Memory Management Functions
# ===-----------------------------------------------------------------------===#

fn calculate_scan_buffer_size(
    batch: Int,
    dim: Int,
    n_chunks: Int,
    dstate: Int
) -> Int:
    """Calculate the size needed for scan intermediate buffers.
    
    Args:
        batch: Batch size
        dim: Hidden dimension size
        n_chunks: Number of chunks
        dstate: State dimension size
        
    Returns:
        Total number of elements needed for scan buffers
    """
    # Each element stores (real, imag) for complex numbers
    # Buffer layout: (batch, dim, n_chunks, dstate * 2)
    return batch * dim * n_chunks * dstate * 2

# ===-----------------------------------------------------------------------===#
# Helper Functions
# ===-----------------------------------------------------------------------===#

fn is_complex_dtype[dtype: DType]() -> Bool:
    """Check if the given dtype is complex."""
    return dtype == DType.complex64 or dtype == DType.complex128

fn get_complex_multiplier[dtype: DType]() -> Int:
    """Get the multiplier for complex numbers (2 for complex, 1 for real)."""
    if is_complex_dtype[dtype]():
        return 2
    else:
        return 1 