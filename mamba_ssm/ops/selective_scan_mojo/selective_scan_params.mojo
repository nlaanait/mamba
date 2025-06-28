# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation
# ===----------------------------------------------------------------------=== #

"""Parameter structures for selective scan forward pass implementation."""

from memory import UnsafePointer
from memory.pointer import AddressSpace
from collections.optional import OptionalReg
from builtin.dtype import DType, Scalar

# ===-----------------------------------------------------------------------===#
# SSMParamsForward
# ===-----------------------------------------------------------------------===#

@fieldwise_init
@register_passable("trivial")
struct SSMParamsForward:
    """Parameter structure for selective scan forward pass.
    
    Contains all the metadata needed to configure and launch the GPU kernel
    for the selective scan forward pass operation.
    """
    
    # Core dimensions
    var batch: Int
    var dim: Int  
    var seqlen: Int
    var dstate: Int
    var n_groups: Int
    var n_chunks: Int
    var dim_ngroups_ratio: Int
    
    # Flags
    var is_variable_B: Bool
    var is_variable_C: Bool
    var delta_softplus: Bool
    var has_z: Bool
    
    # Strides (in elements, not bytes)
    var A_d_stride: UInt32
    var A_dstate_stride: UInt32
    var B_batch_stride: UInt32
    var B_d_stride: UInt32
    var B_dstate_stride: UInt32
    var B_group_stride: UInt32
    var C_batch_stride: UInt32
    var C_d_stride: UInt32
    var C_dstate_stride: UInt32
    var C_group_stride: UInt32
    var u_batch_stride: UInt32
    var u_d_stride: UInt32
    var delta_batch_stride: UInt32
    var delta_d_stride: UInt32
    var z_batch_stride: UInt32
    var z_d_stride: UInt32
    var out_batch_stride: UInt32
    var out_d_stride: UInt32
    var out_z_batch_stride: UInt32
    var out_z_d_stride: UInt32
    
    # Data pointers - using generic Scalar type for flexibility
    var A_ptr: UnsafePointer[Scalar]
    var B_ptr: UnsafePointer[Scalar]
    var C_ptr: UnsafePointer[Scalar]
    var D_ptr: UnsafePointer[Float32]
    var u_ptr: UnsafePointer[Scalar]
    var delta_ptr: UnsafePointer[Scalar]
    var delta_bias_ptr: UnsafePointer[Float32]
    var out_ptr: UnsafePointer[Scalar]
    var x_ptr: UnsafePointer[Scalar]
    var z_ptr: UnsafePointer[Scalar]
    var out_z_ptr: UnsafePointer[Scalar]
    
    fn __init__(inout self):
        """Initialize all fields to zero/default values."""
        self.batch = 0
        self.dim = 0
        self.seqlen = 0
        self.dstate = 0
        self.n_groups = 0
        self.n_chunks = 0
        self.dim_ngroups_ratio = 0
        self.is_variable_B = False
        self.is_variable_C = False
        self.delta_softplus = False
        self.has_z = False
        
        # Initialize strides
        self.A_d_stride = 0
        self.A_dstate_stride = 0
        self.B_batch_stride = 0
        self.B_d_stride = 0
        self.B_dstate_stride = 0
        self.B_group_stride = 0
        self.C_batch_stride = 0
        self.C_d_stride = 0
        self.C_dstate_stride = 0
        self.C_group_stride = 0
        self.u_batch_stride = 0
        self.u_d_stride = 0
        self.delta_batch_stride = 0
        self.delta_d_stride = 0
        self.z_batch_stride = 0
        self.z_d_stride = 0
        self.out_batch_stride = 0
        self.out_d_stride = 0
        self.out_z_batch_stride = 0
        self.out_z_d_stride = 0
        
        # Initialize pointers to null
        self.A_ptr = UnsafePointer[Scalar].null()
        self.B_ptr = UnsafePointer[Scalar].null()
        self.C_ptr = UnsafePointer[Scalar].null()
        self.D_ptr = UnsafePointer[Float32].null()
        self.u_ptr = UnsafePointer[Scalar].null()
        self.delta_ptr = UnsafePointer[Scalar].null()
        self.delta_bias_ptr = UnsafePointer[Float32].null()
        self.out_ptr = UnsafePointer[Scalar].null()
        self.x_ptr = UnsafePointer[Scalar].null()
        self.z_ptr = UnsafePointer[Scalar].null()
        self.out_z_ptr = UnsafePointer[Scalar].null()

# ===-----------------------------------------------------------------------===#
# Kernel Configuration
# ===-----------------------------------------------------------------------===#

@register_passable("trivial")
struct KernelConfig:
    """Configuration for GPU kernel launch parameters."""
    var threads: Int
    var items: Int
    var blocks_per_sm: Int
    
    fn __init__(inout self, threads: Int, items: Int, blocks_per_sm: Int = 2):
        self.threads = threads
        self.items = items
        self.blocks_per_sm = blocks_per_sm

fn get_kernel_config(seqlen: Int) -> KernelConfig:
    """Determine optimal kernel configuration based on sequence length.
    
    Args:
        seqlen: Length of the sequence to process
        
    Returns:
        KernelConfig with optimal thread and item counts
    """
    if seqlen <= 128:
        return KernelConfig(32, 4)
    elif seqlen <= 256:
        return KernelConfig(32, 8)
    elif seqlen <= 512:
        return KernelConfig(32, 16)
    elif seqlen <= 1024:
        return KernelConfig(64, 16)
    else:
        return KernelConfig(128, 16)

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

alias MAX_DSTATE = 256
alias CHUNK_SIZE = 2048
alias M_LOG2E = 1.4426950408889634  # log2(e) 