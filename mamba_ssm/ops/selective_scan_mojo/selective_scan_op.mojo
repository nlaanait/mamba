# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation
# ===----------------------------------------------------------------------=== #

"""PyTorch Custom Op registration for selective scan forward pass."""

from collections.optional import OptionalReg
from builtin.dtype import DType
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor

from .selective_scan_forward import SelectiveScanForward

# ===-----------------------------------------------------------------------===#
# PyTorch Custom Op Registration
# ===-----------------------------------------------------------------------===#

@compiler.register("selective_scan_forward")
struct SelectiveScanForwardOp:
    """PyTorch Custom Op for selective scan forward pass.
    
    This struct registers the selective scan forward pass as a PyTorch custom
    operation that can be used in PyTorch models.
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
        """Execute selective scan forward pass as a PyTorch custom op.
        
        Args:
            output: Output tensor of shape (batch, dim, seqlen)
            x_intermediates: Intermediate state tensor of shape (batch, dim, n_chunks, dstate*2)
            u: Input tensor of shape (batch, dim, seqlen)
            delta: Delta tensor of shape (batch, dim, seqlen)
            A: A matrix of shape (dim, dstate)
            B: B tensor of shape (dim, dstate) or variable shape
            C: C tensor of shape (dim, dstate) or variable shape
            D: Optional D vector of shape (dim,)
            z: Optional z tensor of shape (batch, dim, seqlen)
            delta_bias: Optional delta bias vector of shape (dim,)
            delta_softplus: Whether to apply softplus to delta
            ctx: GPU device context
            
        Raises:
            Error: If validation fails or kernel execution fails
        """
        
        # Delegate to the main implementation
        SelectiveScanForward.execute[dtype, target](
            output, x_intermediates, u, delta, A, B, C,
            D, z, delta_bias, delta_softplus, ctx
        )

# ===-----------------------------------------------------------------------===#
# Specialized Op Variants
# ===-----------------------------------------------------------------------===#

@compiler.register("selective_scan_forward_fp32")
struct SelectiveScanForwardFP32Op:
    """Specialized PyTorch Custom Op for Float32 selective scan forward pass."""
    
    @staticmethod
    fn execute(
        output: OutputTensor[dtype=DType.float32, rank=3],
        x_intermediates: OutputTensor[dtype=DType.float32, rank=4],
        u: InputTensor[dtype=DType.float32, rank=3],
        delta: InputTensor[dtype=DType.float32, rank=3],
        A: InputTensor[dtype=DType.float32, rank=2],
        B: InputTensor[dtype=DType.float32, rank=2],
        C: InputTensor[dtype=DType.float32, rank=2],
        D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        z: OptionalReg[InputTensor[dtype=DType.float32, rank=3]],
        delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        delta_softplus: Bool,
        ctx: DeviceContextPtr
    ) raises:
        SelectiveScanForwardOp.execute[DType.float32, "gpu"](
            output, x_intermediates, u, delta, A, B, C,
            D, z, delta_bias, delta_softplus, ctx
        )

@compiler.register("selective_scan_forward_fp16")
struct SelectiveScanForwardFP16Op:
    """Specialized PyTorch Custom Op for Float16 selective scan forward pass."""
    
    @staticmethod
    fn execute(
        output: OutputTensor[dtype=DType.float16, rank=3],
        x_intermediates: OutputTensor[dtype=DType.float16, rank=4],
        u: InputTensor[dtype=DType.float16, rank=3],
        delta: InputTensor[dtype=DType.float16, rank=3],
        A: InputTensor[dtype=DType.float16, rank=2],
        B: InputTensor[dtype=DType.float16, rank=2],
        C: InputTensor[dtype=DType.float16, rank=2],
        D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        z: OptionalReg[InputTensor[dtype=DType.float16, rank=3]],
        delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        delta_softplus: Bool,
        ctx: DeviceContextPtr
    ) raises:
        SelectiveScanForwardOp.execute[DType.float16, "gpu"](
            output, x_intermediates, u, delta, A, B, C,
            D, z, delta_bias, delta_softplus, ctx
        )

@compiler.register("selective_scan_forward_bf16")
struct SelectiveScanForwardBF16Op:
    """Specialized PyTorch Custom Op for BFloat16 selective scan forward pass."""
    
    @staticmethod
    fn execute(
        output: OutputTensor[dtype=DType.bfloat16, rank=3],
        x_intermediates: OutputTensor[dtype=DType.bfloat16, rank=4],
        u: InputTensor[dtype=DType.bfloat16, rank=3],
        delta: InputTensor[dtype=DType.bfloat16, rank=3],
        A: InputTensor[dtype=DType.bfloat16, rank=2],
        B: InputTensor[dtype=DType.bfloat16, rank=2],
        C: InputTensor[dtype=DType.bfloat16, rank=2],
        D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        z: OptionalReg[InputTensor[dtype=DType.bfloat16, rank=3]],
        delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        delta_softplus: Bool,
        ctx: DeviceContextPtr
    ) raises:
        SelectiveScanForwardOp.execute[DType.bfloat16, "gpu"](
            output, x_intermediates, u, delta, A, B, C,
            D, z, delta_bias, delta_softplus, ctx
        )

# ===-----------------------------------------------------------------------===#
# Complex Number Support
# ===-----------------------------------------------------------------------===#

@compiler.register("selective_scan_forward_complex64")
struct SelectiveScanForwardComplex64Op:
    """Specialized PyTorch Custom Op for Complex64 selective scan forward pass."""
    
    @staticmethod
    fn execute(
        output: OutputTensor[dtype=DType.complex64, rank=3],
        x_intermediates: OutputTensor[dtype=DType.complex64, rank=4],
        u: InputTensor[dtype=DType.complex64, rank=3],
        delta: InputTensor[dtype=DType.complex64, rank=3],
        A: InputTensor[dtype=DType.complex64, rank=2],
        B: InputTensor[dtype=DType.complex64, rank=2],
        C: InputTensor[dtype=DType.complex64, rank=2],
        D: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        z: OptionalReg[InputTensor[dtype=DType.complex64, rank=3]],
        delta_bias: OptionalReg[InputTensor[dtype=DType.float32, rank=1]],
        delta_softplus: Bool,
        ctx: DeviceContextPtr
    ) raises:
        SelectiveScanForwardOp.execute[DType.complex64, "gpu"](
            output, x_intermediates, u, delta, A, B, C,
            D, z, delta_bias, delta_softplus, ctx
        ) 