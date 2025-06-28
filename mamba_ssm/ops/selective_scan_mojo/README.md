# Mojo Selective Scan Forward Pass Implementation

This directory contains a Mojo implementation of the selective scan forward pass for the Mamba architecture. The implementation is designed as a PyTorch Custom Op that can be loaded and used in PyTorch models.

## Overview

The selective scan is a core component of the Mamba architecture that implements a state space model (SSM) with the following computation:

1. **Input Processing**: Apply delta bias and softplus activation to delta values
2. **State Update**: For each timestep, update hidden state: `x[t] = exp(delta[t] * A) * x[t-1] + delta[t] * u[t] * B`
3. **Output Computation**: `y[t] = x[t] * C`
4. **Optional Processing**: Add D*u term and apply SiLU activation if z is provided

## File Structure

```
selective_scan_mojo/
├── __init__.py                    # Package initialization
├── selective_scan_params.mojo     # Parameter structures and types
├── selective_scan_utils.mojo      # Utility functions and validation
├── selective_scan_kernels.mojo    # GPU kernel implementations
├── selective_scan_forward.mojo    # Main forward pass implementation
├── selective_scan_op.mojo         # PyTorch Custom Op registration
├── selective_scan_python.py       # Python integration layer
├── BUILD.bazel                    # Build configuration
├── tests/                         # Test files
│   └── test_selective_scan_params.mojo
└── README.md                      # This file
```

## Key Components

### 1. Parameter Structures (`selective_scan_params.mojo`)

- `SSMParamsForward`: Contains all metadata needed for GPU kernel configuration
- `KernelConfig`: GPU kernel launch parameters
- `get_kernel_config()`: Determines optimal kernel configuration based on sequence length

### 2. Utility Functions (`selective_scan_utils.mojo`)

- `setup_ssm_params_forward()`: Populates parameter structure with tensor metadata
- `validate_inputs()`: Validates input parameters
- `calculate_scan_buffer_size()`: Calculates memory requirements
- Helper functions for complex number support

### 3. GPU Kernel (`selective_scan_kernels.mojo`)

- `selective_scan_fwd_kernel`: Main GPU kernel implementation
- `launch_selective_scan_kernel`: Kernel launch logic with specialization
- Scan operation structures for real and complex numbers

### 4. Main Implementation (`selective_scan_forward.mojo`)

- `SelectiveScanForward`: Main struct providing forward pass functionality
- `selective_scan_forward_mojo`: High-level interface function

### 5. PyTorch Integration (`selective_scan_op.mojo`)

- `SelectiveScanForwardOp`: Generic PyTorch Custom Op
- Specialized variants for different data types (FP32, FP16, BF16, Complex64)

## Features

### Supported Data Types
- Float32 (FP32)
- Float16 (FP16)
- BFloat16 (BF16)
- Complex64

### Variable Tensor Support
- Constant B/C tensors: `(dim, dstate)`
- Variable B/C tensors: `(batch, n_groups, dstate, seqlen)`

### Optional Parameters
- D vector: Adds `D * u` term to output
- z tensor: Applies SiLU activation to output
- delta_bias: Adds bias to delta values
- delta_softplus: Applies softplus activation to delta

### Performance Optimizations
- Chunked processing for long sequences (2048 elements per chunk)
- Adaptive kernel configuration based on sequence length
- Memory coalescing optimizations
- Efficient scan operations using GPU primitives

## Usage

### Basic Usage

```python
from mamba_ssm.ops.selective_scan_mojo import selective_scan_mojo_fn

# Create input tensors
u = torch.randn(batch, dim, seqlen, device='cuda')
delta = torch.randn(batch, dim, seqlen, device='cuda')
A = torch.randn(dim, dstate, device='cuda')
B = torch.randn(dim, dstate, device='cuda')
C = torch.randn(dim, dstate, device='cuda')

# Optional parameters
D = torch.randn(dim, device='cuda')
z = torch.randn(batch, dim, seqlen, device='cuda')
delta_bias = torch.randn(dim, device='cuda')

# Execute selective scan
output, last_state = selective_scan_mojo_fn(
    u, delta, A, B, C,
    D=D, z=z, delta_bias=delta_bias,
    delta_softplus=True,
    use_mojo=True  # Use Mojo implementation
)
```

### Integration with Existing Code

The Mojo implementation is designed to be a drop-in replacement for the existing CUDA implementation:

```python
# Existing code
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
output = selective_scan_fn(u, delta, A, B, C)

# With Mojo support
from mamba_ssm.ops.selective_scan_mojo import selective_scan_mojo_fn
output = selective_scan_mojo_fn(u, delta, A, B, C, use_mojo=True)
```

## Building and Testing

### Build Configuration

The implementation uses Bazel for building:

```bash
# Build the library
bazel build //mamba/mamba_ssm/ops/selective_scan_mojo:selective_scan_mojo

# Run tests
bazel test //mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_params
```

### Testing

```bash
# Run parameter structure tests
mojo mamba/mamba_ssm/ops/selective_scan_mojo/tests/test_selective_scan_params.mojo
```

## Performance

The Mojo implementation is designed to achieve comparable performance to the CUDA implementation:

- **Functional Correctness**: Numerical accuracy within 1e-6 tolerance
- **Performance Target**: Within 10% of CUDA performance for typical workloads
- **Memory Efficiency**: Comparable memory usage to CUDA implementation

## Limitations and Future Work

### Current Limitations
- Forward pass only (backward pass not implemented)
- Limited to NVIDIA GPUs
- Requires Mojo compiler and runtime

### Future Enhancements
- Backward pass implementation
- Support for AMD GPUs
- Additional optimizations for specific hardware
- Integration with more PyTorch features

## Dependencies

- Mojo compiler and runtime
- PyTorch with CUDA support
- NVIDIA GPU with compute capability 7.0+
- Bazel build system

## Contributing

When contributing to this implementation:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure performance benchmarks are maintained
5. Test on multiple GPU architectures

## License

This implementation follows the same license as the original Mamba codebase. 