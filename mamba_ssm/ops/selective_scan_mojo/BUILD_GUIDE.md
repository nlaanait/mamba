# Mojo Selective Scan Build Guide

This guide explains how to build, test, and integrate the Mojo implementation of the selective scan operation into the Mamba project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build Options](#build-options)
3. [Bazel Build](#bazel-build)
4. [Setuptools Build](#setuptools-build)
5. [Testing](#testing)
6. [Benchmarking](#benchmarking)
7. [Integration](#integration)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Mojo Compiler**: Version 0.5.0 or later
   ```bash
   # Install Mojo (follow instructions at https://docs.modular.com/mojo/get-started/)
   curl -fsSL https://get.modular.com | sh -
   modular install mojo
   ```

2. **Python**: 3.9 or later
   ```bash
   python --version  # Should be 3.9+
   ```

3. **PyTorch**: 2.0.0 or later
   ```bash
   pip install torch>=2.0.0
   ```

4. **NumPy**: 1.20.0 or later
   ```bash
   pip install numpy>=1.20.0
   ```

### Optional Software

1. **Bazel**: For Bazel-based builds
   ```bash
   # Install Bazel (https://bazel.build/install)
   curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
   sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
   echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
   sudo apt update && sudo apt install bazel
   ```

2. **CUDA**: For GPU acceleration
   ```bash
   # Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)
   # Ensure PyTorch is built with CUDA support
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Build Options

The Mojo selective scan implementation supports multiple build systems:

1. **Bazel Build** (Recommended for development)
2. **Setuptools Build** (Standard Python packaging)
3. **Manual Build** (Direct Mojo compilation)

## Bazel Build

### Prerequisites

- Bazel installed and configured
- Mojo compiler in PATH
- Access to the Modular Bazel rules

### Build Commands

```bash
# Navigate to the Mamba root directory
cd /path/to/mamba

# Build the Mojo library
bazel build //mamba/mamba_ssm/ops/selective_scan_mojo:selective_scan_mojo_lib

# Build and run tests
bazel test //mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_params

# Build Python integration
bazel build //mamba/mamba_ssm/ops/selective_scan_mojo:selective_scan_mojo_py

# Run integration tests
bazel test //mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_integration

# Run benchmarks
bazel run //mamba/mamba_ssm/ops/selective_scan_mojo:benchmark_selective_scan -- --device cpu --dtype float32
```

### Bazel Configuration

The Bazel build is configured in `BUILD.bazel` with the following targets:

- `selective_scan_mojo_lib`: Main Mojo library
- `selective_scan_mojo_py`: Python integration
- `test_selective_scan_params`: Unit tests
- `test_selective_scan_integration`: Integration tests
- `benchmark_selective_scan`: Performance benchmarks

## Setuptools Build

### Prerequisites

- Mojo compiler in PATH
- Python setuptools

### Build Commands

```bash
# Navigate to the Mojo implementation directory
cd mamba/mamba_ssm/ops/selective_scan_mojo

# Install in development mode
pip install -e .

# Or build and install
python setup_mojo.py build_ext
python setup_mojo.py install
```

### Custom Build Commands

The setuptools build includes custom commands:

- `build_ext`: Compiles Mojo files and builds extensions
- `install`: Installs the package and copies artifacts

## Manual Build

### Direct Mojo Compilation

```bash
# Navigate to the Mojo implementation directory
cd mamba/mamba_ssm/ops/selective_scan_mojo

# Compile individual Mojo files
mojo build selective_scan_params.mojo
mojo build selective_scan_utils.mojo
mojo build selective_scan_kernels.mojo
mojo build selective_scan_forward.mojo
mojo build selective_scan_op.mojo

# Run tests
mojo tests/test_selective_scan_params.mojo
```

## Testing

### Test Runner

Use the comprehensive test runner:

```bash
# Run all tests
python run_tests.py

# Run with specific options
python run_tests.py --device cuda --dtype float16 --quick

# Skip certain test types
python run_tests.py --skip-benchmarks --skip-bazel
```

### Individual Test Types

#### Mojo Unit Tests

```bash
# Run Mojo unit tests directly
mojo tests/test_selective_scan_params.mojo
```

#### Python Integration Tests

```bash
# Run Python integration tests
python tests/test_selective_scan_integration.py
```

#### Bazel Tests

```bash
# Run Bazel tests
bazel test //mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_params
```

### Test Coverage

The test suite covers:

- **Unit Tests**: Parameter structures, utility functions
- **Integration Tests**: PyTorch integration, reference comparison
- **Performance Tests**: Benchmarking against reference implementation
- **Edge Cases**: Different data types, tensor shapes, optional parameters

## Benchmarking

### Performance Benchmarks

```bash
# Run comprehensive benchmarks
python benchmarks/benchmark_selective_scan.py

# Quick benchmarks
python benchmarks/benchmark_selective_scan.py --quick

# GPU benchmarks
python benchmarks/benchmark_selective_scan.py --device cuda --dtype float16

# Custom configuration
python benchmarks/benchmark_selective_scan.py --warmup 20 --runs 200
```

### Benchmark Configurations

The benchmark suite includes:

- **Small**: 1x4x16x4 (batch x dim x seq x state)
- **Medium**: 4x16x64x16
- **Large**: 16x64x256x64
- **Mamba Models**: Configurations matching Mamba-130M, 370M, 790M

### Benchmark Output

Benchmarks generate:

- Console output with timing results
- JSON file with detailed statistics
- Speedup analysis vs reference implementation

## Integration

### PyTorch Integration

The Mojo implementation integrates with PyTorch through:

```python
from mamba_ssm.ops.selective_scan_mojo import selective_scan_mojo_forward

# Use the Mojo implementation
output = selective_scan_mojo_forward(
    u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True
)
```

### Fallback Mechanism

The implementation includes automatic fallback:

1. Try Mojo implementation
2. Fall back to CUDA implementation if available
3. Fall back to reference implementation

### Custom Op Registration

The Mojo implementation registers as a PyTorch custom op:

```python
# Automatic registration on import
import mamba_ssm.ops.selective_scan_mojo

# Manual registration
from mamba_ssm.ops.selective_scan_mojo import register_selective_scan_ops
register_selective_scan_ops()
```

## Troubleshooting

### Common Issues

#### Mojo Compiler Not Found

```bash
# Check Mojo installation
which mojo
mojo --version

# Add Mojo to PATH
export PATH="$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH"
```

#### Build Failures

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Reinstall dependencies
pip install --upgrade torch numpy

# Check Mojo version compatibility
mojo --version  # Should be 0.5.0+
```

#### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .

# Check package structure
python -c "import mamba_ssm.ops.selective_scan_mojo; print('Import successful')"
```

#### Performance Issues

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run CPU benchmarks first
python benchmarks/benchmark_selective_scan.py --device cpu

# Check memory usage
nvidia-smi  # For GPU systems
```

### Debug Mode

Enable debug output:

```bash
# Set debug environment variables
export MOJO_DEBUG=1
export PYTHONVERBOSE=1

# Run with debug output
python run_tests.py --device cpu
```

### Logging

The implementation includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable Mojo-specific logging
logging.getLogger('mamba_ssm.ops.selective_scan_mojo').setLevel(logging.DEBUG)
```

## Performance Optimization

### Compilation Flags

For optimal performance:

```bash
# Set optimization flags
export MOJO_OPT_LEVEL=3
export MOJO_TARGET_CPU=native

# Build with optimizations
mojo build -O3 selective_scan_forward.mojo
```

### GPU Optimization

For GPU performance:

```bash
# Set GPU-specific flags
export MOJO_GPU_ARCH=sm_80  # Adjust for your GPU
export MOJO_GPU_OPT_LEVEL=3

# Build with GPU optimizations
mojo build --target gpu selective_scan_kernels.mojo
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
3. **Make changes**
4. **Run tests**: `python run_tests.py`
5. **Run benchmarks**: `python benchmarks/benchmark_selective_scan.py`
6. **Submit pull request**

### Code Style

- Follow Mojo coding standards
- Use type hints and docstrings
- Add tests for new features
- Update documentation

### Testing Guidelines

- Add unit tests for new functionality
- Ensure integration tests pass
- Run benchmarks to verify performance
- Test on multiple platforms if possible

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the test output for error details
3. Check Mojo and PyTorch compatibility
4. Open an issue with detailed error information

## License

This implementation follows the same license as the main Mamba project (Apache 2.0). 