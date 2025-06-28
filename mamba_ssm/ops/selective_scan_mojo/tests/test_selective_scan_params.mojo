# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified for Mojo implementation
# ===----------------------------------------------------------------------=== #

"""Tests for selective scan parameter structures."""

from testing import assert_equal, assert_true, assert_false
from memory import UnsafePointer
from builtin.dtype import DType, Scalar

from ..selective_scan_params import SSMParamsForward, KernelConfig, get_kernel_config, MAX_DSTATE, CHUNK_SIZE

fn test_ssm_params_forward_init():
    """Test SSMParamsForward initialization."""
    var params = SSMParamsForward()
    
    # Test default values
    assert_equal(params.batch, 0)
    assert_equal(params.dim, 0)
    assert_equal(params.seqlen, 0)
    assert_equal(params.dstate, 0)
    assert_equal(params.n_groups, 0)
    assert_equal(params.n_chunks, 0)
    assert_equal(params.dim_ngroups_ratio, 0)
    
    # Test flags
    assert_false(params.is_variable_B)
    assert_false(params.is_variable_C)
    assert_false(params.delta_softplus)
    assert_false(params.has_z)
    
    # Test pointer initialization
    assert_true(params.A_ptr.is_null())
    assert_true(params.B_ptr.is_null())
    assert_true(params.C_ptr.is_null())
    assert_true(params.D_ptr.is_null())
    assert_true(params.u_ptr.is_null())
    assert_true(params.delta_ptr.is_null())
    assert_true(params.delta_bias_ptr.is_null())
    assert_true(params.out_ptr.is_null())
    assert_true(params.x_ptr.is_null())
    assert_true(params.z_ptr.is_null())
    assert_true(params.out_z_ptr.is_null())

fn test_kernel_config():
    """Test KernelConfig creation and access."""
    let config = KernelConfig(32, 4, 2)
    
    assert_equal(config.threads, 32)
    assert_equal(config.items, 4)
    assert_equal(config.blocks_per_sm, 2)

fn test_get_kernel_config():
    """Test kernel configuration selection based on sequence length."""
    
    # Test different sequence lengths
    let config_128 = get_kernel_config(128)
    assert_equal(config_128.threads, 32)
    assert_equal(config_128.items, 4)
    
    let config_256 = get_kernel_config(256)
    assert_equal(config_256.threads, 32)
    assert_equal(config_256.items, 8)
    
    let config_512 = get_kernel_config(512)
    assert_equal(config_256.threads, 32)
    assert_equal(config_256.items, 16)
    
    let config_1024 = get_kernel_config(1024)
    assert_equal(config_1024.threads, 64)
    assert_equal(config_1024.items, 16)
    
    let config_large = get_kernel_config(2048)
    assert_equal(config_large.threads, 128)
    assert_equal(config_large.items, 16)

fn test_constants():
    """Test constant values."""
    assert_equal(MAX_DSTATE, 256)
    assert_equal(CHUNK_SIZE, 2048)

fn main():
    test_ssm_params_forward_init()
    test_kernel_config()
    test_get_kernel_config()
    test_constants()
    print("All parameter structure tests passed!") 