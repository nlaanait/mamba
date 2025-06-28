#!/usr/bin/env python3
"""
Setup script for Mojo selective scan implementation.
Integrates with the existing Mamba build system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import torch

# Get the directory containing this script
this_dir = Path(__file__).parent.absolute()
mamba_root = this_dir.parent.parent.parent  # mamba/mamba_ssm/ops/selective_scan_mojo -> mamba


class MojoBuildExt(build_ext):
    """Custom build extension for Mojo compilation."""
    
    def run(self):
        """Run the build process."""
        print("Building Mojo selective scan implementation...")
        
        # Check if we're in a Bazel environment
        if self._is_bazel_environment():
            print("Detected Bazel environment, using Bazel build...")
            self._build_with_bazel()
        else:
            print("Using standard setup.py build...")
            self._build_with_setuptools()
        
        # Call the parent build_ext
        super().run()
    
    def _is_bazel_environment(self):
        """Check if we're in a Bazel build environment."""
        return (
            os.environ.get('BAZEL_BUILD') == '1' or
            os.environ.get('BUILD_WORKSPACE_DIRECTORY') is not None or
            os.path.exists('BUILD.bazel') or
            os.path.exists('WORKSPACE')
        )
    
    def _build_with_bazel(self):
        """Build using Bazel."""
        try:
            # Build the Mojo library
            subprocess.run([
                'bazel', 'build', 
                '//mamba/mamba_ssm/ops/selective_scan_mojo:selective_scan_mojo_lib'
            ], check=True, cwd=mamba_root)
            
            # Build tests
            subprocess.run([
                'bazel', 'build', 
                '//mamba/mamba_ssm/ops/selective_scan_mojo:test_selective_scan_params'
            ], check=True, cwd=mamba_root)
            
            print("✓ Bazel build completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Bazel build failed: {e}")
            raise
        except FileNotFoundError:
            print("✗ Bazel not found, falling back to setuptools build")
            self._build_with_setuptools()
    
    def _build_with_setuptools(self):
        """Build using standard setuptools."""
        try:
            # Check if Mojo is available
            mojo_path = shutil.which('mojo')
            if not mojo_path:
                print("⚠ Mojo compiler not found in PATH")
                print("  Please install Mojo and add it to your PATH")
                print("  Visit: https://docs.modular.com/mojo/get-started/")
                return
            
            print(f"Using Mojo compiler: {mojo_path}")
            
            # Compile Mojo files
            mojo_files = [
                'selective_scan_params.mojo',
                'selective_scan_utils.mojo',
                'selective_scan_kernels.mojo',
                'selective_scan_forward.mojo',
                'selective_scan_op.mojo',
            ]
            
            for mojo_file in mojo_files:
                mojo_path = this_dir / mojo_file
                if mojo_path.exists():
                    print(f"Compiling {mojo_file}...")
                    subprocess.run([
                        'mojo', 'build', str(mojo_path)
                    ], check=True, cwd=this_dir)
            
            print("✓ Setuptools build completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Setuptools build failed: {e}")
            raise


class MojoInstall(install):
    """Custom install command for Mojo implementation."""
    
    def run(self):
        """Run the installation process."""
        print("Installing Mojo selective scan implementation...")
        
        # Call the parent install
        super().run()
        
        # Copy compiled artifacts if they exist
        self._copy_mojo_artifacts()
        
        print("✓ Installation completed")
    
    def _copy_mojo_artifacts(self):
        """Copy compiled Mojo artifacts to the installation directory."""
        # This would copy any compiled artifacts to the appropriate location
        # For now, we'll just print a message
        print("  Mojo artifacts will be loaded dynamically at runtime")


def get_mojo_package_data():
    """Get package data for Mojo implementation."""
    return {
        'mamba_ssm.ops.selective_scan_mojo': [
            '*.mojo',
            'tests/*.mojo',
            '*.py',
            'tests/*.py',
            'benchmarks/*.py',
            'README.md',
        ]
    }


def get_mojo_install_requires():
    """Get install requirements for Mojo implementation."""
    base_requires = [
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ]
    
    # Add Mojo-specific requirements if available
    try:
        import mojo
        print("✓ Mojo runtime detected")
    except ImportError:
        print("⚠ Mojo runtime not detected")
        print("  Some features may not be available")
    
    return base_requires


def main():
    """Main setup function."""
    
    # Read the README for long description
    readme_path = this_dir / 'README.md'
    long_description = ""
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            long_description = f.read()
    
    setup(
        name="mamba-ssm-mojo",
        version="0.1.0",
        description="Mojo implementation of Mamba selective scan operations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Mamba Team",
        author_email="",
        url="https://github.com/state-spaces/mamba",
        packages=find_packages(where=str(mamba_root)),
        package_dir={'': str(mamba_root)},
        package_data=get_mojo_package_data(),
        install_requires=get_mojo_install_requires(),
        python_requires=">=3.9",
        cmdclass={
            'build_ext': MojoBuildExt,
            'install': MojoInstall,
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="mamba, selective-scan, mojo, gpu, deep-learning",
        project_urls={
            "Bug Reports": "https://github.com/state-spaces/mamba/issues",
            "Source": "https://github.com/state-spaces/mamba",
            "Documentation": "https://github.com/state-spaces/mamba",
        },
    )


if __name__ == "__main__":
    main() 