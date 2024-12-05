# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package metadata
NAME = "SAM2 Realtime"
VERSION = "1.0"
DESCRIPTION = "SAM2 Realtime: Segment Anything for Video Streams"
URL = "https://github.com/facebookresearch/segment-anything-2"
AUTHOR = "Peter Schroedl"
AUTHOR_EMAIL = "peter_schroedl@me.com"
LICENSE = "Apache 2.0"

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
]

def get_extensions():
    srcs = ["sam2_realtime/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    ext_modules = [CUDAExtension("sam2_realtime._C", srcs, extra_compile_args=compile_args)]
    return ext_modules


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.15",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
