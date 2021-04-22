#!/bin/bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION.

# libcudf examples build script

# Add libcudf examples build scripts down below

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

################################################################################
# Basic example
BASIC_EXAMPLE_DIR=basic
BASIC_EXAMPLE_BUILD_DIR=${BASIC_EXAMPLE_DIR}/build

# Configure
cmake -S ${BASIC_EXAMPLE_DIR} -B ${BASIC_EXAMPLE_BUILD_DIR}
# Build
cmake --build ${BASIC_EXAMPLE_DIR} -j${PARALLEL_LEVEL}
