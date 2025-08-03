#!/bin/bash

# Automatically set environment variables for third-party libraries
# Usage: source set_install_env.sh

WORKSPACE="$(pwd)"
THIRDPARTY_DIR="${WORKSPACE}/third_party"

echo "Checking third_party directory: $THIRDPARTY_DIR"

# Check if third_party directory exists
if [ ! -d "$THIRDPARTY_DIR" ]; then
    echo "Warning: third_party directory not found: $THIRDPARTY_DIR"
    return 1
fi

# Collect all lib paths
LIB_PATHS=""

# Traverse all directories under third_party
for lib_dir in "${THIRDPARTY_DIR}"/*; do
    if [ -d "$lib_dir" ]; then
        lib_path="${lib_dir}/lib"
        if [ -d "$lib_path" ]; then
            if [ -z "$LIB_PATHS" ]; then
                LIB_PATHS="$lib_path"
            else
                LIB_PATHS="${lib_path}:${LIB_PATHS}"
            fi
            echo "Found library path: $lib_path"
        fi
    fi
done

# Set LD_LIBRARY_PATH
if [ -n "$LIB_PATHS" ]; then
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="${LIB_PATHS}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="$LIB_PATHS"
    fi
    echo "Updated LD_LIBRARY_PATH:"
    echo "  $LD_LIBRARY_PATH"
else
    echo "No library paths found in $THIRDPARTY_DIR"
fi

# Set nndeploy library path
NNDEPLOY_LIB_PATH="${WORKSPACE}/lib"

if [ -n "$NNDEPLOY_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="${NNDEPLOY_LIB_PATH}:${LD_LIBRARY_PATH}"
    echo "Added nndeploy library path: $NNDEPLOY_LIB_PATH"
else
    echo "Warning: nndeploy library path not found in $NNDEPLOY_BUILD_DIR"
fi