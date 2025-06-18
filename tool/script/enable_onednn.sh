#!/bin/bash

# export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
# source /opt/intel/oneapi/setvars.sh

# Default oneAPI root path
ONEAPI_ROOT=${1:-"/opt/intel/oneapi"}

# Set LD_LIBRARY_PATH
ONEAPI_LIB="${ONEAPI_ROOT}/compiler/latest/lib"
if [[ ":$LD_LIBRARY_PATH:" != *":$ONEAPI_LIB:"* ]]; then
    export LD_LIBRARY_PATH="${ONEAPI_LIB}:${LD_LIBRARY_PATH}"
fi

# Execute setvars.sh
SETVARS_PATH="${ONEAPI_ROOT}/setvars.sh"
if [ -f "$SETVARS_PATH" ]; then
    source "$SETVARS_PATH"
else
    echo "Error: setvars.sh not found at $SETVARS_PATH"
    exit 1
fi