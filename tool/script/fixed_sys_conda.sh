#!/bin/bash

# System library priority and preload settings script
# Used to resolve system library conflicts in conda environments

# Default system library path
DEFAULT_SYSTEM_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# Default preload library path
DEFAULT_PRELOAD_LIB_PATH="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --system_lib_path)
            SYSTEM_LIB_PATH="$2"
            shift 2
            ;;
        --preload_lib_path) 
            PRELOAD_LIB_PATH="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set system library priority
SYSTEM_LIB_PATH=${SYSTEM_LIB_PATH:-$DEFAULT_SYSTEM_LIB_PATH}
if [[ ":$LD_LIBRARY_PATH:" != *":$SYSTEM_LIB_PATH:"* ]]; then
    export LD_LIBRARY_PATH="${SYSTEM_LIB_PATH}:${LD_LIBRARY_PATH}"
fi

# Set preload library
PRELOAD_LIB_PATH=${PRELOAD_LIB_PATH:-$DEFAULT_PRELOAD_LIB_PATH}
export LD_PRELOAD="${PRELOAD_LIB_PATH}"

# Execute remaining commands
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    exec "${REMAINING_ARGS[@]}"
fi

# When script is not working, execute directly in terminal
# # Set LD_LIBRARY_PATH to prioritize system libraries
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# # Set LD_PRELOAD
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
