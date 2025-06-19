#!/bin/bash

# System library priority and preload settings script
# Used to resolve system library conflicts in conda environments

# 默认系统库路径
DEFAULT_SYSTEM_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# 默认预加载库路径
DEFAULT_PRELOAD_LIB_PATH="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# 解析命令行参数
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

# 设置系统库优先级
SYSTEM_LIB_PATH=${SYSTEM_LIB_PATH:-$DEFAULT_SYSTEM_LIB_PATH}
if [[ ":$LD_LIBRARY_PATH:" != *":$SYSTEM_LIB_PATH:"* ]]; then
    export LD_LIBRARY_PATH="${SYSTEM_LIB_PATH}:${LD_LIBRARY_PATH}"
fi

# 设置预加载库
PRELOAD_LIB_PATH=${PRELOAD_LIB_PATH:-$DEFAULT_PRELOAD_LIB_PATH}
export LD_PRELOAD="${PRELOAD_LIB_PATH}"

# 执行剩余命令
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    exec "${REMAINING_ARGS[@]}"
fi

# 当脚本不生效时，直接在终端执行
# # 设置LD_LIBRARY_PATH优先使用系统库
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# # 或者设置LD_PRELOAD
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

export LD_PRELOAD="/opt/intel/oneapi/dnnl/latest/lib/libdnnl.so.3.7:$LD_PRELOAD"
