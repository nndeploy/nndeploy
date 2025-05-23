#!/bin/bash

# 子模块自动克隆脚本
# 支持克隆全部或单个子模块

# 项目根目录
PROJECT_ROOT="nndeploy"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 错误处理函数
handle_error() {
    echo -e "${RED}错误: $1${NC}"
    exit 1
}

# 检查命令是否成功执行
check_status() {
    if [ $? -ne 0 ]; then
        handle_error "$1"
    else
        echo -e "${GREEN}✓${NC} $2"
    fi
}

# 检查SSH连接
check_ssh_connection() {
    echo -e "${YELLOW}正在检查SSH连接到GitHub...${NC}"
    ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}警告: SSH连接测试失败，请确保已将SSH公钥添加到GitHub账户${NC}"
        echo -e "${YELLOW}继续执行，但可能会遇到权限问题${NC}"
        read -p "是否继续? (y/n): " choice
        if [ "$choice" != "y" ]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ SSH连接正常${NC}"
    fi
}

# 创建目录结构
create_directory_structure() {
    echo -e "${YELLOW}正在创建目录结构...${NC}"
    if [ -d "$THIRD_PARTY_DIR" ]; then
        echo -e "${YELLOW}目录结构已存在，正在删除...${NC}"
        rm -rf "$THIRD_PARTY_DIR"
        check_status "删除已存在目录失败" "已存在目录删除成功"
    fi
    mkdir -p "$THIRD_PARTY_DIR"
    check_status "创建目录失败" "目录结构创建成功"
}

# 克隆子模块函数
clone_submodule() {
    local name=$1
    local url=$2
    local path=$3
    local branch=$4
    
    echo -e "${YELLOW}正在克隆子模块: $name...${NC}"
    
    if [ -d "$path" ]; then
        echo -e "${YELLOW}目录已存在，正在删除: $path${NC}"
        rm -rf "$path"
        check_status "删除已存在子模块目录失败" "已存在子模块目录删除成功"
    fi
    
    if [ -z "$branch" ]; then
        git clone "$url" "$path"
    else
        git clone -b "$branch" "$url" "$path"
    fi
    
    check_status "克隆子模块 $name 失败" "子模块 $name 克隆成功"
}

# 显示帮助信息
show_help() {
    echo -e "用法: $0 [选项] [子模块名称]"
    echo -e "选项:"
    echo -e "  --all         克隆所有子模块(默认行为)"
    echo -e "  --list        列出所有可用子模块"
    echo -e "  --help        显示此帮助信息"
    echo -e ""
    echo -e "子模块名称:"
    echo -e "  gflags, protobuf, onnx, tokenizers-cpp, pybind11,"
    echo -e "  rapidjson, safetensors-cpp, googletest, OpenCL-Headers"
    exit 0
}

# 列出所有子模块
list_submodules() {
    echo -e "可用子模块列表:"
    echo -e "  1. gflags"
    echo -e "  2. protobuf"
    echo -e "  3. onnx"
    echo -e "  4. tokenizers-cpp"
    echo -e "  5. pybind11"
    echo -e "  6. rapidjson"
    echo -e "  7. safetensors-cpp"
    echo -e "  8. googletest"
    echo -e "  9. OpenCL-Headers"
    exit 0
}

# 主函数
main() {
    # 解析命令行参数
    TARGET_MODULE=""
    CLONE_ALL=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                CLONE_ALL=true
                shift
                ;;
            --list)
                list_submodules
                ;;
            --help)
                show_help
                ;;
            gflags|protobuf|onnx|tokenizers-cpp|pybind11|rapidjson|safetensors-cpp|googletest|OpenCL-Headers)
                TARGET_MODULE=$1
                CLONE_ALL=false
                shift
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                show_help
                ;;
        esac
    done
    
    # 检查SSH连接
    check_ssh_connection
    
    # 创建目录结构
    create_directory_structure
    
    # 进入third_party目录
    cd "$THIRD_PARTY_DIR" || handle_error "无法进入目录: $THIRD_PARTY_DIR"
    
    # 定义子模块列表
    declare -A SUBMODULES
    SUBMODULES["gflags"]="git@github.com:gflags/gflags.git"
    SUBMODULES["protobuf"]="git@github.com:protocolbuffers/protobuf.git"
    SUBMODULES["onnx"]="git@github.com:onnx/onnx.git"
    SUBMODULES["tokenizers-cpp"]="git@github.com:mlc-ai/tokenizers-cpp.git"
    SUBMODULES["pybind11"]="git@github.com:pybind/pybind11.git"
    SUBMODULES["rapidjson"]="git@github.com:Tencent/rapidjson.git"
    SUBMODULES["safetensors-cpp"]="git@github.com:nndeploy/safetensors-cpp.git"
    SUBMODULES["googletest"]="git@github.com:google/googletest.git"
    SUBMODULES["OpenCL-Headers"]="git@github.com:KhronosGroup/OpenCL-Headers.git"
    
    # 定义子模块分支
    declare -A SUBMODULE_BRANCHES
    SUBMODULE_BRANCHES["onnx"]="v1.15.0"
    
    # 克隆指定子模块或全部子模块
    if [ "$CLONE_ALL" = true ]; then
        echo -e "${YELLOW}开始克隆所有子模块...${NC}"
        
        for module in "${!SUBMODULES[@]}"; do
            clone_submodule "$module" "${SUBMODULES[$module]}" "$module" "${SUBMODULE_BRANCHES[$module]}"
        done
        
        # 验证子模块数量
        local submodule_count=$(ls -d */ | wc -l)
        if [ "$submodule_count" -eq 9 ]; then
            echo -e "${GREEN}✓ 所有9个子模块均已成功克隆${NC}"
        else
            echo -e "${RED}警告: 只克隆了 $submodule_count/9 个子模块，请检查是否有错误${NC}"
        fi
    else
        echo -e "${YELLOW}开始克隆单个子模块: $TARGET_MODULE...${NC}"
        
        if [[ -z "${SUBMODULES[$TARGET_MODULE]}" ]]; then
            handle_error "未知子模块: $TARGET_MODULE"
        fi
        
        clone_submodule "$TARGET_MODULE" "${SUBMODULES[$TARGET_MODULE]}" "$TARGET_MODULE" "${SUBMODULE_BRANCHES[$TARGET_MODULE]}"
    fi
    
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}操作完成!${NC}"
    echo -e "${GREEN}项目路径: $(realpath $PROJECT_ROOT)${NC}"
    echo -e "${GREEN}=========================================${NC}"
}

# 执行主函数
main "$@"

# usage: ./clone_submodule.sh --all 克隆所有子模块
# usage: ./clone_submodule.sh gflags 克隆gflags子模块
# usage: ./clone_submodule.sh --list 列出所有可用子模块