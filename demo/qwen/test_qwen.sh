#!/usr/bin/bash

MODEL_PATH=/home/lds/models/llama
BUILD_PATH=/home/lds/nndeploy/build
ONNX_DATA=${MODEL_PATH}/onnx/llm.onnx.data
TARGET_LINK=${PWD}
EXE=${BUILD_PATH}/nndeploy_demo_qwen

# symbolic link of llm.onnx.data
if [ -L "${TARGET_LINK}" ]; then
    rm "${TARGET_LINK}"
elif [ -f "${TARGET_LINK}" ]; then
    echo "Error: ${TARGET_LINK} is not a symbolic link. Please remove it manually."
    exit 1
fi

ln -s "${ONNX_DATA}" "${TARGET_LINK}"
echo "Symbolic link created: ${TARGET_LINK} -> current folder"

# params
NAME=NNDEPLOY_QWEN
INFER_TYPE=kInferenceTypeOnnxRuntime
DEVICE=kDeviceTypeCodeCuda:0
MODEL_T=kModelTypeOnnx
PARALLEL_TYPE=kParallelTypeSequential
CONFIG=${MODEL_PATH}/llm_config.json

# exactue 
${EXE} \
    --name ${NAME} \
    --inference_type ${INFER_TYPE} \
    --device_type ${DEVICE} \
    --model_type ${MODEL_T} \
    --is_path \
    --parallel_type ${PARALLEL_TYPE} \
    --config_path ${CONFIG} \

#--name NNDEPLOY_LLAMA2 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /root/workspace/model_zoo/model/llm.onnx --parallel_type kParallelTypeSequential --config_path /root/workspace/model_zoo/model/Qwen-0.5_config.json

