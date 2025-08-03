"""
测试Infer类
"""

import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.op
import nndeploy.inference
import nndeploy.dag
import nndeploy.infer
import torch

# python3 nndeploy/test/infer/test_infer.py
# 
# export LD_LIBRARY_PATH=/path/to/nndeploy/build:$LD_LIBRARY_PATH

def test_infer():
    # 创建输入输出边
    input_edge = nndeploy.dag.Edge("input")
    output_edge = nndeploy.dag.Edge("output") 

    # 测试不同初始化方式
    infer1 = nndeploy.infer.Infer("infer1")
    print(infer1)

    infer2 = nndeploy.infer.Infer("infer2", [input_edge], [output_edge])
    print(infer2)

    infer3 = nndeploy.infer.Infer("infer3", type=nndeploy.base.InferenceType.OnnxRuntime)
    print(infer3)

    infer4 = nndeploy.infer.Infer("infer4", [input_edge], [output_edge], 
                                 nndeploy.base.InferenceType.OnnxRuntime)
    print(infer4)

    # 测试输入输出名称设置
    infer1.set_input_name("input1")
    infer1.set_output_name("output1")
    infer1.set_input_names(["input1", "input2"])
    infer1.set_output_names(["output1", "output2"])

    # 测试推理类型设置
    infer1.set_inference_type(nndeploy.base.InferenceType.OnnxRuntime)

    # 测试参数设置和获取
    param = nndeploy.base.Param()
    infer1.set_param(param)
    ret_param = infer1.get_param()
    print(ret_param)

    # 测试生命周期方法
    # print(infer1.init())
    # print(infer1.run())
    # print(infer1.deinit())

    # 测试获取inference对象
    inference = infer1.get_inference()
    print(inference)

if __name__ == "__main__":
    # print(_C.CodecType.OpenCV)
    # print(_C.CodecFlag.Image)
    test_infer()
