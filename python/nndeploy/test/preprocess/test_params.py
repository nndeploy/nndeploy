
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.preprocess
import torch

# python3 nndeploy/test/preprocess/test_params.py


def test_warp_affine_param():
    # 创建WarpAffineCvtNormTransParam实例
    param = _C.preprocess.WarpAffineCvtNormTransParam()
    # param = _C.WarpAffineCvtNormTransParam()
    
    # 测试transform_属性
    transform = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    param.transform_ = transform.numpy()
    assert (param.transform_ == transform.numpy()).all()
    
    # 测试基本属性
    param.dst_w_ = 224
    param.dst_h_ = 224
    param.src_pixel_type_ = nndeploy.base.PixelType.BGR
    param.dst_pixel_type_ = nndeploy.base.PixelType.RGB
    param.data_type_ = nndeploy.base.DataType("float32")
    param.data_format_ = nndeploy.base.DataFormat.NCHW
    param.h_ = 256
    param.w_ = 256
    param.normalize_ = True
    
    # 测试scale_属性
    scale = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    param.scale_ = scale.numpy()
    assert (param.scale_ == scale.numpy()).all()
    
    # 测试mean_属性
    mean = torch.tensor([123.675, 116.28, 103.53, 0.0], dtype=torch.float32)
    param.mean_ = mean.numpy()
    assert (param.mean_ == mean.numpy()).all()
    
    # 测试std_属性
    std = torch.tensor([58.395, 57.12, 57.375, 1.0], dtype=torch.float32)
    param.std_ = std.numpy()
    assert (param.std_ == std.numpy()).all()
    
    # 测试其他属性
    param.const_value_ = 114
    param.interp_type_ = nndeploy.base.InterpType.Linear
    param.border_type_ = nndeploy.base.BorderType.Constant
    
    # 测试border_val_属性
    border_val = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    param.border_val_ = border_val.numpy()
    assert (param.border_val_ == border_val.numpy()).all()
    
    print(param.serialize())
    # 打印param的类型
    print(type(param))
    # # 打印param的父类型
    print(type(param).__base__)
    print(_C.preprocess.WarpAffineCvtNormTransParam.__base__)
    print(_C.dag.Graph)
    print(_C.dag.Graph.__base__)
    print(_C.inference.InferenceParam)
    print(_C.inference.InferenceParam.__base__)
    inference_param = _C.inference.InferenceParam(nndeploy.base.InferenceType.OnnxRuntime)
    print(inference_param.serialize())

    print("WarpAffineCvtNormTransParam test passed!")
    
def test_cvtcolor_param():
    param = nndeploy.preprocess.CvtcolorParam()
    print(param.serialize())
    print(type(param))
    print(type(param).__base__)
    print(nndeploy.preprocess.CvtcolorParam.__base__)
    print("CvtcolorParam test passed!")
    

def test_convert_to():
    convert_to = nndeploy.preprocess.ConvertTo("convert_to")
    print(convert_to.serialize())
    print(type(convert_to))
    print(type(convert_to).__base__)
    print(nndeploy.preprocess.ConvertTo.__base__)
    print("ConvertTo test passed!")

if __name__ == "__main__":
    test_warp_affine_param()
    test_cvtcolor_param()
    test_convert_to()
