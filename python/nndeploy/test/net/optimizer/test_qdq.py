import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import QuantType, QuantFormat, quantize_static

import nndeploy._nndeploy_internal as _C
from nndeploy.base import DeviceType


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def export_model():
    # 创建模型
    model = TestNet()
    model.eval()

    # 导出PyTorch模型
    dummy_input = torch.randn(1, 3, 224, 224)
    # torch.jit.save(torch.jit.trace(model, dummy_input), "conv_conv.pt")

    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        "test_net_float.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


class CalibrationDataReader:
    def __init__(self, input_name="input"):
        self.input_name = input_name
        self.data_index = 0
        self.data_size = 1024

    def get_next(self):
        if self.data_index >= self.data_size:
            return None
        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.data_index += 1
        return {self.input_name: data}

    def rewind(self):
        self.data_index = 0


def quantize_onnx_model():
    # 量化ONNX模型
    calibration_data_reader = CalibrationDataReader()

    quantize_static(
        "test_net_float.onnx",
        "test_net_qdq.onnx",
        calibration_data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )


def test_qdq_quant():
    interpret = _C.ir.create_interpret(_C.base.ModelType.Onnx)
    assert interpret != None
    interpret.interpret(["test_net_qdq.onnx"])

    md = interpret.get_model_desc()
    assert md != None

    net = _C.net.Net()
    net.setModelDesc(md)

    device = DeviceType("cpu", 0)
    net.setDeviceType(device)
    
    net.init()
   
    net.dump("qdq_opt.dot")


if __name__ == "__main__":

    # # 导出原始模型
    # export_model()

    # # 导出ONNX QDQ量化模型
    # quantize_onnx_model()
    
    
    
    test_qdq_quant()
