import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import DeviceType
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
)
import math


class TestSupportedUnaryOperators(unittest.TestCase):
    def setUp(self):
        """测试前的初始化工作"""
        self.src_shape = [16, 1000]
        # 基础测试数据（通用）
        self.base_np_src = np.random.uniform(-5, 5, self.src_shape).astype(np.float32)
        # 非零输入（用于倒数等操作，避免除零错误）
        self.non_zero_np_src = self.base_np_src.copy()
        self.non_zero_np_src[self.non_zero_np_src == 0] = 0.001
        # 正数输入（用于log、sqrt等操作，避免定义域错误）
        self.positive_np_src = np.random.uniform(0.1, 5, self.src_shape).astype(
            np.float32
        )
        # [-1,1]范围输入（用于acos、asin等操作）
        self.range_11_np_src = np.random.uniform(-1, 1, self.src_shape).astype(
            np.float32
        )

        # 创建CPU和CUDA张量（按输入类型分类）
        # 1. 基础张量
        self.cpu_base_src = create_tensor_from_numpy(self.base_np_src)
        self.cuda_base_src = self.cpu_base_src.to(DeviceType("cuda"))
        # 2. 非零张量
        self.cpu_non_zero_src = create_tensor_from_numpy(self.non_zero_np_src)
        self.cuda_non_zero_src = self.cpu_non_zero_src.to(DeviceType("cuda"))
        # 3. 正数张量
        self.cpu_positive_src = create_tensor_from_numpy(self.positive_np_src)
        self.cuda_positive_src = self.cpu_positive_src.to(DeviceType("cuda"))
        # 4. [-1,1]范围张量
        self.cpu_range_11_src = create_tensor_from_numpy(self.range_11_np_src)
        self.cuda_range_11_src = self.cpu_range_11_src.to(DeviceType("cuda"))

    def _test_operator(
        self, nndeploy_func, torch_func, cpu_src, cuda_src, rtol=1e-02, atol=1e-04
    ):
        """通用测试函数：对比nndeploy与PyTorch结果"""
        # 计算PyTorch参考结果
        torch_tensor = torch.from_numpy(create_numpy_from_tensor(cpu_src))
        torch_result = torch_func(torch_tensor)

        # 计算nndeploy结果（CUDA执行后转回CPU）
        cuda_result = nndeploy_func(cuda_src)
        nndeploy_result = create_numpy_from_tensor(cuda_result.to(DeviceType("cpu")))

        # 精度校验
        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                nndeploy_result,
                rtol=rtol,
                atol=atol,
            ),
            f"Test failed for operator: {nndeploy_func.__name__}",
        )

    # ------------------------------ 支持的算子测试 ------------------------------
    def test_relu(self):
        """测试ReLU（支持）"""
        self._test_operator(
            nndeploy_func=F.relu,
            torch_func=torch.relu,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_hard_sigmoid(self):
        """测试HardSigmoid（支持）"""
        self._test_operator(
            nndeploy_func=F.hardsigmoid,
            torch_func=torch.nn.functional.hardsigmoid,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_selu(self):
        """测试SELU（支持）"""
        self._test_operator(
            nndeploy_func=F.selu,
            torch_func=torch.selu,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_tanh(self):
        """测试Tanh（支持）"""
        self._test_operator(
            nndeploy_func=F.tanh,
            torch_func=torch.tanh,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_abs(self):
        """测试Abs（支持）"""
        self._test_operator(
            nndeploy_func=F.abs,
            torch_func=torch.abs,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_acos(self):
        """测试Acos（支持，输入范围[-1,1]）"""
        self._test_operator(
            nndeploy_func=F.acos,
            torch_func=torch.acos,
            cpu_src=self.cpu_range_11_src,
            cuda_src=self.cuda_range_11_src,
        )

    def test_asin(self):
        """测试Asin（支持，输入范围[-1,1]）"""
        self._test_operator(
            nndeploy_func=F.asin,
            torch_func=torch.asin,
            cpu_src=self.cpu_range_11_src,
            cuda_src=self.cuda_range_11_src,
        )

    def test_atan(self):
        """测试Atan（支持）"""
        self._test_operator(
            nndeploy_func=F.atan,
            torch_func=torch.atan,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_ceil(self):
        """测试Ceil（支持）"""
        self._test_operator(
            nndeploy_func=F.ceil,
            torch_func=torch.ceil,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_cos(self):
        """测试Cos（支持）"""
        self._test_operator(
            nndeploy_func=F.cos,
            torch_func=torch.cos,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_cosh(self):
        """测试Cosh（支持）"""
        self._test_operator(
            nndeploy_func=F.cosh,
            torch_func=torch.cosh,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_erf(self):
        """测试Erf（支持）"""
        self._test_operator(
            nndeploy_func=F.erf,
            torch_func=torch.erf,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_exp(self):
        """测试Exp（支持，输入范围[-5,5]避免数值溢出）"""
        self._test_operator(
            nndeploy_func=F.exp,
            torch_func=torch.exp,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_floor(self):
        """测试Floor（支持）"""
        self._test_operator(
            nndeploy_func=F.floor,
            torch_func=torch.floor,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_log(self):
        """测试Log（支持，输入为正数）"""
        self._test_operator(
            nndeploy_func=F.log,
            torch_func=torch.log,
            cpu_src=self.cpu_positive_src,
            cuda_src=self.cuda_positive_src,
        )

    def test_reciprocal(self):
        """测试Reciprocal（支持，输入非零）"""
        self._test_operator(
            nndeploy_func=F.reciprocal,
            torch_func=torch.reciprocal,
            cpu_src=self.cpu_non_zero_src,
            cuda_src=self.cuda_non_zero_src,
        )

    def test_round(self):
        """测试Round（支持）"""
        self._test_operator(
            nndeploy_func=F.round,
            torch_func=torch.round,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_sigmoid(self):
        """测试Sigmoid（支持）"""
        self._test_operator(
            nndeploy_func=F.sigmoid,
            torch_func=torch.sigmoid,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_sign(self):
        """测试Sign（支持）"""
        self._test_operator(
            nndeploy_func=F.sign,
            torch_func=torch.sign,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_sin(self):
        """测试Sin（支持）"""
        self._test_operator(
            nndeploy_func=F.sin,
            torch_func=torch.sin,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_sinh(self):
        """测试Sinh（支持）"""
        self._test_operator(
            nndeploy_func=F.sinh,
            torch_func=torch.sinh,
            cpu_src=self.cpu_base_src,
            cuda_src=self.cuda_base_src,
        )

    def test_sqrt(self):
        """测试Sqrt（支持，输入为正数）"""
        self._test_operator(
            nndeploy_func=F.sqrt,
            torch_func=torch.sqrt,
            cpu_src=self.cpu_positive_src,
            cuda_src=self.cuda_positive_src,
        )

    def test_tan(self):
        """测试Tan（支持）"""
        safe_tan_np_src = np.random.uniform(
            -math.pi / 2 + 0.1, math.pi / 2 - 0.1, self.src_shape
        ).astype(np.float32)
        cpu_safe_tan_src = create_tensor_from_numpy(safe_tan_np_src)
        cuda_safe_tan_src = cpu_safe_tan_src.to(DeviceType("cuda"))
        self._test_operator(
            nndeploy_func=F.tan,
            torch_func=torch.tan,
            cpu_src=cpu_safe_tan_src,
            cuda_src=cuda_safe_tan_src,
        )


if __name__ == "__main__":
    unittest.main()
