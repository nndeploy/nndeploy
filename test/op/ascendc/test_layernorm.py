#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LayerNorm 算子 Ascend C 实现测试
"""

import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import name_to_device_type_code
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
)


def layernorm_reference(x, weight, bias, eps=1e-5):
    """
    PyTorch LayerNorm 参考实现
    """
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(variance + torch.tensor([eps]))
    y = y * weight + bias
    return y


class TestLayerNormOp(unittest.TestCase):

    def test_layernorm_2d_small(self):
        """测试 2D 输入 [N, H] 小尺寸"""
        N, H = 4, 128
        self._test_layernorm_2d(N, H)

    def test_layernorm_2d_medium(self):
        """测试 2D 输入 [N, H] 中等尺寸"""
        N, H = 8, 512
        self._test_layernorm_2d(N, H)

    def test_layernorm_2d_large(self):
        """测试 2D 输入 [N, H] 大尺寸"""
        N, H = 16, 1024
        self._test_layernorm_2d(N, H)

    def test_layernorm_3d_small(self):
        """测试 3D 输入 [B, S, H] 小尺寸"""
        B, S, H = 2, 128, 128
        self._test_layernorm_3d(B, S, H)

    def test_layernorm_3d_medium(self):
        """测试 3D 输入 [B, S, H] 中等尺寸"""
        B, S, H = 2, 256, 256
        self._test_layernorm_3d(B, S, H)

    def test_layernorm_3d_large(self):
        """测试 3D 输入 [B, S, H] 大尺寸"""
        B, S, H = 2, 512, 512
        self._test_layernorm_3d(B, S, H)

    def _test_layernorm_2d(self, N, H, eps=1e-5):
        """
        测试 2D 输入 [N, H]
        """
        print(f"\n=== Test LayerNorm 2D: N={N}, H={H} ===")

        # 创建输入数据
        np_input = np.random.randn(N, H).astype(np.float32)
        np_weight = np.random.randn(H).astype(np.float32)
        np_bias = np.random.randn(H).astype(np.float32)

        # PyTorch 参考结果
        torch_input = torch.from_numpy(np_input)
        torch_weight = torch.from_numpy(np_weight)
        torch_bias = torch.from_numpy(np_bias)
        torch_result = layernorm_reference(torch_input, torch_weight, torch_bias, eps)

        # nnDeploy Ascend C 实现
        input_tensor = create_tensor_from_numpy(np_input)
        weight_tensor = create_tensor_from_numpy(np_weight)
        bias_tensor = create_tensor_from_numpy(np_bias)

        ascend_input = input_tensor.to(nndeploy.base.DeviceType("ascendcl"))
        ascend_weight = weight_tensor.to(nndeploy.base.DeviceType("ascendcl"))
        ascend_bias = bias_tensor.to(nndeploy.base.DeviceType("ascendcl"))

        # 运行 LayerNorm
        ascend_result = F.layer_norm(ascend_input, ascend_weight, ascend_bias, eps=eps)

        # 转回 CPU 比较
        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))
        nndeploy_result_np = create_numpy_from_tensor(nndeploy_result)

        # 精度对比
        torch_result_np = torch_result.detach().numpy()
        max_diff = np.max(np.abs(nndeploy_result_np - torch_result_np))
        mean_diff = np.mean(np.abs(nndeploy_result_np - torch_result_np))

        print(f"[INFO] Max diff: {max_diff:.6e}")
        print(f"[INFO] Mean diff: {mean_diff:.6e}")

        # 精度标准：float32  atol=1e-5, rtol=1e-4
        passed = np.allclose(nndeploy_result_np, torch_result_np, atol=1e-5, rtol=1e-4)
        print(f"[{'PASS' if passed else 'FAIL'}] Accuracy check")

        self.assertTrue(passed)

    def _test_layernorm_3d(self, B, S, H, eps=1e-5):
        """
        测试 3D 输入 [B, S, H]
        """
        print(f"\n=== Test LayerNorm 3D: B={B}, S={S}, H={H} ===")

        # 创建输入数据
        np_input = np.random.randn(B, S, H).astype(np.float32)
        np_weight = np.random.randn(H).astype(np.float32)
        np_bias = np.random.randn(H).astype(np.float32)

        # PyTorch 参考结果
        torch_input = torch.from_numpy(np_input)
        torch_weight = torch.from_numpy(np_weight)
        torch_bias = torch.from_numpy(np_bias)
        torch_result = layernorm_reference(torch_input, torch_weight, torch_bias, eps)

        # nnDeploy Ascend C 实现
        input_tensor = create_tensor_from_numpy(np_input)
        weight_tensor = create_tensor_from_numpy(np_weight)
        bias_tensor = create_tensor_from_numpy(np_bias)

        ascend_input = input_tensor.to(nndeploy.base.DeviceType("ascendcl"))
        ascend_weight = weight_tensor.to(nndeploy.base.DeviceType("ascendcl"))
        ascend_bias = bias_tensor.to(nndeploy.base.DeviceType("ascendcl"))

        # 运行 LayerNorm
        ascend_result = F.layer_norm(ascend_input, ascend_weight, ascend_bias, eps=eps)

        # 转回 CPU 比较
        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))
        nndeploy_result_np = create_numpy_from_tensor(nndeploy_result)

        # 精度对比
        torch_result_np = torch_result.detach().numpy()
        max_diff = np.max(np.abs(nndeploy_result_np - torch_result_np))
        mean_diff = np.mean(np.abs(nndeploy_result_np - torch_result_np))

        print(f"[INFO] Max diff: {max_diff:.6e}")
        print(f"[INFO] Mean diff: {mean_diff:.6e}")

        # 精度标准：float32  atol=1e-5, rtol=1e-4
        passed = np.allclose(nndeploy_result_np, torch_result_np, atol=1e-5, rtol=1e-4)
        print(f"[{'PASS' if passed else 'FAIL'}] Accuracy check")

        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
