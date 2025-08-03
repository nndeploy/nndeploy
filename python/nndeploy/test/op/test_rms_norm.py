import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


def rsqrt_cpu(x):
    return 1.0 / np.sqrt(x) if x > 0 else np.nan


def CPU_fused_resid_and_RMSNorm(h_decoder_out, h_scale, eps, hidden_units, num_tokens):
    for b in range(num_tokens):
        mean = np.mean(
            h_decoder_out[b * hidden_units:(b + 1) * hidden_units])  # 计算均值
        inv_fenmu = rsqrt_cpu(mean + eps)  # 计算逆平方根
        h_decoder_out[b * hidden_units:(b + 1) *
                      hidden_units] *= inv_fenmu * h_scale  # 缩放


class TestRmsNormOp(unittest.TestCase):

    def test_rms_norm(self):
        num_tokens = 32
        hidden_units = 4096

        np_in1 = np.random.random(
            (num_tokens, hidden_units)).astype(np.float32)
        np_out = np_in1.copy()
        np_in2 = np.random.random(hidden_units).astype(np.float32)
        np_in3 = np.random.random(
            (num_tokens, hidden_units)).astype(np.float32)

        CPU_fused_resid_and_RMSNorm(
            np_out, np_in2, 1e-6, hidden_units, num_tokens)

        tensor1 = create_tensor_from_numpy(np_in1).to(nndeploy.base.DeviceType("cuda"))
        tensor2 = create_tensor_from_numpy(np_in2).to(nndeploy.base.DeviceType("cuda"))
        tensor3 = create_tensor_from_numpy(np_in3).to(nndeploy.base.DeviceType("cuda"))

        nndeploy_result = nndeploy.op.rms_norm(tensor1, tensor2, tensor3)

        self.assertTrue(np.allclose(np_out, create_numpy_from_tensor(nndeploy_result), rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
