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


class TestConvOp(unittest.TestCase):

    def test_conv_without_bias_0(self):
        fm_shape = [5, 3, 224, 224]
        we_shape = [6, 3, 3, 3]

        np_fm = np.random.uniform(2, 5, fm_shape).astype(np.float32)
        np_we = np.random.uniform(2, 5, we_shape).astype(np.float32)

        torch_result = torch.nn.functional.conv2d(
            torch.from_numpy(np_fm),
            torch.from_numpy(np_we),
            bias=None,
            stride=1,
            padding=[0, 0],
            dilation=1,
        )

        fm = create_tensor_from_numpy(np_fm)
        we = create_tensor_from_numpy(np_we)

        x86_fm = fm.to(nndeploy.base.DeviceType("x86"))
        x86_we = we.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.conv(x86_fm, x86_we)

        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-02,
                atol=1e-04,
            )
        )

    def test_conv_with_bias_0(self):
        fm_shape = [5, 3, 224, 224]
        we_shape = [6, 3, 3, 3]

        np_fm = np.random.uniform(-1, 1, fm_shape).astype(np.float32)
        np_we = np.random.uniform(0, 1, we_shape).astype(np.float32)
        np_bias = np.random.uniform(0, 5, we_shape[0]).astype(np.float32)

        torch_result = torch.nn.functional.conv2d(
            torch.from_numpy(np_fm),
            torch.from_numpy(np_we),
            torch.from_numpy(np_bias),
            stride=1,
            padding=[0, 0],
            dilation=1,
        )

        fm = create_tensor_from_numpy(np_fm)
        we = create_tensor_from_numpy(np_we)
        bias = create_tensor_from_numpy(np_bias)

        x86_fm = fm.to(nndeploy.base.DeviceType("x86"))
        x86_we = we.to(nndeploy.base.DeviceType("x86"))
        x86_bias = bias.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.conv(x86_fm, x86_we, x86_bias)

        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-02,
                atol=1e-04,
            )
        )


if __name__ == "__main__":
    unittest.main()
