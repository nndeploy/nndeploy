import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


class TestConvOp(unittest.TestCase):

    def test_conv_without_bias_0(self):
        input_shape = [1, 16, 64, 64]
        weight_shape = [64, 16, 3, 3]

        np_input = np.random.random(input_shape).astype(np.float16)

        np_weight = np.random.random(weight_shape).astype(np.float16)

        torch_result = torch.nn.functional.conv2d(
            torch.tensor(np_input), torch.tensor(np_weight)
        )

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)

        ascend_input = input.to(nndeploy.base.DeviceType("ascendcl"))
        ascend_weight = weight.to(nndeploy.base.DeviceType("ascendcl"))

        ascend_result = F.conv(ascend_input, ascend_weight)

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

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
