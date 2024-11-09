import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test_utils import (
    createTensorFromNumpy,
    createNumpyFromTensor,
    device_name_to_code,
)


class TestConvOp(unittest.TestCase):

    def test_conv_without_bias_0(self):
        input_shape = [1, 32, 64, 64]
        weight_shape = [64, 32, 3, 3]

        np_input = np.random.random(input_shape).astype(np.float32)

        np_weight = np.random.random(weight_shape).astype(np.float32)

        torch_result = torch.nn.functional.conv2d(
            torch.tensor(np_input), torch.tensor(np_weight)
        )

        input = createTensorFromNumpy(np_input)
        weight = createTensorFromNumpy(np_weight)

        nndeploy_result = F.conv(input, weight)
        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                createNumpyFromTensor(nndeploy_result),
                rtol=1e-05,
                atol=1e-08,
            )
        )

    def test_conv_with_bias_0(self):
        input_shape = [1, 32, 64, 64]
        weight_shape = [64, 32, 3, 3]

        np_input = np.random.random(input_shape).astype(np.float32)

        np_weight = np.random.random(weight_shape).astype(np.float32)

        np_bias = np.random.random(weight_shape[0]).astype(np.float32)

        torch_result = torch.nn.functional.conv2d(
            torch.tensor(np_input), torch.tensor(np_weight), torch.tensor(np_bias)
        )

        input = createTensorFromNumpy(np_input)
        weight = createTensorFromNumpy(np_weight)
        bias = createTensorFromNumpy(np_bias)

        nndeploy_result = F.conv(input, weight, bias)
        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                createNumpyFromTensor(nndeploy_result),
                rtol=1e-05,
                atol=1e-08,
            )
        )


if __name__ == "__main__":
    unittest.main()
