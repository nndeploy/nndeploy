import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor


class TestMaxPoolOp(unittest.TestCase):

    def test_maxpool0(self):
        input_shape = [32, 4, 16, 16]
        kernel_size = 2
        stride = 2
        padding = 0
        dilation = 1
        ceil_mode = False

        np_input = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.nn.functional.max_pool2d(
            torch.tensor(np_input),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

        input = create_tensor_from_numpy(np_input)

        nndeploy_result = F.maxpool(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_maxpool1(self):
        input_shape = [32, 4, 16, 16]
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        ceil_mode = True

        np_input = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.nn.functional.max_pool2d(
            torch.tensor(np_input),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

        input = create_tensor_from_numpy(np_input)

        nndeploy_result = F.maxpool(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )


if __name__ == "__main__":
    unittest.main()
