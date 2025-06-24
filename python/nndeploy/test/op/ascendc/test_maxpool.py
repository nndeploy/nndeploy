import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import name_to_device_type_code
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
    name_to_device_type_code,
)

class TestMaxPoolOp(unittest.TestCase):

    def test_maxpool(self):
        input_shape = [1, 3, 320, 320]
        kernel_size = 3
        stride = 1
        padding = 0
        dilation = 1
        ceil_mode = False

        np_input = np.random.uniform(1, 2, input_shape).astype(np.float16)

        torch_result = torch.nn.functional.max_pool2d(
            torch.tensor(np_input),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

        input = create_tensor_from_numpy(np_input)
        ascend_input = input.to(nndeploy.base.DeviceType("ascendcl"))
        

        ascend_result = F.maxpool(
            ascend_input, kernel_size, stride, padding, dilation, ceil_mode
        )

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

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