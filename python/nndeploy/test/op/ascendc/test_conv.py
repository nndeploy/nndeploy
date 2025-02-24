import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import name_to_device_type_code
from nndeploy.test.test_util import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
    name_to_device_type_code,
)

class TestConvOp(unittest.TestCase):

    def test_conv_without_bias_0(self):
        fm_shape = [1, 3, 320, 320]
        we_shape = [3, 3, 3, 3]

        np_fm = np.random.uniform(2, 5, fm_shape).astype(np.float16)
        np_we = np.random.uniform(2, 5, we_shape).astype(np.float16)

        # torch_result = torch.nn.functional.conv2d(
        #     torch.tensor(np_fm), torch.tensor(np_we)
        # )
        torch_result = torch.nn.functional.conv2d(
            torch.from_numpy(np_fm), 
            torch.from_numpy(np_we), 
            bias=None, 
            stride=1, 
            padding=[0, 0], 
            dilation=1
        )

        fm = create_tensor_from_numpy(np_fm)
        we = create_tensor_from_numpy(np_we)

        ascend_fm = fm.to(nndeploy.base.DeviceType("ascendcl")) 
        ascend_we = we.to(nndeploy.base.DeviceType("ascendcl"))

        ascend_result = F.conv(ascend_fm, ascend_we)

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

        # ascend_input1_array = create_numpy_from_tensor(nndeploy_result)
        # diff_input = ascend_input1_array - torch_result.detach().numpy()
        # print("nndeploy和onnx输入的差异:", diff_input)

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