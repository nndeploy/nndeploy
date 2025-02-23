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

class TestMulOp(unittest.TestCase):

    def test_mul(self):
        fm_shape = [1, 3, 256, 256]
        we_shape = [3, 3, 3, 3]

        np_fm = np.random.uniform(2, 5, fm_shape).astype(np.float16)
        np_we = np.random.uniform(2, 5, we_shape).astype(np.float16)

        torch_result = torch.mul(
            torch.tensor(np_fm),
            torch.tensor(np_we),
        )

        fm = create_tensor_from_numpy(np_fm.transpose(0, 2, 3, 1))
        we = create_tensor_from_numpy(np_we.transpose(0, 2, 3, 1))

        ascend_fm = fm.to(nndeploy.base.DeviceType("ascendcl")) 
        ascend_we = we.to(nndeploy.base.DeviceType("ascendcl"))

        ascend_result = F.conv(ascend_fm, ascend_we)

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

        # ascend_input1_array = create_numpy_from_tensor(ascend_input1)
        # diff_input = ascend_input1_array - np_input1
        # print("nndeploy和onnx输入的差异:", diff_input)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result)transpose(0, 3, 1, 2),
                rtol=1e-03,
                atol=1e-04,
            )
        )

if __name__ == "__main__":
    unittest.main()