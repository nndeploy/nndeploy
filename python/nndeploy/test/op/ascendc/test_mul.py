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
        input_shape = [32, 4, 16, 16]

        np_input1 = np.random.random(input_shape).astype(np.float32)
        np_input2 = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.mul(
            torch.tensor(np_input1),
            torch.tensor(np_input2),
        )

        input1 = create_tensor_from_numpy(np_input1)
        input2 = create_tensor_from_numpy(np_input2)

        ascend_input1 = input1.to(nndeploy.base.DeviceType("ascendcl")) 
        ascend_input2 = input2.to(nndeploy.base.DeviceType("ascendcl"))

        ascend_result = F.mul(ascend_input1, ascend_input2)

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

        # ascend_input1_array = create_numpy_from_tensor(ascend_input1)
        # diff_input = ascend_input1_array - np_input1
        # print("nndeploy和onnx输入的差异:", diff_input)

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