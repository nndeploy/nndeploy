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


class TestTransposeOp(unittest.TestCase):
    def test_transpose(self):
        shape = [1, 3, 4, 5]
        perm = [0, 1, 3, 2]
        perm_axis = [2, 3]

        np.random.seed(123)
        np_input = np.random.randn(*shape).astype(np.float32)

        torch_result = torch.from_numpy(np_input).permute(perm)

        input_tensor = create_tensor_from_numpy(np_input)

        x86_input = input_tensor.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.transpose(x86_input, perm)

        nndeploy_result_numpy = create_numpy_from_tensor(
            x86_result.to(nndeploy.base.DeviceType("cpu"))
        )

        self.assertTrue(
            np.allclose(
                torch_result.numpy(),
                nndeploy_result_numpy,
                rtol=1e-05,
                atol=1e-08,
            ),
        )


if __name__ == "__main__":
    unittest.main()