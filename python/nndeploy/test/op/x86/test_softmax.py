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


class TestSoftmaxOp(unittest.TestCase):

    def test_softmax(self):
        matA_shape = [3, 1000]

        np_matA = np.random.uniform(2, 5, matA_shape).astype(np.float32)

        axis = 1
        torch_result = torch.softmax(
            torch.from_numpy(np_matA),
            dim=axis
        )

        matA = create_tensor_from_numpy(np_matA)

        x86_matA = matA.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.softmax(x86_matA, axis)

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
