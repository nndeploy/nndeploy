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

    def test_concat_without_axis(self):
        matA_shape = [1, 6, 3, 3]
        matB_shape = [1, 6, 3, 3]

        np_matA = np.random.uniform(2, 5, matA_shape).astype(np.float32)
        np_matB = np.random.uniform(2, 5, matB_shape).astype(np.float32)

        torch_result = torch.cat(
            (torch.from_numpy(np_matA),
            torch.from_numpy(np_matB)), 
            dim=0
        )

        matA = create_tensor_from_numpy(np_matA)
        matB = create_tensor_from_numpy(np_matB)

        x86_matA = matA.to(nndeploy.base.DeviceType("x86"))
        x86_matB = matB.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.concat(x86_matA, x86_matB)

        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-02,
                atol=1e-04,
            )
        )


    def test_concat_with_axis(self):
        matA_shape = [6, 3, 3]
        matB_shape = [6, 3, 3]

        np_matA = np.random.uniform(2, 5, matA_shape).astype(np.float32)
        np_matB = np.random.uniform(2, 5, matB_shape).astype(np.float32)
        axis = 2

        torch_result = torch.cat(
            (torch.from_numpy(np_matA),
            torch.from_numpy(np_matB)), 
            dim=axis
        )

        matA = create_tensor_from_numpy(np_matA)
        matB = create_tensor_from_numpy(np_matB)

        x86_matA = matA.to(nndeploy.base.DeviceType("x86"))
        x86_matB = matB.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.concat(x86_matA, x86_matB, axis)

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
