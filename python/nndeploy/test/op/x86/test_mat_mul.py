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


class TestMatMulOp(unittest.TestCase):

    def test_2d_matmul_without_bias(self):
        M, K, N = 3, 4, 5
        matA_shape = [M, K]
        matB_shape = [K, N]

        np.random.seed(123)
        np_matA = np.random.uniform(-5, 5, matA_shape).astype(np.float32)
        np_matB = np.random.uniform(-5, 5, matB_shape).astype(np.float32)

        torch_result = torch.matmul(
            torch.from_numpy(np_matA),
            torch.from_numpy(np_matB)
        )

        matA = create_tensor_from_numpy(np_matA)
        matB = create_tensor_from_numpy(np_matB)

        x86_matA = matA.to(nndeploy.base.DeviceType("x86"))
        x86_matB = matB.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.mat_mul(x86_matA, x86_matB)

        nndeploy_result_np = create_numpy_from_tensor(
            x86_result.to(nndeploy.base.DeviceType("cpu"))
        )

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                nndeploy_result_np,
                rtol=1e-02, 
            ),
        )
    

    def test_2d_matmul_with_bias(self):
        M, K, N = 3, 4, 5
        matA_shape = [M, K]
        matB_shape = [K, N]
        bias_shape = [1, N] 

        np.random.seed(456) 
        np_matA = np.random.uniform(-5, 5, matA_shape).astype(np.float32)
        np_matB = np.random.uniform(-5, 5, matB_shape).astype(np.float32)
        np_bias = np.random.uniform(-2, 2, bias_shape).astype(np.float32) 

        torch_result = torch.matmul(
            torch.from_numpy(np_matA),
            torch.from_numpy(np_matB)
        ) + torch.from_numpy(np_bias) 

        matA = create_tensor_from_numpy(np_matA)
        matB = create_tensor_from_numpy(np_matB)
        bias = create_tensor_from_numpy(np_bias) 

        x86_matA = matA.to(nndeploy.base.DeviceType("x86"))
        x86_matB = matB.to(nndeploy.base.DeviceType("x86"))
        x86_bias = bias.to(nndeploy.base.DeviceType("x86")) 

        x86_result = F.mat_mul(x86_matA, x86_matB, x86_bias)

        nndeploy_result_np = create_numpy_from_tensor(
            x86_result.to(nndeploy.base.DeviceType("cpu"))
        )

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                nndeploy_result_np,
                rtol=1e-02,
                atol=1e-04,
            ),
        )


if __name__ == "__main__":
    unittest.main()
