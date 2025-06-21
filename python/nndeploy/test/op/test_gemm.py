import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


class TestGemmOp(unittest.TestCase):

    def test_gemm0(self):
        input_shape = [64, 32]
        weight_shape = [32, 16]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)
        np_bias = np.random.random((input_shape[0], weight_shape[1])).astype(np.float32)

        torch_result = torch.matmul(
            torch.tensor(np_input),
            torch.tensor(np_weight),
        )
        torch_result = torch.add(torch_result, torch.tensor(np_bias))

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)
        bias = create_tensor_from_numpy(np_bias)

        nndeploy_result = F.gemm(input, weight, bias)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_gemm_bias_broadcast(self):
        input_shape = [64, 32]
        weight_shape = [32, 16]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)
        np_bias = np.random.random((1, weight_shape[1])).astype(np.float32)

        torch_result = torch.matmul(
            torch.tensor(np_input),
            torch.tensor(np_weight),
        )
        torch_result = torch.add(torch_result, torch.tensor(np_bias))

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)
        bias = create_tensor_from_numpy(np_bias)

        nndeploy_result = F.gemm(input, weight, bias)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_gemm_alpha(self):
        input_shape = [64, 32]
        weight_shape = [32, 16]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)

        alpha = 2.0
        torch_result = torch.matmul(
            torch.tensor(np_input),
            torch.tensor(np_weight),
        )
        torch_result = alpha * torch_result

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)

        nndeploy_result = F.gemm(input, weight, alpha=alpha)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_gemm_beta(self):
        input_shape = [64, 32]
        weight_shape = [32, 16]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)
        np_bias = np.random.random((input_shape[0], weight_shape[1])).astype(np.float32)

        beta = 0.5
        torch_result = torch.matmul(
            torch.tensor(np_input),
            torch.tensor(np_weight),
        )
        torch_result = torch.add(torch_result, beta * torch.tensor(np_bias))

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)
        bias = create_tensor_from_numpy(np_bias)

        nndeploy_result = F.gemm(input, weight, bias, beta=beta)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_gemm_trans_a(self):
        input_shape = [32, 64]  # Note the change in shape for trans_a
        weight_shape = [32, 16]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)

        torch_result = torch.matmul(
            torch.tensor(np_input).T,  # Transpose input
            torch.tensor(np_weight),
        )

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)

        nndeploy_result = F.gemm(input, weight, trans_a=1)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_gemm_trans_b(self):
        input_shape = [64, 32]
        weight_shape = [16, 32]  # Note the change in shape for trans_b

        np_input = np.random.random(input_shape).astype(np.float32)
        np_weight = np.random.random(weight_shape).astype(np.float32)

        torch_result = torch.matmul(
            torch.tensor(np_input),
            torch.tensor(np_weight).T,  # Transpose weight
        )

        input = create_tensor_from_numpy(np_input)
        weight = create_tensor_from_numpy(np_weight)

        nndeploy_result = F.gemm(input, weight, trans_b=1)

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
