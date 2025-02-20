import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test.test_util import create_tensor_from_numpy, createNumpyFromTensor


class TestFlattenOp(unittest.TestCase):

    def test_flatten0(self):
        input_shape = [32, 64, 32, 32]
        axis = 1

        np_input = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.flatten(torch.tensor(np_input), start_dim=axis)

        input = create_tensor_from_numpy(np_input)

        nndeploy_result = F.flatten(input, axis)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                createNumpyFromTensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )

    def test_flatten1(self):
        input_shape = [32, 64, 32, 32]
        axis = 2

        np_input = np.random.random(input_shape).astype(np.float32)

        # onnx总是将tensor flatten成2D
        torch_result = torch.flatten(
            torch.tensor(np_input), start_dim=0, end_dim=axis - 1
        )
        torch_result = torch.flatten(torch_result, start_dim=1)
        input = create_tensor_from_numpy(np_input)

        nndeploy_result = F.flatten(input, axis)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                createNumpyFromTensor(nndeploy_result),
                rtol=1e-03,
                atol=1e-04,
            )
        )


if __name__ == "__main__":
    unittest.main()
