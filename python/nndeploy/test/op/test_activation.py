import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


class TestActivation(unittest.TestCase):

    def test_relu_0(self):
        input_shape = [32, 4, 16, 16]

        np_input = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.nn.functional.relu(torch.tensor(np_input))

        input = create_tensor_from_numpy(np_input)

        nndeploy_result = F.relu(input)

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
