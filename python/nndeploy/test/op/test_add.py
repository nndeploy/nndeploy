import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor

class TestAddOp(unittest.TestCase):

    def test_add(self):
        input_shape = [32, 4, 16, 16]

        np_input1 = np.random.random(input_shape).astype(np.float32)
        np_input2 = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.add(
            torch.tensor(np_input1),
            torch.tensor(np_input2),
        )

        input1 = create_tensor_from_numpy(np_input1)
        input2 = create_tensor_from_numpy(np_input2)

        nndeploy_result = F.add(input1, input2)

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