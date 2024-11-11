import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test_utils import (
    createTensorFromNumpy,
    createNumpyFromTensor,
    device_name_to_code,
)


class TestActivation(unittest.TestCase):

    def test_relu_0(self):
        input_shape = [32, 4, 16, 16]
        
        np_input = np.random.random(input_shape).astype(np.float32)
        
        torch_result = torch.nn.functional.relu(
            torch.tensor(np_input)
        )

        input = createTensorFromNumpy(np_input)

        nndeploy_result = F.relu(input)

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
