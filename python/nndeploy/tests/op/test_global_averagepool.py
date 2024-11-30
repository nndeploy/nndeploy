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


class TestGlobalAveragePoolOp(unittest.TestCase):

    def test_global_averagepool(self):
        input_shape = [32, 64, 32, 32]

        np_input = np.random.random(input_shape).astype(np.float32)

        torch_result = torch.nn.functional.adaptive_avg_pool2d(
            torch.tensor(np_input),
            output_size=(1, 1),
        )

        input = createTensorFromNumpy(np_input)

        nndeploy_result = F.global_averagepool(input)

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
