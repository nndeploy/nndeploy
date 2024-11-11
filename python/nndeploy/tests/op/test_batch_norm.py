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


class TestBatchNormOp(unittest.TestCase):

    def test_batch_norm_0(self):
        input_shape = [32, 4, 16, 16]
        weight_shape = input_shape[1]

        np_input = np.random.random(input_shape).astype(np.float32)
        np_scale = np.random.random(weight_shape).astype(np.float32)
        np_bias = np.random.random(weight_shape).astype(np.float32)
        np_mean = np.random.random(weight_shape).astype(np.float32)
        np_var = np.random.random(weight_shape).astype(np.float32)

        torch_result = torch.nn.functional.batch_norm(
            torch.tensor(np_input),
            torch.tensor(np_mean),
            torch.tensor(np_var),
            torch.tensor(np_scale),
            torch.tensor(np_bias),
        )

        input = createTensorFromNumpy(np_input)
        scale = createTensorFromNumpy(np_scale)
        bias = createTensorFromNumpy(np_bias)
        mean = createTensorFromNumpy(np_mean)
        var = createTensorFromNumpy(np_var)

        nndeploy_result = F.batch_norm(input, scale, bias, mean, var)

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
