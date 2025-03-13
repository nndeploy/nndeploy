import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F

from nndeploy.test.test_util import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
    name_to_device_type_code,
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

        input = create_tensor_from_numpy(np_input)
        ascend_input = input.to(nndeploy.base.DeviceType("ascendcl"))
        scale = create_tensor_from_numpy(np_scale)
        bias = create_tensor_from_numpy(np_bias)
        mean = create_tensor_from_numpy(np_mean)
        var = create_tensor_from_numpy(np_var)

        ascend_result = F.batch_norm(ascend_input, scale, bias, mean, var)

        nndeploy_result = ascend_result.to(nndeploy.base.DeviceType("cpu"))

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
