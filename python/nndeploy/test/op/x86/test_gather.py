import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
)

class TestGatherOp(unittest.TestCase):

    def test_gather(self):
        data_shape = [1, 5, 10, 8]
        indices_shape = [1, 2, 10, 8]
        axis = 1

        np.random.seed(123)

        np_data = np.random.randn(*data_shape).astype(np.float32)
        max_index = data_shape[axis]
        np_indices = np.random.randint(0, max_index, size=indices_shape).astype(np.int64)

        torch_data = torch.from_numpy(np_data)
        torch_indices = torch.from_numpy(np_indices)
        torch_result = torch.gather(torch_data, axis, torch_indices)
        
        np_indices = np_indices.astype(np.int32)
        data = create_tensor_from_numpy(np_data)
        indices = create_tensor_from_numpy(np_indices)

        x86_data = data.to(nndeploy.base.DeviceType("x86"))
        x86_indices = indices.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.gather(x86_data, x86_indices, axis=axis)

        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))

        self.assertTrue(
            np.allclose(
                torch_result.numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-05,
                atol=1e-08,
            ),
        )

if __name__ == "__main__":
    unittest.main()