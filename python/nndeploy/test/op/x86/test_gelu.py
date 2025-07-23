import unittest
import numpy as np
import torch
import torch.nn.functional as torch_F 
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import name_to_device_type_code
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
)


class TestGeluOp(unittest.TestCase):

    def test_gelu(self):
        src_shape = [1, 3, 16, 1000]
        torch.slice_copy

        np_src = np.random.uniform(-5, 5, src_shape).astype(np.float32)

        torch_result = torch_F.gelu(
            torch.from_numpy(np_src)
        )

        src = create_tensor_from_numpy(np_src)

        x86_src = src.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.gelu(x86_src)

        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))
        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                create_numpy_from_tensor(nndeploy_result),
                rtol=1e-02,
                atol=1e-04,
            ),
        )


if __name__ == "__main__":
    unittest.main()