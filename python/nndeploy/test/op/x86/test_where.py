import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.base import name_to_device_type_code
from nndeploy.device.tensor import (
    create_tensor_from_numpy,
    create_numpy_from_tensor,
)

class TestWhereOp(unittest.TestCase):

    def test_where(self):
        shape = [1, 3,4, 5]

        np.random.seed(123)

        np_condition = np.random.choice([True, False], size=shape)
        np_x = np.random.randn(*shape).astype(np.float32)
        np_y = np.random.randn(*shape).astype(np.float32)

        torch_result = torch.where(
            torch.from_numpy(np_condition),
            torch.from_numpy(np_x),
            torch.from_numpy(np_y)
        )
        
        np_condition = np_condition.astype(np.int8)
        condition = create_tensor_from_numpy(np_condition)

        x = create_tensor_from_numpy(np_x)
        y = create_tensor_from_numpy(np_y)



        x86_condition = condition.to(nndeploy.base.DeviceType("x86"))
        x86_x = x.to(nndeploy.base.DeviceType("x86"))
        x86_y = y.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.where(x86_x, x86_y, x86_condition)

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