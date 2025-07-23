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

class TestSliceOp(unittest.TestCase):

    def _test_slice(self, shape, starts, ends, axes, steps):
        np.random.seed(123)
        np_input = np.random.randn(*shape).astype(np.float32)

        slices = [slice(None)] * len(shape)
        for i, axis in enumerate(axes):
            slices[axis] = slice(starts[i], ends[i], steps[i])
        
        torch_input = torch.from_numpy(np_input)
        torch_result = torch_input[tuple(slices)]

        np_starts = np.array(starts, dtype=np.int32)
        np_ends = np.array(ends, dtype=np.int32)
        np_axes = np.array(axes, dtype=np.int32)
        np_steps = np.array(steps, dtype=np.int32)

        nndeploy_input = create_tensor_from_numpy(np_input)
        nndeploy_starts = create_tensor_from_numpy(np_starts)
        nndeploy_ends = create_tensor_from_numpy(np_ends)
        nndeploy_axes = create_tensor_from_numpy(np_axes)
        nndeploy_steps = create_tensor_from_numpy(np_steps)

        x86_input = nndeploy_input.to(nndeploy.base.DeviceType("x86"))
        x86_starts = nndeploy_starts.to(nndeploy.base.DeviceType("x86"))
        x86_ends = nndeploy_ends.to(nndeploy.base.DeviceType("x86"))
        x86_axes = nndeploy_axes.to(nndeploy.base.DeviceType("x86"))
        x86_steps = nndeploy_steps.to(nndeploy.base.DeviceType("x86"))
        
        x86_result = F.slice(x86_input, x86_starts, x86_ends, x86_axes, x86_steps)

        nndeploy_result_cpu = x86_result.to(nndeploy.base.DeviceType("cpu"))
        
        nndeploy_np_result = create_numpy_from_tensor(nndeploy_result_cpu)

        # print("torch")
        # print(torch_result.numpy())
        # print("nndeploy")
        # print(nndeploy_np_result)

        self.assertTrue(
            np.allclose(
                torch_result.numpy(),
                nndeploy_np_result,
                rtol=1e-05,
                atol=1e-08,
            ),
        )

    def test_slice_basic(self):
        self._test_slice(
            shape=[1, 3, 4, 4],
            starts=[0, 0, 0, 0],
            ends=[1, 3, 112, 112],
            axes=[0, 1, 2, 3],
            steps=[1, 1, 1, 1] # 只支持step=1
        )


if __name__ == "__main__":
    unittest.main()