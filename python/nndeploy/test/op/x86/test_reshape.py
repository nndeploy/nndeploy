import unittest  
import numpy as np  
import torch  
import nndeploy  
from nndeploy.op import functional as F  
from nndeploy.device.tensor import (  
    create_tensor_from_numpy,  
    create_numpy_from_tensor,  
)  
  
class TestReshapeOp(unittest.TestCase):

    def test_reshape_op(self):
        input_shape = [2, 3, 4, 5]
        new_shape = [6, 4, 5]  

        np.random.seed(123)  # for reproducibility
        np_input = np.random.uniform(-10, 10, input_shape).astype(np.float32)

        torch_input = torch.from_numpy(np_input)
        torch_result = torch.reshape(torch_input, new_shape)

        input_tensor = create_tensor_from_numpy(np_input)
        input_shape = create_tensor_from_numpy(np.array(new_shape).astype(np.int64))

        x86_input = input_tensor.to(nndeploy.base.DeviceType("x86"))
        x86_shape = input_shape.to(nndeploy.base.DeviceType("x86"))

        x86_result = F.reshape(x86_input, x86_shape)

        nndeploy_result_cpu = x86_result.to(nndeploy.base.DeviceType("cpu"))

        nndeploy_result_np = create_numpy_from_tensor(nndeploy_result_cpu)

        print("torch")
        print(torch_result.numpy())
        print("nndeploy")
        print(nndeploy_result_np)

        self.assertTrue(
            np.allclose(
                torch_result.detach().numpy(),
                nndeploy_result_np,
                rtol=1e-04,
                atol=1e-05,
            ),
        )

        self.assertEqual(list(nndeploy_result_np.shape), new_shape, "Output shape after reshape is incorrect!")

  
if __name__ == "__main__":  
    unittest.main()