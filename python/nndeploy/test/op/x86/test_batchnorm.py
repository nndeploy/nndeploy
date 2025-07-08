import unittest  
import numpy as np  
import torch  
import nndeploy  
from nndeploy.op import functional as F  
from nndeploy.device.tensor import (  
    create_tensor_from_numpy,  
    create_numpy_from_tensor,  
)  
  
class TestBatchNormOp(unittest.TestCase):  
  
    def test_batch_norm_with_scale_and_shift(self):  
        input_shape = [2, 3, 4, 4]  
        channel_size = input_shape[1]  
          
        np.random.seed(123)
        np_input = np.random.uniform(-1, 1, input_shape).astype(np.float32)  
        np_scale = np.random.uniform(0.5, 1.5, channel_size).astype(np.float32)  
        np_shift = np.random.uniform(-0.5, 0.5, channel_size).astype(np.float32)  
        np_mean = np.random.uniform(-0.1, 0.1, channel_size).astype(np.float32)  
        np_var = np.random.uniform(0.8, 1.2, channel_size).astype(np.float32)  
          
        torch_result = torch.nn.functional.batch_norm(  
            torch.from_numpy(np_input),  
            torch.from_numpy(np_mean),  
            torch.from_numpy(np_var),  
            torch.from_numpy(np_scale),  
            torch.from_numpy(np_shift),  
            training=False,  
            eps=1e-5  
        )  
          
        input_tensor = create_tensor_from_numpy(np_input)  
        scale_tensor = create_tensor_from_numpy(np_scale)  
        shift_tensor = create_tensor_from_numpy(np_shift)  
        mean_tensor = create_tensor_from_numpy(np_mean)  
        var_tensor = create_tensor_from_numpy(np_var)  
          
        cpu_input = input_tensor.to(nndeploy.base.DeviceType("x86"))  
        cpu_scale = scale_tensor.to(nndeploy.base.DeviceType("x86"))  
        cpu_shift = shift_tensor.to(nndeploy.base.DeviceType("x86"))  
        cpu_mean = mean_tensor.to(nndeploy.base.DeviceType("x86"))  
        cpu_var = var_tensor.to(nndeploy.base.DeviceType("x86"))  
          
        x86_result = F.batch_norm(cpu_input, cpu_scale, cpu_shift, cpu_mean, cpu_var)  
        nndeploy_result = x86_result.to(nndeploy.base.DeviceType("cpu"))
        self.assertTrue(  
            np.allclose(  
                torch_result.detach().numpy(),  
                create_numpy_from_tensor(nndeploy_result),  
                rtol=1e-04,  
                atol=1e-05,  
            )  
        )  
  
if __name__ == "__main__":  
    unittest.main()