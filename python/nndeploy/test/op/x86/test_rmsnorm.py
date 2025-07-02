import unittest  
import numpy as np  
import torch  
import nndeploy  
from nndeploy.op import functional as F  
from nndeploy.device.tensor import (  
    create_tensor_from_numpy,  
    create_numpy_from_tensor,  
)  
  
class TestRMSNormOp(unittest.TestCase):  
  
    def test_rms_norm_with_weight(self):  
        input_shape = [2, 2, 4, 6]  
        normalized_shape = [input_shape[-1]] 

        np.random.seed(123)
        np_input = np.random.uniform(-1, 1, input_shape).astype(np.float32)  
        np_weight = np.random.uniform(0.5, 1.5, normalized_shape).astype(np.float32)  
          
        torch_result = torch.nn.functional.rms_norm(  
            torch.from_numpy(np_input),  
            normalized_shape,
            # torch.from_numpy(np_weight),  
            eps=1e-5  
        )  
          
        input_tensor = create_tensor_from_numpy(np_input)  
        weight_tensor = create_tensor_from_numpy(np_weight)  
          
        x86_input = input_tensor.to(nndeploy.base.DeviceType("x86"))  
        x86_weight = weight_tensor.to(nndeploy.base.DeviceType("x86"))  

        x86_result = F.rms_norm(x86_input, x86_weight)  
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