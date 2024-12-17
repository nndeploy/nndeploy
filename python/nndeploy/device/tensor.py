import nndeploy
import numpy as np
import nndeploy._nndeploy_internal as _C
from nndeploy.base import device_name_to_code

# 从numpy array返回一个Tensor
def createTensorFromNumpy(np_data, device="cpu"):
    tensor = _C.device.Tensor(np_data, device_name_to_code["cpu"])
    if device != "cpu":
        device_tensor = tensor.to(device_name_to_code[device])
        return device_tensor
    return tensor

# 从Tensor返回一个numpy array
def createNumpyFromTensor(tensor):
    return np.array(tensor.to(device_name_to_code["cpu"]))