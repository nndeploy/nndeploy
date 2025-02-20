import nndeploy
import numpy as np

from nndeploy.device.tensor import create_numpy_from_tensor, create_tensor_from_numpy
from nndeploy.base.common import name_to_device_type_code


str_to_np_data_types = {
    'float32': np.float32,
    'float16': np.float16
}


