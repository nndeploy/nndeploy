import unittest
import numpy as np
import itertools

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor, str_to_np_data_types

"""
测试nndeploy.device.Tensor 与  numpy array的互相转换
"""


#  生成全排列
def generate_permutations(shapes, data_types, devices):

    permutations = list(itertools.product(shapes, data_types, devices))
    return permutations


class TestNumpy(unittest.TestCase):

    def test_from_to_np(self):
        shape_list = [[32], [32, 32], [8, 16, 16], [4, 8, 8, 8]]
        data_types = ['float32', 'float16']
        # data_types = ['float32', 'float32']
        devices = ['cpu']

        for shape, data_type, device in generate_permutations(shape_list, data_types, devices):
            np_array = np.random.random(shape).astype(
                str_to_np_data_types[data_type])
            tensor = create_tensor_from_numpy(np_array)
            self.assertTrue(np.allclose(np_array, np.array(tensor), rtol=1e-05, atol=1e-08),
                            msg=f"Arrays are not close enough in case: shape={shape}, data_type={data_type}, device={device}")


if __name__ == '__main__':
    unittest.main()
