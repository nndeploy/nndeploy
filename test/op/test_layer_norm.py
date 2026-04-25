import unittest
import numpy as np

from nndeploy.op import functional as F
from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


def numpy_layer_norm(x, weight, bias=None, epsilon=1e-5):
    normalized_rank = len(weight.shape)
    norm_axes = tuple(range(x.ndim - normalized_rank, x.ndim))
    mean = np.mean(x, axis=norm_axes, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=norm_axes, keepdims=True)

    normalized = (x - mean) / np.sqrt(var + epsilon)
    scaled = normalized * weight
    if bias is not None:
        scaled = scaled + bias
    return scaled


class TestLayerNormOp(unittest.TestCase):
    def test_layer_norm_last_dim(self):
        np.random.seed(42)
        np_input = np.random.uniform(-1, 1, [2, 3, 8]).astype(np.float32)
        np_weight = np.random.uniform(0.5, 1.5, [8]).astype(np.float32)
        np_bias = np.random.uniform(-0.2, 0.2, [8]).astype(np.float32)

        input_tensor = create_tensor_from_numpy(np_input)
        weight_tensor = create_tensor_from_numpy(np_weight)
        bias_tensor = create_tensor_from_numpy(np_bias)

        nndeploy_result = F.layer_norm(input_tensor, weight_tensor, bias_tensor)
        np_result = create_numpy_from_tensor(nndeploy_result)

        np_ref = numpy_layer_norm(np_input, np_weight, np_bias)
        self.assertTrue(np.allclose(np_result, np_ref, rtol=1e-5, atol=1e-5))

    def test_layer_norm_multi_dim_no_bias(self):
        np.random.seed(123)
        np_input = np.random.uniform(-1, 1, [2, 3, 4]).astype(np.float32)
        np_weight = np.random.uniform(0.5, 1.5, [3, 4]).astype(np.float32)

        input_tensor = create_tensor_from_numpy(np_input)
        weight_tensor = create_tensor_from_numpy(np_weight)

        nndeploy_result = F.layer_norm(input_tensor, weight_tensor)
        np_result = create_numpy_from_tensor(nndeploy_result)

        np_ref = numpy_layer_norm(np_input, np_weight)
        self.assertTrue(np.allclose(np_result, np_ref, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
