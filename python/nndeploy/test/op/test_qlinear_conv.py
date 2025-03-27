import unittest
import numpy as np
import onnx
from onnx.reference.ops.op_qlinear_conv import QLinearConv as OnnxQLinearConv
import nndeploy
from nndeploy.op import functional as F
from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor


class TestQLinearConvOp(unittest.TestCase):
    # 该测试用例参考onnx
    def test_qlinear_conv(self):

        x = np.array(
            [
                [255, 174, 162, 25, 203, 168, 58],
                [15, 59, 237, 95, 129, 0, 64],
                [56, 242, 153, 221, 168, 12, 166],
                [232, 178, 186, 195, 237, 162, 237],
                [188, 39, 124, 77, 80, 102, 43],
                [127, 230, 21, 83, 41, 40, 134],
                [255, 154, 92, 141, 42, 148, 247],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))

        x_scale = np.float32(0.00369204697)
        x_zero_point = np.uint8(132)

        w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))

        w_scale = np.array([0.00172794575], dtype=np.float32)
        w_zero_point = np.uint8(255)

        y_scale = np.float32(0.00162681262)
        y_zero_point = np.uint8(123)

        expected_output = np.array(
            [
                [0, 81, 93, 230, 52, 87, 197],
                [240, 196, 18, 160, 126, 255, 191],
                [199, 13, 102, 34, 87, 243, 89],
                [23, 77, 69, 60, 18, 93, 18],
                [67, 216, 131, 178, 175, 153, 212],
                [128, 25, 234, 172, 214, 215, 121],
                [0, 101, 163, 114, 213, 107, 8],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))

        x_tensor = create_tensor_from_numpy(x)
        x_scale_tensor = create_tensor_from_numpy(np.array([x_scale], dtype=np.float32))
        x_zero_point_tensor = create_tensor_from_numpy(
            np.array([x_zero_point], dtype=np.uint8)
        )
        w_tensor = create_tensor_from_numpy(w)
        w_scale_tensor = create_tensor_from_numpy(np.array([w_scale], dtype=np.float32))
        w_zero_point_tensor = create_tensor_from_numpy(
            np.array([w_zero_point], dtype=np.uint8)
        )
        y_scale_tensor = create_tensor_from_numpy(np.array([y_scale], dtype=np.float32))
        y_zero_point_tensor = create_tensor_from_numpy(
            np.array([y_zero_point], dtype=np.uint8)
        )

        output_tensor = F.qlinear_conv(
            x_tensor,
            x_scale_tensor,
            x_zero_point_tensor,
            w_tensor,
            w_scale_tensor,
            w_zero_point_tensor,
            y_scale_tensor,
            y_zero_point_tensor,
        )

        output = create_numpy_from_tensor(output_tensor)

        self.assertTrue(np.allclose(output, expected_output, rtol=1e-05, atol=1e-05))


if __name__ == "__main__":
    unittest.main()
