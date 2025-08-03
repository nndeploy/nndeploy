import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


class TestSplitOp(unittest.TestCase):

    def _test_split(self, np_input, split_arg, dim=0):
        torch_input = torch.tensor(np_input)

        # 1. PyTorch 调用（保持原语义）
        torch_outputs = torch.split(torch_input, split_arg, dim=dim)

        # 2. nndeploy 调用
        input_tensor = create_tensor_from_numpy(np_input)

        if isinstance(split_arg, int):
            # ✅ 将 PyTorch 的 split_size 转成 ONNX 的 section 张量
            axis_dim = np_input.shape[dim]
            sections = []
            start = 0
            while start < axis_dim:
                end = min(start + split_arg, axis_dim)
                sections.append(end - start)
                start = end
            split_tensor = create_tensor_from_numpy(np.array(sections, dtype=np.int64))
            nndeploy_outputs = F.split(input_tensor, section=split_tensor, axis=dim)
        else:
            # ✅ list → 直接对齐
            split_tensor = create_tensor_from_numpy(np.array(split_arg, dtype=np.int64))
            nndeploy_outputs = F.split(input_tensor, section=split_tensor, axis=dim)

        # 3. 断言长度和数值
        self.assertEqual(len(torch_outputs), len(nndeploy_outputs))
        for torch_out, nndeploy_out in zip(torch_outputs, nndeploy_outputs):
            np.testing.assert_allclose(
                torch_out.numpy(),
                create_numpy_from_tensor(nndeploy_out),
                rtol=1e-5,
                atol=1e-6
            )

    def test_split_num_outputs_even(self):
        np_input = np.random.rand(6, 4).astype(np.float32)
        self._test_split(np_input, 3, dim=0)

    def test_split_num_outputs_remainder(self):
        np_input = np.random.rand(7, 4).astype(np.float32)
        self._test_split(np_input, 3, dim=0)

    def test_split_num_outputs_dim1(self):
        np_input = np.random.rand(2, 9).astype(np.float32)
        self._test_split(np_input, 4, dim=1)

    def test_split_num_outputs_single_chunk(self):
        np_input = np.random.rand(1, 10).astype(np.float32)
        self._test_split(np_input, 1, dim=0)

    def test_split_num_outputs_last_chunk_small(self):
        np_input = np.random.rand(5, 3, 4).astype(np.float32)
        self._test_split(np_input, 4, dim=1)

    def test_split_sections_list(self):
        np_input = np.random.rand(10, 4).astype(np.float32)
        self._test_split(np_input, [2, 3, 5], dim=0)

    def test_split_sections_list_uneven(self):
        np_input = np.random.rand(6, 8).astype(np.float32)
        self._test_split(np_input, [1, 2, 5], dim=1)

    def test_split_sections_list_3d(self):
        np_input = np.random.rand(4, 6, 5).astype(np.float32)
        self._test_split(np_input, [1, 2, 3], dim=1)

    def test_split_sections_list_axis2(self):
        np_input = np.random.rand(2, 3, 7).astype(np.float32)
        self._test_split(np_input, [3, 4], dim=2)

    def test_split_single_element(self):
        np_input = np.array([[[1.0]]], dtype=np.float32)
        self._test_split(np_input, [1], dim=0)

    def test_split_empty_sections(self):
        np_input = np.random.rand(0, 5).astype(np.float32)
        self._test_split(np_input, [0, 0, 0], dim=0)

    def test_split_negative_axis(self):
        np_input = np.random.rand(3, 4, 5).astype(np.float32)
        self._test_split(np_input, 2, dim=-1)


if __name__ == "__main__":
    unittest.main()