
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

from PIL import Image
import numpy as np
from typing import List

import numpy as np
import cv2

class MakeNumpyGrid(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.MakeNumpyGrid")
        self.set_desc("Concatenate multiple NumPy images (ndarray) into a grid")
        self.rows = 1
        self.cols = 1
        self.resize_h = -1
        self.resize_w = -1
        self.set_input_type(np.ndarray)    # 动态输入：numpy.ndarray
        self.set_output_type(np.ndarray)
        self.set_dynamic_input(True)

    def _ensure_3d(self, arr: np.ndarray) -> np.ndarray:
        """将 HxW 转为 HxWx1, 其他保持不变。"""
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"输入数组维度应为2或3,得到 {arr.ndim}")

    def _resize_np(self, arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        """使用 cv2.resize 统一尺寸。"""
        arr3 = self._ensure_3d(arr)
        h, w, _ = arr3.shape
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
        out = cv2.resize(arr3, dsize=(new_w, new_h), interpolation=interp)
        if out.ndim == 2:  # 单通道情况下，cv2 可能返回 HxW
            out = out[..., None]
        return out

    def run(self) -> bool:
        try:
            # 收集所有输入边的 ndarray
            arrays: List[np.ndarray] = []
            for i in range(len(self.get_all_input())):
                edge = self.get_input(i)
                arr = edge.get(self)
                if arr is not None:
                    if not isinstance(arr, np.ndarray):
                        print(f"MakeNumpyGrid: 输入类型必须为 numpy.ndarray, 得到 {type(arr)}")
                        return nndeploy.base.Status.error()
                    arrays.append(arr)

            expected = self.rows * self.cols
            if len(arrays) != expected:
                print(f"MakeNumpyGrid: 输入图像数量({len(arrays)})与网格(rows*cols={expected})不符")
                return nndeploy.base.Status.error()

            # 可选 resize，否则确保尺寸/通道一致
            if self.resize_h != -1 and self.resize_w != -1:
                arrays = [self._resize_np(a, self.resize_w, self.resize_h) for a in arrays]
            else:
                arrays = [self._ensure_3d(a) for a in arrays]

            # 校验尺寸一致
            h0, w0, c0 = arrays[0].shape
            for idx, a in enumerate(arrays):
                if a.shape != (h0, w0, c0):
                    print(f"MakeNumpyGrid: 第 {idx} 个输入尺寸不一致，得到 {a.shape}，期望 {(h0, w0, c0)}")
                    return nndeploy.base.Status.error()

            # 预分配输出 (rows*h, cols*w, C)
            grid_h = self.rows * h0
            grid_w = self.cols * w0
            out = np.zeros((grid_h, grid_w, c0), dtype=arrays[0].dtype)

            # 逐格拷贝
            for i, a in enumerate(arrays):
                r = i // self.cols
                c = i % self.cols
                y0, y1 = r * h0, (r + 1) * h0
                x0, x1 = c * w0, (c + 1) * w0
                out[y0:y1, x0:x1, :] = a

            # 若是单通道，输出为 HxW
            if c0 == 1:
                out = out[..., 0]

            # 输出到输出边
            output_edge = self.get_output(0)
            output_edge.set(out)
            return nndeploy.base.Status.ok()

        except Exception as e:
            print(f"MakeNumpyGrid 节点运行失败: {e}")
            return nndeploy.base.Status.error()

    def serialize(self):
        self.add_required_param("rows")
        self.add_required_param("cols")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["rows"] = self.rows
        json_obj["cols"] = self.cols
        json_obj["resize_h"] = self.resize_h
        json_obj["resize_w"] = self.resize_w
        return json.dumps(json_obj, ensure_ascii=False, indent=2)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "rows" in json_obj:
                self.rows = int(json_obj["rows"])
            if "cols" in json_obj:
                self.cols = int(json_obj["cols"])
            if "resize_h" in json_obj:
                self.resize_h = int(json_obj["resize_h"])
            if "resize_w" in json_obj:
                self.resize_w = int(json_obj["resize_w"])
            return super().deserialize(target)
        except Exception as e:
            print(f"MakeNumpyGrid 节点反序列化失败: {e}")
            return nndeploy.base.Status.error()


class MakeNumpyGridCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = MakeNumpyGrid(name, inputs, outputs)
        return self.node


make_numpy_grid_node_creator = MakeNumpyGridCreator()
nndeploy.dag.register_node("nndeploy.codec.MakeNumpyGrid", make_numpy_grid_node_creator)