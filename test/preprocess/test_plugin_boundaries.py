import gc
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from nndeploy.dag import Edge
from nndeploy.device import create_tensor_from_numpy
from nndeploy.preprocess import ConvertTo
from nndeploy.infer import Infer
from nndeploy.codec.codec import OpenCvImageDecode, OpenCvImageEncode


class PluginBoundaries(unittest.TestCase):
    def test_cast_and_edge_lifetime(self):
        values = np.array([0, 7, 255], dtype=np.uint8)
        source, result = Edge("input"), Edge("output")
        inputs, outputs = [source], [result]
        node = ConvertTo("cast", inputs, outputs)
        self.assertTrue(source.set(create_tensor_from_numpy(values)))
        ref = weakref.ref(source)
        inputs.clear()
        outputs.clear()
        del source
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertTrue(node.run())
        array = np.asarray(result.get(node))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_array_equal(array, values)
        del node
        gc.collect()
        self.assertIsNone(ref())

    def test_unconfigured_infer(self):
        node = Infer("unconfigured")
        self.assertFalse(node.init())
        self.assertFalse(node.run())
        self.assertTrue(node.deinit())

    def test_codec_failures_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"invalid")
            edge = Edge("image")
            decode = OpenCvImageDecode("read", [], [edge])
            self.assertTrue(decode.set_path(str(path)))
            self.assertFalse(decode.run())
            encode = OpenCvImageEncode("write", [edge], [])
            values = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            self.assertTrue(edge.set(values))
            self.assertTrue(encode.set_path(str(path / "missing.png")))
            self.assertFalse(encode.run())
            self.assertTrue(encode.set_path(str(path)))
            self.assertTrue(encode.run())
            self.assertTrue(decode.run())
            np.testing.assert_array_equal(edge.get(decode), values)
