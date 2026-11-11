import gc
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.dag.graph import Graph


class DagBoundaries(unittest.TestCase):
    def test_uninitialized_graph(self):
        graph = Graph("empty")
        self.assertFalse(graph.run())
        self.assertFalse(graph.synchronize())
        self.assertFalse(graph.interrupt())

    def test_graph_retains_created_node(self):
        graph = Graph("outer")
        node = graph.create_node("nndeploy::dag::Graph", "inner")
        ref = weakref.ref(node)
        del node
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertEqual(graph.get_node("inner").get_name(), "inner")
        del graph
        gc.collect()
        self.assertIsNone(ref())

    def test_load_file_returns_deserialization_status(self):
        graph = Graph("source")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "graph.json"
            path.write_text(graph.serialize())
            self.assertEqual(Graph("target").load_file(path), native.base.StatusCode.Ok)

    def test_numpy_image_layout(self):
        edge = native.dag.Edge("image")
        image = np.arange(120, dtype=np.uint8).reshape(5, 8, 3)
        for view in (image[::-1], image[:, ::2], image.transpose(1, 0, 2)):
            status = edge.set(view)
            if status == native.base.StatusCode.ErrorNotSupport:
                self.skipTest("OpenCV disabled")
            self.assertTrue(status)
            np.testing.assert_array_equal(edge.get(None), view)
        for invalid in (np.array(1), np.arange(3), image[None]):
            with self.assertRaises(ValueError):
                edge.set(invalid)
