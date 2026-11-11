import gc
import subprocess
import sys
import threading
import time
import unittest
import weakref

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.device import Buffer, Tensor


def run_bounded_pipeline():
    edge = native.dag.Edge("frames")
    edge.set_parallel_type(native.base.ParallelType.Pipeline)
    edge.set_queue_max_size(1)
    edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.AllBackpressure)
    edge.mark_graph_output()
    edge.construct()
    node = native.dag.Node("producer", [], [edge])
    frames = [np.array([[i]], dtype=np.uint8) for i in range(4)]
    node.set_output_data(frames[0])
    received = []

    def consume():
        # Let the producer encounter backpressure before reading anything.
        time.sleep(0.1)
        for _ in frames:
            received.append(edge.get_graph_output().item())

    reader = threading.Thread(target=consume)
    reader.start()
    for frame in frames[1:]:
        node.set_output_data(frame)
    reader.join()
    if received != list(range(len(frames))):
        raise AssertionError(received)


class NumpyEdge(unittest.TestCase):
    def setUp(self):
        self.edge = native.dag.Edge("image")
        self.node = native.dag.Node("sender", [], [self.edge])

    def require_opencv(self):
        status = self.edge.set(np.zeros((1, 1), dtype=np.uint8))
        if status == native.base.StatusCode.ErrorNotSupport:
            self.skipTest("OpenCV disabled")
        self.assertTrue(status)

    def test_image_entry_points_have_identical_layout_and_validation(self):
        self.require_opencv()
        image = np.arange(24, dtype=np.int16).reshape(2, 4, 3)
        for setter in (self.edge.set, self.node.set_output_data):
            for view in (image[:, ::2], image[..., ::-1], image.transpose(1, 0, 2)):
                expected = view.copy()
                self.assertTrue(setter(view))
                np.testing.assert_array_equal(self.edge.get(None), expected)
            for invalid in (np.array(1), np.arange(3), image[None],
                            np.empty((1, 1, 0)), np.empty((1, 1, 513))):
                with self.assertRaises(ValueError):
                    setter(invalid)
            invalid = np.ones((2, 2), dtype=object)
            owner = weakref.ref(invalid)
            with self.assertRaises(TypeError):
                setter(invalid)
            del invalid
            gc.collect()
            self.assertIsNone(owner())
            np.testing.assert_array_equal(self.edge.get(None), expected)

    def test_publishing_same_object_releases_old_wrapper(self):
        for factory in (Tensor.from_numpy, Buffer.from_numpy):
            for setter in (self.edge.set, self.node.set_output_data):
                source = np.arange(4, dtype=np.uint8)
                payload = factory(source)
                owner = weakref.ref(payload)
                for _ in range(4):
                    self.assertTrue(setter(payload))
                del payload
                self.assertIsNotNone(owner())
                self.assertTrue(setter("replace"))
                gc.collect()
                self.assertIsNone(owner())

    def test_pipeline_backpressure_with_python_consumer(self):
        self.require_opencv()
        subprocess.run([sys.executable, __file__, "--pipeline"], timeout=15, check=True)


if __name__ == "__main__" and sys.argv[1:] == ["--pipeline"]:
    run_bounded_pipeline()
