"""CPU checks for result ownership, coordinates and operation-local replacement."""

import tempfile
import unittest
import numpy as np

from nndeploy.segment.sam3.modules import Modules, boundary
from nndeploy.segment.sam3.results import ResultStore, decode_mask, encode_mask, mask_box
from nndeploy.segment.sam3.types import SAM3Error, box_xyxy, concept_boxes, fields, points_xy


class Contracts(unittest.TestCase):
    def test_coordinate_semantics(self):
        values = [{"box": [10, 20, 30, 50], "positive": True}]
        self.assertEqual(concept_boxes(values, 100, 200), ([[.2, .175, .2, .15]], [1]))
        self.assertEqual(concept_boxes(values, 100, 200, video=True), ([[.1, .1, .2, .15]], [1]))
        with self.assertRaises(SAM3Error):
            box_xyxy([20, 0, 10, 20], 100, 200)
        with self.assertRaises(SAM3Error):
            points_xy([[100, 20]], [1], 100, 200)

    def test_mask_wire_and_exclusive_bbox(self):
        mask = np.zeros((3, 5), bool); mask[1, 4] = True
        np.testing.assert_array_equal(decode_mask(encode_mask(mask), 3, 5), mask)
        self.assertEqual(mask_box(mask), [4, 1, 5, 2])
        self.assertEqual(mask_box(np.zeros_like(mask)), [0, 0, 0, 0])
        with self.assertRaises(SAM3Error):
            decode_mask(np.ones((3, 5)) * .5, 3, 5)

    def test_result_versions_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory)
            key = store.begin("session", 3)
            store.write(key, {"frame_index": 1, "objects": []}, "forward")
            store.finish(key, "succeeded")
            frames = store.read(key, "session", 3, [0, 1])
            self.assertEqual([f["availability"] for f in frames], ["not_computed", "complete"])
            with self.assertRaises(SAM3Error):
                store.read(key, "session", 4, [1])
            with self.assertRaises(SAM3Error):
                store.read("../other", "session", 3, [1])

    def test_local_replacement_does_not_escape_scope(self):
        @boundary("unit")
        def function(value):
            return value + 1
        modules = Modules({"unit": lambda native, value: native(value) * 2})
        self.assertEqual(function(3), 4)
        with modules.activate():
            self.assertEqual(function(3), 8)
        self.assertEqual(function(3), 4)
        self.assertEqual(modules.calls["unit"], 1)

    def test_no_silent_parameters(self):
        with self.assertRaises(SAM3Error):
            fields({"ignored_parameter": 2}, {"supported_parameter"})


if __name__ == "__main__":
    unittest.main()
