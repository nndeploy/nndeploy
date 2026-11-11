import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from nndeploy import _nndeploy_internal as native
from nndeploy.device import create_tensor_from_numpy


class PluginPostprocess(unittest.TestCase):
    @unittest.skipUnless(hasattr(native, "keypoint"), "keypoint disabled")
    def test_pose_uses_each_batch(self):
        from nndeploy.keypoint.keypoint import KeypointPostProcess

        source, boxes, poses = [native.dag.Edge(k) for k in ("in", "boxes", "poses")]
        post = KeypointPostProcess("pose", [source], [boxes, poses])
        post.get_param().num_keypoints_ = 1
        values = np.zeros((2, 8, 2), dtype=np.float32)
        values[0, :, 0] = [10, 10, 4, 4, 0.8, 10, 10, 1]
        values[1, :, 1] = [20, 20, 4, 4, 0.9, 20, 20, 1]
        tensor = create_tensor_from_numpy(values)
        self.assertTrue(source.set(tensor))
        self.assertTrue(post.run())
        skeletons = poses.get(post).skeletons_
        self.assertEqual([s.index_ for s in skeletons], [0, 1])
        np.testing.assert_allclose([s.score_ for s in skeletons], [0.8, 0.9])

    @unittest.skipUnless(hasattr(native, "ocr"), "ocr disabled")
    def test_ocr_dictionary_boundary(self):
        source, output = [native.dag.Edge(k) for k in ("in", "out")]
        post = native.ocr.RecognizerPostProcess("rec", [source], [output])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dict.json"
            path.write_text(json.dumps({"PostProcess": {"character_dict": ["A"]}}))
            post.get_param().character_path_ = str(path)
            values = np.array([[[[0], [1], [0]]]], dtype=np.float32)
            tensor = create_tensor_from_numpy(values)
            self.assertTrue(source.set(tensor))
            self.assertTrue(post.run())
            self.assertEqual(output.get(post).text_, ["A"])
            for classes, valid in [(3, True), (4, False)]:
                values = np.zeros((1, 1, classes), dtype=np.float32)
                values[0, 0, -1] = 1
                tensor = create_tensor_from_numpy(values)
                self.assertTrue(source.set(tensor))
                self.assertEqual(bool(post.run()), valid)

    @unittest.skipUnless(hasattr(native, "ocr"), "ocr disabled")
    def test_ocr_channel_array_bounds(self):
        for cls in (native.ocr.DetectorPreProcessParam, native.ocr.ClassifierPreProcessParam,
                    native.ocr.RecognizerPreProcessParam):
            param = cls()
            for field in ("scale_", "mean_", "std_"):
                values = np.arange(6, dtype=np.float32)[::2]
                setattr(param, field, values)
                before = param.serialize()
                np.testing.assert_array_equal(getattr(param, field), values)
                for length in (2, 4):
                    with self.assertRaises(ValueError):
                        setattr(param, field, np.ones(length, dtype=np.float32))
                    self.assertEqual(param.serialize(), before)

    def test_empty_target_faces_forward_original_frame(self):
        # Exercise run without downloading the optional model or constructing a session.
        with patch.dict(sys.modules, {m: Mock() for m in (
            "insightface", "nndeploy.face.deep_live_cam", "tqdm", "cv2"
        )}):
            from nndeploy.face.insightface import InsightFaceSwapper
        node = Mock()
        frame = np.array([[[12, 34, 56]]], dtype=np.uint8)
        node.get_input.side_effect = [Mock(get=Mock(return_value=value))
                                      for value in ([], [], frame)]
        self.assertTrue(InsightFaceSwapper.run(node))
        self.assertIs(node.get_output.return_value.set.call_args.args[0], frame)
        node.swapper.get.assert_not_called()

    def test_face_map_skips_empty_source_and_preserves_slot_id(self):
        with patch.dict(sys.modules, {m: Mock() for m in (
            "insightface", "nndeploy.face.deep_live_cam", "tqdm", "cv2"
        )}):
            from nndeploy.face.insightface import FaceIdMap
        for mapping, expected_id in (([], 1), ([3, 7], 7)):
            with self.subTest(mapping=mapping):
                node = Mock()
                targets = [{"id": 0}, {"id": expected_id}]
                face = object()
                inputs = [Mock(get=Mock(return_value=v)) for v in (targets, [], [face])]
                node.get_input.side_effect = inputs
                node.get_all_input.return_value = inputs
                node.map_ids_ = mapping
                self.assertTrue(FaceIdMap.run(node))
                mapped = node.get_output.return_value.set.call_args.args[0]
                self.assertNotIn("source", mapped[0])
                self.assertIs(mapped[1]["source"], face)

    @unittest.skipUnless(hasattr(native, "ocr"), "ocr disabled")
    def test_ocr_shape_setters_follow_numpy_strides(self):
        for cls, field in ((native.ocr.ClassifierPreProcessParam, "cls_image_shape_"),
                           (native.ocr.RecognizerPreProcessParam, "rec_image_shape_")):
            param = cls()
            source = np.array([3, 7, 48, 7, 192], dtype=np.int32)
            for view in (source[::2], source[::-2]):
                setattr(param, field, view)
                np.testing.assert_array_equal(getattr(param, field), view)
