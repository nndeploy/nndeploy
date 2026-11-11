import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from nndeploy import _nndeploy_internal as native


class PluginSafety(unittest.TestCase):
    @unittest.skipUnless(hasattr(native, "segment"), "segment disabled")
    def test_segment_owns_external_pixels(self):
        for field in ("mask_", "score_"):
            result = native.segment.SegmentResult()
            pixels = np.array([[4, 8]], dtype=np.float32)
            source = native.device.Tensor.from_numpy(pixels, native.base.DeviceType(native.base.DeviceTypeCode.cpu, 0))
            setattr(result, field, source)
            pixels.fill(-1)
            retained = getattr(result, field)
            del source, pixels, result
            gc.collect()
            np.testing.assert_array_equal(np.asarray(retained), [[4, 8]])

    @unittest.skipUnless(hasattr(native, "segment"), "segment disabled")
    def test_segment_tensor_lifetime(self):
        from nndeploy.segment.result import SegmentResult
        result = SegmentResult()
        for field in ("mask_", "score_"):
            source = native.device.Tensor("first")
            setattr(result, field, source)
            old = getattr(result, field)
            setattr(result, field, native.device.Tensor("replacement"))
            del source
            gc.collect()
            self.assertEqual(old.get_name(), "first")
            setattr(result, field, None)
            self.assertIsNone(getattr(result, field))
        result.mask_ = native.device.Tensor("last")
        last = result.mask_
        del result
        gc.collect()
        self.assertEqual(last.get_name(), "last")

    @unittest.skipUnless(hasattr(native, "matting"), "matting disabled")
    def test_matting_exports_and_grayscale(self):
        import nndeploy.matting as m
        for name in ("MattingResult", "PPMattingPostParam", "PPMattingPostProcess", "PPMattingGraph", "VisMatting"):
            self.assertIs(getattr(m, name), getattr(native.matting, name))
        image, alpha, out = [native.dag.Edge(name) for name in ("image", "alpha", "out")]
        node = m.VisMatting("vis", [image, alpha], [out])
        values = np.array([[20, 40]], dtype=np.uint8)
        result = m.MattingResult()
        result.shape, result.alpha = [1, 2], [1.0, 0.0]
        self.assertTrue(image.set(values))
        self.assertTrue(alpha.set(result))
        self.assertTrue(node.run())
        np.testing.assert_array_equal(out.get(node), [[[20, 20, 20], [153, 255, 120]]])
        result.alpha = []
        self.assertFalse(node.run())

    @unittest.skipUnless(hasattr(native, "track"), "track disabled")
    def test_fairmot_channel_arrays(self):
        param = native.track.FairMotPreParam()
        for field in ("scale_", "mean_", "std_"):
            self.assertTrue(param.deserialize(json.dumps({field: [1, 2, 3, 9]})))
            self.assertTrue(param.deserialize(json.dumps({field: [4, 5, 6]})))
            self.assertEqual(json.loads(param.serialize())[field], [4, 5, 6, 9])
            for bad in ([1], [1, 2], [1, "wrong", 3], [1, 2, 3, 4, 5]):
                self.assertFalse(param.deserialize(json.dumps({field: bad})))

    @unittest.skipUnless(hasattr(native, "tokenizer_cpp"), "tokenizers-cpp disabled")
    def test_hf_tokenizer_errors_and_retry(self):
        from nndeploy.tokenizer.tokenizer import TokenizerEncodeCpp, TokenizerDecodeCpp
        model = json.dumps({"version":"1.0", "model":{"type":"WordLevel",
                           "vocab":{"[UNK]":0, "hello":1}, "unk_token":"[UNK]"}})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tokenizer.json"
            for cls in (TokenizerEncodeCpp, TokenizerDecodeCpp):
                node = cls("tokenizer")
                param = node.get_param()
                param.tokenizer_type_ = native.tokenizer.TokenizerType.kTokenizerTypeHF
                for is_path, blob in [(True, str(path) + ".missing"), (False, "{"),
                                      (False, '{"model":{"type":"invalid"}}')]:
                    param.is_path_, param.json_blob_ = is_path, blob
                    self.assertFalse(node.init())
                path.write_text(model)
                for is_path, blob in [(True, str(path)), (False, model)]:
                    param.is_path_, param.json_blob_ = is_path, blob
                    self.assertTrue(node.init())
                    if cls is TokenizerEncodeCpp:
                        self.assertEqual(node.encode("hello"), [1])
                    else:
                        self.assertEqual(node.decode([1]), "hello")
