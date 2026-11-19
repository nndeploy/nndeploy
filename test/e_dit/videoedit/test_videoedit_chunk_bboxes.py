# Adapted from SGLang multimodal_gen tests (Apache-2.0).
"""Regression coverage for masks at non-aligned source-frame edges."""

import unittest

from nndeploy.e_dit.algorithms.wan.video.chunk_plan import (
    ChunkPlan,
    ClipPlan,
    assign_chunk_bboxes,
)


def _plan_bbox(mask_bbox, height, width, mode):
    chunk = ChunkPlan(
        w_idx=0, seq_start=0, context=[0], owned=(0, 1), read_range=(0, 1)
    )
    clip = ClipPlan(
        label="long", direction="forward", seq_idx=[0], lead=0, chunks=[chunk]
    )
    assign_chunk_bboxes(
        [clip], [mask_bbox], height, width, mode=mode, expand_scale=1.6
    )
    return chunk


class TestVideoEditChunkBBoxes(unittest.TestCase):
    def test_slides_aligned_crop_to_preserve_each_frame_edge(self):
        cases = [
            ((100, 0, 500, 800), 1080, 1920, 1, 0),
            ((100, 279, 500, 1079), 1080, 1920, 3, 1080),
            ((0, 100, 800, 500), 1920, 1080, 0, 0),
            ((279, 100, 1079, 500), 1920, 1080, 2, 1080),
        ]
        for mode in ("tight", "fixed_size"):
            for mask, height, width, edge, coordinate in cases:
                with self.subTest(mode=mode, mask=mask):
                    chunk = _plan_bbox(mask, height, width, mode)
                    x0, y0, x1, y1 = chunk.bbox
                    self.assertLessEqual(x0, mask[0])
                    self.assertLessEqual(y0, mask[1])
                    self.assertGreater(x1, mask[2])
                    self.assertGreater(y1, mask[3])
                    self.assertEqual(chunk.bbox[edge], coordinate)
                    self.assertEqual(chunk.crop_h, chunk.aligned_h)
                    self.assertEqual(chunk.crop_w, chunk.aligned_w)
                    self.assertEqual(
                        chunk.crop_h if height == 1080 else chunk.crop_w, 1072
                    )

    def test_resamples_full_extent_when_mask_cannot_fit_aligned_crop(self):
        cases = [
            ((100, 0, 500, 1079), 1080, 1920, (0, 0, 1024, 1080), (1088, 1024)),
            ((0, 100, 1079, 500), 1920, 1080, (0, 0, 1080, 1024), (1024, 1088)),
            ((0, 0, 1919, 1079), 1080, 1920, (0, 0, 1920, 1080), (1088, 1920)),
        ]
        for mode in ("tight", "fixed_size"):
            for mask, height, width, bbox, aligned in cases:
                with self.subTest(mode=mode, mask=mask):
                    chunk = _plan_bbox(mask, height, width, mode)
                    self.assertEqual(chunk.bbox, bbox)
                    self.assertEqual((chunk.aligned_h, chunk.aligned_w), aligned)
                    self.assertEqual(chunk.crop_h, bbox[3] - bbox[1])
                    self.assertEqual(chunk.crop_w, bbox[2] - bbox[0])

    def test_interior_crop_keeps_existing_geometry(self):
        for mode in ("tight", "fixed_size", "global"):
            with self.subTest(mode=mode):
                chunk = _plan_bbox((300, 300, 500, 500), 1080, 1920, mode)
                self.assertEqual(chunk.bbox, (160, 160, 640, 640))
                self.assertEqual((chunk.crop_h, chunk.crop_w), (480, 480))
                self.assertEqual((chunk.aligned_h, chunk.aligned_w), (480, 480))


if __name__ == "__main__":
    unittest.main()
