# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import unittest

import numpy as np
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    build_videoedit_bridge,
    materialize_videoedit_pass,
    materialize_videoedit_window,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import (
    build_videoedit_pass_window_specs,
    plan_videoedit_passes,
    shrink_videoedit_bridge,
)


def _frame(value: int, size: tuple[int, int] = (4, 4)) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


def _mask(value: int, size: tuple[int, int] = (4, 4)) -> Image.Image:
    return Image.new("L", size, value)


def _pixel(image: Image.Image) -> int:
    return int(np.asarray(image)[0, 0, 0])


class TestVideoEditPassPlanning(unittest.TestCase):
    def test_reference_at_first_middle_last_and_tie_break(self):
        first = plan_videoedit_passes(7, 0, bridge_overlap=5)
        self.assertEqual(first.long.direction, "forward")
        self.assertEqual(first.long.source_indices, tuple(range(7)))
        self.assertIsNone(first.short)
        self.assertEqual(first.bridge_length, 0)

        middle = plan_videoedit_passes(7, 3, bridge_overlap=5)
        self.assertEqual(middle.long.direction, "forward")
        self.assertEqual(middle.long.source_indices, (3, 4, 5, 6))
        self.assertEqual(middle.long.sequence_indices, (None, 3, 4, 5, 6))
        self.assertIsNotNone(middle.short)
        assert middle.short is not None
        self.assertEqual(middle.short.direction, "backward")
        self.assertEqual(middle.short.source_indices, (2, 1, 0))
        self.assertEqual(middle.short.sequence_indices, (None, 2, 1, 0))
        self.assertEqual(middle.bridge_length, 1)

        last = plan_videoedit_passes(7, 6, bridge_overlap=5)
        self.assertEqual(last.long.direction, "backward")
        self.assertEqual(last.long.source_indices, (6, 5, 4, 3, 2, 1, 0))
        self.assertIsNone(last.short)
        self.assertEqual(last.bridge_length, 0)

    def test_left_longer_and_right_longer(self):
        right = plan_videoedit_passes(8, 2, bridge_overlap=5)
        self.assertEqual(right.long.source_indices, (2, 3, 4, 5, 6, 7))
        assert right.short is not None
        self.assertEqual(right.short.source_indices, (1, 0))

        left = plan_videoedit_passes(8, 5, bridge_overlap=5)
        self.assertEqual(left.long.direction, "backward")
        self.assertEqual(left.long.source_indices, (5, 4, 3, 2, 1, 0))
        assert left.short is not None
        self.assertEqual(left.short.direction, "forward")
        self.assertEqual(left.short.source_indices, (6, 7))

    def test_reference_range_and_bridge_contract(self):
        for ref_frame_idx in (-1, 7):
            with self.subTest(ref_frame_idx=ref_frame_idx):
                with self.assertRaisesRegex(ValueError, "ref_frame_idx"):
                    plan_videoedit_passes(7, ref_frame_idx, bridge_overlap=5)

        self.assertEqual(shrink_videoedit_bridge(1, 20), 1)
        self.assertEqual(shrink_videoedit_bridge(5, 20), 5)
        self.assertEqual(shrink_videoedit_bridge(9, 20), 9)
        self.assertEqual(shrink_videoedit_bridge(9, 8), 5)
        self.assertEqual(shrink_videoedit_bridge(5, 2), 1)
        for invalid in (0, 2, 3, 4, 6):
            with self.subTest(bridge_overlap=invalid):
                with self.assertRaisesRegex(ValueError, "bridge_overlap"):
                    shrink_videoedit_bridge(invalid, 20)
                with self.assertRaisesRegex(ValueError, "bridge_overlap"):
                    plan_videoedit_passes(1, 0, bridge_overlap=invalid)


class TestVideoEditPassMaterialization(unittest.TestCase):
    def setUp(self):
        self.source_frames = [_frame(10 + i) for i in range(8)]
        self.source_masks = [_mask(100 + i) for i in range(8)]
        self.reference = _frame(240)

    def test_long_pass_keeps_source_k_real_mask_and_global_index(self):
        plan = plan_videoedit_passes(8, 2, bridge_overlap=5).long
        sequence = materialize_videoedit_pass(
            plan,
            source_frames=self.source_frames,
            source_masks=self.source_masks,
            reference_frame=self.reference,
        )

        self.assertEqual(sequence.global_indices, (None, 2, 3, 4, 5, 6, 7))
        self.assertEqual(_pixel(sequence.frames[0]), 240)
        self.assertEqual(_pixel(sequence.frames[1]), 12)
        self.assertEqual(int(np.asarray(sequence.masks[0]).sum()), 0)
        self.assertGreater(int(np.asarray(sequence.masks[1]).sum()), 0)
        self.assertEqual(int(np.asarray(sequence.masks[1])[0, 0]), 102)

    def test_bridge_is_long_output_slice_reversed_and_conditioning_only(self):
        plans = plan_videoedit_passes(10, 4, bridge_overlap=5)
        assert plans.short is not None
        long_output = [_frame(value) for value in range(20, 27)]

        bridge = build_videoedit_bridge(long_output, plans.bridge_length)
        self.assertEqual([_pixel(frame) for frame in bridge], [25, 24, 23, 22, 21])

        sequence = materialize_videoedit_pass(
            plans.short,
            source_frames=self.source_frames + [_frame(18), _frame(19)],
            source_masks=self.source_masks + [_mask(108), _mask(109)],
            bridge_frames=bridge,
        )
        self.assertEqual(sequence.global_indices[:5], (None,) * 5)
        self.assertEqual(sequence.global_indices[5:], (3, 2, 1, 0))
        self.assertTrue(all(int(np.asarray(mask).sum()) == 0 for mask in sequence.masks[:5]))
        self.assertEqual([_pixel(frame) for frame in sequence.frames[:5]], [25, 24, 23, 22, 21])

    def test_bridge_holes_fail_positionally(self):
        with self.assertRaisesRegex(RuntimeError, "holes"):
            build_videoedit_bridge([_frame(1), _frame(2), None, _frame(4)], 3)


class TestVideoEditStrictWindowing(unittest.TestCase):
    def test_native_starts_reverse_mirror_and_padding_not_committed(self):
        sequence_indices = (None,) + tuple(range(9))
        specs = build_videoedit_pass_window_specs(
            sequence_indices,
            infer_len=5,
            overlap=1,
        )

        self.assertEqual([spec.start_index for spec in specs], [0, 4, 8])
        self.assertEqual(specs[-1].input_indices, [8, 9, 9, 8, 7])
        self.assertEqual(specs[-1].valid_len, 2)
        self.assertEqual(specs[-1].reflected_count, 3)
        self.assertEqual(specs[-1].commit_local_to_global, {1: 8})
        self.assertNotIn(0, specs[-1].commit_local_to_global)
        self.assertEqual(specs[1].reference_prev_local_idx, 4)
        self.assertEqual(specs[1].overlap_mask_zero_count, 1)
        self.assertEqual(specs[1].commit_start_local_idx, 1)

    def test_full_overlap_propagation_and_later_window_skip(self):
        sequence_indices = (None,) + tuple(range(12))
        specs = build_videoedit_pass_window_specs(
            sequence_indices,
            infer_len=9,
            overlap=5,
        )
        sequence = materialize_videoedit_pass(
            plan_videoedit_passes(12, 0, bridge_overlap=5).long,
            source_frames=[_frame(i) for i in range(12)],
            source_masks=[_mask(255) for _ in range(12)],
            reference_frame=_frame(200),
        )
        previous_output = [_frame(100 + i) for i in range(9)]

        window = materialize_videoedit_window(
            sequence,
            specs[1],
            previous_output_frames=previous_output,
        )

        self.assertEqual([_pixel(frame) for frame in window.frames[:5]], [104, 105, 106, 107, 108])
        self.assertTrue(all(int(np.asarray(mask).sum()) == 0 for mask in window.masks[:5]))
        self.assertGreater(int(np.asarray(window.masks[5]).sum()), 0)
        self.assertTrue(all(local_idx >= 5 for local_idx in specs[1].commit_local_to_global))

    def test_backward_and_short_commits_recover_native_global_order(self):
        plans = plan_videoedit_passes(8, 5, bridge_overlap=5)
        assert plans.short is not None
        committed = []
        for pass_plan in (plans.long, plans.short):
            for spec in build_videoedit_pass_window_specs(
                pass_plan.sequence_indices,
                infer_len=5,
                overlap=1,
            ):
                for local_idx, global_idx in spec.commit_local_to_global.items():
                    committed.append(
                        (
                            global_idx,
                            pass_plan.name,
                            spec.window_index,
                            local_idx,
                        )
                    )

        self.assertEqual(sorted(item[0] for item in committed), list(range(8)))

    def test_strict_window_shape_and_zero_overlap_multi_window(self):
        for invalid in (0, 2, 4, 6):
            with self.subTest(infer_len=invalid):
                with self.assertRaisesRegex(ValueError, "infer_len"):
                    build_videoedit_pass_window_specs((None, 0), infer_len=invalid, overlap=0)

        specs = build_videoedit_pass_window_specs(
            tuple(range(11)), infer_len=5, overlap=0
        )
        self.assertEqual([spec.start_index for spec in specs], [0, 5, 10])
        self.assertTrue(all(spec.reference_prev_local_idx is None for spec in specs))
        self.assertTrue(all(spec.commit_start_local_idx == 0 for spec in specs))


if __name__ == "__main__":
    unittest.main()
