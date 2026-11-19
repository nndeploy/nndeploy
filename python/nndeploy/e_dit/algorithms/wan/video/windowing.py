# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from nndeploy.e_dit.algorithms.wan.video.contracts import VideoEditWindowSpec


@dataclass(frozen=True)
class VideoEditPassPlan:
    """One VideoEdit inference pass in model-time order."""

    name: Literal["long", "short"]
    direction: Literal["forward", "backward"]
    source_indices: tuple[int, ...]
    prefix_kind: Literal["reference", "bridge"]
    prefix_length: int

    @property
    def sequence_indices(self) -> tuple[int | None, ...]:
        """Global source indices, with conditioning-only frames marked ``None``."""

        return (None,) * self.prefix_length + self.source_indices


@dataclass(frozen=True)
class VideoEditSequencePlan:
    """The long pass and optional bridge-seeded short pass for one source video."""

    long: VideoEditPassPlan
    short: VideoEditPassPlan | None
    bridge_length: int


def shrink_videoedit_bridge(requested: int, available: int) -> int:
    """Return the largest legal bridge length within the available long output."""

    if requested < 1:
        raise ValueError(f"bridge_overlap must be >= 1, got {requested}")
    if (requested - 1) % 4 != 0:
        raise ValueError(
            "bridge_overlap must satisfy (b-1)%4==0 (1, 5, 9, ...), "
            f"got {requested}"
        )
    if available < 0:
        raise ValueError(f"available bridge frames must be non-negative, got {available}")
    cap = min(requested, max(available, 1))
    return max(cap - ((cap - 1) % 4), 1)


def plan_videoedit_passes(
    num_frames: int,
    ref_frame_idx: int,
    bridge_overlap: int = 5,
) -> VideoEditSequencePlan:
    """Plan arbitrary-reference VideoEdit passes without changing global indices."""

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not 0 <= ref_frame_idx < num_frames:
        raise ValueError(
            f"ref_frame_idx={ref_frame_idx} out of range [0,{num_frames})"
        )
    # Validate even when the short side is empty: an invalid request must fail rather
    # than become conditionally valid based on the selected reference frame.
    shrink_videoedit_bridge(bridge_overlap, num_frames)

    left = tuple(range(ref_frame_idx))
    right = tuple(range(ref_frame_idx + 1, num_frames))
    if len(right) >= len(left):
        long_direction: Literal["forward", "backward"] = "forward"
        long_indices = (ref_frame_idx,) + right
        short_indices = tuple(reversed(left))
        short_direction: Literal["forward", "backward"] = "backward"
    else:
        long_direction = "backward"
        long_indices = (ref_frame_idx,) + tuple(reversed(left))
        short_indices = right
        short_direction = "forward"

    long_plan = VideoEditPassPlan(
        name="long",
        direction=long_direction,
        source_indices=long_indices,
        prefix_kind="reference",
        prefix_length=1,
    )
    if not short_indices:
        return VideoEditSequencePlan(long=long_plan, short=None, bridge_length=0)

    bridge_length = shrink_videoedit_bridge(bridge_overlap, len(long_indices))
    short_plan = VideoEditPassPlan(
        name="short",
        direction=short_direction,
        source_indices=short_indices,
        prefix_kind="bridge",
        prefix_length=bridge_length,
    )
    return VideoEditSequencePlan(
        long=long_plan,
        short=short_plan,
        bridge_length=bridge_length,
    )

def _native_reverse_mirror_index(index: int, num_frames: int) -> int:
    if num_frames <= 1:
        return 0
    if index < num_frames:
        return index
    return max(num_frames - 1 - (index - num_frames), 0)


def build_videoedit_pass_window_specs(
    sequence_indices: Sequence[int | None],
    infer_len: int = 49,
    overlap: int = 5,
) -> list[VideoEditWindowSpec]:
    """Plan strict native windows over one explicit long/short pass sequence.

    ``input_indices`` are pass-local materialization indices. Commit mappings point
    directly to native source-video indices and omit conditioning frames, propagated
    overlap, and reverse-mirror padding.
    """

    total = len(sequence_indices)
    if total <= 0:
        raise ValueError("sequence_indices must not be empty")
    if infer_len < 1 or (infer_len - 1) % 4 != 0:
        raise ValueError(
            "infer_len must be >= 1 and satisfy (infer_len-1)%4==0, "
            f"got {infer_len}"
        )
    if not 0 <= overlap < infer_len:
        raise ValueError(f"overlap must be in [0, {infer_len}), got {overlap}")

    stride = infer_len - overlap
    starts = [0]
    if total > infer_len:
        next_start = stride
        while next_start + overlap < total:
            starts.append(next_start)
            next_start += stride

    specs: list[VideoEditWindowSpec] = []
    for window_index, start_index in enumerate(starts):
        raw_indices = list(range(start_index, start_index + infer_len))
        input_indices = [
            _native_reverse_mirror_index(index, total) for index in raw_indices
        ]
        valid_len = min(infer_len, total - start_index)
        commit_start = 0 if window_index == 0 else overlap
        commit: dict[int, int] = {}
        for local_idx in range(commit_start, valid_len):
            global_idx = sequence_indices[start_index + local_idx]
            if global_idx is not None:
                commit[local_idx] = global_idx

        uses_previous_output = window_index > 0 and overlap > 0
        reference_global_index = (
            sequence_indices[start_index] if uses_previous_output else None
        )
        specs.append(
            VideoEditWindowSpec(
                window_index=window_index,
                start_index=start_index,
                end_index=min(start_index + infer_len, total),
                input_indices=input_indices,
                commit_local_to_global=commit,
                valid_len=valid_len,
                reflected_count=infer_len - valid_len,
                stride=stride,
                reference_prev_local_idx=stride if uses_previous_output else None,
                reference_global_index=reference_global_index,
                overlap_mask_zero_count=overlap if uses_previous_output else 0,
                commit_start_local_idx=commit_start,
            )
        )
    return specs
