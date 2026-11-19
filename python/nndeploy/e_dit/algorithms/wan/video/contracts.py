# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEditWindowSpec:
    window_index: int
    start_index: int
    end_index: int
    input_indices: list[int]
    commit_local_to_global: dict[int, int]
    valid_len: int | None = None
    reflected_count: int = 0
    stride: int | None = None
    reference_prev_local_idx: int | None = None
    reference_global_index: int | None = None
    overlap_mask_zero_count: int = 0
    commit_start_local_idx: int = 0
