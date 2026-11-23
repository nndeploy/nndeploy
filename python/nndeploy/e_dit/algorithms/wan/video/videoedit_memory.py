# SPDX-License-Identifier: Apache-2.0
"""Bounded staging and lifetime helpers for serial VideoEdit inference."""

import logging
import os
from collections import OrderedDict, deque

import torch

logger = logging.getLogger(__name__)

# The automatic residency profile is measured on one 1920x1088 aligned frame with
# a 49-frame window, which is 13 latent frames at a 8x VAE and 2x patch stride.
# Every reserve below is derived by scaling that measurement with the window's
# latent volume, so the window length itself is bounded only by measured headroom.
PROFILE_LATENT_FRAMES = 13
PROFILE_LATENT_HEIGHT = 136  # 1088 // 8
PROFILE_LATENT_WIDTH = 240  # 1920 // 8
PROFILE_FRAME_CELLS = PROFILE_LATENT_HEIGHT * PROFILE_LATENT_WIDTH
PROFILE_CELLS = PROFILE_LATENT_FRAMES * PROFILE_FRAME_CELLS
PROFILE_MAX_ALIGNED_HEIGHT = 1088
PROFILE_MAX_ALIGNED_WIDTH = 1920


def env_nonnegative_int(name: str, default: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def common_weight_budget(local_bytes, cpu_group=None):
    """All SP ranks use the least available headroom; fail together if too small."""
    import torch.distributed as dist

    value = torch.tensor([int(local_bytes)], dtype=torch.int64)
    if dist.is_initialized():
        if cpu_group is None:
            raise ValueError("Automatic residency requires the SP CPU process group")
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=cpu_group)
    return int(value.item())


def window_frames(latent_frames: int) -> int:
    """Pixel frames in a Wan window: four VAE frames per latent frame, plus one."""
    return 4 * (max(1, int(latent_frames)) - 1) + 1


def feasible_window_frames(latent_height, latent_width, available_bytes, *, sp_degree=2,
                           minimum_reserve_bytes=4 * 1024**3,
                           full_frame_reserve_bytes=16 * 1024**3,
                           margin_bytes=2 * 1024**3, minimum_weight_bytes=0) -> int:
    """Largest window this headroom supports, or 0 when not even one frame fits.

    Mirrors ``VideoEditResidencyPlanner.plan`` scaling so a rejected request can
    name the frame count that would fit the same crop on the same headroom. It is
    a profile estimate: measured high-water marks can raise the reserve further.
    """
    if available_bytes <= 0:
        return 0
    headroom = available_bytes - margin_bytes - minimum_weight_bytes
    if headroom < minimum_reserve_bytes:
        return 0
    cells = (headroom * PROFILE_CELLS) // max(1, full_frame_reserve_bytes * (2 // sp_degree))
    latent_frames = cells // max(1, int(latent_height) * int(latent_width))
    return window_frames(latent_frames) if latent_frames >= 1 else 0


def _headroom_shortfall(shape, available_bytes, reserve_bytes, margin_bytes,
                        minimum_weight_bytes, budget_bytes, weight_limit_bytes,
                        frames_hint) -> str:
    def gib(value):
        return value / 1024**3

    message = (
        f"VideoEdit adaptive residency cannot serve a {window_frames(shape[2])}-frame "
        f"window with latent shape {list(shape)}: {gib(reserve_bytes):.1f} GiB activation "
        f"reserve plus {gib(margin_bytes):.1f} GiB margin out of {gib(available_bytes):.1f} GiB "
        f"headroom leaves {gib(budget_bytes):.1f} GiB for DiT weights, below the "
        f"{gib(minimum_weight_bytes):.1f} GiB one layer needs"
    )
    if weight_limit_bytes:
        message += (
            f" (NNDEPLOY_DIT_WAN_VIDEOEDIT_WEIGHT_BUDGET_MB={weight_limit_bytes // 1024**2} "
            f"caps the weight budget)"
        )
    if frames_hint:
        message += (
            f"; the same crop supports about {frames_hint} frames at this headroom. "
            f"Lower infer_len, crop tighter, or raise the reserve only with measurements"
        )
    return message


class VideoEditResidencyPlanner:
    """Conservative Wan 14B/FA/BF16 profile, with bounded per-shape high-water data.

    The 16 GiB full-frame estimate is above the historical two-rank ~11 GiB
    non-weight footprint; a separate 2 GiB margin protects allocator/workspace
    variation. It is a measured workload profile, not an OOM-proof allocator.
    Observations may increase the reserve but never reduce this shape floor.

    The estimate scales with the window's latent volume, so ``infer_len`` may be
    any 4n+1 frame count the measured headroom can hold: longer windows keep fewer
    DiT layers resident, and windows that no longer leave room for one layer are
    rejected with the frame count the same crop would support.
    """

    def __init__(self):
        self.history = OrderedDict()

    def plan(self, latent_shape, available_bytes, *, sp_degree=2, cfg=True,
             teacache=True, minimum_reserve_bytes=4 * 1024**3,
             full_frame_reserve_bytes=16 * 1024**3, margin_bytes=2 * 1024**3,
             weight_limit_bytes=0, minimum_weight_bytes=0):
        shape = tuple(int(x) for x in latent_shape)
        if (len(shape) != 5 or any(x <= 0 for x in shape)
                or shape[0] != 1 or shape[1] != 16):
            raise ValueError("Automatic residency profile supports batch 1 with 16 Wan "
                             f"latent channels, got {list(shape)}")
        # A 16-pixel aligned crop keeps both latent spatial dimensions even, and
        # the measured profile only covers crops of one 1920x1088 aligned frame.
        if shape[3] % 2 or shape[4] % 2:
            raise ValueError("Automatic residency profile requires a 16-pixel aligned "
                             f"crop, got latent {shape[3]}x{shape[4]}")
        if (max(shape[3:]) > PROFILE_LATENT_WIDTH
                or shape[3] * shape[4] > PROFILE_FRAME_CELLS):
            raise ValueError("Automatic residency profile supports aligned crops of at most "
                             f"{PROFILE_MAX_ALIGNED_WIDTH}x{PROFILE_MAX_ALIGNED_HEIGHT}, got "
                             f"latent {shape[3]}x{shape[4]}")
        if sp_degree not in (1, 2):
            raise ValueError("Automatic residency profile supports SP=1 or SP=2")
        if min(available_bytes, minimum_reserve_bytes, full_frame_reserve_bytes,
               margin_bytes, weight_limit_bytes, minimum_weight_bytes) < 0:
            raise ValueError("Residency memory limits must be nonnegative")
        key = (shape, sp_degree, bool(cfg), bool(teacache))
        cells = shape[2] * shape[3] * shape[4]
        scaled = (full_frame_reserve_bytes * cells * (2 // sp_degree)
                  + PROFILE_CELLS - 1) // PROFILE_CELLS
        history = self.history.get(key, 0)
        reserve = max(minimum_reserve_bytes, scaled, history)
        budget = max(0, available_bytes - reserve - margin_bytes)
        if weight_limit_bytes:
            budget = min(budget, weight_limit_bytes)
        plan = dict(latent_shape=list(shape), available_bytes=available_bytes,
                    estimated_activation_bytes=reserve,
                    historical_activation_bytes=history, margin_bytes=margin_bytes,
                    local_weight_budget_bytes=budget)
        # ``available_bytes=0`` means "geometry only": the pre-flight check runs
        # before any headroom measurement and validates the shape alone.
        if available_bytes and available_bytes < reserve + margin_bytes + minimum_weight_bytes:
            hint = 0 if weight_limit_bytes and weight_limit_bytes < minimum_weight_bytes else (
                feasible_window_frames(
                    shape[3], shape[4], available_bytes, sp_degree=sp_degree,
                    minimum_reserve_bytes=minimum_reserve_bytes,
                    full_frame_reserve_bytes=full_frame_reserve_bytes,
                    margin_bytes=margin_bytes, minimum_weight_bytes=minimum_weight_bytes))
            raise ValueError(_headroom_shortfall(
                shape, available_bytes, reserve, margin_bytes, minimum_weight_bytes,
                budget, weight_limit_bytes, hint))
        return key, plan

    def observe(self, key, *, peak_allocated_bytes, initial_allocated_bytes,
                weight_capacity_bytes):
        # Capacity is an upper bound on simultaneous managed weights. This
        # residual is a lower bound on non-weight peak, not an exact activation
        # measurement; keep the conservative shape estimate and add 25% here.
        residual = max(0, peak_allocated_bytes - initial_allocated_bytes - weight_capacity_bytes)
        previous = self.history.pop(key, 0)
        self.history[key] = max(previous, (residual * 5 + 3) // 4)
        while len(self.history) > 128:
            self.history.popitem(last=False)


class PinnedStagingPool:
    """Two reusable CPU buffers; wait for DMA before overwriting either one.

    Used on one producer thread/copy stream. Pageable masters remain unchanged.
    Allocation failure is observable and falls back to blocking pageable copies.
    """

    def __init__(self, max_bytes: int):
        if max_bytes < 32:
            raise ValueError("staging budget must be at least 32 bytes")
        self.slot_bytes = (max_bytes // 2 // 16) * 16
        self.slots = []
        self.events = [None, None]
        self.cursor = 0
        self.fallback_copies = 0
        self.disabled = False

    @property
    def pinned_bytes(self):
        return sum(t.numel() for t in self.slots if t.is_pinned())

    def copy_to(self, target, source, stream):
        if not target.is_contiguous() or not source.is_contiguous() or self.disabled:
            self.fallback_copies += 1
            target.copy_(source, non_blocking=False)
            return
        self.copy_many(target, [source], stream)

    def copy_many(self, target, sources, stream):
        """Coalesce small, equal-dtype CPU weights into bounded DMA batches."""
        sources = [t.view(-1) for t in sources if t.numel()]
        if not sources:
            return
        dtype = sources[0].dtype
        assert all(t.dtype == dtype and t.device.type == "cpu" for t in sources)
        assert target.is_contiguous() and target.numel() == sum(t.numel() for t in sources)

        def blocking_copy():
            offset = 0
            for source in sources:
                self.fallback_copies += 1
                target.view(-1)[offset : offset + source.numel()].copy_(source, non_blocking=False)
                offset += source.numel()

        if self.disabled:
            blocking_copy()
            return
        if not self.slots:
            try:
                self.slots = [torch.empty(self.slot_bytes, dtype=torch.uint8,
                                          device="cpu", pin_memory=True) for _ in range(2)]
            except (RuntimeError, TypeError):
                self.slots.clear()
                self.disabled = True
                logger.warning("Pinned staging unavailable; using blocking pageable H2D")
                blocking_copy()
                return
        size = self.slot_bytes // sources[0].element_size()
        target = target.view(-1)
        source_index, source_offset = 0, 0
        for offset in range(0, target.numel(), size):
            slot = self.cursor % 2
            self.cursor += 1
            if self.events[slot] is not None:
                self.events[slot].synchronize()
            count = min(size, target.numel() - offset)
            staging = self.slots[slot].view(dtype)[:count]
            filled = 0
            while filled < count:
                source = sources[source_index]
                length = min(count - filled, source.numel() - source_offset)
                staging[filled : filled + length].copy_(source[source_offset : source_offset + length])
                filled += length
                source_offset += length
                if source_offset == source.numel():
                    source_index += 1
                    source_offset = 0
            target[offset : offset + count].copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
            self.events[slot] = event

    def close(self):
        for event in self.events:
            if event is not None:
                event.synchronize()
        self.events = [None, None]
        self.slots.clear()


def select_weight_budget(layer_bytes, budget_bytes, prefetch_cap):
    """Choose a resident prefix and a rolling window within a byte budget.

    Reserve using the largest layers rather than assuming uniform sizes.
    Return (resident_layers, prefetch_size); raise if even one layer cannot fit.
    """
    if not layer_bytes:
        return 0, 1
    if budget_bytes < max(layer_bytes):
        raise ValueError("weight budget cannot hold the largest layer")
    if sum(layer_bytes) <= budget_bytes:
        return len(layer_bytes), 1
    # A zero-lookahead schedule still loads the current layer on demand.
    prefetch = min(max(0, prefetch_cap), len(layer_bytes) - 1)
    while sum(sorted(layer_bytes, reverse=True)[:prefetch + 1]) > budget_bytes:
        prefetch -= 1
    resident, used = 0, 0
    for index, size in enumerate(layer_bytes):
        remaining = sorted(layer_bytes[index + 1:], reverse=True)[:prefetch + 1]
        if used + size + sum(remaining) > budget_bytes:
            break
        used += size
        resident += 1
    return resident, prefetch


def decoded_frames_to_images(decoded_frames, batch_size=4):
    """Bounded D2H pipeline preserving the original FP32 CPU pixel conversion.

    Transfer in the decoded dtype, then convert on CPU. Two frame batches can
    be in flight; the producer event protects CUDA inputs and each CPU batch
    is read only after its copy completes. No full-video FP32 GPU allocation.
    """
    from PIL import Image

    if batch_size < 1:
        raise ValueError("decode transfer batch size must be positive")
    frames = []

    def consume(host):
        for frame in host.float().permute(0, 2, 3, 1):
            arr = (frame.clamp(0, 1).numpy() * 255.0).astype("uint8")
            frames.append(Image.fromarray(arr))

    if decoded_frames.device.type != "cuda":
        for batch in decoded_frames.split(batch_size):
            consume(batch.detach().cpu())
        return frames

    stream = torch.cuda.Stream(device=decoded_frames.device)
    stream.wait_stream(torch.cuda.current_stream(decoded_frames.device))
    decoded_frames.record_stream(stream)
    pending = deque()
    slots = []
    try:
        # Allocate before submitting any copy, allowing a clean pageable fallback.
        shape = (min(batch_size, len(decoded_frames)), *decoded_frames.shape[1:])
        try:
            slots = [torch.empty(shape, dtype=decoded_frames.dtype, pin_memory=True)
                     for _ in range(2)]
        except (RuntimeError, TypeError):
            logger.warning("Pinned decode buffers unavailable; using batchwise blocking D2H")
            for batch in decoded_frames.split(batch_size):
                consume(batch.detach().cpu())
            return frames
        for index, batch in enumerate(decoded_frames.split(batch_size)):
            if len(pending) == 2:
                event, host = pending.popleft()
                event.synchronize()
                consume(host)
            host = slots[index % 2][:len(batch)]
            with torch.cuda.stream(stream):
                host.copy_(batch.detach(), non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
            pending.append((event, host))
        while pending:
            event, host = pending.popleft()
            event.synchronize()
            consume(host)
        return frames
    finally:
        # Also protect pinned destinations when conversion raises mid-pipeline.
        stream.synchronize()
