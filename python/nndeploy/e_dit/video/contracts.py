"""Origin-observable contracts shared by the windowed erase runtime."""

from __future__ import annotations

import math
from typing import Any

from nndeploy.e_dit.video.encoding import VideoEncodingProfile, _optional_string


SUPPORTED_WINDOW_SP_DEGREES = frozenset({1, 2, 4})


def resolve_spatial_alignment(sp_degree: int) -> tuple[int, int]:
    """Return origin-compatible ``(align_w, align_h)`` for an SP topology."""
    if isinstance(sp_degree, bool) or not isinstance(sp_degree, int):
        raise TypeError("sp_degree must be an integer")
    if sp_degree not in SUPPORTED_WINDOW_SP_DEGREES:
        raise ValueError(f"unsupported WINDOW sp_degree: {sp_degree}")
    exponent = int(math.log2(sp_degree))
    align_w = 32 * (2 ** math.ceil(exponent / 2))
    align_h = 32 * (2 ** math.floor(exponent / 2))
    return int(align_w), int(align_h)


def resolve_runtime_sp_degree(server_args: Any) -> int:
    """Read the effective SP degree without introducing a second config source."""
    context = getattr(server_args, "parallel_context", None)
    if context is None or not bool(getattr(context, "enabled", False)):
        return 1
    plan = getattr(context, "plan", None)
    if plan is None:
        raise ValueError("enabled parallel context must provide an SP plan")
    return int(getattr(plan, "sp_degree"))
