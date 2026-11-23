"""EraserDiT erase – timestep preparation stage."""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.utils.latent import build_rope_interpolation_scale


class EraserDiTEraseTimestepPreparationNode(StateNode):
    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        vae = batch.modules["vae"]
        # The baseline hard-codes ``frame_rate=25`` for the RoPE temporal scale;
        # it deliberately does not follow the input video frame rate.
        batch.rope_interpolation_scale = build_rope_interpolation_scale(
            temporal_ratio=int(vae.temporal_compression_ratio),
            frame_rate=int(batch.frame_rate),
            spatial_ratio=int(vae.spatial_compression_ratio),
        )
        self.log_info("rope_interpolation_scale=%s", batch.rope_interpolation_scale)
        return batch
