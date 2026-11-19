"""EraserDiT erase – window validation stage."""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import field_summary
from nndeploy.e_dit.nodes.validators import StageValidators, VerificationResult


class EraserDiTEraseWindowValidationNode(StateNode):
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("video", batch.video, StageValidators.with_dims(5))
        result.add_check("mask", batch.mask, StageValidators.with_dims(5))
        for name in ("vae", "transformer", "scheduler", "text_encoder", "tokenizer"):
            result.add_check(
                f"modules.{name}", batch.modules.get(name), StageValidators.not_none
            )
        return result

    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        if batch.video.shape[0] != 1 or batch.mask.shape[0] != 1:
            raise ValueError("EraserDiT erase stages only support batch size 1")
        self.log_info(
            "%s | %s | bbox=%s",
            field_summary("video", batch.video),
            field_summary("mask", batch.mask),
            batch.bbox,
        )
        return batch
